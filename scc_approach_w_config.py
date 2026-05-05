import os
import time
import csv
import json
import argparse
import multiprocessing as mp
import queue
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import gurobipy as gp

# Compatibility shim: gurobipy <=12 had Model.getColumn; gurobipy 13+ renamed it to getCol.
# Alias so either name works without touching call sites.
if not hasattr(gp.Model, 'getColumn') and hasattr(gp.Model, 'getCol'):
    gp.Model.getColumn = gp.Model.getCol

# ══════════════════════════════════════════════════════════════════════════════
#  Model Manipulation
# ══════════════════════════════════════════════════════════════════════════════

def build_substitution_map(DE, IE):
    """
    Builds a substitution map based on DE (direct elimination) and IE (indirect elimination).
    Returns a map: var -> (rep_var, flip), where flip=0 if var == rep_var, flip=1 if var == 1 - rep_var.
    """
    parent = {}  # var -> (rep, flip)

    def find(var):
        # path compression
        if var not in parent:
            parent[var] = (var, 0)
        rep, flip = parent[var]
        if rep != var:
            new_rep, new_flip = find(rep)
            parent[var] = (new_rep, flip ^ new_flip)
        return parent[var]

    def union(x, y, relation):  # relation = 0 for DE, 1 for IE
        x_rep, x_flip = find(x)
        y_rep, y_flip = find(y)
        if x_rep == y_rep:
            return
        # merge y_rep into x_rep
        parent[y_rep] = (x_rep, x_flip ^ y_flip ^ relation)

    for x, y in DE:
        union(x, y, 0)
    for x, y in IE:
        union(x, y, 1)

    # Normalize the map
    substitution_map = {}
    for var in parent:
        rep, flip = find(var)
        if var != rep:
            substitution_map[var] = (rep, flip)

    return substitution_map

def substitute_variables_in_model(model, substitution_map):
    """
    Manually substitutes variables in a Gurobi model based on a mapping.
    For each entry var -> (rep, flip):
        flip == 0  means  var = rep           (Direct Elim)
        flip == 1  means  var = 1 - rep       (Indirect Elim)
    Rewrites every row a*var in 'constr' as (a)*rep  (flip=0) or (-a)*rep with
    RHS -= a (flip=1); rewrites the objective a*var analogously with ObjCon += a
    for flip=1 to preserve the value of the objective.

    Implementation notes (correctness):
      * gurobipy's LinExpr.getCoeff(i) takes an INTEGER INDEX; it does NOT accept
        a Var.  Passing a Var raises TypeError, which previously fell into a bare
        `except` and silently set rep_coeff = 0 — that would overwrite the rep
        coefficient with 0 + coeff on every touched row.  We look up rep's
        coefficient by scanning the row's entries by VarName instead.
      * gurobipy uses lazy updates (UpdateMode=1 by default).  chgCoeff / remove
        changes are pending until model.update().  We therefore pre-compute all
        coefficient/RHS/Obj deltas in pure Python first, THEN apply them in a
        single batched write, then call update() once.  This also guarantees
        that several sources aliasing into the same rep do not step on each
        other's accumulation.
    """
    from collections import defaultdict

    variables = {v.VarName: v for v in model.getVars()}

    # ---- Pass 1: compute all deltas without mutating the model ----
    # (constr_name -> rep_name -> delta to add to the rep's coefficient)
    rep_coef_delta = defaultdict(lambda: defaultdict(float))
    # (constr_name -> delta to add to RHS)
    rhs_delta = defaultdict(float)
    # (rep_name -> delta to add to rep.Obj), and total ObjCon delta
    obj_delta = defaultdict(float)
    objcon_delta = 0.0
    # List of (constr, var) pairs to zero out.
    zero_coef_pairs = []
    # Per-constraint cache of (name -> coef) so repeated getRow calls are avoided.
    constr_cache = {}

    def get_constr_coefs(c):
        if c.ConstrName in constr_cache:
            return constr_cache[c.ConstrName]
        row = model.getRow(c)
        d = {row.getVar(i).VarName: row.getCoeff(i) for i in range(row.size())}
        constr_cache[c.ConstrName] = d
        return d

    for var_name, (rep_name, flip) in substitution_map.items():
        if var_name not in variables or rep_name not in variables:
            continue
        var = variables[var_name]

        col = model.getColumn(var)
        for i in range(col.size()):
            constr = col.getConstr(i)
            coeff = col.getCoeff(i)
            zero_coef_pairs.append((constr, var))
            if flip == 0:
                rep_coef_delta[constr.ConstrName][rep_name] += coeff
            else:
                rep_coef_delta[constr.ConstrName][rep_name] -= coeff
                rhs_delta[constr.ConstrName] -= coeff

        obj_coeff = var.Obj
        if obj_coeff != 0.0:
            if flip == 0:
                obj_delta[rep_name] += obj_coeff
            else:
                obj_delta[rep_name] -= obj_coeff
                objcon_delta += obj_coeff

    # ---- Pass 2: read current rep coefficients / RHS in one sweep ----
    # Resolve deltas to absolute new values using cached original coefficients.
    constr_by_name = {c.ConstrName: c for c in model.getConstrs()}
    new_rep_coef = []  # list of (constr, rep_var, new_coef)
    new_rhs = []       # list of (constr, new_rhs)
    for cname, rep_deltas in rep_coef_delta.items():
        c = constr_by_name[cname]
        current = get_constr_coefs(c)
        for rep_name, delta in rep_deltas.items():
            rep_var = variables[rep_name]
            new_rep_coef.append((c, rep_var, current.get(rep_name, 0.0) + delta))
    for cname, drhs in rhs_delta.items():
        c = constr_by_name[cname]
        new_rhs.append((c, c.RHS + drhs))

    # ---- Pass 3: apply all mutations ----
    for constr, var in zero_coef_pairs:
        model.chgCoeff(constr, var, 0.0)
    for constr, rep_var, new_coef in new_rep_coef:
        model.chgCoeff(constr, rep_var, new_coef)
    for constr, rhs_val in new_rhs:
        constr.RHS = rhs_val

    # Objective deltas
    for rep_name, d in obj_delta.items():
        rep_var = variables[rep_name]
        rep_var.Obj = rep_var.Obj + d
    if objcon_delta != 0.0:
        model.ObjCon += objcon_delta

    # Zero out the substituted variables' obj and remove them.
    for var_name, _ in substitution_map.items():
        if var_name in variables:
            var = variables[var_name]
            try:
                var.Obj = 0.0
                model.remove(var)
            except Exception:
                pass

    model.update()

def apply_changes_to_model(model, F0, F1, DE, IE):
    """
    Applies all known reductions to a Gurobi model:
    - Substitutes variables based on DE and IE (eliminates aliases into representatives).
    - Fixes variables from F0 and F1 by PHYSICALLY REMOVING them from the model
      (fix-to-0 ->  model.remove(var);
       fix-to-1 ->  for each row c with coef a: c.RHS -= a, ObjCon += var.Obj, then model.remove(var)).
      This makes fixing structurally symmetric with elimination: both shrink the LP instead
      of merely tightening bounds, which is the key to the reduction mattering when
      Gurobi's presolve/propagation is disabled.
    - If a fixed variable has been substituted out (alias), the fixing is translated onto
      its representative via the DE/IE substitution_map, accounting for flip.
    """
    # -------------------------------------------------
    # 1. Build substitution map from DE/IE and translate F0/F1 onto representatives
    # -------------------------------------------------
    substitution_map = build_substitution_map(DE, IE)

    effective_F0 = set()
    effective_F1 = set()
    for v in F0:
        if v in substitution_map:
            rep, flip = substitution_map[v]
            # v = rep (flip=0) or v = 1 - rep (flip=1); v == 0 implies rep == 0 or rep == 1 respectively
            (effective_F0 if flip == 0 else effective_F1).add(rep)
        else:
            effective_F0.add(v)
    for v in F1:
        if v in substitution_map:
            rep, flip = substitution_map[v]
            (effective_F1 if flip == 0 else effective_F0).add(rep)
        else:
            effective_F1.add(v)

    # -------------------------------------------------
    # 2. Apply elimination (removes aliased variables, rewrites coefficients onto reps)
    # -------------------------------------------------
    substitute_variables_in_model(model, substitution_map)

    # -------------------------------------------------
    # 3. Physically remove fixed variables (post-substitution, so reps are the right targets)
    # -------------------------------------------------
    vars_dict = {v.VarName: v for v in model.getVars()}
    fixed_vars = {}

    # Fix-to-1: move the constant contribution to each row's RHS, bump ObjCon, remove var.
    #
    # Correctness note: gurobipy uses lazy updates (UpdateMode=1).  A naive loop
    #   for v in effective_F1:
    #       for (c, a) in col(v):
    #           c.RHS -= a
    # exhibits "last-write-wins" semantics whenever two F1 vars share a row,
    # because each `-=` reads the pre-loop value of c.RHS (pending writes are
    # not visible without an intervening model.update()).  On academictimetablebig
    # this drove two rows to structurally-infeasible RHS and triggered a false
    # 0.28s INFEASIBLE verdict.  See investigation_bug_atb/BUG_REPORT.md.
    # Fix: accumulate per-constraint RHS deltas in Python, then do a single
    # write per constraint at the end.
    rhs_delta = defaultdict(float)
    objcon_delta = 0.0
    f1_to_remove = []
    for v_name in effective_F1:
        if v_name not in vars_dict:
            continue
        var = vars_dict[v_name]
        col = model.getColumn(var)
        for i in range(col.size()):
            c = col.getConstr(i)
            a = col.getCoeff(i)
            rhs_delta[c.ConstrName] += -a
        if var.Obj != 0.0:
            objcon_delta += var.Obj
        f1_to_remove.append(var)
        fixed_vars[v_name] = 1

    # Resolve RHS deltas to absolute values using the current (pre-write) RHS,
    # then apply one write per constraint.
    if rhs_delta:
        constr_by_name = {c.ConstrName: c for c in model.getConstrs()}
        new_rhs = [(constr_by_name[nm], constr_by_name[nm].RHS + d)
                   for nm, d in rhs_delta.items()]
        for c, r in new_rhs:
            c.RHS = r
    if objcon_delta != 0.0:
        model.ObjCon += objcon_delta
    for v in f1_to_remove:
        model.remove(v)

    # Fix-to-0: contribution is 0 everywhere; just remove the variable.
    for v_name in effective_F0:
        if v_name not in vars_dict:
            continue
        # If the name also appeared in effective_F1 (shouldn't happen unless contradictory
        # inputs), the var was already removed; guard with getVarByName.
        var = model.getVarByName(v_name)
        if var is None:
            continue
        model.remove(var)
        fixed_vars[v_name] = 0

    model.update()

    return {
        "fixed": fixed_vars,
        "substitution_map": substitution_map
    }

# ══════════════════════════════════════════════════════════════════════════════
#  Graph Generation & Reductions (Optimized via Integer Mapping)
# ══════════════════════════════════════════════════════════════════════════════

def create_implication_graph(model) -> Tuple[List[List[int]], List[str], int, int]:
    """
    Builds the implication graph using pure integer indexing.
    Node 2*i represents 0*x_i. Node 2*i + 1 represents 1*x_i.
    """
    from mip import ConflictGraph
    cg = ConflictGraph(model)
    
    n_vars = len(model.vars)
    n_nodes = 2 * n_vars
    adj = [[] for _ in range(n_nodes)]
    
    var_to_idx = {v.name: i for i, v in enumerate(model.vars)}
    idx_to_var = [v.name for v in model.vars]
    m_edges = 0

    for i, x in enumerate(model.vars):
        x0 = 2 * i
        x1 = 2 * i + 1

        # xi = 0
        z = cg.conflicting_assignments(x == 0)
        for y in z[0]:
            y0 = 2 * var_to_idx[y.name]
            y1 = y0 + 1
            adj[x0].append(y0)
            adj[y1].append(x1)
            m_edges += 2
        for y in z[1]:
            y0 = 2 * var_to_idx[y.name]
            y1 = y0 + 1
            adj[x0].append(y1)
            adj[y0].append(x1)
            m_edges += 2

        # xi = 1
        o = cg.conflicting_assignments(x)
        for y in o[0]:
            y0 = 2 * var_to_idx[y.name]
            y1 = y0 + 1
            adj[x1].append(y0)
            adj[y1].append(x0)
            m_edges += 2
        for y in o[1]:
            y0 = 2 * var_to_idx[y.name]
            y1 = y0 + 1
            adj[x1].append(y1)
            adj[y0].append(x0)
            m_edges += 2

    return adj, idx_to_var, n_nodes, m_edges

def build_scc_structures(adj: List[List[int]], n_nodes: int):
    """
    Tarjan's Algorithm running entirely on pre-allocated integer arrays.
    """
    index = 0
    indices = [-1] * n_nodes
    lowlink = [-1] * n_nodes
    on_stack = [False] * n_nodes
    stack = []
    sccs = []

    node_to_scc = [-1] * n_nodes

    for v in range(n_nodes):
        if indices[v] == -1:
            call_stack = [(v, 0)]
            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            on_stack[v] = True

            while call_stack:
                curr, edge_idx = call_stack[-1]
                neighbors = adj[curr]

                if edge_idx < len(neighbors):
                    w = neighbors[edge_idx]
                    call_stack[-1] = (curr, edge_idx + 1)
                    if indices[w] == -1:
                        indices[w] = index
                        lowlink[w] = index
                        index += 1
                        stack.append(w)
                        on_stack[w] = True
                        call_stack.append((w, 0))
                    elif on_stack[w]:
                        if indices[w] < lowlink[curr]:
                            lowlink[curr] = indices[w]
                else:
                    call_stack.pop()
                    if call_stack:
                        prev = call_stack[-1][0]
                        if lowlink[curr] < lowlink[prev]:
                            lowlink[prev] = lowlink[curr]

                    # This MUST be inside the `else` block (after all neighbors are explored)
                    if lowlink[curr] == indices[curr]:
                        scc = []
                        scc_id = len(sccs)
                        while True:
                            w = stack.pop()
                            on_stack[w] = False
                            scc.append(w)
                            node_to_scc[w] = scc_id
                            if w == curr:
                                break
                        sccs.append(scc)

    num_sccs = len(sccs)
    
    dag_adj_sets = [set() for _ in range(num_sccs)]
    for u in range(n_nodes):
        scc_u = node_to_scc[u]
        for v in adj[u]:
            scc_v = node_to_scc[v]
            if scc_u != scc_v:
                dag_adj_sets[scc_u].add(scc_v)

    dag_adj = [list(edges) for edges in dag_adj_sets]

    return node_to_scc, num_sccs, dag_adj

def elimination_on_implication_graph_scc(node_to_scc: List[int], n_vars: int, idx_to_var: List[str]):
    """
    Finds direct and indirect eliminations using integer-based arrays.
    """
    dsu_parent = list(range(n_vars))
    dsu_rank = [0] * n_vars
    
    def find(i):
        if dsu_parent[i] != i:
            dsu_parent[i] = find(dsu_parent[i])
        return dsu_parent[i]
    
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            if dsu_rank[root_i] < dsu_rank[root_j]:
                root_i, root_j = root_j, root_i
            dsu_parent[root_j] = root_i
            if dsu_rank[root_i] == dsu_rank[root_j]:
                dsu_rank[root_i] += 1

    scc_to_0_vars = defaultdict(list)
    scc_to_1_vars = defaultdict(list)
    
    for i in range(n_vars):
        scc_0 = node_to_scc[2*i]
        scc_1 = node_to_scc[2*i + 1]
        
        if scc_0 == -1 or scc_1 == -1 or scc_0 == scc_1:
            continue
            
        scc_to_0_vars[scc_0].append(i)
        scc_to_1_vars[scc_1].append(i)

    for scc_id, vars_in_scc in scc_to_0_vars.items():
        if len(vars_in_scc) > 1:
            root_var = vars_in_scc[0]
            for other_var in vars_in_scc[1:]:
                union(root_var, other_var)

    groups_dict = defaultdict(list)
    for i in range(n_vars):
        groups_dict[find(i)].append(i)
        
    direct_eliminations = [set(idx_to_var[i] for i in g) for g in groups_dict.values() if len(g) > 1]
    
    var_to_root = [find(i) for i in range(n_vars)]
    indirect_map_internal = defaultdict(set)
    
    for scc_id, vars_0 in scc_to_0_vars.items():
        if scc_id in scc_to_1_vars: 
            vars_1 = scc_to_1_vars[scc_id]
            for v_a in vars_0:
                for v_b in vars_1:
                    if var_to_root[v_a] != var_to_root[v_b]:
                        name_a = idx_to_var[v_a]
                        name_b = idx_to_var[v_b]
                        indirect_map_internal[name_a].add(name_b)
                        indirect_map_internal[name_b].add(name_a)

    indirect_map = {k: list(v) for k, v in indirect_map_internal.items()}
    return direct_eliminations, indirect_map

def fixing_on_implication_graph_scc_bfs(node_to_scc: List[int], n_vars: int, idx_to_var: List[str], 
                                        num_sccs: int, dag_adj: List[List[int]]):
    """
    Calculates reachability using Breadth-First Search (BFS) on the SCC DAG.
    """
    reach = [set() for _ in range(num_sccs)]
    
    for u in range(num_sccs):
        visited = {u}
        stack = [u]
        while stack:
            curr = stack.pop()
            for v in dag_adj[curr]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        reach[u] = visited
            
    fixing_0 = set()
    fixing_1 = set()
    
    for i in range(n_vars):
        scc_0 = node_to_scc[2*i]
        scc_1 = node_to_scc[2*i + 1]
        
        if scc_0 == -1 or scc_1 == -1:
            continue
            
        # 1x reaches 0x => x forced to 0
        if scc_0 in reach[scc_1]:
            fixing_0.add(idx_to_var[i]) 
            continue 
            
        # 0x reaches 1x => x forced to 1
        if scc_1 in reach[scc_0]:
            fixing_1.add(idx_to_var[i]) 
            
    return fixing_0, fixing_1

def fixing_on_implication_graph_scc_bitset(node_to_scc: List[int], n_vars: int, idx_to_var: List[str], 
                                           num_sccs: int, dag_adj: List[List[int]]):
    """
    Calculates reachability using native Python Arbitrary-Precision Integers.
    Because Tarjan's algorithm produces SCCs in reverse topological order by default,
    range(num_sccs) inherently processes from sinks upwards.
    """
    reach = [1 << i for i in range(num_sccs)]
    
    # Process in reverse topological order (0 to num_sccs-1)
    for u in range(num_sccs):
        u_reach = reach[u]
        for v in dag_adj[u]:
            u_reach |= reach[v]
        reach[u] = u_reach
            
    fixing_0 = set()
    fixing_1 = set()
    
    for i in range(n_vars):
        scc_0 = node_to_scc[2*i]
        scc_1 = node_to_scc[2*i + 1]
        
        if scc_0 == -1 or scc_1 == -1:
            continue
            
        # 1x reaches 0x => x forced to 0
        if reach[scc_1] & (1 << scc_0):
            fixing_0.add(idx_to_var[i]) 
            continue 
            
        # 0x reaches 1x => x forced to 1
        if reach[scc_0] & (1 << scc_1):
            fixing_1.add(idx_to_var[i]) 
            
    return fixing_0, fixing_1

# ══════════════════════════════════════════════════════════════════════════════
#  Worker Process (Isolated Memory Environment)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_worker(filepath_str: str, result_queue: mp.Queue, max_vars: int):
    """
    Executes all preprocessing steps, pushing results to the queue progressively.
    If the worker is killed during a slow step, previous steps are already safely reported!
    """
    try:
        from mip import Model, MINIMIZE, CBC
        mip_model = Model(sense=MINIMIZE, solver_name=CBC)
        mip_model.read(filepath_str)
        
        n_vars = len(mip_model.vars)
        if n_vars > max_vars:
            result_queue.put({"step": "skipped", "reason": "SKIPPED_MAX_VARS", "n_vars": n_vars})
            return

        adj, idx_to_var, n_nodes, m_edges = create_implication_graph(mip_model)
        result_queue.put({
            "step": "graph",
            "n_vars": n_vars, 
            "n_nodes": n_nodes, 
            "m_edges": m_edges
        })

        # --- SCC Step ---
        t_scc_start = time.time()
        node_to_scc, num_sccs, dag_adj = build_scc_structures(adj, n_nodes)
        t_scc = time.time() - t_scc_start
        
        result_queue.put({"step": "scc", "num_sccs": num_sccs, "t_scc": t_scc})
        
        # --- Elimination Step ---
        t_elim_start = time.time()
        d_elim_groups, i_map = elimination_on_implication_graph_scc(node_to_scc, n_vars, idx_to_var)
        t_elim = time.time() - t_elim_start

        d_count = sum(len(g) - 1 for g in d_elim_groups)
        i_count = sum(len(neighs) for neighs in i_map.values()) // 2
        
        DE_pairs = [
            (v1, v2)
            for group in d_elim_groups
            for i, v1 in enumerate(sorted(group))
            for v2 in sorted(group)[i+1:]
        ]

        IE_pairs = []
        seen_ie = set()
        for a, ns in i_map.items():
            for b in ns:
                pair = tuple(sorted((a, b)))
                if pair not in seen_ie:
                    seen_ie.add(pair)
                    IE_pairs.append(pair)
                    
        result_queue.put({
            "step": "elim", "t_elim": t_elim,
            "d_count": d_count, "i_count": i_count,
            "DE_pairs": DE_pairs, "IE_pairs": IE_pairs
        })
        
        # --- Fixing Step (BFS) ---
        t_fix_bfs_start = time.time()
        f0_bfs, f1_bfs = fixing_on_implication_graph_scc_bfs(node_to_scc, n_vars, idx_to_var, num_sccs, dag_adj)
        t_fix_bfs = time.time() - t_fix_bfs_start
        
        result_queue.put({
            "step": "fix_bfs", "t_fix_bfs": t_fix_bfs,
            "f0_bfs": f0_bfs, "f1_bfs": f1_bfs
        })

        # --- Fixing Step (Bitset) ---
        # Isolate the bitset path: it materializes a Python-int reach bitset per
        # SCC and can OOM on huge graphs (e.g. rail03 has 1.29M SCCs over a 30M-
        # edge implication graph).  Failures here must NOT poison the rest of the
        # record — BFS fixings, DE/IE pairs, and timings are already valid.
        try:
            t_fix_bitset_start = time.time()
            f0_bitset, f1_bitset = fixing_on_implication_graph_scc_bitset(node_to_scc, n_vars, idx_to_var, num_sccs, dag_adj)
            t_fix_bitset = time.time() - t_fix_bitset_start

            result_queue.put({
                "step": "fix_bitset", "t_fix_bitset": t_fix_bitset,
                "f0_bitset": f0_bitset, "f1_bitset": f1_bitset
            })
        except Exception as bitset_err:
            import traceback as _tb
            result_queue.put({
                "step": "fix_bitset_failed",
                "t_fix_bitset": time.time() - t_fix_bitset_start,
                "f0_bitset": set(),
                "f1_bitset": set(),
                "bitset_err": f"{type(bitset_err).__name__}: {bitset_err}\n{_tb.format_exc()}"
            })

    except Exception as e:
        import traceback
        result_queue.put({"step": "error", "message": f"{str(e)}\n{traceback.format_exc()}"})

# ══════════════════════════════════════════════════════════════════════════════
#  Main Batch Pipeline (Dictionary Config based)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.json", help="Path to JSON run config.")
    parser.add_argument("--tag", type=str, default="", help="Optional tag for output directory.")
    parser.add_argument("--max_vars", type=int, default=1000000, help="Maximum number of vars allowed.")
    parser.add_argument("--timeout_prep", type=int, default=1800, help="Max seconds per isolated preprocessing step.")
    
    # CLI Overrides (these override config.json values if True)
    parser.add_argument("--no_elim", action="store_true", help="Disable elimination completely.")
    parser.add_argument("--no_fix", action="store_true", help="Disable fixing completely.")
    parser.add_argument("--no_conf", action="store_true", help="Disable config defaults (if applicable).")
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated Gurobi Seed values to run each (instance,config) under. Default: '0,1,2,3,4'."
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=3600,
        help="Per-solve time limit in seconds."
    )
    args = parser.parse_args()

    # Parse seeds
    try:
        seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    except ValueError:
        print(f"[!] Invalid --seeds value '{args.seeds}'. Must be comma-separated integers.")
        return
    if not seed_list:
        seed_list = [0]

    if not os.path.exists(args.config_path):
        print(f"[!] Config file '{args.config_path}' not found.")
        return

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    # Setup results directory
    today = datetime.now().strftime("%Y-%m-%d")
    tag_str = f"_{args.tag}" if args.tag else ""
    results_dir = Path(f"results_{today}{tag_str}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Per-solve Gurobi logs live here; console stays quiet.
    # Resolve to an absolute path so Gurobi (which on Windows sometimes struggles
    # with relative paths) has no ambiguity about where to open the log file.
    logs_dir = (results_dir / "logs").resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "batch_results.csv"
    
    # Added 'run_id' and 'category' to CSV columns. 'seed' supports multi-seed averaging.
    fieldnames = [
        'run_id', 'category', 'problem', 'n', 'm', 'num_sccs',
        'scc_build_time', 'elim_time', 'fix_bfs_time', 'fix_bitset_time',
        'has_reductions',
        'direct_elim_#', 'indirect_elim_#',
        'fixing_0_bfs_#', 'fixing_1_bfs_#',
        'fixing_0_bitset_#', 'fixing_1_bitset_#',
        'config', 'seed', 'solve_status', 'solve_time', 'nodes', 'obj_val', 'obj_bound'
    ]

    # Write headers if new file
    if not csv_path.exists():
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    ctx = mp.get_context('spawn') # Must use 'spawn' to be safe on Windows/Linux with Gurobi

    index = 0
    total_runs = len(config)
    for run_id, run_config in config.items():
        filepath_str = run_config.get("filepath")
        category = run_config.get("category", "Uncategorized")

        if not filepath_str or not category:
            print(f"[!] Skipping {run_id} (missing filepath or category).")
            continue

        filepath = Path(filepath_str)
        if not filepath.exists():
            print(f"[!] Skipping {run_id} (file not found: {filepath_str})")
            continue
            
        index += 1
        print(f"\n[{index}/{total_runs}] Running {run_id}: {filepath.name} (Category: {category})")

        # Combine logic: if 'elimination' is disabled in config OR by CLI flag '--no_elim'
        has_elim = run_config.get("elimination", True) and not args.no_elim
        has_fix = run_config.get("fixing", True) and not args.no_fix
        has_conf = not args.no_conf # Available as a global flag 

        problem_info = {k: "N/A" for k in fieldnames}
        problem_info['run_id'] = run_id
        problem_info['category'] = category
        problem_info['problem'] = filepath.name

        q = ctx.Queue()
        p = ctx.Process(target=preprocess_worker, args=(str(filepath), q, args.max_vars))
        p.start()
        
        # Track the accumulated results from the worker as steps finish
        worker_res = {
            "n_vars": 0, "n_nodes": 0, "m_edges": 0,
            "num_sccs": 0, "t_scc": 0.0,
            "t_elim": 0.0, "d_count": 0, "i_count": 0, "DE_pairs": [], "IE_pairs": [],
            "t_fix_bfs": 0.0, "f0_bfs": set(), "f1_bfs": set(),
            "t_fix_bitset": 0.0, "f0_bitset": set(), "f1_bitset": set(),
            "status": "processing",
            "error_msg": ""
        }

        # Expect sequential step completions
        steps_to_wait = ["graph", "scc", "elim", "fix_bfs", "fix_bitset"]
        current_step_idx = 0

        while current_step_idx < len(steps_to_wait):
            expected_step = steps_to_wait[current_step_idx]
            
            # Loading graph can take longer for massive files
            step_timeout = 7200 if expected_step == "graph" else args.timeout_prep
            
            try:
                msg = q.get(timeout=step_timeout)
                
                if msg["step"] == "skipped":
                    worker_res["status"] = "skipped"
                    worker_res["error_msg"] = msg["reason"]
                    break
                elif msg["step"] == "error":
                    worker_res["status"] = "error"
                    worker_res["error_msg"] = msg["message"]
                    break
                elif msg["step"] == "fix_bitset_failed":
                    # Bitset path crashed (likely OOM on huge graphs).  BFS results
                    # are already in worker_res from the previous step.  Record the
                    # failure but don't abort — DE/IE/F0/F1 (BFS) are still valid.
                    for k, v in msg.items():
                        if k not in ("step", "bitset_err"):
                            worker_res[k] = v
                    worker_res["bitset_err"] = msg.get("bitset_err", "")
                    print(f"    - Step 'fix_bitset' FAILED ({worker_res['bitset_err'].splitlines()[0][:80] if worker_res['bitset_err'] else 'unknown'}); keeping BFS results and continuing.")
                    current_step_idx += 1
                    continue

                # Merge incoming step results
                for k, v in msg.items():
                    if k != "step":
                        worker_res[k] = v

                print(f"    - Completed step: {expected_step}")
                current_step_idx += 1
                
            except queue.Empty:
                if p.is_alive():
                    print(f"    TIMEOUT on step '{expected_step}'. Terminating worker but keeping prior reductions.")
                    p.terminate()
                    p.join()
                    worker_res["status"] = f"timeout_{expected_step}"
                else:
                    print(f"    CRITICAL ERROR: Worker crashed silently during '{expected_step}'. Keeping prior reductions.")
                    worker_res["status"] = f"crash_{expected_step}"
                break
                
        if worker_res["status"] == "processing":
            worker_res["status"] = "success"
            
        # ==========================================
        # Compile Gathered Data
        # ==========================================
        
        has_reductions = (worker_res["d_count"] > 0 or worker_res["i_count"] > 0 or 
                          len(worker_res["f0_bfs"]) > 0 or len(worker_res["f1_bfs"]) > 0 or
                          len(worker_res["f0_bitset"]) > 0 or len(worker_res["f1_bitset"]) > 0)
                          
        problem_info.update({
            'n': worker_res["n_nodes"], 'm': worker_res["m_edges"], 'num_sccs': worker_res["num_sccs"],
            'scc_build_time': round(worker_res["t_scc"], 6),
            'elim_time': round(worker_res["t_elim"], 6),
            'fix_bfs_time': round(worker_res["t_fix_bfs"], 6),
            'fix_bitset_time': round(worker_res["t_fix_bitset"], 6),
            'has_reductions': has_reductions,
            'direct_elim_#': worker_res["d_count"],
            'indirect_elim_#': worker_res["i_count"],
            'fixing_0_bfs_#': len(worker_res["f0_bfs"]),
            'fixing_1_bfs_#': len(worker_res["f1_bfs"]),
            'fixing_0_bitset_#': len(worker_res["f0_bitset"]),
            'fixing_1_bitset_#': len(worker_res["f1_bitset"])
        })

        # SKIPPED is always terminal.  Graph-level crash/timeout means we have
        # nothing usable.  Plain "error" used to be terminal too — but if the
        # worker died AFTER producing usable DE/IE/F0/F1 (e.g. bitset OOM on
        # rail03), short-circuiting here loses the chance to run Baseline +
        # Elimination_Only / Fixing_Only with the BFS-side results we already
        # collected.  So: only abort if reductions are empty.
        terminal_now = (
            worker_res["status"] == "skipped"
            or worker_res["status"].startswith("crash_graph")
            or worker_res["status"].startswith("timeout_graph")
            or (worker_res["status"] == "error" and not has_reductions)
        )
        if terminal_now:
            # Truncate to 200 chars so the actual exception class is visible
            # (40 was eating "MemoryError: ..." and friends).
            problem_info['config'] = f"{worker_res['status']}: {worker_res['error_msg'][:200]}"
            with open(csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
            continue

        # Worker errored AFTER producing reductions (typically bitset OOM after
        # BFS succeeded): record the diagnostic in the row but proceed to
        # Gurobi solves with whatever we have.
        if worker_res["status"] == "error":
            print(f"    [warn] Worker errored late but reductions are usable; "
                  f"proceeding with Gurobi solves.  err: {worker_res['error_msg'][:120]!r}")
            worker_res["status"] = "partial_success"

        if not has_reductions:
            problem_info['config'] = f"No_Reductions_{worker_res['status']}"
            with open(csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
            print(f"    No reductions found for {run_id}. Skipping Gurobi solves.")
            continue

        # 3. Gurobi Solving Logic
        
        # Determine valid preprocessing sets based on worker results
        f0_to_use = worker_res["f0_bfs"] or worker_res["f0_bitset"]
        f1_to_use = worker_res["f1_bfs"] or worker_res["f1_bitset"]
        
        can_elim = (worker_res["d_count"] > 0 or worker_res["i_count"] > 0)
        can_fix = (len(f0_to_use) > 0 or len(f1_to_use) > 0)

        # Build execution configs respecting the JSON config and CLI flags
        configs_to_run = [("Baseline", {})]
        
        if has_elim and can_elim:
            configs_to_run.append(("Elimination_Only", {"DE": worker_res["DE_pairs"], "IE": worker_res["IE_pairs"]}))
            
        if has_fix and can_fix:
            configs_to_run.append(("Fixing_Only", {"F0": f0_to_use, "F1": f1_to_use}))
            
        if (has_elim and can_elim) and (has_fix and can_fix):
            configs_to_run.append(("All_Preprocessing", {
                "F0": f0_to_use, "F1": f1_to_use, 
                "DE": worker_res["DE_pairs"], "IE": worker_res["IE_pairs"]
            }))
        
        try:
            # We ONLY load Gurobi in the main thread! MIP memory is already cleaned up.
            # Use a quiet env so the MPS-read banner doesn't spam the console.
            _read_env = gp.Env(empty=True)
            _read_env.setParam("OutputFlag", 0)
            _read_env.start()
            gurobi_base_model = gp.read(str(filepath), env=_read_env)

            for cfg_name, changes in configs_to_run:
                for seed in seed_list:
                    row = problem_info.copy()
                    row['config'] = f"{cfg_name} (Status: {worker_res['status']})"
                    row['seed'] = seed

                    try:
                        m_solve = gurobi_base_model.copy()
                        # Only two knobs: Presolve=0 isolates our reductions from Gurobi's
                        # own probing/bound-strengthening; everything else (Cuts, Aggregate,
                        # PreSparsify, ...) is orthogonal to our contribution so we leave
                        # Gurobi's defaults in place. Seed controls run-to-run variance.
                        m_solve.setParam("Presolve", 0)
                        m_solve.setParam("TimeLimit", args.time_limit)
                        m_solve.setParam("Seed", seed)

                        # Redirect Gurobi's own output to a per-solve log file so the
                        # batch runner's console stays readable.  Keep the filename
                        # SHORT — Windows MAX_PATH is 260 and the repo/results prefix
                        # can easily eat 200+ chars, leaving little room.  Abbreviate
                        # config names and cap the problem stem at 24 chars.
                        _CFG_ABBR = {
                            "Baseline": "base",
                            "Elimination_Only": "elim",
                            "Fixing_Only": "fix",
                            "All_Preprocessing": "all",
                        }
                        safe_cfg = _CFG_ABBR.get(cfg_name, cfg_name[:6]).lower()
                        safe_problem = filepath.stem[:24]
                        log_path = logs_dir / f"{safe_problem}_{safe_cfg}_s{seed}.log"
                        # Always silence the console; LogFile is best-effort.  If Gurobi
                        # can't open the log file for whatever reason (Windows path quirks,
                        # permissions, ...), just fall through with OutputFlag=0 so the
                        # solve still runs silently.  This keeps the batch resilient.
                        #
                        # CRITICAL: The read env was created with OutputFlag=0 to silence the
                        # MPS-read banner.  That setting propagates through .copy() and would
                        # silence both the console AND the LogFile (OutputFlag=0 overrides
                        # LogToConsole and LogFile).  Re-enable OutputFlag=1 here so Gurobi
                        # actually emits messages, then route them to the file via
                        # LogToConsole=0 + LogFile=...
                        m_solve.setParam("OutputFlag", 1)
                        m_solve.setParam("LogToConsole", 0)
                        try:
                            # Defensive: ensure the directory exists right before use, and
                            # pass an absolute string path so Gurobi has no ambiguity.
                            log_path.parent.mkdir(parents=True, exist_ok=True)
                            m_solve.setParam("LogFile", str(log_path))
                        except Exception as _log_err:
                            print(f"    [warn] LogFile setup failed ({_log_err}); continuing with OutputFlag=0")
                            m_solve.setParam("LogFile", "")
                            m_solve.setParam("OutputFlag", 0)

                        # Apply extracted reductions (physically removes fixed vars)
                        apply_changes_to_model(
                            m_solve,
                            changes.get("F0", set()),
                            changes.get("F1", set()),
                            changes.get("DE", []),
                            changes.get("IE", [])
                        )

                        t_solve_start = time.time()
                        m_solve.optimize()
                        solve_duration = round(time.time() - t_solve_start, 4)

                        row.update({
                            'solve_status': m_solve.Status,
                            'solve_time': solve_duration,
                            'nodes': int(m_solve.NodeCount),
                            'obj_val': m_solve.ObjVal if m_solve.SolCount > 0 else "N/A",
                            'obj_bound': m_solve.ObjBound
                        })

                    except Exception as e:
                        print(f"    Config {cfg_name} (seed={seed}) failed: {e}")
                        row['solve_status'] = f"FAILED: {str(e)[:30]}"

                    finally:
                        with open(csv_path, 'a', newline='') as f:
                            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        except Exception as e:
            print(f"  Gurobi model load failed: {e}")
            problem_info['config'] = "GUROBI_LOAD_FAIL"
            with open(csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)

    print(f"\nBatch processing complete. Results safely saved in {csv_path}")

if __name__ == "__main__":
    mp.freeze_support() # Essential for Windows multiprocessing
    main()

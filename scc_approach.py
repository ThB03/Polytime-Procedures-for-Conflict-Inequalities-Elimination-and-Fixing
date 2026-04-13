import os
import time
import csv
import argparse
import multiprocessing as mp
import queue
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import gurobipy as gp

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
    Uses getColumn() for O(V) constraint lookup and safely handles Objective offsets.
    """
    variables = {v.VarName: v for v in model.getVars()}

    for var_name, (rep_name, flip) in substitution_map.items():
        if var_name not in variables or rep_name not in variables:
            continue

        var = variables[var_name]
        rep = variables[rep_name]

        col = model.getColumn(var)
        
        for i in range(col.size()):
            constr = col.getConstr(i)
            coeff = col.getCoeff(i)
            
            model.chgCoeff(constr, var, 0.0)
            
            expr = model.getRow(constr)
            try:
                rep_coeff = expr.getCoeff(rep)
            except:
                rep_coeff = 0.0  

            if flip == 0:
                model.chgCoeff(constr, rep, rep_coeff + coeff)
            else:
                model.chgCoeff(constr, rep, rep_coeff - coeff)
                constr.RHS -= coeff

        obj_coeff = var.Obj
        if obj_coeff != 0:
            var.Obj = 0.0
            if flip == 0:
                rep.Obj += obj_coeff
            else:
                rep.Obj -= obj_coeff
                model.ObjCon += obj_coeff  # CRITICAL FIX: Add offset instead of flipping ModelSense

        model.remove(var)

    model.update()


def apply_changes_to_model(model, F0, F1, DE, IE, stat, varname):
    """
    Applies all known reductions and constraints to a Gurobi model:
    - Fixes variables from F0 and F1
    - Substitutes variables based on DE and IE (eliminates)
    - Adds constraints for AE (conflict edges)
    """
    vars_dict = {v.VarName: v for v in model.getVars()}
    fixed_vars = {}

    # ------------------------
    # 1. Fix variables in F0 and F1
    # OPTIMIZATION: Loop through the F0/F1 sets, not the whole model
    for v_name in F0:
        if v_name in vars_dict:
            var = vars_dict[v_name]
            var.LB = 0
            var.UB = 0
            fixed_vars[v_name] = 0

    for v_name in F1:
        if v_name in vars_dict:
            var = vars_dict[v_name]
            var.LB = 1
            var.UB = 1
            fixed_vars[v_name] = 1

    # ------------------------
    # 2. Eliminate variables from DE and IE
    substitution_map = build_substitution_map(DE, IE)
    substitute_variables_in_model(model, substitution_map)

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
            ('1'+v1, '1'+v2) 
            for group in d_elim_groups 
            for i, v1 in enumerate(sorted(group)) 
            for v2 in sorted(group)[i+1:]
        ]
        
        IE_pairs = []
        seen_ie = set()
        for a, ns in i_map.items():
            for b in ns:
                pair = tuple(sorted(('1'+a, '1'+b)))
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
        t_fix_bitset_start = time.time()
        f0_bitset, f1_bitset = fixing_on_implication_graph_scc_bitset(node_to_scc, n_vars, idx_to_var, num_sccs, dag_adj)
        t_fix_bitset = time.time() - t_fix_bitset_start
        
        result_queue.put({
            "step": "fix_bitset", "t_fix_bitset": t_fix_bitset,
            "f0_bitset": f0_bitset, "f1_bitset": f1_bitset
        })

    except Exception as e:
        import traceback
        result_queue.put({"step": "error", "message": f"{str(e)}\n{traceback.format_exc()}"})

# ══════════════════════════════════════════════════════════════════════════════
#  Main Batch Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def find_problem_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.mps"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems_dir", type=str, default="problems")
    parser.add_argument("--max_vars", type=int, default=1000000)
    parser.add_argument("--timeout_prep", type=int, default=1800, help="Max seconds per isolated preprocessing step.")
    args = parser.parse_args()

    problems_root = Path(args.problems_dir)
    if not problems_root.exists():
        print(f"Problems directory '{args.problems_dir}' not found")
        return

    files = find_problem_files(problems_root)
    
    skip_set = set()
    skip_file_path = Path("toSkip.txt")
    if skip_file_path.exists():
        with open(skip_file_path, 'r') as f:
            for line in f:
                name = line.strip()
                if name:
                    skip_set.add(name)
        print(f"Loaded {len(skip_set)} instances to skip from toSkip.txt")

    today = datetime.now().strftime("%Y-%m-%d")
    results_dir = Path(f"results_{today}_batch_resilient")
    results_dir.mkdir(exist_ok=True)
    
    csv_path = results_dir / "batch_results.csv"
    
    fieldnames = [
        'problem', 'n', 'm', 'num_sccs',
        'scc_build_time', 'elim_time', 'fix_bfs_time', 'fix_bitset_time',
        'has_reductions', 
        'direct_elim_#', 'indirect_elim_#', 
        'fixing_0_bfs_#', 'fixing_1_bfs_#',
        'fixing_0_bitset_#', 'fixing_1_bitset_#',
        'config', 'solve_status', 'solve_time', 'nodes', 'obj_val', 'obj_bound'
    ]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Must use 'spawn' to be safe on Windows
    ctx = mp.get_context('spawn')

    for idx, filepath in enumerate(files, start=1):
        if filepath.name in skip_set:
            print(f"[{idx}/{len(files)}] Skipping {filepath.name} (found in toSkip.txt)")
            continue
            
        print(f"\n[{idx}/{len(files)}] Processing {filepath.name}")

        problem_info = {k: "N/A" for k in fieldnames}
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
            
            # Loading graph can take longer for massive files, so we give it double time
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
        # Compile Gathered Data (Even if partial!)
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

        if worker_res["status"] in ["skipped", "error"] or worker_res["status"].startswith("crash_graph") or worker_res["status"].startswith("timeout_graph"):
            problem_info['config'] = f"{worker_res['status']}: {worker_res['error_msg'][:40]}"
            with open(csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
            continue

        if not has_reductions:
            problem_info['config'] = f"No_Reductions_{worker_res['status']}"
            with open(csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
            print(f"    No reductions found for {filepath.name}. Skipping Gurobi solves.")
            continue

        # 3. Gurobi Solving Logic (Only runs if reductions were found)
        configs = [("Baseline", {})]
        
        if worker_res["d_count"] > 0 or worker_res["i_count"] > 0:
            configs.append(("Elimination_Only", {"DE": worker_res["DE_pairs"], "IE": worker_res["IE_pairs"]}))
            
        # We prefer BFS fixings, but fallback to bitset if BFS timed out but Bitset somehow didn't (unlikely, but safe)
        f0_to_use = worker_res["f0_bfs"] or worker_res["f0_bitset"]
        f1_to_use = worker_res["f1_bfs"] or worker_res["f1_bitset"]
        
        if len(f0_to_use) > 0 or len(f1_to_use) > 0:
            configs.append(("Fixing_Only", {"F0": f0_to_use, "F1": f1_to_use}))
            
        if (worker_res["d_count"] > 0 or worker_res["i_count"] > 0) and (len(f0_to_use) > 0 or len(f1_to_use) > 0):
            configs.append(("All_Preprocessing", {"F0": f0_to_use, "F1": f1_to_use, "DE": worker_res["DE_pairs"], "IE": worker_res["IE_pairs"]}))
        
        try:
            # We ONLY load Gurobi in the main thread! MIP memory is already cleaned up.
            gurobi_base_model = gp.read(str(filepath))
            
            for cfg_name, changes in configs:
                row = problem_info.copy()
                # Document if config is paired with a partial timeout
                row['config'] = f"{cfg_name} (Status: {worker_res['status']})"
                
                try:
                    m_solve = gurobi_base_model.copy()
                    m_solve.setParam("Presolve", 0)
                    m_solve.setParam("TimeLimit", 3600)
                    
                    # Additional presolving disabled
                    m_solve.setParam('AggFill', 0)
                    m_solve.setParam('Aggregate', 0)
                    m_solve.setParam('DualReductions', 0)
                    m_solve.setParam('PreCrush', 0)
                    m_solve.setParam('PreDepRow', 0)
                    m_solve.setParam('PreDual', 0)
                    m_solve.setParam('PreMIQCPForm', 0)
                    m_solve.setParam('PrePasses', 0)
                    m_solve.setParam('PreQLinearize', 0)
                    m_solve.setParam('PreSOS1BigM', 0)
                    m_solve.setParam('PreSOS1Encoding', 0)
                    m_solve.setParam('PreSOS2BigM', 0)
                    m_solve.setParam('PreSOS2Encoding', 0)
                    m_solve.setParam('PreSparsify', 0)
                    m_solve.setParam('Cuts', 0)

        
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
                    print(f"    Config {cfg_name} failed: {e}")
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

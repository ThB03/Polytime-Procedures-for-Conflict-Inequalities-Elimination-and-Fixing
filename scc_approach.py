import os
import time
import csv
import argparse
import gc
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import gurobipy as gp
from mip import Model, MINIMIZE, CBC, ConflictGraph

# Import the model application helper
from main_gurobi import apply_changes_to_model

# ══════════════════════════════════════════════════════════════════════════════
#  Graph Generation (No NetworkX)
# ══════════════════════════════════════════════════════════════════════════════

def create_implication_graph(model: Model) -> Tuple[Dict[str, List[str]], List[str], int, int]:
    """
    Builds the implication graph natively using python dictionaries.
    Completely bypasses NetworkX to prevent massive memory overhead.
    """
    cg = ConflictGraph(model)
    adj = defaultdict(list)
    nodes = []
    m_edges = 0

    # Initialize nodes array to ensure we capture disconnected variables too
    for x in model.vars:
        nodes.append('0' + x.name)
        nodes.append('1' + x.name)

    for x in model.vars:
        x_name = x.name
        
        # xi = 0
        z = cg.conflicting_assignments(x == 0)
        for y in z[0]:
            adj['0' + x_name].append('0' + y.name)
            adj['1' + y.name].append('1' + x_name)
            m_edges += 2
            
        for y in z[1]:
            adj['0' + x_name].append('1' + y.name)
            adj['0' + y.name].append('1' + x_name)
            m_edges += 2

        # xi = 1
        o = cg.conflicting_assignments(x)
        for y in o[0]:
            adj['1' + x_name].append('0' + y.name)
            adj['1' + y.name].append('0' + x_name)
            m_edges += 2
            
        for y in o[1]:
            adj['1' + x_name].append('1' + y.name)
            adj['0' + y.name].append('0' + x_name)
            m_edges += 2

    return adj, nodes, len(nodes), m_edges

# ══════════════════════════════════════════════════════════════════════════════
#  Helper Utilities
# ══════════════════════════════════════════════════════════════════════════════

def stat(node: str) -> int:
    return int(node[0])

def varname(node: str) -> str:
    return node[1:]

class DisjointSetUnion:
    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
    
    def groups(self):
        groups_dict = defaultdict(list)
        for e in self.parent:
            groups_dict[self.find(e)].append(e)
        return groups_dict


# ══════════════════════════════════════════════════════════════════════════════
#  Core SCC Graph Processing (Native Python, ZERO NetworkX)
# ══════════════════════════════════════════════════════════════════════════════

def build_scc_structures(adj: Dict[str, List[str]], nodes: List[str]):
    """
    Finds SCCs and builds the condensed DAG using an Iterative Tarjan's Algorithm.
    Operates strictly on native dictionaries and lists for maximum memory efficiency.
    """
    
    # 1. ITERATIVE TARJAN'S ALGORITHM
    # (Iterative prevents 'RecursionError' on massive graphs of 100k+ vars)
    index = 0
    indices = {}
    lowlink = {}
    on_stack = set()
    stack = []
    sccs = []

    for v in nodes:
        if v not in indices:
            call_stack = [(v, 0)]
            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            on_stack.add(v)

            while call_stack:
                curr, edge_idx = call_stack[-1]
                neighbors = adj.get(curr, [])

                # If there are still neighbors to visit for the current node
                if edge_idx < len(neighbors):
                    w = neighbors[edge_idx]
                    call_stack[-1] = (curr, edge_idx + 1) # Advance pointer for next time
                    
                    if w not in indices:
                        indices[w] = index
                        lowlink[w] = index
                        index += 1
                        stack.append(w)
                        on_stack.add(w)
                        call_stack.append((w, 0)) # "Recurse" into w
                    elif w in on_stack:
                        lowlink[curr] = min(lowlink[curr], indices[w])
                else:
                    # Finished all neighbors, time to pop from call stack
                    call_stack.pop()
                    if call_stack:
                        prev, _ = call_stack[-1]
                        lowlink[prev] = min(lowlink[prev], lowlink[curr])

                    # If curr is the root of an SCC
                    if lowlink[curr] == indices[curr]:
                        scc = []
                        while True:
                            w = stack.pop()
                            on_stack.remove(w)
                            scc.append(w)
                            if w == curr:
                                break
                        sccs.append(scc)

    num_sccs = len(sccs)

    # 2. MAP NODES TO SCCs
    node_to_scc = {}
    # AMAZING TRICK: Tarjan's outputs SCCs in Reverse Topological Order!
    # The first SCC found is guaranteed to be a "sink" (leaf).
    for scc_id, scc_nodes in enumerate(sccs):
        for node in scc_nodes:
            node_to_scc[node] = scc_id
            
    # Free memory instantly
    del sccs 
            
    # Map variable strings back to their states
    var_to_scc = {}
    all_vars_set = set()
    for node in nodes:
        st = int(node[0])
        var = node[1:]
        all_vars_set.add(var)
        if var not in var_to_scc:
            var_to_scc[var] = [None, None]
        var_to_scc[var][st] = node_to_scc[node]
    all_vars = sorted(list(all_vars_set))

    # 3. BUILD CONDENSED DAG
    # Using sets to prevent duplicate edges between SCCs, then converting to lists
    dag_adj_sets = [set() for _ in range(num_sccs)]
    for u, neighbors in adj.items():
        scc_u = node_to_scc[u]
        for v in neighbors:
            scc_v = node_to_scc.get(v)
            if scc_v is not None and scc_u != scc_v:
                dag_adj_sets[scc_u].add(scc_v)

    dag_adj = [list(edges) for edges in dag_adj_sets]
    
    del node_to_scc
    del dag_adj_sets
    
    # Because Tarjan's spits out leaves first and roots last,
    # the SCC IDs (0, 1, 2...) are ALREADY exactly the reverse topological sort!
    reversed_topo = list(range(num_sccs))

    return var_to_scc, all_vars, num_sccs, reversed_topo, dag_adj


# ══════════════════════════════════════════════════════════════════════════════
#  Elimination
# ══════════════════════════════════════════════════════════════════════════════
# (No changes needed here)

def elimination_on_implication_graph_scc(all_vars: List[str], var_to_scc: Dict[str, List[int]]):
    dsu_vars = DisjointSetUnion(all_vars)
    indirect_map_internal = defaultdict(set)
    scc_to_0_vars = defaultdict(list)
    scc_to_1_vars = defaultdict(list)
    
    for v in all_vars:
        scc_0, scc_1 = var_to_scc[v]
        if scc_0 is None or scc_1 is None:
            continue
            
        if scc_0 == scc_1:
            print(f"WARNING: Variable {v} is infeasible (0{v} <=> 1{v}).")
            continue
            
        scc_to_0_vars[scc_0].append(v)
        scc_to_1_vars[scc_1].append(v)

    for scc_id, vars_in_scc in scc_to_0_vars.items():
        if len(vars_in_scc) > 1:
            root_var = vars_in_scc[0]
            for other_var in vars_in_scc[1:]:
                dsu_vars.union(root_var, other_var)
                
    final_groups = dsu_vars.groups()
    direct_eliminations = [set(g) for g in final_groups.values() if len(g) > 1]
    var_to_root = {v: dsu_vars.find(v) for v in all_vars}
    
    for scc_id, vars_0 in scc_to_0_vars.items():
        if scc_id in scc_to_1_vars: 
            vars_1 = scc_to_1_vars[scc_id]
            for v_a in vars_0:
                for v_b in vars_1:
                    if var_to_root[v_a] != var_to_root[v_b]:
                        indirect_map_internal[v_a].add(v_b)
                        indirect_map_internal[v_b].add(v_a)

    indirect_map = {k: list(v) for k, v in indirect_map_internal.items()}
    return dsu_vars, direct_eliminations, indirect_map


# ══════════════════════════════════════════════════════════════════════════════
#  Fixing 
# ══════════════════════════════════════════════════════════════════════════════
# (No changes needed here)

def fixing_on_implication_graph_scc(all_vars: List[str], var_to_scc: Dict[str, List[int]], 
                                    num_sccs: int, reversed_topo: List[int], 
                                    adj: List[List[int]]):
    
    num_words = (num_sccs + 63) // 64
    reach = np.zeros((num_sccs, num_words), dtype=np.uint64)
    
    for i in range(num_sccs):
        word_idx = i // 64
        bit_idx = np.uint64(i % 64)
        reach[i, word_idx] |= (np.uint64(1) << bit_idx)

    for u in reversed_topo:
        for v in adj[u]:
            reach[u] |= reach[v]
            
    fixing_0 = set()
    fixing_1 = set()
    
    for v in all_vars:
        scc_0, scc_1 = var_to_scc[v]
        
        if scc_0 is None or scc_1 is None:
            continue
            
        word_0 = scc_0 // 64
        bit_0 = np.uint64(scc_0 % 64)
        if reach[scc_1, word_0] & (np.uint64(1) << bit_0):
            fixing_0.add(v) 
            continue 
            
        word_1 = scc_1 // 64
        bit_1 = np.uint64(scc_1 % 64)
        if reach[scc_0, word_1] & (np.uint64(1) << bit_1):
            fixing_1.add(v) 
            
    return fixing_0, fixing_1


# ══════════════════════════════════════════════════════════════════════════════
#  Batch Solving Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def find_problem_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.mps"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems_dir", type=str, default="problems")
    parser.add_argument("--max_vars", type=int, default=100000)
    args = parser.parse_args()

    problems_root = Path(args.problems_dir)
    if not problems_root.exists():
        print(f"Problems directory '{args.problems_dir}' not found")
        return

    files = find_problem_files(problems_root)
    
    # Check for toSkip.txt and load instances to skip
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
        'scc_build_time', 'elim_time', 'fix_time',
        'has_reductions', 
        'direct_elim_#', 'indirect_elim_#', 'fixing_0_#', 'fixing_1_#',
        'config', 'solve_status', 'solve_time', 'nodes', 'obj_val', 'obj_bound'
    ]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for idx, filepath in enumerate(files, start=1):
        if filepath.name in skip_set:
            print(f"[{idx}/{len(files)}] Skipping {filepath.name} (found in toSkip.txt)")
            continue
            
        print(f"[{idx}/{len(files)}] Processing {filepath.name}")

        problem_info = {k: "N/A" for k in fieldnames}
        problem_info['problem'] = filepath.name

        try:
            mip_model = Model(sense=MINIMIZE, solver_name=CBC)
            mip_model.read(str(filepath))
            
            n_vars = len(mip_model.vars)
            print(f"  Variables: {n_vars}")
            
            if n_vars > args.max_vars:
                problem_info['config'] = "SKIPPED_MAX_VARS"
                with open(csv_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
                del mip_model
                gc.collect()
                continue

            # --- GRAPH GENERATION ---
            adj_dict, nodes, n_nodes, m_edges = create_implication_graph(mip_model)
            problem_info.update({'n': n_nodes, 'm': m_edges})
            
            try:
                t_scc_start = time.time()
                # Feed the raw dictionary into our optimized function
                var_to_scc, all_vars, num_sccs, reversed_topo, adj = build_scc_structures(adj_dict, nodes)
                t_scc = time.time() - t_scc_start
                
                # Nuke raw graph dictionary immediately
                del adj_dict
                del nodes
                gc.collect()
                
                t_elim_start = time.time()
                dsu_v, d_elim_groups, i_map = elimination_on_implication_graph_scc(all_vars, var_to_scc)
                t_elim = time.time() - t_elim_start
                
                t_fix_start = time.time()
                f0, f1 = fixing_on_implication_graph_scc(all_vars, var_to_scc, num_sccs, reversed_topo, adj)
                t_fix = time.time() - t_fix_start
                
                del var_to_scc
                del all_vars
                del reversed_topo
                del adj
                gc.collect()

                d_count = sum(len(g) - 1 for g in d_elim_groups)
                i_count = sum(len(neighs) for neighs in i_map.values()) // 2
                has_reductions = (d_count > 0 or i_count > 0 or len(f0) > 0 or len(f1) > 0)
                
                problem_info.update({
                    'num_sccs': num_sccs,
                    'scc_build_time': round(t_scc, 6),
                    'elim_time': round(t_elim, 6),
                    'fix_time': round(t_fix, 6),
                    'has_reductions': has_reductions,
                    'direct_elim_#': d_count,
                    'indirect_elim_#': i_count,
                    'fixing_0_#': len(f0),
                    'fixing_1_#': len(f1)
                })

            except Exception as e:
                print(f"    SCC Reduction Pipeline failed: {e}")
                problem_info['config'] = f"REDUCTION_FAIL: {str(e)[:40]}"
                with open(csv_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
                del mip_model
                gc.collect()
                continue

            # 3. Gurobi Solving Logic
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
                        
            del d_elim_groups
            del i_map
            del dsu_v
            gc.collect()

            configs = [("Baseline", {})]
            if d_count > 0 or i_count > 0:
                configs.append(("Elimination_Only", {"DE": DE_pairs, "IE": IE_pairs}))
            if len(f0) > 0 or len(f1) > 0:
                configs.append(("Fixing_Only", {"F0": f0, "F1": f1}))
            if len(configs) == 3:
                configs.append(("All_Preprocessing", {"F0": f0, "F1": f1, "DE": DE_pairs, "IE": IE_pairs}))
            
            if not has_reductions:
                problem_info['config'] = "No_Reductions_Found"
                with open(csv_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
                print(f"   No reductions found for {filepath.name}. Skipping Gurobi solves.")
                del mip_model
                gc.collect()
                continue

            try:
                gurobi_base_model = gp.read(str(filepath))
                for cfg_name, changes in configs:
                    row = problem_info.copy()
                    row['config'] = cfg_name
                    
                    try:
                        m_solve = gurobi_base_model.copy()
                        m_solve.setParam("Presolve", 0)
                        m_solve.setParam("TimeLimit", 3600)

                        apply_changes_to_model(
                            m_solve, 
                            changes.get("F0", set()), 
                            changes.get("F1", set()), 
                            changes.get("DE", []), 
                            changes.get("IE", []), 
                            []
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
                        if 'm_solve' in locals():
                            del m_solve
                            gc.collect()
            except Exception as e:
                print(f"  Gurobi model load failed: {e}")
                problem_info['config'] = "GUROBI_LOAD_FAIL"
                with open(csv_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)

        except Exception as e:
            print(f"  Critical error on {filepath.name}: {e}")
            problem_info['config'] = f"CRITICAL_ERROR: {str(e)[:50]}"
            with open(csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
                
        finally:
            for var in ['mip_model', 'gurobi_base_model', 'DE_pairs', 'IE_pairs', 'f0', 'f1', 'adj_dict']:
                if var in locals():
                    del locals()[var]
            gc.collect()

    print(f"Batch processing complete. Results safely saved in {csv_path}")

if __name__ == "__main__":
    main()

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

# Import the model application helper
from main_gurobi import apply_changes_to_model

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
    reversed_topo = list(range(num_sccs))

    return node_to_scc, num_sccs, reversed_topo, dag_adj

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

def fixing_on_implication_graph_scc(node_to_scc: List[int], n_vars: int, idx_to_var: List[str], 
                                    num_sccs: int, reversed_topo: List[int], 
                                    dag_adj: List[List[int]]):
    """
    Calculates reachability using native Python Arbitrary-Precision Integers.
    Drastically outperforms numpy looping for bitwise operations.
    """
    reach = [1 << i for i in range(num_sccs)]
    
    for u in reversed_topo:
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
    This entire function runs in a completely separate Windows process.
    If it hits a MemoryError and crashes, the main python script survives!
    """
    try:
        from mip import Model, MINIMIZE, CBC
        mip_model = Model(sense=MINIMIZE, solver_name=CBC)
        mip_model.read(filepath_str)
        
        n_vars = len(mip_model.vars)
        if n_vars > max_vars:
            result_queue.put({"status": "skipped", "reason": "SKIPPED_MAX_VARS", "n_vars": n_vars})
            return

        adj, idx_to_var, n_nodes, m_edges = create_implication_graph(mip_model)
        
        # --- NEW LOGIC: Send signal that file reading & graph building is DONE! ---
        result_queue.put({
            "status": "graph_ready",
            "n_vars": n_vars, 
            "n_nodes": n_nodes, 
            "m_edges": m_edges
        })

        t_scc_start = time.time()
        node_to_scc, num_sccs, reversed_topo, dag_adj = build_scc_structures(adj, n_nodes)
        t_scc = time.time() - t_scc_start
        
        t_elim_start = time.time()
        d_elim_groups, i_map = elimination_on_implication_graph_scc(node_to_scc, n_vars, idx_to_var)
        t_elim = time.time() - t_elim_start
        
        t_fix_start = time.time()
        f0, f1 = fixing_on_implication_graph_scc(node_to_scc, n_vars, idx_to_var, num_sccs, reversed_topo, dag_adj)
        t_fix = time.time() - t_fix_start

        d_count = sum(len(g) - 1 for g in d_elim_groups)
        i_count = sum(len(neighs) for neighs in i_map.values()) // 2
        has_reductions = (d_count > 0 or i_count > 0 or len(f0) > 0 or len(f1) > 0)
        
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

        # Send the calculated results back to the main script!
        result_queue.put({
            "status": "success",
            "num_sccs": num_sccs,
            "t_scc": t_scc, "t_elim": t_elim, "t_fix": t_fix,
            "d_count": d_count, "i_count": i_count,
            "has_reductions": has_reductions,
            "f0": f0, "f1": f1, "DE_pairs": DE_pairs, "IE_pairs": IE_pairs
        })

    except Exception as e:
        result_queue.put({"status": "error", "message": str(e)})

# ══════════════════════════════════════════════════════════════════════════════
#  Main Batch Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def find_problem_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.mps"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems_dir", type=str, default="problems")
    parser.add_argument("--max_vars", type=int, default=1000000)
    parser.add_argument("--timeout_prep", type=int, default=3600, help="Max seconds for SCC preprocessing.")
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
        'scc_build_time', 'elim_time', 'fix_time',
        'has_reductions', 
        'direct_elim_#', 'indirect_elim_#', 'fixing_0_#', 'fixing_1_#',
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
            
        print(f"[{idx}/{len(files)}] Processing {filepath.name}")

        problem_info = {k: "N/A" for k in fieldnames}
        problem_info['problem'] = filepath.name

        # ==========================================
        # MULTIPROCESSING TIMEOUT BLOCK (WINDOWS SAFE)
        # ==========================================
        q = ctx.Queue()
        p = ctx.Process(target=preprocess_worker, args=(str(filepath), q, args.max_vars))
        p.start()
        
        # 1. Wait for File Load & Graph Build (Generous 2-hour timeout just in case it hangs entirely)
        try:
            init_msg = q.get(timeout=7200)
        except queue.Empty:
            if p.is_alive():
                print(f"    TIMEOUT: File loading and graph creation exceeded 2 hours. Terminating worker.")
                p.terminate()
                p.join()
                problem_info['config'] = "TIMEOUT_FILE_LOAD"
            else:
                print(f"    CRITICAL ERROR: Worker process crashed silently during load (Likely Out of Memory).")
                problem_info['config'] = "CRITICAL_ERROR: Load Crash"
            
            with open(csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
            continue
            
        # Handle early termination cases (Skip, Error)
        if init_msg["status"] == "skipped":
            print(f"  Variables: {init_msg['n_vars']}")
            problem_info['config'] = init_msg["reason"]
            with open(csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
            continue
            
        elif init_msg["status"] == "error":
            print(f"    Graph Generation failed: {init_msg['message']}")
            problem_info['config'] = f"GRAPH_FAIL: {init_msg['message'][:40]}"
            with open(csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
            continue
            
        elif init_msg["status"] == "graph_ready":
            print(f"  Variables: {init_msg['n_vars']}")
            
            # 2. Graph is built! NOW start the strict timeout just for the SCC algorithms!
            p.join(timeout=args.timeout_prep)

            if p.is_alive():
                print(f"    TIMEOUT: Preprocessing exceeded {args.timeout_prep}s. Terminating worker.")
                p.terminate()
                p.join() # Clean up zombie process
                problem_info['config'] = "TIMEOUT_PREPROCESSING"
                with open(csv_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
                continue

            # If the worker finished (or crashed during SCC), grab the final results
            try:
                final_res = q.get_nowait()
            except queue.Empty:
                print(f"    CRITICAL ERROR: Worker process crashed silently during SCC (Likely Out of Memory).")
                problem_info['config'] = "CRITICAL_ERROR: SCC Crash"
                with open(csv_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
                continue

            # Handle errors thrown during the SCC/Elim/Fixing math
            if final_res["status"] == "error":
                print(f"    SCC Reduction Pipeline failed: {final_res['message']}")
                problem_info['config'] = f"REDUCTION_FAIL: {final_res['message'][:40]}"
                with open(csv_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
                continue

            # --- SUCCESS PATH ---
            problem_info.update({
                'n': init_msg["n_nodes"], 'm': init_msg["m_edges"], 'num_sccs': final_res["num_sccs"],
                'scc_build_time': round(final_res["t_scc"], 6),
                'elim_time': round(final_res["t_elim"], 6),
                'fix_time': round(final_res["t_fix"], 6),
                'has_reductions': final_res["has_reductions"],
                'direct_elim_#': final_res["d_count"],
                'indirect_elim_#': final_res["i_count"],
                'fixing_0_#': len(final_res["f0"]),
                'fixing_1_#': len(final_res["f1"])
            })

            if not final_res["has_reductions"]:
                problem_info['config'] = "No_Reductions_Found"
                with open(csv_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
                print(f"   No reductions found for {filepath.name}. Skipping Gurobi solves.")
                continue

            # 3. Gurobi Solving Logic (Only runs if reductions were found)
            configs = [("Baseline", {})]
            if final_res["d_count"] > 0 or final_res["i_count"] > 0:
                configs.append(("Elimination_Only", {"DE": final_res["DE_pairs"], "IE": final_res["IE_pairs"]}))
            if len(final_res["f0"]) > 0 or len(final_res["f1"]) > 0:
                configs.append(("Fixing_Only", {"F0": final_res["f0"], "F1": final_res["f1"]}))
            if len(configs) == 3:
                configs.append(("All_Preprocessing", {"F0": final_res["f0"], "F1": final_res["f1"], "DE": final_res["DE_pairs"], "IE": final_res["IE_pairs"]}))
            
            try:
                # We ONLY load Gurobi in the main thread! MIP memory is already cleaned up.
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
            except Exception as e:
                print(f"  Gurobi model load failed: {e}")
                problem_info['config'] = "GUROBI_LOAD_FAIL"
                with open(csv_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)

    print(f"Batch processing complete. Results safely saved in {csv_path}")

if __name__ == "__main__":
    mp.freeze_support() # Essential for Windows multiprocessing
    main()

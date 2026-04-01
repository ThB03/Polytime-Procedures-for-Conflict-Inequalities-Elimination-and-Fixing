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
#  Graph Generation & Reductions (Now safe to run in a Worker Process)
# ══════════════════════════════════════════════════════════════════════════════

def create_implication_graph(model) -> Tuple[Dict[str, List[str]], List[str], int, int]:
    from mip import ConflictGraph
    cg = ConflictGraph(model)
    adj = defaultdict(list)
    nodes = []
    m_edges = 0

    for x in model.vars:
        nodes.append('0' + x.name)
        nodes.append('1' + x.name)

    for x in model.vars:
        x_name = x.name
        z = cg.conflicting_assignments(x == 0)
        for y in z[0]:
            adj['0' + x_name].append('0' + y.name)
            adj['1' + y.name].append('1' + x_name)
            m_edges += 2
        for y in z[1]:
            adj['0' + x_name].append('1' + y.name)
            adj['0' + y.name].append('1' + x_name)
            m_edges += 2

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

def build_scc_structures(adj: Dict[str, List[str]], nodes: List[str]):
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

                if edge_idx < len(neighbors):
                    w = neighbors[edge_idx]
                    call_stack[-1] = (curr, edge_idx + 1)
                    if w not in indices:
                        indices[w] = index
                        lowlink[w] = index
                        index += 1
                        stack.append(w)
                        on_stack.add(w)
                        call_stack.append((w, 0))
                    elif w in on_stack:
                        lowlink[curr] = min(lowlink[curr], indices[w])
                else:
                    call_stack.pop()
                    if call_stack:
                        prev, _ = call_stack[-1]
                        lowlink[prev] = min(lowlink[prev], lowlink[curr])

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
    node_to_scc = {}
    for scc_id, scc_nodes in enumerate(sccs):
        for node in scc_nodes:
            node_to_scc[node] = scc_id
            
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

    dag_adj_sets = [set() for _ in range(num_sccs)]
    for u, neighbors in adj.items():
        scc_u = node_to_scc[u]
        for v in neighbors:
            scc_v = node_to_scc.get(v)
            if scc_v is not None and scc_u != scc_v:
                dag_adj_sets[scc_u].add(scc_v)

    dag_adj = [list(edges) for edges in dag_adj_sets]
    reversed_topo = list(range(num_sccs))

    return var_to_scc, all_vars, num_sccs, reversed_topo, dag_adj

def elimination_on_implication_graph_scc(all_vars: List[str], var_to_scc: Dict[str, List[int]]):
    dsu_vars = DisjointSetUnion(all_vars)
    indirect_map_internal = defaultdict(set)
    scc_to_0_vars = defaultdict(list)
    scc_to_1_vars = defaultdict(list)
    
    for v in all_vars:
        scc_0, scc_1 = var_to_scc[v]
        if scc_0 is None or scc_1 is None or scc_0 == scc_1:
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

        adj_dict, nodes, n_nodes, m_edges = create_implication_graph(mip_model)
        
        t_scc_start = time.time()
        var_to_scc, all_vars, num_sccs, reversed_topo, adj = build_scc_structures(adj_dict, nodes)
        t_scc = time.time() - t_scc_start
        
        t_elim_start = time.time()
        dsu_v, d_elim_groups, i_map = elimination_on_implication_graph_scc(all_vars, var_to_scc)
        t_elim = time.time() - t_elim_start
        
        t_fix_start = time.time()
        f0, f1 = fixing_on_implication_graph_scc(all_vars, var_to_scc, num_sccs, reversed_topo, adj)
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
            "n_vars": n_vars, "n_nodes": n_nodes, "m_edges": m_edges, "num_sccs": num_sccs,
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
    parser.add_argument("--max_vars", type=int, default=100000)
    parser.add_argument("--timeout_prep", type=int, default=3600, help="Max seconds for preprocessing.")
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
        
        # Wait up to `timeout_prep` seconds for the worker to finish
        p.join(timeout=args.timeout_prep)

        if p.is_alive():
            print(f"    TIMEOUT: Preprocessing exceeded {args.timeout_prep}s. Terminating worker.")
            p.terminate()
            p.join() # Clean up zombie process
            problem_info['config'] = "TIMEOUT_PREPROCESSING"
            with open(csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
            continue

        # If the worker finished (or crashed), try to get the results
        try:
            res = q.get_nowait()
        except queue.Empty:
            # If the queue is empty, the process died violently (e.g., MemoryError crash)
            print(f"    CRITICAL ERROR: Worker process crashed silently (Likely Out of Memory).")
            problem_info['config'] = "CRITICAL_ERROR: Process Crashed"
            with open(csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
            continue

        # Handle explicit messages sent from the worker
        if res["status"] == "skipped":
            print(f"  Variables: {res['n_vars']}")
            problem_info['config'] = res["reason"]
            with open(csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
            continue

        elif res["status"] == "error":
            print(f"    SCC Reduction Pipeline failed: {res['message']}")
            problem_info['config'] = f"REDUCTION_FAIL: {res['message'][:40]}"
            with open(csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
            continue

        # --- SUCCESS PATH ---
        print(f"  Variables: {res['n_vars']}")
        problem_info.update({
            'n': res["n_nodes"], 'm': res["m_edges"], 'num_sccs': res["num_sccs"],
            'scc_build_time': round(res["t_scc"], 6),
            'elim_time': round(res["t_elim"], 6),
            'fix_time': round(res["t_fix"], 6),
            'has_reductions': res["has_reductions"],
            'direct_elim_#': res["d_count"],
            'indirect_elim_#': res["i_count"],
            'fixing_0_#': len(res["f0"]),
            'fixing_1_#': len(res["f1"])
        })

        if not res["has_reductions"]:
            problem_info['config'] = "No_Reductions_Found"
            with open(csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
            print(f"   No reductions found for {filepath.name}. Skipping Gurobi solves.")
            continue

        # 3. Gurobi Solving Logic (Only runs if reductions were found)
        configs = [("Baseline", {})]
        if res["d_count"] > 0 or res["i_count"] > 0:
            configs.append(("Elimination_Only", {"DE": res["DE_pairs"], "IE": res["IE_pairs"]}))
        if len(res["f0"]) > 0 or len(res["f1"]) > 0:
            configs.append(("Fixing_Only", {"F0": res["f0"], "F1": res["f1"]}))
        if len(configs) == 3:
            configs.append(("All_Preprocessing", {"F0": res["f0"], "F1": res["f1"], "DE": res["DE_pairs"], "IE": res["IE_pairs"]}))
        
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

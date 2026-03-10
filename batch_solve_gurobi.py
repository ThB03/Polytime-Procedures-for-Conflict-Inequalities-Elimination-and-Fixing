import os
import time
import csv
import json
import argparse
import itertools
from pathlib import Path
from datetime import datetime

import numpy as np
import networkx as nx
import gurobipy as gp
from mip import Model, MINIMIZE, CBC

# Optional import for GPU acceleration
try:
    import torch
except ImportError:
    torch = None

# Helpers from existing modules
from implication_graph import (
    create_conflict_graph,
    transitive_closure,
    elimination_on_implication_graph,
    fixing_on_implication_graph,
    elimination_on_implication_graph_torch,
    fixing_on_implication_graph_torch,
    varname
)
from main_gurobi import apply_changes_to_model

def find_problem_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.mps"))

def main():
    parser = argparse.ArgumentParser(description="Batch resilient Gurobi analysis with GPU-accelerated reductions.")
    parser.add_argument("--problems_dir", type=str, default="problems", help="Directory containing .mps files.")
    parser.add_argument("--max_vars", type=int, default=100000, help="Maximum number of variables allowed.")
    args = parser.parse_args()

    problems_root = Path(args.problems_dir)
    if not problems_root.exists():
        print(f"Problems directory '{args.problems_dir}' not found")
        return

    files = find_problem_files(problems_root)
    today = datetime.now().strftime("%Y-%m-%d")
    results_dir = Path(f"results_{today}_batch_resilient")
    results_dir.mkdir(exist_ok=True)
    
    csv_path = results_dir / "batch_results.csv"
    fieldnames = [
        'problem', 'n', 'm', 
        'time_matrix_torch', 'time_bfs', 'time_bfs_torch',
        'has_reductions', 'elim_time', 'fix_time',
        'direct_elim_#', 'indirect_elim_#', 'fixing_0_#', 'fixing_1_#',
        'config', 'solve_status', 'solve_time', 'nodes', 'obj_val', 'obj_bound'
    ]

    # Initialize CSV with header
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    methods = ['matrix_torch', 'bfs', 'bfs_torch']

    for idx, filepath in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] Processing {filepath.name}")
        
        problem_info = {k: "N/A" for k in fieldnames}
        problem_info['problem'] = filepath.name

        try:
            # 1. Model & Graph Creation
            mip_model = Model(sense=MINIMIZE, solver_name=CBC)
            mip_model.read(str(filepath))
            
            n_vars = len(mip_model.vars)
            if n_vars > args.max_vars:
                print(f"  Skipped: {n_vars} vars exceeds limit.")
                problem_info['config'] = "SKIPPED_MAX_VARS"
                with open(csv_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
                continue

            _, G = create_conflict_graph(mip_model)
            nodes = list(G.nodes)
            n_nodes, m_edges = len(nodes), G.number_of_edges()
            index = {node: i for i, node in enumerate(nodes)}
            all_var_names = sorted(list(set(varname(nd) for nd in nodes)))
            num_total_vars = len(all_var_names)
            
            problem_info.update({'n': n_nodes, 'm': m_edges})

            # 2. Closure & Reduction Benchmarking
            method_times = {}
            best_reductions = None # We will store (F0, F1, direct_elim, indirect_map)
            
            for m_name in methods:
                try:
                    start_time = time.time()
                    is_torch = m_name in ('matrix_torch', 'bfs_torch') and torch is not None
                    
                    if is_torch:
                        # GPU pipeline: Closure -> Elim -> Fix (staying on GPU)
                        R = transitive_closure(G, nodes=nodes, method=m_name, reflexive=True, return_matrix=True)
                        t_closure = time.time() - start_time
                        
                        t_elim_start = time.time()
                        dsu_v, d_elim_groups, i_map = elimination_on_implication_graph_torch(R, index, nodes)
                        t_elim = time.time() - t_elim_start
                        
                        t_fix_start = time.time()
                        f0, f1 = fixing_on_implication_graph_torch(R, index, nodes, dsu_vars=dsu_v, indirect_map=i_map)
                        t_fix = time.time() - t_fix_start
                    else:
                        # CPU pipeline
                        reach_dict = transitive_closure(G, nodes=nodes, method=m_name, reflexive=True)
                        t_closure = time.time() - start_time
                        
                        t_elim_start = time.time()
                        dsu_v, d_elim_groups, i_map = elimination_on_implication_graph(G, reach_bitsets=reach_dict, index=index)
                        t_elim = time.time() - t_elim_start
                        
                        t_fix_start = time.time()
                        f0, f1 = fixing_on_implication_graph(G, dsu_vars=dsu_v, indirect_map=i_map, reach_bitsets=reach_dict, index=index)
                        t_fix = time.time() - t_fix_start

                    method_times[f'time_{m_name}'] = round(t_closure, 6)
                    
                    # Track reductions from the first successful method (usually matrix_torch)
                    if best_reductions is None:
                        # Calculate counts based on requirements
                        direct_count = num_total_vars - len(d_elim_groups)
                        indirect_count = sum(len(neighs) for neighs in i_map.values()) // 2
                        
                        best_reductions = (f0, f1, d_elim_groups, i_map, t_elim, t_fix, direct_count, indirect_count)

                except Exception as e:
                    print(f"    Method {m_name} failed: {e}")
                    method_times[f'time_{m_name}'] = "N/A"
            
            problem_info.update(method_times)

            if best_reductions is None:
                raise ValueError("All reduction methods failed.")

            f0, f1, direct_elim, indirect_map, elim_time, fix_time, d_count, i_count = best_reductions
            
            has_reductions = (d_count > 0 or i_count > 0 or len(f0) > 0 or len(f1) > 0)
            
            problem_info.update({
                'has_reductions': has_reductions,
                'elim_time': round(elim_time, 6),
                'fix_time': round(fix_time, 6),
                'direct_elim_#': d_count,
                'indirect_elim_#': i_count,
                'fixing_0_#': len(f0),
                'fixing_1_#': len(f1)
            })

            # 3. Gurobi Solving Logic
            # Build pairs for Gurobi application layer
            DE_pairs = [('1'+v1, '1'+v2) for group in direct_elim for i, v1 in enumerate(sorted(group)) for v2 in sorted(group)[i+1:]]
            IE_pairs = []
            seen_ie = set()
            for a, ns in indirect_map.items():
                for b in ns:
                    pair = tuple(sorted(('1'+a, '1'+b)))
                    if pair not in seen_ie:
                        seen_ie.add(pair)
                        IE_pairs.append(pair)

            configs = [("Baseline", {})]
            if d_count > 0 or i_count > 0:
                configs.append(("Elimination_Only", {"DE": DE_pairs, "IE": IE_pairs}))
            if len(f0) > 0 or len(f1) > 0:
                configs.append(("Fixing_Only", {"F0": f0, "F1": f1}))
            if has_reductions:
                configs.append(("All_Preprocessing", {"F0": f0, "F1": f1, "DE": DE_pairs, "IE": IE_pairs}))

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
                            m_solve, changes.get("F0", set()), changes.get("F1", set()), 
                            changes.get("DE", []), changes.get("IE", []), []
                        )

                        t_start = time.time()
                        m_solve.optimize()
                        solve_duration = round(time.time() - t_start, 4)
                        
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

        except Exception as e:
            print(f"  Critical error on {filepath.name}: {e}")
            problem_info['config'] = f"CRITICAL_ERROR: {str(e)[:50]}"
            with open(csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)

    print(f"Batch processing complete. Results in {csv_path}")

if __name__ == "__main__":
    main()

import os
import time
import csv
from pathlib import Path
import numpy as np
import gurobipy as gp
from mip import Model, MINIMIZE, CBC

# Helpers from existing modules
from implication_graph import (
    create_conflict_graph,
    transitive_closure,
    elimination_on_implication_graph,
    fixing_on_implication_graph,
)
from main_gurobi import apply_changes_to_model

def find_problem_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.mps"))

def main():
    problems_root = Path("problems")
    if not problems_root.exists():
        print("Problems directory not found")
        return

    files = find_problem_files(problems_root)
    today = time.strftime("%Y-%m-%d")
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

    max_vars = 100000
    methods = ['matrix_torch', 'bfs', 'bfs_torch']

    for idx, filepath in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] Processing {filepath.name}")
        if not filepath.name.startswith("air03.mps"):
            continue
        # Prepare the base dictionary with N/A
        problem_info = {k: "N/A" for k in fieldnames}
        problem_info['problem'] = filepath.name

        try:
            # 1. Model & Graph Creation
            mip_model = Model(sense=MINIMIZE, solver_name=CBC)
            mip_model.read(str(filepath))
            
            n_vars = len(mip_model.vars)
            if n_vars > max_vars:
                print(f"  Skipped: {n_vars} vars exceeds limit.")
                problem_info['config'] = "SKIPPED_MAX_VARS"
                with open(csv_path, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(problem_info)
                continue

            _, G = create_conflict_graph(mip_model)
            nodes = list(G.nodes)
            n, m = len(nodes), G.number_of_edges()
            index = {node: i for i, node in enumerate(nodes)}
            problem_info.update({'n': n, 'm': m})

            # 2. Closure Benchmarking
            method_times, reach_bitsets = {}, None
            for m_name in methods:
                try:
                    start = time.time()
                    res = transitive_closure(G, nodes=nodes, method=m_name, reflexive=True)
                    if reach_bitsets is None: reach_bitsets = res
                    method_times[f'time_{m_name}'] = round(time.time() - start, 6)
                except Exception:
                    method_times[f'time_{m_name}'] = "N/A"
            problem_info.update(method_times)

            if reach_bitsets is None: raise ValueError("All closures failed.")

            # 3. Dense Matrix Optimization
            total_reach_edges = sum(len(v) for v in reach_bitsets.values())
            reach_matrix = None
            if total_reach_edges > n * 10:
                try:
                    reach_matrix = np.zeros((n, n), dtype=bool)
                    for u in nodes:
                        ui = index[u]
                        for v in reach_bitsets[u]: reach_matrix[ui, index[v]] = True
                except Exception: pass

            # 4. Reduction Computation
            t_elim_start = time.time()
            dsu_vars, direct_elim, indirect_map = elimination_on_implication_graph(
                G, reach_bitsets=reach_bitsets, reach_matrix=reach_matrix, index=index
            )
            elim_time = round(time.time() - t_elim_start, 6)

            t_fix_start = time.time()
            f0, f1 = fixing_on_implication_graph(
                G, dsu_vars=dsu_vars, indirect_map=indirect_map, 
                reach_bitsets=reach_bitsets, reach_matrix=reach_matrix, index=index
            )
            fix_time = round(time.time() - t_fix_start, 6)

            direct_count = len(direct_elim)
            indirect_count = len(set(tuple(sorted((a, b))) for a, ns in indirect_map.items() for b in ns))
            has_elim = (direct_count > 0 or indirect_count > 0)
            has_fix = (len(f0) > 0 or len(f1) > 0)
            
            problem_info.update({
                'has_reductions': (has_elim or has_fix),
                'elim_time': elim_time, 'fix_time': fix_time,
                'direct_elim_#': direct_count, 'indirect_elim_#': indirect_count,
                'fixing_0_#': len(f0), 'fixing_1_#': len(f1)
            })

            # 5. Gurobi Logic
            configs = [("Baseline", {})]
            if has_elim:
                # Helper to build pairs for apply_changes_to_model
                DE_pairs = [('1'+v1, '1'+v2) for group in direct_elim for i, v1 in enumerate(sorted(group)) for v2 in sorted(group)[i+1:]]
                IE_pairs = list(set(tuple(sorted(('1'+a, '1'+b))) for a, ns in indirect_map.items() for b in ns))
                configs.append(("Elimination_Only", {"DE": DE_pairs, "IE": IE_pairs}))
            
            if has_fix:
                configs.append(("Fixing_Only", {"F0": f0, "F1": f1}))
            
            if has_elim and has_fix:
                configs.append(("All_Preprocessing", {"F0": f0, "F1": f1, "DE": DE_pairs, "IE": IE_pairs}))

            # If no reductions, we still want to log the Baseline/Preprocessing results
            try:
                gurobi_base_model = gp.read(str(filepath))
                for cfg_name, changes in configs:
                    # Prepare a row for this specific configuration attempt
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
                        
                        # Update row with successful solve data
                        row.update({
                            'solve_status': m_solve.Status,
                            'solve_time': solve_duration,
                            'nodes': int(m_solve.NodeCount),
                            'obj_val': m_solve.ObjVal if m_solve.SolCount > 0 else "N/A",
                            'obj_bound': m_solve.ObjBound
                        })
                        
                    except Exception as e:
                        print(f"   Config {cfg_name} failed: {e}")
                        # Mark the solve status as failed in the CSV
                        row['solve_status'] = f"FAILED: {str(e)[:30]}"
                    
                    finally:
                        # This block runs regardless of success or failure
                        with open(csv_path, 'a', newline='') as f:
                            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
            except Exception as e:
                print(f"  Gurobi model load failed: {e}")
                # Log the data we have even if Gurobi fails
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
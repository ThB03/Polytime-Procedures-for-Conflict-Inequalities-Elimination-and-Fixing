import networkx as nx
from mip import *
import time
import csv
import gurobipy as gp
from collections import deque
from pathlib import Path
from datetime import datetime
import os
import itertools
import json
import argparse
# Import implication functions
from implication_graph import create_conflict_graph, transitive_closure, elimination_on_implication_graph, fixing_on_implication_graph
# Import the original, working model-update functions from main_gurobi
from main_gurobi import apply_changes_to_model as main_apply_changes_to_model
from main_gurobi import build_substitution_map as main_build_substitution_map
from main_gurobi import substitute_variables_in_model as main_substitute_variables_in_model
def varname(node: str) -> str:
    return node[1:]

def run_analysis_gurobi(filepath: str, tag: str, max_vars: int, results_dir: str, has_elim: bool, has_fix: bool, method: str):
    # Read with mip and Gurobi
    mip_model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)
    mip_model.read(filepath)
    gurobi_model = gp.read(filepath)

    # Create implication graph
    all_vars = [v.VarName for v in gurobi_model.getVars()]
    total_vars = len(all_vars)
    if total_vars > max_vars:
        print(f"Skipping {filepath} due to variable limit ({total_vars} > {max_vars})")
        return
    _, G = create_conflict_graph(mip_model)

    # Compute transitive closure
    t_closure_start = time.time()
    reach = transitive_closure(G, method=method, reflexive=True)
    t_closure = time.time() - t_closure_start

    # Run elimination
    t_elim_start = time.time()
    dsu_vars, direct_eliminations, indirect_map = elimination_on_implication_graph(G, reach_bitsets=reach) if has_elim else (None, [], {})
    t_elim = time.time() - t_elim_start

    # Run fixing
    t_fix_start = time.time()
    F0, F1 = fixing_on_implication_graph(G, dsu_vars=dsu_vars, indirect_map=indirect_map, reach_bitsets=reach) if has_fix else (set(), set())
    t_fix = time.time() - t_fix_start

    # Translate implication results into the variable-pair format expected by main_gurobi
    # direct_eliminations: list of sets of variable names (e.g. {'x1','x2'})
    # indirect_map: dict var -> list of vars
    DE_pairs = []
    for group in direct_eliminations:
        group_list = sorted(list(group))
        for i in range(len(group_list)):
            for j in range(i + 1, len(group_list)):
                DE_pairs.append(('1'+group_list[i], '1'+group_list[j]))

    IE_pairs = []
    seen = set()
    for a, neighs in indirect_map.items():
        for b in neighs:
            pair = tuple(sorted(('1'+a, '1'+b)))
            if pair not in seen:
                seen.add(pair)
                IE_pairs.append(pair)


    # Configurations
    configs = [("No Preprocessing", {})]
    if has_elim:
        configs.append(("Elimination Only", {"DE": DE_pairs, "IE": IE_pairs}))
    if has_fix:
        configs.append(("Fixing Only", {"F0": F0, "F1": F1}))
    if has_elim and has_fix:
        configs.append(("All Preprocessing", {
            "F0": F0, "F1": F1, "DE": DE_pairs, "IE": IE_pairs
        }))

    results = []

    for name, changes in configs:
        model_copy = gurobi_model.copy()
        model_copy.setParam("TimeLimit", 3600)

        # Disable Gurobi preprocessing
        model_copy.setParam('Presolve', 0)
        model_copy.setParam('AggFill', 0)
        model_copy.setParam('Aggregate', 0)
        model_copy.setParam('DualReductions', 0)
        model_copy.setParam('PreCrush', 0)
        model_copy.setParam('PreDepRow', 0)
        model_copy.setParam('PreDual', 0)
        model_copy.setParam('PreMIQCPForm', 0)
        model_copy.setParam('PrePasses', 0)
        model_copy.setParam('PreQLinearize', 0)
        model_copy.setParam('PreSOS1BigM', 0)
        model_copy.setParam('PreSOS1Encoding', 0)
        model_copy.setParam('PreSOS2BigM', 0)
        model_copy.setParam('PreSOS2Encoding', 0)
        model_copy.setParam('PreSparsify', 0)
        model_copy.setParam('Cuts', 0)

        # Apply changes
        t_apply_start = time.time()
        # Use the original apply_changes implementation from main_gurobi.
        # main_apply_changes_to_model expects (model, F0, F1, DE, IE, AE)
        F0_set = changes.get("F0", set())
        F1_set = changes.get("F1", set())
        DE_arg = changes.get("DE", DE_pairs) if changes.get("DE", None) is not None else DE_pairs
        IE_arg = changes.get("IE", IE_pairs) if changes.get("IE", None) is not None else IE_pairs
        try:
            main_apply_changes_to_model(model_copy, F0_set, F1_set, DE_arg, IE_arg, [])
        except TypeError:
            # fallback if signature differs: try without AE
            main_apply_changes_to_model(model_copy, F0_set, F1_set, DE_arg, IE_arg)
        t_apply = time.time() - t_apply_start

        # Solve model
        try:
            t_solve_start = time.time()
            model_copy.optimize()
            t_solve = time.time() - t_solve_start

            # Extract solution info
            obj_value = obj_bound = obj_gap = node_count = None
            status = model_copy.Status
            node_count = model_copy.NodeCount

            if status == gp.GRB.OPTIMAL:
                obj_value = model_copy.ObjVal
                obj_bound = model_copy.ObjBound
                obj_gap = 0.0
            elif status in [gp.GRB.SUBOPTIMAL, gp.GRB.TIME_LIMIT, gp.GRB.INTERRUPTED]:
                if model_copy.SolCount > 0:
                    obj_value = model_copy.ObjVal
                obj_bound = model_copy.ObjBound
                if obj_value is not None and obj_bound != 0:
                    obj_gap = abs(obj_bound - obj_value) / abs(obj_value)

            nF0 = len(changes.get("F0", set()))
            nF1 = len(changes.get("F1", set()))
            nDE = len(DE_arg) if DE_arg is not None else 0
            nIE = len(IE_arg) if IE_arg is not None else 0
            eliminated = len(set(x for pair in DE_arg + IE_arg for x in pair))
            nAE = 0  # No added edges in this approach

            results.append([
                name, method, nF0, round(100 * nF0 / total_vars, 1) if total_vars > 0 else 0,
                nF1, round(100 * nF1 / total_vars, 1) if total_vars > 0 else 0,
                nDE, nIE, eliminated, round(100 * eliminated / total_vars, 1) if total_vars > 0 else 0,
                nAE,
                round(t_closure, 8),
                round(t_fix, 8) if "Fixing" in name else 0,
                round(t_elim, 8) if "Elimination" in name else 0,
                0,  # No add conflict time
                0,  # No BFS augmented
                0,  # No fix augmented
                0,  # No elim augmented
                round(t_apply, 4),
                round(t_solve, 4),
                obj_value, node_count, obj_bound, obj_gap
            ])
        except Exception as e:
            print(f"Skipping {name} for {filepath} with method {method} due to Gurobi error: {e}")
            continue

    # Save results
    headers = [
        "Configuration", "Method", "#F0", "%F0", "#F1", "%F1",
        "#DE", "#IE", "#Eliminated", "%Eliminated", "#AE",
        "Closure Time", "Fix Time", "Eliminate Time", "AddConflict Time",
        "BFS Time (Augmented)", "Fix Time (Augmented)", "Eliminate Time (Augmented)",
        "ApplyChanges Time", "Solve Time", "Objective Value",
        "#Nodes", "Objective Bound", "Objective Gap"
    ]

    filename = Path(filepath).stem
    csv_path = Path(results_dir) / f"{filename}_{method}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)

    print(f"Results saved to: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Run Gurobi analysis with implication graph from a JSON config file.")
    parser.add_argument("config_path", type=str, help="Path to the JSON config file.")
    parser.add_argument("--max_vars", type=int, default=15000, help="Max number of variables allowed.")
    parser.add_argument("--tag", type=str, default="", help="Optional tag for CSV files.")
    parser.add_argument("--no_elim", action="store_true", help="Disable elimination.")
    parser.add_argument("--no_fix", action="store_true", help="Disable fixing.")
    args = parser.parse_args()

    # Load config
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    # Setup results directory
    today = datetime.now().strftime("%Y-%m-%d")
    tag = f"_{args.tag}" if args.tag else ""
    results_dir = Path(f"results_{today}{tag}_implication")
    results_dir.mkdir(parents=True, exist_ok=True)

    index = 0
    for run_id, run_config in config.items():
        filepath = run_config.get("filepath")
        category = run_config.get("category")

        if not filepath or not category:
            print(f"[!] Skipping {run_id} (missing filepath or category).")
            continue

        try:
            index += 1
            print(f"[{index}] Running {run_id}: {filepath}")

            # Flags from config, overridden by CLI
            has_elim = run_config.get("elimination", True) and not args.no_elim
            has_fix = run_config.get("fixing", True) and not args.no_fix

            for method in ['bfs']:
                run_analysis_gurobi(
                    filepath=filepath,
                    tag=category,
                    max_vars=args.max_vars,
                    results_dir=results_dir,
                    has_elim=has_elim,
                    has_fix=has_fix,
                    method=method
                )

        except Exception as e:
            import traceback
            print(f"[!] Error in {run_id} with method {method}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
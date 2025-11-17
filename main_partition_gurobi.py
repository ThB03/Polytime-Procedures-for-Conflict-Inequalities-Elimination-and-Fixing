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
import nxmetis

def part(node: str) -> str:
    return ('1' if node[0] == '0' else '0') + node[1:]

def stat(node: str) -> int:
    return int(node[0])

def varname(node: str) -> str:
    return node[1:]

def create_conflict_graph(model: Model, parts: int):
    """Returns the conflict graph and a list with partitioned subgraphs."""
    cg = mip.ConflictGraph(model)
    graph = nx.Graph()

    for var in model.vars:
        graph.add_node('0' + var.name)
        graph.add_node('1' + var.name)

    for var in model.vars:
        zero_conflicts = cg.conflicting_assignments(var == 0)
        for conflict_var in zero_conflicts[0]:
            graph.add_edge('0' + var.name, '1' + conflict_var.name)
        for conflict_var in zero_conflicts[1]:
            graph.add_edge('0' + var.name, '0' + conflict_var.name)

        one_conflicts = cg.conflicting_assignments(var)
        for conflict_var in one_conflicts[0]:
            graph.add_edge('1' + var.name, '1' + conflict_var.name)
        for conflict_var in one_conflicts[1]:
            graph.add_edge('1' + var.name, '0' + conflict_var.name)

    parts = max(1, parts)
    node_count = graph.number_of_nodes()
    if parts == 1 or node_count == 0:
        return graph, [graph]

    partitions = []
    try:
        _, node_groups = nxmetis.partition(graph, min(parts, node_count))
        for nodes in node_groups:
            if not nodes:
                continue
            partitions.append(graph.subgraph(nodes).copy())
    except Exception:
        partitions = [graph]

    if not partitions:
        partitions = [graph]
    print(f"Graph partitioned into {len(partitions)} parts.")
    return graph, partitions


def write_status_file(csv_path: Path, status: str, message: str):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Status", "Message"])
        writer.writerow([status, message])

def modified_bfs(G: nx.Graph, s: str):
    start_time = time.time()
    dist = {v : -1 for v in G.nodes}
    pred = {v : v for v in G.nodes}

    for v in G.nodes:
        dist[v] = -1
        pred[v] = -1

    dist[s] = 0
    dist[part(s)] = 1
    pred[part(s)] = s

    if G.has_node(part(s)): 
        Q = deque([part(s)])

        while Q:
            v = Q.popleft()
            for u in G.neighbors(v):
                if u == part(v) or dist[u] > -1:
                    continue
                dist[u] = dist[v] + 1
                pred[u] = v
                if G.has_node(part(u)):
                    dist[part(u)] = dist[v] + 2
                    pred[part(u)] = u
                    Q.append(part(u))

    end_time = time.time()

    return dist, pred, end_time - start_time

def fix(G: nx.Graph, dist):
    start_time = time.time()

    F0, F1 = set(), set()

    for u in G.nodes:
        for s, t in itertools.combinations(G.neighbors(u), 2):
            if s == part(u) or t == part(u):
                continue
            if dist[s][t] < 0:
                continue
            if dist[s][t] % 2 == 0:
                continue

            if stat(u) == 1:
                F0.add(varname(u))
            elif stat(u) == 0:
                F1.add(varname(u))

    end_time = time.time()
    return F0, F1, end_time - start_time

def eliminate(G: nx.Graph, dist, pred):
    start_time = time.time()
    DE, IE = set(), set()

    for s in G.nodes:
        for t in G.neighbors(s):
            if t == part(s):
                continue
            if dist[s][t] < 0:
                continue
            if dist[s][t] % 2 == 0:
                continue

            u = t
            while u != s:
                if dist[s][u]%2 == 1:
                    u = pred[s][u]
                    continue

                if stat(s) == stat(u):
                        DE.add((s, u))
                else:
                    if s != part(u) and (u, s) not in IE:
                        IE.add((s, u))
                    
                if pred[s][u] == u:
                    u = s
                else:
                    u = pred[s][u]

    end_time = time.time()
    return DE, IE, end_time - start_time


def add_conflict(G: nx.Graph, dist, stat):
    start_time = time.time()
    AE = set()

    for s in G.nodes:
        for t in G.nodes:
            if dist[s][t] < 0 or dist[s][t]% 2 == 0:
                continue
            edge = (part(s),part(t))
            if (edge not in G.edges) and (edge not in AE) and ((part(t),part(s)) not in AE):
                AE.add(edge)

    end_time = time.time()
    return AE, end_time - start_time


# ANS
def ANS(graph: nx.graph, varsNames: list):

    indirect, time = SimpleIndirectElim(graph, varsNames)

    conflicts = ANSConflicts(graph, indirect)
    elimination = ANSElimination(graph, indirect)
    fixing = ANSFixing(graph, indirect)


    return conflicts, elimination, fixing, time

def SimpleIndirectElim(graph: nx.graph, varsNames: list):
    startTime = time.time()
    pointer1 = 0
    pointer2 = 1
    ie = set()
    while pointer1 < len(varsNames):
        pointer2 = pointer1 + 1
        while pointer2 < len(varsNames):
            if graph.has_edge('0x' + varsNames[pointer1], '0x' + varsNames[pointer2]) and graph.has_edge('1x' + varsNames[pointer1], '1x' + varsNames[pointer2]):
                ie.add([varsNames[pointer1], varsNames[pointer2]])
            pointer2 += 1
        pointer1 += 1
    
    return ie, time.time() - startTime


def ANSConflicts(graph: nx.graph, ie: set):
    startTime = time.time()
    newConflicts = set()
    for pair in ie:
        for l in graph.neighbors('0x' + pair[0]):
            for k in graph.neighbors('0x' + pair[1]):
                if not ((l,k) in newConflicts or (k,l) in newConflicts) and not graph.has_edge(l,k):
                    newConflicts.add((l,k))

    for pair in ie:
        for l in graph.neighbors('1x' + pair[0]):
            for k in graph.neighbors('1x' + pair[1]):
                if not ((l,k) in newConflicts or (k,l) in newConflicts) and not graph.has_edge(l,k):
                    newConflicts.add((l,k))
    return newConflicts, time.time() - startTime

def ANSElimination(graph: nx.graph, ie: set):
    startTime = time.time()
    indirectElimination = set()
    directElimination = set()
    for pair1 in ie:
        for pair2 in ie:
            if pair1==pair2: continue
            if graph.has_edge('0x' + pair1[0], '0x' + pair2[0]) and graph.has_edge('0x' + pair1[1], '0x' + pair2[1]):
                indirectElimination.add(pair1[0], pair2[0])
                indirectElimination.add(pair1[1], pair2[1])
                directElimination.add(pair1[0], pair2[1])
                directElimination.add(pair1[1], pair2[0])

            if graph.has_edge('0x' + pair1[1], '0x' + pair2[0]) and graph.has_edge('0x' + pair1[0], '0x' + pair2[1]):
                indirectElimination.add(pair1[1], pair2[0])
                indirectElimination.add(pair1[0], pair2[1])
                directElimination.add(pair1[0], pair2[0])
                directElimination.add(pair1[1], pair2[1])
    
    return indirectElimination, directElimination, time.time() - startTime

def ANSFixing(graph: nx.graph, ie: set):
    startTime = time.time()
    zeroFixing = set()
    oneFixing = set()
    for pair in ie:
        node1 = '1x' + pair[0]
        node2 = '1x' + pair[1]

        for neighbor in graph.neighbors(node1):
            if graph.has_edge(node2, neighbor):
                if neighbor[0] == '0':
                    oneFixing.add(neighbor[1:])
                else:
                    zeroFixing.add(neighbor[1:])

    for pair in ie:
        node1 = '0x' + pair[0]
        node2 = '0x' + pair[1]

        for neighbor in graph.neighbors(node1):
            if graph.has_edge(node2, neighbor):
                if neighbor[0] == '0':
                    oneFixing.add(neighbor[1:])
                else:
                    zeroFixing.add(neighbor[1:])
    
    return zeroFixing, oneFixing, time.time() - startTime

def build_substitution_map(DE, IE):
    """
    Builds a substitution map based on DE (direct eelimination) and IE (indirect elimination).
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
    from gurobipy import LinExpr
    
    variables = {v.VarName: v for v in model.getVars()}

    for var_name, (rep_name, flip) in substitution_map.items():
        if var_name not in variables or rep_name not in variables:
            continue

        var = variables[var_name]
        rep = variables[rep_name]

        # Substitute in all constraints
        for constr in model.getConstrs():
            expr = model.getRow(constr)
            coeff = expr.getCoeff(var)
            if coeff == 0:
                continue

            model.chgCoeff(constr, var, 0)
            if flip == 0:
                model.chgCoeff(constr, rep, expr.getCoeff(rep) + coeff)
            else:
                # x_var = 1 - x_rep ⇒ x_var = 1 - x ⇒ c·x_var → c·(1 - x_rep)
                model.chgCoeff(constr, rep, expr.getCoeff(rep) - coeff)
                rhs = model.getAttr("RHS", [constr])[0]
                sense = constr.Sense
                if sense == '=':
                    model.setAttr("RHS", [constr], [rhs - coeff])
                elif sense == '<':
                    model.setAttr("RHS", [constr], [rhs - coeff])
                elif sense == '>':
                    model.setAttr("RHS", [constr], [rhs - coeff])

        # Substitute in objective
        obj_coeff = var.Obj
        if obj_coeff != 0:
            if flip == 0:
                rep.Obj += obj_coeff
            else:
                rep.Obj -= obj_coeff
                model.ModelSense = -model.ModelSense  # flip sense if needed (depending on objective semantics)
            var.Obj = 0

        # Remove the eliminated variable from the model
        model.remove(var)

    model.update()

def apply_changes_to_model(model, F0, F1, DE, IE, AE):
    """
    Applies all known reductions and constraints to a Gurobi model:
    - Fixes variables from F0 and F1
    - Substitutes variables based on DE and IE (eliminates)
    - Adds constraints for AE (conflict edges)
    """

    # ------------------------
    # 1. Fix variables in F0 and F1
    fixed_vars = {}
    for var in model.getVars():
        if var.VarName in F0:
            var.LB = 0
            var.UB = 0
            fixed_vars[var.VarName] = 0
        elif var.VarName in F1:
            var.LB = 1
            var.UB = 1
            fixed_vars[var.VarName] = 1

    # ------------------------
    # 2. Eliminate variables from DE and IE
    substitution_map = build_substitution_map(DE, IE)
    substitute_variables_in_model(model, substitution_map)

    # ------------------------
    # 3. Add constraints for AE (conflict edges)
    vars_dict = {v.VarName: v for v in model.getVars()}

    for u, v in AE:
        su, sv = stat(u), stat(v)
        vu = vars_dict.get(varname(u))
        vv = vars_dict.get(varname(v))

        if vu is None or vv is None:
            continue

        if su == 0 and sv == 0:
            # x_u + x_v >= 1 ⇒ -x_u - x_v ≤ -1
            model.addConstr(vu + vv >= 1, name=f"conflict_{u}_{v}")
        elif su == 1 and sv == 1:
            # (1 - x_u) + (1 - x_v) >= 1 ⇒ x_u + x_v ≤ 1
            model.addConstr(vu + vv <= 1, name=f"conflict_{u}_{v}")
        elif su == 0 and sv == 1:
            # x_u + (1 - x_v) >= 1 ⇒ x_u - x_v >= 0 ⇒ x_u ≥ x_v
            model.addConstr(vu >= vv, name=f"conflict_{u}_{v}")
        elif su == 1 and sv == 0:
            # (1 - x_u) + x_v ≥ 1 ⇒ -x_u + x_v ≥ 0 ⇒ x_v ≥ x_u
            model.addConstr(vv >= vu, name=f"conflict_{u}_{v}")

    model.update()
    return {
        "fixed": fixed_vars,
        "substitution_map": substitution_map
    }

def run_analysis_gurobi(
    filepath: str,
    tag: str,
    max_vars: int,
    results_dir: str,
    has_elim: bool,
    has_fix: bool,
    partition_threshold: int
):
    # Read with mip and Gurobi
    mip_model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)
    mip_model.read(filepath)
    gurobi_model = gp.read(filepath)

    # Create conflict graph
    all_vars = [v.VarName for v in gurobi_model.getVars()]
    total_vars = len(all_vars)
    filename = Path(filepath).stem
    csv_path = Path(results_dir) / f"{filename}.csv"

    if total_vars > max_vars:
        print(f"Skipping {filepath} due to variable limit ({total_vars} > {max_vars})")
        write_status_file(
            csv_path,
            "SKIPPED",
            f"Variables {total_vars} exceed limit {max_vars}"
        )
        return

    write_status_file(csv_path, "RUNNING", f"Started at {datetime.now().isoformat()}")

    try:
        threshold = max(1, partition_threshold)
        partition_count = (len(mip_model.vars) // threshold) + 1
        full_graph, partitions = create_conflict_graph(mip_model, partition_count)
        if not partitions:
            partitions = [full_graph]

        processed_partitions = []
        partition_edges = []
        F0_orig, F1_orig = set(), set()
        DE_orig, IE_orig = set(), set()
        AE = set()

        t_bfs = t_fix = t_elim = t_conf = 0.0

        for subgraph in partitions:
            if subgraph.number_of_nodes() == 0:
                continue

            processed_partitions.append(subgraph)
            dist_all = {}
            pred_all = {}
            start_bfs = time.time()
            for node in subgraph.nodes:
                dist, pred, _ = modified_bfs(subgraph, node)
                dist_all[node] = dist
                pred_all[node] = pred
            t_bfs += time.time() - start_bfs

            if has_fix:
                f0, f1, fix_time = fix(subgraph, dist_all)
                F0_orig |= set(f0)
                F1_orig |= set(f1)
                t_fix += fix_time
            if has_elim:
                de, ie, elim_time = eliminate(subgraph, dist_all, pred_all)
                DE_orig |= set(de)
                IE_orig |= set(ie)
                t_elim += elim_time

            if has_fix or has_elim:
                ae, conf_time = add_conflict(subgraph, dist_all, stat)
            else:
                ae, conf_time = set(), 0
            partition_edges.append(ae)
            AE |= set(ae)
            t_conf += conf_time

        has_conf = bool(AE)

        t_bfs_aug = t_fix_aug = t_elim_aug = 0.0
        F0_all, F1_all = set(), set()
        DE_all, IE_all = set(), set()

        if has_fix or has_elim or has_conf:
            for subgraph, ae in zip(processed_partitions, partition_edges):
                augmented_graph = subgraph.copy()
                augmented_graph.add_edges_from(ae)

                dist_aug = {}
                pred_aug = {}
                start_bfs_aug = time.time()
                for node in augmented_graph.nodes:
                    dist, pred, _ = modified_bfs(augmented_graph, node)
                    dist_aug[node] = dist
                    pred_aug[node] = pred
                t_bfs_aug += time.time() - start_bfs_aug

                if has_fix:
                    f0_aug, f1_aug, fix_time_aug = fix(augmented_graph, dist_aug)
                    F0_all |= set(f0_aug)
                    F1_all |= set(f1_aug)
                    t_fix_aug += fix_time_aug
                if has_elim:
                    de_aug, ie_aug, elim_time_aug = eliminate(augmented_graph, dist_aug, pred_aug)
                    DE_all |= set(de_aug)
                    IE_all |= set(ie_aug)
                    t_elim_aug += elim_time_aug

        # Configurations (skip comparisons when no changes)
        has_elim_changes = has_elim and (bool(DE_orig) or bool(IE_orig))
        has_fix_changes = has_fix and (bool(F0_orig) or bool(F1_orig))
        has_conf_changes = bool(AE)
        any_changes = has_elim_changes or has_fix_changes or has_conf_changes

        configs = [("No Preprocessing", {})]
        if has_elim_changes:
            configs.append(("Elimination Only", {"DE": DE_orig, "IE": IE_orig}))
        if has_fix_changes:
            configs.append(("Fixing Only", {"F0": F0_orig, "F1": F1_orig}))
        if has_conf_changes:
            configs.append(("Added Edges Only", {"AE": AE}))
        if any_changes:
            configs.append(("All Preprocessing", {
                "F0": F0_all, "F1": F1_all, "DE": DE_all, "IE": IE_all, "AE": AE
            }))
        else:
            print("No elimination, fixing, or added-edge changes detected; skipping comparative runs.")

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
            apply_changes_to_model(
                model_copy,
                F0=changes.get("F0", []),
                F1=changes.get("F1", []),
                DE=changes.get("DE", []),
                IE=changes.get("IE", []),
                AE=changes.get("AE", [])
            )
            t_apply = time.time() - t_apply_start

            # Solve model
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

            # Stats
            nF0 = len(changes.get("F0", []))
            nF1 = len(changes.get("F1", []))
            nDE = len(changes.get("DE", []))
            nIE = len(changes.get("IE", []))
            eliminated = len(set(x[0] for x in changes.get("DE", []) or changes.get("IE", [])))
            nAE = len(changes.get("AE", []))

            results.append([
                name, nF0, round(100 * nF0 / total_vars, 1),
                nF1, round(100 * nF1 / total_vars, 1),
                nDE, nIE, eliminated, round(100 * eliminated / total_vars, 1),
                nAE,
                round(t_bfs, 4),
                round(t_fix, 4) if "Fixing" in name else 0,
                round(t_elim, 4) if "Elimination" in name else 0,
                round(t_conf, 4) if "AE" in changes else 0,
                round(t_bfs_aug, 4) if name == "All Preprocessing" else 0,
                round(t_fix_aug, 4) if name == "All Preprocessing" else 0,
                round(t_elim_aug, 4) if name == "All Preprocessing" else 0,
                round(t_apply, 4),
                round(t_solve, 4),
                obj_value, node_count, obj_bound, obj_gap
            ])

        # Save results
        headers = [
            "Configuration", "#F0", "%F0", "#F1", "%F1",
            "#DE", "#IE", "#Eliminated", "%Eliminated", "#AE",
            "BFS Time", "Fix Time", "Eliminate Time", "AddConflict Time",
            "BFS Time (Augmented)", "Fix Time (Augmented)", "Eliminate Time (Augmented)",
            "ApplyChanges Time", "Solve Time", "Objective Value",
            "#Nodes", "Objective Bound", "Objective Gap"
        ]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(results)

        print(f"Results saved to: {csv_path}")
    except Exception as exc:
        write_status_file(csv_path, "ERROR", str(exc)[:500])
        raise

def main():
    parser = argparse.ArgumentParser(description="Run Gurobi analysis from a JSON config file.")
    parser.add_argument("config_path", type=str, help="Path to the JSON config file.")
    parser.add_argument("--max_vars", type=int, default=10000000, help="Max number of variables allowed.")
    parser.add_argument("--tag", type=str, default="", help="Optional tag for CSV files.")
    parser.add_argument("--no_elim", action="store_true", help="Disable elimination.")
    parser.add_argument("--no_fix", action="store_true", help="Disable fixing.")
    parser.add_argument("--no_conf", action="store_true", help="Disable conflict edge addition.")
    parser.add_argument(
        "--partition-threshold",
        type=int,
        default=15000,
        help="Maximum number of variables assigned to a single partition when building the conflict graph."
    )
    args = parser.parse_args()

    # Load config
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    # Setup results directory
    today = datetime.now().strftime("%Y-%m-%d")
    tag = f"_{args.tag}" if args.tag else ""
    results_dir = Path(f"results_{today}{tag}")
    results_dir.mkdir(parents=True, exist_ok=True)

    default_partition_threshold = max(1, args.partition_threshold)
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
            has_conf = not args.no_conf
            partition_threshold = run_config.get("partition_threshold", default_partition_threshold)
            try:
                partition_threshold = int(partition_threshold)
            except (TypeError, ValueError):
                partition_threshold = default_partition_threshold
            partition_threshold = max(1, partition_threshold)

            run_analysis_gurobi(
                filepath=filepath,
                tag=category,
                max_vars=args.max_vars,
                results_dir=results_dir,
                has_elim=has_elim,
                has_fix=has_fix,
                partition_threshold=partition_threshold
            )

        except Exception as e:
            import traceback
            print(f"[!] Error in {run_id}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()

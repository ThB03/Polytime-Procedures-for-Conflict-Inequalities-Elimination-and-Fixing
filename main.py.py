import networkx as nx
from mip import *
import time
import csv
from collections import deque
from pathlib import Path
from datetime import datetime
import os
import itertools
import json
import argparse

def part(node: str) -> str:
    return ('1' if node[0] == '0' else '0') + node[1:]

def stat(node: str) -> int:
    return int(node[0])

def varname(node: str) -> str:
    return node[1:]

def create_conflict_graph(model: Model):
    # Returns the conflict graph for the model and a list with all partitions

    cg = mip.ConflictGraph(model)
    g = nx.Graph()
    count = 0
    names = []
    for x in model.vars:
        g.add_node('0' + x.name)
        g.add_node('1' + x.name)
        names.append(x.name)
        count+= 1


    for x in model.vars:
        # xi = 0
        z = cg.conflicting_assignments(x == 0)
        for y in z[0]:
            g.add_edge('0'+ x.name, '1' + y.name)
            
        for y in z[1]:
            g.add_edge('0'+ x.name, '0' + y.name)
        # xi = 1
        o = cg.conflicting_assignments(x)
        for y in o[0]:
            g.add_edge('1'+ x.name, '1' + y.name)
        
        for y in o[1]:
            g.add_edge('1'+ x.name, '0' + y.name)

    return g

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
                    if s != part(u) and (u[1:], s[1:]) not in DE:
                        DE.add((s[1:], u[1:]))
                else:
                    if s != part(u) and (u[1:], s[1:]) not in IE:
                        IE.add((s[1:], u[1:]))
                    
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


def add_conflicts(graph: nx.Graph, AE):
    for u, v in AE:
        graph.add_edge(u,v)

def analyze_model(model: Model, tag: str, has_fix: bool = True, has_elim: bool = True):
    # Step 1: Build conflict graph
    graph = create_conflict_graph(model)
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    fixing_quantity = []
    fixing_time = []
    elimination_quantity = []
    elimination_time = []
    conflict_quantity = []
    conflict_time = []

    for i in range(0,3):
        total_F0 = set()
        total_F1 = set()
        total_DE = set()
        total_IE = set()
        total_AE = set()
        total_F0_nem = set()
        total_F1_nem = set()
        total_DE_nem = set()
        total_IE_nem = set()
        total_AE_nem = set()

        total_bfs_time = 0
        total_fixing_time = 0
        total_elimination_time = 0
        total_conflict_time = 0
        total_nem_fixing_time = 0
        total_nem_elimination_time = 0
        total_nem_conflict_time = 0
        total_nem_preprocess_time = 0

        # Step 2: Compute distances and predecessors using modified BFS
        dist_all = {}
        pred_all = {}
        start_bfs = time.time()
        for node in graph.nodes:
            dist, pred, _ = modified_bfs(graph, node)
            dist_all[node] = dist
            pred_all[node] = pred
        bfsTime = time.time() - start_bfs
        total_bfs_time += bfsTime


        # Step 3: Improved techniques
        
        # Finding conflicts and updating the graph
        ae, conflictTime = add_conflict(graph, dist_all, stat)
        add_conflicts(graph, ae)

        if has_fix: f0, f1, fixingTime = fix(graph, dist_all)
        if has_elim: de, ie, eliminationTime = eliminate(graph, dist_all, pred_all)

        if has_fix: total_fixing_time += fixingTime
        if has_elim: total_elimination_time += eliminationTime
        total_conflict_time += conflictTime

        # Step 4: ANS methods
        var_names = list(set(var.name for var in model.vars))
        nem_conflicts, nem_eliminations, nem_fixing, nem_preprocess = ANS(graph, var_names)

        if has_fix: nem_F0, nem_F1, nem_F_time = nem_fixing
        if has_elim: nem_IE, nem_DE, nem_E_time = nem_eliminations
        nem_AE, nem_C_time = nem_conflicts

        if has_fix: total_nem_fixing_time += nem_F_time
        if has_elim: total_nem_elimination_time += nem_E_time
        total_nem_conflict_time += nem_C_time
        total_nem_preprocess_time += nem_preprocess

        if has_fix: total_F0 = total_F0 | f0
        if has_fix: total_F1 = total_F1 | f1
        if has_elim: total_DE = total_DE | de
        if has_elim: total_IE = total_IE | ie
        total_AE = total_AE | ae
        if has_fix: total_F0_nem = total_F0_nem | nem_F0
        if has_fix: total_F1_nem = total_F1_nem | nem_F1
        if has_elim: total_DE_nem = total_DE_nem | nem_DE
        if has_elim: total_IE_nem = total_IE_nem | nem_IE
        total_AE_nem = total_AE_nem | nem_AE

        # Results per type
        fixing_quantity.append({
            "category": tag,
            "n": n,
            "m": m,
            "HSC_zero_fixing": len(total_F0),
            "ANS_zero_fixing": len(total_F0_nem),
            "imp_zero_fixing (%)": ((len(total_F0) - len(total_F0_nem)) / len(total_F0_nem)) * 100 if total_F0_nem else 'N/A',
            "HSC_one_fixing": len(total_F1),
            "ANS_one_fixing": len(total_F1_nem),
            "imp_one_fixing (%)": ((len(total_F1) - len(total_F1_nem)) / len(total_F1_nem)) * 100 if total_F1_nem else 'N/A',
        })
        fixing_time.append({
            "category": tag,
            "n": n,
            "m": m,
            "bfs_time (s)": total_bfs_time,
            "ANS_preprocess_time (s)": total_nem_preprocess_time,
            "HSC_fixing_time (s)": total_fixing_time,
            "ANS_fixing_time (s)": total_nem_fixing_time,
        })

        elimination_quantity.append({
            "category": tag,
            "n": n,
            "m": m,
            "HSC_direct_elimination": len(total_DE),
            "ANS_direct_elimination": len(total_DE_nem),
            "imp_direct_elimination (%)": ((len(total_DE) - len(total_DE_nem)) / len(total_DE_nem)) * 100 if total_DE_nem else 'N/A',
            "HSC_indirect_elimination": len(total_IE),
            "ANS_indirect_elimination": len(total_IE_nem),
            "imp_indirect_elimination (%)": ((len(total_IE) - len(total_IE_nem)) / len(total_IE_nem)) * 100 if total_IE_nem else 'N/A',
        })
        elimination_time.append({
            "category": tag,
            "n": n,
            "m": m,
            "bfs_time (s)": total_bfs_time,
            "ANS_preprocess_time (s)": total_nem_preprocess_time,
            "HSC_elimination_time (s)": total_elimination_time,
            "ANS_elimination_time (s)": total_nem_elimination_time,
        })

        conflict_quantity.append({
            "category": tag,
            "n": n,
            "m": m,
            "HSC_conflicts": len(total_AE),
            "ANS_conflicts": len(total_AE_nem),
            "imp_conflicts (%)": ((len(total_AE) - len(total_AE_nem)) / len(total_AE_nem)) * 100 if total_AE_nem else 'N/A',
        })
        conflict_time.append({
            "category": tag,
            "n": n,
            "m": m,
            "bfs_time (s)": total_bfs_time,
            "ANS_preprocess_time (s)": total_nem_preprocess_time,
            "HSC_conflict_time (s)": total_conflict_time,
            "ANS_conflict_time (s)": total_nem_conflict_time,
        })

    return fixing_quantity, fixing_time, elimination_quantity, elimination_time, conflict_quantity, conflict_time

def save_results(output_csv: str, model_name: str, results: dict):
    results_with_instance = {"instance": model_name}
    results_with_instance.update(results)

    write_header = not os.path.exists(output_csv)
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results_with_instance.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(results_with_instance)


def run_analysis(filepath: str, tag: str, max_vars: int, results_dir: str, has_elim: bool, has_fix: bool):
    model_name = os.path.basename(filepath)
    print(f"Analyzing {filepath} under tag {tag}")

    model = Model(sense=minimize, solver_name=CBC)
    model.read(filepath)

    if len(model.vars) > max_vars:
        print(f"Skipping {model_name} due to var count ({len(model.vars)}) > {max_vars}")
        return

    (fixing_quantity, fixing_time,
     elimination_quantity, elimination_time,
     conflict_quantity, conflict_time) = analyze_model(model, tag, has_elim=has_elim, has_fix=has_fix)

    for i in range(3):
        save_results(os.path.join(results_dir, f"fixing_quantity_g{i}.csv"), model_name, fixing_quantity[i])
        save_results(os.path.join(results_dir, f"fixing_time_g{i}.csv"), model_name, fixing_time[i])
        save_results(os.path.join(results_dir, f"elimination_quantity_g{i}.csv"), model_name, elimination_quantity[i])
        save_results(os.path.join(results_dir, f"elimination_time_g{i}.csv"), model_name, elimination_time[i])
        save_results(os.path.join(results_dir, f"conflict_quantity_g{i}.csv"), model_name, conflict_quantity[i])
        save_results(os.path.join(results_dir, f"conflict_time_g{i}.csv"), model_name, conflict_time[i])

def main():
    parser = argparse.ArgumentParser(description="Run analysis from a config file.")
    parser.add_argument("config_path", type=str, help="Path to the JSON config file.")
    parser.add_argument("--max-vars", type=int, default=15000, help="Maximum number of variables allowed.")
    args = parser.parse_args()

    config_base = os.path.splitext(os.path.basename(args.config_path))[0]
    today = datetime.now().strftime("%Y-%m-%d")
    results_dir = os.path.join(f"results_{today}_{config_base}")
    os.makedirs(results_dir, exist_ok=True)

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    index = 0
    for run_id, run_config in config.items():
        filepath = run_config.get('filepath')
        tag = run_config.get('category')
        has_elim = run_config.get('elimination', False)
        has_fix = run_config.get('fixing', False)

        if not filepath or not tag:
            print(f"Skipping {run_id}: missing filepath or category.")
            continue

        index += 1
        try:
            print(f"Running index {index} [{tag}] -> {filepath}")
            run_analysis(filepath, tag, args.max_vars, results_dir, has_elim, has_fix)
        except MemoryError as e:
            print(f"MemoryError on {filepath}: {e}")
            raise e
        except Exception as e:
            print(f"Error on {filepath}: {e}")
            raise e

if __name__ == "__main__":
    main()
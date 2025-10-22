import os
import time
import json
import csv
import itertools
from collections import deque
from datetime import datetime
from pathlib import Path

import networkx as nx
import mip

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

from mip import *


# -------------------------
# Device diagnostics
# -------------------------

def cuda_diagnostics():
    info = {}
    info["torch_installed"] = bool(_HAS_TORCH)
    if _HAS_TORCH:
        try:
            info["torch_version"] = getattr(torch, "__version__", "?")
            info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
            info["cuda_available"] = torch.cuda.is_available()
            info["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if torch.cuda.is_available():
                try:
                    info["device_name_0"] = torch.cuda.get_device_name(0)
                except Exception as e:
                    info["device_name_0"] = f"error: {e}"
        except Exception as e:
            info["diagnostic_error"] = str(e)
    return info

def print_cuda_diagnostics(prefix="[CUDA]"):
    info = cuda_diagnostics()
    print(f"{prefix} torch_installed={info.get('torch_installed')} torch_version={info.get('torch_version')}")
    print(f"{prefix} torch_cuda_version={info.get('torch_cuda_version')} cuda_available={info.get('cuda_available')}")
    print(f"{prefix} device_count={info.get('device_count')} device_name_0={info.get('device_name_0')}")

def choose_backend(requested: str = "auto"):
    requested = (requested or "auto").lower()
    cuda_ok = _HAS_TORCH and torch.cuda.is_available()
    if requested == "cuda":
        if not _HAS_TORCH:
            raise RuntimeError("Requested CUDA but torch is not installed.")
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False. Check drivers and CUDA-enabled PyTorch.")
        return "cuda"
    if requested == "cpu":
        return "cpu"
    # auto
    return "cuda" if cuda_ok else "cpu"


# -------------------------
# Utilities from main.py
# -------------------------

def part(node: str) -> str:
    return ('1' if node[0] == '0' else '0') + node[1:]

def stat(node: str) -> int:
    return int(node[0])

def varname(node: str) -> str:
    return node[1:]

def create_conflict_graph(model: Model):
    cg = mip.ConflictGraph(model)
    g = nx.Graph()
    for x in model.vars:
        g.add_node('0' + x.name)
        g.add_node('1' + x.name)

    for x in model.vars:
        # xi = 0
        z = cg.conflicting_assignments(x == 0)
        for y in z[0]:
            g.add_edge('0' + x.name, '1' + y.name)
        for y in z[1]:
            g.add_edge('0' + x.name, '0' + y.name)
        # xi = 1
        o = cg.conflicting_assignments(x)
        for y in o[0]:
            g.add_edge('1' + x.name, '1' + y.name)
        for y in o[1]:
            g.add_edge('1' + x.name, '0' + y.name)

    return g


# -------------------------
# Hop graph (H) construction
# -------------------------

class HopGraph:
    def __init__(self, G: nx.Graph):
        self.G = G
        # Maps
        self.nodes = list(G.nodes)
        self.n = len(self.nodes)
        self.id_of = {v: i for i, v in enumerate(self.nodes)}
        self.node_of = {i: v for v, i in self.id_of.items()}
        # Precompute part() as indices
        self.part_idx = self._build_part_index()
        # Build H in CSR with witness per hop edge
        self.indptr, self.indices, self.witness = self._build_h_csr()

    def _build_part_index(self):
        part_idx = [0] * self.n
        for v, i in self.id_of.items():
            pv = part(v)
            part_idx[i] = self.id_of[pv]
        return part_idx

    def _build_h_csr(self):
        # For undirected G edge {a,b}, add directed hops:
        #   a -> part(b) (witness=b) if b != part(a)
        #   b -> part(a) (witness=a) if a != part(b)
        n = self.n
        adj = [[] for _ in range(n)]
        wit = [[] for _ in range(n)]

        for a, b in self.G.edges:
            ia = self.id_of[a]
            ib = self.id_of[b]
            ipb = self.part_idx[ib]
            ipa = self.part_idx[ia]

            if ib != self.part_idx[ia]:
                adj[ia].append(ipb)
                wit[ia].append(ib)  # witness is b
            if ia != self.part_idx[ib]:
                adj[ib].append(ipa)
                wit[ib].append(ia)  # witness is a

        # Build CSR arrays
        indptr = [0]
        indices = []
        witness = []
        for i in range(n):
            nbrs = adj[i]
            ws = wit[i]
            indptr.append(indptr[-1] + len(nbrs))
            indices.extend(nbrs)
            witness.extend(ws)

        if _HAS_NUMPY:
            indptr = np.asarray(indptr, dtype=np.int64)
            indices = np.asarray(indices, dtype=np.int64)
            witness = np.asarray(witness, dtype=np.int64)
        return indptr, indices, witness


# -------------------------
# GPU/CPU BFS on H (CSR)
# -------------------------

def _bfs_on_H_torch(indptr, indices, witness, src: int):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    indptr_t = torch.as_tensor(indptr, device=device, dtype=torch.long)
    indices_t = torch.as_tensor(indices, device=device, dtype=torch.long)
    witness_t = torch.as_tensor(witness, device=device, dtype=torch.long)

    n = indptr_t.numel() - 1
    dist = torch.full((n,), -1, device=device, dtype=torch.int32)
    pred = torch.full((n,), -1, device=device, dtype=torch.int64)
    wit_to = torch.full((n,), -1, device=device, dtype=torch.int64)

    frontier = torch.zeros((n,), device=device, dtype=torch.bool)
    dist[src] = 0
    frontier[src] = True
    depth = 0

    while bool(frontier.any().item()):
        rows = torch.nonzero(frontier, as_tuple=False).flatten()
        if rows.numel() == 0:
            break
        starts = indptr_t[rows]
        ends = indptr_t[rows + 1]
        degs = ends - starts
        total = int(degs.sum().item())
        if total == 0:
            break

        offsets = torch.cumsum(degs, dim=0) - degs
        rep_rows = torch.repeat_interleave(rows, degs)
        rep_starts = torch.repeat_interleave(starts, degs)
        rep_offsets = torch.repeat_interleave(offsets, degs)
        edge_idx = torch.arange(total, device=device)
        flat_ptrs = rep_starts + (edge_idx - rep_offsets)
        nbrs = indices_t[flat_ptrs]
        wits = witness_t[flat_ptrs]

        unseen_mask = dist[nbrs] == -1
        if unseen_mask.any():
            targets = nbrs[unseen_mask]
            dist[targets] = depth + 1
            pred[targets] = rep_rows[unseen_mask]
            wit_to[targets] = wits[unseen_mask]
            next_frontier = torch.zeros_like(frontier)
            next_frontier[targets] = True
            frontier = next_frontier
            depth += 1
        else:
            break

    return dist, pred, wit_to


def _bfs_on_H_cpu(indptr, indices, witness, src: int):
    # Lightweight CPU fallback using numpy arrays if present; else lists
    n = (len(indptr) - 1)
    dist = [-1] * n
    pred = [-1] * n
    wit_to = [-1] * n
    q = deque([src])
    dist[src] = 0
    while q:
        v = q.popleft()
        start = indptr[v]
        end = indptr[v + 1]
        for e in range(start, end):
            u = indices[e]
            if dist[u] != -1:
                continue
            dist[u] = dist[v] + 1
            pred[u] = v
            wit_to[u] = witness[e]
            q.append(u)
    return dist, pred, wit_to


def bfs_on_H(H: HopGraph, src_part_idx: int, backend: str = "auto"):
    use_cuda = (backend == "cuda")
    if use_cuda:
        return _bfs_on_H_torch(H.indptr, H.indices, H.witness, src_part_idx)
    else:
        return _bfs_on_H_cpu(H.indptr, H.indices, H.witness, src_part_idx)


# -----------------------------------------
# Map H-BFS results back to G semantics
# -----------------------------------------

def derive_dist_pred_for_source(G: nx.Graph, H: HopGraph, s_name: str, backend: str = "auto"):
    s_idx = H.id_of[s_name]
    ps_idx = H.part_idx[s_idx]
    # Run BFS on H from part(s)
    distH, predH, witH = bfs_on_H(H, ps_idx, backend=backend)

    # Convert tensors to CPU arrays if needed
    if _HAS_TORCH and isinstance(distH, torch.Tensor):
        distH = distH.cpu().numpy()
        predH = predH.cpu().numpy()
        witH = witH.cpu().numpy()

    # Initialize outputs like modified_bfs
    dist = {v: -1 for v in G.nodes}
    pred = {v: -1 for v in G.nodes}

    s = s_name
    ps = H.node_of[ps_idx]
    dist[s] = 0
    dist[ps] = 1
    pred[ps] = s

    # For each reachable part-node p in H with distance k,
    # set dist/pred for its witness u and for p itself.
    n = H.n
    for p_idx in range(n):
        k = distH[p_idx]
        if k == -1:
            continue
        p_name = H.node_of[p_idx]
        u_idx = int(witH[p_idx])
        if u_idx != -1:
            u_name = H.node_of[u_idx]
            even = 2 * int(k)
            odd = even + 1
            if dist[u_name] == -1 or dist[u_name] > even:
                dist[u_name] = even
                # predecessor of u is previous part-node in H
                prev_idx = int(predH[p_idx])
                pred[u_name] = H.node_of[prev_idx] if prev_idx != -1 else ps
            if dist[p_name] == -1 or dist[p_name] > odd:
                dist[p_name] = odd
                pred[p_name] = u_name

    return dist, pred


# -------------------------
# Existing higher-level ops
# -------------------------

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
                if dist[s][u] % 2 == 1:
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


def add_conflict(G: nx.Graph, dist, _stat_fn):
    start_time = time.time()
    AE = set()
    for s in G.nodes:
        for t in G.nodes:
            if dist[s][t] < 0 or dist[s][t] % 2 == 0:
                continue
            edge = (part(s), part(t))
            if (edge not in G.edges) and (edge not in AE) and ((part(t), part(s)) not in AE):
                AE.add(edge)
    end_time = time.time()
    return AE, end_time - start_time


def add_conflicts(graph: nx.Graph, AE):
    for u, v in AE:
        graph.add_edge(u, v)


# -------------------------
# ANS helpers (copied)
# -------------------------

def ANS(graph: nx.graph, varsNames: list):
    indirect, tm = SimpleIndirectElim(graph, varsNames)
    conflicts = ANSConflicts(graph, indirect)
    elimination = ANSElimination(graph, indirect)
    fixing = ANSFixing(graph, indirect)
    return conflicts, elimination, fixing, tm


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
                if not ((l, k) in newConflicts or (k, l) in newConflicts) and not graph.has_edge(l, k):
                    newConflicts.add((l, k))
    for pair in ie:
        for l in graph.neighbors('1x' + pair[0]):
            for k in graph.neighbors('1x' + pair[1]):
                if not ((l, k) in newConflicts or (k, l) in newConflicts) and not graph.has_edge(l, k):
                    newConflicts.add((l, k))
    return newConflicts, time.time() - startTime


def ANSElimination(graph: nx.graph, ie: set):
    startTime = time.time()
    indirectElimination = set()
    directElimination = set()
    for pair1 in ie:
        for pair2 in ie:
            if pair1 == pair2:
                continue
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


# -------------------------
# Analyze using H + BFS
# -------------------------

def analyze_model_gpu(model: Model, tag: str, has_fix: bool = True, has_elim: bool = True, device: str = "auto"):
    graph = create_conflict_graph(model)
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    fixing_quantity = []
    fixing_time = []
    elimination_quantity = []
    elimination_time = []
    conflict_quantity = []
    conflict_time = []

    for _ in range(0, 3):
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

        # Build H once
        H = HopGraph(graph)

        # Compute distances and predecessors via H-BFS
        dist_all = {}
        pred_all = {}
        backend = choose_backend(device)
        device_str = "GPU (CUDA)" if backend == "cuda" else "CPU"
        print(f"[BFS] Device: {device_str}")
        if backend == "cuda":
            torch.cuda.synchronize()
        start_bfs = time.time()
        for node in graph.nodes:
            dist, pred = derive_dist_pred_for_source(graph, H, node, backend=backend)
            dist_all[node] = dist
            pred_all[node] = pred
        if backend == "cuda":
            torch.cuda.synchronize()
        bfsTime = time.time() - start_bfs
        total_bfs_time += bfsTime
        print(f"[BFS] Time on {device_str}: {bfsTime:.6f}s")

        # Improved techniques (as in main.py)
        ae, conflictTime = add_conflict(graph, dist_all, stat)
        add_conflicts(graph, ae)

        if has_fix:
            f0, f1, fixingTime = fix(graph, dist_all)
            total_fixing_time += fixingTime
            total_F0 |= f0
            total_F1 |= f1
        if has_elim:
            de, ie, eliminationTime = eliminate(graph, dist_all, pred_all)
            total_elimination_time += eliminationTime
            total_DE |= de
            total_IE |= ie
        total_AE |= ae

        # ANS methods
        var_names = list(set(var.name for var in model.vars))
        nem_conflicts, nem_eliminations, nem_fixing, nem_preprocess = ANS(graph, var_names)
        total_nem_preprocess_time += nem_preprocess
        nem_AE, nem_C_time = nem_conflicts
        total_nem_conflict_time += nem_C_time
        if has_fix:
            nem_F0, nem_F1, nem_F_time = nem_fixing
            total_nem_fixing_time += nem_F_time
            total_F0_nem |= nem_F0
            total_F1_nem |= nem_F1
        if has_elim:
            nem_IE, nem_DE, nem_E_time = nem_eliminations
            total_nem_elimination_time += nem_E_time
            total_DE_nem |= nem_DE
            total_IE_nem |= nem_IE
        total_AE_nem |= nem_AE

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


def run_analysis(filepath: str, tag: str, max_vars: int, results_dir: str, has_elim: bool, has_fix: bool, device: str = "auto"):
    model_name = os.path.basename(filepath)
    print(f"[GPU] Analyzing {filepath} under tag {tag}")

    model = Model(sense=minimize, solver_name=CBC)
    model.read(filepath)

    if len(model.vars) > max_vars:
        print(f"Skipping {model_name} due to var count ({len(model.vars)}) > {max_vars}")
        return

    (fixing_quantity, fixing_time,
     elimination_quantity, elimination_time,
     conflict_quantity, conflict_time) = analyze_model_gpu(model, tag, has_elim=has_elim, has_fix=has_fix, device=device)

    for i in range(3):
        save_results(os.path.join(results_dir, f"fixing_quantity_g{i}.csv"), model_name, fixing_quantity[i])
        save_results(os.path.join(results_dir, f"fixing_time_g{i}.csv"), model_name, fixing_time[i])
        save_results(os.path.join(results_dir, f"elimination_quantity_g{i}.csv"), model_name, elimination_quantity[i])
        save_results(os.path.join(results_dir, f"elimination_time_g{i}.csv"), model_name, elimination_time[i])
        save_results(os.path.join(results_dir, f"conflict_quantity_g{i}.csv"), model_name, conflict_quantity[i])
        save_results(os.path.join(results_dir, f"conflict_time_g{i}.csv"), model_name, conflict_time[i])


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run GPU-optimized analysis (BFS on hop graph H).")
    parser.add_argument("config_path", type=str, help="Path to the JSON config file.")
    parser.add_argument("--max-vars", type=int, default=15000, help="Maximum number of variables allowed.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Execution device for BFS: auto, cpu, or cuda.")
    args = parser.parse_args()

    config_base = os.path.splitext(os.path.basename(args.config_path))[0]
    today = datetime.now().strftime("%Y-%m-%d")
    results_dir = os.path.join(f"results_{today}_{config_base}_gpu")
    os.makedirs(results_dir, exist_ok=True)

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    print_cuda_diagnostics()

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
            print(f"[GPU] Running index {index} [{tag}] -> {filepath}")
            run_analysis(filepath, tag, args.max_vars, results_dir, has_elim, has_fix, device=args.device)
        except MemoryError as e:
            print(f"MemoryError on {filepath}: {e}")
            raise e
        except Exception as e:
            print(f"Error on {filepath}: {e}")
            raise e


if __name__ == "__main__":
    main()

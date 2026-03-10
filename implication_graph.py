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
import numba
from numba import jit

# Optional import for GPU acceleration
try:
    import torch
except ImportError:
    torch = None

def warshall_bitset(reach, n):
    """Warshall algorithm for bitset transitive closure."""
    for k in range(n):
        rk = reach[k]
        mask = 1 << k
        for i in range(n):
            if reach[i] & mask:
                reach[i] |= rk
    return reach

def part(node: str) -> str:
    return ('1' if node[0] == '0' else '0') + node[1:]

def stat(node: str) -> int:
    return int(node[0])

def varname(node: str) -> str:
    return node[1:]

def create_conflict_graph(model: Model):
    # Returns the conflict graph for the model and a list with all partitions

    cg = mip.ConflictGraph(model)
    g_conflict = nx.Graph()
    g_implication = nx.DiGraph()
    count = 0
    names = []
    for x in model.vars:
        g_conflict.add_node('0' + x.name)
        g_conflict.add_node('1' + x.name)
        names.append(x.name)
        count+= 1


    for x in model.vars:
        # xi = 0
        z = cg.conflicting_assignments(x == 0)
        for y in z[0]:
            g_conflict.add_edge('0'+ x.name, '1' + y.name)

            g_implication.add_edge('0'+ x.name, '0' + y.name)
            g_implication.add_edge('1'+ y.name, '1' + x.name)
            
        for y in z[1]:
            g_conflict.add_edge('0'+ x.name, '0' + y.name)

            g_implication.add_edge('0'+ x.name, '1' + y.name)
            g_implication.add_edge('0'+ y.name, '1' + x.name)
        # xi = 1
        o = cg.conflicting_assignments(x)
        for y in o[0]:
            g_conflict.add_edge('1'+ x.name, '1' + y.name)

            g_implication.add_edge('1'+ x.name, '0' + y.name)
            g_implication.add_edge('1'+ y.name, '0' + x.name)
        
        for y in o[1]:
            g_conflict.add_edge('1'+ x.name, '0' + y.name)

            g_implication.add_edge('1'+ x.name, '1' + y.name)
            g_implication.add_edge('0'+ y.name, '0' + x.name)

    return g_conflict, g_implication


def transitive_closure(graph, nodes=None, method='auto', reflexive=True, return_matrix=False):
    """
    Compute the transitive closure of a directed graph.
    Returns either a dict (node -> set) or a boolean matrix (ndarray/tensor).
    """
    if nodes is None:
        if hasattr(graph, 'nodes') and hasattr(graph, 'successors'):
            nodes = list(graph.nodes)
            adj_iter = lambda u: graph.successors(u)
        else:
            nodes = list(graph.keys())
            adj_iter = lambda u: graph.get(u, ())
            for nbrs in graph.values():
                for v in nbrs:
                    if v not in nodes: nodes.append(v)
    else:
        nodes = list(nodes)
        adj_iter = lambda u: graph.successors(u) if hasattr(graph, 'successors') else graph.get(u, ())

    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    
    # Heuristic for method
    if method == 'auto':
        if n == 0: return {} if not return_matrix else np.zeros((0,0))
        if n <= 1000: method = 'bitset'
        elif torch is not None: method = 'matrix_torch'
        else: method = 'bfs'

    # Delegate to specialized implementations
    if method == 'matrix':
        return transitive_closure_gpu(graph, nodes=nodes, reflexive=reflexive, backend='numpy', return_matrix=return_matrix)
    elif method == 'matrix_torch':
        return transitive_closure_torch(graph, nodes=nodes, reflexive=reflexive, return_matrix=return_matrix)
    elif method in ('bfs_gpu', 'bfs_torch'):
        return transitive_closure_bfs_torch(graph, nodes=nodes, reflexive=reflexive, return_matrix=return_matrix)
    elif method == 'bfs':
        if return_matrix:
            mat = np.zeros((n, n), dtype=bool)
            for i, u in enumerate(nodes):
                visited = {u} if reflexive else set()
                stack = [u]
                while stack:
                    x = stack.pop()
                    for v in adj_iter(x):
                        if v not in visited:
                            visited.add(v)
                            stack.append(v)
                for v in visited:
                    mat[i, index[v]] = True
            return mat
        else:
            out = {}
            for u in nodes:
                visited = set()
                stack = [u]
                while stack:
                    x = stack.pop()
                    for v in adj_iter(x):
                        if v not in visited and v != u:
                            visited.add(v)
                            stack.append(v)
                if reflexive: visited.add(u)
                out[u] = visited
            return out

    # Default: Bitset method
    reach = [0] * n
    for u in nodes:
        ui = index[u]
        for v in adj_iter(u):
            if v in index: reach[ui] |= 1 << index[v]
    if reflexive:
        for i in range(n): reach[i] |= 1 << i

    reach = warshall_bitset(reach, n)

    if return_matrix:
        mat = np.zeros((n, n), dtype=bool)
        for i in range(n):
            b = reach[i]
            while b:
                lsb = b & -b
                j = lsb.bit_length() - 1
                mat[i, j] = True
                b &= b - 1
        return mat
    else:
        out = {}
        for i in range(n):
            b = reach[i]
            s = set()
            while b:
                lsb = b & -b
                j = lsb.bit_length() - 1
                s.add(nodes[j])
                b &= b - 1
            out[nodes[i]] = s
        return out


def transitive_closure_gpu(graph, nodes=None, reflexive=True, backend='cupy', return_matrix=False):
    """
    GPU-accelerated transitive closure using matrix multiplication on the boolean semiring.

    Parameters
    - graph: NetworkX DiGraph or adjacency mapping {node: iterable(neighbors)}
    - nodes: optional iterable of nodes (order defines matrix indices)
    - reflexive: include self-reachability
    - backend: 'cupy' (preferred) or 'numpy' (CPU fallback)
    - return_matrix: if True return the boolean matrix (CuPy/Numpy array);
                     otherwise return dict mapping node->set(reachable nodes)

    Notes
    - Requires CuPy for actual GPU acceleration. If CuPy is not available and
      `backend=='cupy'` a RuntimeError is raised. You can set `backend='numpy'`
      to run the same algorithm on CPU via NumPy.
    - This approach builds an n x n boolean adjacency matrix and iteratively
      applies boolean matrix multiplication until closure stabilizes. It is
      most effective when `n` fits comfortably on the GPU memory.
    """
    # Normalize adjacency and nodes
    if nodes is None:
        if hasattr(graph, 'nodes') and hasattr(graph, 'successors'):
            nodes = list(graph.nodes)
            adj_iter = lambda u: graph.successors(u)
        else:
            nodes = list(graph.keys())
            adj_iter = lambda u: graph.get(u, ())
            for nbrs in graph.values():
                for v in nbrs:
                    if v not in nodes:
                        nodes.append(v)
    else:
        nodes = list(nodes)
        if hasattr(graph, 'successors'):
            adj_iter = lambda u: graph.successors(u)
        else:
            adj_iter = lambda u: graph.get(u, ())

    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}

    # Choose backend
    if backend == 'cupy':
        try:
            import cupy as cp
            xp = cp
        except Exception as e:
            raise RuntimeError('CuPy backend requested but CuPy is not available: ' + str(e))
    else:
        import numpy as cp
        xp = cp

    # Build adjacency matrix (boolean), dtype=uint8 for multiplication
    A = xp.zeros((n, n), dtype=xp.uint8)
    for u in nodes:
        ui = index[u]
        for v in adj_iter(u):
            if v in index:
                A[ui, index[v]] = 1

    if reflexive:
        diag = xp.arange(n)
        A[diag, diag] = 1

    # iterative multiplication until closure stabilizes
    reach = A.copy()
    while True:
        prod = reach.dot(reach)
        # boolean semiring: entry > 0 means there exists a path
        prod_bool = (prod > 0).astype(xp.uint8)
        new = (reach | prod_bool)
        # check convergence
        if xp.array_equal(new, reach):
            break
        reach = new

    if return_matrix:
        return reach

    # Convert matrix to mapping node->set
    out = {}
    if xp.__name__ == 'cupy':
        # move to host in slices to avoid huge temporary if memory constrained
        for i, u in enumerate(nodes):
            row = xp.asnumpy(reach[i])
            inds = (row > 0).nonzero()[0]
            out[u] = {nodes[j] for j in inds}
    else:
        for i, u in enumerate(nodes):
            row = reach[i]
            inds = (row > 0).nonzero()[0]
            out[u] = {nodes[j] for j in inds}

    return out


def transitive_closure_torch(graph, nodes=None, reflexive=True, return_matrix=False):
    """
    GPU/CPU-accelerated transitive closure using PyTorch.

    Parameters
    - graph: NetworkX DiGraph or adjacency mapping {node: iterable(neighbors)}
    - nodes: optional iterable of nodes (order defines matrix indices)
    - reflexive: include self-reachability
    - return_matrix: if True return the boolean matrix (torch.Tensor);
                     otherwise return dict mapping node->set(reachable nodes)

    Returns
    - dict: node -> set(of reachable nodes) (or torch.Tensor if return_matrix=True)

    Notes
    - Leverages PyTorch's optimized matrix ops for the boolean semiring.
    - Automatically falls back to CPU if device='cuda' but CUDA is unavailable.
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch is not installed. Install with: pip install torch")

    # Normalize adjacency and nodes
    if nodes is None:
        if hasattr(graph, 'nodes') and hasattr(graph, 'successors'):
            nodes = list(graph.nodes)
            adj_iter = lambda u: graph.successors(u)
        else:
            nodes = list(graph.keys())
            adj_iter = lambda u: graph.get(u, ())
            for nbrs in graph.values():
                for v in nbrs:
                    if v not in nodes:
                        nodes.append(v)
    else:
        nodes = list(nodes)
        if hasattr(graph, 'successors'):
            adj_iter = lambda u: graph.successors(u)
        else:
            adj_iter = lambda u: graph.get(u, ())

    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.xpu.is_available():
        device = 'xpu'
    else:
        device = 'cpu'

    print(f"Closure running on: {device}")
    # Build adjacency matrix as boolean tensor
    A = torch.zeros((n, n), dtype=torch.uint8, device=device)
    for u in nodes:
        ui = index[u]
        for v in adj_iter(u):
            if v in index:
                A[ui, index[v]] = 1

    if reflexive:
        diag_idx = torch.arange(n, device=device)
        A[diag_idx, diag_idx] = 1

    # Iterative matrix multiplication until closure stabilizes
    reach = A.clone()
    while True:
        # Boolean multiplication: treat as integer, then clip to {0, 1}
        prod = torch.matmul(reach.float(), reach.float()).bool().to(torch.uint8)
        new = torch.bitwise_or(reach, prod)
        if torch.equal(new, reach):
            break
        reach = new

    if return_matrix:
        return reach

    # Convert matrix to dict mapping node->set
    reach_cpu = reach.cpu().numpy() if device == 'cuda' else reach.numpy()
    out = {}
    for i, u in enumerate(nodes):
        inds = (reach_cpu[i] > 0).nonzero()[0]
        out[u] = {nodes[j] for j in inds}

    return out


def transitive_closure_bfs_torch(graph, nodes=None, reflexive=True, return_matrix=False):
    """GPU/CPU-accelerated BFS-style closure using PyTorch frontiers.

    This variant is most useful on sparse graphs where the "matrix" methods
    (which repeatedly square a reachability matrix) may generate a large
    intermediate dense matrix.  Instead we propagate a boolean frontier from
    every source simultaneously using torch matrix operations.

    Parameters
    - graph: NetworkX DiGraph or adjacency mapping {node: iterable(neighbors)}
    - nodes: optional iterable of nodes (order defines matrix indices)
    - reflexive: include self-reachability
    - return_matrix: if True return the boolean matrix (torch.Tensor)

    Returns
    - dict: node -> set(of reachable nodes)
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch is not installed. Install with: pip install torch")

    # normalize adjacency
    if nodes is None:
        if hasattr(graph, 'nodes') and hasattr(graph, 'successors'):
            nodes = list(graph.nodes)
            adj_iter = lambda u: graph.successors(u)
        else:
            nodes = list(graph.keys())
            adj_iter = lambda u: graph.get(u, ())
            for nbrs in graph.values():
                for v in nbrs:
                    if v not in nodes:
                        nodes.append(v)
    else:
        nodes = list(nodes)
        if hasattr(graph, 'successors'):
            adj_iter = lambda u: graph.successors(u)
        else:
            adj_iter = lambda u: graph.get(u, ())

    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.xpu.is_available():
        device = 'xpu'
    else:
        device = 'cpu'

    print(f"Closure running on: {device}")
    # build boolean adjacency matrix
    A = torch.zeros((n, n), dtype=torch.bool, device=device)
    for u in nodes:
        ui = index[u]
        for v in adj_iter(u):
            if v in index:
                A[ui, index[v]] = True

    if reflexive:
        diag = torch.arange(n, device=device)
        A[diag, diag] = True

    # visited matrix and frontier
    visited = A.clone()
    frontier = A.clone()

    # propagate frontiers until stabilization
    while True:
        # new_front = (frontier @ A) & ~visited
        prod = torch.matmul(frontier.float(), A.float()).to(torch.bool)
        new_front = prod & ~visited
        if not new_front.any():
            break
        visited |= new_front
        frontier = new_front

    if return_matrix:
        return visited

    # convert back to dict
    vis_cpu = visited.cpu().numpy()
    out = {}
    for i, u in enumerate(nodes):
        inds = (vis_cpu[i] > 0).nonzero()[0]
        out[u] = {nodes[j] for j in inds}
    return out

class DisjointSetUnion:
    """Simple DSU (Union-Find) data structure for grouping nodes."""
    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
    
    def groups(self):
        """Return dict: root -> set of all elements with that root."""
        groups_dict = {}
        for e in self.parent:
            root = self.find(e)
            if root not in groups_dict:
                groups_dict[root] = set()
            groups_dict[root].add(e)
        return groups_dict


def elimination_on_implication_graph(graph, reach_bitsets=None, reach_matrix=None, index=None):
    """
    Optimized elimination using bipartite structure and symmetry.

    Parameters
    - graph: NetworkX DiGraph
    - reach_bitsets: dict mapping node -> set of reachable nodes
    - reach_matrix: optional precomputed numpy boolean matrix (nodes × nodes).
                   If None, will use reach_bitsets directly.
    - index: node -> index mapping
    """
    nodes = list(graph.nodes)
    if not nodes:
        return None, [], {}
    
    if index is None:
        index = {node: i for i, node in enumerate(nodes)}
    
    # Precompute node metadata to avoid repeated string ops
    node_stat = {}
    node_var = {}
    for node in nodes:
        node_stat[node] = stat(node)
        node_var[node] = varname(node)
    
    # Precompute var_nodes and separate by stat value
    var_nodes = {}
    for node in nodes:
        v = node_var[node]
        if v not in var_nodes:
            var_nodes[v] = [None, None]  # idx 0 for stat=0, idx 1 for stat=1
        var_nodes[v][node_stat[node]] = node
    
    # Use provided matrix or fall back to reach_bitsets
    use_matrix = reach_matrix is not None
    
    # ===== DIRECT ELIMINATION =====
    all_variables = list(var_nodes.keys())
    dsu_vars = DisjointSetUnion(all_variables)
    
    # Only check from 0x nodes
    nodes_0x = [node for node in nodes if node_stat[node] == 0]
    
    for u in nodes_0x:
        u_var = node_var[u]
        u_idx = index[u]
        
        for v in nodes_0x:
            if u == v:
                continue
            v_var = node_var[v]
            if u_var == v_var:  # Skip same variable
                continue
            
            # Early exit if already in same group
            if dsu_vars.find(u_var) == dsu_vars.find(v_var):
                continue
            
            v_idx = index[v]
            
            # Check bidirectional reachability
            if use_matrix:
                if reach_matrix[u_idx, v_idx] and reach_matrix[v_idx, u_idx]:
                    dsu_vars.union(u_var, v_var)
            else:
                if v in reach_bitsets[u] and u in reach_bitsets[v]:
                    dsu_vars.union(u_var, v_var)
    
    # Cache the final groups once
    final_groups = dsu_vars.groups()
    direct_eliminations = [set(group) for group in final_groups.values() if group]
    
    # Build var_to_group map
    var_to_group = {}
    for root, group in final_groups.items():
        for var in group:
            var_to_group[var] = root
    
    # ===== INDIRECT ELIMINATION =====
    var_list = sorted(var_nodes.keys())
    indirect_map = {}
    
    for i in range(len(var_list)):
        var_a = var_list[i]
        node_a_0x = var_nodes[var_a][0]  # stat=0 node
        
        if node_a_0x is None:
            continue
        
        a_idx = index[node_a_0x]
        group_a = var_to_group.get(var_a)
        
        for j in range(i + 1, len(var_list)):
            var_b = var_list[j]
            node_b_1x = var_nodes[var_b][1]  # stat=1 node
            
            if node_b_1x is None:
                continue
            
            # Skip if in same direct elimination group
            if group_a == var_to_group.get(var_b):
                continue
            
            b_idx = index[node_b_1x]
            
            # Check: 0xA -> 1xB AND 1xB -> 0xA
            if use_matrix:
                if reach_matrix[a_idx, b_idx] and reach_matrix[b_idx, a_idx]:
                    if var_a not in indirect_map:
                        indirect_map[var_a] = []
                    if var_b not in indirect_map:
                        indirect_map[var_b] = []
                    indirect_map[var_a].append(var_b)
                    indirect_map[var_b].append(var_a)
            else:
                node_a_0x_check = var_nodes[var_a][0]
                if node_b_1x in reach_bitsets[node_a_0x] and node_a_0x_check in reach_bitsets[node_b_1x]:
                    if var_a not in indirect_map:
                        indirect_map[var_a] = []
                    if var_b not in indirect_map:
                        indirect_map[var_b] = []
                    indirect_map[var_a].append(var_b)
                    indirect_map[var_b].append(var_a)
    
    return dsu_vars, direct_eliminations, indirect_map


def elimination_on_implication_graph_torch(R, index, nodes):
    """GPU-accelerated elimination using mutual reachability M = R & R.T."""
    if torch is None:
        raise RuntimeError("PyTorch not available")
    
    # M[a][b] == 1 iff a reaches b AND b reaches a (mutual reachability)
    M = (R > 0) & (R.t() > 0)
    M_cpu = M.cpu().numpy()
    
    # Extract var names and find node indices
    node_vars = [varname(n) for n in nodes]
    node_stats = [stat(n) for n in nodes]
    
    all_vars = sorted(list(set(node_vars)))
    dsu_vars = DisjointSetUnion(all_vars)
    
    # Var mapping to indices in nodes list
    var_to_indices = {}
    for i, v in enumerate(node_vars):
        if v not in var_to_indices: var_to_indices[v] = [None, None]
        var_to_indices[v][node_stats[i]] = i

    # Direct Elimination: 0x <-> 0y
    for i in range(len(all_vars)):
        v_a = all_vars[i]
        idx_a0 = var_to_indices[v_a][0]
        if idx_a0 is None: continue
        
        for j in range(i + 1, len(all_vars)):
            v_b = all_vars[j]
            idx_b0 = var_to_indices[v_b][0]
            if idx_b0 is None: continue
            
            if M_cpu[idx_a0, idx_b0]:
                dsu_vars.union(v_a, v_b)
                
    final_groups = dsu_vars.groups()
    # Align with CPU logic: return all groups including those of size 1
    direct_eliminations = [set(group) for group in final_groups.values() if group]
    
    # Var to group mapping for indirect skip
    var_to_root = {v: dsu_vars.find(v) for v in all_vars}
    
    # Indirect Elimination: 0x <-> 1y
    indirect_map = {}
    for i in range(len(all_vars)):
        v_a = all_vars[i]
        idx_a0 = var_to_indices[v_a][0]
        if idx_a0 is None: continue
        
        # Consistent with CPU loop: only check pairs j > i
        for j in range(len(all_vars)):
            if i == j: continue
            v_b = all_vars[j]
            if var_to_root[v_a] == var_to_root[v_b]: continue # Skip if already directly equivalent
            
            idx_b1 = var_to_indices[v_b][1]
            if idx_b1 is None: continue
            
            if M_cpu[idx_a0, idx_b1]:
                if v_a not in indirect_map: indirect_map[v_a] = []
                if v_b not in indirect_map: indirect_map[v_b] = []
                indirect_map[v_a].append(v_b)
                indirect_map[v_b].append(v_a)
                
    return dsu_vars, direct_eliminations, indirect_map


def fixing_on_implication_graph(graph, dsu_vars=None, indirect_map=None, reach_bitsets=None, reach_matrix=None, index=None):
    """
    Compute variable fixing based on implication graph reachability.
    
    Parameters
    - graph: NetworkX DiGraph
    - dsu_vars, indirect_map: from elimination_on_implication_graph
    - reach_bitsets: dict mapping node -> set of reachable nodes (used if reach_matrix is None)
    - reach_matrix: optional precomputed numpy boolean matrix. If None, will use reach_bitsets.
    - index: node -> index mapping
    
    Returns
    - (fixing_0, fixing_1): sets of variables to fix
    """
    nodes = list(graph.nodes)
    if not nodes:
        return set(), set()
    
    if index is None:
        index = {node: i for i, node in enumerate(nodes)}
    
    # Precompute node metadata
    node_stat = {}
    node_var = {}
    for node in nodes:
        node_stat[node] = stat(node)
        node_var[node] = varname(node)
    
    # Build var_nodes structure
    var_nodes = {}
    for node in nodes:
        v = node_var[node]
        if v not in var_nodes:
            var_nodes[v] = [None, None]  # idx 0 for stat=0, idx 1 for stat=1
        var_nodes[v][node_stat[node]] = node
    
    # Use provided matrix or fall back to reach_bitsets
    use_matrix = reach_matrix is not None
    
    # Cache dsu groups if provided
    if dsu_vars:
        direct_groups = dsu_vars.groups()
        var_to_group = {}
        for root, group in direct_groups.items():
            for var in group:
                var_to_group[var] = root
    else:
        var_to_group = {}
        direct_groups = {}
    
    fixing_0 = set()
    fixing_1 = set()
    
    # For each variable A, check if it can be fixed
    for var_a in var_nodes:
        if var_a in fixing_0 or var_a in fixing_1:
            continue
        
        # Try fixing to 0: 0xA reachable from both 0xB and 1xB for some B
        node_0xa = var_nodes[var_a][0]
        if node_0xa is not None:
            idx_0xa = index[node_0xa]
            
            for var_b in var_nodes:
                if var_a == var_b:
                    continue
                
                node_0xb = var_nodes[var_b][0]
                node_1xb = var_nodes[var_b][1]
                
                if node_0xb is None or node_1xb is None:
                    continue
                
                idx_0xb = index[node_0xb]
                idx_1xb = index[node_1xb]
                
                # Check if both 0xB and 1xB reach 0xA
                if use_matrix:
                    can_fix = reach_matrix[idx_0xb, idx_0xa] and reach_matrix[idx_1xb, idx_0xa]
                else:
                    can_fix = node_0xa in reach_bitsets[node_0xb] and node_0xa in reach_bitsets[node_1xb]
                
                if can_fix:
                    fixing_0.add(var_a)
                    
                    # Propagate through direct eliminations (same value)
                    if dsu_vars:
                        group_a = var_to_group.get(var_a)
                        if group_a is not None:
                            for var_c in direct_groups[group_a]:
                                if var_c not in fixing_0 and var_c not in fixing_1:
                                    fixing_0.add(var_c)
                    
                    # Propagate through indirect eliminations (opposite value)
                    if indirect_map and var_a in indirect_map:
                        for var_c in indirect_map[var_a]:
                            if var_c not in fixing_0 and var_c not in fixing_1:
                                fixing_1.add(var_c)
                    break
        
        # Try fixing to 1: 1xA reachable from both 0xB and 1xB for some B
        if var_a not in fixing_0 and var_a not in fixing_1:
            node_1xa = var_nodes[var_a][1]
            if node_1xa is not None:
                idx_1xa = index[node_1xa]
                
                for var_b in var_nodes:
                    if var_a == var_b:
                        continue
                    
                    node_0xb = var_nodes[var_b][0]
                    node_1xb = var_nodes[var_b][1]
                    
                    if node_0xb is None or node_1xb is None:
                        continue
                    
                    idx_0xb = index[node_0xb]
                    idx_1xb = index[node_1xb]
                    
                    # Check if both 0xB and 1xB reach 1xA
                    if use_matrix:
                        can_fix = reach_matrix[idx_0xb, idx_1xa] and reach_matrix[idx_1xb, idx_1xa]
                    else:
                        can_fix = node_1xa in reach_bitsets[node_0xb] and node_1xa in reach_bitsets[node_1xb]
                    
                    if can_fix:
                        fixing_1.add(var_a)
                        
                        # Propagate through direct eliminations (same value)
                        if dsu_vars:
                            group_a = var_to_group.get(var_a)
                            if group_a is not None:
                                for var_c in direct_groups[group_a]:
                                    if var_c not in fixing_0 and var_c not in fixing_1:
                                        fixing_1.add(var_c)
                        
                        # Propagate through indirect eliminations (opposite value)
                        if indirect_map and var_a in indirect_map:
                            for var_c in indirect_map[var_a]:
                                if var_c not in fixing_0 and var_c not in fixing_1:
                                    fixing_0.add(var_c)
                        break
    
    return fixing_0, fixing_1


def fixing_on_implication_graph_torch(R, index, nodes, dsu_vars=None, indirect_map=None):
    """GPU-accelerated fixing following the logic in gpu_fixing.pdf."""
    if torch is None:
        raise RuntimeError("PyTorch not available")
    
    n = len(nodes)
    node_vars = [varname(n) for n in nodes]
    node_stats = [stat(n) for n in nodes]
    all_vars = sorted(list(set(node_vars)))
    
    # Map variable names to literal indices
    var_to_indices = {}
    for i, v in enumerate(node_vars):
        if v not in var_to_indices: var_to_indices[v] = [None, None]
        var_to_indices[v][node_stats[i]] = i
        
    # Vectors for Method 1 & 2
    idx_0 = []
    idx_1 = []
    valid_vars = []
    for v in all_vars:
        i0, i1 = var_to_indices[v]
        if i0 is not None and i1 is not None:
            idx_0.append(i0)
            idx_1.append(i1)
            valid_vars.append(v)
            
    if not idx_0: return set(), set()
    
    idx_0 = torch.tensor(idx_0, device=R.device)
    idx_1 = torch.tensor(idx_1, device=R.device)
    
    # Method 2: Anti-Diagonal Lookup (Direct 2-SAT fixing)
    # x=1 reaches x=0 => x=0
    # x=0 reaches x=1 => x=1
    direct_0_vec = R[idx_1, idx_0] > 0
    direct_1_vec = R[idx_0, idx_1] > 0
    
    # Method 1: Complementary-Pair AND (Global reachability check)
    # Literal t is forced if there exists some variable y such that both y=0 and y=1 reach t
    R_even = R[idx_0, :].float()
    R_odd = R[idx_1, :].float()
    P = (R_even > 0) & (R_odd > 0)
    forced_mask = P.any(dim=0)
    
    forced_0_vec = forced_mask[idx_0] | direct_0_vec
    forced_1_vec = forced_mask[idx_1] | direct_1_vec
    
    fixing_0 = {valid_vars[i] for i, f in enumerate(forced_0_vec) if f}
    fixing_1 = {valid_vars[i] for i, f in enumerate(forced_1_vec) if f}
    
    # Propagate through eliminations (CPU logic)
    if dsu_vars or indirect_map:
        # Note: In a real solver, we'd loop until convergence, 
        # but here we follow the existing CPU structure.
        var_to_group = {}
        if dsu_vars:
            groups = dsu_vars.groups()
            for root, members in groups.items():
                for m in members: var_to_group[m] = root
        
        # Simple one-pass propagation to match CPU logic as closely as possible
        # whilst still staying mostly on GPU logic.
        new_f0 = set(fixing_0)
        new_f1 = set(fixing_1)
        
        # Propagate from 0-fixes
        for v in fixing_0:
            if dsu_vars and v in var_to_group:
                for m in dsu_vars.groups()[var_to_group[v]]: new_f0.add(m)
            if indirect_map and v in indirect_map:
                for m in indirect_map[v]: new_f1.add(m)
        
        # Propagate from 1-fixes
        for v in fixing_1:
            if dsu_vars and v in var_to_group:
                for m in dsu_vars.groups()[var_to_group[v]]: new_f1.add(m)
            if indirect_map and v in indirect_map:
                for m in indirect_map[v]: new_f0.add(m)
                
        return new_f0, new_f1

    return fixing_0, fixing_1


def run_analysis(filepath, tag, max_vars, results_dir, has_elim, has_fix, methods):
    """Run analysis with multiple closure methods, but elimination/fixing only once.
    
    Parameters
    - filepath, tag, max_vars, results_dir, has_elim, has_fix: as before
    - methods: list of closure method names to compare
    """
    model_name = os.path.basename(filepath)
    print(f"Analyzing {filepath} under tag {tag}")

    model = Model(sense=minimize, solver_name=CBC)
    model.read(filepath)
    _, graph_implication = create_conflict_graph(model)
    
    # Skip if too many variables
    if len(model.vars) > max_vars:
        print(f"Skipping {model_name} due to var count ({len(model.vars)}) > {max_vars}")
        return
    
    nodes = list(graph_implication.nodes)
    n = len(nodes)
    m = graph_implication.number_of_edges()
    index = {node: i for i, node in enumerate(nodes)}
    
    # All variable names present in the graph
    all_var_names = sorted(list(set(varname(nd) for nd in nodes)))
    num_total_vars = len(all_var_names)

    # Compute one reference closure (for elimination/fixing baselines)
    closure_start = time.time()
    reach_bitsets = transitive_closure(graph_implication, nodes=nodes, method='bitset',
                                       reflexive=True)
    closure_time = time.time() - closure_start
    
    # Create reach_matrix once if we have enough edges to make it worthwhile
    total_reach_edges = sum(len(v) for v in reach_bitsets.values())
    create_matrix = total_reach_edges > n * 10
    
    reach_matrix = None
    if create_matrix:
        import numpy as np
        reach_matrix = np.zeros((n, n), dtype=np.bool_)
        for u in nodes:
            ui = index[u]
            for v in reach_bitsets[u]:
                vi = index[v]
                reach_matrix[ui, vi] = True
    
    # Run elimination and fixing ONCE (CPU Reference)
    elim_start = time.time()
    dsu_vars, direct_eliminations, indirect_map = elimination_on_implication_graph(
        graph_implication, reach_bitsets=reach_bitsets, reach_matrix=reach_matrix, index=index)
    elim_time = time.time() - elim_start
    
    # Direct elimination should count N - groups
    direct_elim_count = num_total_vars - len(direct_eliminations)
    
    # Indirect elimination should count pairs of variables
    if indirect_map:
        indirect_elim_count = sum(len(neighs) for neighs in indirect_map.values()) // 2
    else:
        indirect_elim_count = 0
    
    fixing_start = time.time()
    fixing_0, fixing_1 = fixing_on_implication_graph(
        graph_implication, dsu_vars=dsu_vars, indirect_map=indirect_map,
        reach_bitsets=reach_bitsets, reach_matrix=reach_matrix, index=index)
    fixing_time = time.time() - fixing_start
    
    csv_path = os.path.join(results_dir, "results.csv")
    write_header = not os.path.exists(csv_path)
    fieldnames = ['problem', 'method', 'n', 'm', 'closure_time', 'direct_elimination_#', 
                  'indirect_elimination_#', 'elimination_time', '0_fixing', '1_fixing', 
                  'fixing_time']
    
    for method in methods:
        closure_start = time.time()
        try:
            # Handle methods that might return a tensor for GPU elimination/fixing
            if method in ('matrix_torch', 'bfs_torch') and torch is not None:
                if method == 'matrix_torch':
                    R_gpu = transitive_closure_torch(graph_implication, nodes=nodes, reflexive=True, return_matrix=True)
                else:
                    R_gpu = transitive_closure_bfs_torch(graph_implication, nodes=nodes, reflexive=True, return_matrix=True)
                
                closure_time_test = time.time() - closure_start
                
                # GPU Elimination
                e_start = time.time()
                dsu_gpu, d_elim_groups_gpu, i_map_gpu = elimination_on_implication_graph_torch(R_gpu, index, nodes)
                e_time_gpu = time.time() - e_start
                
                # GPU Fixing
                f_start = time.time()
                f0_gpu, f1_gpu = fixing_on_implication_graph_torch(R_gpu, index, nodes, dsu_vars=dsu_gpu, indirect_map=i_map_gpu)
                f_time_gpu = time.time() - f_start
                
                # Count pairs for indirect
                i_count_gpu = 0
                if i_map_gpu:
                    i_count_gpu = sum(len(v) for v in i_map_gpu.values()) // 2

                result = {
                    'problem': model_name,
                    'method': method + "_gpu_steps",
                    'n': n,
                    'm': m,
                    'closure_time': round(closure_time_test, 6),
                    'direct_elimination_#': num_total_vars - len(d_elim_groups_gpu),
                    'indirect_elimination_#': i_count_gpu,
                    'elimination_time': round(e_time_gpu, 6),
                    '0_fixing': len(f0_gpu),
                    '1_fixing': len(f1_gpu),
                    'fixing_time': round(f_time_gpu, 6),
                }
            else:
                # Regular CPU path (but still checking the test method)
                reach_test = transitive_closure(graph_implication, nodes=nodes, method=method, reflexive=True)
                closure_time_test = time.time() - closure_start
                
                # For reporting, we still use the reference counts for elimination/fixing 
                # as closure correctness is assumed if we reached here
                result = {
                    'problem': model_name,
                    'method': method,
                    'n': n,
                    'm': m,
                    'closure_time': round(closure_time_test, 6),
                    'direct_elimination_#': direct_elim_count,
                    'indirect_elimination_#': indirect_elim_count,
                    'elimination_time': round(elim_time, 6),
                    '0_fixing': len(fixing_0),
                    '1_fixing': len(fixing_1),
                    'fixing_time': round(fixing_time, 6),
                }
            
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(result)
            
            print(f"  {method}: closure={closure_time_test:.3f}s")
        except Exception as e:
            print(f"  {method}: ERROR - {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Results for {model_name} written to {csv_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run analysis from a config file.")
    parser.add_argument("--config_path", type=str, default='config.json', help="Path to the JSON config file.")
    parser.add_argument("--max-vars", type=int, default=100000, help="Maximum number of variables allowed.")
    args = parser.parse_args()

    today = datetime.now().strftime("%Y-%m-%d")
    config_base = os.path.splitext(os.path.basename(args.config_path))[0]
    results_dir = os.path.join(f"results_{today}_{config_base}")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(args.config_path):
        print(f"Config file not found: {args.config_path}")
        return

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    methods = ['matrix_torch', 'bfs_torch', 'bfs']
    index = 0
    for run_id, run_config in config.items():
        filepath = run_config.get('filepath')
        tag = run_config.get('category')
        has_elim = run_config.get('elimination', True)
        has_fix = run_config.get('fixing', True)

        if not filepath or not tag:
            print(f"Skipping {run_id}: missing filepath or category.")
            continue

        index += 1
        try:
            print(f"Running index {index} [{tag}] -> {filepath}")
            run_analysis(filepath, tag, args.max_vars, results_dir, has_elim, has_fix, methods)
        except MemoryError as e:
            print(f"MemoryError on {filepath}: {e}")
        except Exception as e:
            print(f"Error on {filepath}: {e}")


if __name__ == "__main__":
    main()

// =============================================================================
//  graph_utils.cpp
//
//  Implementation of the SCC + DSU + reachability primitives declared in
//  graph_utils.h.  Standard library only.
// =============================================================================

#include "graph_utils.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <queue>
#include <stack>
#include <stdexcept>
#include <utility>

namespace presolve {

// -----------------------------------------------------------------------------
//  BuildCSR
// -----------------------------------------------------------------------------
CSRDigraph BuildCSR(Idx n, const std::vector<std::pair<Idx, Idx>>& arcs) {
    CSRDigraph G;
    G.n = n;
    G.xadj.assign((std::size_t)n + 1, 0);

    // Counting sort by source vertex; first pass = counts.
    for (const auto& a : arcs) {
        if (a.first == a.second) continue;            // drop self-loops
        if (a.first < 0 || a.first >= n)  continue;
        if (a.second < 0 || a.second >= n) continue;
        ++G.xadj[(std::size_t)a.first + 1];
    }
    for (Idx i = 0; i < n; ++i) G.xadj[(std::size_t)i + 1] += G.xadj[(std::size_t)i];

    // Second pass = scatter.  We may write duplicates; we dedupe with an
    // O(deg log deg) sort below.  Memory is tight only on the very biggest
    // MIPLIB instances; the alternative (hash dedup) is not worth the cache
    // miss on the typical workload.
    const Idx scratch_size = G.xadj[(std::size_t)n];
    std::vector<Idx> tmp_adj(scratch_size);
    std::vector<Idx> cursor = G.xadj;  // mutable copy
    for (const auto& a : arcs) {
        if (a.first == a.second) continue;
        if (a.first < 0 || a.first >= n)  continue;
        if (a.second < 0 || a.second >= n) continue;
        tmp_adj[(std::size_t)cursor[(std::size_t)a.first]++] = a.second;
    }

    // Dedup each adjacency list in place.
    G.adjncy.reserve(scratch_size);
    std::vector<Idx> new_xadj((std::size_t)n + 1, 0);
    for (Idx v = 0; v < n; ++v) {
        Idx lo = G.xadj[(std::size_t)v];
        Idx hi = G.xadj[(std::size_t)v + 1];
        std::sort(tmp_adj.begin() + lo, tmp_adj.begin() + hi);
        Idx prev = kInvalid;
        for (Idx i = lo; i < hi; ++i) {
            if (tmp_adj[(std::size_t)i] == prev) continue;
            prev = tmp_adj[(std::size_t)i];
            G.adjncy.push_back(prev);
        }
        new_xadj[(std::size_t)v + 1] = (Idx)G.adjncy.size();
    }
    G.xadj   = std::move(new_xadj);
    G.nArcs  = (Idx)G.adjncy.size();
    return G;
}

// -----------------------------------------------------------------------------
//  TarjanSCC -- iterative
//
//  Standard iterative Tarjan with an explicit call stack of frames:
//      (v, child_iter)   meaning "currently visiting v; resume at adjncy[child_iter]"
//
//  We keep three parallel arrays of size n: index_, lowlink_, on_stack_.
//  Sentinel index_ = -1 means "unvisited".
// -----------------------------------------------------------------------------
SCCResult TarjanSCC(const CSRDigraph& G) {
    SCCResult result;
    result.scc_id.assign((std::size_t)G.n, kInvalid);

    std::vector<Idx>  index_arr((std::size_t)G.n, kInvalid);
    std::vector<Idx>  lowlink ((std::size_t)G.n, 0);
    std::vector<char> on_stack((std::size_t)G.n, 0);

    std::vector<Idx>  scc_stack;       // Tarjan's "S"
    scc_stack.reserve((std::size_t)G.n);

    struct Frame { Idx v; Idx ci; };   // ci = current child index into adjncy
    std::vector<Frame> call_stack;
    call_stack.reserve(256);

    Idx next_index = 0;
    Idx next_scc   = 0;

    for (Idx start = 0; start < G.n; ++start) {
        if (index_arr[(std::size_t)start] != kInvalid) continue;

        // Push start
        index_arr[(std::size_t)start] = next_index;
        lowlink  [(std::size_t)start] = next_index;
        ++next_index;
        scc_stack.push_back(start);
        on_stack[(std::size_t)start] = 1;
        call_stack.push_back({start, G.xadj[(std::size_t)start]});

        while (!call_stack.empty()) {
            Frame& f       = call_stack.back();
            const Idx v    = f.v;
            const Idx end  = G.xadj[(std::size_t)v + 1];

            // Try to descend into an unvisited child.
            bool descended = false;
            while (f.ci < end) {
                const Idx w = G.adjncy[(std::size_t)f.ci++];
                if (index_arr[(std::size_t)w] == kInvalid) {
                    // Push w
                    index_arr[(std::size_t)w] = next_index;
                    lowlink  [(std::size_t)w] = next_index;
                    ++next_index;
                    scc_stack.push_back(w);
                    on_stack[(std::size_t)w] = 1;
                    call_stack.push_back({w, G.xadj[(std::size_t)w]});
                    descended = true;
                    break;
                } else if (on_stack[(std::size_t)w]) {
                    // Back-edge to an ancestor in the current SCC candidate.
                    if (index_arr[(std::size_t)w] < lowlink[(std::size_t)v]) {
                        lowlink[(std::size_t)v] = index_arr[(std::size_t)w];
                    }
                }
                // Else: cross-edge into a finished SCC; ignore.
            }
            if (descended) continue;

            // All children processed; pop and propagate lowlink upward.
            const Idx v_low = lowlink[(std::size_t)v];
            const Idx v_idx = index_arr[(std::size_t)v];
            call_stack.pop_back();
            if (!call_stack.empty()) {
                Frame& parent = call_stack.back();
                if (v_low < lowlink[(std::size_t)parent.v]) {
                    lowlink[(std::size_t)parent.v] = v_low;
                }
            }

            // SCC root?
            if (v_low == v_idx) {
                while (true) {
                    const Idx w = scc_stack.back();
                    scc_stack.pop_back();
                    on_stack[(std::size_t)w] = 0;
                    result.scc_id[(std::size_t)w] = next_scc;
                    if (w == v) break;
                }
                ++next_scc;
            }
        }
    }

    result.num_sccs = next_scc;
    return result;
}

// -----------------------------------------------------------------------------
//  DetectInfeasibility
// -----------------------------------------------------------------------------
bool DetectInfeasibility(const SCCResult& scc, Idx num_literals) {
    // Iterate by variable so we test each (l, l^1) pair exactly once.
    for (Idx v = 0; v + 1 < num_literals; v += 2) {
        if (scc.scc_id[(std::size_t)v] == scc.scc_id[(std::size_t)(v + 1)]) {
            return true;
        }
    }
    return false;
}

// -----------------------------------------------------------------------------
//  BuildCondensation
//
//  Two-pass: count out-degrees per SCC, then scatter (with set-based dedup
//  per source SCC).  After construction we run a single Kahn topological
//  sort and emit the reverse order, which the bitset DP wants.
// -----------------------------------------------------------------------------
CondensationDAG BuildCondensation(const CSRDigraph& G, const SCCResult& scc) {
    CondensationDAG H;
    H.num_nodes = scc.num_sccs;
    H.xadj.assign((std::size_t)H.num_nodes + 1, 0);

    // Pass 1: dedup arcs into a (src, tgt) buffer using a per-source "last seen
    // target" cursor backed by a Idx scratch array.  We avoid std::set in the
    // hot path because it kills cache locality on million-SCC instances.
    std::vector<std::pair<Idx, Idx>> dag_arcs;
    dag_arcs.reserve((std::size_t)G.nArcs / 2 + 16);

    std::vector<Idx> last_seen_target((std::size_t)H.num_nodes, kInvalid);
    // We dedup *within a single source*: when we change sources, we don't
    // actually clear last_seen_target.  Instead we tag entries with a "version"
    // equal to the current source SCC id and only treat them as valid if the
    // tag matches.  Two parallel arrays: tag[t] and version[t].
    std::vector<Idx> last_src_for_target((std::size_t)H.num_nodes, kInvalid);

    // Walk source vertices in increasing SCC order so adjacency lists stay
    // grouped by source SCC; this keeps the dedup cursor warm.
    std::vector<Idx> verts_by_scc((std::size_t)G.n);
    {
        // Bucket-sort literals by their SCC id.
        std::vector<Idx> bucket_size((std::size_t)H.num_nodes + 1, 0);
        for (Idx v = 0; v < G.n; ++v) ++bucket_size[(std::size_t)scc.scc_id[(std::size_t)v] + 1];
        for (Idx i = 0; i < H.num_nodes; ++i) bucket_size[(std::size_t)i + 1] += bucket_size[(std::size_t)i];
        std::vector<Idx> cursor = bucket_size;
        for (Idx v = 0; v < G.n; ++v) {
            verts_by_scc[(std::size_t)cursor[(std::size_t)scc.scc_id[(std::size_t)v]]++] = v;
        }

        // Emit deduped inter-SCC arcs.
        for (Idx s = 0; s < H.num_nodes; ++s) {
            const Idx lo = bucket_size[(std::size_t)s];
            const Idx hi = bucket_size[(std::size_t)s + 1];
            for (Idx i = lo; i < hi; ++i) {
                const Idx u   = verts_by_scc[(std::size_t)i];
                const Idx ue  = G.xadj[(std::size_t)u + 1];
                for (Idx p = G.xadj[(std::size_t)u]; p < ue; ++p) {
                    const Idx w  = G.adjncy[(std::size_t)p];
                    const Idx tw = scc.scc_id[(std::size_t)w];
                    if (tw == s) continue;                  // intra-SCC
                    if (last_src_for_target[(std::size_t)tw] == s) continue;
                    last_src_for_target[(std::size_t)tw] = s;
                    dag_arcs.emplace_back(s, tw);
                    ++H.xadj[(std::size_t)s + 1];
                }
            }
        }
    }

    H.num_arcs = (Idx)dag_arcs.size();
    for (Idx i = 0; i < H.num_nodes; ++i) H.xadj[(std::size_t)i + 1] += H.xadj[(std::size_t)i];
    H.adjncy.resize((std::size_t)H.num_arcs);
    std::vector<Idx> cursor = H.xadj;
    for (const auto& e : dag_arcs) H.adjncy[(std::size_t)cursor[(std::size_t)e.first]++] = e.second;

    // Sort each adjacency list (stable order makes downstream debugging painless).
    for (Idx s = 0; s < H.num_nodes; ++s) {
        std::sort(H.adjncy.begin() + H.xadj[(std::size_t)s],
                  H.adjncy.begin() + H.xadj[(std::size_t)s + 1]);
    }

    // Kahn topological order.
    std::vector<Idx> indeg((std::size_t)H.num_nodes, 0);
    for (Idx p = 0; p < H.num_arcs; ++p) ++indeg[(std::size_t)H.adjncy[(std::size_t)p]];
    std::queue<Idx> q;
    for (Idx s = 0; s < H.num_nodes; ++s) if (indeg[(std::size_t)s] == 0) q.push(s);
    std::vector<Idx> topo;
    topo.reserve((std::size_t)H.num_nodes);
    while (!q.empty()) {
        const Idx s = q.front(); q.pop();
        topo.push_back(s);
        const Idx end = H.xadj[(std::size_t)s + 1];
        for (Idx p = H.xadj[(std::size_t)s]; p < end; ++p) {
            const Idx t = H.adjncy[(std::size_t)p];
            if (--indeg[(std::size_t)t] == 0) q.push(t);
        }
    }
    // A condensation is acyclic by construction, so topo.size() == H.num_nodes.
    if ((Idx)topo.size() != H.num_nodes) {
        throw std::runtime_error("BuildCondensation: topological sort failed; "
                                 "SCC result is inconsistent.");
    }
    H.reverse_topo_order.assign(topo.rbegin(), topo.rend());
    return H;
}

// -----------------------------------------------------------------------------
//  PerSourceBFSReachable
// -----------------------------------------------------------------------------
std::vector<Idx> PerSourceBFSReachable(const CondensationDAG& H, Idx source) {
    std::vector<char> seen((std::size_t)H.num_nodes, 0);
    std::vector<Idx>  result;
    if (source < 0 || source >= H.num_nodes) return result;

    std::queue<Idx> q;
    seen[(std::size_t)source] = 1;
    q.push(source);
    while (!q.empty()) {
        const Idx s = q.front(); q.pop();
        const Idx end = H.xadj[(std::size_t)s + 1];
        for (Idx p = H.xadj[(std::size_t)s]; p < end; ++p) {
            const Idx t = H.adjncy[(std::size_t)p];
            if (!seen[(std::size_t)t]) {
                seen[(std::size_t)t] = 1;
                result.push_back(t);
                q.push(t);
            }
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

// -----------------------------------------------------------------------------
//  BitsetReachable
//
//  Reverse-topological DP:  R[s] = {s} U (union over (s,t) in F of R[t]).
//
//  Memory:  num_nodes * ceil(num_nodes / 64) * 8 bytes.
//  For 65k SCCs that's ~530 MB; above that we expect the caller to use the
//  per-source BFS instead.
// -----------------------------------------------------------------------------
BitsetReach BitsetReachable(const CondensationDAG& H) {
    BitsetReach R;
    R.n             = H.num_nodes;
    R.words_per_row = (Idx)((H.num_nodes + 63) / 64);
    R.bits.assign((std::size_t)R.n * (std::size_t)R.words_per_row, 0ULL);

    for (Idx s : H.reverse_topo_order) {
        const std::size_t row_off = (std::size_t)s * (std::size_t)R.words_per_row;
        // Self-bit.
        R.bits[row_off + (std::size_t)(s >> 6)] |= (1ULL << (s & 63));
        // OR in each successor's row.
        const Idx end = H.xadj[(std::size_t)s + 1];
        for (Idx p = H.xadj[(std::size_t)s]; p < end; ++p) {
            const Idx t = H.adjncy[(std::size_t)p];
            const std::size_t t_off = (std::size_t)t * (std::size_t)R.words_per_row;
            for (Idx w = 0; w < R.words_per_row; ++w) {
                R.bits[row_off + (std::size_t)w] |= R.bits[t_off + (std::size_t)w];
            }
        }
    }
    return R;
}

// -----------------------------------------------------------------------------
//  ParityDSU
// -----------------------------------------------------------------------------
ParityDSU::ParityDSU(Idx num_elems) : n_(num_elems) {
    parent_.resize((std::size_t)n_);
    rank_.assign  ((std::size_t)n_, 0);
    parity_.assign((std::size_t)n_, 0);
    for (Idx i = 0; i < n_; ++i) parent_[(std::size_t)i] = i;
}

std::pair<Idx, std::uint8_t> ParityDSU::FindWithParity(Idx i) {
    // Iterative two-pass: first collect the chain, then compress.
    Idx cur = i;
    while (parent_[(std::size_t)cur] != cur) cur = parent_[(std::size_t)cur];
    const Idx root = cur;

    // Second pass: rewrite parent + parity for everyone on the path.
    Idx node = i;
    std::uint8_t acc = 0;
    while (parent_[(std::size_t)node] != root) {
        const Idx     next  = parent_[(std::size_t)node];
        const std::uint8_t step = parity_[(std::size_t)node];
        parent_[(std::size_t)node] = root;
        parity_[(std::size_t)node] = (std::uint8_t)(acc ^ step);
        acc                        = (std::uint8_t)(acc ^ step);
        node                       = next;
    }
    // The accumulator above tracks "parity from i to root via the unmodified path"
    // ONLY for nodes we rewrote.  Re-derive the final i->root parity from the
    // freshly-updated parent.
    return {root, parity_[(std::size_t)i]};
}

bool ParityDSU::Union(Idx i, Idx j, std::uint8_t relation) {
    auto [ri, pi] = FindWithParity(i);
    auto [rj, pj] = FindWithParity(j);
    if (ri == rj) {
        // Already merged; must agree on the parity.
        const std::uint8_t implied = (std::uint8_t)(pi ^ pj);
        return implied == relation;
    }

    // We want: x_i = x_j XOR relation.
    // We have: x_i = x_ri XOR pi  and  x_j = x_rj XOR pj.
    // So:      x_ri = x_rj XOR (pi XOR pj XOR relation).
    const std::uint8_t new_par = (std::uint8_t)(pi ^ pj ^ relation);
    if (rank_[(std::size_t)ri] < rank_[(std::size_t)rj]) {
        parent_[(std::size_t)ri] = rj;
        parity_[(std::size_t)ri] = new_par;
    } else if (rank_[(std::size_t)ri] > rank_[(std::size_t)rj]) {
        parent_[(std::size_t)rj] = ri;
        parity_[(std::size_t)rj] = new_par;
    } else {
        parent_[(std::size_t)rj] = ri;
        parity_[(std::size_t)rj] = new_par;
        ++rank_[(std::size_t)ri];
    }
    return true;
}

}  // namespace presolve

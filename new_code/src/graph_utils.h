// =============================================================================
//  graph_utils.h
//
//  Dependency-free graph primitives backing the SCIP presolver plugin.
//
//  All routines operate on a CSR-encoded directed graph over [0, n) literals,
//  where literal index 2*i corresponds to the "0-literal" of binary variable i
//  (i.e. x_i = 0) and literal index 2*i + 1 corresponds to the "1-literal"
//  (i.e. x_i = 1).  The partner of literal l is therefore l ^ 1.
//
//  The header is intentionally STL-only so the SCC / DSU / reachability code
//  can be unit-tested without a SCIP installation.  See test/test_graph_utils.cpp.
//
//  Reference: Section 5 ("Implication Graphs and Polytime Procedures") of
//  Barbosa and Validi, "Polytime Procedures for Conflict Inequalities,
//  Elimination, and Fixing" (IJOC, under revision).
// =============================================================================

#ifndef PRESOLVE_GRAPH_UTILS_H_
#define PRESOLVE_GRAPH_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace presolve {

// Index types are kept signed-32 because the implication digraphs we see on
// the full MIPLIB-2017 set top out around 30M arcs (rail03) and 1.5M literals;
// signed 32-bit indices simplify Tarjan's "on stack / low-link / index"
// bookkeeping with a clean -1 sentinel.
using Idx = std::int32_t;

inline constexpr Idx kInvalid = -1;

// Compressed-sparse-row digraph.
//
// xadj has size n+1; out-neighbours of literal v are
//   adjncy[xadj[v] .. xadj[v+1])
//
// The CSR is built once from a triple stream and never mutated; that keeps the
// SCC / BFS passes branch-free in the hot loop.
struct CSRDigraph {
    Idx              n     = 0;   // number of literals (always 2 * #binary vars)
    Idx              nArcs = 0;   // |A|
    std::vector<Idx> xadj;        // size n+1
    std::vector<Idx> adjncy;      // size nArcs

    // Returns the partner literal of v under the (x=0 <-> x=1) involution.
    static inline Idx partner(Idx v) noexcept { return v ^ 1; }
};

// Build a CSR digraph from an unsorted list of (tail, head) arcs.
// Self-loops and duplicate arcs are dropped (an arc is "duplicate" if the
// exact (tail, head) pair has been emitted before).
//
// Complexity: O(n + |arcs|) time, O(n + |arcs|) space.
CSRDigraph BuildCSR(Idx                                       n,
                    const std::vector<std::pair<Idx, Idx>>&   arcs);

// =============================================================================
//  Tarjan SCC (iterative)
// =============================================================================
//
// Returns:
//   scc_id[v]  -- SCC index of literal v, in [0, num_sccs).
//
// We use the iterative variant so we never blow the C stack on instances such
// as rail03 (1.5M literals) where the recursion depth could otherwise exceed
// the default 8 MiB thread stack.
//
// Complexity: O(n + |A|).
struct SCCResult {
    Idx              num_sccs = 0;
    std::vector<Idx> scc_id;        // size n
};

SCCResult TarjanSCC(const CSRDigraph& G);

// Returns true iff there exists a literal v with scc_id[v] == scc_id[v^1].
// In that case the 2-SAT instance is infeasible -- the literal in the SCC
// would imply its own negation.  See Theorem 7 of the main paper.
bool DetectInfeasibility(const SCCResult& scc, Idx num_literals);

// =============================================================================
//  SCC condensation DAG
// =============================================================================
//
// Builds the DAG H = (C, F) where C is the set of SCCs and (S, T) in F iff
// there exists an arc (u, v) in A with scc(u) = S and scc(v) = T (no self
// loops on H).  Duplicate inter-SCC arcs are coalesced.
//
// Also returns a reverse-topological ordering of C -- useful for the
// bitset reachability DP below.
//
// Complexity: O(n + |A|) time and space.
struct CondensationDAG {
    Idx              num_nodes = 0;        // |C|
    Idx              num_arcs  = 0;        // |F|
    std::vector<Idx> xadj;                 // size num_nodes + 1
    std::vector<Idx> adjncy;               // size num_arcs
    std::vector<Idx> reverse_topo_order;   // size num_nodes
};

CondensationDAG BuildCondensation(const CSRDigraph& G, const SCCResult& scc);

// =============================================================================
//  Reachability on the condensation DAG
// =============================================================================
//
// For each SCC, we want to know which SCCs are reachable from it (excluding
// itself by convention; callers add the self bit when interpreting the
// result).  We expose two backends and a thin dispatcher:
//
//   * PerSourceBFSReachable -- O(|C| + |F|) per source query, O(|C|) space.
//     Use when |C|^2 / word_size is uncomfortably large (rule of thumb:
//     |C| > 2^17).
//
//   * BitsetReachable       -- single reverse-topological DP that fills a
//     dense bitset for every SCC in O(|C| * (|C| + |F|) / w) time and
//     O(|C|^2 / w) memory.  Wins when the DAG is dense, but allocate carefully:
//     a 100k-SCC bitset over 100k targets needs ~1.25 GB.
//
// For the AE's "open presolve pipeline" experiments we set the threshold
// kBitsetReachThreshold conservatively at 65536 SCCs and fall back to per-source
// BFS above that.  This is the same threshold the bitset DP in the paper's
// supplement uses (Section EC.4 of the online supplement).
inline constexpr Idx kBitsetReachThreshold = 1 << 16;

// Returns set of forced SCCs given a single source SCC.  The returned vector
// is sorted by SCC id and excludes the source itself.
std::vector<Idx> PerSourceBFSReachable(const CondensationDAG& H, Idx source);

// Computes a |C|-by-|C| dense reachability bitset.  reach[i] has bit j set iff
// SCC j is reachable from SCC i (including i itself).
//
// Bit-packed; row stride is words_per_row = (|C| + 63) / 64.
struct BitsetReach {
    Idx                     n              = 0;   // |C|
    Idx                     words_per_row  = 0;
    std::vector<std::uint64_t> bits;              // size n * words_per_row
    inline bool Test(Idx src, Idx tgt) const noexcept {
        return (bits[(std::size_t)src * (std::size_t)words_per_row + (std::size_t)(tgt >> 6)]
                >> (tgt & 63)) & 1ULL;
    }
};

BitsetReach BitsetReachable(const CondensationDAG& H);

// =============================================================================
//  Disjoint-set union with parity (a.k.a. weighted DSU)
// =============================================================================
//
// Each element carries a parity bit relative to its current root.  Union
// arguments take a relation flag r in {0, 1}:  r = 0  ->  x = y
//                                              r = 1  ->  x = 1 - y
//
// FindWithParity(i) returns (root, p) where p in {0, 1} encodes the parity
// from i to its root: x_i = x_root  if p = 0, and x_i = 1 - x_root  if p = 1.
//
// Returns false from Union iff the two elements were already in the same set
// AND the requested relation is inconsistent with the stored parity
// (infeasibility detection -- though if the SCC step has already passed,
// this never fires; we keep the check anyway as a defensive assertion).
//
// Complexity: O(p alpha(p)) for p elements over O(p) operations.
class ParityDSU {
public:
    explicit ParityDSU(Idx num_elems);
    std::pair<Idx, std::uint8_t> FindWithParity(Idx i);

    // Returns true if the union was consistent (possibly a no-op), false
    // if it conflicts with an earlier union.  Callers should treat false
    // as a hard infeasibility signal.
    bool Union(Idx i, Idx j, std::uint8_t relation);

    Idx num_elems() const noexcept { return n_; }
private:
    Idx                       n_;
    std::vector<Idx>          parent_;
    std::vector<std::uint8_t> rank_;
    std::vector<std::uint8_t> parity_;   // parity to parent_[i]
};

}  // namespace presolve

#endif  // PRESOLVE_GRAPH_UTILS_H_

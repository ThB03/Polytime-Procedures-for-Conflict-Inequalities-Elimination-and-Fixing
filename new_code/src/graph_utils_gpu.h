// =============================================================================
//  graph_utils_gpu.h
//
//  GPU backend for the reachability pass of the implication-graph presolve
//  pipeline.  Drop-in replacement for the CPU bitset reachability backend
//  in graph_utils.h: same Test(s_from, s_to) interface, computed on the GPU
//  via topological-wavefront bitset propagation.
//
//  Why GPU?  The reach phase is the only stage of the pipeline whose
//  asymptotic cost is super-linear in the condensation size |C|.  On the
//  largest instances we have measured (rail01: |C| approximately 235k,
//  3.2 s on CPU; truly large instances with |C| > 500k would exceed
//  practical CPU budgets).  Each level of the DAG is fully parallel; a
//  GPU with thousands of cores collapses the per-level OR loop from
//  O(|C_level| * |C| / 64) sequential work to ~O(|C| / 64) parallel work.
//
//  Memory budget: the dense transitive-closure matrix is |C|^2 / 8 bytes.
//  RTX PRO 5000 Blackwell (48 GiB) holds the closure for |C| up to about
//  620k in a single allocation; larger graphs need batched columns
//  (out of scope for the first version).
//
//  Build: this header is safe to include from .cpp (no CUDA types leak).
//  The implementation is in graph_utils_gpu.cu and links libcudart at
//  build time (controlled by CMake option IMPLGRAPH_HAS_CUDA).
// =============================================================================

#ifndef PRESOLVE_GRAPH_UTILS_GPU_H_
#define PRESOLVE_GRAPH_UTILS_GPU_H_

#include "graph_utils.h"

#include <cstdint>
#include <vector>

namespace presolve {

// Same shape as BitsetReach in graph_utils.h.  We keep a separate type so
// callers can choose backends explicitly; the Test(s, t) bit layout is
// identical so the two are mechanically interchangeable.
struct BitsetReachGPU {
    std::vector<std::uint64_t>  bits;     // row-major, n rows x n_words columns
    Idx                         n        = 0;
    int                         n_words  = 0;

    bool Test(Idx s_from, Idx s_to) const noexcept {
        return (bits[(std::size_t)s_from * (std::size_t)n_words +
                     (std::size_t)(s_to >> 6)] >> (s_to & 63)) & 1ULL;
    }
};

// Compute the transitive closure of the DAG H on the GPU.  Returns a row-major
// bitset matrix in pinned host memory; the matrix is dense, with row v holding
// the reach set of node v.
//
// On any CUDA error (out of memory, no device, kernel launch failure), the
// returned struct has n == 0 and the caller should fall back to the CPU
// implementation.
//
// `verbose`: 0 silent, 1 one-line summary, 2 per-wave timing.
BitsetReachGPU BitsetReachableGPUImpl(const CondensationDAG& H, int verbose = 0);

// Convenience wrapper that always succeeds: if GPU is unavailable or OOMs,
// returns an empty BitsetReachGPU.  This is the entry point the plug-in calls.
inline BitsetReachGPU BitsetReachableGPU(const CondensationDAG& H, int verbose = 0) {
    return BitsetReachableGPUImpl(H, verbose);
}

// Fused reach-and-query: builds the transitive closure of H on the GPU and
// directly answers, for each binary i in [0..num_binaries), whether the
// "scc(2i+1) reaches scc(2i)" and "scc(2i) reaches scc(2i+1)" tests hold.
//
// Output encoding (per binary):
//      0 = no force
//      1 = scc(2i+1) reaches scc(2i)     => force x_i = 0 (F_0)
//      2 = scc(2i)   reaches scc(2i+1)   => force x_i = 1 (F_1)
//      3 = both directions reach          => infeasible
//
// Avoids the |C|^2 / 8 byte DMA back to host that BitsetReachableGPU does;
// only num_binaries bytes are copied back.  This is the API the SCIP
// plug-in calls in production; BitsetReachableGPU stays for correctness
// validation in the bench tool.
//
// On any failure (no CUDA, OOM, kernel error) returns an empty vector.
std::vector<std::uint8_t>
ComputeForcedLiteralsGPU(const CondensationDAG& H,
                         const std::vector<Idx>& scc_id,  // size = 2 * num_binaries
                         Idx num_binaries,
                         int verbose = 0);

// GPU CSR build: same contract as BuildCSR (graph_utils.h), but the
// sort + dedup of adjacency lists runs on the GPU via CUB primitives.
//
// Pipeline:
//   1. Copy arcs to device.
//   2. Radix-sort all arcs lexicographically by (src, tgt).  Now all arcs
//      with the same source are contiguous, and within each source group
//      they are sorted by target.
//   3. cub::DeviceSelect::Unique removes consecutive duplicate (src, tgt)
//      pairs.  Because step 2 sorted globally, this also removes per-source
//      duplicates.
//   4. cub::DeviceHistogram computes per-source out-degrees, then
//      cub::DeviceScan::ExclusiveSum builds xadj.
//   5. Copy xadj and adjncy back to host.
//
// On any CUDA failure returns an empty CSRDigraph; caller should fall back
// to the CPU BuildCSR.  Self-loops and out-of-range arcs are filtered out
// identically to the CPU version.
CSRDigraph BuildCSRGPU(Idx n,
                       const std::vector<std::pair<Idx, Idx>>& arcs,
                       int verbose = 0);

// Probe whether a usable CUDA device is present.  Cheap, returns immediately.
// Used by the plug-in's "auto" dispatch logic to decide CPU-vs-GPU.
bool CudaIsAvailable();

// Free pinned host memory if any was held back.  No-op in the current
// implementation but reserved for future pool-based optimisations.
void GpuShutdown();

}  // namespace presolve

#endif  // PRESOLVE_GRAPH_UTILS_GPU_H_

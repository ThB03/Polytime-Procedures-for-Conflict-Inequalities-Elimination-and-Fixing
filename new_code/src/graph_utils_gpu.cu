// =============================================================================
//  graph_utils_gpu.cu
//
//  GPU backend implementation for the implication-graph reachability pass.
//  See graph_utils_gpu.h for the design rationale.
//
//  Algorithm (topological-wavefront bitset propagation):
//      1. CPU: compute level[v] = 1 + max(level[w] for w in out-neighbours of v)
//         in O(|C| + |F|) via a reverse-topological-order scan.
//      2. Group nodes by level.  All nodes in level k can be processed in
//         parallel because their children live in levels < k.
//      3. GPU: launch one kernel per level (a "wave").  Each thread block
//         processes one node v; the threads in the block cooperatively
//         compute  R[v] = {v} U union over children w of R[w].
//
//  Memory layout: row-major  uint64_t reach[|C|][n_words]  where
//  n_words = ceil(|C| / 64).  Each row holds the dense bitset of nodes
//  reachable from one source node.
//
//  Kernel: one thread block per node, blockDim.x threads per block.
//      thread t handles word indices  t, t + blockDim.x, t + 2*blockDim.x, ...
//      for each word w, accumulator = (self-bit if word contains v else 0);
//                       for each child c, accumulator |= reach[c * n_words + w];
//                       reach[v * n_words + w] = accumulator;
//  This pattern is fully coalesced on read (consecutive threads read
//  consecutive words of the same child row) and on write.
// =============================================================================

#include "graph_utils_gpu.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <vector>

namespace presolve {

namespace {

// Cheap macro for CUDA error checks that returns an empty result on failure.
#define CUDA_TRY(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        std::fprintf(stderr, "[implgraph-gpu] CUDA error at %s:%d: %s\n", \
                     __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return BitsetReachGPU{}; \
    } \
} while (0)

// Per-node propagation kernel.  Each block handles one node v.
//
// nodes_in_wave : array of |wave| node IDs to process this launch.
// xadj, adjncy  : CSR of the condensation DAG H.
// reach         : flat uint64_t array of size |C| * n_words; row v contains
//                 the reach bitset of node v.  Already initialised to zero
//                 on the host side before the first wave.
// n_words       : number of uint64_t words per row.
__global__ void reach_wave_kernel(const Idx* __restrict__ nodes_in_wave,
                                  int                               n_nodes,
                                  const Idx* __restrict__ xadj,
                                  const Idx* __restrict__ adjncy,
                                  std::uint64_t*       __restrict__ reach,
                                  int                               n_words) {
    const int bid = blockIdx.x;
    if (bid >= n_nodes) return;
    const Idx v        = nodes_in_wave[bid];
    const Idx lo       = xadj[v];
    const Idx hi       = xadj[v + 1];
    const Idx self_word = v >> 6;
    const std::uint64_t self_bit  = (std::uint64_t)1 << (v & 63);

    std::uint64_t* Rv = reach + (std::size_t)v * (std::size_t)n_words;

    for (int word = threadIdx.x; word < n_words; word += blockDim.x) {
        std::uint64_t acc = (word == (int)self_word) ? self_bit : 0;
        // Sequential over children; reads to reach[c * n_words + word] are
        // coalesced across the threads of this block (each thread holds a
        // different `word` within the same row).
        for (Idx i = lo; i < hi; ++i) {
            const Idx c = adjncy[i];
            acc |= reach[(std::size_t)c * (std::size_t)n_words + word];
        }
        Rv[word] = acc;
    }
}

// CPU helper: compute the level of each node, where leaves have level 0 and
// level[v] = 1 + max(level[w] for w in out-neighbours of v).  Also returns
// the grouping of nodes by level (waves[k] = sorted list of nodes at level k).
//
// We need a true topological order to assign levels correctly; the H struct
// already exposes one in xadj / adjncy because BuildCondensation outputs the
// DAG in reverse-topological order of construction.  We compute Kahn-style
// from leaves to be safe.
struct WaveLayout {
    std::vector<int>                          level;
    std::vector<std::vector<Idx>>   waves;
};

WaveLayout ComputeWaves(const CondensationDAG& H) {
    WaveLayout L;
    const Idx n = H.num_nodes;
    L.level.assign(n, -1);

    // Iterative DFS to compute level in reverse-topological-order.  We use an
    // explicit stack of (node, child-iterator) frames to avoid stack overflow
    // on deep DAGs (rail-style graphs can have tens of thousands of levels).
    struct Frame { Idx v; Idx ci; };
    std::vector<Frame> stack;
    stack.reserve(256);

    for (Idx start = 0; start < n; ++start) {
        if (L.level[start] != -1) continue;
        stack.emplace_back(Frame{start, H.xadj[start]});
        while (!stack.empty()) {
            Frame& f = stack.back();
            if (f.ci < H.xadj[f.v + 1]) {
                const Idx c = H.adjncy[f.ci++];
                if (L.level[c] == -1) {
                    stack.emplace_back(Frame{c, H.xadj[c]});
                }
                continue;
            }
            // All children of f.v processed: assign level.
            int max_child = -1;
            for (Idx i = H.xadj[f.v]; i < H.xadj[f.v + 1]; ++i) {
                const int lc = L.level[H.adjncy[i]];
                if (lc > max_child) max_child = lc;
            }
            L.level[f.v] = max_child + 1;
            stack.pop_back();
        }
    }

    // Group nodes by level.
    int max_level = -1;
    for (Idx v = 0; v < n; ++v) {
        if (L.level[v] > max_level) max_level = L.level[v];
    }
    L.waves.assign((std::size_t)(max_level + 1), {});
    for (Idx v = 0; v < n; ++v) {
        L.waves[(std::size_t)L.level[v]].push_back(v);
    }
    return L;
}

}  // namespace

// Query kernel: for each binary i in [0, num_binaries), reads the two bits
// (s0_arr[i] reaches s1_arr[i])  and  (s1_arr[i] reaches s0_arr[i])
// from the reach matrix, encodes the status into status[i].
__global__ void query_forced_kernel(const Idx*           __restrict__ s0_arr,
                                    const Idx*           __restrict__ s1_arr,
                                    Idx                                num_binaries,
                                    const std::uint64_t* __restrict__ reach,
                                    int                                n_words,
                                    std::uint8_t*        __restrict__ status) {
    const Idx tid = (Idx)blockIdx.x * (Idx)blockDim.x + (Idx)threadIdx.x;
    if (tid >= num_binaries) return;
    const Idx s0 = s0_arr[tid];
    const Idx s1 = s1_arr[tid];

    const std::uint64_t bit_s0_to_s1 =
        reach[(std::size_t)s0 * (std::size_t)n_words + (s1 >> 6)] >> (s1 & 63);
    const std::uint64_t bit_s1_to_s0 =
        reach[(std::size_t)s1 * (std::size_t)n_words + (s0 >> 6)] >> (s0 & 63);
    const std::uint8_t a = (std::uint8_t)(bit_s0_to_s1 & 1ULL);  // forces x = 1
    const std::uint8_t b = (std::uint8_t)(bit_s1_to_s0 & 1ULL);  // forces x = 0

    // status encoding: 0=none, 1=force0, 2=force1, 3=infeasible
    status[tid] = (std::uint8_t)((b ? 1u : 0u) | (a ? 2u : 0u));
}

std::vector<std::uint8_t>
ComputeForcedLiteralsGPU(const CondensationDAG&  H,
                         const std::vector<Idx>& scc_id,
                         Idx                     num_binaries,
                         int                     verbose) {
    std::vector<std::uint8_t> empty_result;
    if (num_binaries == 0 || H.num_nodes == 0) return empty_result;
    if (!CudaIsAvailable()) return empty_result;

    const Idx n        = H.num_nodes;
    const int n_words  = (int)((n + 63) / 64);
    const std::size_t bytes_reach =
        (std::size_t)n * (std::size_t)n_words * sizeof(std::uint64_t);

    // ---- 1. Compute waves on host (same as BitsetReachableGPUImpl) -------
    WaveLayout L = ComputeWaves(H);
    if (verbose >= 1) {
        std::fprintf(stdout, "[implgraph-gpu] fused-query: |C|=%u, |F|=%u, "
                             "n_words=%d, depth=%zu, n_bins=%u\n",
                     (unsigned)n, (unsigned)H.num_arcs, n_words, L.waves.size(),
                     (unsigned)num_binaries);
        std::fflush(stdout);
    }

    // ---- 2. Pre-compute s0/s1 arrays on host -----------------------------
    std::vector<Idx> s0_host(num_binaries), s1_host(num_binaries);
    for (Idx i = 0; i < num_binaries; ++i) {
        s0_host[(std::size_t)i] = scc_id[(std::size_t)(2 * i    )];
        s1_host[(std::size_t)i] = scc_id[(std::size_t)(2 * i + 1)];
    }

    // ---- 3. GPU memory check ---------------------------------------------
    {
        std::size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        if (bytes_reach + (std::size_t)256 * 1024 * 1024 > free_mem) {
            if (verbose >= 1) {
                std::fprintf(stderr,
                    "[implgraph-gpu] fused-query: matrix needs %.2f GiB but "
                    "only %.2f GiB free; refusing\n",
                    bytes_reach / (1024.0 * 1024.0 * 1024.0),
                    free_mem    / (1024.0 * 1024.0 * 1024.0));
            }
            return empty_result;
        }
    }

    // ---- 4. Allocate device buffers --------------------------------------
    Idx*           d_xadj   = nullptr;
    Idx*           d_adjncy = nullptr;
    std::uint64_t* d_reach  = nullptr;
    Idx*           d_wave   = nullptr;
    Idx*           d_s0     = nullptr;
    Idx*           d_s1     = nullptr;
    std::uint8_t*  d_status = nullptr;
    auto cleanup = [&]() {
        cudaFree(d_xadj); cudaFree(d_adjncy);
        cudaFree(d_reach); cudaFree(d_wave);
        cudaFree(d_s0); cudaFree(d_s1); cudaFree(d_status);
    };

    if (cudaMalloc(&d_xadj,   ((std::size_t)n + 1) * sizeof(Idx)) != cudaSuccess ||
        cudaMalloc(&d_adjncy, (std::size_t)H.num_arcs * sizeof(Idx)) != cudaSuccess ||
        cudaMalloc(&d_reach,  bytes_reach) != cudaSuccess ||
        cudaMalloc(&d_wave,   (std::size_t)n * sizeof(Idx)) != cudaSuccess ||
        cudaMalloc(&d_s0,     (std::size_t)num_binaries * sizeof(Idx)) != cudaSuccess ||
        cudaMalloc(&d_s1,     (std::size_t)num_binaries * sizeof(Idx)) != cudaSuccess ||
        cudaMalloc(&d_status, (std::size_t)num_binaries * sizeof(std::uint8_t)) != cudaSuccess) {
        cleanup();
        return empty_result;
    }

    cudaMemcpy(d_xadj,   H.xadj.data(),
               ((std::size_t)n + 1) * sizeof(Idx), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjncy, H.adjncy.data(),
               (std::size_t)H.num_arcs * sizeof(Idx), cudaMemcpyHostToDevice);
    cudaMemset(d_reach,  0, bytes_reach);
    cudaMemcpy(d_s0,     s0_host.data(),
               (std::size_t)num_binaries * sizeof(Idx), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s1,     s1_host.data(),
               (std::size_t)num_binaries * sizeof(Idx), cudaMemcpyHostToDevice);

    // ---- 5. Reach waves --------------------------------------------------
    cudaEvent_t e_reach_start, e_reach_end, e_query_end;
    cudaEventCreate(&e_reach_start);
    cudaEventCreate(&e_reach_end);
    cudaEventCreate(&e_query_end);
    cudaEventRecord(e_reach_start);

    const int threads_per_block = 128;
    for (std::size_t k = 0; k < L.waves.size(); ++k) {
        const auto& wave = L.waves[k];
        if (wave.empty()) continue;
        cudaMemcpy(d_wave, wave.data(),
                   wave.size() * sizeof(Idx), cudaMemcpyHostToDevice);
        const int blocks = (int)wave.size();
        reach_wave_kernel<<<blocks, threads_per_block>>>(
            d_wave, blocks, d_xadj, d_adjncy, d_reach, n_words);
        cudaError_t le = cudaGetLastError();
        if (le != cudaSuccess) {
            std::fprintf(stderr,
                "[implgraph-gpu] fused-query: kernel failed at wave %zu: %s\n",
                k, cudaGetErrorString(le));
            cleanup();
            return empty_result;
        }
    }
    cudaEventRecord(e_reach_end);

    // ---- 6. Query kernel: per-binary forced-literal status ---------------
    {
        const int q_threads = 256;
        const int q_blocks  = ((int)num_binaries + q_threads - 1) / q_threads;
        query_forced_kernel<<<q_blocks, q_threads>>>(
            d_s0, d_s1, num_binaries, d_reach, n_words, d_status);
        cudaError_t le = cudaGetLastError();
        if (le != cudaSuccess) {
            std::fprintf(stderr,
                "[implgraph-gpu] fused-query: query kernel failed: %s\n",
                cudaGetErrorString(le));
            cleanup();
            return empty_result;
        }
    }
    cudaEventRecord(e_query_end);

    // ---- 7. Copy back small result vector --------------------------------
    std::vector<std::uint8_t> status((std::size_t)num_binaries);
    cudaMemcpy(status.data(), d_status,
               (std::size_t)num_binaries * sizeof(std::uint8_t),
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    float ms_reach = 0.0f, ms_query = 0.0f;
    cudaEventElapsedTime(&ms_reach, e_reach_start, e_reach_end);
    cudaEventElapsedTime(&ms_query, e_reach_end,   e_query_end);
    cudaEventDestroy(e_reach_start);
    cudaEventDestroy(e_reach_end);
    cudaEventDestroy(e_query_end);

    cleanup();

    if (verbose >= 1) {
        std::fprintf(stdout,
            "[implgraph-gpu] fused-query: reach %.2f ms, query %.2f ms, "
            "%.2f GiB matrix (not copied back)\n",
            ms_reach, ms_query,
            bytes_reach / (1024.0 * 1024.0 * 1024.0));
        std::fflush(stdout);
    }
    return status;
}

// ---- BuildCSRGPU: sort + dedup adjacency lists on the GPU via CUB --------
//
// CUB primitives used:
//   cub::DeviceRadixSort::SortPairs   -- sort (src, tgt) pairs by composite key
//   cub::DeviceSelect::Unique         -- compress consecutive duplicates
//   cub::DeviceScan::ExclusiveSum     -- prefix-sum row counts into xadj
//
// All three are 1-shot primitives that take a temp-storage workspace; we
// query the required size, allocate, run, and free.
//
// Key encoding: pack (src, tgt) into a single 64-bit key with src in the
// high 32 bits and tgt in the low 32 bits.  Radix-sorting these keys is
// then a single 64-bit sort and produces the desired lexicographic order.
//
// Counts kernel: after sort+dedup we have N_unique arcs sorted by source.
// A single kernel pass over the sorted arcs increments xadj[src+1] for
// each arc; an exclusive prefix scan then produces the final xadj.

namespace {

// Compose (src, tgt) -> uint64_t.  src in high 32 bits ensures the sort
// orders by src first, tgt second.  Both args are 32-bit unsigned views of
// the underlying int32_t (we already filtered out negatives in the host
// validation pass).
__device__ __host__ inline std::uint64_t pack_key(Idx s, Idx t) {
    return ((std::uint64_t)(std::uint32_t)s << 32) | (std::uint64_t)(std::uint32_t)t;
}
__device__ __host__ inline Idx key_src(std::uint64_t k) {
    return (Idx)(std::uint32_t)(k >> 32);
}
__device__ __host__ inline Idx key_tgt(std::uint64_t k) {
    return (Idx)(std::uint32_t)(k & 0xFFFFFFFFull);
}

// Kernel: increment row_count[key_src(keys[i])] for each unique arc i.
// (NOT row_count[s+1] -- cub::ExclusiveSum produces xadj[i] = sum of
// row_count[0..i-1], so we need row_count[v] = degree of vertex v for
// the resulting xadj[v+1] to equal the cumulative arc count through v.)
// We use atomicAdd because multiple arcs in the same source bin will hit
// the same counter cell; this is the cheapest correct way to count.
__global__ void count_sources_kernel(const std::uint64_t* __restrict__ keys,
                                     int                                n_unique,
                                     Idx*                __restrict__ row_count,
                                     Idx                               n_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_unique) return;
    const Idx s = key_src(keys[tid]);
    if (s < 0 || s >= n_nodes) return;
    atomicAdd((unsigned*)&row_count[s], 1u);
}

// Kernel: split unique keys into adjncy[] = key_tgt(keys[i]).
__global__ void unpack_tgts_kernel(const std::uint64_t* __restrict__ keys,
                                   int                                n_unique,
                                   Idx*                __restrict__ adjncy) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_unique) return;
    adjncy[tid] = key_tgt(keys[tid]);
}

}  // namespace

CSRDigraph BuildCSRGPU(Idx                                    n,
                       const std::vector<std::pair<Idx, Idx>>& arcs,
                       int                                    verbose) {
    CSRDigraph G;
    G.n = n;
    if (n == 0 || arcs.empty() || !CudaIsAvailable()) {
        G.xadj.assign((std::size_t)n + 1, 0);
        return G;
    }

    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3);
    cudaEventRecord(e0);

    // ---- 1. Filter & pack host arcs into a (key) buffer ------------------
    //
    // We mirror BuildCSR's filtering rules: drop self-loops and out-of-range
    // endpoints.  Doing this on the host keeps the GPU code simple; the
    // filter pass is O(|arcs|) and runs in a few ms even for rail01-scale.
    std::vector<std::uint64_t> h_keys;
    h_keys.reserve(arcs.size());
    for (const auto& a : arcs) {
        if (a.first == a.second) continue;
        if (a.first < 0 || a.first  >= n) continue;
        if (a.second < 0 || a.second >= n) continue;
        h_keys.push_back(pack_key(a.first, a.second));
    }
    const int n_in = (int)h_keys.size();
    if (n_in == 0) {
        G.xadj.assign((std::size_t)n + 1, 0);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaEventDestroy(e2); cudaEventDestroy(e3);
        return G;
    }

    // ---- 2. Allocate device buffers --------------------------------------
    std::uint64_t* d_keys_in  = nullptr;
    std::uint64_t* d_keys_out = nullptr;
    std::uint64_t* d_unique   = nullptr;
    int*           d_n_unique = nullptr;
    Idx*           d_row_count = nullptr;
    Idx*           d_xadj    = nullptr;
    Idx*           d_adjncy  = nullptr;
    void*          d_tmp     = nullptr;
    std::size_t    tmp_bytes_sort = 0;
    std::size_t    tmp_bytes_uniq = 0;
    std::size_t    tmp_bytes_scan = 0;

    auto cleanup = [&]() {
        cudaFree(d_keys_in); cudaFree(d_keys_out);
        cudaFree(d_unique);  cudaFree(d_n_unique);
        cudaFree(d_row_count); cudaFree(d_xadj);
        cudaFree(d_adjncy);  cudaFree(d_tmp);
    };

    if (cudaMalloc(&d_keys_in,  (std::size_t)n_in * sizeof(std::uint64_t)) != cudaSuccess ||
        cudaMalloc(&d_keys_out, (std::size_t)n_in * sizeof(std::uint64_t)) != cudaSuccess ||
        cudaMalloc(&d_unique,   (std::size_t)n_in * sizeof(std::uint64_t)) != cudaSuccess ||
        cudaMalloc(&d_n_unique, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&d_row_count, ((std::size_t)n + 1) * sizeof(Idx)) != cudaSuccess ||
        cudaMalloc(&d_xadj,     ((std::size_t)n + 1) * sizeof(Idx)) != cudaSuccess) {
        cleanup();
        G.xadj.assign((std::size_t)n + 1, 0);
        return G;
    }

    cudaMemcpy(d_keys_in, h_keys.data(),
               (std::size_t)n_in * sizeof(std::uint64_t),
               cudaMemcpyHostToDevice);
    cudaMemset(d_row_count, 0, ((std::size_t)n + 1) * sizeof(Idx));

    // ---- 3. Radix-sort keys in [0, n_in) ---------------------------------
    cub::DeviceRadixSort::SortKeys(nullptr, tmp_bytes_sort,
                                   d_keys_in, d_keys_out, n_in);

    // ---- 4. Unique pass to dedup -----------------------------------------
    cub::DeviceSelect::Unique(nullptr, tmp_bytes_uniq,
                              d_keys_out, d_unique, d_n_unique, n_in);

    // ---- 5. Exclusive scan to build xadj from row_count ------------------
    cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes_scan,
                                  d_row_count, d_xadj, (int)(n + 1));

    const std::size_t tmp_bytes =
        std::max(tmp_bytes_sort, std::max(tmp_bytes_uniq, tmp_bytes_scan));
    if (cudaMalloc(&d_tmp, tmp_bytes) != cudaSuccess) {
        cleanup();
        G.xadj.assign((std::size_t)n + 1, 0);
        return G;
    }

    // Execute the three CUB primitives in order.
    cub::DeviceRadixSort::SortKeys(d_tmp, tmp_bytes_sort,
                                   d_keys_in, d_keys_out, n_in);
    cub::DeviceSelect::Unique(d_tmp, tmp_bytes_uniq,
                              d_keys_out, d_unique, d_n_unique, n_in);

    int n_unique = 0;
    cudaMemcpy(&n_unique, d_n_unique, sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(e1);

    // ---- 6. Count sources and run prefix scan -----------------------------
    if (n_unique > 0) {
        if (cudaMalloc(&d_adjncy, (std::size_t)n_unique * sizeof(Idx)) != cudaSuccess) {
            cleanup();
            G.xadj.assign((std::size_t)n + 1, 0);
            return G;
        }
        const int block = 256;
        const int grid  = (n_unique + block - 1) / block;
        count_sources_kernel<<<grid, block>>>(d_unique, n_unique, d_row_count, n);
        unpack_tgts_kernel<<<grid, block>>>  (d_unique, n_unique, d_adjncy);
    }
    cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes_scan,
                                  d_row_count, d_xadj, (int)(n + 1));
    cudaEventRecord(e2);

    // ---- 7. Copy results back --------------------------------------------
    G.xadj.assign((std::size_t)n + 1, 0);
    cudaMemcpy(G.xadj.data(), d_xadj,
               ((std::size_t)n + 1) * sizeof(Idx),
               cudaMemcpyDeviceToHost);
    G.adjncy.assign((std::size_t)n_unique, 0);
    if (n_unique > 0) {
        cudaMemcpy(G.adjncy.data(), d_adjncy,
                   (std::size_t)n_unique * sizeof(Idx),
                   cudaMemcpyDeviceToHost);
    }
    G.nArcs = (Idx)n_unique;
    cudaEventRecord(e3);
    cudaDeviceSynchronize();

    if (verbose >= 1) {
        float ms_sort_uniq = 0, ms_count_scan = 0, ms_copy = 0;
        cudaEventElapsedTime(&ms_sort_uniq, e0, e1);
        cudaEventElapsedTime(&ms_count_scan, e1, e2);
        cudaEventElapsedTime(&ms_copy, e2, e3);
        std::fprintf(stdout,
            "[implgraph-gpu] BuildCSRGPU: %d arcs -> %d unique  "
            "(sort+unique %.2fms, count+scan %.2fms, copyback %.2fms)\n",
            n_in, n_unique, ms_sort_uniq, ms_count_scan, ms_copy);
        std::fflush(stdout);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaEventDestroy(e2); cudaEventDestroy(e3);
    cleanup();
    return G;
}

bool CudaIsAvailable() {
    int dev_count = 0;
    cudaError_t e = cudaGetDeviceCount(&dev_count);
    if (e != cudaSuccess) return false;
    return dev_count > 0;
}

void GpuShutdown() {
    // No pool to free in the current implementation.  Future work: keep a
    // pinned-host arena and free it here so a long-lived SCIP process does
    // not accumulate allocations across solves.
}

BitsetReachGPU BitsetReachableGPUImpl(const CondensationDAG& H, int verbose) {
    BitsetReachGPU R;
    const Idx n = H.num_nodes;
    if (n == 0) return R;
    if (!CudaIsAvailable()) {
        if (verbose >= 1) {
            std::fprintf(stderr, "[implgraph-gpu] no CUDA device; returning empty\n");
        }
        return R;
    }

    const int n_words = (int)((n + 63) / 64);

    // ---- 1. Compute topological wavefront on host ------------------------
    WaveLayout L = ComputeWaves(H);
    if (verbose >= 1) {
        std::fprintf(stdout, "[implgraph-gpu] |C|=%u, |F|=%u, n_words=%d, "
                             "depth=%zu\n",
                     (unsigned)n, (unsigned)H.num_arcs, n_words, L.waves.size());
        std::fflush(stdout);
    }

    // ---- 2. Allocate GPU memory -----------------------------------------
    //
    // Total reach matrix: n * n_words * 8 bytes.  Bail out cleanly if this
    // does not fit so the caller can fall back to CPU.
    const std::size_t bytes_reach = (std::size_t)n * (std::size_t)n_words * sizeof(std::uint64_t);
    {
        std::size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        if (bytes_reach + (std::size_t)256 * 1024 * 1024 > free_mem) {
            if (verbose >= 1) {
                std::fprintf(stderr,
                    "[implgraph-gpu] reach matrix needs %.2f GiB but only "
                    "%.2f GiB free; refusing\n",
                    bytes_reach / (1024.0 * 1024.0 * 1024.0),
                    free_mem    / (1024.0 * 1024.0 * 1024.0));
            }
            return R;
        }
    }

    Idx* d_xadj = nullptr;
    Idx* d_adjncy = nullptr;
    std::uint64_t* d_reach = nullptr;
    Idx* d_wave = nullptr;

    CUDA_TRY(cudaMalloc(&d_xadj,   ((std::size_t)n + 1) * sizeof(Idx)));
    CUDA_TRY(cudaMalloc(&d_adjncy, (std::size_t)H.num_arcs * sizeof(Idx)));
    CUDA_TRY(cudaMalloc(&d_reach,  bytes_reach));
    CUDA_TRY(cudaMalloc(&d_wave,   (std::size_t)n * sizeof(Idx)));

    CUDA_TRY(cudaMemcpy(d_xadj,   H.xadj.data(),
                        ((std::size_t)n + 1) * sizeof(Idx),
                        cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_adjncy, H.adjncy.data(),
                        (std::size_t)H.num_arcs * sizeof(Idx),
                        cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemset(d_reach, 0, bytes_reach));

    // ---- 3. Launch waves in order ---------------------------------------
    const int threads_per_block = 128;
    cudaEvent_t start_evt, stop_evt;
    cudaEventCreate(&start_evt);
    cudaEventCreate(&stop_evt);
    cudaEventRecord(start_evt);

    for (std::size_t k = 0; k < L.waves.size(); ++k) {
        const auto& wave = L.waves[k];
        if (wave.empty()) continue;
        CUDA_TRY(cudaMemcpy(d_wave, wave.data(),
                            wave.size() * sizeof(Idx),
                            cudaMemcpyHostToDevice));
        const int blocks = (int)wave.size();
        reach_wave_kernel<<<blocks, threads_per_block>>>(
            d_wave, blocks, d_xadj, d_adjncy, d_reach, n_words);
        cudaError_t le = cudaGetLastError();
        if (le != cudaSuccess) {
            std::fprintf(stderr,
                "[implgraph-gpu] kernel launch failed at wave %zu: %s\n",
                k, cudaGetErrorString(le));
            cudaFree(d_xadj); cudaFree(d_adjncy);
            cudaFree(d_reach); cudaFree(d_wave);
            return BitsetReachGPU{};
        }
    }
    CUDA_TRY(cudaDeviceSynchronize());

    cudaEventRecord(stop_evt);
    cudaEventSynchronize(stop_evt);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_evt, stop_evt);
    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);

    // ---- 4. Copy reach matrix back to host ------------------------------
    R.n        = n;
    R.n_words  = n_words;
    R.bits.assign((std::size_t)n * (std::size_t)n_words, 0);
    CUDA_TRY(cudaMemcpy(R.bits.data(), d_reach, bytes_reach, cudaMemcpyDeviceToHost));

    cudaFree(d_xadj);
    cudaFree(d_adjncy);
    cudaFree(d_reach);
    cudaFree(d_wave);

    if (verbose >= 1) {
        std::fprintf(stdout,
            "[implgraph-gpu] reach: %d waves, %.2f ms GPU, %.2f GiB matrix\n",
            (int)L.waves.size(), ms,
            bytes_reach / (1024.0 * 1024.0 * 1024.0));
        std::fflush(stdout);
    }
    return R;
}

}  // namespace presolve

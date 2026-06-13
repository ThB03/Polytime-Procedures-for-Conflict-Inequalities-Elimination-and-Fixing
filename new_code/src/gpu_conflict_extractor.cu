// =============================================================================
//  gpu_conflict_extractor.cu
//
//  GPU implementation of the single-constraint residual conflict extractor.
//  See gpu_conflict_extractor.h for the design and algorithm.
//
//  Kernel structure: one CUDA block per constraint.  Threads in the block
//  cooperate to (a) gather the binary subset of that constraint's columns
//  into shared memory, (b) iterate over the O(k^2) pairs of binaries, and
//  (c) atomically append conflict arcs into a flat output buffer.
//
//  Output buffer: a single int32_t array of size 2 * arc_buf_cap.  The
//  number actually written lives in a separate counter (arc_count).  If
//  the counter exceeds arc_buf_cap, later atomicAdd attempts still bump
//  the counter but the host clips at arc_buf_cap and warns.
//
//  Dedup: post-kernel we pack (src, tgt) as uint64_t, radix-sort, and
//  cub::DeviceSelect::Unique, then unpack.  Same idea as BuildCSRGPU.
// =============================================================================

#include "gpu_conflict_extractor.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <vector>

namespace presolve {

namespace {

#define CUDA_TRY_OR_EMPTY(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        std::fprintf(stderr, "[gpu-conflict] CUDA error at %s:%d: %s\n", \
                     __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return std::vector<int32_t>{}; \
    } \
} while (0)

// Per-row kernel.  Each block handles one constraint row.
//
//   row_xadj[nrows + 1]            CSR row pointers
//   row_colidx[nnz]                column indices, sized to total nonzero count
//   row_coef[nnz]                  coefficients
//   row_lhs[nrows], row_rhs[nrows] per-row LHS / RHS bounds
//   row_sum_min[nrows]             pre-computed sum over all cols of min(coef*x)
//   row_sum_max[nrows]             pre-computed sum over all cols of max(coef*x)
//   binary_index[ncols]            -1 for non-binary, else 0..nbin-1
//   nbin                           number of binaries
//   arc_buf, arc_count, arc_cap    output buffer + atomic counter + capacity
//   feastol                        infeasibility tolerance
__global__ void extract_conflicts_kernel(
    const int*    __restrict__ row_xadj,
    const int*    __restrict__ row_colidx,
    const double* __restrict__ row_coef,
    const double* __restrict__ row_lhs,
    const double* __restrict__ row_rhs,
    const double* __restrict__ row_sum_min,
    const double* __restrict__ row_sum_max,
    const int*    __restrict__ binary_index,
    int                        nbin,
    int                        nrows,
    int                        max_row_width,
    int*          __restrict__ arc_buf,
    unsigned int* __restrict__ arc_count,
    int                        arc_cap,
    double                     feastol)
{
    const int row = blockIdx.x;
    if (row >= nrows) return;

    const int  lo  = row_xadj[row];
    const int  hi  = row_xadj[row + 1];
    const double rhs = row_rhs[row];
    const double lhs = row_lhs[row];
    const double sum_min = row_sum_min[row];
    const double sum_max = row_sum_max[row];

    // Pass 1: count binary columns in this row.  Use shared atomic.
    __shared__ int   s_n_bin;
    extern __shared__ unsigned char s_mem[];
    int*    s_bin_idx  = reinterpret_cast<int*>   (s_mem);
    double* s_bin_coef = reinterpret_cast<double*>(s_bin_idx + 1024);

    if (threadIdx.x == 0) s_n_bin = 0;
    __syncthreads();

    for (int i = lo + (int)threadIdx.x; i < hi; i += (int)blockDim.x) {
        const int c = row_colidx[i];
        const int b = binary_index[c];
        if (b >= 0) {
            const int slot = atomicAdd(&s_n_bin, 1);
            if (slot < 1024) {
                s_bin_idx [slot] = b;
                s_bin_coef[slot] = row_coef[i];
            }
        }
    }
    __syncthreads();

    const int n_bin_row = s_n_bin;
    if (n_bin_row < 2)                 return;          // need at least one pair
    if (n_bin_row > max_row_width)     return;          // skip pathologically wide rows
    if (n_bin_row > 1024)              return;          // shared-mem cap

    // Pre-compute bin_min[i] / bin_max[i] for each binary in this row.  We
    // do it in-place on s_bin_coef -- saves shared memory.  Actually we
    // need both the coef AND the (min, max) so let's just compute on the
    // fly inside the inner loop; it's only two comparisons per pair.

    // Pass 3: iterate ALL pairs.  Pair (i, j) with i < j.  Total pairs =
    // n_bin_row * (n_bin_row - 1) / 2.  Distribute across threads.
    const int total_pairs = (n_bin_row * (n_bin_row - 1)) / 2;
    for (int p = (int)threadIdx.x; p < total_pairs; p += (int)blockDim.x) {
        // Convert linear pair index p -> (i, j), i < j.
        // Solve: p = i*(2*n - i - 1)/2 + (j - i - 1).  We can compute
        // i = floor((2*n - 1 - sqrt((2*n - 1)^2 - 8*p)) / 2)  but that's
        // expensive.  Use a simple unranking loop -- O(n_bin_row) per
        // pair but with cache-friendly memory access.  For n_bin_row up
        // to a few hundred this is cheap.
        int rem = p, i = 0;
        while (true) {
            const int row_pairs = n_bin_row - 1 - i;
            if (rem < row_pairs) break;
            rem -= row_pairs;
            ++i;
        }
        const int j = i + 1 + rem;

        const int    bi   = s_bin_idx[i];
        const int    bj   = s_bin_idx[j];
        const double ai   = s_bin_coef[i];
        const double aj   = s_bin_coef[j];
        const double mini = (ai < 0) ? ai : 0.0;
        const double maxi = (ai < 0) ? 0.0 : ai;
        const double minj = (aj < 0) ? aj : 0.0;
        const double maxj = (aj < 0) ? 0.0 : aj;

        // Sum of min/max OVER OTHER COLUMNS (binaries and continuous).
        // Both binary i and binary j are excluded.
        const double rest_min = sum_min - mini - minj;
        const double rest_max = sum_max - maxi - maxj;

        for (int vi = 0; vi < 2; ++vi) {
            for (int vj = 0; vj < 2; ++vj) {
                const double pair_contrib = ai * (double)vi + aj * (double)vj;
                const double total_min    = rest_min + pair_contrib;
                const double total_max    = rest_max + pair_contrib;
                const bool   over_rhs     = (total_min > rhs + feastol);
                const bool   under_lhs    = (total_max < lhs - feastol);
                if (!over_rhs && !under_lhs) continue;

                // (xi = vi) AND (xj = vj) infeasible.  Emit two arcs:
                //   (xi = vi) -> (xj = 1 - vj)
                //   (xj = vj) -> (xi = 1 - vi)
                const int lit_i      = 2 * bi + vi;
                const int lit_j      = 2 * bj + vj;
                const int lit_i_neg  = 2 * bi + (1 - vi);
                const int lit_j_neg  = 2 * bj + (1 - vj);

                const unsigned int pos = atomicAdd(arc_count, 2u);
                if ((int)pos + 3 < arc_cap) {
                    arc_buf[2 * pos    ] = lit_i;
                    arc_buf[2 * pos + 1] = lit_j_neg;
                    arc_buf[2 * pos + 2] = lit_j;
                    arc_buf[2 * pos + 3] = lit_i_neg;
                }
            }
        }
    }
}

}  // namespace

std::vector<int32_t>
ExtractConflictsGPU(const ConstraintMatrix& M, const GpuExtractOptions& opts) {
    std::vector<int32_t> empty;
    if (M.n_rows == 0 || M.n_binaries == 0 || M.row_coef.empty()) {
        return empty;
    }

    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::fprintf(stderr, "[gpu-conflict] no CUDA device available\n");
        return empty;
    }

    // ---- 1. Pre-compute per-row sum_min / sum_max on host ---------------
    //
    // sum_min[r] = sum over all i in row r of  min(coef[r][i] * x_i)
    // sum_max[r] = sum over all i in row r of  max(coef[r][i] * x_i)
    // For binary x:  min = min(0, coef), max = max(0, coef).
    // For continuous x in [lb, ub]:  min = min(coef*lb, coef*ub), etc.
    std::vector<double> row_sum_min(M.n_rows, 0.0);
    std::vector<double> row_sum_max(M.n_rows, 0.0);
    for (int r = 0; r < M.n_rows; ++r) {
        for (int i = M.row_xadj[r]; i < M.row_xadj[r + 1]; ++i) {
            const int    c = M.row_colidx[i];
            const double a = M.row_coef[i];
            const bool   is_bin = (M.binary_index[c] >= 0);
            double mn, mx;
            if (is_bin) {
                mn = (a < 0.0) ? a   : 0.0;
                mx = (a < 0.0) ? 0.0 : a;
            } else {
                const double v0 = a * M.col_lb[c];
                const double v1 = a * M.col_ub[c];
                mn = (v0 < v1) ? v0 : v1;
                mx = (v0 < v1) ? v1 : v0;
            }
            row_sum_min[r] += mn;
            row_sum_max[r] += mx;
        }
    }

    if (opts.verbose) {
        std::fprintf(stdout, "[gpu-conflict] matrix: n_cols=%d n_rows=%d nnz=%zu "
                             "n_binaries=%d\n",
                     M.n_cols, M.n_rows, M.row_coef.size(), M.n_binaries);
        std::fflush(stdout);
    }

    // ---- 2. Allocate device buffers --------------------------------------
    int*    d_row_xadj    = nullptr;
    int*    d_row_colidx  = nullptr;
    double* d_row_coef    = nullptr;
    double* d_row_lhs     = nullptr;
    double* d_row_rhs     = nullptr;
    double* d_row_sum_min = nullptr;
    double* d_row_sum_max = nullptr;
    int*    d_binary_idx  = nullptr;
    int*    d_arc_buf     = nullptr;
    unsigned int* d_arc_count = nullptr;

    auto cleanup = [&]() {
        cudaFree(d_row_xadj);    cudaFree(d_row_colidx); cudaFree(d_row_coef);
        cudaFree(d_row_lhs);     cudaFree(d_row_rhs);
        cudaFree(d_row_sum_min); cudaFree(d_row_sum_max);
        cudaFree(d_binary_idx);  cudaFree(d_arc_buf);    cudaFree(d_arc_count);
    };

    const std::size_t nnz = M.row_coef.size();
    CUDA_TRY_OR_EMPTY(cudaMalloc(&d_row_xadj,    ((std::size_t)M.n_rows + 1) * sizeof(int)));
    CUDA_TRY_OR_EMPTY(cudaMalloc(&d_row_colidx,  nnz * sizeof(int)));
    CUDA_TRY_OR_EMPTY(cudaMalloc(&d_row_coef,    nnz * sizeof(double)));
    CUDA_TRY_OR_EMPTY(cudaMalloc(&d_row_lhs,     (std::size_t)M.n_rows * sizeof(double)));
    CUDA_TRY_OR_EMPTY(cudaMalloc(&d_row_rhs,     (std::size_t)M.n_rows * sizeof(double)));
    CUDA_TRY_OR_EMPTY(cudaMalloc(&d_row_sum_min, (std::size_t)M.n_rows * sizeof(double)));
    CUDA_TRY_OR_EMPTY(cudaMalloc(&d_row_sum_max, (std::size_t)M.n_rows * sizeof(double)));
    CUDA_TRY_OR_EMPTY(cudaMalloc(&d_binary_idx,  (std::size_t)M.n_cols * sizeof(int)));
    CUDA_TRY_OR_EMPTY(cudaMalloc(&d_arc_buf,     (std::size_t)opts.arc_buf_cap * 2 * sizeof(int)));
    CUDA_TRY_OR_EMPTY(cudaMalloc(&d_arc_count,   sizeof(unsigned int)));

    cudaMemcpy(d_row_xadj,    M.row_xadj.data(),    ((std::size_t)M.n_rows + 1) * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_colidx,  M.row_colidx.data(),  nnz * sizeof(int),                            cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_coef,    M.row_coef.data(),    nnz * sizeof(double),                         cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_lhs,     M.row_lhs.data(),     (std::size_t)M.n_rows * sizeof(double),       cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_rhs,     M.row_rhs.data(),     (std::size_t)M.n_rows * sizeof(double),       cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_sum_min, row_sum_min.data(),   (std::size_t)M.n_rows * sizeof(double),       cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_sum_max, row_sum_max.data(),   (std::size_t)M.n_rows * sizeof(double),       cudaMemcpyHostToDevice);
    cudaMemcpy(d_binary_idx,  M.binary_index.data(),(std::size_t)M.n_cols * sizeof(int),          cudaMemcpyHostToDevice);
    cudaMemset(d_arc_count, 0, sizeof(unsigned int));

    // ---- 3. Launch the kernel -------------------------------------------
    const int threads = 128;
    const int blocks  = M.n_rows;
    // Shared memory: s_bin_idx[1024] + s_bin_coef[1024] = 4 KB + 8 KB = 12 KB
    // Plus s_n_bin: 4 bytes.  Round up.
    const size_t shmem_bytes = 1024 * sizeof(int) + 1024 * sizeof(double);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);

    extract_conflicts_kernel<<<blocks, threads, shmem_bytes>>>(
        d_row_xadj, d_row_colidx, d_row_coef,
        d_row_lhs,  d_row_rhs,
        d_row_sum_min, d_row_sum_max,
        d_binary_idx, M.n_binaries, M.n_rows,
        opts.max_row_width,
        d_arc_buf, d_arc_count, opts.arc_buf_cap,
        opts.feastol);

    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {
        std::fprintf(stderr, "[gpu-conflict] kernel launch failed: %s\n",
                     cudaGetErrorString(le));
        cleanup();
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        return empty;
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms_kernel = 0.0f;
    cudaEventElapsedTime(&ms_kernel, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);

    // ---- 4. Read back arc count ------------------------------------------
    unsigned int n_written = 0;
    cudaMemcpy(&n_written, d_arc_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    const int n_arcs_raw = (int)std::min<unsigned int>(n_written, (unsigned int)opts.arc_buf_cap);
    if ((int)n_written > opts.arc_buf_cap) {
        std::fprintf(stderr,
            "[gpu-conflict] WARNING: kernel wanted %u arcs but buffer "
            "capped at %d; output truncated -- raise --cap\n",
            n_written, opts.arc_buf_cap);
    }
    if (opts.verbose) {
        std::fprintf(stdout, "[gpu-conflict] kernel: %d arcs in %.2f ms\n",
                     n_arcs_raw, ms_kernel);
        std::fflush(stdout);
    }

    // ---- 5. Copy arcs back to host (host-side dedup is cheap at our scale) --
    std::vector<int32_t> result((std::size_t)n_arcs_raw * 2);
    if (n_arcs_raw > 0) {
        cudaMemcpy(result.data(), d_arc_buf,
                   (std::size_t)n_arcs_raw * 2 * sizeof(int),
                   cudaMemcpyDeviceToHost);
    }
    cleanup();

    // ---- 6. Host-side dedup ----------------------------------------------
    //
    // Pack each (src, tgt) into a uint64_t, sort, std::unique.  Dedup turns
    // the typical 5-10x duplicate emission rate (same conflict implied by
    // multiple constraints) into a clean unique arc set.
    if (opts.dedup && n_arcs_raw > 1) {
        std::vector<std::uint64_t> keys((std::size_t)n_arcs_raw);
        for (int i = 0; i < n_arcs_raw; ++i) {
            const std::uint32_t s = (std::uint32_t)result[2 * i];
            const std::uint32_t t = (std::uint32_t)result[2 * i + 1];
            keys[(std::size_t)i] = ((std::uint64_t)s << 32) | (std::uint64_t)t;
        }
        std::sort(keys.begin(), keys.end());
        auto new_end = std::unique(keys.begin(), keys.end());
        const std::size_t n_unique = (std::size_t)(new_end - keys.begin());
        result.assign(n_unique * 2, 0);
        for (std::size_t i = 0; i < n_unique; ++i) {
            result[2 * i    ] = (int32_t)(std::uint32_t)(keys[i] >> 32);
            result[2 * i + 1] = (int32_t)(std::uint32_t)(keys[i] & 0xFFFFFFFFu);
        }
        if (opts.verbose) {
            std::fprintf(stdout, "[gpu-conflict] dedup: %d -> %zu arcs\n",
                         n_arcs_raw, n_unique);
        }
    }

    if (opts.verbose) {
        std::fprintf(stdout, "[gpu-conflict] returning %zu arc pairs\n",
                     result.size() / 2);
        std::fflush(stdout);
    }
    return result;
}

}  // namespace presolve

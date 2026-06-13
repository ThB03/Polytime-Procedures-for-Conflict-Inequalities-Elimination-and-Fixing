// =============================================================================
//  gpu_conflict_extractor.h
//
//  Standalone GPU-native conflict graph builder.  Reads a constraint matrix
//  (extracted from SCIP) and emits all binary-binary conflict arcs that
//  follow from per-constraint residual analysis.  Designed as a drop-in
//  replacement for scripts/build_conflict_graph.py (CBC) on instances where
//  CBC's CoinConflictGraph runs out of memory.
//
//  Algorithm (single-constraint residual):
//
//      For each constraint c:   lhs[c]  <=  sum_i coef[c][i] * x_i  <=  rhs[c]
//      pre-compute:
//          sum_min[c] = sum over all i of  min(coef[c][i] * x_i)
//          sum_max[c] = sum over all i of  max(coef[c][i] * x_i)
//      For each pair (i, j) of binary variables in constraint c:
//          For each (vi, vj) in {0, 1}^2:
//              residual_min = sum_min[c]
//                             - bin_min[c][i] - bin_min[c][j]
//                             + vi * coef[c][i] + vj * coef[c][j]
//              residual_max = sum_max[c]
//                             - bin_max[c][i] - bin_max[c][j]
//                             + vi * coef[c][i] + vj * coef[c][j]
//              if residual_min > rhs[c] + eps    or
//                 residual_max < lhs[c] - eps:
//                  emit conflict (vi, vj)
//
//  This captures conflicts CBC discovers via single-constraint analysis,
//  which is the bulk of binary-binary conflicts on most MIPLIB instances.
//  CBC's full LP-propagation chains across constraints are NOT replicated
//  here; that's future work (a multi-pass GPU propagator would handle it).
//
//  Output: list of (src_lit, tgt_lit) pairs, where literal lit(v, a) is
//  encoded as 2 * binary_index(v) + a, matching the rest of the codebase.
// =============================================================================

#ifndef PRESOLVE_GPU_CONFLICT_EXTRACTOR_H_
#define PRESOLVE_GPU_CONFLICT_EXTRACTOR_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace presolve {

// Host-side flat constraint matrix.  All arrays are sized to fit completely
// into one cudaMalloc on the GPU and are designed for coalesced reads.
struct ConstraintMatrix {
    // Per-column data, indexed by SCIP's column ordering.  binary_index[c]
    // is -1 if column c is not an active binary variable, else its 0..p-1
    // dense index used in the output arc encoding.
    std::vector<int>      binary_index;     // size n_cols
    std::vector<double>   col_lb;           // size n_cols
    std::vector<double>   col_ub;           // size n_cols
    std::vector<std::string> binary_names;  // size p (output ordering)

    // CSR storage of the rows we extracted (linear, setppc, logicor).  Each
    // row has lhs/rhs bounds; lhs[r] = -infty / rhs[r] = +infty mark
    // one-sided rows.
    std::vector<int>      row_xadj;         // size n_rows + 1
    std::vector<int>      row_colidx;       // size nnz
    std::vector<double>   row_coef;         // size nnz
    std::vector<double>   row_lhs;          // size n_rows
    std::vector<double>   row_rhs;          // size n_rows

    int n_cols = 0;
    int n_rows = 0;
    int n_binaries = 0;
};

struct GpuExtractOptions {
    // Hard cap on the number of (src, tgt) arc pairs the GPU buffer holds.
    // Each pair is 2 int32_t = 8 bytes; default 2B pairs = 16 GiB.  Sized
    // for the 48 GiB RTX PRO 5000; lower on smaller cards.  Some MIPLIB
    // instances with dense binary constraints produce >1B raw arcs before
    // dedup (e.g. rail-family, nw04-family).
    int     arc_buf_cap = 2000000000;

    // Skip rows wider than max_row_width binaries.  Per-row work scales
    // O(k^2) in binary count k; rows with k > 1024 typically come from
    // knapsack-style constraints where this approach is suboptimal anyway.
    int     max_row_width = 1024;

    // Numerical tolerance for residual-bound infeasibility checks.
    double  feastol = 1e-9;

    // If true, run cub::DeviceRadixSort + DeviceSelect::Unique to dedup
    // arcs (multiple constraints often imply the same conflict).
    bool    dedup = true;

    // If true, print one-line summaries of each phase.
    bool    verbose = false;
};

// Extract binary-binary conflict arcs from the constraint matrix on the GPU.
// Returns a flat array of literal pairs: result[2k], result[2k+1] is the
// k-th arc (src_lit -> tgt_lit).  Literals are encoded as
// 2 * binary_index + value (value in {0, 1}).
//
// Both directions of each conflict are emitted: (vi, vj) infeasible yields
//      (lit(i, vi), lit(j, 1 - vj))    and    (lit(j, vj), lit(i, 1 - vi))
// matching CBC's symmetric ConflictGraph output.
//
// On any CUDA failure returns an empty vector.
std::vector<int32_t>
ExtractConflictsGPU(const ConstraintMatrix& M, const GpuExtractOptions& opts = {});

}  // namespace presolve

#endif  // PRESOLVE_GPU_CONFLICT_EXTRACTOR_H_

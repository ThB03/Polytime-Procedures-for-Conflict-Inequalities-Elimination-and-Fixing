// =============================================================================
//  cbc_extractor.cpp
//
//  Manual probing through OsiClpSolverInterface + CglProbing pre-pass.
//  See header for design notes.
//
//  Build deps:
//      coinor-libcbc-dev   (provides CBC + CglProbing)
//      coinor-libcgl-dev
//      coinor-libosi-dev
//      coinor-libclp-dev
//      coinor-libcoinutils-dev
//  Linux:  libCbc.so, libCgl.so, libOsiClp.so, libClp.so, libCoinUtils.so
// =============================================================================

#include "cbc_extractor.h"

#include <CglProbing.hpp>
#include <CoinHelperFunctions.hpp>
#include <OsiClpSolverInterface.hpp>
#include <OsiCuts.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <unistd.h>

namespace presolve {

namespace {

// RAII wrapper around mkstemps to ensure the temp file gets unlinked even
// on early return / exception.
struct TempMpsFile {
    char path[64];
    bool valid = false;
    TempMpsFile() {
        std::snprintf(path, sizeof(path), "/tmp/implgraph_cbc_XXXXXX.mps");
        int fd = mkstemps(path, 4);
        if (fd >= 0) { close(fd); valid = true; }
    }
    ~TempMpsFile() { if (valid) unlink(path); }
};

// Convenience: log timing if verbose.
inline double seconds_since(std::clock_t start) {
    return (double)(std::clock() - start) / (double)CLOCKS_PER_SEC;
}

}  // namespace

std::vector<std::pair<Idx, Idx>>
ExtractImplicationArcsViaCBC(SCIP*                                       scip,
                              const std::unordered_map<SCIP_VAR*, Idx>&   var_to_index,
                              const CbcExtractOptions&                    opts) {
    std::vector<std::pair<Idx, Idx>> arcs;
    if (var_to_index.empty()) return arcs;
    const std::clock_t t0 = std::clock();

    // --- 1. Write SCIP's currently-transformed problem to a temp MPS. ---
    //
    // We use the transformed (post-presolve-so-far) problem so any reductions
    // SCIP has already applied are reflected, and our probing operates on
    // the live constraint matrix that downstream presolve will see.
    TempMpsFile tmp;
    if (!tmp.valid) return arcs;
    if (SCIPwriteTransProblem(scip, tmp.path, "mps", FALSE) != SCIP_OKAY) {
        return arcs;
    }
    if (opts.verbose) {
        std::printf("[cbc_extractor]   wrote SCIP MPS to %s in %.2fs\n",
                    tmp.path, seconds_since(t0));
    }

    // --- 2. Read into CBC's Osi solver. ---
    OsiClpSolverInterface solver;
    solver.messageHandler()->setLogLevel(opts.cbc_log_level);
    {
        const std::clock_t t1 = std::clock();
        if (solver.readMps(tmp.path, "mps") < 0) return arcs;
        if (opts.verbose) {
            std::printf("[cbc_extractor]   CBC read MPS in %.2fs (n=%d cols, m=%d rows)\n",
                        seconds_since(t1), solver.getNumCols(), solver.getNumRows());
        }
    }
    const int ncols = solver.getNumCols();
    if (ncols == 0) return arcs;

    // --- 3. Build CBC col -> our Idx lookup via variable names. ---
    //
    // SCIP transformed names look like "t_<orig_name>" or just "<orig_name>";
    // we try both and also fall back to the original SCIP variable name.
    std::unordered_map<std::string, Idx> name_to_our_idx;
    name_to_our_idx.reserve(var_to_index.size() * 2);
    for (const auto& kv : var_to_index) {
        const std::string n(SCIPvarGetName(kv.first));
        name_to_our_idx[n] = kv.second;
        if (n.size() >= 2 && n[0] == 't' && n[1] == '_') {
            name_to_our_idx[n.substr(2)] = kv.second;
        } else {
            name_to_our_idx["t_" + n] = kv.second;
        }
    }
    std::vector<Idx> col_to_our_idx((std::size_t)ncols, kInvalid);
    int n_mapped = 0;
    for (int c = 0; c < ncols; ++c) {
        auto it = name_to_our_idx.find(solver.getColName(c));
        if (it != name_to_our_idx.end()) {
            col_to_our_idx[(std::size_t)c] = it->second;
            ++n_mapped;
        }
    }
    if (opts.verbose) {
        std::printf("[cbc_extractor]   mapped %d / %d CBC columns to SCIP binaries\n",
                    n_mapped, ncols);
    }
    if (n_mapped == 0) return arcs;

    // --- 4. Optional CglProbing pre-pass for unconditional tightenings. ---
    if (opts.run_cgl_prepass) {
        const std::clock_t t2 = std::clock();
        CglProbing probing;
        probing.setMaxPass(1);
        probing.setMaxProbe(100);
        probing.setMaxLook(50);
        probing.setUsingObjective(0);
        probing.setRowCuts(0);                  // we want col cuts only
        OsiCuts cs;
        try {
            probing.generateCuts(solver, cs);
        } catch (...) {
            // Some CglProbing versions throw on infeasibility; ignore here.
        }
        // CglProbing's col cuts encode unconditional tightenings (root-level
        // fixings).  We translate each fixing into a "self-arc" from the
        // forced literal's partner to the forced literal, which the
        // downstream SCC pipeline picks up as a forced literal.
        for (int i = 0; i < cs.sizeColCuts(); ++i) {
            const OsiColCut& cc = cs.colCut(i);
            const CoinPackedVector& lb = cc.lbs();
            const CoinPackedVector& ub = cc.ubs();
            for (int k = 0; k < lb.getNumElements(); ++k) {
                const int col = lb.getIndices()[k];
                const double v = lb.getElements()[k];
                if (col < 0 || col >= ncols || col_to_our_idx[col] == kInvalid) continue;
                if (v > 0.5) {
                    // forced to 1: emit (lit(col, 0) -> lit(col, 1))
                    arcs.emplace_back((Idx)(2 * col_to_our_idx[col] + 0),
                                      (Idx)(2 * col_to_our_idx[col] + 1));
                }
            }
            for (int k = 0; k < ub.getNumElements(); ++k) {
                const int col = ub.getIndices()[k];
                const double v = ub.getElements()[k];
                if (col < 0 || col >= ncols || col_to_our_idx[col] == kInvalid) continue;
                if (v < 0.5) {
                    arcs.emplace_back((Idx)(2 * col_to_our_idx[col] + 1),
                                      (Idx)(2 * col_to_our_idx[col] + 0));
                }
            }
        }
        if (opts.verbose) {
            std::printf("[cbc_extractor]   CglProbing pre-pass: %d col cuts -> %zu arcs (%.2fs)\n",
                        cs.sizeColCuts(), arcs.size(), seconds_since(t2));
        }
    }

    // --- 5. Initial LP solve. ---
    if (opts.per_lp_time_limit_s > 0.0) {
        solver.setIntParam(OsiMaxNumIteration, 10000);
    }
    {
        const std::clock_t t3 = std::clock();
        solver.initialSolve();
        if (opts.verbose) {
            std::printf("[cbc_extractor]   initial LP: %.2fs, %s\n",
                        seconds_since(t3),
                        solver.isProvenOptimal() ? "optimal" :
                          solver.isProvenPrimalInfeasible() ? "infeasible" : "unknown");
        }
        if (!solver.isProvenOptimal()) return arcs;
    }

    // --- 6. Manual probing loop. ---
    std::vector<double> orig_lb(solver.getColLower(),
                                solver.getColLower() + ncols);
    std::vector<double> orig_ub(solver.getColUpper(),
                                solver.getColUpper() + ncols);

    const std::clock_t t4 = std::clock();
    int n_probes = 0;
    int n_self_infeasible = 0;
    int n_pinned_arcs = 0;

    for (int src_col = 0; src_col < ncols; ++src_col) {
        if (col_to_our_idx[(std::size_t)src_col] == kInvalid) continue;
        if (!solver.isBinary(src_col)) continue;
        if (orig_lb[(std::size_t)src_col] > 0.5 ||
            orig_ub[(std::size_t)src_col] < 0.5) continue;          // already fixed

        for (int src_val = 0; src_val < 2; ++src_val) {
            if (opts.max_lp_resolves > 0 && n_probes >= opts.max_lp_resolves) {
                if (opts.verbose) {
                    std::printf("[cbc_extractor]   probe cap hit at %d resolves\n",
                                n_probes);
                }
                goto done;
            }
            ++n_probes;

            solver.setColBounds(src_col, (double)src_val, (double)src_val);
            solver.resolve();

            const Idx src_lit =
                (Idx)(2 * col_to_our_idx[(std::size_t)src_col] + src_val);

            if (solver.isProvenPrimalInfeasible()) {
                // src_lit is infeasible -> partner is forced.
                arcs.emplace_back(src_lit, (Idx)(src_lit ^ 1));
                ++n_self_infeasible;
            } else if (solver.isProvenOptimal()) {
                const double* lb = solver.getColLower();
                const double* ub = solver.getColUpper();
                for (int tgt_col = 0; tgt_col < ncols; ++tgt_col) {
                    if (tgt_col == src_col) continue;
                    if (col_to_our_idx[(std::size_t)tgt_col] == kInvalid) continue;
                    if (!solver.isBinary(tgt_col)) continue;
                    if (orig_lb[(std::size_t)tgt_col] > 0.5 ||
                        orig_ub[(std::size_t)tgt_col] < 0.5) continue;
                    if (lb[tgt_col] > 0.5) {
                        const Idx tgt_lit =
                            (Idx)(2 * col_to_our_idx[(std::size_t)tgt_col] + 1);
                        arcs.emplace_back(src_lit, tgt_lit);
                        ++n_pinned_arcs;
                    } else if (ub[tgt_col] < 0.5) {
                        const Idx tgt_lit =
                            (Idx)(2 * col_to_our_idx[(std::size_t)tgt_col] + 0);
                        arcs.emplace_back(src_lit, tgt_lit);
                        ++n_pinned_arcs;
                    }
                }
            }
            // Restore bounds and continue.
            solver.setColBounds(src_col, orig_lb[(std::size_t)src_col],
                                          orig_ub[(std::size_t)src_col]);
        }
    }
done:
    if (opts.verbose) {
        std::printf("[cbc_extractor]   manual probing: %d resolves, "
                    "%d self-infeasible, %d pinned-pair arcs (%.2fs)\n",
                    n_probes, n_self_infeasible, n_pinned_arcs,
                    seconds_since(t4));
        std::printf("[cbc_extractor]   total: %zu arcs in %.2fs\n",
                    arcs.size(), seconds_since(t0));
    }
    return arcs;
}

}  // namespace presolve

// =============================================================================
//  cbc_extractor.h
//
//  CBC-based implication-graph extractor.  This is the C++ analog of
//  python-mip's `mip.ConflictGraph(model)`: take a MIP, run CBC's probing
//  pass, and return the discovered binary-binary implications as a list of
//  arcs in our literal-index format.
//
//  Algorithm (manual probing through OsiClpSolverInterface, matching what
//  python-mip does internally on CBC 2.10.x):
//
//      for each binary var v with bounds {0, 1}:
//          for each src_val in {0, 1}:
//              save bounds; tighten v to {src_val, src_val}
//              solver.resolve()                        // warm-started LP
//              if (LP infeasible)
//                  emit arc (lit(v, src_val) -> lit(v, 1 - src_val))
//              else
//                  for each OTHER binary w:
//                      check tightened bounds; emit arc if pinned
//              restore bounds
//
//  Optionally we also do a CglProbing pre-pass to catch unconditional
//  tightenings cheaply.
//
//  This is much richer than scanning SCIP's constraint handlers individually
//  (which only sees the implications already encoded in setppc/varbound/...
//  constraints) because the LP propagation chains together inferences
//  across constraints.
// =============================================================================

#ifndef PRESOLVE_CBC_EXTRACTOR_H_
#define PRESOLVE_CBC_EXTRACTOR_H_

#include "graph_utils.h"
#include "scip/scip.h"

#include <unordered_map>
#include <utility>
#include <vector>

namespace presolve {

struct CbcExtractOptions {
    // Hard cap on the number of LP resolves to perform.  0 = unlimited.
    // Each binary literal is one resolve, so 2 * num_binaries is the full
    // probing pass.  Set a cap when num_binaries is large.
    int  max_lp_resolves    = 0;

    // CBC log level (0 = silent, 1 = summary, 3 = noisy).
    int  cbc_log_level      = 0;

    // Per-LP-solve time limit in seconds (0 = no limit).  Manual probing
    // does up to 2 * num_binaries LP resolves; capping each LP at 1s keeps
    // the worst case bounded.
    double per_lp_time_limit_s = 1.0;

    // If true, also run CglProbing once before the manual loop, to absorb
    // its unconditional tightenings into our arc set.
    bool run_cgl_prepass   = true;

    // If true, dump per-stage timings on stdout.
    bool verbose           = false;
};

// Extract binary-binary implications from SCIP's currently-transformed
// problem via CBC's probing.  The (active) binary variables in SCIP must
// already have been put into `var_to_index` by the caller; the returned
// arc list uses our literal-index format (2 * var_to_index[v]    for v=0,
// 2 * var_to_index[v] + 1 for v=1).
//
// Self-arcs (where probing concludes the source literal is infeasible) are
// encoded as  (src_lit -> src_lit ^ 1)  -- i.e. an arc from a literal to
// its partner -- which the downstream SCC pipeline interprets correctly:
// scc(src_lit) == scc(part(src_lit)) triggers infeasibility, while
// part(src_lit) reaching src_lit yields a forced-literal fixing.
//
// Returns an empty vector on any internal failure (could not write the SCIP
// problem to a temp MPS, CBC could not read it, etc.); callers should treat
// an empty return as "use the fallback extractor".
std::vector<std::pair<Idx, Idx>>
ExtractImplicationArcsViaCBC(
    SCIP*                                       scip,
    const std::unordered_map<SCIP_VAR*, Idx>&   var_to_index,
    const CbcExtractOptions&                    opts = {});

}  // namespace presolve

#endif  // PRESOLVE_CBC_EXTRACTOR_H_

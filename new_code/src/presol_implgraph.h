// =============================================================================
//  presol_implgraph.h
//
//  SCIP presolver plugin "implgraph": runs the implication-graph elimination
//  and fixing pipeline of Section 5 of Barbosa and Validi (IJOC, under
//  revision) directly inside SCIP's presolving loop.
//
//  Hooked plugins return reductions to SCIP via:
//      SCIPfixVar         -- for F_0 and F_1
//      SCIPaggregateVars  -- for direct (x_i = x_j) and indirect (x_i = 1 - x_j)
//                            eliminations
//      SCIPdelCons        -- for redundant set-packing constraints reachable
//                            through the digraph (we do NOT do this unless the
//                            user explicitly enables the experimental knob)
//
//  Usage from cmain.cpp:
//      SCIP_CALL( SCIPincludePresolImplgraph(scip) );
//
//  The plugin is conservative: it only operates on binary variables and only
//  uses SCIP's already-built implication / clique tables (no extra probing).
//  This keeps the plugin compatible with the rest of SCIP's presolve and
//  fixes the AE concern that our reductions should run "on top of" the
//  solver's own presolve rather than as a black-box replacement.
// =============================================================================

#ifndef PRESOLVE_PRESOL_IMPLGRAPH_H_
#define PRESOLVE_PRESOL_IMPLGRAPH_H_

#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

// Plugin parameters (registered under "presolving/implgraph/...")
//
//   ENABLED              bool   default TRUE  -- run at all?
//   MAX_LITERALS         int    default 0     -- skip if > this many literals
//                                              (0 = no cap; the published
//                                              experiments use 0 = uncapped)
//   USE_BITSET_REACH     int    default -1    -- -1 = auto threshold of 65k
//                                              SCCs; 0 = always BFS; 1 = always
//                                              bitset DP
//   APPLY_FIXINGS        bool   default TRUE
//   APPLY_AGGREGATIONS   bool   default TRUE
//   VERBOSE              int    default 1     -- 0 = silent, 1 = summary,
//                                              2 = per-phase, 3 = per-reduction
//   EXTRACTOR            int    default 0     -- 0 = SCIP constraint handlers
//                                              (fast, narrow); 1 = CBC manual
//                                              LP probing (currently broken --
//                                              do not use); 2 = both
//   ARCSFILE             string default ""    -- path to an external arcs
//                                              file produced by
//                                              scripts/build_conflict_graph.py
//                                              (Python + python-mip wrapping
//                                              CBC's CoinConflictGraph).
//                                              When non-empty, the loaded
//                                              arcs are UNIONED with the
//                                              constraint-handler arcs.

SCIP_RETCODE SCIPincludePresolImplgraph(SCIP* scip);

#ifdef __cplusplus
}
#endif

#endif  // PRESOLVE_PRESOL_IMPLGRAPH_H_

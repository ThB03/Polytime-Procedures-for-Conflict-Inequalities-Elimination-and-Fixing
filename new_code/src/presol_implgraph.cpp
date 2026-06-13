// =============================================================================
//  presol_implgraph.cpp
//
//  SCIP presolver plugin implementation.  See presol_implgraph.h for the
//  public interface and design notes.
//
//  Plug-in lifecycle:
//      PRESOLINIT  -- nothing (everything is allocated in EXEC)
//      PRESOLEXEC  -- the main routine
//      PRESOLEXIT  -- nothing
//
//  PRESOLEXEC stages:
//      1. Extract SCIP's implication and clique tables into a CSR digraph D.
//      2. Run Tarjan SCC on D; bail out infeasible if scc(v) == scc(v ^ 1).
//      3. Build the SCC condensation H = (C, F) and a topological order.
//      4. Direct + indirect elimination via parity-DSU over the SCC partition.
//      5. Forced-literal fixing via reachability on H:  v is forced iff
//         scc(v ^ 1) reaches scc(v) in H.
//      6. Apply: SCIPaggregateVars for DE/IE, SCIPfixVar for F_0/F_1.
//
//  We never call SCIPaggregateVars on a pair that contains a variable that
//  has already been fixed in this round.  Similarly, we recheck SCIPvarIsActive
//  before every application -- earlier presolvers in the same round may have
//  fixed or aggregated the variable already, in which case we skip.
// =============================================================================

#include "presol_implgraph.h"
#include "graph_utils.h"
#include "cbc_extractor.h"
#ifdef IMPLGRAPH_HAS_CUDA
#include "graph_utils_gpu.h"
#endif

#include "scip/scipdefplugins.h"
#include "scip/cons_setppc.h"
#include "scip/cons_logicor.h"
#include "scip/cons_varbound.h"
// SCIP_BOUNDTYPE comes in via scip/type_lp.h, transitively included by scip/scip.h.

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#define PRESOL_NAME            "implgraph"
#define PRESOL_DESC            "Implication-graph SCC + reachability presolver "  \
                               "(Barbosa & Validi, IJOC, under revision)"
// SCIP applies presolvers in order of priority (larger first).  Path-B
// placement: we run BEFORE PaPILO's MILP presolver (priority 9999999), so
// PaPILO can't subsume our DE/IE/F0/F1 before we see them.  The trade-off
// is that we now run before probing has populated SCIP's implication table,
// so we have to read directly from clique-bearing constraint handlers
// (cons_setppc / cons_logicor) in addition to SCIPgetCliques().
#define PRESOL_PRIORITY        (10000001)
#define PRESOL_MAXROUNDS       (1)         // fire AT MOST once per SCIP solve.
                                           // The plug-in's extract + SCC + reach
                                           // is paid in full on every fire; in
                                           // practice we never observe second-round
                                           // reductions that the first round
                                           // missed (PaPILO has fully converged
                                           // by then), so capping to 1 round
                                           // turns the per-solve overhead from
                                           // 3-5x into 1x with no loss of yield.
#define PRESOL_TIMING          SCIP_PRESOLTIMING_EXHAUSTIVE

namespace {  // implementation detail

using presolve::Idx;
using presolve::kInvalid;

// Per-variable book-keeping built once per PRESOLEXEC call.
//
//   active_binary_vars[i]  -- the SCIP_VAR* of binary variable i in our
//                             dense indexing.  Only "active" binary vars
//                             (i.e. SCIPvarIsActive and not yet fixed) make
//                             it into this array.
//
// Literal indexing convention (see graph_utils.h):
//   literal 2*i     <-> "x_i = 0"
//   literal 2*i + 1 <-> "x_i = 1"
struct LiteralMap {
    std::vector<SCIP_VAR*>                 active_binary_vars;
    std::unordered_map<SCIP_VAR*, Idx>     var_to_index;   // SCIP_VAR* -> i

    Idx NumLiterals() const noexcept {
        return (Idx)(2 * active_binary_vars.size());
    }
};

LiteralMap BuildLiteralMap(SCIP* scip) {
    LiteralMap M;
    const int nbinvars = SCIPgetNBinVars(scip);
    SCIP_VAR** const binvars = SCIPgetVars(scip);  // first nbinvars are binary
    M.active_binary_vars.reserve((std::size_t)nbinvars);
    M.var_to_index.reserve((std::size_t)nbinvars * 2);

    for (int i = 0; i < nbinvars; ++i) {
        SCIP_VAR* v = binvars[i];
        if (!SCIPvarIsActive(v)) continue;
        if (SCIPvarGetType(v) != SCIP_VARTYPE_BINARY) continue;
        // Skip already-fixed binaries.
        if (SCIPvarGetLbGlobal(v) > 0.5 || SCIPvarGetUbGlobal(v) < 0.5) continue;

        M.var_to_index[v] = (Idx)M.active_binary_vars.size();
        M.active_binary_vars.push_back(v);
    }
    return M;
}

// Emit two implication arcs for a single conflict (x_u = a) /\ (x_v = b) -- i.e.
// the assignment is infeasible:
//     (x_u = a)        -> (x_v = 1 - b)         (negation of v if u takes a)
//     (x_v = b)        -> (x_u = 1 - a)         (negation of u if v takes b)
// In literal-index space, with l(i, a) = 2*i + a:
//     emit (l(u, a),      l(v, 1 - b))
//     emit (l(v, b),      l(u, 1 - a))
inline void EmitConflict(std::vector<std::pair<Idx, Idx>>& arcs,
                         Idx u_var, std::uint8_t u_val,
                         Idx v_var, std::uint8_t v_val) {
    const Idx lu_a   = 2 * u_var + u_val;
    const Idx lv_b   = 2 * v_var + v_val;
    const Idx lv_nb  = 2 * v_var + (Idx)(1 - v_val);
    const Idx lu_na  = 2 * u_var + (Idx)(1 - u_val);
    if (lu_a != lv_nb) arcs.emplace_back(lu_a, lv_nb);
    if (lv_b != lu_na) arcs.emplace_back(lv_b, lu_na);
}

// Extract all binary-binary implications + clique conflicts into an arc list.
//
// We pull from three SCIP sources:
//
//   (a) SCIPvarGetImplics / SCIPvarGetNImpls : direct binary->binary implications
//   (b) SCIPvarGetCliques  / SCIPgetCliques  : at-most-one cliques on
//                                              binary literals (= a set of
//                                              pairwise conflicts)
//   (c) setpartitioning / setpacking constraints (from cons_setppc): each
//       contributes a clique on the literal {x = 1} for every column.
//
// We DELIBERATELY do not extract from generic linear constraints; SCIP's own
// probing presolver already lifts those into (a) and (b), and re-walking the
// linear matrix would slow us down without producing new implications on
// MIPLIB instances.
std::vector<std::pair<Idx, Idx>> ExtractImplicationArcs(SCIP*             scip,
                                                       const LiteralMap& M,
                                                       int               verbose = 0) {
    std::vector<std::pair<Idx, Idx>> arcs;

    // Per-source timing (verbose >= 2) so we can localize bottlenecks like
    // the 830s SCIPgetCliques / lazy-table population we hit on acc-tight5.
    const SCIP_Real t_start_a = SCIPgetSolvingTime(scip);
    std::size_t n_before_a = arcs.size();

    // ---- (a) Direct binary implications. ---------------------------------
    // SCIP stores per-variable lower-bound and upper-bound implications.
    // For a binary variable v, only two "modes" are interesting:
    //   v = 1  (== lower bound of v changed to 1)  =>  implies w >= / <= ...
    //   v = 0  (== upper bound of v changed to 0)  =>  implies ...
    for (Idx ui = 0; ui < (Idx)M.active_binary_vars.size(); ++ui) {
        SCIP_VAR* u = M.active_binary_vars[(std::size_t)ui];
        for (int u_val = 0; u_val < 2; ++u_val) {
            SCIP_Bool         varfixing = (u_val == 1) ? TRUE : FALSE;
            int               nimpls    = SCIPvarGetNImpls(u, varfixing);
            SCIP_VAR**        impl_vars = SCIPvarGetImplVars(u, varfixing);
            SCIP_BOUNDTYPE*   impl_btys = SCIPvarGetImplTypes(u, varfixing);
            SCIP_Real*        impl_bnds = SCIPvarGetImplBounds(u, varfixing);

            for (int k = 0; k < nimpls; ++k) {
                SCIP_VAR* w = impl_vars[k];
                if (SCIPvarGetType(w) != SCIP_VARTYPE_BINARY) continue;
                auto it = M.var_to_index.find(w);
                if (it == M.var_to_index.end()) continue;
                const Idx wi = it->second;

                // u = u_val implies w is bounded:
                //   bound type SCIP_BOUNDTYPE_LOWER, bound 1.0  => w = 1
                //   bound type SCIP_BOUNDTYPE_UPPER, bound 0.0  => w = 0
                int w_val;
                if (impl_btys[k] == SCIP_BOUNDTYPE_LOWER && impl_bnds[k] >= 0.5) {
                    w_val = 1;
                } else if (impl_btys[k] == SCIP_BOUNDTYPE_UPPER && impl_bnds[k] <= 0.5) {
                    w_val = 0;
                } else {
                    continue;  // fractional implication; not a binary-binary edge
                }
                // u = u_val -> w = w_val.  In literal space:
                arcs.emplace_back((Idx)(2 * ui + u_val), (Idx)(2 * wi + w_val));
                // Contrapositive: w = 1 - w_val -> u = 1 - u_val.
                arcs.emplace_back((Idx)(2 * wi + (1 - w_val)),
                                  (Idx)(2 * ui + (1 - u_val)));
            }
        }
    }

    const SCIP_Real t_end_a   = SCIPgetSolvingTime(scip);
    const std::size_t n_after_a = arcs.size();
    if (verbose >= 2) {
        std::fprintf(stdout, "[implgraph]   src(a) impls: %zu arcs in %.2fs\n",
                     n_after_a - n_before_a, t_end_a - t_start_a);
        std::fflush(stdout);
    }

    // ---- (b) Cliques. ----------------------------------------------------
    // A clique is a set of literals (possibly negated) that are pairwise in
    // conflict (at most one can be true).  SCIP exposes both an array of all
    // cliques and per-variable references; we walk the global list.
    SCIP_CLIQUE** const cliques  = SCIPgetCliques(scip);
    const int           ncliques = SCIPgetNCliques(scip);
    for (int c = 0; c < ncliques; ++c) {
        SCIP_CLIQUE* clq    = cliques[c];
        const int    nvars  = SCIPcliqueGetNVars(clq);
        SCIP_VAR**   cvars  = SCIPcliqueGetVars(clq);
        SCIP_Bool*   cvals  = SCIPcliqueGetValues(clq);  // TRUE if positive literal

        // Map clique members to (lit_var_idx, lit_val) pairs we know about.
        struct LitRef { Idx var; std::uint8_t val; };
        std::vector<LitRef> members;
        members.reserve((std::size_t)nvars);
        for (int j = 0; j < nvars; ++j) {
            auto it = M.var_to_index.find(cvars[j]);
            if (it == M.var_to_index.end()) continue;
            members.push_back({ it->second, (std::uint8_t)(cvals[j] ? 1 : 0) });
        }
        // Quadratic on clique size; cliques are typically small (median ~3 on
        // MIPLIB) so this is fine.  For pathological 1k-literal cliques the
        // O(clique^2) emit is still cheap compared to the SCC pass that follows.
        for (std::size_t a = 0; a < members.size(); ++a) {
            for (std::size_t b = a + 1; b < members.size(); ++b) {
                EmitConflict(arcs,
                             members[a].var, members[a].val,
                             members[b].var, members[b].val);
            }
        }
    }
    const SCIP_Real t_end_b   = SCIPgetSolvingTime(scip);
    const std::size_t n_after_b = arcs.size();
    if (verbose >= 2) {
        std::fprintf(stdout, "[implgraph]   src(b) cliques: %zu arcs in %.2fs (ncliques=%d)\n",
                     n_after_b - n_after_a, t_end_b - t_end_a, ncliques);
        std::fflush(stdout);
    }

    // ---- (c) Direct constraint scan -- needed because SCIP's clique table
    //     is populated by other presolvers' callbacks, which haven't run
    //     yet at our (high) priority placement.  We walk the set-packing /
    //     set-partitioning constraint handler and the logicor constraint
    //     handler directly and emit pairwise conflict arcs.
    // ----------------------------------------------------------------------
    SCIP_CONSHDLR* setppc_hdlr = SCIPfindConshdlr(scip, "setppc");
    if (setppc_hdlr != NULL) {
        const int nconss = SCIPconshdlrGetNConss(setppc_hdlr);
        SCIP_CONS** const conss = SCIPconshdlrGetConss(setppc_hdlr);
        for (int c = 0; c < nconss; ++c) {
            SCIP_CONS* cons = conss[c];
            // Only PARTITIONING (sum == 1) and PACKING (sum <= 1) produce
            // pairwise (x_i = 1, x_j = 1) conflicts; COVERING (sum >= 1)
            // does not.
            const SCIP_SETPPCTYPE t = SCIPgetTypeSetppc(scip, cons);
            if (t != SCIP_SETPPCTYPE_PARTITIONING &&
                t != SCIP_SETPPCTYPE_PACKING) continue;
            const int    nvars = SCIPgetNVarsSetppc(scip, cons);
            SCIP_VAR**   vars  = SCIPgetVarsSetppc(scip, cons);
            struct LR { Idx var; std::uint8_t val; };
            std::vector<LR> members;
            members.reserve((std::size_t)nvars);
            for (int j = 0; j < nvars; ++j) {
                auto it = M.var_to_index.find(vars[j]);
                if (it == M.var_to_index.end()) continue;
                members.push_back({ it->second, 1 });  // positive literal
            }
            for (std::size_t a = 0; a < members.size(); ++a) {
                for (std::size_t b = a + 1; b < members.size(); ++b) {
                    EmitConflict(arcs,
                                 members[a].var, members[a].val,
                                 members[b].var, members[b].val);
                }
            }
        }
    }

    const SCIP_Real t_end_c   = SCIPgetSolvingTime(scip);
    const std::size_t n_after_c = arcs.size();
    if (verbose >= 2) {
        std::fprintf(stdout, "[implgraph]   src(c) setppc: %zu arcs in %.2fs\n",
                     n_after_c - n_after_b, t_end_c - t_end_b);
        std::fflush(stdout);
    }

    // ---- (d) Binary logicor (sum_{i in L} l_i >= 1) for |L| == 2 ----
    //     A two-literal logicor over l1 and l2 means l1 OR l2 = true; the
    //     contrapositive is: !l1 implies l2, and !l2 implies l1.  These
    //     can produce NON-bipartite arcs (e.g. (x_i = 0) -> (x_j = 1)),
    //     which is exactly what our SCC pipeline needs to find cycles.
    SCIP_CONSHDLR* logicor_hdlr = SCIPfindConshdlr(scip, "logicor");
    if (logicor_hdlr != NULL) {
        const int nconss = SCIPconshdlrGetNConss(logicor_hdlr);
        SCIP_CONS** const conss = SCIPconshdlrGetConss(logicor_hdlr);
        for (int c = 0; c < nconss; ++c) {
            SCIP_CONS* cons = conss[c];
            const int    nvars = SCIPgetNVarsLogicor(scip, cons);
            if (nvars != 2) continue;          // binary clauses only
            SCIP_VAR** vars = SCIPgetVarsLogicor(scip, cons);
            // Each var in a logicor is a literal: positive if it appears
            // directly, negative if it is a negated variable.  pyscipopt
            // and the SCIP C API expose negation via SCIPvarIsNegated /
            // SCIPvarGetNegatedVar.  We resolve to (active_var, polarity).
            auto resolve = [&](SCIP_VAR* v) -> std::pair<Idx, std::uint8_t> {
                std::uint8_t pol = 1;
                if (SCIPvarIsNegated(v)) {
                    pol = 0;
                    v   = SCIPvarGetNegatedVar(v);
                }
                auto it = M.var_to_index.find(v);
                if (it == M.var_to_index.end()) return { kInvalid, 0 };
                return { it->second, pol };
            };
            auto [ui, upol] = resolve(vars[0]);
            auto [vi, vpol] = resolve(vars[1]);
            if (ui == kInvalid || vi == kInvalid) continue;
            // Constraint: lit(u, upol) OR lit(v, vpol).  Negation -> impl:
            //   lit(u, 1 - upol)  ->  lit(v, vpol)
            //   lit(v, 1 - vpol)  ->  lit(u, upol)
            arcs.emplace_back((Idx)(2 * ui + (1 - upol)),
                              (Idx)(2 * vi + vpol));
            arcs.emplace_back((Idx)(2 * vi + (1 - vpol)),
                              (Idx)(2 * ui + upol));
        }
    }

    const SCIP_Real t_end_d   = SCIPgetSolvingTime(scip);
    const std::size_t n_after_d = arcs.size();
    if (verbose >= 2) {
        std::fprintf(stdout, "[implgraph]   src(d) logicor: %zu arcs in %.2fs\n",
                     n_after_d - n_after_c, t_end_d - t_end_c);
        std::fflush(stdout);
    }

    // ---- (e) Variable-bound constraints  lhs <= x + c*y <= rhs  with x,y
    //     both binary.  Each side of the inequality fixes a specific
    //     (x-literal, y-literal) combination as infeasible, which is exactly
    //     a binary conflict.  These constraints often produce NON-bipartite
    //     arcs (a 0-literal of x implying a 1-literal of y, etc.).
    SCIP_CONSHDLR* vb_hdlr = SCIPfindConshdlr(scip, "varbound");
    if (vb_hdlr != NULL) {
        const int nconss = SCIPconshdlrGetNConss(vb_hdlr);
        SCIP_CONS** const conss = SCIPconshdlrGetConss(vb_hdlr);
        for (int c = 0; c < nconss; ++c) {
            SCIP_CONS* cons = conss[c];
            SCIP_VAR*  x    = SCIPgetVarVarbound  (scip, cons);
            SCIP_VAR*  y    = SCIPgetVbdvarVarbound(scip, cons);
            const SCIP_Real cy  = SCIPgetVbdcoefVarbound(scip, cons);
            const SCIP_Real lhs = SCIPgetLhsVarbound(scip, cons);
            const SCIP_Real rhs = SCIPgetRhsVarbound(scip, cons);
            if (SCIPvarGetType(x) != SCIP_VARTYPE_BINARY ||
                SCIPvarGetType(y) != SCIP_VARTYPE_BINARY) continue;
            auto itx = M.var_to_index.find(x);
            auto ity = M.var_to_index.find(y);
            if (itx == M.var_to_index.end() || ity == M.var_to_index.end())
                continue;
            const Idx xi = itx->second;
            const Idx yi = ity->second;
            // For each combination (a, b) in {0, 1}^2, check feasibility:
            //   value = a + cy * b
            //   infeasible iff value < lhs - 1e-9 OR value > rhs + 1e-9
            for (int a = 0; a < 2; ++a) {
                for (int b = 0; b < 2; ++b) {
                    const double value = (double)a + cy * (double)b;
                    const bool infeasible =
                        (value < lhs - 1e-9) || (value > rhs + 1e-9);
                    if (!infeasible) continue;
                    // (x = a) /\ (y = b) is infeasible:
                    //   emit (x = a) -> (y = 1 - b)
                    //   emit (y = b) -> (x = 1 - a)
                    EmitConflict(arcs, xi, (std::uint8_t)a, yi, (std::uint8_t)b);
                }
            }
        }
    }

    const SCIP_Real t_end_e   = SCIPgetSolvingTime(scip);
    const std::size_t n_after_e = arcs.size();
    if (verbose >= 2) {
        std::fprintf(stdout, "[implgraph]   src(e) varbound: %zu arcs in %.2fs\n",
                     n_after_e - n_after_d, t_end_e - t_end_d);
        std::fflush(stdout);
    }
    return arcs;
}

// Apply one direct elimination (x_i = x_j) via SCIPaggregateVars.
//
// SCIP's contract for SCIPaggregateVars on binaries:
//      SCIPaggregateVars(scip, varx, vary, scalarx, scalary, rhs, ...)
//   enforces:   scalarx * varx + scalary * vary = rhs.
//
// For x_i = x_j we use:   1*x_i + (-1)*x_j = 0.
// For x_i = 1 - x_j we use: 1*x_i + 1*x_j = 1.
SCIP_RETCODE TryAggregate(SCIP*       scip,
                          SCIP_VAR*   xi,
                          SCIP_VAR*   xj,
                          bool        indirect,  // false: xi = xj; true: xi = 1 - xj
                          int*        n_aggregated,
                          SCIP_Bool*  infeasible,
                          int         verbose) {
    if (xi == NULL || xj == NULL) return SCIP_OKAY;
    if (xi == xj) return SCIP_OKAY;
    if (!SCIPvarIsActive(xi) || !SCIPvarIsActive(xj)) return SCIP_OKAY;
    // SCIPaggregateVars requires both vars to be in LOOSE or COLUMN status;
    // calling it on a variable that is already AGGREGATED, MULTAGGR, NEGATED,
    // FIXED, or ORIGINAL segfaults (the SCIPvarIsActive check is not enough
    // because it returns TRUE for some statuses that aggregation rejects).
    const SCIP_VARSTATUS sti = SCIPvarGetStatus(xi);
    const SCIP_VARSTATUS stj = SCIPvarGetStatus(xj);
    if (sti != SCIP_VARSTATUS_LOOSE && sti != SCIP_VARSTATUS_COLUMN) return SCIP_OKAY;
    if (stj != SCIP_VARSTATUS_LOOSE && stj != SCIP_VARSTATUS_COLUMN) return SCIP_OKAY;
    if (SCIPvarGetLbGlobal(xi) > 0.5 || SCIPvarGetUbGlobal(xi) < 0.5) return SCIP_OKAY;
    if (SCIPvarGetLbGlobal(xj) > 0.5 || SCIPvarGetUbGlobal(xj) < 0.5) return SCIP_OKAY;
    // After resolving any internal aliasing, ensure the two probvars differ.
    // SCIP's SCIPvarGetProbvar may return the same active variable for both
    // xi and xj if one is implicitly equivalent to the other.
    SCIP_VAR* px = SCIPvarGetProbvar(xi);
    SCIP_VAR* py = SCIPvarGetProbvar(xj);
    if (px == py) return SCIP_OKAY;
    // Defensive: also reject MULTAGGR/AGGREGATED probvars (chains).
    if (px == NULL || py == NULL) return SCIP_OKAY;
    if (SCIPvarGetStatus(px) != SCIP_VARSTATUS_LOOSE &&
        SCIPvarGetStatus(px) != SCIP_VARSTATUS_COLUMN) return SCIP_OKAY;
    if (SCIPvarGetStatus(py) != SCIP_VARSTATUS_LOOSE &&
        SCIPvarGetStatus(py) != SCIP_VARSTATUS_COLUMN) return SCIP_OKAY;

    SCIP_Bool aggregated = FALSE;
    if (!indirect) {
        SCIP_CALL( SCIPaggregateVars(scip, xi, xj,  1.0, -1.0, 0.0,
                                     infeasible, /*redundant=*/NULL, &aggregated) );
    } else {
        SCIP_CALL( SCIPaggregateVars(scip, xi, xj,  1.0,  1.0, 1.0,
                                     infeasible, /*redundant=*/NULL, &aggregated) );
    }
    if (aggregated) {
        ++(*n_aggregated);
        if (verbose >= 3) {
            SCIPinfoMessage(scip, NULL, "[implgraph] aggregate %s = %s%s\n",
                            SCIPvarGetName(xi),
                            indirect ? "1 - " : "",
                            SCIPvarGetName(xj));
        }
    }
    return SCIP_OKAY;
}

// Load implications from an external arcs file.  The file format is the same
// "<bit><varname> <bit><varname>" per line accepted by standalone_runner --
// each token starts with '0' or '1' (the literal's value) followed by the
// SCIP variable name as it appears in the source MPS.  Lines beginning with
// '#' and blank lines are skipped.  Tokens whose variable name does not map
// to any active binary in our LiteralMap are silently ignored (those vars
// were already fixed/aggregated by earlier presolvers).
//
// The reader is tolerant of SCIP's "t_" name prefix on transformed variables:
// it builds the lookup map with BOTH the bare name and the "t_"-prefixed form
// (and the de-prefixed form, when SCIP names start with "t_").  This lets the
// same arcs file be reused across presolve rounds even if SCIP renames vars.
//
// Returns SCIP_OKAY on success.  Errors opening/parsing the file are logged
// at verbose >= 1 but never abort the presolver; on parse error we just stop
// reading and use whatever arcs we already loaded.
//
// On return, *out_n_loaded contains the number of arcs successfully appended
// to `arcs`, and *out_n_skipped the number of arcs dropped because at least
// one endpoint's variable was no longer in our literal map.
// Forward decl of the data struct so the loader can cache state across rounds.
struct PresolDataImplgraph;

SCIP_RETCODE LoadArcsFromFile(SCIP*                            scip,
                              const char*                      path,
                              const LiteralMap&                M,
                              std::vector<std::pair<Idx,Idx>>& arcs,
                              int                              verbose,
                              int*                             out_n_loaded,
                              int*                             out_n_skipped,
                              PresolDataImplgraph*             data = nullptr);

// SCIP callback wrappers ----------------------------------------------------

struct PresolDataImplgraph {
    SCIP_Bool enabled            = TRUE;
    int       max_literals       = 0;
    int       use_bitset_reach   = -1;
    SCIP_Bool apply_fixings      = TRUE;
    SCIP_Bool apply_aggregations = TRUE;
    int       verbose            = 1;
    // Extractor selection: 0 = constraint handlers (fast, narrow), 1 = CBC
    // CglProbing (richer in principle but currently disabled by default --
    // the manual-LP-probing path produces no implications on real MIPs;
    // implementing proper CglProbing-based extraction is future work).
    int       extractor          = 0;
    int       cbc_max_resolves   = 0;     // 0 = no cap
    int       cbc_log_level      = 0;
    // Reach-phase backend selection:
    //   -1 = auto: GPU if available and H.num_nodes > use_gpu_threshold;
    //              else bitset DP if H.num_nodes <= kBitsetReachThreshold;
    //              else per-source BFS.
    //    0 = force CPU (existing behaviour).
    //    1 = force GPU (errors out / falls back to CPU if CUDA unavailable).
    int       use_gpu            = -1;
    int       use_gpu_threshold  = 0;      // |C| above which auto-mode picks GPU;
                                           // 0 = always GPU when CUDA available
    // Path to an external arcs file produced by
    // scripts/build_conflict_graph.py (Python + python-mip wrapping CBC's
    // CoinConflictGraph).  When non-empty, the listed implications are
    // UNIONED with the constraint-handler arcs before SCC analysis -- so
    // the file strictly adds information and never replaces it.  An empty
    // string disables the file loader; the plug-in then runs exactly as
    // before, using only SCIP's own extractor.
    char*     arcs_file          = NULL;

    // Across-round cache for parsed arcs.  We hold the (string, bit, string,
    // bit) tuples as parsed from the file on the FIRST EXEC call, then on
    // every subsequent EXEC call we just rebuild the name->Idx map from the
    // current LiteralMap and emit arcs.  This avoids re-doing the disk read
    // and string parse on every presolve round (which mattered: on irp's
    // 1.9M-arc file the per-round reload was the dominant overhead in
    // on-top mode).
    struct CachedArc { std::string a_name; std::uint8_t a_val;
                       std::string b_name; std::uint8_t b_val; };
    std::vector<CachedArc> arcs_cache;
    std::string            arcs_cache_path;   // empty => cache not built
};

// Definition of LoadArcsFromFile.  Loads the arcs file from disk on the
// FIRST EXEC call (parsed into data->arcs_cache as string tuples) and on
// subsequent calls walks the cache instead, which saves re-reading the
// (potentially multi-million-line) file every presolve round.  The arc
// indices DO have to be recomputed every round because vars get fixed /
// aggregated between rounds, but the name -> Idx lookup is cheap.
SCIP_RETCODE LoadArcsFromFile(SCIP*                            scip,
                              const char*                      path,
                              const LiteralMap&                M,
                              std::vector<std::pair<Idx,Idx>>& arcs,
                              int                              verbose,
                              int*                             out_n_loaded,
                              int*                             out_n_skipped,
                              PresolDataImplgraph*             data) {
    if (out_n_loaded)  *out_n_loaded  = 0;
    if (out_n_skipped) *out_n_skipped = 0;
    if (path == NULL || path[0] == '\0') return SCIP_OKAY;

    // First call: parse the file into data->arcs_cache.  Subsequent calls:
    // skip the parse and walk the cache.
    if (data != nullptr && data->arcs_cache_path != std::string(path)) {
        // Path changed or first call -- rebuild cache.
        data->arcs_cache.clear();
        data->arcs_cache_path = path;

        FILE* fp = std::fopen(path, "r");
        if (fp == NULL) {
            if (verbose >= 1) {
                SCIPinfoMessage(scip, NULL,
                    "[implgraph] arcsfile: could not open '%s' (errno=%d)\n",
                    path, errno);
            }
            return SCIP_OKAY;
        }
        char line[4096];
        int n_parse_err = 0;
        while (std::fgets(line, sizeof(line), fp) != NULL) {
            char* hash = std::strchr(line, '#');
            if (hash) *hash = '\0';
            char* a = std::strtok(line, " \t\r\n");
            if (a == NULL) continue;
            char* b = std::strtok(NULL, " \t\r\n");
            if (b == NULL) { ++n_parse_err; continue; }
            if ((a[0] != '0' && a[0] != '1') || a[1] == '\0' ||
                (b[0] != '0' && b[0] != '1') || b[1] == '\0') {
                ++n_parse_err;
                continue;
            }
            PresolDataImplgraph::CachedArc e;
            e.a_val  = (std::uint8_t)(a[0] == '1' ? 1 : 0);
            e.a_name = std::string(a + 1);
            e.b_val  = (std::uint8_t)(b[0] == '1' ? 1 : 0);
            e.b_name = std::string(b + 1);
            data->arcs_cache.push_back(std::move(e));
        }
        std::fclose(fp);
        if (verbose >= 1) {
            SCIPinfoMessage(scip, NULL,
                "[implgraph] arcsfile '%s': parsed and cached %zu arcs, "
                "%d parse errors\n",
                path, data->arcs_cache.size(), n_parse_err);
        }
    }

    // Resolve names to Idx using the CURRENT LiteralMap (which changes round
    // to round as vars get fixed / aggregated).  Build a small name->Idx
    // hashmap tolerant of SCIP "t_" prefix in either direction.
    std::unordered_map<std::string, Idx> name_to_idx;
    name_to_idx.reserve(M.active_binary_vars.size() * 2);
    for (Idx i = 0; i < (Idx)M.active_binary_vars.size(); ++i) {
        const std::string n(SCIPvarGetName(M.active_binary_vars[(std::size_t)i]));
        name_to_idx[n] = i;
        if (n.size() >= 2 && n[0] == 't' && n[1] == '_') {
            name_to_idx[n.substr(2)] = i;
        } else {
            name_to_idx[std::string("t_") + n] = i;
        }
    }

    int n_loaded = 0, n_skipped = 0;
    // If we don't have a `data` pointer (legacy call path) the cache won't
    // be populated; fall back to parsing the file on every call.  That's
    // exactly the pre-cache behaviour.
    auto try_emit = [&](const std::string& a_name, std::uint8_t a_val,
                        const std::string& b_name, std::uint8_t b_val) {
        auto ita = name_to_idx.find(a_name);
        auto itb = name_to_idx.find(b_name);
        if (ita == name_to_idx.end() || itb == name_to_idx.end()) {
            ++n_skipped;
            return;
        }
        arcs.emplace_back((Idx)(2 * ita->second + a_val),
                          (Idx)(2 * itb->second + b_val));
        ++n_loaded;
    };

    if (data != nullptr && !data->arcs_cache.empty()) {
        arcs.reserve(arcs.size() + data->arcs_cache.size());
        for (const auto& e : data->arcs_cache) {
            try_emit(e.a_name, e.a_val, e.b_name, e.b_val);
        }
    } else {
        // No cache available (data == nullptr); read the file inline.
        FILE* fp = std::fopen(path, "r");
        if (fp == NULL) return SCIP_OKAY;
        char line[4096];
        while (std::fgets(line, sizeof(line), fp) != NULL) {
            char* hash = std::strchr(line, '#');
            if (hash) *hash = '\0';
            char* a = std::strtok(line, " \t\r\n");
            if (a == NULL) continue;
            char* b = std::strtok(NULL, " \t\r\n");
            if (b == NULL) continue;
            if ((a[0] != '0' && a[0] != '1') || a[1] == '\0' ||
                (b[0] != '0' && b[0] != '1') || b[1] == '\0') continue;
            try_emit(std::string(a + 1), (std::uint8_t)(a[0] == '1' ? 1 : 0),
                     std::string(b + 1), (std::uint8_t)(b[0] == '1' ? 1 : 0));
        }
        std::fclose(fp);
    }

    if (out_n_loaded)  *out_n_loaded  = n_loaded;
    if (out_n_skipped) *out_n_skipped = n_skipped;
    if (verbose >= 1) {
        SCIPinfoMessage(scip, NULL,
            "[implgraph] arcsfile '%s': %d arcs into pipeline "
            "(%d skipped: vars no longer active)\n",
            path, n_loaded, n_skipped);
    }
    return SCIP_OKAY;
}

SCIP_DECL_PRESOLFREE(presolFreeImplgraph) {
    auto* data = (PresolDataImplgraph*)SCIPpresolGetData(presol);
    delete data;
    SCIPpresolSetData(presol, NULL);
    return SCIP_OKAY;
}

SCIP_DECL_PRESOLEXEC(presolExecImplgraph) {
    auto* data = (PresolDataImplgraph*)SCIPpresolGetData(presol);
    *result    = SCIP_DIDNOTRUN;
    if (!data->enabled)                          return SCIP_OKAY;
    if (SCIPgetNBinVars(scip) <= 1)              return SCIP_OKAY;

    auto literal_map = BuildLiteralMap(scip);
    const Idx nlit = literal_map.NumLiterals();
    if (nlit < 4)                                return SCIP_OKAY;  // nothing to do
    if (data->max_literals > 0 && (int)nlit > data->max_literals) {
        if (data->verbose >= 1) {
            SCIPinfoMessage(scip, NULL,
                "[implgraph] skipping: %d literals exceeds cap %d\n",
                (int)nlit, data->max_literals);
        }
        return SCIP_OKAY;
    }
    *result = SCIP_DIDNOTFIND;

    // --- (1) Build CSR --------------------------------------------------
    const SCIP_Real t0 = SCIPgetSolvingTime(scip);
    std::vector<std::pair<Idx, Idx>> arcs;
    if (data->extractor == 0 || data->extractor == 2) {
        auto a1 = ExtractImplicationArcs(scip, literal_map, data->verbose);
        arcs.insert(arcs.end(),
                    std::make_move_iterator(a1.begin()),
                    std::make_move_iterator(a1.end()));
    }
    if (data->extractor == 1 || data->extractor == 2) {
#ifdef IMPLGRAPH_HAS_CBC
        presolve::CbcExtractOptions opts;
        opts.max_lp_resolves = data->cbc_max_resolves;
        opts.cbc_log_level   = data->cbc_log_level;
        opts.verbose         = (data->verbose >= 2);
        auto a2 = presolve::ExtractImplicationArcsViaCBC(scip,
                    literal_map.var_to_index, opts);
        arcs.insert(arcs.end(),
                    std::make_move_iterator(a2.begin()),
                    std::make_move_iterator(a2.end()));
#else
        if (data->verbose >= 1) {
            SCIPinfoMessage(scip, NULL,
                "[implgraph] CBC extractor requested but plug-in built without "
                "IMPLGRAPH_HAS_CBC; falling back to constraint-handler "
                "extractor only.\n");
        }
        if (data->extractor == 1) {
            // User explicitly asked for CBC-only -- run the fallback anyway
            // so we don't produce an empty arc set.
            auto a3 = ExtractImplicationArcs(scip, literal_map, data->verbose);
            arcs.insert(arcs.end(),
                        std::make_move_iterator(a3.begin()),
                        std::make_move_iterator(a3.end()));
        }
#endif
    }
    // --- External Python+CBC arcs file (union) --------------------------
    //
    // If the user pre-built a conflict graph via
    //     python3 scripts/build_conflict_graph.py instance.mps instance.arcs
    // and set `presolving/implgraph/arcsfile=instance.arcs`, union those arcs
    // into our list.  CBC's ConflictGraph (wrapped by python-mip) chains LP
    // propagation across general linear constraints and so produces strictly
    // more binary-binary implications than SCIP's per-constraint extractors,
    // which only see what setppc/varbound/logicor explicitly encode.
    if (data->arcs_file != NULL && data->arcs_file[0] != '\0') {
        int n_loaded = 0, n_skipped = 0;
        SCIP_CALL( LoadArcsFromFile(scip, data->arcs_file, literal_map,
                                    arcs, data->verbose,
                                    &n_loaded, &n_skipped, data) );
        (void)n_loaded;  (void)n_skipped;
    }
    const SCIP_Real t1 = SCIPgetSolvingTime(scip);
    if (arcs.empty()) {
        if (data->verbose >= 1) {
            SCIPinfoMessage(scip, NULL,
                "[implgraph] no implications extracted; nothing to do (%.2fs)\n",
                t1 - t0);
        }
        // We did do extraction work; signal "ran but found nothing" so the
        // overhead is correctly accounted in SCIP's presolving statistics.
        *result = SCIP_DIDNOTFIND;
        return SCIP_OKAY;
    }
    presolve::CSRDigraph D = presolve::BuildCSR(nlit, arcs);
    // Free the temporary arc list eagerly; the CSR copy is now the source of truth.
    std::vector<std::pair<Idx, Idx>>().swap(arcs);

    const SCIP_Real t2 = SCIPgetSolvingTime(scip);
    if (data->verbose >= 1) {
        SCIPinfoMessage(scip, NULL,
            "[implgraph] D = (n=%d literals, |A|=%d arcs) built in %.2fs\n",
            (int)D.n, (int)D.nArcs, t2 - t0);
    }

    // --- (2) Tarjan SCC -------------------------------------------------
    presolve::SCCResult scc = presolve::TarjanSCC(D);
    const SCIP_Real t3 = SCIPgetSolvingTime(scip);

    // Infeasibility check: scc(v) == scc(v ^ 1).
    if (presolve::DetectInfeasibility(scc, nlit)) {
        if (data->verbose >= 1) {
            SCIPinfoMessage(scip, NULL,
                "[implgraph] INFEASIBLE: literal and its partner share an SCC\n");
        }
        *result = SCIP_CUTOFF;
        return SCIP_OKAY;
    }

    // --- (3) Condensation DAG ------------------------------------------
    presolve::CondensationDAG H = presolve::BuildCondensation(D, scc);
    const SCIP_Real t4 = SCIPgetSolvingTime(scip);
    if (data->verbose >= 2) {
        SCIPinfoMessage(scip, NULL,
            "[implgraph] SCC: |C|=%d (%.2fs); condensation |F|=%d (%.2fs)\n",
            (int)scc.num_sccs, t3 - t2, (int)H.num_arcs, t4 - t3);
    }

    // --- (4) Parity-DSU sweep -> DE, IE --------------------------------
    //
    // For each SCC S, pick a representative literal rS, and for every other
    // v in S, union(var(v), var(rS), stat(v) XOR stat(rS)).  Then read
    // each variable's (root, parity) to enumerate DE and IE pairs.
    const Idx p = (Idx)literal_map.active_binary_vars.size();
    presolve::ParityDSU dsu(p);

    {
        std::vector<Idx> rep_of_scc((std::size_t)scc.num_sccs, kInvalid);
        for (Idx l = 0; l < nlit; ++l) {
            const Idx s = scc.scc_id[(std::size_t)l];
            if (rep_of_scc[(std::size_t)s] == kInvalid) {
                rep_of_scc[(std::size_t)s] = l;
                continue;
            }
            const Idx           r        = rep_of_scc[(std::size_t)s];
            const Idx           var_l    = l >> 1;
            const Idx           var_r    = r >> 1;
            const std::uint8_t  stat_l   = (std::uint8_t)(l  & 1);
            const std::uint8_t  stat_r   = (std::uint8_t)(r  & 1);
            // x_l = x_r XOR (stat_l XOR stat_r), but literal "x_i = a" means
            // the binary variable x_i takes value a; the parity-DSU works on
            // the variable, so we need x_var_l = x_var_r XOR rel where
            //   rel = 0  iff  (l is the a=stat_r literal of var_l) and stat_l == stat_r
            // which simplifies to:  rel = stat_l XOR stat_r.
            const std::uint8_t  rel      = (std::uint8_t)(stat_l ^ stat_r);
            if (var_l == var_r) continue;  // same binary -- already in same SCC
                                          // would have meant infeasibility,
                                          // which we tested above
            if (!dsu.Union(var_l, var_r, rel)) {
                if (data->verbose >= 1) {
                    SCIPinfoMessage(scip, NULL,
                        "[implgraph] INFEASIBLE: inconsistent parity in SCC\n");
                }
                *result = SCIP_CUTOFF;
                return SCIP_OKAY;
            }
        }
    }
    const SCIP_Real t5 = SCIPgetSolvingTime(scip);

    // Apply DE / IE.
    int n_aggregated_direct   = 0;
    int n_aggregated_indirect = 0;
    SCIP_Bool infeasible      = FALSE;
    if (data->apply_aggregations) {
        if (data->verbose >= 2) {
            std::fprintf(stdout, "[implgraph]   aggregate-apply: entering loop, p=%d\n", (int)p);
            std::fflush(stdout);
        }
        int loop_counter = 0;
        for (Idx i = 0; i < p; ++i) {
            auto [r, par] = dsu.FindWithParity(i);
            if (r == i) continue;
            SCIP_VAR* xi = literal_map.active_binary_vars[(std::size_t)i];
            SCIP_VAR* xr = literal_map.active_binary_vars[(std::size_t)r];
            const bool indirect = (par == 1);
            ++loop_counter;
            if (data->verbose >= 2 && (loop_counter <= 5 || loop_counter % 1000 == 0)) {
                std::fprintf(stdout,
                    "[implgraph]   agg #%d: i=%d (%s, st=%d, lb=%g, ub=%g) "
                    "r=%d (%s, st=%d, lb=%g, ub=%g) indirect=%d\n",
                    loop_counter, (int)i,
                    xi ? SCIPvarGetName(xi) : "NULL",
                    xi ? (int)SCIPvarGetStatus(xi) : -1,
                    xi ? SCIPvarGetLbGlobal(xi) : -1.0,
                    xi ? SCIPvarGetUbGlobal(xi) : -1.0,
                    (int)r,
                    xr ? SCIPvarGetName(xr) : "NULL",
                    xr ? (int)SCIPvarGetStatus(xr) : -1,
                    xr ? SCIPvarGetLbGlobal(xr) : -1.0,
                    xr ? SCIPvarGetUbGlobal(xr) : -1.0,
                    indirect ? 1 : 0);
                std::fflush(stdout);
            }
            if (!indirect) {
                SCIP_CALL( TryAggregate(scip, xi, xr, false,
                                        &n_aggregated_direct, &infeasible,
                                        data->verbose) );
            } else {
                SCIP_CALL( TryAggregate(scip, xi, xr, true,
                                        &n_aggregated_indirect, &infeasible,
                                        data->verbose) );
            }
            if (infeasible) { *result = SCIP_CUTOFF; return SCIP_OKAY; }
        }
    }
    const SCIP_Real t6 = SCIPgetSolvingTime(scip);

    // --- (5) Forced-literal reachability -------------------------------
    // For each variable i, check whether scc(2*i) reaches scc(2*i + 1) in H
    // (forcing x_i = 1) or vice versa (forcing x_i = 0).
    //
    // Backend selection (in priority order):
    //   (a) GPU (graph_utils_gpu.cu) -- used if use_gpu == 1, or if
    //       use_gpu == -1 (auto) and CUDA is available and
    //       H.num_nodes > use_gpu_threshold;
    //   (b) CPU bitset DP -- used if use_bitset_reach == 1, or if
    //       use_bitset_reach == -1 (auto) and H.num_nodes <= kBitsetReachThreshold;
    //   (c) CPU per-source BFS -- the fallback for very large condensations.
    bool try_gpu = false;
#ifdef IMPLGRAPH_HAS_CUDA
    if (data->use_gpu == 1) {
        try_gpu = true;
    } else if (data->use_gpu < 0) {
        try_gpu = presolve::CudaIsAvailable()
                  && (int)H.num_nodes > data->use_gpu_threshold;
    }
#endif
    const bool use_bitset =
        (data->use_bitset_reach == 1) ||
        (data->use_bitset_reach < 0 && H.num_nodes <= presolve::kBitsetReachThreshold);

    int n_fix0 = 0, n_fix1 = 0;
    bool gpu_succeeded = false;
#ifdef IMPLGRAPH_HAS_CUDA
    if (try_gpu) {
        const SCIP_Real tg0 = SCIPgetSolvingTime(scip);
        // Fused reach-and-query: builds the |C|x|C| transitive closure on
        // the GPU, then runs a small kernel that answers, for each binary i,
        // whether scc(2i+1) reaches scc(2i) (-> F_0) or vice versa (-> F_1).
        // Only the per-binary status vector (1 byte / binary) is DMA'd back,
        // not the full |C|^2/8 byte matrix.  On rail01-scale graphs this is
        // the difference between sub-100 ms total and ~2 s.
        std::vector<std::uint8_t> status =
            presolve::ComputeForcedLiteralsGPU(H, scc.scc_id, p, data->verbose);
        const SCIP_Real tg1 = SCIPgetSolvingTime(scip);
        if (!status.empty()) {
            gpu_succeeded = true;
            if (data->verbose >= 1) {
                SCIPinfoMessage(scip, NULL,
                    "[implgraph] GPU fused-query: |C|=%u processed in %.2fs\n",
                    (unsigned)H.num_nodes, tg1 - tg0);
            }
            for (Idx i = 0; i < p; ++i) {
                const std::uint8_t s = status[(std::size_t)i];
                if (s == 0) continue;
                SCIP_VAR* xi = literal_map.active_binary_vars[(std::size_t)i];
                if (!SCIPvarIsActive(xi)) continue;
                if (SCIPvarGetLbGlobal(xi) > 0.5 || SCIPvarGetUbGlobal(xi) < 0.5) continue;
                // s: bit 0 = force 0 (s1->s0), bit 1 = force 1 (s0->s1)
                if (s == 3) {
                    *result = SCIP_CUTOFF; return SCIP_OKAY;
                }
                if ((s & 1u) && data->apply_fixings) {
                    SCIP_Bool fixed = FALSE;
                    SCIP_CALL( SCIPfixVar(scip, xi, 0.0, &infeasible, &fixed) );
                    if (infeasible) { *result = SCIP_CUTOFF; return SCIP_OKAY; }
                    if (fixed) ++n_fix0;
                } else if ((s & 2u) && data->apply_fixings) {
                    SCIP_Bool fixed = FALSE;
                    SCIP_CALL( SCIPfixVar(scip, xi, 1.0, &infeasible, &fixed) );
                    if (infeasible) { *result = SCIP_CUTOFF; return SCIP_OKAY; }
                    if (fixed) ++n_fix1;
                }
            }
        } else if (data->verbose >= 1) {
            SCIPinfoMessage(scip, NULL,
                "[implgraph] GPU reach unavailable; falling back to CPU\n");
        }
    }
#endif

    if (!gpu_succeeded && use_bitset) {
        presolve::BitsetReach R = presolve::BitsetReachable(H);
        for (Idx i = 0; i < p; ++i) {
            SCIP_VAR* xi = literal_map.active_binary_vars[(std::size_t)i];
            if (!SCIPvarIsActive(xi)) continue;
            if (SCIPvarGetLbGlobal(xi) > 0.5 || SCIPvarGetUbGlobal(xi) < 0.5) continue;

            const Idx s0 = scc.scc_id[(std::size_t)(2 * i    )];
            const Idx s1 = scc.scc_id[(std::size_t)(2 * i + 1)];

            const bool s0_reaches_s1 = R.Test(s0, s1);  // x_i = 0 => x_i = 1: forces x_i = 1
            const bool s1_reaches_s0 = R.Test(s1, s0);  // x_i = 1 => x_i = 0: forces x_i = 0

            if (s0_reaches_s1 && s1_reaches_s0) {       // would have been caught earlier
                *result = SCIP_CUTOFF; return SCIP_OKAY;
            }
            if (s1_reaches_s0 && data->apply_fixings) {
                SCIP_Bool fixed = FALSE;
                SCIP_CALL( SCIPfixVar(scip, xi, 0.0, &infeasible, &fixed) );
                if (infeasible) { *result = SCIP_CUTOFF; return SCIP_OKAY; }
                if (fixed) ++n_fix0;
            } else if (s0_reaches_s1 && data->apply_fixings) {
                SCIP_Bool fixed = FALSE;
                SCIP_CALL( SCIPfixVar(scip, xi, 1.0, &infeasible, &fixed) );
                if (infeasible) { *result = SCIP_CUTOFF; return SCIP_OKAY; }
                if (fixed) ++n_fix1;
            }
        }
    } else if (!gpu_succeeded) {
        // Per-source BFS.  We query at most one BFS per SCC that contains a
        // literal whose partner is in a different SCC; for very large H
        // (rail03-style) this is essential to stay within memory budget.
        //
        // We cache "reached-from-S" sets only when both literals' SCCs need
        // a query, which is the common case.  A simple memoization map is
        // enough -- O(|C|) entries amortized.
        std::unordered_map<Idx, std::vector<Idx>> cache;
        auto reached_from = [&](Idx s) -> const std::vector<Idx>& {
            auto it = cache.find(s);
            if (it != cache.end()) return it->second;
            auto v = presolve::PerSourceBFSReachable(H, s);
            auto [ins, ok] = cache.emplace(s, std::move(v));
            return ins->second;
        };
        auto reaches = [&](Idx s_from, Idx s_to) -> bool {
            const auto& r = reached_from(s_from);
            return std::binary_search(r.begin(), r.end(), s_to);
        };

        for (Idx i = 0; i < p; ++i) {
            SCIP_VAR* xi = literal_map.active_binary_vars[(std::size_t)i];
            if (!SCIPvarIsActive(xi)) continue;
            if (SCIPvarGetLbGlobal(xi) > 0.5 || SCIPvarGetUbGlobal(xi) < 0.5) continue;

            const Idx s0 = scc.scc_id[(std::size_t)(2 * i    )];
            const Idx s1 = scc.scc_id[(std::size_t)(2 * i + 1)];
            const bool s0_reaches_s1 = reaches(s0, s1);
            const bool s1_reaches_s0 = reaches(s1, s0);

            if (s0_reaches_s1 && s1_reaches_s0) {
                *result = SCIP_CUTOFF; return SCIP_OKAY;
            }
            if (s1_reaches_s0 && data->apply_fixings) {
                SCIP_Bool fixed = FALSE;
                SCIP_CALL( SCIPfixVar(scip, xi, 0.0, &infeasible, &fixed) );
                if (infeasible) { *result = SCIP_CUTOFF; return SCIP_OKAY; }
                if (fixed) ++n_fix0;
            } else if (s0_reaches_s1 && data->apply_fixings) {
                SCIP_Bool fixed = FALSE;
                SCIP_CALL( SCIPfixVar(scip, xi, 1.0, &infeasible, &fixed) );
                if (infeasible) { *result = SCIP_CUTOFF; return SCIP_OKAY; }
                if (fixed) ++n_fix1;
            }
        }
    }
    const SCIP_Real t7 = SCIPgetSolvingTime(scip);

    // --- (6) Report -----------------------------------------------------
    *nfixedvars   += n_fix0 + n_fix1;
    *naggrvars    += n_aggregated_direct + n_aggregated_indirect;
    if (n_fix0 + n_fix1 + n_aggregated_direct + n_aggregated_indirect > 0) {
        *result = SCIP_SUCCESS;
    }
    if (data->verbose >= 1) {
        SCIPinfoMessage(scip, NULL,
            "[implgraph] DE=%d  IE=%d  F0=%d  F1=%d  "
            "(extract=%.2fs scc=%.2fs cond=%.2fs dsu=%.2fs apply=%.2fs reach=%.2fs)\n",
            n_aggregated_direct, n_aggregated_indirect, n_fix0, n_fix1,
            t1 - t0, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6);
    }
    return SCIP_OKAY;
}

}  // namespace

extern "C" SCIP_RETCODE SCIPincludePresolImplgraph(SCIP* scip) {
    auto* data = new PresolDataImplgraph();

    SCIP_PRESOL* presol = NULL;
    SCIP_CALL( SCIPincludePresolBasic(scip, &presol,
                                      PRESOL_NAME, PRESOL_DESC,
                                      PRESOL_PRIORITY, PRESOL_MAXROUNDS,
                                      PRESOL_TIMING, presolExecImplgraph,
                                      (SCIP_PRESOLDATA*)data) );
    SCIP_CALL( SCIPsetPresolFree(scip, presol, presolFreeImplgraph) );

    SCIP_CALL( SCIPaddBoolParam(scip, "presolving/" PRESOL_NAME "/enabled",
        "Enable the implication-graph SCC + reachability presolver",
        &data->enabled, FALSE, TRUE, NULL, NULL) );

    SCIP_CALL( SCIPaddIntParam(scip, "presolving/" PRESOL_NAME "/maxliterals",
        "Skip presolver if implication graph would have more than this many "
        "literals (0 = no cap)",
        &data->max_literals, FALSE, 0, 0, INT_MAX, NULL, NULL) );

    SCIP_CALL( SCIPaddIntParam(scip, "presolving/" PRESOL_NAME "/usebitsetreach",
        "Reachability backend: -1 auto, 0 per-source BFS, 1 bitset DP",
        &data->use_bitset_reach, FALSE, -1, -1, 1, NULL, NULL) );

    SCIP_CALL( SCIPaddBoolParam(scip, "presolving/" PRESOL_NAME "/applyfixings",
        "Apply zero/one fixings (F0/F1) via SCIPfixVar",
        &data->apply_fixings, FALSE, TRUE, NULL, NULL) );

    SCIP_CALL( SCIPaddBoolParam(scip, "presolving/" PRESOL_NAME "/applyaggregations",
        "Apply direct/indirect eliminations (DE/IE) via SCIPaggregateVars",
        &data->apply_aggregations, FALSE, TRUE, NULL, NULL) );

    SCIP_CALL( SCIPaddIntParam(scip, "presolving/" PRESOL_NAME "/verbose",
        "Verbosity 0..3 (0 silent)",
        &data->verbose, FALSE, 1, 0, 3, NULL, NULL) );

    SCIP_CALL( SCIPaddIntParam(scip, "presolving/" PRESOL_NAME "/extractor",
        "Implication extractor: 0 = SCIP constraint handlers (fast, narrow), "
        "1 = CBC manual LP-probing (BROKEN: 830s+ on acc-tight5 with no "
        "added arcs -- do NOT use), 2 = both (union).  The rich CBC pipeline "
        "now lives in scripts/build_conflict_graph.py and is consumed via "
        "the arcsfile parameter below.",
        &data->extractor, FALSE, 0, 0, 2, NULL, NULL) );

    SCIP_CALL( SCIPaddIntParam(scip,
        "presolving/" PRESOL_NAME "/cbcmaxresolves",
        "Cap on the number of LP resolves CBC probing performs "
        "(0 = unlimited; consider capping on >100k-binary instances)",
        &data->cbc_max_resolves, FALSE, 0, 0, INT_MAX, NULL, NULL) );

    SCIP_CALL( SCIPaddIntParam(scip,
        "presolving/" PRESOL_NAME "/cbcloglevel",
        "CBC internal log level (0 silent .. 3 noisy)",
        &data->cbc_log_level, FALSE, 0, 0, 3, NULL, NULL) );

    SCIP_CALL( SCIPaddStringParam(scip,
        "presolving/" PRESOL_NAME "/arcsfile",
        "Path to an external arcs file (one '<bit><name> <bit><name>' per line) "
        "produced by scripts/build_conflict_graph.py, i.e. CBC's "
        "CoinConflictGraph via python-mip; arcs are UNIONED with SCIP "
        "constraint-handler arcs.  Empty string disables.",
        &data->arcs_file, FALSE, "", NULL, NULL) );

    SCIP_CALL( SCIPaddIntParam(scip,
        "presolving/" PRESOL_NAME "/usegpu",
        "Reach-phase backend: -1 auto (GPU if CUDA available and "
        "|C| > usegputhreshold), 0 force CPU, 1 force GPU.  When the "
        "plug-in is built without CUDA (no graph_utils_gpu.cu) this "
        "parameter is ignored.",
        &data->use_gpu, FALSE, -1, -1, 1, NULL, NULL) );

    SCIP_CALL( SCIPaddIntParam(scip,
        "presolving/" PRESOL_NAME "/usegputhreshold",
        "Auto-mode threshold on |C| (condensation size) above which the "
        "GPU backend is preferred over the CPU bitset DP.  Default 0 = "
        "always GPU when CUDA is available, which is right for any modern "
        "Hopper/Blackwell device.  Raise it on cards with limited memory "
        "to keep small instances on CPU.",
        &data->use_gpu_threshold, FALSE, 0, 0, INT_MAX, NULL, NULL) );

    return SCIP_OKAY;
}

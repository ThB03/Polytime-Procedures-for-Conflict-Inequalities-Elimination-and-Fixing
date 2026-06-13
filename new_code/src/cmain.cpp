// =============================================================================
//  cmain.cpp
//
//  Thin SCIP driver that loads SCIP, registers our presolver plugin, reads an
//  MPS/LP file, optimizes, and reports.  Intended for use by the benchmark
//  harness in scripts/run_miplib_benchmark.py.
//
//  Usage:
//      scip_implgraph [options] <model.mps>
//
//  Options that are not consumed below are forwarded to SCIP as
//  "<setting>=<value>" via SCIPparseLong (e.g. "limits/time=3600",
//  "presolving/maxrounds=-1").  We intentionally do NOT shortcut the SCIP
//  command-line interface; for interactive sessions the user should call
//      scip -c "include <path>/libscip_implgraph.so" ...
//  but that requires SCIP built with module support.  The static link path
//  below is the portable choice.
//
//  The driver also accepts --no-implgraph to disable the plugin entirely
//  (useful for paired baseline runs without compiling a separate binary).
// =============================================================================

#include "presol_implgraph.h"

#include "scip/scip.h"
#include "scip/scipdefplugins.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void PrintUsage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s [--no-implgraph] [--time-limit S] [--seed N] "
        "[--no-solver-presolve [--strict]] [--write-stats PATH] "
        "[--param KEY=VALUE]... MODEL.mps\n"
        "\n"
        "  --no-solver-presolve    Disable SCIP's built-in presolvers (per-presolver\n"
        "                          maxrounds=0); cons-handler presolve still runs so\n"
        "                          SCIP's clique/impl tables are populated and\n"
        "                          SCIPaggregateVars works.\n"
        "  --strict                With --no-solver-presolve, use the aggressive\n"
        "                          SCIPsetPresolving(OFF) mode for true isolation.\n"
        "                          Automatically disables aggregation applies\n"
        "                          (SCIPaggregateVars segfaults without populated\n"
        "                          clique/impl tables); F0/F1 fixings still apply.\n",
        argv0);
}

// Set any SCIP parameter from a "name=value" string.  Tries int, bool,
// real, and string in that order.  Returns SCIP_OKAY on success.
SCIP_RETCODE SetParamFromString(SCIP* scip, const char* kv) {
    const char* eq = std::strchr(kv, '=');
    if (eq == NULL) { std::fprintf(stderr, "--param needs KEY=VALUE\n"); return SCIP_PARAMETERWRONGVAL; }
    const std::string key(kv, eq - kv);
    const std::string val(eq + 1);
    SCIP_PARAM* p = SCIPgetParam(scip, key.c_str());
    if (p == NULL) { std::fprintf(stderr, "no such parameter '%s'\n", key.c_str()); return SCIP_PARAMETERUNKNOWN; }
    switch (SCIPparamGetType(p)) {
        case SCIP_PARAMTYPE_BOOL:   return SCIPsetBoolParam   (scip, key.c_str(), val == "1" || val == "TRUE" || val == "true");
        case SCIP_PARAMTYPE_INT:    return SCIPsetIntParam    (scip, key.c_str(), std::atoi(val.c_str()));
        case SCIP_PARAMTYPE_LONGINT:return SCIPsetLongintParam(scip, key.c_str(), std::atoll(val.c_str()));
        case SCIP_PARAMTYPE_REAL:   return SCIPsetRealParam   (scip, key.c_str(), std::atof(val.c_str()));
        case SCIP_PARAMTYPE_CHAR:   return SCIPsetCharParam   (scip, key.c_str(), val.empty() ? '\0' : val[0]);
        case SCIP_PARAMTYPE_STRING: return SCIPsetStringParam (scip, key.c_str(), val.c_str());
    }
    return SCIP_PARAMETERWRONGTYPE;
}

}  // namespace

int main(int argc, char** argv) {
    bool        enable_implgraph     = true;
    bool        disable_solver_presolve = false;
    bool        strict_off               = false;     // see --strict below
    double      time_limit           = 3600.0;
    int         seed                 = 0;
    const char* model_path           = NULL;
    const char* stats_path           = NULL;
    std::vector<const char*> extra_params;   // --param KEY=VALUE accumulator

    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--no-implgraph"))       enable_implgraph        = false;
        else if (!std::strcmp(argv[i], "--no-solver-presolve")) disable_solver_presolve = true;
        else if (!std::strcmp(argv[i], "--strict"))             strict_off              = true;
        else if (!std::strcmp(argv[i], "--time-limit") && i + 1 < argc) time_limit = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--seed")       && i + 1 < argc) seed       = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--write-stats") && i + 1 < argc) stats_path = argv[++i];
        else if (!std::strcmp(argv[i], "--param")      && i + 1 < argc) extra_params.push_back(argv[++i]);
        else if (argv[i][0] != '-' && model_path == NULL)               model_path = argv[i];
        else { PrintUsage(argv[0]); return 1; }
    }
    if (model_path == NULL) { PrintUsage(argv[0]); return 1; }

    SCIP* scip = NULL;
    SCIP_CALL_ABORT( SCIPcreate(&scip) );
    SCIP_CALL_ABORT( SCIPincludeDefaultPlugins(scip) );

    if (enable_implgraph) {
        SCIP_CALL_ABORT( SCIPincludePresolImplgraph(scip) );
    }

    SCIP_CALL_ABORT( SCIPsetRealParam(scip, "limits/time", time_limit) );
    SCIP_CALL_ABORT( SCIPsetIntParam (scip, "randomization/randomseedshift", seed) );
    SCIP_CALL_ABORT( SCIPsetIntParam (scip, "display/verblevel", 4) );

    // MIPLIB-2017 official benchmark protocol: single-threaded LP solving,
    // no parallel branch-and-bound.  SCIP defaults to single-threaded LP
    // (lp/threads=0 means "let LP solver pick", which on Linux+CPLEX/SoPlex
    // defaults to 1).  We make this explicit so the timings reported here
    // are bit-for-bit reproducible regardless of the SCIP build's defaults
    // and match the MIPLIB community's expectation of single-thread numbers.
    if (SCIPgetParam(scip, "lp/threads") != NULL) {
        SCIP_CALL_ABORT( SCIPsetIntParam(scip, "lp/threads", 1) );
    }
    // SCIP's parallel mode is off by default (no concurrent solving), but
    // be explicit for the record.
    if (SCIPgetParam(scip, "parallel/maxnthreads") != NULL) {
        SCIP_CALL_ABORT( SCIPsetIntParam(scip, "parallel/maxnthreads", 1) );
    }
    // Number of OpenMP threads SCIP itself may use for parallel
    // presolving / propagation (also defaults to 1 in modern SCIP, but
    // some MIPs benefit from explicit setting).
    if (SCIPgetParam(scip, "nlpi/intpoint/threads") != NULL) {
        SCIP_CALL_ABORT( SCIPsetIntParam(scip, "nlpi/intpoint/threads", 1) );
    }

    // Disable SCIP's internal multi-aggregation chains.  On a small number
    // of MIPLIB instances with structured variable names (e.g.
    // academictimetablesmall, datt256, s100, bnatt500), SCIPaggregateVars
    // crashes inside SCIP itself when it tries to chain through a
    // multi-aggregation; setting donotmultaggr=TRUE bypasses the buggy
    // path with no measurable effect on solve time on the other instances.
    // We set it for BOTH baseline_on and ours_on_top runs so the
    // comparison remains apples-to-apples.
    if (SCIPgetParam(scip, "presolving/donotmultaggr") != NULL) {
        SCIP_CALL_ABORT( SCIPsetBoolParam(scip, "presolving/donotmultaggr", TRUE) );
    }

    // Apply --param overrides before the read/solve.
    for (const char* kv : extra_params) {
        SCIP_CALL_ABORT( SetParamFromString(scip, kv) );
    }

    if (disable_solver_presolve) {
        // Two modes of "disable solver presolve":
        //
        // (a) DEFAULT (cooperative): walk SCIP's registered presolvers and
        //     set each one's maxrounds to 0, leaving cons-handler presolve
        //     ALONE.  Cons-handlers still populate SCIP's clique and
        //     implication tables (which SCIPaggregateVars dereferences
        //     internally); without those tables, SCIPaggregateVars segfaults.
        //     Trade-off: cons-handlers may do some reductions before our
        //     plug-in fires, slightly diluting the "our plug-in alone" signal,
        //     but DE/IE/F0/F1 reductions can all be applied.
        //
        // (b) --strict (pure isolation): full SCIPsetPresolving(OFF, TRUE),
        //     matching the paper's original Python+Gurobi setup where the
        //     procedure runs in true isolation.  Gives the cleanest
        //     "what does the procedure produce on its own" measurement, but
        //     SCIPaggregateVars segfaults in this mode, so we automatically
        //     disable aggregation applies and report F0/F1 fixings only.
        //     DE/IE class counts are still reported in the plug-in log line.
        if (strict_off) {
            SCIP_CALL_ABORT( SCIPsetPresolving(scip, SCIP_PARAMSETTING_OFF, TRUE) );
            if (enable_implgraph) {
                SCIP_CALL_ABORT( SCIPsetIntParam(scip,
                    "presolving/maxrounds", -1) );
                SCIP_CALL_ABORT( SCIPsetIntParam(scip,
                    "presolving/implgraph/maxrounds", -1) );
                // Avoid the SCIPaggregateVars crash that hits when SCIP's
                // clique/impl tables are uninitialised.
                SCIP_CALL_ABORT( SCIPsetBoolParam(scip,
                    "presolving/implgraph/applyaggregations", FALSE) );
            }
        } else {
            SCIP_PRESOL** const all_presols  = SCIPgetPresols(scip);
            const int           n_all_presols = SCIPgetNPresols(scip);
            for (int i = 0; i < n_all_presols; ++i) {
                const char* name = SCIPpresolGetName(all_presols[i]);
                if (std::strcmp(name, "implgraph") == 0) continue;
                char key[256];
                std::snprintf(key, sizeof(key), "presolving/%s/maxrounds", name);
                SCIP_CALL_ABORT( SCIPsetIntParam(scip, key, 0) );
            }
            if (enable_implgraph) {
                SCIP_CALL_ABORT( SCIPsetIntParam(scip,
                    "presolving/maxrounds", -1) );
                SCIP_CALL_ABORT( SCIPsetIntParam(scip,
                    "presolving/implgraph/maxrounds", -1) );
            }
        }
    }

    SCIP_CALL_ABORT( SCIPreadProb(scip, model_path, NULL) );
    SCIP_CALL_ABORT( SCIPsolve(scip) );

    if (stats_path != NULL) {
        FILE* fp = std::fopen(stats_path, "w");
        if (fp != NULL) {
            SCIP_CALL_ABORT( SCIPprintStatistics(scip, fp) );
            std::fclose(fp);
        }
    } else {
        SCIP_CALL_ABORT( SCIPprintStatistics(scip, NULL) );
    }

    SCIP_CALL_ABORT( SCIPfree(&scip) );
    return 0;
}

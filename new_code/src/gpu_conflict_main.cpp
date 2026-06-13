// =============================================================================
//  gpu_conflict_main.cpp
//
//  Standalone CLI driver for the GPU-native conflict graph builder.  Reads
//  an MPS file with SCIP, extracts the linear/setppc/logicor constraint
//  matrix, runs the GPU residual extractor (gpu_conflict_extractor.cu), and
//  writes a binary-binary conflict arcs file in the format consumed by the
//  SCIP plug-in via presolving/implgraph/arcsfile.
//
//  Drop-in replacement for scripts/build_conflict_graph.py (CBC's
//  CoinConflictGraph + python-mip) on instances where CBC OOMs.  The output
//  format is identical:
//      <bit><varname>   <bit><varname>      # one arc per line
//  with <bit> in {'0','1'}, varname the MPS variable name, and lines
//  starting with '#' treated as comments.
//
//  Usage:
//      gpu_conflict_graph  <input.mps>  <output.arcs>
//          [--cap N] [--max-row-width K] [--no-dedup] [--verbose]
// =============================================================================

#include "gpu_conflict_extractor.h"

#include "scip/scip.h"
#include "scip/scipdefplugins.h"
#include "scip/cons_linear.h"
#include "scip/cons_setppc.h"
#include "scip/cons_logicor.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using presolve::ConstraintMatrix;
using presolve::GpuExtractOptions;

namespace {

void PrintUsage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s <input.mps> <output.arcs> "
        "[--cap N] [--max-row-width K] [--no-dedup] [--verbose]\n",
        argv0);
}

// Iterate cons_linear, cons_setppc, cons_logicor and append each row to the
// constraint matrix.  We use SCIP's getters so coefficient signs / bounds
// are normalized to the constraint handler's canonical form.
//
// Variable indexing: we map SCIP_VAR* -> column index in extraction order
// (first encountered = column 0).  The binary subset gets a dense 0..p-1
// index in `binary_index`.
SCIP_RETCODE BuildMatrix(SCIP* scip, ConstraintMatrix& M, bool verbose) {
    // Collect variables; assign binary subset to dense indices.
    SCIP_VAR** const vars = SCIPgetVars(scip);
    const int n_vars = SCIPgetNVars(scip);
    std::unordered_map<SCIP_VAR*, int> var_to_col;
    var_to_col.reserve((std::size_t)n_vars * 2);
    M.n_cols = n_vars;
    M.binary_index.assign((std::size_t)n_vars, -1);
    M.col_lb.assign((std::size_t)n_vars, 0.0);
    M.col_ub.assign((std::size_t)n_vars, 1.0);
    M.binary_names.reserve(256);

    int n_bin = 0;
    for (int c = 0; c < n_vars; ++c) {
        SCIP_VAR* v = vars[c];
        var_to_col[v] = c;
        M.col_lb[(std::size_t)c] = SCIPvarGetLbGlobal(v);
        M.col_ub[(std::size_t)c] = SCIPvarGetUbGlobal(v);
        if (SCIPvarGetType(v) == SCIP_VARTYPE_BINARY
            && SCIPvarIsActive(v)
            && M.col_lb[(std::size_t)c] < 0.5
            && M.col_ub[(std::size_t)c] > 0.5) {
            M.binary_index[(std::size_t)c] = n_bin++;
            M.binary_names.emplace_back(SCIPvarGetName(v));
        }
    }
    M.n_binaries = n_bin;

    // CSR accumulators.
    M.row_xadj.clear();   M.row_xadj.reserve(4096);
    M.row_xadj.push_back(0);
    M.row_colidx.clear(); M.row_colidx.reserve(65536);
    M.row_coef.clear();   M.row_coef.reserve(65536);
    M.row_lhs.clear();    M.row_lhs.reserve(4096);
    M.row_rhs.clear();    M.row_rhs.reserve(4096);

    auto add_row = [&](double lhs, double rhs,
                       SCIP_VAR** rvars, double* rcoefs, int rnvars) {
        int n_with_col = 0;
        for (int i = 0; i < rnvars; ++i) {
            // We need the PROBVAR (the active representative).  For binaries,
            // SCIPvarGetProbvar returns the active var; for aggregated, the
            // chain leader.  We use the var as-is to keep column indexing
            // aligned with SCIPgetVars().
            auto it = var_to_col.find(rvars[i]);
            if (it == var_to_col.end()) continue;
            M.row_colidx.push_back(it->second);
            M.row_coef  .push_back(rcoefs[i]);
            ++n_with_col;
        }
        if (n_with_col == 0) return;
        M.row_xadj.push_back((int)M.row_colidx.size());
        M.row_lhs.push_back(lhs);
        M.row_rhs.push_back(rhs);
    };

    // ---- cons_linear ----------------------------------------------------
    SCIP_CONSHDLR* hdlr_lin = SCIPfindConshdlr(scip, "linear");
    if (hdlr_lin != nullptr) {
        const int n_lin = SCIPconshdlrGetNConss(hdlr_lin);
        SCIP_CONS** const conss = SCIPconshdlrGetConss(hdlr_lin);
        for (int c = 0; c < n_lin; ++c) {
            SCIP_CONS* cons = conss[c];
            const int nvars  = SCIPgetNVarsLinear(scip, cons);
            SCIP_VAR**  rvars = SCIPgetVarsLinear(scip, cons);
            SCIP_Real*  rcoef = SCIPgetValsLinear(scip, cons);
            const double lhs = SCIPgetLhsLinear(scip, cons);
            const double rhs = SCIPgetRhsLinear(scip, cons);
            add_row(lhs, rhs, rvars, rcoef, nvars);
        }
        if (verbose) {
            std::fprintf(stdout, "[gpu-conflict] cons_linear: %d constraints\n", n_lin);
        }
    }

    // ---- cons_setppc ----------------------------------------------------
    SCIP_CONSHDLR* hdlr_spc = SCIPfindConshdlr(scip, "setppc");
    if (hdlr_spc != nullptr) {
        const int n_spc = SCIPconshdlrGetNConss(hdlr_spc);
        SCIP_CONS** const conss = SCIPconshdlrGetConss(hdlr_spc);
        std::vector<double> coefs;
        coefs.reserve(1024);
        for (int c = 0; c < n_spc; ++c) {
            SCIP_CONS* cons = conss[c];
            const int nvars = SCIPgetNVarsSetppc(scip, cons);
            SCIP_VAR**  rvars = SCIPgetVarsSetppc(scip, cons);
            coefs.assign((std::size_t)nvars, 1.0);
            const SCIP_SETPPCTYPE t = SCIPgetTypeSetppc(scip, cons);
            double lhs = -SCIPinfinity(scip);
            double rhs =  SCIPinfinity(scip);
            switch (t) {
                case SCIP_SETPPCTYPE_PARTITIONING: lhs = 1.0; rhs = 1.0; break;
                case SCIP_SETPPCTYPE_PACKING:                  rhs = 1.0; break;
                case SCIP_SETPPCTYPE_COVERING:     lhs = 1.0;           break;
            }
            add_row(lhs, rhs, rvars, coefs.data(), nvars);
        }
        if (verbose) {
            std::fprintf(stdout, "[gpu-conflict] cons_setppc: %d constraints\n", n_spc);
        }
    }

    // ---- cons_logicor (literals sum to >= 1) ----------------------------
    //
    // logicor stores literals; SCIPgetVarsLogicor returns the literal vars,
    // which may be negated copies of the active variables.  To stay aligned
    // with the var_to_col map (which has the ACTIVE vars), we resolve each
    // literal to its active var and flip the coefficient sign accordingly.
    //
    // A logicor constraint  l1 + l2 + ... + lk >= 1
    // where lk = (1 - xk) for negated literals becomes
    //  sum_{positive} xi  -  sum_{negated} xi  >=  1 - (# negated)
    SCIP_CONSHDLR* hdlr_lor = SCIPfindConshdlr(scip, "logicor");
    if (hdlr_lor != nullptr) {
        const int n_lor = SCIPconshdlrGetNConss(hdlr_lor);
        SCIP_CONS** const conss = SCIPconshdlrGetConss(hdlr_lor);
        std::vector<SCIP_VAR*> rvars; std::vector<double> rcoefs;
        rvars.reserve(1024); rcoefs.reserve(1024);
        for (int c = 0; c < n_lor; ++c) {
            SCIP_CONS* cons = conss[c];
            const int nvars  = SCIPgetNVarsLogicor(scip, cons);
            SCIP_VAR**  lvars = SCIPgetVarsLogicor(scip, cons);
            rvars.clear(); rcoefs.clear();
            int n_neg = 0;
            for (int i = 0; i < nvars; ++i) {
                SCIP_VAR* lv = lvars[i];
                if (SCIPvarIsNegated(lv)) {
                    rvars.push_back(SCIPvarGetNegatedVar(lv));
                    rcoefs.push_back(-1.0);
                    ++n_neg;
                } else {
                    rvars.push_back(lv);
                    rcoefs.push_back(1.0);
                }
            }
            const double lhs = 1.0 - (double)n_neg;
            const double rhs = SCIPinfinity(scip);
            add_row(lhs, rhs, rvars.data(), rcoefs.data(), (int)rvars.size());
        }
        if (verbose) {
            std::fprintf(stdout, "[gpu-conflict] cons_logicor: %d constraints\n", n_lor);
        }
    }

    M.n_rows = (int)M.row_lhs.size();
    if (verbose) {
        std::fprintf(stdout,
            "[gpu-conflict] matrix built: n_cols=%d n_rows=%d nnz=%zu n_binaries=%d\n",
            M.n_cols, M.n_rows, M.row_coef.size(), M.n_binaries);
    }
    return SCIP_OKAY;
}

void WriteArcs(const std::vector<int32_t>& arcs,
               const ConstraintMatrix& M,
               const std::string& path,
               bool verbose) {
    // Build inverse: binary_index -> name string.  M.binary_names already
    // has them in the right order.
    std::ofstream out(path);
    if (!out) {
        std::fprintf(stderr, "could not open %s for writing\n", path.c_str());
        std::exit(2);
    }
    out << "# GPU-extracted conflict graph; format: <bit><name> <bit><name>\n";
    const std::size_t n_pairs = arcs.size() / 2;
    out << "# " << n_pairs << " arcs over " << M.n_binaries << " binaries\n";
    int n_written = 0;
    for (std::size_t k = 0; k < n_pairs; ++k) {
        const int32_t la = arcs[2 * k];
        const int32_t lb = arcs[2 * k + 1];
        const int va = la >> 1, ba = la & 1;
        const int vb = lb >> 1, bb = lb & 1;
        if (va < 0 || va >= M.n_binaries || vb < 0 || vb >= M.n_binaries) continue;
        out << ba << M.binary_names[(std::size_t)va] << ' '
            << bb << M.binary_names[(std::size_t)vb] << '\n';
        ++n_written;
    }
    if (verbose) {
        std::fprintf(stdout, "[gpu-conflict] wrote %d arcs to %s\n",
                     n_written, path.c_str());
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { PrintUsage(argv[0]); return 1; }
    const std::string mps_path  = argv[1];
    const std::string arcs_path = argv[2];

    GpuExtractOptions opts;
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--cap") && i + 1 < argc) {
            opts.arc_buf_cap = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--max-row-width") && i + 1 < argc) {
            opts.max_row_width = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--no-dedup")) {
            opts.dedup = false;
        } else if (!std::strcmp(argv[i], "--verbose")) {
            opts.verbose = true;
        } else {
            PrintUsage(argv[0]); return 1;
        }
    }

    SCIP* scip = nullptr;
    SCIP_CALL_ABORT( SCIPcreate(&scip) );
    SCIP_CALL_ABORT( SCIPincludeDefaultPlugins(scip) );

    // Silence SCIP's chatter; we only care about our own logs.
    SCIP_CALL_ABORT( SCIPsetIntParam(scip, "display/verblevel", 0) );

    if (opts.verbose) {
        std::fprintf(stdout, "[gpu-conflict] reading %s ...\n", mps_path.c_str());
    }
    SCIP_CALL_ABORT( SCIPreadProb(scip, mps_path.c_str(), nullptr) );
    SCIP_CALL_ABORT( SCIPtransformProb(scip) );

    ConstraintMatrix M;
    SCIP_CALL_ABORT( BuildMatrix(scip, M, opts.verbose) );

    auto arcs = presolve::ExtractConflictsGPU(M, opts);

    WriteArcs(arcs, M, arcs_path, opts.verbose);

    SCIP_CALL_ABORT( SCIPfree(&scip) );
    return 0;
}

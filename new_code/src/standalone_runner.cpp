// =============================================================================
//  standalone_runner.cpp
//
//  No-SCIP standalone runner for the implication-graph pipeline.
//
//  Reads an implication-arc text file in the format produced by the original
//  Python pipeline (and accepted by the validation script in
//  scripts/dump_python_arcs.py).  Each non-blank, non-comment line has the form
//
//      <src-literal>  <tgt-literal>
//
//  where each literal is the variable name prefixed with '0' (for x = 0) or
//  '1' (for x = 1), exactly as in the original code.  Example:
//
//      0x1   0x2
//      1x2   1x1
//      0x3   1x4
//      ...
//
//  Lines starting with '#' are ignored.  The runner emits, on stdout, the same
//  four reduction sets the Python pipeline produces:
//
//      DE  count  pair-list (var = var)
//      IE  count  pair-list (var = 1 - var)
//      F0  count  var-list
//      F1  count  var-list
//
//  Exit code is 0 on success, 1 on infeasibility (a literal and its partner
//  end up in the same SCC), 2 on parse errors.
//
//  This binary is the cleanest way to bit-for-bit validate the C++ core
//  against the Python reference without installing SCIP.  The recommended
//  workflow is:
//
//      python scripts/dump_python_arcs.py instance.mps > arcs.txt
//      ./bin/standalone_runner arcs.txt > cpp_reductions.txt
//      python scripts/dump_python_reductions.py instance.mps > py_reductions.txt
//      diff cpp_reductions.txt py_reductions.txt
//
//  (The two dump_* scripts are sketched in scripts/README.md.)
// =============================================================================

#include "graph_utils.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using presolve::Idx;
using presolve::kInvalid;

struct LiteralIndex {
    std::vector<std::string>               varname_of_var;
    std::unordered_map<std::string, Idx>   var_of_name;

    // Returns the literal index (2 * var_idx + status) for a token like "0x1".
    // Inserts the variable if it has not been seen before.
    Idx LiteralIndexOf(const std::string& tok, int line) {
        if (tok.size() < 2 || (tok[0] != '0' && tok[0] != '1')) {
            std::fprintf(stderr,
                "parse error on line %d: literal '%s' must start with '0' or '1'\n",
                line, tok.c_str());
            std::exit(2);
        }
        const int status = (tok[0] == '1');
        const std::string vn = tok.substr(1);
        auto it = var_of_name.find(vn);
        if (it == var_of_name.end()) {
            const Idx new_idx = (Idx)varname_of_var.size();
            var_of_name.emplace(vn, new_idx);
            varname_of_var.push_back(vn);
            return 2 * new_idx + status;
        }
        return 2 * it->second + status;
    }
};

struct ParsedArcs {
    LiteralIndex                            idx;
    std::vector<std::pair<Idx, Idx>>        arcs;
};

ParsedArcs ParseArcFile(const std::string& path) {
    ParsedArcs P;
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "could not open '%s'\n", path.c_str());
        std::exit(2);
    }
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        // Strip comments and trim.
        auto hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        std::istringstream iss(line);
        std::string a, b, extra;
        if (!(iss >> a)) continue;          // blank line
        if (!(iss >> b)) {
            std::fprintf(stderr, "parse error on line %d: expected two tokens\n", lineno);
            std::exit(2);
        }
        if (iss >> extra) {
            std::fprintf(stderr, "parse error on line %d: stray token '%s'\n",
                         lineno, extra.c_str());
            std::exit(2);
        }
        const Idx la = P.idx.LiteralIndexOf(a, lineno);
        const Idx lb = P.idx.LiteralIndexOf(b, lineno);
        P.arcs.emplace_back(la, lb);
    }
    return P;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s arcs.txt\n", argv[0]);
        return 2;
    }

    auto parsed = ParseArcFile(argv[1]);
    const Idx nlit = (Idx)(2 * parsed.idx.varname_of_var.size());
    if (nlit == 0) {
        std::fprintf(stderr, "no literals\n");
        return 2;
    }

    auto D = presolve::BuildCSR(nlit, parsed.arcs);
    std::vector<std::pair<Idx, Idx>>().swap(parsed.arcs);

    auto scc = presolve::TarjanSCC(D);
    if (presolve::DetectInfeasibility(scc, nlit)) {
        std::printf("# INFEASIBLE: literal and partner share an SCC\n");
        return 1;
    }

    auto H = presolve::BuildCondensation(D, scc);

    // ---- DE / IE via parity-DSU sweep ----------------------------------
    const Idx p = (Idx)parsed.idx.varname_of_var.size();
    presolve::ParityDSU dsu(p);
    std::vector<Idx> rep_of_scc((std::size_t)scc.num_sccs, kInvalid);
    for (Idx l = 0; l < nlit; ++l) {
        const Idx s = scc.scc_id[(std::size_t)l];
        if (rep_of_scc[(std::size_t)s] == kInvalid) {
            rep_of_scc[(std::size_t)s] = l;
            continue;
        }
        const Idx     r       = rep_of_scc[(std::size_t)s];
        const Idx     var_l   = l >> 1;
        const Idx     var_r   = r >> 1;
        if (var_l == var_r) continue;
        const std::uint8_t rel = (std::uint8_t)((l & 1) ^ (r & 1));
        if (!dsu.Union(var_l, var_r, rel)) {
            std::printf("# INFEASIBLE: inconsistent parity in SCC\n");
            return 1;
        }
    }
    std::vector<std::pair<Idx, Idx>>  DE;          // var_i = var_root
    std::vector<std::pair<Idx, Idx>>  IE;          // var_i = 1 - var_root
    for (Idx i = 0; i < p; ++i) {
        auto [r, par] = dsu.FindWithParity(i);
        if (r == i) continue;
        (par == 0 ? DE : IE).emplace_back(i, r);
    }

    // ---- F0 / F1 via reachability on H ---------------------------------
    std::vector<Idx> F0, F1;
    if (H.num_nodes <= presolve::kBitsetReachThreshold) {
        auto R = presolve::BitsetReachable(H);
        for (Idx i = 0; i < p; ++i) {
            const Idx s0 = scc.scc_id[(std::size_t)(2 * i    )];
            const Idx s1 = scc.scc_id[(std::size_t)(2 * i + 1)];
            const bool s0_to_s1 = R.Test(s0, s1);
            const bool s1_to_s0 = R.Test(s1, s0);
            if (s1_to_s0)      F0.push_back(i);
            else if (s0_to_s1) F1.push_back(i);
        }
    } else {
        std::unordered_map<Idx, std::vector<Idx>> cache;
        auto reaches = [&](Idx s_from, Idx s_to) -> bool {
            auto it = cache.find(s_from);
            if (it == cache.end()) {
                auto v = presolve::PerSourceBFSReachable(H, s_from);
                it = cache.emplace(s_from, std::move(v)).first;
            }
            return std::binary_search(it->second.begin(), it->second.end(), s_to);
        };
        for (Idx i = 0; i < p; ++i) {
            const Idx s0 = scc.scc_id[(std::size_t)(2 * i    )];
            const Idx s1 = scc.scc_id[(std::size_t)(2 * i + 1)];
            if (reaches(s1, s0))      F0.push_back(i);
            else if (reaches(s0, s1)) F1.push_back(i);
        }
    }

    // ---- Emit ---------------------------------------------------------
    auto name = [&](Idx i) { return parsed.idx.varname_of_var[(std::size_t)i].c_str(); };

    std::printf("# n=%d  |A|=%d  |C|=%d  |F|=%d\n",
                (int)D.n, (int)D.nArcs, (int)scc.num_sccs, (int)H.num_arcs);

    std::printf("DE %zu\n", DE.size());
    std::sort(DE.begin(), DE.end(), [&](const auto& a, const auto& b) {
        return std::strcmp(name(a.first), name(b.first)) < 0;
    });
    for (const auto& e : DE) std::printf("  %s = %s\n", name(e.first), name(e.second));

    std::printf("IE %zu\n", IE.size());
    std::sort(IE.begin(), IE.end(), [&](const auto& a, const auto& b) {
        return std::strcmp(name(a.first), name(b.first)) < 0;
    });
    for (const auto& e : IE) std::printf("  %s = 1 - %s\n", name(e.first), name(e.second));

    std::sort(F0.begin(), F0.end(), [&](Idx a, Idx b) {
        return std::strcmp(name(a), name(b)) < 0;
    });
    std::printf("F0 %zu\n", F0.size());
    for (Idx i : F0) std::printf("  %s\n", name(i));

    std::sort(F1.begin(), F1.end(), [&](Idx a, Idx b) {
        return std::strcmp(name(a), name(b)) < 0;
    });
    std::printf("F1 %zu\n", F1.size());
    for (Idx i : F1) std::printf("  %s\n", name(i));

    return 0;
}

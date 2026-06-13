// =============================================================================
//  bench_reach_gpu.cpp
//
//  Standalone benchmark and correctness test for the GPU reachability backend.
//  Loads an implication-arc file (same format as standalone_runner), builds
//  the implication digraph, runs Tarjan SCC + condensation, then runs the
//  CPU BitsetReachable and the GPU BitsetReachableGPU side-by-side and:
//
//    (a) verifies that R_cpu.Test(s, t) == R_gpu.Test(s, t) for every (s, t)
//        pair the implication-graph plug-in would query (one per active
//        binary in each direction);
//    (b) reports wall-clock timings of both backends.
//
//  Build: linked from CMakeLists.txt as target `bench_reach_gpu` when
//  IMPLGRAPH_HAS_CUDA is enabled.
//
//  Usage:
//      ./bench_reach_gpu  arcs.txt  [--repeat N]
// =============================================================================

#include "graph_utils.h"
#include "graph_utils_gpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using presolve::Idx;
using presolve::kInvalid;

namespace {

struct ParsedArcs {
    std::vector<std::pair<Idx, Idx>>     arcs;
    std::unordered_map<std::string, Idx> name_to_var;
    std::vector<std::string>             var_names;
};

// Parse "<bit><name> <bit><name>" lines (same format as standalone_runner).
ParsedArcs Parse(const std::string& path) {
    ParsedArcs P;
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "could not open %s\n", path.c_str());
        std::exit(2);
    }
    auto lit_of = [&](const std::string& tok) -> Idx {
        if (tok.size() < 2 || (tok[0] != '0' && tok[0] != '1')) {
            std::fprintf(stderr, "bad token %s\n", tok.c_str());
            std::exit(2);
        }
        const int v = (tok[0] == '1') ? 1 : 0;
        const std::string n = tok.substr(1);
        auto it = P.name_to_var.find(n);
        Idx vidx;
        if (it != P.name_to_var.end()) {
            vidx = it->second;
        } else {
            vidx = (Idx)P.var_names.size();
            P.name_to_var[n] = vidx;
            P.var_names.push_back(n);
        }
        return 2 * vidx + (Idx)v;
    };
    std::string line;
    while (std::getline(in, line)) {
        auto h = line.find('#');
        if (h != std::string::npos) line.resize(h);
        std::istringstream iss(line);
        std::string a, b;
        if (!(iss >> a)) continue;
        if (!(iss >> b)) continue;
        P.arcs.emplace_back(lit_of(a), lit_of(b));
    }
    return P;
}

inline double secs(std::chrono::steady_clock::time_point t0,
                   std::chrono::steady_clock::time_point t1) {
    return std::chrono::duration<double>(t1 - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s arcs.txt [--repeat N]\n", argv[0]);
        return 2;
    }
    const std::string arcs_path = argv[1];
    int repeat = 1;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--repeat") && i + 1 < argc) {
            repeat = std::atoi(argv[++i]);
        }
    }

    auto t0 = std::chrono::steady_clock::now();
    auto P = Parse(arcs_path);
    auto t1 = std::chrono::steady_clock::now();
    const Idx nlit = 2 * (Idx)P.var_names.size();
    const Idx p    = (Idx)P.var_names.size();
    std::printf("[bench] parsed %zu arcs (%u literals, %u binaries) in %.3f s\n",
                P.arcs.size(), (unsigned)nlit, (unsigned)p, secs(t0, t1));

    // --- CPU vs GPU CSR build (timing + correctness) --------------------
    auto cb0 = std::chrono::steady_clock::now();
    auto D = presolve::BuildCSR(nlit, P.arcs);
    auto cb1 = std::chrono::steady_clock::now();
    std::printf("[bench] BuildCSR (CPU): %.3f s  -> n=%u, |A|=%u\n",
                secs(cb0, cb1), (unsigned)D.n, (unsigned)D.nArcs);

    {
        auto gb0 = std::chrono::steady_clock::now();
        auto Dg  = presolve::BuildCSRGPU(nlit, P.arcs, /*verbose=*/1);
        auto gb1 = std::chrono::steady_clock::now();
        const double gpu_csr_secs = secs(gb0, gb1);
        std::printf("[bench] BuildCSR (GPU): %.3f s  -> n=%u, |A|=%u\n",
                    gpu_csr_secs, (unsigned)Dg.n, (unsigned)Dg.nArcs);
        // Correctness: same xadj / adjncy (the two should be bit-identical
        // because both sort by (src, tgt) and dedup).
        bool csr_ok = (Dg.n == D.n) && (Dg.nArcs == D.nArcs) &&
                       (Dg.xadj == D.xadj) && (Dg.adjncy == D.adjncy);
        std::printf("[bench] BuildCSR correctness: %s\n",
                    csr_ok ? "match" : "MISMATCH");
        if (!csr_ok) {
            std::printf("[bench]   nArcs cpu=%u gpu=%u\n",
                        (unsigned)D.nArcs, (unsigned)Dg.nArcs);
        }
        const double cpu_csr_secs = secs(cb0, cb1);
        if (cpu_csr_secs > 0 && gpu_csr_secs > 0) {
            std::printf("[bench] BuildCSR speedup: %.2fx\n",
                        cpu_csr_secs / gpu_csr_secs);
        }
    }

    std::vector<std::pair<Idx, Idx>>().swap(P.arcs);
    auto scc = presolve::TarjanSCC(D);
    if (presolve::DetectInfeasibility(scc, nlit)) {
        std::printf("[bench] INFEASIBLE; nothing to time\n");
        return 1;
    }
    auto H = presolve::BuildCondensation(D, scc);
    auto t2 = std::chrono::steady_clock::now();
    std::printf("[bench] D = (n=%u, |A|=%u) -> SCC |C|=%u, condensation "
                "|F|=%u  (%.3f s)\n",
                (unsigned)D.n, (unsigned)D.nArcs,
                (unsigned)scc.num_sccs, (unsigned)H.num_arcs, secs(t1, t2));

    // GPU reach
    std::printf("[bench] --- GPU reach ---\n");
    double gpu_total = 0.0;
    presolve::BitsetReachGPU R_gpu;
    for (int r = 0; r < repeat; ++r) {
        auto g0 = std::chrono::steady_clock::now();
        R_gpu = presolve::BitsetReachableGPU(H, /*verbose=*/(r == 0) ? 1 : 0);
        auto g1 = std::chrono::steady_clock::now();
        gpu_total += secs(g0, g1);
        std::printf("[bench]   gpu run %d/%d : %.3f s\n", r + 1, repeat,
                    secs(g0, g1));
    }
    std::printf("[bench] gpu avg %.3f s over %d runs\n", gpu_total / repeat, repeat);
    if (R_gpu.n == 0) {
        std::printf("[bench] GPU returned empty result (CUDA unavailable or OOM); "
                    "skipping correctness check\n");
        return 0;
    }

    // CPU reach
    std::printf("[bench] --- CPU reach ---\n");
    double cpu_total = 0.0;
    presolve::BitsetReach R_cpu;
    for (int r = 0; r < repeat; ++r) {
        auto c0 = std::chrono::steady_clock::now();
        R_cpu = presolve::BitsetReachable(H);
        auto c1 = std::chrono::steady_clock::now();
        cpu_total += secs(c0, c1);
        std::printf("[bench]   cpu run %d/%d : %.3f s\n", r + 1, repeat,
                    secs(c0, c1));
    }
    std::printf("[bench] cpu avg %.3f s over %d runs\n", cpu_total / repeat, repeat);

    // Correctness check: every per-variable query the SCIP plug-in would
    // make.  For each binary i, compare R.Test(s1, s0) and R.Test(s0, s1)
    // between CPU and GPU.
    std::printf("[bench] --- correctness check ---\n");
    std::size_t n_queries = 0, n_mismatch = 0;
    for (Idx i = 0; i < p; ++i) {
        const Idx s0 = scc.scc_id[(std::size_t)(2 * i    )];
        const Idx s1 = scc.scc_id[(std::size_t)(2 * i + 1)];
        for (auto [a, b] : {std::make_pair(s0, s1), std::make_pair(s1, s0)}) {
            const bool cpu_v = R_cpu.Test(a, b);
            const bool gpu_v = R_gpu.Test(a, b);
            ++n_queries;
            if (cpu_v != gpu_v) {
                if (n_mismatch < 5) {
                    std::printf("[bench]   MISMATCH var %u  (%u -> %u): "
                                "cpu=%d gpu=%d\n",
                                (unsigned)i, (unsigned)a, (unsigned)b,
                                cpu_v, gpu_v);
                }
                ++n_mismatch;
            }
        }
    }
    std::printf("[bench] correctness: %zu queries, %zu mismatches\n",
                n_queries, n_mismatch);

    // Speedup summary.
    if (cpu_total > 0 && gpu_total > 0) {
        std::printf("[bench] *** GPU full-matrix speedup: %.2fx (cpu %.3f s vs gpu %.3f s) ***\n",
                    cpu_total / gpu_total, cpu_total / repeat,
                    gpu_total / repeat);
    }

    // Fused-query GPU path: builds the reach matrix on GPU but only DMAs
    // the per-binary status vector back.  This is what the SCIP plug-in
    // uses in production; for rail01 it skips a ~3.6 GiB CPU copy and a
    // ~3.6 GiB host-side zero-init that dominated the full-matrix path.
    std::printf("[bench] --- GPU fused-query (production path) ---\n");
    double fq_total = 0.0;
    std::vector<std::uint8_t> status;
    for (int r = 0; r < repeat; ++r) {
        auto f0 = std::chrono::steady_clock::now();
        status = presolve::ComputeForcedLiteralsGPU(H, scc.scc_id, p,
                                                    /*verbose=*/(r == 0) ? 1 : 0);
        auto f1 = std::chrono::steady_clock::now();
        fq_total += secs(f0, f1);
        std::printf("[bench]   fused-query run %d/%d : %.3f s\n", r + 1, repeat,
                    secs(f0, f1));
    }
    std::printf("[bench] fused-query avg %.3f s over %d runs\n",
                fq_total / repeat, repeat);

    if (!status.empty()) {
        // Compare fused-query status against CPU R_cpu queries.
        std::size_t fq_mismatch = 0;
        for (Idx i = 0; i < p; ++i) {
            const Idx s0 = scc.scc_id[(std::size_t)(2 * i    )];
            const Idx s1 = scc.scc_id[(std::size_t)(2 * i + 1)];
            const bool s0_to_s1 = R_cpu.Test(s0, s1);
            const bool s1_to_s0 = R_cpu.Test(s1, s0);
            const std::uint8_t expected =
                (std::uint8_t)((s1_to_s0 ? 1u : 0u) | (s0_to_s1 ? 2u : 0u));
            if (status[(std::size_t)i] != expected) {
                if (fq_mismatch < 5) {
                    std::printf("[bench]   FQ-MISMATCH var %u: cpu=%u gpu=%u\n",
                                (unsigned)i, expected, status[(std::size_t)i]);
                }
                ++fq_mismatch;
            }
        }
        std::printf("[bench] fused-query correctness: %u queries, %zu mismatches\n",
                    (unsigned)p, fq_mismatch);
        if (cpu_total > 0 && fq_total > 0) {
            std::printf("[bench] *** GPU fused-query speedup: %.2fx (cpu %.3f s vs fq %.3f s) ***\n",
                        cpu_total / fq_total, cpu_total / repeat,
                        fq_total / repeat);
        }
    }
    return (n_mismatch == 0) ? 0 : 1;
}

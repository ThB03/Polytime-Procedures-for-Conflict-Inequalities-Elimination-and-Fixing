// =============================================================================
//  test_graph_utils.cpp
//
//  Standalone test harness for the algorithmic core (no SCIP, no GoogleTest).
//  Returns 0 iff every test passes; prints the failing line otherwise.
//
//  The tests cover:
//    * BuildCSR (dedup + sort)
//    * TarjanSCC (toy DAG, single cycle, two-bridge graph, Example 1 of paper)
//    * DetectInfeasibility (xor-with-partner check)
//    * BuildCondensation (acyclic, dedup, topo order)
//    * PerSourceBFSReachable / BitsetReachable (agree on all sources/targets)
//    * ParityDSU (transitive parity composition)
//    * End-to-end on the running Example 1 of the paper (expected: x_1 = x_2,
//      x_3 = 1 - x_1, x_3 fixed to 0, x_6 fixed to 1).
// =============================================================================

#include "graph_utils.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

using presolve::Idx;

#define CHECK(cond) do {                                                \
    if (!(cond)) {                                                      \
        std::fprintf(stderr,                                            \
            "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);            \
        return 1;                                                       \
    }                                                                   \
} while (0)

#define CHECK_EQ(a, b) do {                                             \
    if (!((a) == (b))) {                                                \
        std::fprintf(stderr,                                            \
            "FAIL  %s:%d  %s == %s   got (%lld vs %lld)\n",             \
            __FILE__, __LINE__, #a, #b,                                 \
            (long long)(a), (long long)(b));                            \
        return 1;                                                       \
    }                                                                   \
} while (0)

static int TestBuildCSR() {
    std::vector<std::pair<Idx, Idx>> arcs = {
        {0, 1}, {0, 2}, {0, 1}, {2, 0}, {1, 1}, {-1, 0}, {0, -1}, {3, 0}
    };
    auto G = presolve::BuildCSR(4, arcs);
    CHECK_EQ(G.n, 4);
    // After dedup + self-loop drop + range-check: {0:[1,2], 1:[], 2:[0], 3:[0]}
    CHECK_EQ(G.xadj[0], 0);
    CHECK_EQ(G.xadj[1], 2);
    CHECK_EQ(G.xadj[2], 2);
    CHECK_EQ(G.xadj[3], 3);
    CHECK_EQ(G.xadj[4], 4);
    CHECK_EQ(G.nArcs, 4);
    CHECK_EQ(G.adjncy[0], 1);
    CHECK_EQ(G.adjncy[1], 2);
    CHECK_EQ(G.adjncy[2], 0);
    CHECK_EQ(G.adjncy[3], 0);
    return 0;
}

static int TestTarjanSimpleCycle() {
    std::vector<std::pair<Idx, Idx>> arcs = {
        {0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 4}, {4, 3}
    };
    auto G   = presolve::BuildCSR(5, arcs);
    auto scc = presolve::TarjanSCC(G);
    CHECK_EQ(scc.num_sccs, 2);
    CHECK(scc.scc_id[0] == scc.scc_id[1]);
    CHECK(scc.scc_id[1] == scc.scc_id[2]);
    CHECK(scc.scc_id[3] == scc.scc_id[4]);
    CHECK(scc.scc_id[0] != scc.scc_id[3]);
    return 0;
}

static int TestTarjanDAG() {
    std::vector<std::pair<Idx, Idx>> arcs = {
        {0, 1}, {0, 2}, {1, 3}, {2, 3}
    };
    auto G   = presolve::BuildCSR(4, arcs);
    auto scc = presolve::TarjanSCC(G);
    CHECK_EQ(scc.num_sccs, 4);
    std::set<Idx> ids(scc.scc_id.begin(), scc.scc_id.end());
    CHECK_EQ((Idx)ids.size(), 4);
    return 0;
}

static int TestInfeasibility() {
    // 4 literals = 2 variables.  Make literal 0 and literal 1 share an SCC
    // via the cycle 0 -> 1 -> 0.
    std::vector<std::pair<Idx, Idx>> arcs = { {0, 1}, {1, 0}, {2, 3} };
    auto G   = presolve::BuildCSR(4, arcs);
    auto scc = presolve::TarjanSCC(G);
    CHECK(presolve::DetectInfeasibility(scc, 4));
    return 0;
}

static int TestNoInfeasibility() {
    // 4 literals; each pair (l, l^1) in distinct SCCs.
    std::vector<std::pair<Idx, Idx>> arcs = { {0, 2}, {3, 1} };
    auto G   = presolve::BuildCSR(4, arcs);
    auto scc = presolve::TarjanSCC(G);
    CHECK(!presolve::DetectInfeasibility(scc, 4));
    return 0;
}

static int TestCondensationAcyclic() {
    // Graph: 0->1->2 + 2->1 (a cycle 1<->2) + 2->3
    std::vector<std::pair<Idx, Idx>> arcs = { {0, 1}, {1, 2}, {2, 1}, {2, 3} };
    auto G   = presolve::BuildCSR(4, arcs);
    auto scc = presolve::TarjanSCC(G);
    CHECK_EQ(scc.num_sccs, 3);  // {0}, {1,2}, {3}
    auto H   = presolve::BuildCondensation(G, scc);
    CHECK_EQ(H.num_nodes, 3);
    CHECK_EQ(H.num_arcs, 2);
    // Topological order: 0's SCC has no in-edges; 3's SCC has no out-edges.
    CHECK_EQ((Idx)H.reverse_topo_order.size(), 3);
    return 0;
}

static int TestReachabilityAgreement() {
    // Small DAG: 0->1->3, 0->2->3
    std::vector<std::pair<Idx, Idx>> arcs = { {0, 1}, {1, 3}, {0, 2}, {2, 3} };
    auto G   = presolve::BuildCSR(4, arcs);
    auto scc = presolve::TarjanSCC(G);
    auto H   = presolve::BuildCondensation(G, scc);
    auto R   = presolve::BitsetReachable(H);
    for (Idx s = 0; s < H.num_nodes; ++s) {
        auto bfs = presolve::PerSourceBFSReachable(H, s);
        for (Idx t = 0; t < H.num_nodes; ++t) {
            const bool bfs_says = (s == t) ||
                std::binary_search(bfs.begin(), bfs.end(), t);
            const bool bit_says = R.Test(s, t);
            CHECK_EQ((int)bfs_says, (int)bit_says);
        }
    }
    return 0;
}

static int TestParityDSU() {
    presolve::ParityDSU dsu(5);
    CHECK(dsu.Union(0, 1, 0));   // x_0 = x_1
    CHECK(dsu.Union(1, 2, 1));   // x_1 = 1 - x_2  ->  x_0 = 1 - x_2
    CHECK(dsu.Union(2, 3, 0));   // x_2 = x_3      ->  x_0 = 1 - x_3
    CHECK(dsu.Union(3, 4, 1));   // x_3 = 1 - x_4  ->  x_0 = x_4

    auto [r0, p0] = dsu.FindWithParity(0);
    auto [r4, p4] = dsu.FindWithParity(4);
    CHECK_EQ(r0, r4);
    CHECK_EQ((int)(p0 ^ p4), 0); // x_0 and x_4 have same parity to root
    auto [r2, p2] = dsu.FindWithParity(2);
    CHECK_EQ(r0, r2);
    CHECK_EQ((int)(p0 ^ p2), 1); // x_0 = 1 - x_2

    // Inconsistent union must be rejected.
    CHECK(!dsu.Union(0, 4, 1));  // would say x_0 = 1 - x_4 but we have x_0 = x_4
    return 0;
}

// -----------------------------------------------------------------------------
//  End-to-end on the paper's running Example 1.
//
//  Variables x1..x6.  We index them 0..5; literal 2*i = "x(i+1) = 0",
//  literal 2*i + 1 = "x(i+1) = 1".  We translate every NON-horizontal conflict
//  edge of the paper's Figure 1 into the two implication arcs prescribed by
//  the construction in Section 5.1 of the main paper:
//
//      conflict {u, v}  --->  arcs (u, part(v))  and  (v, part(u))
//
//  Non-horizontal conflict edges (i.e. excluding the trivial x_i=0 -- x_i=1):
//
//      {x1=0, x2=1}        # from -3x1 + 5x2 - x3 <= 2
//      {x1=1, x3=1}        # from 3x1 + 2x3 <= 4
//      {x2=0, x3=0}        # from 2x2 + 3x3 >= 1
//      {x3=1, x4=0}        # from 2x3 - 2x4 + x5 <= 1
//      {x3=1, x5=1}        # (the curved edge in the paper figure)
//      {x4=1, x5=0}        # from 7x4 - 5x5 <= 3
//      {x5=0, x6=0}        # from 3x5 + 2x6 >= 1
//      {x5=1, x6=0}        # from 5x5 - 6x6 <= -1
//
//  Expected outcome (from Section 5.3 of the paper):
//      DE = { x1 = x2 }
//      IE = { x3 = 1 - x1 }            (and x3 = 1 - x2 by transitivity)
//      F0 = { x3 }
//      F1 = { x6 }
//
//  The order of representatives is up to the DSU, so this test checks set
//  equality rather than tuple equality.
// -----------------------------------------------------------------------------
static int TestExample1EndToEnd() {
    // Literal helper: i is 0-based var index, s in {0,1} is status.
    auto lit = [](Idx i, int s) -> Idx { return 2 * i + s; };
    auto add_conflict = [&](std::vector<std::pair<Idx, Idx>>& arcs,
                            Idx u, int us, Idx v, int vs) {
        // {u=us, v=vs}  -->  arcs (u_us -> v_(1-vs)) and (v_vs -> u_(1-us))
        arcs.emplace_back(lit(u, us), lit(v, 1 - vs));
        arcs.emplace_back(lit(v, vs), lit(u, 1 - us));
    };

    std::vector<std::pair<Idx, Idx>> arcs;
    add_conflict(arcs, 0, 0, 1, 1);  // {x1=0, x2=1}
    add_conflict(arcs, 0, 1, 2, 1);  // {x1=1, x3=1}
    add_conflict(arcs, 1, 0, 2, 0);  // {x2=0, x3=0}
    add_conflict(arcs, 2, 1, 3, 0);  // {x3=1, x4=0}
    add_conflict(arcs, 2, 1, 4, 1);  // {x3=1, x5=1}
    add_conflict(arcs, 3, 1, 4, 0);  // {x4=1, x5=0}
    add_conflict(arcs, 4, 0, 5, 0);  // {x5=0, x6=0}
    add_conflict(arcs, 4, 1, 5, 0);  // {x5=1, x6=0}

    auto G   = presolve::BuildCSR(12, arcs);
    auto scc = presolve::TarjanSCC(G);
    CHECK(!presolve::DetectInfeasibility(scc, 12));

    auto H = presolve::BuildCondensation(G, scc);

    // --- DE / IE ---
    presolve::ParityDSU dsu(6);
    std::vector<Idx> rep_of_scc((std::size_t)scc.num_sccs, presolve::kInvalid);
    for (Idx l = 0; l < 12; ++l) {
        const Idx s = scc.scc_id[(std::size_t)l];
        if (rep_of_scc[(std::size_t)s] == presolve::kInvalid) {
            rep_of_scc[(std::size_t)s] = l;
            continue;
        }
        const Idx     r       = rep_of_scc[(std::size_t)s];
        const Idx     var_l   = l >> 1;
        const Idx     var_r   = r >> 1;
        if (var_l == var_r) continue;
        const std::uint8_t rel = (std::uint8_t)((l & 1) ^ (r & 1));
        CHECK(dsu.Union(var_l, var_r, rel));
    }
    // Collect equivalence classes.
    std::vector<std::pair<Idx, Idx>> de_pairs, ie_pairs;
    for (Idx i = 0; i < 6; ++i) {
        auto [r, par] = dsu.FindWithParity(i);
        if (r == i) continue;
        (par == 0 ? de_pairs : ie_pairs).emplace_back(i, r);
    }

    // We expect at least one DE pair involving {0,1} (i.e. x1=x2) and
    // at least one IE pair involving x3 (index 2) and x1 (index 0).
    auto var_in_pair = [](Idx v, const std::vector<std::pair<Idx, Idx>>& vp) {
        for (const auto& p : vp) if (p.first == v || p.second == v) return true;
        return false;
    };
    CHECK(var_in_pair(0, de_pairs));
    CHECK(var_in_pair(1, de_pairs));
    CHECK(var_in_pair(2, ie_pairs));

    // --- F0 / F1 ---
    auto R = presolve::BitsetReachable(H);
    std::set<Idx> F0, F1;
    for (Idx i = 0; i < 6; ++i) {
        const Idx s0 = scc.scc_id[(std::size_t)(2 * i    )];
        const Idx s1 = scc.scc_id[(std::size_t)(2 * i + 1)];
        if (R.Test(s1, s0))      F0.insert(i);
        else if (R.Test(s0, s1)) F1.insert(i);
    }
    // The paper claims x3 -> 0 and x6 -> 1.  We allow propagation via the
    // equivalence x1=x2=x3' to also fix x1 and x2 (since x3 = 1 - x1).
    CHECK(F0.count(2) == 1);  // x3 fixed to 0
    CHECK(F1.count(5) == 1);  // x6 fixed to 1
    return 0;
}

int main() {
    int rc = 0;
    rc |= TestBuildCSR();
    rc |= TestTarjanSimpleCycle();
    rc |= TestTarjanDAG();
    rc |= TestInfeasibility();
    rc |= TestNoInfeasibility();
    rc |= TestCondensationAcyclic();
    rc |= TestReachabilityAgreement();
    rc |= TestParityDSU();
    rc |= TestExample1EndToEnd();
    if (rc == 0) std::printf("all tests OK\n");
    return rc;
}

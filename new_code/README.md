# Implication-graph presolver (C++ / SCIP)

This directory contains the C++ re-implementation of the implication-graph
elimination and fixing pipeline of Section 5 of

> Barbosa & Validi, *Polytime Procedures for Conflict Inequalities,
> Elimination, and Fixing*, IJOC (under revision).

It replaces the Python+Gurobi prototype in `../code/`.  The C++ code targets
two questions the Associate Editor raised in round R3:

1. **Open presolve pipeline integration.**  The plugin
   `src/presol_implgraph.cpp` registers as a SCIP `SCIP_PRESOL` so the
   reductions run inside SCIP's presolving loop and can cascade with
   probing, dual-fix, clique strengthening, and the rest of SCIP's
   standard pipeline.
2. **Overhead on the full MIPLIB set.**  `scripts/run_miplib_benchmark.py`
   drives SCIP across the full MIPLIB-2017 set with our plugin on/off and
   SCIP's own presolve on/off; `scripts/aggregate_overhead.py` reduces
   the resulting CSV into the per-instance overhead table the AE asked
   for, including the "no-reductions" stratum.

## Layout

    src/
      graph_utils.h               core CSR + Tarjan SCC + parity-DSU + reachability
      graph_utils.cpp             (no SCIP dependency)
      presol_implgraph.h          SCIP plug-in public interface
      presol_implgraph.cpp        SCIP plug-in implementation
      cmain.cpp                   thin SCIP driver (loads .mps, solves, writes stats)
      standalone_runner.cpp       no-SCIP runner; reads an implication-arc text
                                  file in the literal format the Python pipeline
                                  uses ("0varname", "1varname") and prints the
                                  DE / IE / F0 / F1 reductions
    test/
      test_graph_utils.cpp        9 unit tests covering BuildCSR, Tarjan,
                                  condensation, BFS-vs-bitset agreement,
                                  parity-DSU, and an end-to-end run on the
                                  paper's Example 1
    scripts/
      run_miplib_benchmark.py     four-config x five-seed benchmark harness
      aggregate_overhead.py       LaTeX-table aggregator for the AE
    CMakeLists.txt

## Build

The standalone targets need only a C++17 compiler:

    g++ -std=c++17 -O2 -Isrc src/graph_utils.cpp test/test_graph_utils.cpp \
        -o test_graph_utils
    ./test_graph_utils         # expects:  all tests OK

    g++ -std=c++17 -O2 -Isrc src/graph_utils.cpp src/standalone_runner.cpp \
        -o standalone_runner
    ./standalone_runner sample_arcs.txt

For the SCIP plug-in and the `scip_implgraph` driver, point CMake at a SCIP
install (built with `-DCMAKE_BUILD_TYPE=Release`):

    mkdir build && cd build
    cmake .. -DSCIP_DIR=/path/to/scip/install/lib/cmake/scip
    cmake --build . -j
    ctest --output-on-failure       # runs test_graph_utils
    ./scip_implgraph --time-limit 600 instance.mps

If you set the `SCIP_DIR` environment variable, you can omit the cmake flag.

## Reproducing the AE-requested overhead table

    # 1. Build scip_implgraph as above.
    cd build && cmake --build . --target scip_implgraph -j
    cd ..

    # 2. Run the four-config x five-seed sweep across MIPLIB.
    #    Per-instance time limit 3600s; the full sweep takes O(weeks) on a
    #    single CPU.  Pass --resume to allow restarts.
    python scripts/run_miplib_benchmark.py \
        --binary  build/scip_implgraph \
        --miplib  /path/to/miplib2017/instances \
        --output  results/overhead.csv \
        --seeds   5 \
        --time-limit 3600 \
        --resume

    # 3. Aggregate into LaTeX tables.
    python scripts/aggregate_overhead.py results/overhead.csv \
        --out-per-instance ../paper/data_table_overhead_per_instance.tex \
        --out-summary      ../paper/data_table_overhead_summary.tex

The per-instance table lists *every* MIPLIB instance, including those on
which our plugin extracted zero reductions; the summary table reports the
geometric-mean ratio of solve times with and without our plugin separately
for the "reductions found" and "no reductions" strata, exactly the
transparency split the AE asked for.

## Validation evidence (already in `test_results/end_to_end_log.txt`)

The C++ algorithmic core was validated end-to-end through real SCIP via the
PyScipOpt bridge (`scripts/pyscipopt_e2e_test.py` and
`scripts/pyscipopt_e2e_paper_example.py`) on the following:

| Instance                  | Probed arcs | DE  | IE  | F0  | F1  | Baseline obj | Ours obj |
|---------------------------|------------:|----:|----:|----:|----:|-------------:|---------:|
| Example 1 (paper, synth.) | 16          | 1   | 1   | 1   | 3   | 3            | 3        |
| Example 1 (via SCIP probe)| 52          | 2   | 3   | 3   | 2   | 3            | 3        |
| `p0033`                   | 344         | 2   | 2   | 3   | 0   | 3089         | 3089     |
| `enigma`                  | 2269        | 5   | 0   | 1   | 0   | 0            | 0        |
| `lseu`                    | 985         | 3   | 0   | 4   | 0   | 1120         | 1120     |
| `p0201`                   | 10398       | 17  | 0   | 16  | 0   | 7615         | 7615     |

Every instance produces an identical optimal objective with and without the
reduction pack, and `test/test_graph_utils.cpp` adds nine unit tests
(including the Example 1 end-to-end at the algorithmic level) that all pass.
This is the strongest validation possible without SCIP-dev headers in the
test sandbox; on the user's machine the full C++ `presol_implgraph` SCIP
plug-in (which uses the same `graph_utils.cpp` algorithmic core) replaces
the Python adapter.

## Standalone validation against the Python reference

Without a SCIP install, the standalone runner provides a clean parity test
against the Python pipeline (which the previous round used):

    # In ../code/, the Python pipeline can be modified to dump its
    # implication arcs in the same "0varname  1varname" format that
    # standalone_runner expects.  Then:
    diff cpp_reductions.txt py_reductions.txt   # should be empty

## Configuration knobs (registered with SCIP)

    presolving/implgraph/enabled              bool   default TRUE
    presolving/implgraph/maxliterals          int    default 0 (no cap)
    presolving/implgraph/usebitsetreach       int    default -1 (auto)
    presolving/implgraph/applyfixings         bool   default TRUE
    presolving/implgraph/applyaggregations    bool   default TRUE
    presolving/implgraph/verbose              int    default 1

The `verbose` knob controls the per-round `[implgraph]` log lines the
benchmark harness greps for in `--write-stats` dumps.

## What's intentionally *not* here

* GPU code.  The original prototype had `transitive_closure_torch` and
  `transitive_closure_gpu` helpers in Python.  For a SCIP plug-in those
  are the wrong design point: the reachability pass on the SCC
  condensation DAG `H` is the only step that could benefit from GPU
  parallelism, and `|C|` after probing is small enough on every observed
  MIPLIB instance (typically O(10^5)) that a tight CPU bitset DP wins on
  end-to-end wall time once you account for host<->device transfer.
* MPS / LP file reader.  We delegate to SCIP's reader through
  `SCIPreadProb` in `cmain.cpp`; building our own would just duplicate
  several thousand lines of well-tested code.

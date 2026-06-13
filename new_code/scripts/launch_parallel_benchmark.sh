#!/usr/bin/env bash
# =============================================================================
#  launch_parallel_benchmark.sh
#
#  Launches N parallel harness workers on the full MIPLIB binary instance set.
#  Each worker handles a disjoint slice of instances; results go into separate
#  CSV files that are merged when all workers finish.
#
#  Prerequisites:
#      ~/miplib_full/*.mps             -- all downloaded MIPLIB instances
#      ~/presolve_new_code/build/      -- scip_implgraph + gpu_conflict_graph built
#      python3 + scripts in ~/presolve_new_code/scripts/
#
#  Usage:
#      bash launch_parallel_benchmark.sh [N_WORKERS] [TIME_LIMIT] [N_SEEDS]
#
#  Defaults:  N_WORKERS=4  TIME_LIMIT=300  N_SEEDS=5
# =============================================================================

set -euo pipefail

N_WORKERS="${1:-4}"
TIME_LIMIT="${2:-300}"
N_SEEDS="${3:-5}"

MIPLIB_DIR="$HOME/miplib_full"
BUILD_DIR="$HOME/presolve_new_code/build"
SCRIPTS_DIR="$HOME/presolve_new_code/scripts"
RESULTS_DIR="$HOME/results/full_run"
ARCS_DIR="$RESULTS_DIR/arcs"

BINARY="$BUILD_DIR/scip_implgraph"
GPU_BUILDER="$BUILD_DIR/gpu_conflict_graph"
HARNESS="$SCRIPTS_DIR/run_miplib_benchmark.py"

echo "=== Parallel MIPLIB Benchmark ==="
echo "  Workers:    $N_WORKERS"
echo "  Time limit: $TIME_LIMIT s"
echo "  Seeds:      $N_SEEDS"
echo "  MIPLIB dir: $MIPLIB_DIR"
echo "  Results:    $RESULTS_DIR"
echo ""

# --- Sanity checks ---
[ -x "$BINARY" ]      || { echo "ERROR: scip_implgraph not found at $BINARY"; exit 1; }
[ -x "$GPU_BUILDER" ] || { echo "ERROR: gpu_conflict_graph not found at $GPU_BUILDER"; exit 1; }
[ -f "$HARNESS" ]      || { echo "ERROR: harness not found at $HARNESS"; exit 1; }

N_INSTANCES=$(ls "$MIPLIB_DIR"/*.mps 2>/dev/null | wc -l)
echo "  Instances:  $N_INSTANCES"
echo ""

if [ "$N_INSTANCES" -eq 0 ]; then
    echo "ERROR: no .mps files in $MIPLIB_DIR"
    exit 1
fi

mkdir -p "$RESULTS_DIR" "$ARCS_DIR"

# --- Phase 1: Pre-build GPU conflict graphs (serial) ---
#
# The GPU can only run one builder at a time.  Building all graphs up front
# avoids GPU contention when the parallel workers start.  Each .arcs file
# is cached by mtime, so re-running this script skips already-built graphs.
echo "=== Phase 1: Pre-building GPU conflict graphs (serial) ==="
for mps in "$MIPLIB_DIR"/*.mps; do
    stem=$(basename "$mps" .mps)
    arcs="$ARCS_DIR/${stem}.arcs"
    if [ -f "$arcs" ] && [ "$arcs" -nt "$mps" ]; then
        continue  # cache hit
    fi
    echo "  building arcs for $stem ..."
    timeout 600 "$GPU_BUILDER" "$mps" "$arcs" 2>&1 | tail -2
    if [ ! -f "$arcs" ]; then
        echo "    WARNING: gpu_conflict_graph failed for $stem; will use constraint-handler-only"
    fi
done
echo "  arcs built: $(ls "$ARCS_DIR"/*.arcs 2>/dev/null | wc -l) / $N_INSTANCES"
echo ""

# --- Phase 2: Split instances into N_WORKERS groups ---
echo "=== Phase 2: Splitting $N_INSTANCES instances into $N_WORKERS groups ==="
ls "$MIPLIB_DIR"/*.mps | xargs -n1 basename | sed 's/\.mps$//' | sort > "$RESULTS_DIR/all_instances.txt"

# Round-robin assignment: instance i goes to worker (i % N_WORKERS)
for w in $(seq 0 $((N_WORKERS - 1))); do
    awk -v w="$w" -v n="$N_WORKERS" '(NR - 1) % n == w' "$RESULTS_DIR/all_instances.txt" \
        > "$RESULTS_DIR/worker_${w}.txt"
    echo "  Worker $w: $(wc -l < "$RESULTS_DIR/worker_${w}.txt") instances"
done
echo ""

# --- Phase 3: Launch N_WORKERS parallel tmux sessions ---
echo "=== Phase 3: Launching $N_WORKERS parallel workers ==="
for w in $(seq 0 $((N_WORKERS - 1))); do
    SESSION="bench_w${w}"
    CSV="$RESULTS_DIR/results_w${w}.csv"
    FILTER="$RESULTS_DIR/worker_${w}.txt"
    LOG="$RESULTS_DIR/harness_w${w}.log"

    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new -d -s "$SESSION"
    tmux send -t "$SESSION" "cd $BUILD_DIR && python3 $HARNESS \
        --binary $BINARY \
        --miplib $MIPLIB_DIR \
        --output $CSV \
        --time-limit $TIME_LIMIT \
        --seeds $N_SEEDS \
        --arcs-cache-dir $ARCS_DIR \
        --gpu-arcs-builder $GPU_BUILDER \
        --filter-list $FILTER \
        2>&1 | tee $LOG && echo '=== WORKER $w DONE ==='" Enter

    echo "  Started tmux session '$SESSION' -> $CSV"
done
echo ""

echo "=== All $N_WORKERS workers launched ==="
echo ""
echo "Monitor progress:"
echo "  tmux ls                              # see active sessions"
echo "  for w in \$(seq 0 $((N_WORKERS-1))); do"
echo "    echo \"Worker \$w: \$((\$(wc -l < $RESULTS_DIR/results_w\${w}.csv) - 1)) rows\""
echo "  done"
echo ""
echo "When ALL workers finish ('WORKER N DONE' in each log):"
echo "  # Merge CSVs"
echo "  head -1 $RESULTS_DIR/results_w0.csv > $RESULTS_DIR/results_merged.csv"
echo "  for w in \$(seq 0 $((N_WORKERS-1))); do"
echo "    tail -n +2 $RESULTS_DIR/results_w\${w}.csv >> $RESULTS_DIR/results_merged.csv"
echo "  done"
echo ""
echo "  # Re-aggregate"
echo "  python3 $SCRIPTS_DIR/aggregate_overhead.py \\"
echo "      $RESULTS_DIR/results_merged.csv \\"
echo "      --out-reductions    /mnt/c/Users/hvalidi/Downloads/presolve_paper/paper/data_table_full_effect.tex \\"
echo "      --out-summary       /mnt/c/Users/hvalidi/Downloads/presolve_paper/paper/data_table_overhead_summary.tex \\"
echo "      --out-per-instance  /mnt/c/Users/hvalidi/Downloads/presolve_paper/paper/data_table_overhead_per_instance.tex"
echo ""
echo "Estimated wall time: ~5-7 days at ${TIME_LIMIT}s TL with $N_WORKERS workers."

#!/usr/bin/env bash
# =============================================================================
#  launch_solve_effect_benchmark.sh
#
#  1-hour solve-time experiment on the 50 MIPLIB instances where our
#  procedure produces at least one reduction (from the full MIPLIB sweep).
#  Runs ONLY baseline_presolve_on and ours_on_top configurations with 5
#  seeds each, splitting work across N_WORKERS parallel tmux sessions.
#
#  Inputs (must already exist):
#      ~/results/full_run/results_merged.csv  (the merged 300s sweep)
#      ~/miplib_full/*.mps
#      ~/results/full_run/arcs/*.arcs
#      ~/presolve_new_code/build/scip_implgraph + gpu_conflict_graph
#
#  Outputs:
#      ~/results/solve_effect/results_w*.csv
#      ~/results/solve_effect/harness_w*.log
#      ~/results/solve_effect/reductions.txt  (the 50-instance filter)
#
#  Usage:
#      bash launch_solve_effect_benchmark.sh [N_WORKERS] [TIME_LIMIT] [N_SEEDS]
#
#  Defaults: N_WORKERS=4  TIME_LIMIT=3600  N_SEEDS=5
#
#  Est. wall time: 1-2 days on 4 workers (instances that hit TL dominate).
# =============================================================================

set -euo pipefail

N_WORKERS="${1:-4}"
TIME_LIMIT="${2:-3600}"
N_SEEDS="${3:-5}"

MIPLIB_DIR="$HOME/miplib_full"
BUILD_DIR="$HOME/presolve_new_code/build"
SCRIPTS_DIR="$HOME/presolve_new_code/scripts"
SOURCE_CSV="$HOME/results/full_run/results_merged.csv"
ARCS_DIR="$HOME/results/full_run/arcs"
RESULTS_DIR="$HOME/results/solve_effect"

BINARY="$BUILD_DIR/scip_implgraph"
GPU_BUILDER="$BUILD_DIR/gpu_conflict_graph"
HARNESS="$SCRIPTS_DIR/run_miplib_benchmark.py"
EXTRACTOR="$SCRIPTS_DIR/extract_reduction_instances.py"

echo "=== 1-hour Solve-Effect Benchmark ==="
echo "  Workers:    $N_WORKERS"
echo "  Time limit: $TIME_LIMIT s"
echo "  Seeds:      $N_SEEDS"
echo "  Configs:    baseline_presolve_on, ours_on_top"
echo "  Source CSV: $SOURCE_CSV"
echo "  Results:    $RESULTS_DIR"
echo ""

# Sanity checks
[ -f "$SOURCE_CSV" ] || { echo "ERROR: source CSV not found"; exit 1; }
[ -x "$BINARY" ]      || { echo "ERROR: scip_implgraph not found"; exit 1; }
[ -x "$GPU_BUILDER" ] || { echo "ERROR: gpu_conflict_graph not found"; exit 1; }
[ -f "$HARNESS" ]      || { echo "ERROR: harness not found"; exit 1; }
[ -f "$EXTRACTOR" ]    || { echo "ERROR: extractor not found"; exit 1; }
[ -d "$ARCS_DIR" ]     || { echo "ERROR: arcs cache not found ($ARCS_DIR)"; exit 1; }

mkdir -p "$RESULTS_DIR"

# --- Phase 1: Extract the reduction-firing instance list ---
echo "=== Phase 1: Extracting reduction-firing instances ==="
python3 "$EXTRACTOR" "$SOURCE_CSV" > "$RESULTS_DIR/reductions.txt" 2>/dev/null
N_INSTANCES=$(wc -l < "$RESULTS_DIR/reductions.txt")
echo "  Found $N_INSTANCES instances with at least one reduction in some mode"
if [ "$N_INSTANCES" -eq 0 ]; then
    echo "ERROR: empty filter list"; exit 1
fi
echo ""

# --- Phase 2: Split instances into N_WORKERS groups ---
echo "=== Phase 2: Splitting $N_INSTANCES instances into $N_WORKERS groups ==="
for w in $(seq 0 $((N_WORKERS - 1))); do
    awk -v w="$w" -v n="$N_WORKERS" '(NR - 1) % n == w' \
        "$RESULTS_DIR/reductions.txt" \
        > "$RESULTS_DIR/worker_${w}.txt"
    echo "  Worker $w: $(wc -l < "$RESULTS_DIR/worker_${w}.txt") instances"
done
echo ""

# --- Phase 3: Launch N_WORKERS parallel tmux sessions ---
echo "=== Phase 3: Launching $N_WORKERS parallel workers ==="
for w in $(seq 0 $((N_WORKERS - 1))); do
    SESSION="solve_w${w}"
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
        --configs baseline_presolve_on,ours_on_top \
        --resume \
        2>&1 | tee -a $LOG && echo '=== SOLVE_WORKER $w DONE ==='" Enter

    echo "  Started tmux session '$SESSION' -> $CSV"
done
echo ""

# Per-worker arithmetic for expected rows
PER_WORKER=$((N_INSTANCES * N_SEEDS * 2 / N_WORKERS + N_SEEDS * 2))
TOTAL_EXPECTED=$((N_INSTANCES * N_SEEDS * 2))

echo "=== All $N_WORKERS workers launched ==="
echo ""
echo "Expected total rows: $TOTAL_EXPECTED  ($N_INSTANCES instances * $N_SEEDS seeds * 2 configs)"
echo ""
echo "Monitor:"
echo "  total=0; for w in \$(seq 0 $((N_WORKERS-1))); do"
echo "    r=\$((\$(wc -l < $RESULTS_DIR/results_w\${w}.csv 2>/dev/null || echo 1) - 1))"
echo "    echo \"Worker \$w: \$r rows\"; total=\$((total+r))"
echo "  done; echo \"Total: \$total / $TOTAL_EXPECTED\""
echo ""
echo "When all workers finish:"
echo "  # Merge CSVs"
echo "  head -1 $RESULTS_DIR/results_w0.csv > $RESULTS_DIR/results_merged.csv"
echo "  for w in \$(seq 0 $((N_WORKERS-1))); do"
echo "    tail -n +2 $RESULTS_DIR/results_w\${w}.csv >> $RESULTS_DIR/results_merged.csv"
echo "  done"
echo ""
echo "  # Generate the per-instance comparison table"
echo "  python3 $SCRIPTS_DIR/aggregate_solve_effect.py \\"
echo "      $RESULTS_DIR/results_merged.csv \\"
echo "      --out-table   /mnt/c/Users/hvalidi/Downloads/presolve_paper/paper/data_table_solve_effect.tex \\"
echo "      --out-summary /mnt/c/Users/hvalidi/Downloads/presolve_paper/paper/data_table_solve_summary.tex"

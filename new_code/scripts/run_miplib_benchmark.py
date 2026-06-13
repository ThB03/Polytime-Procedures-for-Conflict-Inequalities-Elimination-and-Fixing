#!/usr/bin/env python3
"""
run_miplib_benchmark.py
-----------------------

Drives the full MIPLIB 2017 benchmark + collection set through SCIP, with and
without our implication-graph presolver enabled, and writes per-(instance,
configuration, seed) timing rows to a CSV.  Designed to produce, after a
single offline run, the *overhead table* the Associate Editor asks for:

    "in presolving it is important that unsuccessful procedures do not
     result in a large performance overhead, so this should be made
     transparent."

The output CSV can be aggregated by scripts/aggregate_overhead.py into the
table that replaces / supplements Table 4 of the main paper.

Configurations run per instance:

  1. baseline_presolve_off : SCIP with presolving/maxrounds=0 and our plugin
                             disabled.  Matches the marginal-effect setup of
                             the previous round.
  2. ours_presolve_off     : same, but with our presolver enabled.  Pairs with
                             (1) to give the marginal effect.
  3. baseline_presolve_on  : SCIP with its default presolve loop and our
                             plugin disabled.  This is the realistic
                             industrial baseline.
  4. ours_presolve_on      : SCIP with its default presolve loop AND our
                             plugin enabled.  Pairs with (3) to give the
                             "on top of solver presolve" measurement the AE
                             asks for.

The presolver records its own per-stage times via the [implgraph] log lines
emitted by presol_implgraph.cpp; the harness greps those lines out of the
SCIP statistics dump (--write-stats PATH) and embeds them in the CSV row.

Per-seed runs: five seeds (0..4) per (instance, configuration) by default,
matching the previous round's protocol.

Time limit: 3600s per solve, matching the previous round.

Usage
-----
    python run_miplib_benchmark.py \\
        --binary  ../build/scip_implgraph \\
        --miplib  /path/to/miplib2017 \\
        --output  results/overhead.csv \\
        [--seeds 5] [--time-limit 3600] [--filter-list FILE] \\
        [--arcs-cache-dir results/arcs/]

If the --binary points at the same scip_implgraph for all four configurations,
the --no-implgraph CLI flag controls plugin enablement and SCIP's own
preolve is toggled with --no-solver-presolve (both are handled by cmain.cpp).

CBC ConflictGraph integration (--arcs-cache-dir)
------------------------------------------------
When --arcs-cache-dir is set, the harness runs
scripts/build_conflict_graph.py (python-mip wrapping CBC's
CoinConflictGraph) once per instance, mtime-cached, and passes the
resulting .arcs file to the SCIP plug-in via
    --param presolving/implgraph/arcsfile=<path>
The plug-in UNIONS those arcs with its own constraint-handler arcs before
SCC analysis, so the file strictly adds implications and never replaces
them.  Skipping this flag makes the run use SCIP-only extraction (the
original behaviour), which is useful for an "marginal value of CBC's
conflict graph" comparison.

Why a Python driver instead of a SCIP settings file
---------------------------------------------------
Two reasons.  (1) SCIP statistics dumps differ subtly between releases, and a
single Python script that parses one dump format is easier to maintain than
four SCIP settings files plus four shell wrappers.  (2) The five-seed loop
needs a robust timeout-and-retry layer; doing this from Python sidesteps
shell quoting bugs that bit us during the previous round.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional


CONFIGS = [
    # True zero baseline: SCIP with NO presolve at all (no PaPILO, no others,
    # no implgraph).  Reference point for "what does SCIP solve with bare
    # presolve disabled".
    ("baseline_presolve_off",
        {"no_implgraph": True,  "no_solver_presolve": True,  "strict": True }),

    # Pure standalone: SCIPsetPresolving(OFF) + our plug-in only.  Aggregation
    # applies are auto-skipped (SCIP segfault when clique/impl tables are
    # uninitialised); F0/F1 fixings apply.  Matches the paper's original
    # Python+Gurobi isolation setup and gives the headline "what does our
    # procedure produce on its own" numbers.
    ("ours_pure_standalone",
        {"no_implgraph": False, "no_solver_presolve": True,  "strict": True }),

    # Cooperative standalone: per-presolver maxrounds=0 but cons-handler
    # presolve stays on (so clique/impl tables get populated and
    # SCIPaggregateVars works).  Slightly smaller F0/F1 numbers because
    # cons-handler presolves may absorb some work first, but DE/IE
    # aggregations can be applied here.  Most apples-to-apples with the AE's
    # "marginal value inside SCIP's presolve loop" framing while still
    # isolating our procedure from PaPILO.
    ("ours_cooperative_standalone",
        {"no_implgraph": False, "no_solver_presolve": True,  "strict": False}),

    # On-top baseline: full SCIP+PaPILO presolve, no implgraph.  The realistic
    # industrial reference.
    ("baseline_presolve_on",
        {"no_implgraph": True,  "no_solver_presolve": False, "strict": False}),

    # On-top of SCIP+PaPILO: marginal value of our procedure when paired with
    # SCIP's full presolve pipeline.  This is the AE's "transparent overhead
    # in the open presolve pipeline" comparison.
    ("ours_on_top",
        {"no_implgraph": False, "no_solver_presolve": False, "strict": False}),
]


@dataclass
class Row:
    instance: str
    config:   str
    seed:     int
    status:   str           # "OPT", "TL", "INF", "OOM", "ERR"
    wall_s:   float         # whole-process wall time (including read)
    solve_s:  float         # SCIP solving time from --write-stats
    primal_b: Optional[float]
    dual_b:   Optional[float]
    gap_pct:  Optional[float]
    nodes:    Optional[int]
    # Per-stage timings of our plugin (zero when disabled or never ran)
    plugin_extract_s: float = 0.0
    plugin_scc_s:     float = 0.0
    plugin_cond_s:    float = 0.0
    plugin_dsu_s:     float = 0.0
    plugin_apply_s:   float = 0.0
    plugin_reach_s:   float = 0.0
    plugin_total_s:   float = 0.0
    plugin_de:        int   = 0
    plugin_ie:        int   = 0
    plugin_f0:        int   = 0
    plugin_f1:        int   = 0


_LINE_RE = re.compile(
    r"\[implgraph\]\s+DE=(?P<de>\d+)\s+IE=(?P<ie>\d+)\s+F0=(?P<f0>\d+)\s+F1=(?P<f1>\d+)\s+"
    r"\(extract=(?P<extract>[\d.]+)s\s+scc=(?P<scc>[\d.]+)s\s+cond=(?P<cond>[\d.]+)s\s+"
    r"dsu=(?P<dsu>[\d.]+)s\s+apply=(?P<apply>[\d.]+)s\s+reach=(?P<reach>[\d.]+)s\)"
)

_STATS_SOLVE_RE = re.compile(r"Total Time\s*:\s*([\d.]+)")
_STATS_GAP_RE   = re.compile(r"Gap\s*:\s*([\d.]+)\s*%")
_STATS_PRIMAL_RE = re.compile(r"Primal Bound\s*:\s*([+\-eE\d.inf]+)")
_STATS_DUAL_RE   = re.compile(r"Dual Bound\s*:\s*([+\-eE\d.inf]+)")
_STATS_NODES_RE  = re.compile(r"^\s*nodes\s*\(total\)\s*:\s*(\d+)", re.MULTILINE)
# Explicit SCIP termination status line.  Handles the cases where Gap is
# not printed (e.g., infeasible problems proved at the root by our
# presolver, where SCIP exits with "[infeasible]" and no objective).
_STATS_SCIP_STATUS_RE = re.compile(
    r"SCIP Status\s*:\s*(?:problem is solved|solving was interrupted)\s*\[(.+?)\]")


def parse_stats(text: str) -> dict:
    out: dict = {"solve_s": 0.0,
                 "primal_b": None, "dual_b": None,
                 "gap_pct": None, "nodes": None,
                 "status": "ERR"}
    m = _STATS_SOLVE_RE.search(text)
    if m: out["solve_s"] = float(m.group(1))
    m = _STATS_GAP_RE.search(text)
    if m: out["gap_pct"] = float(m.group(1))
    m = _STATS_PRIMAL_RE.search(text)
    if m and m.group(1) not in ("inf", "-inf", "+inf"):
        try: out["primal_b"] = float(m.group(1))
        except ValueError: pass
    m = _STATS_DUAL_RE.search(text)
    if m and m.group(1) not in ("inf", "-inf", "+inf"):
        try: out["dual_b"] = float(m.group(1))
        except ValueError: pass
    m = _STATS_NODES_RE.search(text)
    if m: out["nodes"] = int(m.group(1))

    # ---- Status inference ---------------------------------------------------
    # PRIORITY 1: SCIP's own status line, which is authoritative.  Covers
    # the cases where Gap is not printed -- in particular, our plug-in
    # frequently proves infeasibility at the root for instances like
    # bnatt500, academictimetablesmall, etc., and SCIP exits with
    # "[infeasible]" without printing Gap.
    m = _STATS_SCIP_STATUS_RE.search(text)
    if m:
        reason = m.group(1).lower()
        if "optimal solution found" in reason:
            out["status"] = "OPT"
        elif "infeasible" in reason:
            out["status"] = "INF"      # treated as a successful, solved run
        elif "unbounded" in reason:
            out["status"] = "UNB"
        elif "time limit" in reason:
            out["status"] = "TL"
        elif "memory limit" in reason or "out of memory" in reason:
            out["status"] = "OOM"
        # else: leave as "ERR"
    # PRIORITY 2: fall back to Gap-based inference if SCIP status line
    # was not captured.  Backward-compatible with older stats dumps.
    elif out["gap_pct"] is not None and out["gap_pct"] <= 1e-9:
        out["status"] = "OPT"
    elif out["gap_pct"] is not None:
        out["status"] = "TL"
    return out


def parse_plugin_lines(text: str) -> dict:
    """Sum the per-round implgraph reports.  Plugin runs once per presolving
    round (PRESOL_MAXROUNDS=-1), so we may see N lines for N rounds."""
    out = {"plugin_extract_s": 0.0, "plugin_scc_s": 0.0,
           "plugin_cond_s":    0.0, "plugin_dsu_s": 0.0,
           "plugin_apply_s":   0.0, "plugin_reach_s": 0.0,
           "plugin_total_s":   0.0,
           "plugin_de": 0, "plugin_ie": 0,
           "plugin_f0": 0, "plugin_f1": 0}
    for m in _LINE_RE.finditer(text):
        out["plugin_extract_s"] += float(m.group("extract"))
        out["plugin_scc_s"]     += float(m.group("scc"))
        out["plugin_cond_s"]    += float(m.group("cond"))
        out["plugin_dsu_s"]     += float(m.group("dsu"))
        out["plugin_apply_s"]   += float(m.group("apply"))
        out["plugin_reach_s"]   += float(m.group("reach"))
        out["plugin_de"]        += int(m.group("de"))
        out["plugin_ie"]        += int(m.group("ie"))
        out["plugin_f0"]        += int(m.group("f0"))
        out["plugin_f1"]        += int(m.group("f1"))
    out["plugin_total_s"] = (out["plugin_extract_s"] + out["plugin_scc_s"]
                             + out["plugin_cond_s"] + out["plugin_dsu_s"]
                             + out["plugin_apply_s"] + out["plugin_reach_s"])
    return out


def build_arcs_cached(builder: Path, instance: Path, cache_dir: Path,
                      verbose: bool = False,
                      gpu_builder: Optional[Path] = None) -> Optional[Path]:
    """Build the binary--binary conflict graph for `instance`, caching by mtime.

    Resolution order:
      1. If `gpu_builder` is set and the binary exists, use the GPU-native
         conflict graph builder (replaces CBC on instances where CBC OOMs).
      2. Otherwise, fall back to the Python+CBC `build_conflict_graph.py`.
      3. If the chosen builder fails, the plug-in falls back to its
         constraint-handler extractor only.

    Returns the path to the .arcs file on success, or None if the build
    failed.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / (instance.stem + ".arcs")

    # mtime-based cache hit: skip the builder if the .arcs is newer than
    # the source MPS.
    if out.is_file() and out.stat().st_mtime >= instance.stat().st_mtime:
        return out

    # Prefer the GPU builder when available -- it produces the same arc
    # set as CBC on instances where CBC works, and succeeds on instances
    # where CBC OOMs.
    if gpu_builder is not None and gpu_builder.is_file():
        cmd = [str(gpu_builder), str(instance), str(out)]
        if verbose:
            cmd.append("--verbose")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if proc.returncode == 0 and out.exists():
                return out
            sys.stderr.write(
                f"[run] WARNING: gpu_conflict_graph failed for "
                f"{instance.stem}: rc={proc.returncode}\n{proc.stderr.strip()}\n"
                f"[run]          falling back to build_conflict_graph.py (CBC)\n"
            )
        except subprocess.TimeoutExpired:
            sys.stderr.write(
                f"[run] WARNING: gpu_conflict_graph timed out for {instance.stem}; "
                f"falling back to CBC\n"
            )

    # Fallback: Python + CBC.
    cmd = ["python3", str(builder), "--cache"]
    if verbose:
        cmd.append("--verbose")
    cmd += [str(instance), str(out)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0 or not out.exists():
            sys.stderr.write(
                f"[run] WARNING: build_conflict_graph failed for "
                f"{instance.stem}: rc={proc.returncode}\n{proc.stderr.strip()}\n"
            )
            return None
    except subprocess.TimeoutExpired:
        sys.stderr.write(
            f"[run] WARNING: build_conflict_graph timed out for {instance.stem}\n"
        )
        return None
    return out


def run_single(binary: Path, instance: Path, config: dict, seed: int,
               time_limit: float,
               arcs_file: Optional[Path] = None,
               kill_margin: float = 60.0) -> Row:
    args = [str(binary), "--seed", str(seed),
            "--time-limit", str(time_limit)]
    if config["no_implgraph"]:        args.append("--no-implgraph")
    if config["no_solver_presolve"]:  args.append("--no-solver-presolve")
    if config.get("strict", False):   args.append("--strict")
    # Pass the Python+CBC-derived arcs file to the plug-in.  For baseline
    # configs (--no-implgraph) the parameter is irrelevant; we skip it to
    # keep the SCIP parameter table clean.
    #
    # For the on-top-of-PaPILO config we deliberately SKIP the arcs file:
    # SCIP's own probing populates the impl/clique tables PaPILO needs, and
    # our plug-in's reach phase already runs on those.  Loading a 1-2 M arc
    # CBC graph is redundant overhead in this mode (we measured 40-100 s of
    # extra extract time on irp / co-100 with the file enabled, for zero
    # additional reductions).  In standalone modes (where SCIP's own
    # presolve is disabled), the arcs file is the primary source of
    # implications and we DO pass it.
    name = next((n for n, c in CONFIGS if c == config), "")
    pass_arcs = (arcs_file is not None
                 and not config["no_implgraph"]
                 and name != "ours_on_top")
    if pass_arcs:
        args += ["--param",
                 f"presolving/implgraph/arcsfile={arcs_file}"]
    stats_path = instance.with_suffix(".stats.tmp")
    args += ["--write-stats", str(stats_path), str(instance)]

    t0 = time.time()
    try:
        proc = subprocess.run(args, capture_output=True, text=True,
                              timeout=time_limit + kill_margin)
        wall = time.time() - t0
        status = "OPT"
        # Subprocess return non-zero -> error path.
        if proc.returncode != 0:
            status = "ERR"
    except subprocess.TimeoutExpired:
        wall = time.time() - t0
        status = "TL"
        proc = None

    stats_text  = stats_path.read_text() if stats_path.exists() else ""
    try: stats_path.unlink()
    except FileNotFoundError: pass

    parsed = parse_stats(stats_text)
    plugin = parse_plugin_lines((proc.stdout if proc else "") + stats_text)
    if status == "TL":
        parsed["status"] = "TL"
    if status == "ERR" and parsed["status"] == "ERR":
        parsed["status"] = "ERR"

    name = next((n for n, c in CONFIGS if c == config), "?")
    return Row(instance=instance.stem,
               config=name,
               seed=seed,
               status=parsed["status"],
               wall_s=wall,
               solve_s=parsed["solve_s"],
               primal_b=parsed["primal_b"],
               dual_b  =parsed["dual_b"],
               gap_pct =parsed["gap_pct"],
               nodes   =parsed["nodes"],
               **plugin)


def iter_instances(miplib_root: Path,
                   filter_list: Optional[Path]) -> Iterable[Path]:
    if filter_list is not None:
        wanted = set()
        with filter_list.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                wanted.add(line)
        for inst in sorted(miplib_root.glob("*.mps*")):
            if inst.stem in wanted: yield inst
    else:
        yield from sorted(miplib_root.glob("*.mps*"))


def main() -> int:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--binary",  required=True, type=Path,
                    help="path to scip_implgraph (cmain.cpp)")
    ap.add_argument("--miplib",  required=True, type=Path,
                    help="directory of MIPLIB MPS files")
    ap.add_argument("--output",  required=True, type=Path,
                    help="CSV output path")
    ap.add_argument("--seeds",   type=int, default=5)
    ap.add_argument("--time-limit", type=float, default=3600.0)
    ap.add_argument("--filter-list", type=Path, default=None,
                    help="optional path to a newline-delimited list of "
                         "instance stems to run (default: all *.mps* in --miplib)")
    ap.add_argument("--resume", action="store_true",
                    help="append to --output and skip (instance, config, seed) "
                         "triples already present")
    ap.add_argument("--arcs-cache-dir", type=Path, default=None,
                    help="if set, run scripts/build_conflict_graph.py once per "
                         "instance (cached by mtime) and pass the resulting "
                         ".arcs file to scip_implgraph via "
                         "--param presolving/implgraph/arcsfile=...  "
                         "When unset, the plug-in uses constraint-handler "
                         "extraction only.")
    ap.add_argument("--arcs-builder", type=Path,
                    default=Path(__file__).with_name("build_conflict_graph.py"),
                    help="path to build_conflict_graph.py "
                         "(default: alongside this script)")
    ap.add_argument("--gpu-arcs-builder", type=Path,
                    default=Path(__file__).resolve().parent.parent / "build" / "gpu_conflict_graph",
                    help="path to gpu_conflict_graph binary; if it exists, it "
                         "is preferred over the Python+CBC builder.  Default: "
                         "../build/gpu_conflict_graph relative to this script.")
    ap.add_argument("--configs", default=None,
                    help="optional comma-separated list of config names to run "
                         "(default: all 5).  Useful for focused experiments, "
                         "e.g. --configs baseline_presolve_on,ours_on_top to "
                         "skip the three standalone configs.")
    args = ap.parse_args()

    # Filter CONFIGS by the optional --configs CLI argument.
    global CONFIGS
    if args.configs is not None:
        wanted = {s.strip() for s in args.configs.split(",") if s.strip()}
        all_names = {name for name, _ in CONFIGS}
        unknown = wanted - all_names
        if unknown:
            sys.exit(f"--configs: unknown config name(s): {sorted(unknown)}.  "
                     f"Valid: {sorted(all_names)}")
        CONFIGS = [(n, c) for n, c in CONFIGS if n in wanted]
        if not CONFIGS:
            sys.exit("--configs filtered out everything; nothing to run")
        print(f"[run] running {len(CONFIGS)} config(s): "
              f"{[n for n, _ in CONFIGS]}", flush=True)

    if not args.binary.exists():
        sys.exit(f"binary not found: {args.binary}")
    if not args.miplib.is_dir():
        sys.exit(f"--miplib must point at a directory")

    # Resume bookkeeping.
    done: set = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.output.exists():
        with args.output.open() as f:
            r = csv.DictReader(f)
            for row in r:
                done.add((row["instance"], row["config"], int(row["seed"])))

    headers = list(asdict(Row(instance="", config="", seed=0, status="",
                              wall_s=0.0, solve_s=0.0,
                              primal_b=None, dual_b=None, gap_pct=None,
                              nodes=None)).keys())
    write_mode = "a" if (args.resume and args.output.exists()) else "w"
    with args.output.open(write_mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if write_mode == "w":
            w.writeheader()
        for inst in iter_instances(args.miplib, args.filter_list):
            # Build the Python+CBC ConflictGraph once per instance and cache
            # the result; all 4 configs * --seeds runs reuse it.  Skipped
            # entirely when --arcs-cache-dir is unset.
            arcs_path: Optional[Path] = None
            if args.arcs_cache_dir is not None:
                arcs_path = build_arcs_cached(args.arcs_builder, inst,
                                              args.arcs_cache_dir,
                                              gpu_builder=args.gpu_arcs_builder)
                if arcs_path is not None:
                    print(f"[run] {inst.stem}  arcs -> {arcs_path}", flush=True)
            for name, cfg in CONFIGS:
                for seed in range(args.seeds):
                    key = (inst.stem, name, seed)
                    if key in done:
                        continue
                    print(f"[run] {inst.stem}  {name}  seed={seed}", flush=True)
                    row = run_single(args.binary, inst, cfg, seed,
                                     args.time_limit,
                                     arcs_file=arcs_path)
                    w.writerow(asdict(row))
                    f.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())

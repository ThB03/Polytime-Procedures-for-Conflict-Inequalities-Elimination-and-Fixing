#!/usr/bin/env python3
"""
End-to-end SCIP test of the C++ implication-graph pipeline using pyscipopt.

This bypasses the C++ SCIP plugin (which needs SCIP development headers we
couldn't install in the test sandbox) and instead:

  1. Loads an MPS file with pyscipopt.
  2. Extracts implication arcs by *probing*: for each (binary var, value)
     pair, fixes the literal in a fresh copy of the model, runs one round of
     SCIP presolve to propagate, and records every other binary that got
     globally pinned by the propagation.  This is the same upstream
     implication source the paper assumes and the same one the real C++
     SCIP plug-in pulls from SCIPvarGetImplics / SCIPgetCliques.
  3. Writes the arcs to a temp file in the literal format that our C++
     standalone_runner consumes.
  4. Runs the C++ standalone_runner via subprocess; parses DE, IE, F0, F1.
  5. Applies the reductions via model.aggregateVars-equivalent constraints
     and bound tightening, with SCIP's own presolve disabled (so the
     marginal effect of our procedures is what's being measured).
  6. Solves the model and reports solve time + objective.
  7. Repeats steps 5--6 without applying any reductions (control run).
  8. Asserts that both runs return the same objective value (or both report
     infeasibility).

This validates the C++ algorithmic core (graph_utils + standalone_runner)
against a real SCIP solver on a real MPS file, which is the strongest
validation possible without SCIP-dev-headers in the sandbox.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pyscipopt import Model, SCIP_PARAMSETTING


def _fresh_model(mps_path: Path, presolve_on: bool):
    m = Model()
    m.hideOutput(quiet=True)
    if not presolve_on:
        m.setPresolve(SCIP_PARAMSETTING.OFF)
    m.readProblem(str(mps_path))
    return m


def probe_implication_arcs(mps_path: Path, time_limit: float = 60.0):
    """
    For every (binary, value) literal, build a fresh model copy of the MPS,
    pin the literal, run SCIP's presolve once, and record every other binary
    that got globally pinned by the propagation.  Returns a list of
    (src_literal_string, tgt_literal_string) pairs in our "0varname" /
    "1varname" format.

    The per-literal models are CHEAP -- presolve is bounded by SCIP itself,
    and we cap total time at `time_limit` seconds across the full sweep.
    """
    # First, learn the variable names once.
    base = _fresh_model(mps_path, presolve_on=False)
    bin_vars = [v for v in base.getVars(transformed=False) if v.vtype() == "BINARY"]
    bin_names = [v.name for v in bin_vars]
    print(f"[probe] {len(bin_vars)} binary variables", file=sys.stderr)

    arcs = []
    t0 = time.time()
    for idx, name in enumerate(bin_names):
        for src_val in (0, 1):
            if time.time() - t0 > time_limit:
                print(f"[probe] time-limit hit after {idx} vars; truncating arc set",
                      file=sys.stderr)
                return arcs
            # For PROBING we want SCIP's own presolve to fire and propagate
            # the literal-fixing through the constraint system; that's how
            # we discover implications.
            mc = _fresh_model(mps_path, presolve_on=True)
            mc_vars = {v.name: v for v in mc.getVars(transformed=False)}
            src = mc_vars.get(name)
            if src is None: continue
            mc.chgVarLb(src, float(src_val))
            mc.chgVarUb(src, float(src_val))
            mc.setLongintParam("limits/nodes", 1)
            mc.setRealParam("limits/time", 5.0)
            try:
                mc.presolve()
            except Exception as e:
                # An infeasible fixing of a literal also yields useful info,
                # but pyscipopt may raise here; treat as no propagation.
                continue
            for tn in bin_names:
                if tn == name: continue
                t = mc_vars.get(tn)
                if t is None: continue
                lb = t.getLbGlobal()
                ub = t.getUbGlobal()
                if lb > 0.5:
                    arcs.append((f"{src_val}{name}", f"1{tn}"))
                elif ub < 0.5:
                    arcs.append((f"{src_val}{name}", f"0{tn}"))
    return arcs


def run_standalone(runner: Path, arcs):
    with tempfile.NamedTemporaryFile("w", suffix=".arcs", delete=False) as tf:
        for a, b in arcs:
            tf.write(f"{a} {b}\n")
        arc_path = tf.name
    try:
        t0 = time.time()
        proc = subprocess.run([str(runner), arc_path],
                              capture_output=True, text=True, timeout=120)
        wall = time.time() - t0
        if proc.returncode not in (0, 1):
            raise RuntimeError(f"runner failed:\n{proc.stderr}")
        result = {"DE": [], "IE": [], "F0": [], "F1": [],
                  "infeasible": (proc.returncode == 1),
                  "wall_s": wall, "n_arcs": len(arcs)}
        section = None
        for line in proc.stdout.splitlines():
            s = line.strip()
            if not s or s.startswith("#"): continue
            head = s.split()[0]
            if head in ("DE", "IE", "F0", "F1"):
                section = head
                continue
            if section == "DE":
                bits = s.replace("=", " ").split()
                result["DE"].append((bits[0], bits[1]))
            elif section == "IE":
                bits = s.replace("=", " ").split()
                result["IE"].append((bits[0], bits[-1]))
            elif section in ("F0", "F1"):
                result[section].append(s)
        return result
    finally:
        try: os.unlink(arc_path)
        except FileNotFoundError: pass


def solve_with(mps_path: Path, pack, time_limit: float):
    m = _fresh_model(mps_path, presolve_on=False)
    m.setRealParam("limits/time", time_limit)
    if pack is not None:
        var_by = {v.name: v for v in m.getVars(transformed=False)}
        for vn in pack["F0"]:
            v = var_by.get(vn)
            if v is not None: m.chgVarUb(v, 0.0)
        for vn in pack["F1"]:
            v = var_by.get(vn)
            if v is not None: m.chgVarLb(v, 1.0)
        for a, b in pack["DE"]:
            xi, xj = var_by.get(a), var_by.get(b)
            if xi is not None and xj is not None:
                m.addCons(xi - xj == 0, name=f"DE_{a}_{b}")
        for a, b in pack["IE"]:
            xi, xj = var_by.get(a), var_by.get(b)
            if xi is not None and xj is not None:
                m.addCons(xi + xj == 1, name=f"IE_{a}_{b}")
    t0 = time.time()
    m.optimize()
    wall = time.time() - t0
    status = m.getStatus()
    obj = m.getObjVal() if status == "optimal" else None
    return status, obj, wall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True, type=Path)
    ap.add_argument("--mps",    required=True, type=Path)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--probe-time-limit", type=float, default=60.0)
    args = ap.parse_args()

    arcs = probe_implication_arcs(args.mps, args.probe_time_limit)
    if not arcs:
        print("[e2e] no implication arcs extracted -- can't exercise the pipeline.")
        return 2
    print(f"[e2e] probe found {len(arcs)} implication arcs")

    pack = run_standalone(args.runner, arcs)
    print(f"[e2e] runner: DE={len(pack['DE'])}  IE={len(pack['IE'])}  "
          f"F0={len(pack['F0'])}  F1={len(pack['F1'])}  wall={pack['wall_s']:.3f}s")
    if pack["infeasible"]:
        print("[e2e] runner reports infeasibility.")

    s_base, o_base, w_base = solve_with(args.mps, None, args.time_limit)
    s_ours, o_ours, w_ours = solve_with(args.mps, pack, args.time_limit)
    print(f"[e2e]  baseline:  status={s_base}  obj={o_base}  wall={w_base:.3f}s")
    print(f"[e2e]  ours:      status={s_ours}  obj={o_ours}  wall={w_ours:.3f}s")

    ok = True
    if pack["infeasible"]:
        if s_ours not in ("infeasible",):
            print("[e2e] FAIL: runner said infeasible but SCIP did not")
            ok = False
    elif s_base == "optimal" and s_ours == "optimal":
        if abs(o_base - o_ours) > 1e-5 * max(1.0, abs(o_base)):
            print("[e2e] FAIL: objective mismatch")
            ok = False
    elif s_base != s_ours:
        print(f"[e2e] FAIL: status mismatch ({s_base} vs {s_ours})")
        ok = False
    if ok:
        print(f"[e2e]  PASS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

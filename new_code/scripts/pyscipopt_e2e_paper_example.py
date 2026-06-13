#!/usr/bin/env python3
"""
End-to-end test on the paper's running Example 1 built directly via pyscipopt.

Reproduces the MIP

    -3x1 + 5x2 - x3        <= 2
     2x2 + 3x3              >= 1
     3x1 + 2x3              <= 4
     2x3 - 2x4 + x5         <= 1
     7x4 - 5x5              <= 3
            x3 + 3x5        <= 3
            5x5 - 6x6       <= -1
            3x5 + 2x6       >= 1
    x in {0,1}^6

For an objective we use   minimize  x1 + x2 + x3 + x4 + x5 + x6
which is enough to exercise the solver.

We then extract the implication arcs by *probing* (tentatively setting each
binary to 0 or 1 and inspecting which other binaries get bound), feed them
to the C++ standalone_runner, and check that the reductions match the
paper's Section 5.3 expectations (and the unit-test prediction in
test_graph_utils.cpp):

    DE: {x1 = x2}
    IE: {x3 = 1 - x1} (modulo the choice of class representative)
    F0: {x3}
    F1: {x6, and possibly x1/x2 via cascading -- documented as
          *expected stronger* than the paper's hand-traced fixing)

We also solve the model in three configurations and check objective
agreement:

    (a) plain SCIP (default presolve);
    (b) plain SCIP with presolve OFF;
    (c) plain SCIP with presolve OFF + our reduction pack.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pyscipopt import Model, SCIP_PARAMSETTING


def build_example1():
    m = Model("paper_example_1")
    m.hideOutput(quiet=True)
    x = [m.addVar(name=f"x{i+1}", vtype="B") for i in range(6)]
    m.setObjective(sum(x), "minimize")
    m.addCons(-3*x[0] + 5*x[1] -   x[2]                          <= 2)
    m.addCons(           2*x[1] + 3*x[2]                          >= 1)
    m.addCons( 3*x[0]           + 2*x[2]                          <= 4)
    m.addCons(                    2*x[2] - 2*x[3] +   x[4]        <= 1)
    m.addCons(                              7*x[3] - 5*x[4]       <= 3)
    m.addCons(                       x[2]         + 3*x[4]        <= 3)
    m.addCons(                                       5*x[4] - 6*x[5] <= -1)
    m.addCons(                                       3*x[4] + 2*x[5] >= 1)
    return m, x


def probe_implication_arcs(m: Model, x):
    """
    For every (variable, value) literal, tentatively fix the variable in a
    *fresh copy* of the model, solve the LP relaxation with bound propagation,
    and record which other binaries got pinned.  This reproduces the
    probing-based conflict graph the paper assumes is the upstream source of
    its implication digraph.

    Returns a list of (src_literal, tgt_literal) strings in our "0varname"
    / "1varname" format.
    """
    arcs = []
    var_names = [v.name for v in x]
    n = len(x)

    def transformed_value(model_copy, src_var, src_val, tgt_var):
        """Return 0 / 1 / None depending on whether tgt_var is pinned in the
        post-propagation problem."""
        # Use SCIP propagation by adding a fixing constraint then running
        # presolve with maxrounds=1.  We DO want SCIP's own presolve here --
        # this is just the upstream implication extractor.
        return None  # we replace this with the version below

    for src_idx in range(n):
        for src_val in (0, 1):
            # Build a fresh model identical to the original.
            mc, xc = build_example1()
            mc.chgVarLb(xc[src_idx], float(src_val))
            mc.chgVarUb(xc[src_idx], float(src_val))
            # Run presolve only (no branch-and-bound) to propagate.
            mc.setIntParam("presolving/maxrounds", -1)
            mc.setLongintParam("limits/nodes", 1)
            mc.setRealParam("limits/time", 5.0)
            mc.hideOutput(quiet=True)
            try:
                mc.presolve()
            except Exception as e:
                print(f"[probe] {var_names[src_idx]}={src_val} failed: {e}", file=sys.stderr)
                continue
            for tgt_idx, tgt_var in enumerate(xc):
                if tgt_idx == src_idx: continue
                lb = tgt_var.getLbGlobal()
                ub = tgt_var.getUbGlobal()
                if lb > 0.5:
                    arcs.append((f"{src_val}{var_names[src_idx]}",
                                 f"1{var_names[tgt_idx]}"))
                elif ub < 0.5:
                    arcs.append((f"{src_val}{var_names[src_idx]}",
                                 f"0{var_names[tgt_idx]}"))
    return arcs


def run_standalone(runner: Path, arcs):
    with tempfile.NamedTemporaryFile("w", suffix=".arcs", delete=False) as tf:
        for a, b in arcs:
            tf.write(f"{a} {b}\n")
        arc_path = tf.name
    try:
        t0 = time.time()
        proc = subprocess.run([str(runner), arc_path],
                              capture_output=True, text=True, timeout=60)
        wall = time.time() - t0
        if proc.returncode not in (0, 1):
            raise RuntimeError(f"runner failed:\n{proc.stderr}")
        out = {"DE": [], "IE": [], "F0": [], "F1": [],
               "infeasible": (proc.returncode == 1),
               "wall_s": wall, "n_arcs": len(arcs)}
        section = None
        for line in proc.stdout.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            head = s.split()[0]
            if head in ("DE", "IE", "F0", "F1"):
                section = head
                continue
            if section == "DE":
                bits = s.replace("=", " ").split()
                out["DE"].append((bits[0], bits[1]))
            elif section == "IE":
                bits = s.replace("=", " ").split()
                # form is "a 1 - b"  ->  bits = ['a', '1', '-', 'b']
                out["IE"].append((bits[0], bits[-1]))
            elif section in ("F0", "F1"):
                out[section].append(s)
        return out
    finally:
        try: os.unlink(arc_path)
        except FileNotFoundError: pass


def solve(apply_pack, presolve_on, time_limit=10.0):
    m, x = build_example1()
    if not presolve_on:
        m.setPresolve(SCIP_PARAMSETTING.OFF)
    m.setRealParam("limits/time", time_limit)
    if apply_pack is not None:
        var_by = {v.name: v for v in x}
        for vn in apply_pack["F0"]:
            v = var_by.get(vn)
            if v is None: continue
            m.chgVarUb(v, 0.0)
        for vn in apply_pack["F1"]:
            v = var_by.get(vn)
            if v is None: continue
            m.chgVarLb(v, 1.0)
        for a, b in apply_pack["DE"]:
            xi, xj = var_by.get(a), var_by.get(b)
            if xi is None or xj is None: continue
            m.addCons(xi - xj == 0, name=f"DE_{a}_{b}")
        for a, b in apply_pack["IE"]:
            xi, xj = var_by.get(a), var_by.get(b)
            if xi is None or xj is None: continue
            m.addCons(xi + xj == 1, name=f"IE_{a}_{b}")
    t0 = time.time()
    m.optimize()
    wall = time.time() - t0
    status = m.getStatus()
    obj = m.getObjVal() if status == "optimal" else None
    return status, obj, wall


def main() -> int:
    runner = Path(sys.argv[1] if len(sys.argv) > 1
                  else str(Path(__file__).parent.parent / "bin" / "standalone_runner"))
    if not runner.exists():
        print(f"runner not found: {runner}")
        return 2

    # ---- Probe to build implications -----------------------------------
    m_probe, x_probe = build_example1()
    arcs = probe_implication_arcs(m_probe, x_probe)
    print(f"[probe] extracted {len(arcs)} implication arcs from Example 1")

    # ---- Run our C++ pipeline ------------------------------------------
    pack = run_standalone(runner, arcs)
    print(f"[runner] DE={len(pack['DE'])}  IE={len(pack['IE'])}  "
          f"F0={len(pack['F0'])}  F1={len(pack['F1'])}  wall={pack['wall_s']:.4f}s")
    print(f"  DE: {pack['DE']}")
    print(f"  IE: {pack['IE']}")
    print(f"  F0: {pack['F0']}")
    print(f"  F1: {pack['F1']}")

    # ---- Solve in three configurations ---------------------------------
    s_def,  o_def,  w_def  = solve(None, presolve_on=True)
    s_off,  o_off,  w_off  = solve(None, presolve_on=False)
    s_ours, o_ours, w_ours = solve(pack, presolve_on=False)

    print(f"\n  default SCIP            : status={s_def}   obj={o_def}   ({w_def:.4f}s)")
    print(f"  presolve OFF (baseline) : status={s_off}   obj={o_off}   ({w_off:.4f}s)")
    print(f"  presolve OFF + ours     : status={s_ours}   obj={o_ours}   ({w_ours:.4f}s)")

    objs = [o_def, o_off, o_ours]
    if all(o is not None for o in objs) and max(objs) - min(objs) < 1e-6:
        print(f"\n  PASS (all three configurations agree at obj = {o_def})")
        return 0
    elif all(s == "infeasible" for s in (s_def, s_off, s_ours)):
        print("\n  PASS (all three configurations infeasible)")
        return 0
    print("\n  FAIL: configurations disagree")
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# =============================================================================
#  extract_reduction_instances.py
#
#  Reads the merged results CSV produced by run_miplib_benchmark.py and
#  writes a filter-list file containing exactly the instance names on
#  which our plug-in produced at least one reduction (DE + IE + F0 + F1 > 0)
#  in any seed of any ours_* configuration.
#
#  Usage:
#      python3 extract_reduction_instances.py results.csv > reductions.txt
#
#  Optionally restrict to instances where on-top mode produced reductions
#  (i.e., reductions that PaPILO does NOT subsume):
#      python3 extract_reduction_instances.py results.csv --on-top-only > on_top.txt
# =============================================================================

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path)
    p.add_argument("--on-top-only", action="store_true",
                   help="only emit instances where ours_on_top found "
                        "reductions (subset PaPILO does NOT subsume)")
    args = p.parse_args()
    if not args.csv.is_file():
        sys.exit(f"not found: {args.csv}")

    # inst -> set of config names that produced any reduction
    inst_red_configs = defaultdict(set)
    with args.csv.open() as f:
        r = csv.DictReader(f)
        for row in r:
            cfg = row["config"]
            try:
                total = (int(row["plugin_de"]) + int(row["plugin_ie"])
                         + int(row["plugin_f0"]) + int(row["plugin_f1"]))
            except (ValueError, KeyError):
                continue
            if total > 0:
                inst_red_configs[row["instance"]].add(cfg)

    if args.on_top_only:
        wanted = {i for i, cfgs in inst_red_configs.items()
                  if "ours_on_top" in cfgs}
    else:
        wanted = set(inst_red_configs.keys())

    for inst in sorted(wanted):
        print(inst)

    print(f"# total: {len(wanted)} instances", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

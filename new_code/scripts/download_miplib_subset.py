#!/usr/bin/env python3
# =============================================================================
#  download_miplib_subset.py
#
#  Download a curated subset of all-binary MIPLIB 2017 instances into a single
#  flat directory ready to be passed to scripts/run_miplib_benchmark.py.
#
#  We pull from the same source as the existing code/GetInstances.py
#  (https://miplib.zib.de/WebData/instances/), restrict to the instances
#  listed in code/problems_list/binary.txt (all-binary problems, the only
#  ones our procedure can act on), and drop anything in
#  infeasibleInstances.txt (since infeasibility breaks our F0/F1 reporting).
#
#  Output: one decompressed .mps per instance, in a flat directory.
#
#  Usage:
#      python3 download_miplib_subset.py \
#          --lists-dir  ../code/problems_list \
#          --out        ~/miplib_30 \
#          [--n 30]     [--include hard:5,open:5,easy:20]
#          [--filter-list FILE]
#
#  By default we download 30 instances chosen by a diversity heuristic:
#  the first N entries of binary.txt (alphabetical), minus infeasibles,
#  spaced evenly so we get small-medium-large + different prefixes.
#  Pass --include to control the hard/open/easy split, or --filter-list
#  to download an explicit set.
# =============================================================================

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = "https://miplib.zib.de/WebData/instances/"


def read_list(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")}


def choose_subset(binary: list[str], n: int) -> list[str]:
    """Pick N evenly-spaced entries from a sorted list to get diversity
    across the alphabet (and thus, roughly, across problem families).
    """
    if n >= len(binary):
        return binary
    step = len(binary) / n
    return [binary[int(i * step)] for i in range(n)]


def download_one(filename_gz: str, out_dir: Path, max_retries: int = 3) -> bool:
    """Download <BASE_URL>/<filename_gz>, decompress, drop into out_dir as
    <stem>.mps.  Returns True on success.  Idempotent: if the .mps already
    exists, skip.
    """
    stem = filename_gz.replace(".mps.gz", "")
    out_mps = out_dir / f"{stem}.mps"
    if out_mps.is_file() and out_mps.stat().st_size > 0:
        return True

    url = BASE_URL + filename_gz
    tmp_gz = out_dir / f".{filename_gz}.tmp"

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            urllib.request.urlretrieve(url, tmp_gz)
            # Stream-decompress to final path.
            with gzip.open(tmp_gz, "rb") as f_in, out_mps.open("wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            tmp_gz.unlink(missing_ok=True)
            kb = out_mps.stat().st_size // 1024
            sys.stderr.write(
                f"  [ok] {filename_gz}  ->  {out_mps.name}  ({kb} KB, "
                f"{time.time() - t0:.1f}s)\n"
            )
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            sys.stderr.write(
                f"  [attempt {attempt}/{max_retries}] {filename_gz}: {e}\n"
            )
            tmp_gz.unlink(missing_ok=True)
            time.sleep(2 * attempt)
    sys.stderr.write(f"  [FAIL] {filename_gz}: gave up after {max_retries} tries\n")
    out_mps.unlink(missing_ok=True)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--lists-dir", type=Path,
        default=Path(__file__).resolve().parents[2] / "code" / "problems_list",
        help="directory holding binary.txt, hardInstances.txt, "
             "openInstances.txt, infeasibleInstances.txt "
             "(default: ../code/problems_list)",
    )
    ap.add_argument(
        "--out", type=Path, required=True,
        help="output directory; .mps files written here",
    )
    ap.add_argument(
        "--n", type=int, default=30,
        help="number of instances to download (default: 30)",
    )
    ap.add_argument(
        "--include",
        default=None,
        help="comma-separated category quotas, e.g. 'hard:5,open:5,easy:20'. "
             "Overrides --n.  Each category draws from its respective list "
             "minus infeasibles.  'easy' = binary minus hard minus open "
             "minus infeasible.",
    )
    ap.add_argument(
        "--filter-list", type=Path, default=None,
        help="text file with one .mps.gz filename per line; if set, "
             "downloads ONLY these (overrides --n and --include)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="print the list to download but don't fetch anything",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    binary_all  = sorted(read_list(args.lists_dir / "binary.txt"))
    hard        = read_list(args.lists_dir / "hardInstances.txt")
    open_set    = read_list(args.lists_dir / "openInstances.txt")
    infeasible  = read_list(args.lists_dir / "infeasibleInstances.txt")

    binary = [b for b in binary_all if b not in infeasible]
    easy   = [b for b in binary if b not in hard and b not in open_set]
    hard_in_binary = [b for b in binary if b in hard]
    open_in_binary = [b for b in binary if b in open_set]

    sys.stderr.write(
        f"[catalog] binary={len(binary_all)}, infeasible={len(infeasible)}, "
        f"hard-in-binary={len(hard_in_binary)}, "
        f"open-in-binary={len(open_in_binary)}, easy-in-binary={len(easy)}\n"
    )

    to_download: list[str] = []
    if args.filter_list is not None:
        wanted = set()
        with args.filter_list.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if not line.endswith(".mps.gz"):
                    line = line + ".mps.gz"
                wanted.add(line)
        to_download = sorted(wanted)
    elif args.include is not None:
        quotas: dict[str, int] = {}
        for tok in args.include.split(","):
            k, v = tok.split(":")
            quotas[k.strip()] = int(v.strip())
        for cat, n in quotas.items():
            pool = {"hard": hard_in_binary, "open": open_in_binary,
                    "easy": easy, "binary": binary}.get(cat)
            if pool is None:
                sys.stderr.write(f"error: unknown category '{cat}'\n")
                return 2
            to_download.extend(choose_subset(pool, n))
    else:
        to_download = choose_subset(binary, args.n)

    # Deduplicate, preserving order.
    seen: set[str] = set()
    to_download = [x for x in to_download if not (x in seen or seen.add(x))]

    sys.stderr.write(f"[plan] downloading {len(to_download)} instances "
                     f"to {args.out}\n")
    for fn in to_download:
        sys.stderr.write(f"  - {fn}\n")
    if args.dry_run:
        return 0

    n_ok, n_fail = 0, 0
    for fn in to_download:
        if download_one(fn, args.out):
            n_ok += 1
        else:
            n_fail += 1

    sys.stderr.write(f"[done] {n_ok} ok, {n_fail} failed\n")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

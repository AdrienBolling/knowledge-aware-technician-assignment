"""Merge per-agent hvp_eval part-roots into one scenario directory.

The parallel 5M ladder runs each agent in its own ``--out-root``
(``reports/hvp_vl_parts/<agent>/very_long/``) so concurrent runs never
touch the same CSV; this consolidates them into the canonical
``reports/hvp_eval_v3/very_long/`` layout the analyzer expects.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True,
                    help="root holding <agent>/very_long/ part dirs")
    ap.add_argument("--dest", required=True,
                    help="destination scenario dir (e.g. .../very_long)")
    args = ap.parse_args()

    eps, steps, manifest = [], [], None
    parts = sorted(Path(args.parts).glob("*/very_long"))
    for p in parts:
        e = p / "episodes.csv"
        if not e.is_file():
            print(f"[skip] {p.parent.name}: no episodes.csv")
            continue
        eps.append(pd.read_csv(e))
        steps.append(pd.read_csv(p / "steps.csv.gz"))
        manifest = manifest or (p / "manifest.json")
        print(f"[ok]   {p.parent.name}")
    if not eps:
        print("nothing to merge")
        return 1

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    pd.concat(eps, ignore_index=True).to_csv(dest / "episodes.csv", index=False)
    pd.concat(steps, ignore_index=True).to_csv(
        dest / "steps.csv.gz", index=False, compression="gzip"
    )
    if manifest:
        shutil.copy(manifest, dest / "manifest.json")
    print(f"merged {len(eps)} parts -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Merge per-agent hvp_eval part-roots into one scenario directory.

The parallel 5M ladder runs each agent in its own ``--out-root``
(``reports/hvp_vl_parts/<agent>/<scenario>/``) so concurrent runs never
touch the same CSV; this consolidates them into the canonical
``reports/hvp_eval_v4/<scenario>/`` layout the analyzer expects.

Non-destructive (2026-08-04): if the destination already holds results,
rows for agents NOT present in the parts are preserved — the historical
behaviour overwrote the destination with only the merged parts, which
silently discarded the rest of the roster.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def _agent_col(df: pd.DataFrame) -> str | None:
    for c in ("agent", "agent_key", "agent_name"):
        if c in df.columns:
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True,
                    help="root holding <agent>/<scenario>/ part dirs")
    ap.add_argument("--dest", required=True,
                    help="destination scenario dir (e.g. .../very_long)")
    ap.add_argument("--scenario", default="very_long",
                    help="scenario subdirectory name inside each part "
                         "(default: very_long)")
    args = ap.parse_args()

    eps, steps, manifest = [], [], None
    parts = sorted(Path(args.parts).glob(f"*/{args.scenario}"))
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

    new_eps = pd.concat(eps, ignore_index=True)
    new_steps = pd.concat(steps, ignore_index=True)

    # Preserve existing rows for agents absent from the parts.
    col = _agent_col(new_eps)
    dest_eps = dest / "episodes.csv"
    if col and dest_eps.is_file():
        old = pd.read_csv(dest_eps)
        if _agent_col(old) == col:
            merged_agents = set(new_eps[col].unique())
            keep = old[~old[col].isin(merged_agents)]
            if len(keep):
                print(f"[keep] {len(keep)} existing episode rows for "
                      f"{sorted(keep[col].unique())}")
                new_eps = pd.concat([keep, new_eps], ignore_index=True)
        dest_steps = dest / "steps.csv.gz"
        if dest_steps.is_file():
            old_s = pd.read_csv(dest_steps)
            scol = _agent_col(old_s)
            if scol:
                keep_s = old_s[~old_s[scol].isin(set(new_steps[scol].unique())
                                                 if scol in new_steps else set())]
                if len(keep_s):
                    new_steps = pd.concat([keep_s, new_steps],
                                          ignore_index=True)

    new_eps.to_csv(dest / "episodes.csv", index=False)
    new_steps.to_csv(dest / "steps.csv.gz", index=False, compression="gzip")
    if manifest and not (dest / "manifest.json").is_file():
        shutil.copy(manifest, dest / "manifest.json")
    print(f"merged {len(eps)} parts -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

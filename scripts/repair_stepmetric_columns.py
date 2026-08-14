"""Retroactively repair snapshot-contaminated step-metric columns.

Before 2026-08-14, ``eval_human_vs_performance.py::run_episode`` copied the
TERMINAL step's ``info["metrics"]`` dict verbatim into the episode KPI row,
so every StepMetric column in ``episodes.csv`` (``repair_quality``,
``mttr_rolling``, ``repair_time_delta``, ``repair_time_delta_per``) was a
single last-decision snapshot — one assignment's quality, one rolling
window out of the whole episode — instead of an episode aggregate.

This script recomputes the columns that are recoverable from the recorded
per-step series and overwrites them in place:

* ``repair_quality``        — mean over recorded decisions
* ``repair_time_delta_per`` — mean over recorded decisions
* ``mttr_rolling``          — mean over recorded decisions, EXCLUDING the
                              0.0 placeholder emitted before the first
                              completed repair

``repair_time_delta`` (absolute) is NOT in the step records and cannot be
recovered — it keeps its terminal-snapshot value and is listed as such in
the sidecar manifest.  For runs recorded with ``--record-every N > 1``
(very_long / lifecycle) the recomputed means are subsample estimates of the
same quantity the fixed harness now computes exactly.

Idempotent: values are always recomputed from ``steps.csv.gz``; the first
run preserves the original file as ``episodes.csv.presnapshotfix``.

Usage:
    python scripts/repair_stepmetric_columns.py --roots reports/hvp_eval_v6w reports/hvp_v6w_parts
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPAIR_COLS = ["repair_quality", "repair_time_delta_per", "mttr_rolling"]
UNRECOVERABLE = ["repair_time_delta"]


def repair_dir(d: Path) -> dict | None:
    ep_f, st_f = d / "episodes.csv", d / "steps.csv.gz"
    if not (ep_f.is_file() and st_f.is_file()):
        return None
    ep = pd.read_csv(ep_f)
    st = pd.read_csv(st_f)
    if "agent" not in ep.columns or "agent" not in st.columns:
        return None
    if "episode" not in st.columns:
        st = st.assign(episode=0)
    cols = [c for c in REPAIR_COLS if c in ep.columns and c in st.columns]
    if not cols:
        return None

    stats: dict[str, float] = {}
    n_samples: list[int] = []
    ep = ep.set_index(["agent", "episode"], drop=False)
    for (a, e), grp in st.groupby(["agent", "episode"]):
        if (a, e) not in ep.index:
            continue
        n_samples.append(len(grp))
        for c in cols:
            vals = grp[c].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if c == "mttr_rolling":
                vals = vals[vals != 0.0]
            if vals.size == 0:
                continue
            new = float(vals.mean())
            old = float(ep.loc[(a, e), c])
            key = f"max_abs_delta_{c}"
            stats[key] = max(stats.get(key, 0.0), abs(new - old))
            ep.loc[(a, e), c] = new
    ep = ep.reset_index(drop=True)

    backup = ep_f.with_suffix(".csv.presnapshotfix")
    if not backup.exists():
        ep_f.rename(backup)
    ep.to_csv(ep_f, index=False)

    manifest = {
        "repaired_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "columns_repaired": cols,
        "columns_still_terminal_snapshot": [
            c for c in UNRECOVERABLE if c in ep.columns
        ],
        "rows": int(len(ep)),
        "step_samples_per_row_min": int(min(n_samples)) if n_samples else 0,
        "step_samples_per_row_median": (
            int(np.median(n_samples)) if n_samples else 0
        ),
        **{k: round(v, 6) for k, v in stats.items()},
    }
    (d / "stepmetric_repair.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    args = ap.parse_args()

    n = 0
    for root in args.roots:
        for ep_f in sorted(Path(root).rglob("episodes.csv")):
            m = repair_dir(ep_f.parent)
            if m is None:
                continue
            n += 1
            deltas = {
                k.removeprefix("max_abs_delta_"): v
                for k, v in m.items()
                if k.startswith("max_abs_delta_")
            }
            print(
                f"[repaired] {ep_f.parent}  rows={m['rows']}  "
                f"samples/row(min/med)={m['step_samples_per_row_min']}/"
                f"{m['step_samples_per_row_median']}  max|Δ|={deltas}"
            )
    print(f"{n} episode files repaired")
    return 0


if __name__ == "__main__":
    main()

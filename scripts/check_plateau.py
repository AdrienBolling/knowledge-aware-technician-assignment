"""Decide whether a training run has plateaued (post-v3 queue gate).

Pulls the per-episode ``train/return`` and ``train/length`` history of a
wandb run and tests whether the *per-decision* return (return/length —
episode returns scale with the sampled horizon, so raw returns are not
comparable across episodes) is still improving.

Rule: split the episode series into quarters; compare the mean
normalised return of the last quarter (m4) against the previous one
(m3).  "Still improving" = m4 exceeds m3 by more than 2 % of |m3| AND
by more than 0.5 standard errors of the last quarter — both clauses
must fire, so noise alone cannot trigger an extension.

Exit codes (consumed by scripts/serval_post_v3_queue.sh):
  0  plateau — proceed to benchmarks
  1  still improving — extend training
  2  indeterminate (API unreachable, too little data) — the queue
     proceeds to benchmarks but logs an ALERT

Usage::

    uv run python scripts/check_plateau.py --log reports/train_hc_v3.log
    uv run python scripts/check_plateau.py --run bolling-adrien/kata-set-transformer/srnes3kk
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np


def run_path_from_log(log: Path) -> str | None:
    """Extract entity/project/run_id from the wandb banner in a log."""
    m = None
    for line in log.read_text(errors="replace").splitlines():
        found = re.search(r"wandb\.ai/([\w-]+)/([\w-]+)/runs/([\w]+)", line)
        if found:
            m = found  # keep the LAST run mentioned (extensions append)
    return f"{m.group(1)}/{m.group(2)}/{m.group(3)}" if m else None


def fetch_history(run_path: str, tries: int = 3):
    import wandb

    for attempt in range(tries):
        try:
            api = wandb.Api(timeout=60)
            run = api.run(run_path)
            hist = run.history(
                keys=["train/return", "train/length"], samples=10000,
                pandas=True,
            )
            return hist
        except Exception as exc:  # network flake / auth — retry
            print(f"[check_plateau] fetch attempt {attempt + 1} failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            time.sleep(30)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help="wandb run path entity/project/run_id")
    ap.add_argument("--log", default=None,
                    help="training log to extract the run path from")
    ap.add_argument("--rel-threshold", type=float, default=0.02)
    ap.add_argument("--min-episodes", type=int, default=80)
    args = ap.parse_args()

    run_path = args.run
    if run_path is None and args.log:
        run_path = run_path_from_log(Path(args.log))
    if run_path is None:
        print("[check_plateau] no run path found", flush=True)
        return 2

    hist = fetch_history(run_path)
    if hist is None or len(hist) == 0:
        print(f"[check_plateau] no history for {run_path}", flush=True)
        return 2

    hist = hist.dropna(subset=["train/return", "train/length"])
    lengths = hist["train/length"].to_numpy(dtype=float)
    returns = hist["train/return"].to_numpy(dtype=float)
    ok = lengths > 0
    r = returns[ok] / lengths[ok]
    n = len(r)
    if n < args.min_episodes:
        print(f"[check_plateau] only {n} episodes — indeterminate", flush=True)
        return 2

    q = n // 4
    m3, m4 = float(np.mean(r[2 * q:3 * q])), float(np.mean(r[3 * q:]))
    se4 = float(np.std(r[3 * q:]) / max(np.sqrt(n - 3 * q), 1.0))
    improving = (
        (m4 - m3) > args.rel_threshold * max(abs(m3), 1e-9)
        and (m4 - m3) > 0.5 * se4
    )
    print(f"[check_plateau] {run_path}: n={n} m3={m3:.5f} m4={m4:.5f} "
          f"delta={(m4 - m3):.5f} se4={se4:.5f} -> "
          f"{'IMPROVING' if improving else 'PLATEAU'}", flush=True)
    return 1 if improving else 0


if __name__ == "__main__":
    sys.exit(main())

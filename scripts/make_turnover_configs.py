"""Write the two neutral-turnover variants of the S5 lifecycle scenario.

Reviewer point: S5 (``run_configs/benchmark_suite/lifecycle.json``) retires
the most knowledgeable technicians.  That is a targeted key-person stress
test, not neutral turnover.  This script derives two evaluation-only
variants from lifecycle.json.  Everything except the fields below stays
identical (layout sampling, horizon, hires, machine events, reward, sim).

``lifecycle_random_retire.json``
    Every ``retire_technician`` event uses ``select: "random"``.  Times and
    counts do not change.

``lifecycle_random_timing.json``
    Random selection as above.  In addition, each technician retire/hire
    pair (a retirement and the hire that follows it) moves to a time drawn
    once, uniformly over the evaluation horizon, with a fixed seed.  Each
    hire keeps its delay after its retirement.  The drawn times are sorted
    and given to the pairs in their original order, so the pair sequence
    (2 trainees, 2 juniors, 1 trainee) stays the same.  The unpaired expert
    hire and all machine events keep their times.

The times are written into the JSON, so the configs are static.  Run again
to regenerate them (the output is byte-identical for the same seed):

    uv run --no-sync python scripts/make_turnover_configs.py
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

SRC = Path("run_configs/benchmark_suite/lifecycle.json")
OUT_DIR = Path("run_configs/benchmark_suite")
# Evaluation horizon of the ``lifecycle`` harness profile
# (scripts/eval_human_vs_performance.py SCENARIOS["lifecycle"]["sim"]).
HORIZON = 5_000_000.0
SEED = 20260916


def _pairs(events: list[dict]) -> list[tuple[int, int]]:
    """Index pairs (retire, next add_technician after it) in time order."""
    order = sorted(range(len(events)), key=lambda i: float(events[i]["time"]))
    pairs, used = [], set()
    for pos, i in enumerate(order):
        if events[i]["kind"] != "retire_technician":
            continue
        hire = next(
            (j for j in order[pos + 1:]
             if events[j]["kind"] == "add_technician" and j not in used),
            None,
        )
        if hire is None:
            raise SystemExit(f"retirement at {events[i]['time']} has no hire")
        used.add(hire)
        pairs.append((i, hire))
    return pairs


def random_retire(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    for ev in out["gym"]["lifecycle_events"]:
        if ev["kind"] == "retire_technician":
            ev["select"] = "random"
    return out


def random_timing(cfg: dict, *, seed: int, horizon: float) -> dict:
    out = random_retire(cfg)
    events = out["gym"]["lifecycle_events"]
    pairs = _pairs(events)
    delays = [float(events[h]["time"]) - float(events[r]["time"])
              for r, h in pairs]
    # Draw retirement times so that every hire still falls inside the
    # horizon; sort them and keep the original pair order.
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0.0, horizon - max(delays), size=len(pairs)))
    for (r, h), t, d in zip(pairs, times, delays):
        t = max(1.0, float(round(t)))
        events[r]["time"] = t
        events[h]["time"] = t + d
    events.sort(key=lambda e: float(e["time"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--horizon", type=float, default=HORIZON)
    args = ap.parse_args()

    cfg = json.loads(SRC.read_text())
    variants = {
        "lifecycle_random_retire": random_retire(cfg),
        "lifecycle_random_timing": random_timing(
            cfg, seed=args.seed, horizon=args.horizon
        ),
    }
    for name, v in variants.items():
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(v, indent=2) + "\n")
        print(f"wrote {path}")
        for ev in v["gym"]["lifecycle_events"]:
            print(f"  {ev['time']:>10.0f}  {ev['kind']:<18s} x{ev['count']}"
                  f"  {ev.get('select') or ev.get('template') or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

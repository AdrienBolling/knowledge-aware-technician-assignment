"""Results table of the S5 deployment-adaptation study.

Reads the runs written by scripts/dgy_live_s5_queue.sh under reports/live_s5:

  <label>/lifecycle/episodes.csv        one live episode (learning or control)
  bench/<key>/lifecycle/episodes.csv    deterministic benchmark of a checkpoint
  <label>/lifecycle/updates.csv         PPO update trajectory of a live run

The baseline is the benchmark rerun of the base policy (bench/ft_quality) with
the same code and settings, NOT the published S5 row: the published row was
produced with a random PYTHONHASHSEED, under which machine-id hash collisions
can drop a machine from the observations, so it is not reproducible here.

Final fleet knowledge and terminal MTTR come from the step records.  They are
read from <run>/lifecycle/steps.csv.gz when present, otherwise from
reports/live_s5/step_metrics.csv (columns run, mttr_final, know_final), which
is cheaper to produce on the evaluation host:

  python - <<'EOF'
  import glob, pandas as pd
  rows = []
  for p in sorted(glob.glob("reports/live_s5/**/lifecycle/steps.csv.gz", recursive=True)):
      d = pd.read_csv(p, usecols=["agent", "episode", "step", "sim_time",
                                  "mttr_rolling", "fleet_knowledge"])
      for (a, e), h in d.groupby(["agent", "episode"]):
          g = h[(h.mttr_rolling > 0) & (h.step >= 52)]
          hi = g.sim_time.max()
          rows.append({"run": p.split("/")[2] if "/bench/" not in p else "bench_" + p.split("/")[3],
                       "mttr_final": float(g[g.sim_time >= hi * 0.97].mttr_rolling.mean()),
                       "know_final": float(h.sort_values("step").fleet_knowledge.iloc[-1])})
  pd.DataFrame(rows).to_csv("reports/live_s5/step_metrics.csv", index=False)
  EOF

Usage: python scripts/live_s5_report.py [--root reports/live_s5] [--md]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

BASELINE = "bench_ft_quality"
# run key -> (row label, what the run is)
RUNS = [
    ("bench_ft_quality", "HTT-RL qua. (baseline rerun)", "argmax, no learning"),
    ("ft_quality_sto", "Control: sampled actions", "sampled, no learning"),
    ("ft_quality_live_sto", "Live, sampled actions", "sampled, PPO updates"),
    ("ft_quality_live_det", "Live, argmax actions", "argmax, PPO updates"),
    ("bench_ft_quality_live_sto", "Live sampled: final weights", "argmax, no learning"),
    ("bench_ft_quality_live_det", "Live argmax: final weights", "argmax, no learning"),
    ("bench_ft_quality_s5ft", "10-episode fine-tune", "argmax, no learning"),
]


def load_episodes(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(str(root / "**" / "lifecycle" / "episodes.csv"), recursive=True)):
        parts = Path(p).parts
        run = "bench_" + parts[-3] if "bench" in parts else parts[-3]
        d = pd.read_csv(p)
        d.insert(0, "run", run)
        rows.append(d)
    if not rows:
        raise SystemExit(f"no episodes.csv under {root}")
    return pd.concat(rows, ignore_index=True)


def step_metrics(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(str(root / "**" / "lifecycle" / "steps.csv.gz"), recursive=True)):
        parts = Path(p).parts
        run = "bench_" + parts[-3] if "bench" in parts else parts[-3]
        d = pd.read_csv(p, usecols=["agent", "episode", "step", "sim_time",
                                    "mttr_rolling", "fleet_knowledge"])
        g = d[(d.mttr_rolling > 0) & (d.step >= 52)]
        hi = g.sim_time.max()
        rows.append({"run": run,
                     "mttr_final": float(g[g.sim_time >= hi * 0.97].mttr_rolling.mean()),
                     "know_final": float(d.sort_values("step").fleet_knowledge.iloc[-1])})
    csv = root / "step_metrics.csv"
    if csv.is_file():
        rows += [r for r in pd.read_csv(csv).to_dict("records")
                 if r["run"] not in {x["run"] for x in rows}]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["run", "mttr_final", "know_final"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="reports/live_s5")
    ap.add_argument("--md", action="store_true", help="markdown table")
    args = ap.parse_args()
    root = Path(args.root)

    ep = load_episodes(root).set_index("run")
    ep["disr"] = ep.ill_technician_count / ep.finished_products * 1000.0
    sm = step_metrics(root)
    if len(sm):
        ep = ep.join(sm.set_index("run"), how="left")
    if BASELINE not in ep.index:
        raise SystemExit(f"baseline run {BASELINE} is missing from {root}")
    base = ep.loc[BASELINE]

    cols = [("finished_products", "Products", "{:.0f}"), ("delta", "vs base", "{:+.1f}%"),
            ("mttr", "MTTR", "{:.1f}"), ("disr", "Disr/10^3", "{:.0f}"),
            ("mttr_final", "Term. MTTR", "{:.1f}"), ("know_final", "Knowledge", "{:.0f}"),
            ("repair_quality", "Quality", "{:.3f}"),
            ("fleet_availability_rate", "Avail.", "{:.3f}"),
            ("wall_s", "Wall h", "{:.1f}")]
    table = []
    for run, label, how in RUNS:
        if run not in ep.index:
            table.append([label, how] + ["--"] * len(cols))
            continue
        r = ep.loc[run]
        vals = []
        for key, _, fmt in cols:
            if key == "delta":
                v = 100.0 * (r.finished_products / base.finished_products - 1.0)
            elif key == "wall_s":
                v = r.wall_s / 3600.0
            else:
                v = r.get(key, float("nan"))
            vals.append("--" if pd.isna(v) else fmt.format(v))
        table.append([label, how] + vals)
    head = ["Run", "Acting"] + [c[1] for c in cols]
    if args.md:
        print("| " + " | ".join(head) + " |")
        print("|" + "|".join(["---"] * len(head)) + "|")
        for row in table:
            print("| " + " | ".join(row) + " |")
    else:
        w = [max(len(str(x[i])) for x in [head] + table) for i in range(len(head))]
        print("  ".join(h.ljust(w[i]) for i, h in enumerate(head)))
        for row in table:
            print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(row)))
    print(f"\nbaseline: {BASELINE} = {base.finished_products:.0f} products "
          f"(published S5 row of HTT-RL qua.: 30601, different PYTHONHASHSEED)")

    for p in sorted(glob.glob(str(root / "*" / "lifecycle" / "updates.csv"))):
        d = pd.read_csv(p)
        k = max(1, len(d) // 5)
        first, last = d.head(k), d.tail(k)
        print(f"\n{Path(p).parts[-3]}: {len(d)} updates over {d.step.max():,} decisions, "
              f"{d.update_s.mean():.0f} s per update")
        for col in ("approx_kl", "entropy", "clip_fraction", "vf_loss", "lr"):
            if col in d:
                fmt = ".2e" if col == "lr" else ".4f"
                print(f"   {col:14s} first fifth {first[col].mean():{fmt}}  "
                      f"last fifth {last[col].mean():{fmt}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate the analysis notebook suite from this single source file.

Run::

    uv run python analysis/_build_notebooks.py

Each ``_make_<name>()`` function below returns a list of (cell_type, source)
tuples; the harness at the bottom wraps them into valid .ipynb JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

Cells = list[tuple[str, str]]


def _md(c: Cells, text: str) -> None:
    c.append(("markdown", text))


def _code(c: Cells, src: str) -> None:
    c.append(("code", src))


# ---------------------------------------------------------------------------
# Common preamble — used at the top of every notebook
# ---------------------------------------------------------------------------

PREAMBLE = '''import sys, pathlib
# Make ``utils.py`` importable whether the notebook is opened from
# project root or from analysis/.
_here = pathlib.Path.cwd()
for cand in (_here, _here / "analysis", _here.parent):
    if (cand / "utils.py").exists():
        sys.path.insert(0, str(cand))
        break

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    list_runs, latest_run, load_run, load_runs,
    group_columns, group_columns_by_prefix, strip_prefix, rolling_mean,
)

plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 200)'''


SELECT_RUN = '''# Pick a run.  ``latest_run()`` returns the most recently modified
# experiment in ``reports/``.  Override ``EXP_ID = "..."`` to analyse
# a specific one.
EXP_ID = latest_run()
print("Analysing:", EXP_ID)

ep, st, cfg = load_run(EXP_ID)
print(f"  episode_metrics.csv: shape={ep.shape}")
print(f"  step_metrics.csv:    shape={st.shape if st is not None else 'absent'}")'''


# ---------------------------------------------------------------------------
# 00 — Overview
# ---------------------------------------------------------------------------


def _make_00_overview() -> Cells:
    c: Cells = []
    _md(c, """# 00 — Overview

Welcome to the analysis suite.  Every notebook here reads the CSV
reports written by `Experiment.run()` under `reports/<exp_id>/`:

* `episode_metrics.csv` — one row per episode (training, eval, …).
* `step_metrics.csv`    — one row per env step.
* `config.json`         — the env + agent + experiment config snapshot.

This first notebook shows:

1. Which runs are available on disk.
2. How to load one.
3. The high-level structure of the resulting DataFrames.
4. A one-glance summary of the run.""")

    _code(c, PREAMBLE)

    _md(c, """## Available runs""")
    _code(c, """runs_df = list_runs()
runs_df""")

    _md(c, """## Load the latest run""")
    _code(c, SELECT_RUN)

    _md(c, """## Episode metrics — columns by prefix

The dataframe is wide.  Columns are namespaced with a prefix that
tells you what they are:

| Prefix | Meaning |
|---|---|
| `metric/` | Episode-level KPIs (`metric/mttr`, `metric/total_breakdowns`, …) |
| `step_mean/` | Mean of each per-step metric over the episode |
| `reward_sum/` / `reward_mean/` | Total + mean per-step reward components |
| `update/` | Agent-update metrics (PPO loss, KL, …) |
| `assignments/` | Per-tech repair-assignment count |
| `machine_maintenance/` / `machine_production/` / `machine_breakdowns/` | Per-machine episode stats |""")
    _code(c, """groups = group_columns_by_prefix(ep)
for prefix in sorted(groups):
    print(f"  {prefix + '/':<25} {len(groups[prefix])} columns")""")

    _md(c, """## Step metrics — head""")
    _code(c, """if st is not None and not st.empty:
    display(st.head(5))
else:
    print("no step_metrics.csv for this run")""")

    _md(c, """## High-level summary""")
    _code(c, """summary = (
    ep.groupby("phase")
      .agg(
          n_episodes=("episode", "count"),
          mean_return=("return", "mean"),
          last_return=("return", "last"),
          mean_length=("length", "mean"),
          total_wall_clock_s=("wall_clock_seconds", "sum"),
      )
      .round(3)
)
summary""")

    _md(c, """## Config snapshot""")
    _code(c, """print("agent:    ", cfg.get("agent", {}).get("agent_type"))
print("seed:     ", cfg.get("experiment", {}).get("seed"))
print("n_eps:    ", cfg.get("experiment", {}).get("n_episodes"))
print("obs_repr: ", cfg.get("env", {}).get("gym", {}).get("observation_representation"))
print("obs_mode: ", cfg.get("env", {}).get("gym", {}).get("observation_mode"))
print("randomized:", cfg.get("env", {}).get("randomized_scenario", {}).get("enabled"))""")

    return c


# ---------------------------------------------------------------------------
# 01 — Learning curves
# ---------------------------------------------------------------------------


def _make_01_learning_curves() -> Cells:
    c: Cells = []
    _md(c, """# 01 — Learning curves

Was the agent actually learning?  Plots episode return + agent-update
metrics (loss, entropy, KL, …) over the course of training, broken
out by phase (``train`` / ``inline_eval`` / ``eval``).""")

    _code(c, PREAMBLE)
    _code(c, SELECT_RUN)

    _md(c, """## Episode return

Raw + rolling-window average for the training phase.""")
    _code(c, """train = ep[ep["phase"] == "train"].copy()
train["return_smooth"] = rolling_mean(train["return"], window=max(5, len(train)//30 or 1))

fig, ax = plt.subplots()
ax.plot(train["episode"], train["return"], lw=0.5, color="C0", alpha=0.4, label="raw")
ax.plot(train["episode"], train["return_smooth"], lw=2, color="C0", label="rolling avg")

# Overlay eval points if present
for phase, marker, color in [("inline_eval", "o", "C1"), ("eval", "s", "C2")]:
    sub = ep[ep["phase"] == phase]
    if not sub.empty:
        ax.scatter(sub["episode"], sub["return"], s=30, marker=marker, color=color, label=phase, zorder=3)

ax.set(xlabel="episode", ylabel="episode return", title="Episode return over training")
ax.legend()
plt.show()""")

    _md(c, """## Agent update metrics (PPO)

`loss`, `entropy`, plus PPO-specific `pg_loss`, `vf_loss`, `approx_kl`,
`clip_fraction`, `lr` if recorded.""")
    _code(c, """update_cols = ["loss", "entropy"] + group_columns(train, "update")
update_cols = [c for c in update_cols if c in train.columns and train[c].notna().any()]

if update_cols:
    n = len(update_cols)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.6 * nrows), squeeze=False)
    for ax, col in zip(axes.flat, update_cols):
        ax.plot(train["episode"], train[col], lw=1)
        ax.set_title(col.split('/', 1)[-1])
        ax.set_xlabel("episode")
    for ax in axes.flat[len(update_cols):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()
else:
    print("No update metrics in this run (heuristic baseline?)")""")

    _md(c, """## Train vs eval gap

If `inline_eval` is enabled, the gap between train and eval returns
is the standard "is the policy overfitting the latest rollout"
diagnostic.""")
    _code(c, """if "inline_eval" in ep["phase"].unique():
    fig, ax = plt.subplots()
    for phase, color in [("train", "C0"), ("inline_eval", "C1")]:
        sub = ep[ep["phase"] == phase].copy()
        sub["smooth"] = rolling_mean(sub["return"], window=max(3, len(sub)//30 or 1))
        ax.plot(sub["episode"], sub["smooth"], label=phase, color=color, lw=2)
    ax.set(xlabel="episode", ylabel="return (rolling avg)", title="Train vs inline-eval")
    ax.legend()
    plt.show()
else:
    print("This run has no inline_eval phase to compare against.")""")

    return c


# ---------------------------------------------------------------------------
# 02 — Reward decomposition
# ---------------------------------------------------------------------------


def _make_02_reward_decomposition() -> Cells:
    c: Cells = []
    _md(c, """# 02 — Reward decomposition

Where did the agent's total return come from?  The runner records
both the **sum** and the **mean** of every enabled reward component
per episode (``reward_sum/<name>`` and ``reward_mean/<name>``).""")

    _code(c, PREAMBLE)
    _code(c, SELECT_RUN)

    _md(c, """## Cumulative contribution per component (training)""")
    _code(c, """train = ep[ep["phase"] == "train"]
sum_cols = group_columns(train, "reward_sum")
contribs = train[sum_cols].sum().sort_values()

fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(contribs))))
contribs.plot.barh(ax=ax)
ax.set(xlabel="total contribution to return over all train episodes",
       title="Reward component breakdown — totals")
ax.set_yticklabels(strip_prefix(list(contribs.index), "reward_sum"))
plt.show()""")

    _md(c, """## Reward-component evolution over episodes

Stacked area: each episode's return decomposed by component.  Helps
identify whether one component dominates (e.g. `assignment` runs
positive while `wait_time` runs negative).""")
    _code(c, """labels = strip_prefix(sum_cols, "reward_sum")
stacked = train[sum_cols].copy()
stacked.columns = labels

# Separate positive and negative components so the plot isn't dominated
# by the zero line
pos = stacked.clip(lower=0)
neg = stacked.clip(upper=0)

fig, ax = plt.subplots()
ax.stackplot(train["episode"], pos.T, labels=pos.columns, alpha=0.7)
ax.stackplot(train["episode"], neg.T, alpha=0.7)
ax.axhline(0, color="k", lw=0.5)
ax.set(xlabel="episode", ylabel="reward contribution", title="Reward decomposition over training")
ax.legend(loc="best", fontsize=8, ncol=2)
plt.show()""")

    _md(c, """## Mean per-step component values""")
    _code(c, """mean_cols = group_columns(train, "reward_mean")
mean_df = train[mean_cols].agg(["mean", "std"]).T
mean_df.index = strip_prefix(list(mean_df.index), "reward_mean")
mean_df.sort_values("mean", ascending=False).round(4)""")

    return c


# ---------------------------------------------------------------------------
# 03 — Episode KPIs
# ---------------------------------------------------------------------------


def _make_03_episode_kpis() -> Cells:
    c: Cells = []
    _md(c, """# 03 — Manufacturing KPIs

The non-reward, operations-oriented metrics computed at episode end:
MTTR, fleet availability, throughput, total breakdowns / repairs,
finished products, etc.  These are the indicators a *factory
operator* would care about.""")

    _code(c, PREAMBLE)
    _code(c, SELECT_RUN)

    _md(c, """## Side-by-side KPI evolution""")
    _code(c, """metric_cols = group_columns(ep, "metric")
metric_cols = [c for c in metric_cols if ep[c].notna().any()]
print(f"{len(metric_cols)} episode metric columns")

if metric_cols:
    n = len(metric_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.8 * nrows), squeeze=False)
    train = ep[ep["phase"] == "train"]
    for ax, col in zip(axes.flat, metric_cols):
        ax.plot(train["episode"], train[col], lw=1, color="C0", alpha=0.4)
        ax.plot(train["episode"], rolling_mean(train[col]), lw=2, color="C0")
        ax.set_title(col.split('/', 1)[-1])
        ax.set_xlabel("episode")
    for ax in axes.flat[len(metric_cols):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()""")

    _md(c, """## Sim time vs wall-clock time""")
    _code(c, """if "wall_clock_seconds" in ep.columns and "sim_time_end" in ep.columns:
    fig, ax = plt.subplots()
    ax.scatter(ep["wall_clock_seconds"], ep["sim_time_end"], c=ep["episode"], cmap="viridis", s=15)
    ax.set(xlabel="wall-clock seconds", ylabel="final sim time", title="Episode duration (real vs sim)")
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("episode")
    plt.show()
else:
    print("Wall-clock / sim-time columns not in this run.")""")

    _md(c, """## Distribution of finished products / MTTR""")
    _code(c, """interest = [c for c in ["metric/finished_products", "metric/mttr",
                            "metric/fleet_availability_rate", "metric/throughput_rate"]
            if c in ep.columns and ep[c].notna().any()]
if interest:
    fig, axes = plt.subplots(1, len(interest), figsize=(4 * len(interest), 4))
    if len(interest) == 1: axes = [axes]
    for ax, col in zip(axes, interest):
        ep[col].dropna().plot.hist(ax=ax, bins=20)
        ax.set_title(col.split('/', 1)[-1])
    plt.tight_layout(); plt.show()""")

    return c


# ---------------------------------------------------------------------------
# 04 — Per-technician
# ---------------------------------------------------------------------------


def _make_04_per_technician() -> Cells:
    c: Cells = []
    _md(c, """# 04 — Per-technician analysis

How the agent distributed work across the fleet, and how each
technician's knowledge / fatigue / specialisation evolved.""")

    _code(c, PREAMBLE)
    _code(c, SELECT_RUN)

    _md(c, """## Total assignments per technician across the run""")
    _code(c, """assign_cols = group_columns(ep, "assignments")
if assign_cols:
    totals = ep[assign_cols].sum().sort_values(ascending=False)
    totals.index = strip_prefix(list(totals.index), "assignments")
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(totals))))
    totals.plot.barh(ax=ax, color="C0", edgecolor="black", linewidth=0.5)
    ax.invert_yaxis()
    ax.set(xlabel="repairs assigned", title="Workload distribution across the fleet")
    plt.show()
else:
    print("No assignments/* columns in this run.")""")

    _md(c, """## Assignment-share evolution (training only)

A line per technician showing what *fraction* of an episode's repairs
went to them.  Drift towards 1 / n_techs ⇒ uniform; spikes ⇒
favouritism.""")
    _code(c, """train = ep[ep["phase"] == "train"]
if assign_cols and not train.empty:
    counts = train[assign_cols].copy()
    counts.columns = strip_prefix(list(counts.columns), "assignments")
    shares = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)
    shares["episode"] = train["episode"].values

    fig, ax = plt.subplots()
    for col in shares.columns[:-1]:
        ax.plot(shares["episode"], rolling_mean(shares[col]), label=col, lw=1.5)
    ax.axhline(1.0 / max(1, len(shares.columns) - 1), ls="--", color="grey", lw=0.8,
               label="uniform")
    ax.set(xlabel="episode", ylabel="share of repairs", title="Assignment share per tech (rolling)")
    ax.legend(fontsize=8, ncol=2)
    plt.show()""")

    _md(c, """## Per-tech knowledge / fatigue evolution (step level)

If step metrics are available, plot mean ``tech_knowledge/<t>`` and
``tech_fatigue/<t>`` over the steps of the last training episode.""")
    _code(c, """if st is not None and not st.empty:
    last_ep = st[(st["phase"] == "train") & (st["episode"] == st[st["phase"] == "train"]["episode"].max())]
    if last_ep.empty:
        last_ep = st[st["episode"] == st["episode"].max()]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for kind, ax in zip(("tech_knowledge", "tech_fatigue"), axes):
        cols = group_columns(last_ep, kind)
        for col in cols:
            ax.plot(last_ep["step"], last_ep[col], label=col.split('/', 1)[1], lw=1.2)
        ax.set(xlabel="step within episode", ylabel=kind,
               title=f"{kind} on episode {int(last_ep['episode'].iloc[0])}")
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()
else:
    print("No step metrics available.")""")

    _md(c, """## Specialisation — what component types each tech handled

Cross-tab of (assigned tech) × (ticket_component_type) from the step
log, normalised per row so you can see relative specialisation.""")
    _code(c, """if st is not None and not st.empty and "ticket_component_type" in st.columns:
    train_steps = st[st["phase"] == "train"]
    if not train_steps.empty:
        # Action index → no per-tech name in step log; key on action int
        ct = pd.crosstab(train_steps["action"], train_steps["ticket_component_type"])
        # Drop empty component-type column if any
        ct = ct.loc[:, ct.sum() > 0]
        norm = ct.div(ct.sum(axis=1).replace(0, 1), axis=0)
        fig, ax = plt.subplots(figsize=(min(12, 1 + 0.8 * ct.shape[1]), 0.4 * ct.shape[0] + 1))
        im = ax.imshow(norm.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels(ct.columns, rotation=30, ha="right")
        ax.set_yticks(range(ct.shape[0])); ax.set_yticklabels(["tech_" + str(i) for i in ct.index])
        ax.set_title("Specialisation — % of each tech's repairs by component")
        plt.colorbar(im, ax=ax, label="row-normalised share")
        plt.tight_layout(); plt.show()""")

    return c


# ---------------------------------------------------------------------------
# 05 — Per-machine
# ---------------------------------------------------------------------------


def _make_05_per_machine() -> Cells:
    c: Cells = []
    _md(c, """# 05 — Per-machine analysis

How time on each machine was split between productive processing,
maintenance (broken / waiting for repair), and how often each broke
down.  Identifies the bottleneck and reliability-sensitive machines.""")

    _code(c, PREAMBLE)
    _code(c, SELECT_RUN)

    _md(c, """## Total maintenance / production time per machine across the run""")
    _code(c, """train = ep[ep["phase"] == "train"]

maint = train[group_columns(train, "machine_maintenance")].sum()
prod  = train[group_columns(train, "machine_production")].sum()
brk   = train[group_columns(train, "machine_breakdowns")].sum()
for s, p in ((maint, "machine_maintenance"), (prod, "machine_production"), (brk, "machine_breakdowns")):
    s.index = strip_prefix(list(s.index), p)

if not maint.empty:
    df = pd.concat([prod, maint], axis=1, keys=["production", "maintenance"]).fillna(0)
    df = df.sort_values("maintenance", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.32 * len(df))))
    df.plot.barh(stacked=True, ax=ax, color=["C2", "C3"], edgecolor="black", linewidth=0.3)
    ax.set(xlabel="cumulative sim time (across episodes)", title="Time-on-machine breakdown")
    plt.tight_layout(); plt.show()""")

    _md(c, """## Breakdowns per machine""")
    _code(c, """if not brk.empty:
    s = brk.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(3, 0.32 * len(s))))
    s.plot.barh(ax=ax, color="C3", edgecolor="black", linewidth=0.3)
    ax.invert_yaxis()
    ax.set(xlabel="total breakdowns (across episodes)", title="Reliability — breakdowns per machine")
    plt.tight_layout(); plt.show()""")

    _md(c, """## Utilisation = production / (production + maintenance)""")
    _code(c, """if not maint.empty:
    util = prod / (prod + maint).replace(0, np.nan)
    util = util.sort_values()
    fig, ax = plt.subplots(figsize=(10, max(3, 0.32 * len(util))))
    util.plot.barh(ax=ax, color="C0", edgecolor="black", linewidth=0.3)
    ax.set(xlabel="production / (production + maintenance)", title="Per-machine utilisation",
           xlim=(0, 1))
    plt.tight_layout(); plt.show()""")

    return c


# ---------------------------------------------------------------------------
# 06 — Cross-experiment comparison
# ---------------------------------------------------------------------------


def _make_06_compare_runs() -> Cells:
    c: Cells = []
    _md(c, """# 06 — Compare multiple runs

Place several `exp_id`s in the list below to compare them side-by-side
(e.g. PPO vs heuristic baselines, different seeds, ablations).""")

    _code(c, PREAMBLE)

    _code(c, '''# Edit this list — defaults to "the 3 most recently modified runs".
runs_df = list_runs()
EXP_IDS = runs_df[runs_df["has_episodes"]].head(3)["exp_id"].tolist()
print("Comparing:", EXP_IDS)
data = load_runs(EXP_IDS, with_steps=False)''')

    _md(c, """## Return curves overlaid""")
    _code(c, """fig, ax = plt.subplots()
for exp_id, (ep, _, _) in data.items():
    train = ep[ep["phase"] == "train"]
    if train.empty: continue
    smooth = rolling_mean(train["return"], window=max(3, len(train)//30 or 1))
    ax.plot(train["episode"], smooth, label=exp_id, lw=2)
ax.set(xlabel="episode", ylabel="return (rolling avg)", title="Train return")
ax.legend(fontsize=8)
plt.show()""")

    _md(c, """## Final-return distribution per run""")
    _code(c, """rows = []
for exp_id, (ep, _, _) in data.items():
    train = ep[ep["phase"] == "train"]
    tail_n = min(20, max(1, len(train) // 10))
    last = train["return"].tail(tail_n)
    for v in last:
        rows.append({"exp_id": exp_id, "return": v})
final_df = pd.DataFrame(rows)

if not final_df.empty:
    fig, ax = plt.subplots()
    final_df.boxplot(column="return", by="exp_id", ax=ax, rot=20)
    ax.set_title("Distribution of returns in the last 10% of training")
    plt.suptitle("")
    ax.set_ylabel("return")
    plt.tight_layout(); plt.show()""")

    _md(c, """## Headline-metric comparison table""")
    _code(c, """rows = []
for exp_id, (ep, _, cfg) in data.items():
    train = ep[ep["phase"] == "train"]
    tail = train.tail(max(1, len(train) // 10))
    rows.append({
        "exp_id": exp_id,
        "agent": cfg.get("agent", {}).get("agent_type"),
        "seed": cfg.get("experiment", {}).get("seed"),
        "n_train_episodes": int(len(train)),
        "mean_return_tail": float(tail["return"].mean()) if not tail.empty else float("nan"),
        "best_return": float(train["return"].max()) if not train.empty else float("nan"),
        "total_wall_clock_s": float(train["wall_clock_seconds"].sum()) if "wall_clock_seconds" in train.columns else float("nan"),
    })
pd.DataFrame(rows).set_index("exp_id").round(3)""")

    return c


# ---------------------------------------------------------------------------
# Build & write
# ---------------------------------------------------------------------------


def _to_notebook(cells: Cells) -> dict:
    nb_cells = []
    for cell_type, source in cells:
        cell = {"cell_type": cell_type, "metadata": {}, "source": source}
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOKS: list[tuple[str, callable]] = [
    ("00_overview.ipynb", _make_00_overview),
    ("01_learning_curves.ipynb", _make_01_learning_curves),
    ("02_reward_decomposition.ipynb", _make_02_reward_decomposition),
    ("03_episode_kpis.ipynb", _make_03_episode_kpis),
    ("04_per_technician.ipynb", _make_04_per_technician),
    ("05_per_machine.ipynb", _make_05_per_machine),
    ("06_compare_runs.ipynb", _make_06_compare_runs),
]


def main() -> None:
    here = Path(__file__).resolve().parent
    for filename, make in NOTEBOOKS:
        nb = _to_notebook(make())
        path = here / filename
        path.write_text(json.dumps(nb, indent=1))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

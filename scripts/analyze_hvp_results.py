"""Analyse the human-vs-performance benchmark artefacts.

Reads ``reports/hvp_eval/<scenario>/{episodes.csv,steps.csv.gz}`` and
produces, per scenario:

* ``horizon_metrics.csv``  — per (agent, episode, window) metrics where the
  windows split each episode's sim-time axis at the quartiles:
  short = [0, T/4), medium = [T/4, T/2), long = [T/2, T]
* ``kpi_table.tex``        — episode-level KPI comparison (mean ± std),
  best value per metric in bold
* ``horizon_table.tex``    — per-window MTTR / throughput / availability /
  knowledge-growth comparison, best per column in bold
* ``rank_summary.csv``     — mean rank per agent across directional metrics

Run after ``scripts/eval_human_vs_performance.py``::

    uv run python scripts/analyze_hvp_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("reports/hvp_eval")
SCENARIOS = ["massive_scale", "small_scale", "baseline", "very_long"]

AGENT_LABELS = {
    "human": r"HC-RL (ours)",
    "performance": r"PO-RL (ours)",
    "random": "Random",
    "round_robin": "RoundRobin",
    "least_busy": "LeastBusy",
    "least_fatigued": "LeastFatigued",
    "shortest_queue": "ShortestQueue",
}
AGENT_ORDER = list(AGENT_LABELS)

# Episode-level KPIs: (column, pretty name, direction, format)
# direction: +1 higher-better, -1 lower-better, 0 report-only (no bold, no rank)
KPI_SPEC = [
    ("finished_products", "Finished", +1, "{:.0f}"),
    ("throughput_rate", r"Thrpt.\ rate", +1, "{:.2f}"),
    ("mttr", "MTTR", -1, "{:.1f}"),
    ("mtbf", "MTBF", 0, "{:.0f}"),
    ("fleet_availability_rate", "Avail.", +1, "{:.3f}"),
    ("workload_balance", "Balance", +1, "{:.3f}"),
    ("ill_technician_count", "Disrupt.", -1, "{:.1f}"),
    ("total_breakdowns", "Breakd.", 0, "{:.0f}"),
]

# Windowed metrics: (key, pretty, direction, format)
WINDOW_SPEC = [
    ("mttr", "MTTR", -1, "{:.1f}"),
    ("thrpt", "Thrpt", +1, "{:.2f}"),
    ("avail", "Avail", +1, "{:.3f}"),
    ("dknow_k", r"$\Delta$Know", +1, "{:.1f}"),
]

WINDOWS = ["short", "medium", "long"]


def window_of(sim_time: np.ndarray, horizon: float) -> np.ndarray:
    q1, q2 = horizon / 4.0, horizon / 2.0
    return np.where(sim_time < q1, "short", np.where(sim_time < q2, "medium", "long"))


def horizon_metrics(steps: pd.DataFrame, horizon: float) -> pd.DataFrame:
    """Per (agent, episode, window) metrics from the step series."""
    rows = []
    for (agent, ep), g in steps.groupby(["agent", "episode"], sort=False):
        g = g.sort_values("sim_time")
        win = window_of(g["sim_time"].to_numpy(), horizon)
        for w in WINDOWS:
            sel = g[win == w]
            lo = {"short": 0.0, "medium": horizon / 4, "long": horizon / 2}[w]
            hi = {"short": horizon / 4, "medium": horizon / 2, "long": horizon}[w]
            dur = hi - lo
            if len(sel) == 0:
                rows.append({"agent": agent, "episode": ep, "window": w,
                             "n_decisions": 0, "mttr": np.nan, "thrpt": np.nan,
                             "avail": np.nan, "queue": np.nan, "quality": np.nan,
                             "rtd_per": np.nan, "fat_mean": np.nan,
                             "fat_std": np.nan, "dknow": np.nan, "dknow_k": np.nan})
                continue
            # windowed throughput: finished-product delta across the window,
            # anchored on the last observation before the window opens.
            # The denominator is the *covered* duration — an episode that
            # terminates inside the window (step cap) must not have its
            # rate diluted by time it never simulated.
            before = g[g["sim_time"] < lo]
            fin_lo = float(before["finished_products"].iloc[-1]) if len(before) else 0.0
            fin_hi = float(sel["finished_products"].iloc[-1])
            t_end = float(g["sim_time"].max())
            covered = max(1e-9, min(hi, t_end) - lo)
            know_lo = float(before["fleet_knowledge"].iloc[-1]) if len(before) else float(sel["fleet_knowledge"].iloc[0])
            know_hi = float(sel["fleet_knowledge"].iloc[-1])
            rows.append({
                "agent": agent, "episode": ep, "window": w,
                "n_decisions": int(len(sel)),
                "covered": covered,
                "mttr": float(sel["mttr_rolling"].mean()),
                "thrpt": (fin_hi - fin_lo) / covered * 1000.0,  # products / 1k time
                "avail": float(sel["fleet_availability"].mean()),
                "queue": float(sel["queue_size"].mean()),
                "quality": float(sel["repair_quality"].mean()),
                "rtd_per": float(sel["repair_time_delta_per"].mean()),
                "fat_mean": float(sel["fatigue_mean"].mean()),
                "fat_std": float(sel["fatigue_std"].mean()),
                "dknow": know_hi - know_lo,
                "dknow_k": (know_hi - know_lo) / 1000.0,
            })
    return pd.DataFrame(rows)


def _fmt_pm(mean: float, std: float, fmt: str, *, bold: bool, n: int) -> str:
    if np.isnan(mean):
        return "--"
    body = fmt.format(mean)
    if n > 1 and not np.isnan(std):
        body += r" \tiny$\pm$" + fmt.format(std)
    return r"\textbf{" + body + "}" if bold else body


def kpi_table(episodes: pd.DataFrame, scenario: str) -> tuple[str, pd.DataFrame]:
    """Episode-level KPI LaTeX rows + per-agent rank frame."""
    agents = [a for a in AGENT_ORDER if a in set(episodes["agent"])]
    stats = {}
    for a in agents:
        sub = episodes[episodes["agent"] == a]
        stats[a] = {c: (sub[c].mean(), sub[c].std(ddof=0), len(sub))
                    for c, *_ in KPI_SPEC if c in sub.columns}
    # best per metric + ranks
    best, ranks = {}, {a: [] for a in agents}
    for c, _, direction, _ in KPI_SPEC:
        if direction == 0 or c not in stats[agents[0]]:
            continue
        vals = {a: stats[a][c][0] for a in agents}
        ordered = sorted(agents, key=lambda a: -direction * vals[a])
        best[c] = ordered[0]
        for r, a in enumerate(ordered):
            ranks[a].append(r + 1)
    rank_df = pd.DataFrame({
        "agent": agents,
        "mean_rank": [float(np.mean(ranks[a])) for a in agents],
    }).sort_values("mean_rank")

    best_overall = rank_df.iloc[0]["agent"]
    lines = []
    for a in agents:
        cells = [AGENT_LABELS[a]]
        for c, _, direction, fmt in KPI_SPEC:
            if c not in stats[a]:
                cells.append("--")
                continue
            mean, std, n = stats[a][c]
            cells.append(_fmt_pm(mean, std, fmt,
                                 bold=(best.get(c) == a), n=n))
        mr = float(rank_df.loc[rank_df["agent"] == a, "mean_rank"].iloc[0])
        mr_txt = f"{mr:.2f}"
        if a == best_overall:
            mr_txt = r"\textbf{" + mr_txt + "}"
        cells.append(mr_txt)
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines), rank_df


def horizon_table(hm: pd.DataFrame) -> str:
    """Per-window LaTeX rows: agents x (window x WINDOW_SPEC)."""
    agents = [a for a in AGENT_ORDER if a in set(hm["agent"])]
    cell_stats: dict[tuple, tuple] = {}
    for w in WINDOWS:
        for key, _, direction, _ in WINDOW_SPEC:
            for a in agents:
                sub = hm[(hm["agent"] == a) & (hm["window"] == w)][key]
                cell_stats[(a, w, key)] = (sub.mean(), sub.std(ddof=0), len(sub))
    best: dict[tuple, str] = {}
    for w in WINDOWS:
        for key, _, direction, _ in WINDOW_SPEC:
            vals = {a: cell_stats[(a, w, key)][0] for a in agents}
            valid = {a: v for a, v in vals.items() if not np.isnan(v)}
            if valid:
                best[(w, key)] = max(valid, key=lambda a: direction * valid[a])
    lines = []
    for a in agents:
        cells = [AGENT_LABELS[a]]
        for w in WINDOWS:
            for key, _, _, fmt in WINDOW_SPEC:
                mean, std, n = cell_stats[(a, w, key)]
                cells.append(_fmt_pm(mean, std, fmt,
                                     bold=(best.get((w, key)) == a), n=0))
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines)


def main() -> int:
    all_ranks = []
    for scenario in SCENARIOS:
        d = ROOT / scenario
        if not (d / "episodes.csv").is_file():
            print(f"[skip] {scenario}: no artefacts yet")
            continue
        manifest = json.loads((d / "manifest.json").read_text())
        horizon = float(manifest["max_eval_sim_time"])
        episodes = pd.read_csv(d / "episodes.csv")
        steps = pd.read_csv(d / "steps.csv.gz")

        hm = horizon_metrics(steps, horizon)
        hm.to_csv(d / "horizon_metrics.csv", index=False)

        kpi_rows, rank_df = kpi_table(episodes, scenario)
        (d / "kpi_table.tex").write_text(kpi_rows + "\n")
        rank_df.to_csv(d / "rank_summary.csv", index=False)
        all_ranks.append(rank_df.assign(scenario=scenario))

        hz_rows = horizon_table(hm)
        (d / "horizon_table.tex").write_text(hz_rows + "\n")

        print(f"=== {scenario} (horizon {horizon:.0f}) ===")
        print(rank_df.to_string(index=False))
        # quick human-vs-performance readout per window
        pivot = hm.groupby(["agent", "window"])[["mttr", "thrpt", "dknow", "fat_std"]].mean()
        print(pivot.loc[[a for a in ["human", "performance"] if a in pivot.index.get_level_values(0)]]
              .to_string())
        print()

    if all_ranks:
        overall = (pd.concat(all_ranks)
                   .groupby("agent")["mean_rank"].mean()
                   .sort_values())
        print("=== overall mean rank across scenarios ===")
        print(overall.to_string())
        overall.to_csv(ROOT / "overall_rank.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

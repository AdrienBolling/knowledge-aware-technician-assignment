"""Paper figures for the human-vs-performance benchmark.

Produces two PDF figures in ``paper/figures/``:

* ``results_over_time.pdf``  — 3x2 grid: one row per scenario, columns =
  rolling MTTR / windowed throughput rate vs simulation time, all seven
  agents.  Curves are rolling-mean smoothed; shaded bands mark the
  short / medium / long horizon windows used by the tables.
* ``results_very_long.pdf`` — same two panels for the very-long-horizon
  industrial run (generated only when its artefacts exist).
* ``results_ablation_hvp.pdf`` — 1x3 row: fleet knowledge volume, fleet
  fatigue dispersion (std), and mean fatigue over time for the two
  trained agents only (the objectives ablation).

Run after ``scripts/eval_human_vs_performance.py``::

    uv run python scripts/plot_hvp_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("reports/hvp_eval")
OUT = Path("paper/figures")

SCENARIOS = [
    ("massive_scale", "Industrial scale"),
    ("small_scale", "Small scale"),
    ("baseline", "Baseline scale"),
]

# Categorical slots in the palette's fixed order (CVD-validated).
PALETTE = {
    "human": "#2a78d6",           # blue
    "performance": "#1baf7a",     # aqua
    "random": "#eda100",          # yellow
    "round_robin": "#008300",     # green
    "least_busy": "#4a3aa7",      # violet
    "least_fatigued": "#e34948",  # red
    "shortest_queue": "#e87ba4",  # magenta
}
LABELS = {
    "human": "HC-RL (ours)",
    "performance": "PO-RL (ours)",
    "random": "Random",
    "round_robin": "RoundRobin",
    "least_busy": "LeastBusy",
    "least_fatigued": "LeastFatigued",
    "shortest_queue": "ShortestQueue",
}
TRAINED = ("human", "performance")

GRID = "#e1e0d9"
MUTED = "#898781"
INK = "#52514e"

plt.rcParams.update({
    "font.size": 8,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def _grid_series(g: pd.DataFrame, col: str, tgrid: np.ndarray) -> np.ndarray:
    """Interpolate one episode's series onto the uniform time grid."""
    t = g["sim_time"].to_numpy()
    v = g[col].to_numpy(dtype=float)
    ok = ~np.isnan(v)
    if ok.sum() < 2:
        return np.full_like(tgrid, np.nan, dtype=float)
    return np.interp(tgrid, t[ok], v[ok], left=np.nan, right=np.nan)


def mean_curve(steps: pd.DataFrame, agent: str, col: str,
               tgrid: np.ndarray) -> np.ndarray:
    eps = []
    for _, g in steps[steps["agent"] == agent].groupby("episode"):
        eps.append(_grid_series(g.sort_values("sim_time"), col, tgrid))
    return np.nanmean(np.vstack(eps), axis=0) if eps else np.full_like(tgrid, np.nan)


def smooth(curve: np.ndarray, frac: float = 0.05) -> np.ndarray:
    """Rolling-mean smoothing over ``frac`` of the series length."""
    w = max(3, int(len(curve) * frac))
    kernel = np.ones(w) / w
    pad = np.pad(curve, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(pad, kernel, mode="valid")


def throughput_rate_curve(steps: pd.DataFrame, agent: str,
                          tgrid: np.ndarray) -> np.ndarray:
    """Windowed throughput: d(finished)/dt on the uniform grid, smoothed,
    in products per 1k sim-time units."""
    eps = []
    for _, g in steps[steps["agent"] == agent].groupby("episode"):
        fin = _grid_series(g.sort_values("sim_time"), "finished_products", tgrid)
        rate = np.gradient(fin, tgrid) * 1000.0
        eps.append(rate)
    if not eps:
        return np.full_like(tgrid, np.nan)
    return smooth(np.nanmean(np.vstack(eps), axis=0))


def _style(agent: str) -> dict:
    is_trained = agent in TRAINED
    return dict(
        color=PALETTE[agent],
        linewidth=1.8 if is_trained else 0.9,
        alpha=1.0 if is_trained else 0.85,
        zorder=3 if is_trained else 2,
    )


def _q_markers(ax, horizon: float, *, label_windows: bool = False) -> None:
    """Dashed Q1/Q2 boundaries + optional short/medium/long band labels."""
    q1, q2 = horizon / 4 / 1000.0, horizon / 2 / 1000.0
    hk = horizon / 1000.0
    ax.axvspan(0, q1, color="#f0efec", zorder=0)
    ax.axvspan(q2, hk, color="#f0efec", zorder=0)
    for q in (q1, q2):
        ax.axvline(q, color=MUTED, linewidth=0.6,
                   linestyle=(0, (4, 3)), zorder=1)
    if label_windows:
        for x, txt in ((q1 / 2, "short"), ((q1 + q2) / 2, "medium"),
                       ((q2 + hk) / 2, "long")):
            ax.text(x, 0.98, txt, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6, color=MUTED)


def fig_over_time() -> None:
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 6.2))
    for ri, (scenario, title) in enumerate(SCENARIOS):
        d = ROOT / scenario
        steps = pd.read_csv(d / "steps.csv.gz")
        horizon = float(json.loads((d / "manifest.json").read_text())["max_eval_sim_time"])
        tgrid = np.linspace(horizon * 0.01, horizon, 320)
        tk = tgrid / 1000.0

        ax_m, ax_t = axes[ri, 0], axes[ri, 1]
        for agent in PALETTE:
            m = smooth(mean_curve(steps, agent, "mttr_rolling", tgrid))
            ax_m.plot(tk, m, label=LABELS[agent], **_style(agent))
            r = throughput_rate_curve(steps, agent, tgrid)
            ax_t.plot(tk, r, label=LABELS[agent], **_style(agent))
        for ax in (ax_m, ax_t):
            _q_markers(ax, horizon, label_windows=(ri == 0))
        # robust MTTR y-window: disruption spikes are real but rare;
        # clip the axis so the persistent band stays readable.
        ydata = np.concatenate([ln.get_ydata() for ln in ax_m.get_lines()
                                if len(ln.get_ydata()) > 10])
        ydata = ydata[~np.isnan(ydata)]
        ax_m.set_ylim(np.nanmin(ydata) * 0.93, np.nanpercentile(ydata, 97.5) * 1.08)
        ax_m.set_ylabel(f"{title}\nrolling MTTR")
        ax_t.set_ylabel("thrpt (prod. / $10^3$ t.u.)")
        if ri == 2:
            ax_m.set_xlabel(r"simulation time ($\times 10^3$)")
            ax_t.set_xlabel(r"simulation time ($\times 10^3$)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=4,
                     bbox_to_anchor=(0.5, -0.005), fontsize=7,
                     columnspacing=1.0, handlelength=1.4,
                     title="lines: rolling mean over 5% of the horizon",
                     title_fontsize=6.5)
    leg.get_title().set_color(INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "results_over_time.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "results_over_time.pdf")


def fig_very_long() -> None:
    d = ROOT / "very_long"
    if not (d / "steps.csv.gz").is_file():
        print("[skip] very_long: no artefacts yet")
        return
    steps = pd.read_csv(d / "steps.csv.gz")
    horizon = float(json.loads((d / "manifest.json").read_text())["max_eval_sim_time"])
    tgrid = np.linspace(horizon * 0.01, horizon, 480)
    tk = tgrid / 1000.0
    fig, (ax_m, ax_t) = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for agent in PALETTE:
        m = smooth(mean_curve(steps, agent, "mttr_rolling", tgrid))
        ax_m.plot(tk, m, label=LABELS[agent], **_style(agent))
        r = throughput_rate_curve(steps, agent, tgrid)
        ax_t.plot(tk, r, label=LABELS[agent], **_style(agent))
    for ax in (ax_m, ax_t):
        _q_markers(ax, horizon, label_windows=True)
        ax.set_xlabel(r"simulation time ($\times 10^3$)")
    ydata = np.concatenate([ln.get_ydata() for ln in ax_m.get_lines()
                            if len(ln.get_ydata()) > 10])
    ydata = ydata[~np.isnan(ydata)]
    ax_m.set_ylim(np.nanmin(ydata) * 0.93, np.nanpercentile(ydata, 97.5) * 1.08)
    ax_m.set_ylabel("rolling MTTR")
    ax_t.set_ylabel("thrpt (prod. / $10^3$ t.u.)")
    handles, labels = ax_m.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=7,
                     bbox_to_anchor=(0.5, -0.09), fontsize=6.5,
                     columnspacing=0.9, handlelength=1.3,
                     title="lines: rolling mean over 5% of the horizon",
                     title_fontsize=6)
    leg.get_title().set_color(INK)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(OUT / "results_very_long.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "results_very_long.pdf")


def fig_ablation() -> None:
    panels = [
        ("fleet_knowledge", "fleet knowledge volume"),
        ("fatigue_std", "fatigue dispersion (std)"),
        ("fatigue_mean", "mean fatigue"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    scenario = "massive_scale"
    d = ROOT / scenario
    steps = pd.read_csv(d / "steps.csv.gz")
    horizon = float(json.loads((d / "manifest.json").read_text())["max_eval_sim_time"])
    tgrid = np.linspace(horizon * 0.01, horizon, 320)
    tk = tgrid / 1000.0
    for ax, (col, ylabel) in zip(axes, panels):
        for agent in TRAINED:
            curve = mean_curve(steps, agent, col, tgrid)
            if col != "fleet_knowledge":
                curve = smooth(curve)
            ax.plot(tk, curve, label=LABELS[agent], **_style(agent))
        _q_markers(ax, horizon)
        ax.set_xlabel(r"simulation time ($\times 10^3$)")
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "results_ablation_hvp.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "results_ablation_hvp.pdf")


if __name__ == "__main__":
    fig_over_time()
    fig_very_long()
    fig_ablation()

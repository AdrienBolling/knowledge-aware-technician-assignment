"""Financial implications: top-profit policy maps (Rodríguez et al. 2022,
Eq. 5, re-instantiated on FactoReal's own counts).

Per episode three counts suffice, all already in the benchmark tables:

    P  finished products
    M  maintenance interventions (completed repairs)
    D  human disruptions (technician absence events)

    profit       = pi_p * P - c_m * M - c_h * D
    profit ratio R = P - (h * D + m * M)        (divided by pi_p)

with h = c_h / pi_p the cost of one disruption and m = c_m / pi_p the cost
of one intervention, both as fractions of the profit of one product.  The
(h, m) plane is swept; each cell is coloured by the policy with the highest
R, the dotted line is that policy's break-even (R = 0), and the region
beyond it — where every policy of the roster runs at a loss — is washed
out (Rodríguez's light colours).

Inputs: reports/hvp_eval_v6w/<scenario>/episodes.csv (the paper's own
episodes); optional exhaustion-only disruptions apply the per-type shares
of reports/hvp_eval_disr to the v6w totals.

Outputs (``--out``, default ``financial/``):
    agent_scenario.csv        P, M, D per (agent, scenario) + slopes and break-evens
    profit_maps/<scenario>.csv  per-cell winner and best R over the (h, m) sweep
    map_summary.csv           share of the gain region won by each agent
    frontier_summary.csv      leader at the origin, first challenger per axis, break-evens
    frontier_table.tex        the same as LaTeX rows
    paper/figures/panels/profit_<scenario>.pdf + profit_legend.pdf  (LaTeX subfigures)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = "reports/hvp_eval_v6w"
DISR = "reports/hvp_eval_disr"
SCEN = {"small_scale": "S1 Small", "baseline": "S2 Baseline",
        "massive_scale": "S3 Industrial", "very_long": "S4 Very-long",
        "lifecycle": "S5 Lifecycle"}
SCEN_ORDER = list(SCEN)
DEPLOYABLE = ["hc_v6", "ft_quality", "empirical_topsis", "empirical_spt", "batch_milp",
              "shortest_queue", "least_fatigued", "round_robin", "least_busy",
              "train_weakest", "random", "a2c_mlp", "grpo_mlp", "dql_mlp"]
INFORMED = ["topsis", "shortest_processing", "optimal_assignment", "reserve_specialist",
            "greedy_reward"]
ROSTERS = {"deployable": DEPLOYABLE, "twin": DEPLOYABLE + ["po_v6"],
           "all": DEPLOYABLE + ["po_v6"] + INFORMED}
LABEL = {"hc_v6": "HTT-RL", "ft_quality": r"HTT-RL$_{qua.}$", "po_v6": "PO-HTT-RL",
         "empirical_topsis": "Emp-Topsis", "empirical_spt": "Emp-Spt", "batch_milp": "BatchMilp",
         "shortest_queue": "ShortestQueue (= LeastFatigued)", "least_fatigued": "LeastFatigued",
         "round_robin": "RoundRobin", "least_busy": "LeastBusy", "train_weakest": "TrainWeakest",
         "random": "Random", "a2c_mlp": "A2C-MLP", "grpo_mlp": "GRPO-MLP", "dql_mlp": "DDQN-MLP",
         "topsis": "Topsis*", "shortest_processing": "Spt*", "optimal_assignment": "Hungarian*",
         "reserve_specialist": "ReserveSpec*", "greedy_reward": "GreedyReward*"}
TEX = {"hc_v6": "HTT-RL", "ft_quality": r"HTT-RL\textsubscript{qua.}", "po_v6": "PO-HTT-RL",
       "empirical_topsis": r"\textsc{Emp-Topsis}", "empirical_spt": r"\textsc{Emp-Spt}",
       "batch_milp": r"\textsc{BatchMilp}", "shortest_queue": r"\textsc{ShortestQueue}",
       "least_fatigued": r"\textsc{LeastFatigued}", "round_robin": r"\textsc{RoundRobin}",
       "least_busy": r"\textsc{LeastBusy}", "train_weakest": r"\textsc{TrainWeakest}",
       "random": r"\textsc{Random}", "a2c_mlp": "A2C-MLP", "grpo_mlp": "GRPO-MLP",
       "dql_mlp": "DDQN-MLP", "topsis": r"\textsc{Topsis}$^{*}$",
       "shortest_processing": r"\textsc{Spt}$^{*}$", "optimal_assignment": r"\textsc{Hungarian}$^{*}$",
       "reserve_specialist": r"\textsc{ReserveSpec}$^{*}$", "greedy_reward": r"\textsc{GreedyReward}$^{*}$"}
# Accents shared with the scenario panels (make_scenario_figures.ACCENT); the
# rules that win somewhere get their own hue (Dark2), MLP anchors pale,
# informed baselines in the remaining hues.  ShortestQueue and LeastFatigued
# produce identical episodes under the shared action mask and share a colour.
COLOR = {"hc_v6": "#0072B2", "ft_quality": "#009E73", "po_v6": "#E69F00",
         "empirical_topsis": "#D55E00", "empirical_spt": "#CC79A7", "random": "#4D4D4D",
         "batch_milp": "#7570B3", "least_busy": "#E6AB02", "shortest_queue": "#A6761D",
         "least_fatigued": "#A6761D", "round_robin": "#66A61E", "train_weakest": "#8C8C8C",
         "a2c_mlp": "#E6D8A8", "grpo_mlp": "#D4E6A8", "dql_mlp": "#A8D4E6",
         "topsis": "#F0E442", "shortest_processing": "#56B4E9", "optimal_assignment": "#999933",
         "reserve_specialist": "#882255", "greedy_reward": "#117733"}
GRID = 200
PANEL = (2.1, 1.9)   # inches; 0.32 * 6.5 in = 2.08 in in the manuscript
SWEEP = 1.15         # axes extend 15% past the roster's largest break-even
_DTYPES = ("injury", "exhaustion", "vacation")


def load_counts(disruptions: str = "all") -> pd.DataFrame:
    rows = []
    for s in SCEN_ORDER:
        df = pd.read_csv(f"{ROOT}/{s}/episodes.csv")
        g = df.groupby("agent").agg(n_episodes=("episode", "size"),
                                    P=("finished_products", "mean"),
                                    M=("total_repairs", "mean"),
                                    D=("ill_technician_count", "mean"))
        try:
            dd = pd.read_csv(f"{DISR}/{s}/episodes.csv")
            cols = [f"disruptions_{t}" for t in _DTYPES]
            gd = dd.groupby("agent")[cols].mean()
            g["share_exhaustion"] = gd["disruptions_exhaustion"] / gd[cols].sum(axis=1)
        except FileNotFoundError:
            g["share_exhaustion"] = np.nan
        g["D_all"] = g["D"]
        if disruptions == "exhaustion":
            g["D"] = g["D_all"] * g["share_exhaustion"]
        g["scenario"] = s
        rows.append(g.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out["M_per_product"] = out["M"] / out["P"]
    out["D_per_product"] = out["D"] / out["P"]
    out["break_even_m"] = out["P"] / out["M"]
    out["break_even_h"] = out["P"] / out["D"]
    out["scenario"] = pd.Categorical(out["scenario"], SCEN_ORDER, ordered=True)
    return out.sort_values(["scenario", "P"], ascending=[True, False]).reset_index(drop=True)


def sweep(g: pd.DataFrame, grid: int = GRID):
    """Winner map over (h, m).  ``g`` indexed by agent with P, M, D."""
    P, M, D = g["P"], g["M"], g["D"]
    hmax = SWEEP * float((P / D).max())
    mmax = SWEEP * float((P / M).max())
    hs = np.linspace(0.0, hmax, grid)
    ms = np.linspace(0.0, mmax, grid)
    H, Mg = np.meshgrid(hs, ms, indexing="ij")
    agents = list(g.index)
    stack = np.stack([P[a] - H * D[a] - Mg * M[a] for a in agents])
    widx = stack.argmax(axis=0)
    best = stack.max(axis=0)
    return hs, ms, widx, best, agents, stack


def break_even(g, leader, other, key):
    """Cost on one axis (the other at 0) at which ``other`` overtakes
    ``leader`` while still profitable itself; None if never, 0 if
    ``other`` already leads at the origin."""
    dP = g.loc[leader, "P"] - g.loc[other, "P"]
    dq = g.loc[leader, key] - g.loc[other, key]
    if dP <= 0:
        return 0.0
    if dq <= 0:
        return None
    x = float(dP / dq)
    return x if x <= g.loc[other, "P"] / g.loc[other, key] else None


def first_challenger(g, leader, key):
    best_ag, best_x = None, np.inf
    for a in g.index:
        if a == leader:
            continue
        x = break_even(g, leader, a, key)
        if x is not None and 0 < x < best_x:
            best_ag, best_x = a, x
    return best_ag, best_x


def draw_panel(ax, hs, ms, widx, best, agents, title):
    img = np.zeros(widx.shape + (3,))
    for i, a in enumerate(agents):
        img[widx == i] = matplotlib.colors.to_rgb(COLOR.get(a, "#000000"))
    loss = best < 0
    img[loss] = img[loss] * 0.3 + 0.7          # washed out: every policy at a loss
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", aspect="auto",
              extent=(0, hs[-1], 0, ms[-1]), interpolation="nearest")
    H, Mg = np.meshgrid(hs, ms, indexing="ij")
    cs = ax.contour(H, Mg, best, levels=[0.0], colors="black", linewidths=0.9,
                    linestyles="dotted")   # dotted profit border
    for coll in getattr(cs, "collections", [cs]):
        coll.set_dashes([(0, (1.2, 1.6))]) if hasattr(coll, "set_dashes") else None
    ax.set_xlabel(r"$h$: disruption cost (products)")
    ax.set_ylabel(r"$m$: intervention cost (products)")
    ax.set_title(title)
    ax.tick_params(length=2, pad=1.5)
    return {agents[i] for i in np.unique(widx)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="financial")
    ap.add_argument("--figures", default="paper/figures/panels")
    ap.add_argument("--roster", default="deployable", choices=list(ROSTERS))
    ap.add_argument("--disruptions", default="all", choices=["all", "exhaustion"])
    ap.add_argument("--grid", type=int, default=GRID)
    ap.add_argument("--fig-suffix", default="")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "profit_maps").mkdir(parents=True, exist_ok=True)
    counts = load_counts(args.disruptions)
    counts.to_csv(out / "agent_scenario.csv", index=False)
    roster = [a for a in ROSTERS[args.roster]]

    plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.3,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "pdf.fonttype": 42})
    summary, frontier, tex_rows, seen = [], [], [], set()
    for s in SCEN_ORDER:
        g = counts[counts.scenario == s].set_index("agent")
        g = g.loc[[a for a in roster if a in g.index]].dropna(subset=["D"])
        hs, ms, widx, best, agents, stack = sweep(g, args.grid)
        Hh, Mm = np.meshgrid(hs, ms, indexing="ij")
        cells = pd.DataFrame({"h": Hh.ravel(), "m": Mm.ravel(),
                              "winner": np.array(agents)[widx.ravel()],
                              "best_profit_ratio": best.ravel()})
        cells.to_csv(out / "profit_maps" / f"{s}.csv", index=False)
        gain = cells[cells.best_profit_ratio >= 0].winner.value_counts(normalize=True)
        for a in agents:
            summary.append({"scenario": s, "agent": a,
                            "share_of_gain_region": float(gain.get(a, 0.0)),
                            "P": g.loc[a, "P"], "M": g.loc[a, "M"], "D": g.loc[a, "D"]})
        leader = g["P"].idxmax()
        ch, xh = first_challenger(g, leader, "D")
        cm, xm = first_challenger(g, leader, "M")
        frontier.append({"scenario": s, "leader": leader, "leader_P": g.loc[leader, "P"],
                         "h_challenger": ch, "h_break_even": xh if ch else np.nan,
                         "leader_break_even_h": g.loc[leader, "break_even_h"],
                         "m_challenger": cm, "m_break_even": xm if cm else np.nan,
                         "leader_break_even_m": g.loc[leader, "break_even_m"],
                         "leader_M_per_product": g.loc[leader, "M_per_product"],
                         "leader_D_per_product": g.loc[leader, "D_per_product"]})
        fmt = lambda x, c: "---" if c is None else f"{x:.3f} ({TEX[c]})"
        tex_rows.append(f"{SCEN[s]} & {TEX[leader]} & {g.loc[leader, 'P']:,.0f} & "
                        f"{fmt(xh, ch)} & {g.loc[leader, 'break_even_h']:.2f} & "
                        f"{fmt(xm, cm)} & {g.loc[leader, 'break_even_m']:.3f} \\\\")
        seen |= set(agents[i] for i in np.unique(widx))
        if args.no_figures:
            continue
        fig, ax = plt.subplots(figsize=PANEL)
        draw_panel(ax, hs, ms, widx, best, agents, SCEN[s])
        fig.subplots_adjust(left=0.2, right=0.97, bottom=0.21, top=0.89)
        Path(args.figures).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(args.figures) / f"profit_{s}{args.fig_suffix}.pdf")
        plt.close(fig)
        print(f"{SCEN[s]:14s} leader {LABEL[leader]:14s} gain region: "
              + ", ".join(f"{LABEL[a]} {100*v:.0f}%" for a, v in gain.items()))

    pd.DataFrame(summary).to_csv(out / "map_summary.csv", index=False)
    pd.DataFrame(frontier).to_csv(out / "frontier_summary.csv", index=False)
    (out / "frontier_table.tex").write_text("\n".join(tex_rows) + "\n")
    if not args.no_figures:
        order = [a for a in roster if a in seen]
        handles = [Patch(facecolor=COLOR[a], edgecolor="none", label=LABEL[a]) for a in order]
        handles.append(Patch(facecolor="#d9d9d9", edgecolor="none", label="loss region (washed out)"))
        handles.append(Line2D([], [], color="black", lw=0.9, ls=(0, (1.2, 1.6)),
                              label="profit border ($R = 0$ of the best policy)"))
        fig = plt.figure(figsize=PANEL)
        fig.legend(handles=handles, loc="center", frameon=False, fontsize=6.5,
                   handlelength=1.6, labelspacing=0.55, borderaxespad=0)
        fig.savefig(Path(args.figures) / f"profit_legend{args.fig_suffix}.pdf")
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

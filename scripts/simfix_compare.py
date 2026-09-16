#!/usr/bin/env python
"""Compare the simulator-fix measurement configs (scripts/dgy_simfix_measure.sh).

Layout of ``--root``: ``<config>/<agent>/<scenario>/{episodes.csv,steps.csv.gz}``.

Configs (switches of kata.core.legacy):
  legacy  KATA_LEGACY_BUFFER_INTERRUPT=1  KATA_LEGACY_MACHINE_TRACKING=1
  fix1    KATA_LEGACY_BUFFER_INTERRUPT=0  KATA_LEGACY_MACHINE_TRACKING=1
  fixed   KATA_LEGACY_BUFFER_INTERRUPT=0  KATA_LEGACY_MACHINE_TRACKING=0

Output (markdown):
  1. switch check: the switch columns of every episodes.csv against its config;
  2. one KPI table per scenario (S5 first): products and their change against
     the legacy rerun and against the published v6w rows, products lost and
     scrapped, availability, MTBF, breakdowns (filed = exact count, tracked =
     the MTBF denominator), episode-mean MTTR, disruptions per 10^3 products,
     final fleet knowledge;
  3. summary score and rank among the measured agents, per config and per
     scenario, with the logic of scripts/summary_scores.py (paper variant:
     products, episode-mean MTTR, disruptions/10^3 products, final knowledge;
     score = mean % distance to the best value of the field).  The published
     v6w rows of the same agents are scored the same way as a reference.

Learned agents run under ``legacy`` on S3 and S5 only; the published column
is the reference for the other scenarios.  The published v6w runs used other
hardware and no PYTHONHASHSEED pin, so a legacy rerun can differ from them.

Usage (repo root):
  PYTHONPATH=src:scripts python scripts/simfix_compare.py \
      --root reports/simfix --published reports/hvp_eval_v6w \
      --out reports/simfix/SIMFIX_COMPARE.md
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summary_scores import KPIS, PAPER_VARIANT, VARIANTS, score  # noqa: E402

CONFIGS = ("legacy", "fix1", "fixed")
# config -> (legacy_buffer_interrupt, legacy_machine_tracking)
SWITCHES = {"legacy": (1, 1), "fix1": (0, 1), "fixed": (0, 0)}
SCENARIOS = [
    ("lifecycle", "S5 lifecycle"),
    ("very_long", "S4 very long"),
    ("massive_scale", "S3 industrial"),
    ("baseline", "S2 baseline"),
    ("small_scale", "S1 small"),
]
AGENT_ORDER = ["ft_quality", "hc_v6", "topsis", "shortest_processing", "random"]
LABELS = {
    "ft_quality": "HTT-RL qua.",
    "hc_v6": "HTT-RL ref.",
    "topsis": "Topsis",
    "shortest_processing": "Spt",
    "random": "Random",
}
SCORE_KPIS = VARIANTS[PAPER_VARIANT] + ["know"]


def _label(agent: str) -> str:
    return LABELS.get(agent, agent)


def _agent_key(agent: str) -> tuple[int, str]:
    return (AGENT_ORDER.index(agent) if agent in AGENT_ORDER else len(AGENT_ORDER), agent)


def _final_knowledge(steps: Path, agent: str | None = None) -> float:
    """Mean over episodes of the last recorded fleet knowledge, in 10^3."""
    if not steps.is_file():
        return math.nan
    cols = ["agent", "episode", "step", "fleet_knowledge"]
    st = pd.read_csv(steps, usecols=cols)
    if agent is not None:
        st = st[st.agent == agent]
    if st.empty:
        return math.nan
    last = st.sort_values("step").groupby("episode").fleet_knowledge.last()
    return float(last.mean()) / 1e3


def kpis_from_episodes(ep: pd.DataFrame, steps: Path, agent: str) -> dict:
    def col(name: str) -> float:
        return float(ep[name].mean()) if name in ep and len(ep) else math.nan

    prod = col("finished_products")
    ill = col("ill_technician_count")
    return {
        "n": int(len(ep)),
        "prod": prod,
        "lost": col("products_lost"),
        "scrapped": col("products_scrapped"),
        "avail": col("fleet_availability_rate"),
        "mtbf": col("mtbf"),
        "bd_filed": col("breakdowns_filed"),
        "bd_tracked": col("machine_breakdowns"),
        "mttr_mean": col("mttr"),
        "disr": ill / prod * 1000.0 if prod and prod > 0 else math.nan,
        "know": _final_knowledge(steps, agent),
        "sw_buffer": col("legacy_buffer_interrupt"),
        "sw_tracking": col("legacy_machine_tracking"),
        "wall_h": col("wall_s") / 3600.0,
    }


def load_measured(root: Path) -> pd.DataFrame:
    rows = []
    for config in CONFIGS:
        for ep_path in sorted((root / config).glob("*/*/episodes.csv")):
            agent, scenario = ep_path.parent.parent.name, ep_path.parent.name
            ep = pd.read_csv(ep_path)
            ep = ep[ep.agent == agent]
            if ep.empty:
                continue
            k = kpis_from_episodes(ep, ep_path.with_name("steps.csv.gz"), agent)
            rows.append({"config": config, "agent": agent, "scenario": scenario, **k})
    return pd.DataFrame(rows)


def load_published(published: Path | None, pairs: set[tuple[str, str]]) -> pd.DataFrame:
    """Published v6w KPIs for the (agent, scenario) pairs that were measured."""
    if published is None or not published.is_dir():
        return pd.DataFrame()
    rows = []
    for scenario in sorted({s for _, s in pairs}):
        ep_path = published / scenario / "episodes.csv"
        if not ep_path.is_file():
            continue
        ep_all = pd.read_csv(ep_path)
        agents = sorted({a for a, s in pairs if s == scenario})
        steps = published / scenario / "steps.csv.gz"
        st = None
        if steps.is_file():
            st = pd.read_csv(steps, usecols=["agent", "episode", "step", "fleet_knowledge"])
            st = st[st.agent.isin(agents)]
        for agent in agents:
            ep = ep_all[ep_all.agent == agent]
            if ep.empty:
                continue
            k = kpis_from_episodes(ep, Path("/nonexistent"), agent)
            if st is not None:
                s = st[st.agent == agent]
                if not s.empty:
                    last = s.sort_values("step").groupby("episode").fleet_knowledge.last()
                    k["know"] = float(last.mean()) / 1e3
            rows.append({"config": "published", "agent": agent, "scenario": scenario, **k})
    return pd.DataFrame(rows)


def _fmt(v: float, spec: str) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return format(v, spec)


def _pct(new: float, ref: float) -> str:
    if not (math.isfinite(new) and math.isfinite(ref)) or ref == 0:
        return "n/a"
    return f"{100.0 * (new / ref - 1.0):+.2f}%"


def switch_check(df: pd.DataFrame) -> list[str]:
    out = ["## Switch check", ""]
    bad = []
    for _, r in df.iterrows():
        want = SWITCHES[r['config']]
        got = (r['sw_buffer'], r['sw_tracking'])
        if not all(math.isfinite(g) for g in got) or tuple(int(g) for g in got) != want:
            bad.append(f"- {r['config']}/{r['agent']}/{r['scenario']}: switches {got}, expected {want}")
    if bad:
        out += ["**Mismatch** between the episodes.csv switch columns and the config:", "", *bad]
    else:
        out.append(f"All {len(df)} parts ran with the switches of their config.")
    return out + [""]


def scenario_tables(df: pd.DataFrame, pub: pd.DataFrame) -> list[str]:
    out = []
    for scenario, title in SCENARIOS:
        sub = df[df.scenario == scenario]
        if sub.empty:
            continue
        out += [
            f"## {title} (`{scenario}`)",
            "",
            "| Agent | Config | Eps | Products | Δ% vs legacy rerun | Pub. v6w products "
            "| Δ% vs published v6w | Lost | Scrapped | Availability | MTBF "
            "| Breakdowns filed | Breakdowns tracked | MTTR (ep. mean) "
            "| Disr./10³ prod. | Final knowledge ×10³ |",
            "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
        ]
        for agent in sorted(sub.agent.unique(), key=_agent_key):
            legacy = sub[(sub.agent == agent) & (sub.config == "legacy")]
            legacy_prod = float(legacy["prod"].iloc[0]) if len(legacy) else math.nan
            p = pub[(pub.agent == agent) & (pub.scenario == scenario)] if len(pub) else pub
            pub_prod = float(p["prod"].iloc[0]) if len(p) else math.nan
            for config in CONFIGS:
                r = sub[(sub.agent == agent) & (sub.config == config)]
                if r.empty:
                    continue
                r = r.iloc[0]
                out.append(
                    f"| {_label(agent)} | {config} | {r['n']} | {_fmt(r['prod'], '.1f')} "
                    f"| {_pct(r['prod'], legacy_prod) if config != 'legacy' else '—'} "
                    f"| {_fmt(pub_prod, '.1f')} | {_pct(r['prod'], pub_prod)} "
                    f"| {_fmt(r['lost'], '.1f')} | {_fmt(r['scrapped'], '.1f')} "
                    f"| {_fmt(r['avail'], '.3f')} | {_fmt(r['mtbf'], '.0f')} "
                    f"| {_fmt(r['bd_filed'], '.0f')} | {_fmt(r['bd_tracked'], '.0f')} "
                    f"| {_fmt(r['mttr_mean'], '.1f')} | {_fmt(r['disr'], '.0f')} "
                    f"| {_fmt(r['know'], '.1f')} |"
                )
        out += [""]
    return out


def score_tables(df: pd.DataFrame, pub: pd.DataFrame) -> list[str]:
    kpi_names = ", ".join(KPIS[k][1] for k in SCORE_KPIS)
    out = [
        "## Summary score and rank among the measured agents",
        "",
        f"Score = mean % distance to the best value of the field (lower is better); "
        f"KPIs: {kpi_names} (the `{PAPER_VARIANT}` variant of scripts/summary_scores.py). "
        "The field is the set of agents measured under that config at that scenario, "
        "so fields with fewer agents are not comparable with full fields.",
        "",
    ]
    frames = [("published v6w (same agents)", pub)] if len(pub) else []
    frames += [(c, df[df.config == c]) for c in CONFIGS]
    for name, sub in frames:
        if sub.empty:
            continue
        cells: dict[str, dict[str, str]] = {}
        overall: dict[str, list[float]] = {}
        scen_cols = []
        for scenario, title in SCENARIOS:
            s = sub[sub.scenario == scenario].set_index("agent")
            s = s.dropna(subset=SCORE_KPIS)
            if s.empty:
                continue
            scen_cols.append((scenario, title))
            sc = score(s, SCORE_KPIS, list(s.index))
            ranks = sc.rank(method="min").astype(int)
            for agent, v in sc.items():
                cells.setdefault(agent, {})[scenario] = f"{v:.1f} ({ranks[agent]}/{len(sc)})"
                overall.setdefault(agent, []).append(float(v))
        if not cells:
            continue
        out += [
            f"### {name}",
            "",
            "| Agent | " + " | ".join(t for _, t in scen_cols) + " | Mean over measured scenarios |",
            "|---|" + "|".join(["--:"] * (len(scen_cols) + 1)) + "|",
        ]
        for agent in sorted(cells, key=_agent_key):
            row = [cells[agent].get(s, "—") for s, _ in scen_cols]
            vals = overall[agent]
            row.append(f"{np.mean(vals):.1f} (n={len(vals)})")
            out.append(f"| {_label(agent)} | " + " | ".join(row) + " |")
        out.append("")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("reports/simfix"))
    ap.add_argument("--published", type=Path, default=Path("reports/hvp_eval_v6w"))
    ap.add_argument("--out", type=Path, default=None, help="markdown file (default: stdout)")
    args = ap.parse_args()

    df = load_measured(args.root)
    if df.empty:
        print(f"no episodes.csv under {args.root}/<config>/<agent>/<scenario>/", file=sys.stderr)
        return 1
    pairs = set(zip(df.agent, df.scenario))
    pub = load_published(args.published, pairs)

    lines = [
        "# Simulator-fix measurement",
        "",
        f"Root `{args.root}`; published reference `{args.published}`"
        + ("" if len(pub) else " (not found: published columns are n/a)")
        + f"; {len(df)} parts.",
        "",
        "Configs: `legacy` = both pre-fix switches on; `fix1` = buffer fix only "
        "(decision-boundary tracking kept); `fixed` = both fixes.  "
        "`Lost` = products created − finished − WIP − scrapped (0 unless the simulator "
        "destroys products).  `Breakdowns filed` is exact in every config; "
        "`Breakdowns tracked` is the MTBF denominator (decision-boundary samples under "
        "legacy tracking).  Parts per config: "
        + ", ".join(f"{c} {int((df.config == c).sum())}" for c in CONFIGS)
        + ".",
        "",
    ]
    lines += switch_check(df)
    lines += scenario_tables(df, pub)
    lines += score_tables(df, pub)
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

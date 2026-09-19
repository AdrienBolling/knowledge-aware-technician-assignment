"""Compare the sequential non-RL baselines with the published v6w benchmark.

Reads the parts of ``scripts/dgy_seqbase_queue.sh``
(``<root>/<agent>/<scenario>/{episodes.csv,steps.csv.gz}``) and the published
tree (``reports/hvp_eval_v6w``, through ``summary_scores.load_metrics``).
Per scenario it reports:

1. products, episode-mean MTTR, disruptions per 10^3 products, final fleet
   knowledge and ms per decision (whole decision loop = wall time / decisions;
   the planner alone where the part recorded it) for the new baselines and
   reference policies of the paper;
2. summary scores of the paper variant (``summary_scores.PAPER_VARIANT``:
   mean % distance to the best value per KPI, final knowledge included)
   (a) against the best values of the PUBLISHED field and (b) in a field that
   also contains the new baselines;
3. the rank of each new baseline among the 15 policies of the paper's main
   table (``summary_scores.REP_COLS``), with its score against the published
   bests and in the extended field.

Wall-clock numbers of different trees come from different machines and loads;
compare them within one tree only.

Usage::

    uv run --no-sync python scripts/seqbase_compare.py --root reports/seqbase \\
        [--published reports/hvp_eval_v6w] [--out reports/seqbase/SEQBASE_COMPARE.md]
"""
from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summary_scores as ss  # noqa: E402

NEW = ("rolling_mpc", "greedy_train_reward")
REFERENCE = ("hc_v6", "ft_quality", "topsis", "shortest_processing", "greedy_reward")
LABELS = {**ss.PLAIN, "rolling_mpc": "RollingMpc", "greedy_train_reward": "GreedyTrain"}


def new_metrics(scen_dir: Path) -> pd.Series | None:
    """KPI row of one part, with the columns of ``summary_scores.scenario_metrics``."""
    ep_path = scen_dir / "episodes.csv"
    if not ep_path.is_file():
        return None
    ep = pd.read_csv(ep_path)
    prod = float(ep.finished_products.mean())
    row = {
        "prod": prod, "mttr_mean": float(ep.mttr.mean()),
        "avail": float(ep.fleet_availability_rate.mean()),
        "ill": float(ep.ill_technician_count.mean()), "n": int(ep.episode.nunique()),
    }
    row["disr"] = row["ill"] / prod * 1000.0 if prod > 0 else np.nan
    row["ms_decision"] = float(1000.0 * ep.wall_s.sum() / max(ep.n_steps.sum(), 1))
    row["planner_ms"] = float(ep.planner_ms_median.mean()) if "planner_ms_median" in ep else np.nan
    st_path = scen_dir / "steps.csv.gz"
    if st_path.is_file():
        st = pd.read_csv(st_path, usecols=ss.STEP_COLS)
        per_ep = [(ss.terminal_mttr(h), float(h.sort_values("step").fleet_knowledge.iloc[-1]))
                  for _, h in st.groupby("episode")]
        row["mttr_final"] = float(np.mean([x for x, _ in per_ep]))
        row["know"] = float(np.mean([y for _, y in per_ep])) / 1e3
    else:
        row["mttr_final"], row["know"] = np.nan, np.nan
    return pd.Series(row)


def published_ms(published: Path, scenario: str) -> dict[str, float]:
    ep = pd.read_csv(published / scenario / "episodes.csv")
    g = ep.groupby("agent")[["wall_s", "n_steps"]].sum()
    return (1000.0 * g.wall_s / g.n_steps.clip(lower=1)).to_dict()


def scores_vs(m: pd.DataFrame, kpis: list[str], best_rows: list[str], rows: list[str]) -> pd.Series:
    """Mean % distance to the best value over ``best_rows``, for ``rows``
    (negative = better than that best)."""
    out = pd.Series(0.0, index=rows)
    for k in kpis:
        d, _ = ss.KPIS[k]
        best = m.loc[best_rows, k].max() if d > 0 else m.loc[best_rows, k].min()
        out += ((best - m.loc[rows, k]) if d > 0 else (m.loc[rows, k] - best)) / best * 100.0
    return out / len(kpis)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="reports/seqbase")
    ap.add_argument("--published", default=ss.ROOT)
    ap.add_argument("--parts", default=ss.PARTS)
    ap.add_argument("--extra", default=ss.EXTRA)
    ap.add_argument("--agents", default=",".join(NEW))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    root, published = Path(args.root), Path(args.published)
    agents = [a for a in args.agents.split(",") if a]
    ss.ROOT, ss.PARTS = str(published), args.parts

    notes = io.StringIO()
    with redirect_stdout(notes):
        metrics = ss.load_metrics(args.extra if args.extra and Path(args.extra).is_file() else None)
    field = ss.main_field(metrics)
    base_kpis = ss.VARIANTS[ss.PAPER_VARIANT]
    rep = [k for k, _ in ss.REP_COLS]

    lines = [f"# Sequential baselines vs the published benchmark", "",
             f"root `{root}`, published `{published}` (field: {len(field)} agents), "
             f"score variant `{ss.PAPER_VARIANT}` (" + ", ".join(ss.KPIS[k][1] for k in base_kpis)
             + ", final knowledge)", ""]
    for note in notes.getvalue().splitlines():
        lines.append(f"> {note}")
    per_scen: dict[str, dict] = {}
    for s, short in ss.SCEN:
        m = metrics[s].copy()
        new_rows = {}
        for a in agents:
            row = new_metrics(root / a / s)
            if row is not None:
                new_rows[a] = row
        have = list(new_rows)
        if new_rows:
            m = pd.concat([m.drop(index=have, errors="ignore"), pd.DataFrame(new_rows).T.astype(float)])
        if not have:
            lines += [f"## {short} ({s}): no parts", ""]
            continue
        kpis = base_kpis + (["know"] if s in ss.KNOW_SCEN else [])
        pub_ms = published_ms(published, s)
        lines += [f"## {short} ({s})", "",
                  "| Policy | Episodes | Products | MTTR (ep. mean) | Disr./10^3 products | Final knowledge (x10^3) "
                  "| ms/decision | planner ms (median) |",
                  "|---|--:|--:|--:|--:|--:|--:|--:|"]
        for a in have + [r for r in REFERENCE if r in m.index]:
            r = m.loc[a]
            ms = r.get("ms_decision") if a in have else pub_ms.get(a, np.nan)
            pl = r.get("planner_ms", np.nan) if a in have else np.nan
            lines.append(f"| {LABELS.get(a, a)} | {int(r['n'])} | {r['prod']:.0f} | {r['mttr_mean']:.1f} | "
                         f"{r['disr']:.1f} | {r['know']:.2f} | {ms:.2f} | "
                         + ("" if not np.isfinite(pl) else f"{pl:.2f}") + " |")
        vs_pub = scores_vs(m, kpis, field, have)
        ext = field + have
        vs_ext = scores_vs(m, kpis, ext, ext)
        lines += ["", "| Policy | score vs published bests | score in extended field | rank in extended field "
                  "| rank among the 15 (published bests) | rank among the 15 (extended field) |",
                  "|---|--:|--:|--:|--:|--:|"]
        rep_here = [k for k in rep if k in m.index]
        for a in have:
            rank_ext = int((vs_ext < vs_ext[a]).sum()) + 1
            pub15 = scores_vs(m, kpis, field, rep_here + [a])
            ext15 = vs_ext.loc[rep_here + [a]]
            lines.append(f"| {LABELS.get(a, a)} | {vs_pub[a]:.2f} | {vs_ext[a]:.2f} | {rank_ext} of {len(ext)} | "
                         f"{int((pub15 < pub15[a]).sum()) + 1} of {len(rep_here) + 1} | "
                         f"{int((ext15 < ext15[a]).sum()) + 1} of {len(rep_here) + 1} |")
        per_scen[short] = {"pub": vs_pub, "ext": vs_ext, "pub15": {a: scores_vs(m, kpis, field, rep_here + [a])
                                                                    for a in have}}
        lines.append("")

    if per_scen:
        lines += ["## Overall (mean over the scenarios with parts)", "",
                  "| Policy | Scenarios | score vs published bests | rank among the 15 (published bests) "
                  "| score in extended field | rank among the 15 (extended field) |", "|---|--:|--:|--:|--:|--:|"]
        for a in agents:
            shorts = [s for s in per_scen if a in per_scen[s]["pub"].index]
            if not shorts:
                continue
            pub = float(np.mean([per_scen[s]["pub"][a] for s in shorts]))
            ext = float(np.mean([per_scen[s]["ext"][a] for s in shorts]))
            rep_all = [k for k in rep if all(k in per_scen[s]["ext"].index for s in shorts)]
            pub15 = pd.DataFrame({s: per_scen[s]["pub15"][a] for s in shorts}).loc[rep_all + [a]].mean(axis=1)
            ext15 = pd.DataFrame({s: per_scen[s]["ext"].loc[rep_all + [a]] for s in shorts}).mean(axis=1)
            lines.append(f"| {LABELS.get(a, a)} | {len(shorts)} | {pub:.2f} | "
                         f"{int((pub15 < pub15[a]).sum()) + 1} of {len(rep_all) + 1} | {ext:.2f} | "
                         f"{int((ext15 < ext15[a]).sum()) + 1} of {len(rep_all) + 1} |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

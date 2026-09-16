"""Merge the neutral-turnover S5 parts and write a markdown summary.

Inputs (from scripts/dgy_turnover_queue.sh, pulled to the local repo):
  reports/hvp_turnover_parts/<agent>/<scenario>/   scenario in
      lifecycle_random_retire, lifecycle_random_timing, lifecycle (targeted
      rerun with the same code)
Outputs:
  reports/hvp_eval_turnover/<scenario>/            merged (merge_hvp_parts.py)
  reports/hvp_eval_turnover/summary.md

KPIs per policy: products, episode-mean MTTR, disruptions per 10^3 products,
final fleet knowledge.  Summary score = scripts/summary_scores.py ``score()``
(mean % distance to the field's best value) with KPIs prod, mttr_mean, disr,
know.  The field is the set of evaluated policies present in every column of
the comparison, without the production-only twin (the paper treats it as an
ablation, not a field member); the twin gets its distance to the same best
values.

Usage (from the repo root):
  uv run --no-sync python scripts/turnover_summary.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summary_scores import KPIS, score  # noqa: E402

VARIANTS = {
    "lifecycle_random_retire": "Random retirement (same times)",
    "lifecycle_random_timing": "Random retirement and random timing",
}
TARGETED = "lifecycle"
TWIN = "po_v6"
SCORE_KPIS = ["prod", "mttr_mean", "disr", "know"]
LABELS = {
    "ft_quality": "HTT-RL qua.", "hc_v6": "HTT-RL ref.", "po_v6": "PO twin",
    "topsis": "Topsis*", "shortest_processing": "Spt*",
    "optimal_assignment": "Hungarian*", "reserve_specialist": "ReserveSpec*",
    "shortest_queue": "ShortestQueue", "least_fatigued": "LeastFatigued",
    "round_robin": "RoundRobin", "least_busy": "LeastBusy",
    "train_weakest": "TrainWeakest", "random": "Random",
    "a2c_mlp": "A2C-MLP", "grpo_mlp": "GRPO-MLP", "dql_mlp": "DDQN-MLP",
}


def merge(parts: Path, dest: Path, scenario: str) -> bool:
    if not any(parts.glob(f"*/{scenario}/episodes.csv")):
        return False
    subprocess.run(
        [sys.executable, str(Path(__file__).parent / "merge_hvp_parts.py"),
         "--parts", str(parts), "--dest", str(dest / scenario),
         "--scenario", scenario],
        check=True, stdout=subprocess.DEVNULL,
    )
    logs = [pd.read_csv(p) for p in sorted(parts.glob(f"*/{scenario}/lifecycle_events.csv"))]
    if logs:
        pd.concat(logs, ignore_index=True).to_csv(
            dest / scenario / "lifecycle_events.csv", index=False)
    return True


def metrics(scen_dir: Path, extra_steps: list[Path] = ()) -> pd.DataFrame:
    """Per-agent KPIs; ``extra_steps`` fill agents absent from steps.csv.gz."""
    ep = pd.read_csv(scen_dir / "episodes.csv")
    m = ep.groupby("agent").agg(
        prod=("finished_products", "mean"), mttr_mean=("mttr", "mean"),
        ill=("ill_technician_count", "mean"))
    m["disr"] = m.ill / m["prod"] * 1000.0
    cols = ["agent", "episode", "step", "fleet_knowledge"]
    st = pd.read_csv(scen_dir / "steps.csv.gz", usecols=cols)
    for path in extra_steps:
        missing = set(m.index) - set(st.agent.unique())
        if not missing:
            break
        x = pd.read_csv(path, usecols=cols)
        st = pd.concat([st, x[x.agent.isin(missing)]], ignore_index=True)
    know = st.sort_values("step").groupby(["agent", "episode"]).fleet_knowledge.last()
    m["know"] = know.groupby("agent").mean().reindex(m.index) / 1e3
    return m


def scored(m: pd.DataFrame, field: list[str]) -> pd.DataFrame:
    """Score/rank over ``field``; other rows get distance to the field best."""
    out = m.copy()
    s = score(m, SCORE_KPIS, field)
    out["score"] = np.nan
    out.loc[s.index, "score"] = s
    out["rank"] = out.loc[field, "score"].rank(method="min")
    extra = [a for a in m.index if a not in field]
    if extra:
        sub = m.loc[field]
        tot = pd.Series(0.0, index=extra)
        for k in SCORE_KPIS:
            d, _ = KPIS[k]
            best = sub[k].max() if d > 0 else sub[k].min()
            v = m.loc[extra, k]
            tot += (best - v) / best * 100.0 if d > 0 else (v - best) / best * 100.0
        out.loc[extra, "score"] = tot / len(SCORE_KPIS)
    return out


def pct(a: float, b: float) -> str:
    return "n/a" if not (np.isfinite(a) and np.isfinite(b)) else f"{(a - b) / b * 100:+.1f}%"


def fmt(v, nd=0):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:,.{nd}f}"


def retiree_block(scen_dir: Path) -> list[str]:
    p = scen_dir / "lifecycle_events.csv"
    if not p.is_file():
        return ["No lifecycle event log for this scenario."]
    lc = pd.read_csv(p)
    ret = lc[lc.kind == "retire_technician"]
    seqs = {a: tuple(zip(g.time.round(0), g.target)) for a, g in ret.groupby("agent")}
    distinct = set(seqs.values())
    lines = [f"Retirements logged for {len(seqs)} policies; "
             f"{len(distinct)} distinct retiree sequence(s)."]
    if len(distinct) == 1:
        lines.append("Retirees (identical for all policies): " + ", ".join(
            f"`{t}` @ {tm:,.0f}" for tm, t in next(iter(distinct))))
    else:
        for a, s in sorted(seqs.items()):
            lines.append(f"- {LABELS.get(a, a)}: " + ", ".join(f"`{t}` @ {tm:,.0f}" for tm, t in s))
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="reports/hvp_turnover_parts")
    ap.add_argument("--dest", default="reports/hvp_eval_turnover")
    ap.add_argument("--published", default="reports/hvp_eval_v6w/lifecycle")
    ap.add_argument("--published-parts", default="reports/hvp_v6w_parts")
    args = ap.parse_args()
    parts, dest = Path(args.parts), Path(args.dest)

    have = {s: merge(parts, dest, s) for s in [*VARIANTS, TARGETED]}
    # The published tree lacks step records for the MLP anchors; their
    # v6w parts carry them.
    pub = metrics(Path(args.published), sorted(
        Path(args.published_parts).glob(f"*/{TARGETED}/steps.csv.gz")))
    same = metrics(dest / TARGETED) if have[TARGETED] else None

    md = ["# Neutral-turnover variants of S5 (lifecycle)", "",
          "Reviewer point: S5 retires the technicians with the most knowledge "
          "(targeted key-person stress test). The two variants retire "
          "technicians at random; the second one also draws the retire/hire "
          "times. Everything else is identical to S5. Eval seed 20260722, "
          "one episode per policy.", "",
          "Columns: products; MTTR = episode-mean MTTR (t.u.); disr. = "
          "technician disruptions per 10³ products; know. = final fleet "
          "knowledge (×10³); score = mean % distance to the field's best value "
          "over these four KPIs (`summary_scores.score`, lower is better). The "
          "field is the set of policies evaluated in the variant, without the "
          "PO twin (its score is its distance to the same best values). "
          "\"Targeted (same code)\" = S5 rerun on dgy with this code and "
          "`PYTHONHASHSEED=0`; \"Targeted (published)\" = "
          "`reports/hvp_eval_v6w/lifecycle`. * = informed baseline.", ""]

    for v, title in VARIANTS.items():
        md += [f"## {title} (`{v}`)", ""]
        if not have[v]:
            md += ["No finished parts yet.", ""]
            continue
        m = metrics(dest / v)
        pols = [a for a in m.index if a in pub.index]
        field = [a for a in pols if a != TWIN]
        mv = scored(m.loc[pols], field)
        mp = scored(pub.loc[pols], field)
        ms = None
        if same is not None:
            spols = [a for a in pols if a in same.index]
            sfield = [a for a in spols if a != TWIN]
            ms = scored(same.loc[spols], sfield) if len(sfield) >= 2 else None
        missing = sorted(set(m.index) - set(pols))
        if missing:
            md += [f"Not in the published S5 table (left out): {missing}.", ""]
        note = (f"Field: {len(field)} policies (variant and published columns)")
        if ms is not None:
            note += f"; same-code targeted column: {len(sfield)} policies"
            if len(sfield) < len(field):
                note += (" (the missing policies have no same-code targeted run "
                         "yet, so this score uses a smaller field)")
        md += [note + ".", ""]
        md += ["| Policy | Products | MTTR | Disr. | Know. | Score | Rank "
               "| Targeted products (same code) | Targeted score (same code) "
               "| Targeted products (published) | Targeted score (published) "
               "| Targeted rank (published) |",
               "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for a in mv.sort_values("score").index:
            r, p = mv.loc[a], mp.loc[a]
            s_prod = s_score = "n/a"
            if ms is not None and a in ms.index:
                s_prod, s_score = fmt(ms.loc[a, "prod"]), fmt(ms.loc[a, "score"], 1)
            rank = "twin" if a == TWIN else fmt(r["rank"])
            prank = "twin" if a == TWIN else fmt(p["rank"])
            md.append(
                f"| {LABELS.get(a, a)} | {fmt(r['prod'])} | {fmt(r.mttr_mean, 1)} "
                f"| {fmt(r.disr, 0)} | {fmt(r.know, 1)} | {fmt(r.score, 1)} | {rank} "
                f"| {s_prod} | {s_score} | {fmt(p['prod'])} | {fmt(p.score, 1)} | {prank} |")
        md.append("")

        # Findings
        md += ["**Findings**", ""]
        lead = mv.loc[field].sort_values("score").index[0]
        lead_p = mv.loc[field, "prod"].idxmax()
        md.append(f"- Best score: {LABELS.get(lead, lead)} ({mv.loc[lead, 'score']:.1f}). "
                  f"Most products: {LABELS.get(lead_p, lead_p)} ({mv.loc[lead_p, 'prod']:,.0f}).")
        if "ft_quality" in mv.index:
            q = mv.loc["ft_quality"]
            md.append(f"- HTT-RL qua.: rank {q['rank']:.0f}/{len(field)} by score "
                      f"(published targeted S5 on the same field: rank "
                      f"{mp.loc['ft_quality', 'rank']:.0f}); products {pct(q['prod'], pub.loc['ft_quality', 'prod'])} "
                      f"vs published targeted S5"
                      + (f", {pct(q['prod'], same.loc['ft_quality', 'prod'])} vs same-code targeted S5"
                         if same is not None and "ft_quality" in same.index else "") + ".")
            if lead != "ft_quality":
                md.append(f"- HTT-RL qua. does not lead: gap to the leader "
                          f"{q.score - mv.loc[lead, 'score']:+.1f} score points, products "
                          f"{pct(q['prod'], mv.loc[lead, 'prod'])} vs {LABELS.get(lead, lead)}.")
            else:
                md.append("- HTT-RL qua. leads the field.")
        else:
            md.append("- HTT-RL qua. has no finished part yet.")
        if TWIN in mv.index:
            for ref in ("ft_quality", "hc_v6"):
                if ref not in mv.index:
                    continue
                def gap(df, k):
                    return pct(df.loc[TWIN, k], df.loc[ref, k])
                line = (f"- PO twin vs {LABELS[ref]}: products {gap(mv, 'prod')}, "
                        f"know. {gap(mv, 'know')}, disr. {gap(mv, 'disr')}, "
                        f"MTTR {gap(mv, 'mttr_mean')} (published targeted S5: products "
                        f"{gap(pub, 'prod')}, know. {gap(pub, 'know')}, disr. {gap(pub, 'disr')}")
                if same is not None and {TWIN, ref} <= set(same.index):
                    line += (f"; same-code targeted S5: products {gap(same, 'prod')}, "
                             f"know. {gap(same, 'know')}, disr. {gap(same, 'disr')}")
                md.append(line + ").")
        else:
            md.append("- PO twin has no finished part yet.")
        md += [""] + retiree_block(dest / v) + [""]

    if same is not None:
        md += ["## Targeted S5 rerun (same code) vs published", "",
               "| Policy | Products (same code) | Products (published) | Δ |",
               "|---|--:|--:|--:|"]
        for a in same.index:
            if a in pub.index:
                md.append(f"| {LABELS.get(a, a)} | {fmt(same.loc[a, 'prod'])} "
                          f"| {fmt(pub.loc[a, 'prod'])} | {pct(same.loc[a, 'prod'], pub.loc[a, 'prod'])} |")
        md += [""] + retiree_block(dest / TARGETED) + [""]

    out = dest / "summary.md"
    dest.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

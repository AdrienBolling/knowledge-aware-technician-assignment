"""Stability of the summary-score ranking across factory layouts (S4, S5).

The career-length scenarios evaluate one factory layout, sampled from the
eval seed (published generation: 20260722).  reports/hvp_s45 holds one episode
per policy on five more layouts (eval seeds 20260909-20260913, one per run).
For each factory, this script scores the main field of tab:results_dist with
the paper's summary score (per-KPI best values taken inside that factory),
ranks the policies of tab:results_dist, and summarizes the ranks and scores
across factories.

HTT-RL is represented by its best checkpoint.  The final checkpoint was not
evaluated on the new layouts, so the field of every factory excludes it.

Outputs (reports/rank_stability/):
  scores.csv          scenario, factory, agent, score, rank
  factories.csv       scenario, factory, seed, machines, scarcest type and count
  rank_stability.tex  tabular body of the appendix table
  summary.txt         concordance statistics and per-policy statistics

Usage: python scripts/make_rank_stability.py   (from the repository root)
"""
import contextlib
import importlib
import io
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multirun_data as md  # noqa: E402
import summary_scores as ss  # noqa: E402

OUT = Path('reports/rank_stability')
SCEN = [('very_long', 'S4 -- Very-long'), ('lifecycle', 'S5 -- Lifecycle')]
PUBLISHED_SEED = 20260722
RUN_SEED_BASE = 20260908
DISPLAYED = [k for k, _ in ss.REP_COLS]
LABEL = dict(ss.REP_COLS)


def main_field(agents):
    """Main field of tab:results_dist (informed swap), restricted to `agents`."""
    field = {a for a in agents
             if a not in ss.EXCLUDE and a not in ss.ORACLES and a not in ss.INFORMED_SWAP}
    field |= {v for v in ss.INFORMED_SWAP.values() if v in agents}
    return sorted(field)


def factory_kpis(scenario):
    """{factory: KPI table}; factory 0 = published layout, 1-5 = hvp_s45 runs."""
    pub = md.episodes(scenario, source='v6w').assign(factory=0)
    new = md.episodes(scenario)
    new = new.assign(factory=new.episode)
    out = {}
    for f, g in pd.concat([pub, new], ignore_index=True).groupby('factory'):
        m = g.groupby('agent')[['prod', 'mttr_mean', 'ill', 'know']].mean()
        m['disr'] = m.ill / m['prod'] * 1000.0
        out[int(f)] = m
    return out


def _machines_by_type(obj, depth=0):
    if hasattr(obj, 'machines_by_type'):
        return obj.machines_by_type
    if depth < 2 and isinstance(obj, (tuple, list)):
        for x in obj:
            r = _machines_by_type(x, depth + 1)
            if r is not None:
                return r
    if depth < 2 and hasattr(obj, '__dict__'):
        for v in vars(obj).values():
            if hasattr(v, 'machines_by_type'):
                return v.machines_by_type
    return None


def factory_layouts(scenario, seeds):
    """Initial machine count and scarcest machine type of each factory."""
    ev = importlib.import_module('eval_human_vs_performance')
    rows = []
    for f, seed in seeds.items():
        ev.EVAL_SEED = seed
        with contextlib.redirect_stdout(io.StringIO()):
            _, factory, *_ = ev.build_scenario(scenario)
            built = factory()
        counts = {t: len(v) for t, v in _machines_by_type(built).items()}
        low = min(counts.values())
        scarce = sorted(t for t, n in counts.items() if n == low)
        rows.append({'scenario': scenario, 'factory': f, 'seed': seed,
                     'machines': sum(counts.values()), 'scarcest': '/'.join(scarce),
                     'scarcest_n': low})
    return pd.DataFrame(rows)


def kendall_w(ranks):
    """Kendall's coefficient of concordance with tie correction.

    ranks: DataFrame, rows = policies, columns = factories (average ranks).
    Returns W, chi-square statistic, p-value.
    """
    n, m = ranks.shape
    total = ranks.sum(axis=1)
    s = float(((total - total.mean()) ** 2).sum())
    ties = sum(float(((c ** 3) - c).sum())
               for col in ranks.columns for c in [ranks[col].value_counts().to_numpy()])
    w = 12.0 * s / (m ** 2 * (n ** 3 - n) - m * ties)
    chi2 = m * (n - 1) * w
    return w, chi2, float(stats.chi2.sf(chi2, n - 1))


def fmt_score(mean, std, rank_of_mean):
    cell = f'{mean:.1f}_{{\\pm {std:.1f}}}'
    if rank_of_mean == 1:
        return f'$\\mathbf{{{mean:.1f}}}_{{\\pm {std:.1f}}}$'
    if rank_of_mean == 2:
        return f'$\\underline{{{mean:.1f}}}_{{\\pm {std:.1f}}}$'
    return f'${cell}$'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    score_rows, layouts, summary, blocks = [], [], [], {}
    for scen, label in SCEN:
        kpis = ss.VARIANTS[ss.PAPER_VARIANT] + (['know'] if scen in ss.KNOW_SCEN else [])
        tables = factory_kpis(scen)
        common = set.intersection(*[set(t.index) for t in tables.values()])
        field = main_field(common)
        missing = [a for a in DISPLAYED if a not in field]
        if missing:
            raise SystemExit(f'{scen}: displayed policies missing from the field: {missing}')
        scores, ranks_min, ranks_avg = {}, {}, {}
        for f, m in tables.items():
            sc = ss.score(m, kpis, field)
            shown = sc[DISPLAYED]
            scores[f] = shown
            ranks_min[f] = shown.round(6).rank(method='min')
            ranks_avg[f] = shown.round(6).rank(method='average')
            for a in DISPLAYED:
                score_rows.append({'scenario': scen, 'factory': f, 'agent': a,
                                   'score': shown[a], 'rank': int(ranks_min[f][a])})
        S = pd.DataFrame(scores)
        Rmin = pd.DataFrame(ranks_min)
        Ravg = pd.DataFrame(ranks_avg)
        seeds = {0: PUBLISHED_SEED} | {f: RUN_SEED_BASE + f for f in S.columns if f > 0}
        lay = factory_layouts(scen, seeds)
        layouts.append(lay)
        w, chi2, p = kendall_w(Ravg)
        taus = [stats.kendalltau(Ravg[i], Ravg[j]).statistic
                for i, j in itertools.combinations(Ravg.columns, 2)]
        taus_pub = [stats.kendalltau(Ravg[0], Ravg[j]).statistic for j in Ravg.columns if j > 0]
        stat = pd.DataFrame({
            'mean_score': S.mean(axis=1), 'std_score': S.std(axis=1, ddof=1),
            'mean_rank': Rmin.mean(axis=1), 'std_rank': Rmin.std(axis=1, ddof=1),
            'best_rank': Rmin.min(axis=1), 'worst_rank': Rmin.max(axis=1),
            'first': (Rmin == 1).sum(axis=1), 'rank_published': Rmin[0]})
        summary.append(f'== {scen}: factories {list(S.columns)}, field {len(field)} policies')
        summary.append(f'   Kendall W = {w:.3f} (chi2 = {chi2:.1f}, df = {len(DISPLAYED) - 1}, p = {p:.2g})')
        summary.append(f'   pairwise Kendall tau-b: mean {np.mean(taus):.3f}, min {np.min(taus):.3f}, max {np.max(taus):.3f}')
        summary.append(f'   tau-b vs published factory: ' + ', '.join(f'{t:.2f}' for t in taus_pub))
        summary.append(lay.to_string(index=False))
        summary.append(stat.round(2).sort_values('mean_score').to_string())
        summary.append('   rank-1 policy per factory: ' + ', '.join(
            f'F{f}={",".join(Rmin.index[Rmin[f] == 1])}' for f in Rmin.columns))
        blocks[scen] = (S, Rmin, stat, lay, w, p)

    pd.DataFrame(score_rows).to_csv(OUT / 'scores.csv', index=False)
    pd.concat(layouts, ignore_index=True).to_csv(OUT / 'factories.csv', index=False)
    (OUT / 'summary.txt').write_text('\n'.join(summary) + '\n')

    lines = []
    for a in DISPLAYED:
        cells = [LABEL[a]]
        for scen, _ in SCEN:
            S, Rmin, stat, _, _, _ = blocks[scen]
            order = stat.mean_score.round(1).rank(method='dense')
            for f in Rmin.columns:
                r = int(Rmin.loc[a, f])
                cells.append(f'\\textbf{{{r}}}' if r == 1 else str(r))
            cells.append(fmt_score(stat.loc[a, 'mean_score'], stat.loc[a, 'std_score'],
                                   int(order[a])))
        lines.append(' & '.join(cells) + r' \\')
    lines.append(r'\midrule')
    for key, name in (('machines', 'Machines'), ('scarcest', 'Scarcest type')):
        cells = [name]
        for scen, _ in SCEN:
            lay = blocks[scen][3]
            for _, row in lay.iterrows():
                if key == 'machines':
                    cells.append(str(row.machines))
                else:
                    short = '/'.join(t[:4] for t in row.scarcest.split('/'))
                    cells.append(f'{short}\\,{row.scarcest_n}')
            cells.append('')
        lines.append(' & '.join(cells) + r' \\')
    (OUT / 'rank_stability.tex').write_text('\n'.join(lines) + '\n')
    print('\n'.join(summary))
    print(f'\nwritten: {OUT}/rank_stability.tex, scores.csv, factories.csv, summary.txt')


if __name__ == '__main__':
    main()

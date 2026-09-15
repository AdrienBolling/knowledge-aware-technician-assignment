"""How the multi-episode data change the published results.

Compares the published generation (S4/S5: one episode) with the five-run
generation, on the main field of tab:results_dist (informed swap on; HTT-RL is
represented by its best checkpoint only, so hc_v6_last is left out of the field).

Prints, per scenario:
  1. KPI mean +- std for the displayed policies, next to the published value;
  2. summary scores: published definition (score of the mean KPIs) and the
     mean +- std of per-episode scores (episode e is the same world for every
     agent, so per-episode scores are paired);
  3. paired comparisons for the claims of the Results section.

Usage: python scripts/multirun_report.py [--md]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multirun_data as md  # noqa: E402

SCORE_KPIS = [('prod', +1), ('mttr_mean', -1), ('disr', -1), ('know', +1)]
FIELD = ['hc_v6', 'ft_quality', 'ft_quality_last', 'ft_fatigue', 'ft_fatigue_last',
         'ft_protect', 'ft_protect_last', 'ft_gini', 'ft_gini_last',
         'topsis', 'shortest_processing', 'optimal_assignment',
         'shortest_queue', 'least_fatigued', 'round_robin', 'least_busy', 'train_weakest', 'random',
         'a2c_mlp', 'a2c_mlp_last', 'grpo_mlp', 'grpo_mlp_last', 'dql_mlp', 'dql_mlp_last']
SHOWN = [('hc_v6', 'HTT-RL ref.'), ('ft_quality', 'HTT-RL qua.'), ('topsis', 'Topsis*'),
         ('shortest_processing', 'Spt*'), ('shortest_queue', 'ShortQ'), ('least_fatigued', 'LeastFat'),
         ('round_robin', 'RoundR'), ('least_busy', 'LeastBusy'), ('train_weakest', 'TrainW'),
         ('random', 'Random'), ('optimal_assignment', 'Hungarian*'), ('a2c_mlp', 'A2C'),
         ('grpo_mlp', 'GRPO'), ('dql_mlp', 'DDQN')]
PAIRS = [('ft_quality', 'topsis'), ('ft_quality', 'shortest_processing'),
         ('ft_quality', 'optimal_assignment'), ('ft_quality', 'hc_v6'), ('hc_v6', 'topsis'),
         ('hc_v6', 'po_v6')]
LABEL = dict(SHOWN) | {'po_v6': 'PO-HTT-RL'}


def score_table(kpi):
    """Summary score per agent from a KPI table (index = agent)."""
    sub = kpi.loc[[a for a in FIELD if a in kpi.index]]
    out = pd.Series(0.0, index=sub.index)
    for k, d in SCORE_KPIS:
        best = sub[k].max() if d > 0 else sub[k].min()
        out += (best - sub[k]) / best * 100.0 if d > 0 else (sub[k] - best) / best * 100.0
    return out / len(SCORE_KPIS)


def mean_kpis(ep):
    m = ep.groupby('agent')[['prod', 'mttr_mean', 'ill', 'know']].mean()
    m['disr'] = m.ill / m['prod'] * 1000.0
    return m


def per_episode_scores(ep):
    rows = {}
    for e, g in ep.groupby('episode'):
        rows[e] = score_table(g.set_index('agent'))
    return pd.DataFrame(rows)  # index agent, columns episode


def paired(ep, a, b, key):
    x = ep[ep.agent == a].set_index('episode')[key]
    y = ep[ep.agent == b].set_index('episode')[key]
    idx = x.index.intersection(y.index)
    d = x[idx] - y[idx]
    p = stats.ttest_rel(x[idx], y[idx]).pvalue if len(idx) > 1 and d.std() > 0 else np.nan
    return len(idx), float(x[idx].mean()), float(y[idx].mean()), float(d.mean()), float(d.std(ddof=1)), p, int((d > 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    overall_pub, overall_new = [], []
    for scen in md.SCENARIOS:
        ep = md.episodes(scen)
        pub = md.episodes(scen, source='v6w')
        print(f'\n################ {scen}  (n episodes: new {ep.groupby("agent").size().median():.0f}, published {pub.groupby("agent").size().median():.0f})')
        st = md.agent_stats(ep)
        pm = mean_kpis(pub)
        print(f'{"policy":12s} ' + '  '.join(f'{k:>26s}' for k, _ in SCORE_KPIS))
        for a, lab in SHOWN:
            cells = []
            for k, _ in SCORE_KPIS:
                m, s = st.loc[a, (k, 'mean')], st.loc[a, (k, 'std')]
                ref = pm.loc[a, k]
                z = (ref - m) / s if s > 0 else np.nan
                cells.append(f'{m:9.1f}±{s:7.1f} (pub {ref:8.1f}, z{z:+.1f})')
            print(f'{lab:12s} ' + '  '.join(f'{c:>26s}' for c in cells))
        s_pub = score_table(pm)
        s_new = score_table(mean_kpis(ep))
        pes = per_episode_scores(ep)
        overall_pub.append(s_pub); overall_new.append(s_new)
        print(f'\n{"policy":12s} {"score pub":>9s} {"rank":>4s} | {"score new":>9s} {"±std(ep)":>8s} {"rank":>4s}')
        rp = s_pub.rank(method='min'); rn = s_new.rank(method='min')
        for a, lab in SHOWN:
            print(f'{lab:12s} {s_pub[a]:9.1f} {rp[a]:4.0f} | {s_new[a]:9.1f} {pes.loc[a].std(ddof=1):8.1f} {rn[a]:4.0f}')
        print('\npaired comparisons (a - b):')
        for a, b in PAIRS:
            for key in ('prod', 'disr', 'know'):
                n, xa, xb, dm, ds, p, wins = paired(ep, a, b, key)
                if n:
                    print(f'  {LABEL[a]:11s} vs {LABEL[b]:10s} {key:5s} n={n} {xa:9.1f} vs {xb:9.1f}  diff {dm:+8.1f} ({100*dm/xb:+5.1f}%) sd {ds:7.1f}  p={p:.3f}  a>b in {wins}/{n}')
            sa = pes.loc[a] if a in pes.index else None
            sb = pes.loc[b] if b in pes.index else None
            if sa is not None and sb is not None:
                d = sa - sb
                p = stats.ttest_rel(sa, sb).pvalue if len(d) > 1 else np.nan
                print(f'  {LABEL[a]:11s} vs {LABEL[b]:10s} score n={len(d)} {sa.mean():9.2f} vs {sb.mean():9.2f}  diff {d.mean():+.2f} sd {d.std(ddof=1):.2f}  p={p:.3f}  a<b in {(d < 0).sum()}/{len(d)}')
    op = pd.concat(overall_pub, axis=1).mean(axis=1)
    on = pd.concat(overall_new, axis=1).mean(axis=1)
    print('\n################ overall (mean over scenarios)')
    rp = op.rank(method='min'); rn = on.rank(method='min')
    for a, lab in SHOWN:
        print(f'{lab:12s} pub {op[a]:5.1f} (rank {rp[a]:2.0f})   new {on[a]:5.1f} (rank {rn[a]:2.0f})')


if __name__ == '__main__':
    main()

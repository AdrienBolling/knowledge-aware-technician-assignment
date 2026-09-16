"""Sensitivity of the summary score to KPI weights, Pareto analysis, and
dependence of the ranks on the roster.

The summary score of tab:results_dist is, per scenario, the mean of four
per-KPI percentage gaps to the best value of the benchmark field
(summary_scores.main_field, 26 agents):

    gap_k(a) = 100 * (best_k - v_k(a)) / best_k   benefit KPI (products, final knowledge)
    gap_k(a) = 100 * (v_k(a) - best_k) / best_k   cost KPI (episode-mean MTTR, disr./10^3 products)
    score(a) = sum_k w_k * gap_k(a),  w = (1/4, 1/4, 1/4, 1/4) in the paper

1. Weight sensitivity.  N weight vectors are drawn uniformly on the 4-KPI
   simplex (Dirichlet(1, 1, 1, 1), fixed seed).  Per draw, the 15 policies
   of tab:results_dist are ranked on the weighted score of each scenario and
   on the overall score (mean of the five scenario scores, same weights).
   Ranks use the "min" convention: tied policies share the best rank, so
   P(rank 1) can add to more than 1.  Rank quantiles use the inverted CDF,
   so they are ranks that occur.  The four single-KPI corners (all weight on
   one KPI) are reported separately.
2. Pareto analysis.  Per scenario, a policy is dominated when another
   displayed policy is at least as good on all four KPIs and strictly better
   on one.  Exactly equal KPI vectors are ties, not dominance.
3. Roster dependence, equal weights.  Roster changes: each agent of the field
   removed (from the field and, if displayed, from the displayed policies),
   GreedyReward added (to both), and the best values taken over the 15
   displayed policies only.  Each change has two effects, reported apart:
   (a) direct effect: the ranks and leader of the original displayed
       policies with the original bests, against the ranks and leader of the
       changed displayed policies with their own bests (columns direct_*);
   (b) reference-value effect: the order of the policies present in both
       rosters, with the original bests against the new bests (columns
       ref_*).  A different best value b' gives gap' = (b/b') * gap +
       constant, so for the ranks it acts as a change of that KPI's weight by
       the factor b/b'.

Outputs (``--out``, default ``kpi_weights/``):
    gaps.csv            per-KPI gaps and the equal-weight score, every field agent
    rank_stats.csv      P(rank 1), median and 5-95% rank per scenario and overall
    corners.csv         score and rank at the four single-KPI corners
    pairwise.csv        P(row policy scores lower than column policy), per scenario and overall
    overall_diff.csv    per-KPI mean-gap difference and overall-score difference quantiles, leading pairs
    rank1_weights.csv   mean and minimum weight per KPI over the draws in which a policy ranks first
    pareto.csv          dominators, Pareto membership and ties per scenario
    roster.csv          scores and ranks with field bests vs displayed-only bests
    roster_changes.csv  direct and reference-value effects of each roster change
    appendix_table.tex  body rows of tab:kpi_weights

Usage: uv run --no-sync python scripts/kpi_weight_sensitivity.py [--n 20000] [--seed 20260916] [--out kpi_weights]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('ss', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'summary_scores.py'))
ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)

N_DRAWS = 20_000
SEED = 20260916
# The four scored KPIs of the paper variant, in the order of the gaps.
KPIS = ss.VARIANTS[ss.PAPER_VARIANT] + ['know']
KPI_LABEL = {'prod': 'products', 'mttr_mean': 'MTTR', 'disr': 'disruptions', 'know': 'knowledge'}
SHOWN = [k for k, _ in ss.REP_COLS]
HEAD = dict(ss.REP_COLS)
SCEN_ID = {s: f'S{i + 1}' for i, (s, _) in enumerate(ss.SCEN)}
OVERALL = 'Overall'
# Rank intervals reported in the appendix table.
TRACKED = ['hc_v6', 'ft_quality', 'topsis', 'shortest_processing', 'optimal_assignment']
# Overall-score pairs whose difference is reported with quantiles.
PAIRS = [('hc_v6', 'topsis'), ('hc_v6', 'ft_quality'), ('hc_v6', 'shortest_processing'),
         ('topsis', 'shortest_processing'), ('ft_quality', 'topsis')]
TOL = 1e-9


def gap_matrix(m, field, kpis=KPIS):
    """Per-KPI percentage gap to the best value over ``field`` (rows: field agents)."""
    sub = m.loc[field]
    out = {}
    for k in kpis:
        d, _ = ss.KPIS[k]
        best = sub[k].max() if d > 0 else sub[k].min()
        out[k] = (best - sub[k]) / best * 100.0 if d > 0 else (sub[k] - best) / best * 100.0
    return pd.DataFrame(out)[list(kpis)]


def draw_weights(n=N_DRAWS, seed=SEED, k=len(KPIS)):
    """``n`` weight vectors uniform on the k-simplex (Dirichlet(1, ..., 1))."""
    return np.random.default_rng(seed).dirichlet(np.ones(k), size=n)


def weighted_scores(G, W):
    """(draws, policies) weighted scores; G is (policies, KPIs), W is (draws, KPIs)."""
    return np.atleast_2d(W) @ np.asarray(G, dtype=float).T


def min_ranks(S, tol=TOL):
    """Competition ranks per row (1 + number of strictly lower scores); ties share a rank."""
    S = np.atleast_2d(S)
    return 1 + (S[:, None, :] < S[:, :, None] - tol).sum(axis=2)


def rank_summary(R, names):
    q = np.quantile(R, [0.05, 0.5, 0.95], axis=0, method='inverted_cdf')
    return pd.DataFrame({'agent': names, 'p_rank1': (R == 1).mean(axis=0),
                         'rank_q05': q[0].astype(int), 'rank_median': q[1].astype(int),
                         'rank_q95': q[2].astype(int)})


def dominators(V, directions):
    """Pareto analysis of the rows of V (policies x KPIs, raw values).

    ``directions`` is +1 for a benefit KPI and -1 for a cost KPI.  Returns the
    number of rows that dominate each row, the indices of those rows, and,
    per row, the rows with an exactly equal KPI vector.
    """
    X = np.asarray(V, dtype=float) * np.asarray(directions, dtype=float)
    geq = (X[:, None, :] >= X[None, :, :]).all(axis=2)   # [i, j]: i >= j everywhere
    gt = (X[:, None, :] > X[None, :, :]).any(axis=2)     # [i, j]: i > j somewhere
    dom = geq & gt                                        # [i, j]: i dominates j
    equal = (X[:, None, :] == X[None, :, :]).all(axis=2)
    np.fill_diagonal(equal, False)
    return dom.sum(axis=0), [list(np.flatnonzero(col)) for col in dom.T], [list(np.flatnonzero(row)) for row in equal]


def pareto_table(metrics, shown, field):
    directions = [ss.KPIS[k][0] for k in KPIS]
    rows = []
    for s, _ in ss.SCEN:
        m = metrics[s]
        n_dom, by, ties = dominators(m.loc[shown, KPIS].to_numpy(), directions)
        n_dom_field, _, _ = dominators(m.loc[field, KPIS].to_numpy(), directions)
        in_field = dict(zip(field, n_dom_field))
        for i, a in enumerate(shown):
            rows.append({'scenario': SCEN_ID[s], 'agent': a, 'n_dominators': int(n_dom[i]),
                         'pareto': bool(n_dom[i] == 0), 'dominated_by': ';'.join(shown[j] for j in by[i]),
                         'n_dominators_field': int(in_field[a]),
                         'tied_with': ';'.join(shown[j] for j in ties[i])})
    return pd.DataFrame(rows)


def bests(m, over, kpis=KPIS):
    """Per-KPI best value over the agents ``over``."""
    return pd.Series({k: (m.loc[over, k].max() if ss.KPIS[k][0] > 0 else m.loc[over, k].min()) for k in kpis})


def ranks_table(metrics, bests_over, shown):
    """Equal-weight scores and min-ranks of ``shown`` with per-KPI bests over ``bests_over``."""
    cols = {SCEN_ID[s]: gap_matrix(metrics[s], bests_over).loc[shown].mean(axis=1) for s, _ in ss.SCEN}
    t = pd.DataFrame(cols)
    t[OVERALL] = t.mean(axis=1)
    r = pd.DataFrame({c: min_ranks(t[c].to_numpy()[None, :])[0] for c in t.columns}, index=t.index)
    return t, r


def _leaders(r, c):
    return ';'.join(r.index[r[c] == 1])


def roster_changes(metrics, field, shown, extra_agents=(ss.GREEDY,)):
    """Direct and reference-value effects of roster changes on the equal-weight ranks.

    Changes: 'bests:displayed' (bests over the displayed policies only, the
    policies do not change), 'drop:<agent>' for each field agent (removed from
    the field and from the displayed policies), and 'add:<agent>' for each of
    ``extra_agents`` (added to both).  Per change and scenario:

    direct effect (original displayed policies with the original bests against
    the changed displayed policies with the new bests)
        direct_leader_original, direct_leader_changed, direct_leader_change
        agent_rank                 rank of the removed agent in the original roster,
                                   or of the added agent in the changed roster
        direct_n_rank_changes      common policies whose rank number changes
        direct_max_rank_shift
    reference-value effect (common policies only, original against new bests)
        ref_leader_before, ref_leader_after, ref_n_order_changes,
        ref_max_order_shift, ref_max_score_change,
        ref_max_rescale            max_k |b_k / b'_k - 1| (over the scenarios for Overall)
    """
    t_f, r_f = ranks_table(metrics, field, shown)
    t_s, r_s = ranks_table(metrics, shown, shown)
    roster = pd.DataFrame([{'scenario': c, 'agent': a, 'score_field': t_f.loc[a, c], 'rank_field': int(r_f.loc[a, c]),
                            'score_shown': t_s.loc[a, c], 'rank_shown': int(r_s.loc[a, c])}
                           for c in t_f.columns for a in shown])
    # (name, changed field, changed displayed policies, removed or added agent)
    changes = [('bests:displayed', list(shown), list(shown), None)]
    changes += [(f'drop:{a}', [x for x in field if x != a], [x for x in shown if x != a], a) for a in field]
    changes += [(f'add:{a}', field + [a], shown + [a], a) for a in extra_agents
                if a not in field and all(a in metrics[s].index for s, _ in ss.SCEN)]
    r_orig = r_f
    rows = []
    for name, field2, shown2, agent in changes:
        common = [a for a in shown if a in shown2]
        _, r_new = ranks_table(metrics, field2, shown2)
        t_ref0, r_ref0 = ranks_table(metrics, field, common)
        t_ref1, r_ref1 = ranks_table(metrics, field2, common)
        resc = {SCEN_ID[s]: float((bests(metrics[s], field) / bests(metrics[s], field2) - 1).abs().max()) for s, _ in ss.SCEN}
        resc[OVERALL] = max(resc.values())
        for c in r_orig.columns:
            direct = (r_new.loc[common, c] - r_orig.loc[common, c]).abs()
            order = (r_ref1[c] - r_ref0[c]).abs()
            if agent in r_orig.index:
                agent_rank = int(r_orig.loc[agent, c])
            elif agent in r_new.index:
                agent_rank = int(r_new.loc[agent, c])
            else:
                agent_rank = pd.NA
            rows.append({'change': name, 'scenario': c,
                         'direct_leader_original': _leaders(r_orig, c), 'direct_leader_changed': _leaders(r_new, c),
                         'direct_leader_change': _leaders(r_orig, c) != _leaders(r_new, c), 'agent_rank': agent_rank,
                         'direct_n_rank_changes': int((direct > 0).sum()), 'direct_max_rank_shift': int(direct.max()),
                         'ref_leader_before': _leaders(r_ref0, c), 'ref_leader_after': _leaders(r_ref1, c),
                         'ref_n_order_changes': int((order > 0).sum()), 'ref_max_order_shift': int(order.max()),
                         'ref_max_score_change': float((t_ref1[c] - t_ref0[c]).abs().max()),
                         'ref_max_rescale': resc[c]})
    out = pd.DataFrame(rows)
    out['agent_rank'] = out['agent_rank'].astype('Int64')
    return roster, out


def sensitivity(metrics, field, shown, W):
    """Rank statistics, single-KPI corners, pairwise win probabilities, overall-score
    differences of PAIRS, and the weights of the draws in which each policy ranks first."""
    G = {SCEN_ID[s]: gap_matrix(metrics[s], field).loc[shown] for s, _ in ss.SCEN}
    G[OVERALL] = sum(G[SCEN_ID[s]] for s, _ in ss.SCEN) / len(ss.SCEN)
    stats, corners, pairwise, diffs, rank1 = [], [], [], [], []
    for c, g in G.items():
        S = weighted_scores(g.to_numpy(), W)
        R = min_ranks(S)
        eq = weighted_scores(g.to_numpy(), np.full(len(KPIS), 1 / len(KPIS)))[0]
        st = rank_summary(R, shown)
        st.insert(0, 'scenario', c)
        st['score_equal'] = eq
        st['rank_equal'] = min_ranks(eq[None, :])[0]
        stats.append(st)
        for j, k in enumerate(KPIS):
            e = np.zeros(len(KPIS)); e[j] = 1.0
            sc = weighted_scores(g.to_numpy(), e)[0]
            corners.append(pd.DataFrame({'scenario': c, 'corner': k, 'agent': shown, 'score': sc,
                                         'rank': min_ranks(sc[None, :])[0]}))
        for j, a in enumerate(shown):
            ww = W[R[:, j] == 1]
            if len(ww):
                row = {'scenario': c, 'agent': a, 'n_rank1': len(ww)}
                row.update({f'w_mean_{k}': ww[:, i].mean() for i, k in enumerate(KPIS)})
                row.update({f'w_min_{k}': ww[:, i].min() for i, k in enumerate(KPIS)})
                row['p_prod_know_majority'] = float((ww[:, KPIS.index('prod')] + ww[:, KPIS.index('know')] > 0.5).mean())
                rank1.append(row)
        P = (S[:, :, None] < S[:, None, :] - TOL).mean(axis=0)
        pw = pd.DataFrame(P, index=shown, columns=shown)
        pw.insert(0, 'scenario', c)
        pairwise.append(pw.rename_axis('agent').reset_index())
        if c == OVERALL:
            idx = {a: i for i, a in enumerate(shown)}
            for a, b in PAIRS:
                d = S[:, idx[a]] - S[:, idx[b]]
                q = np.quantile(d, [0.05, 0.5, 0.95])
                row = {'a': a, 'b': b}
                row.update({f'gap_diff_{k}': g.loc[a, k] - g.loc[b, k] for k in KPIS})
                row.update({'diff_equal': eq[idx[a]] - eq[idx[b]], 'p_a_lower': float((d < -TOL).mean()),
                            'diff_q05': q[0], 'diff_median': q[1], 'diff_q95': q[2]})
                diffs.append(row)
    return (pd.concat(stats, ignore_index=True), pd.concat(corners, ignore_index=True),
            pd.concat(pairwise, ignore_index=True), pd.DataFrame(diffs), pd.DataFrame(rank1))


def _prob(p):
    """Probability with two decimals, rounded half up; '<0.01' for a small nonzero value."""
    if 0 < p < 0.005:
        return '$<$0.01'
    return str(Decimal(repr(float(p))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def emit_tex(stats, pareto, n_draws):
    """Body rows of tab:kpi_weights.

    Per scenario: the Pareto-optimal displayed policies (Table 7 order), the
    (up to) three policies with the highest P(rank 1) (ties in Table 7
    order), and median [5%--95%] rank of the TRACKED policies.
    """
    lines = [f'% ---- tab:kpi_weights body (N = {n_draws} Dirichlet(1,1,1,1) weight draws) ----']
    for c in [SCEN_ID[s] for s, _ in ss.SCEN] + [OVERALL]:
        st = stats[stats.scenario == c].set_index('agent').loc[SHOWN]
        if c == OVERALL:
            lines.append(r'\midrule')
            front = '--'
        else:
            p = pareto[pareto.scenario == c].set_index('agent')
            front = ', '.join(HEAD[a] for a in SHOWN if p.loc[a, 'pareto'])
        lead = st[st.p_rank1 > 0].sort_values('p_rank1', ascending=False, kind='stable')
        lead = ', '.join(f'{HEAD[a]} {_prob(r.p_rank1)}' for a, r in lead.head(3).iterrows())
        cells = ' & '.join(f'{int(st.loc[a, "rank_median"])} [{int(st.loc[a, "rank_q05"])}--{int(st.loc[a, "rank_q95"])}]'
                           for a in TRACKED)
        lines.append(f'{c} & {front} & {lead} & {cells} \\\\')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--n', type=int, default=N_DRAWS, help='number of weight draws')
    ap.add_argument('--seed', type=int, default=SEED, help='seed of the weight draws')
    ap.add_argument('--out', default='kpi_weights', help='output directory')
    ap.add_argument('--extra', default=ss.EXTRA, help='step records of agents without local steps (summary_scores --extra)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pd.set_option('display.width', 200)

    metrics = ss.load_metrics(args.extra)
    field = ss.main_field(metrics)
    shown = [a for a in SHOWN if a in field]
    assert shown == SHOWN, f'displayed policies missing from the field: {sorted(set(SHOWN) - set(shown))}'
    print(f'# field: {len(field)} agents, displayed: {len(shown)}, KPIs: {KPIS}, N = {args.n}, seed = {args.seed}')

    gaps = []
    for s, _ in ss.SCEN:
        g = gap_matrix(metrics[s], field)
        g['score_equal'] = g[KPIS].mean(axis=1)
        chk = ss.score(metrics[s], KPIS, field)
        assert np.allclose(g['score_equal'], chk.loc[g.index]), s
        g.insert(0, 'scenario', SCEN_ID[s])
        gaps.append(g.rename_axis('agent').reset_index())
    pd.concat(gaps, ignore_index=True).to_csv(f'{args.out}/gaps.csv', index=False, float_format='%.6g')

    W = draw_weights(args.n, args.seed)
    stats, corners, pairwise, diffs, rank1 = sensitivity(metrics, field, shown, W)
    stats.to_csv(f'{args.out}/rank_stats.csv', index=False, float_format='%.6g')
    corners.to_csv(f'{args.out}/corners.csv', index=False, float_format='%.6g')
    pairwise.to_csv(f'{args.out}/pairwise.csv', index=False, float_format='%.6g')
    diffs.to_csv(f'{args.out}/overall_diff.csv', index=False, float_format='%.6g')
    rank1.to_csv(f'{args.out}/rank1_weights.csv', index=False, float_format='%.6g')
    pareto = pareto_table(metrics, shown, field)
    pareto.to_csv(f'{args.out}/pareto.csv', index=False)
    roster, changes = roster_changes(metrics, field, shown)
    roster.to_csv(f'{args.out}/roster.csv', index=False, float_format='%.6g')
    changes.to_csv(f'{args.out}/roster_changes.csv', index=False, float_format='%.6g')
    with open(f'{args.out}/appendix_table.tex', 'w', encoding='utf-8') as f:
        f.write(emit_tex(stats, pareto, args.n))

    lab = ss.PLAIN
    print('\n## weight sensitivity (displayed policies, ranks among the 15)')
    for c in stats.scenario.unique():
        st = stats[stats.scenario == c].sort_values(['p_rank1', 'rank_median'], ascending=[False, True])
        print(f'\n{c}:')
        view = st.assign(agent=st.agent.map(lab))[['agent', 'p_rank1', 'rank_median', 'rank_q05', 'rank_q95', 'score_equal', 'rank_equal']]
        print(view.round(3).to_string(index=False))
    print('\n## single-KPI corners: rank-1 policies')
    for (c, k), g in corners.groupby(['scenario', 'corner'], sort=False):
        print(f'  {c:8s} {KPI_LABEL[k]:12s} ' + ', '.join(lab[a] for a in g.agent[g['rank'] == 1]))
    print('\n## weights of the rank-1 draws (mean / min per KPI; share with products + knowledge > 1/2)')
    for _, r in rank1.iterrows():
        print(f'  {r.scenario:8s} {lab[r.agent]:14s} n={r.n_rank1:6d} mean ' + ' '.join(f'{r[f"w_mean_{k}"]:.2f}' for k in KPIS)
              + ' | min ' + ' '.join(f'{r[f"w_min_{k}"]:.3f}' for k in KPIS) + f' | {r.p_prod_know_majority:.3f}')
    print('\n## overall-score differences (a - b) under the weight draws')
    print(diffs.assign(a=diffs.a.map(lab), b=diffs.b.map(lab)).round(3).to_string(index=False))
    print('\n## Pareto sets (displayed policies)')
    for c, g in pareto.groupby('scenario', sort=False):
        front = [lab[a] for a in g.agent[g.pareto]]
        ties = sorted({tuple(sorted((a, t))) for a, tw in zip(g.agent, g.tied_with) if tw for t in tw.split(';')})
        print(f'  {c}: {len(front)} non-dominated: {", ".join(front)}')
        print(f'      dominators: ' + ', '.join(f'{lab[a]}={n}' for a, n in zip(g.agent, g.n_dominators) if n))
        if ties:
            print(f'      exact ties: ' + ', '.join(f'{lab[a]} = {lab[b]}' for a, b in ties))
        off = g[g.pareto & (g.n_dominators_field > 0)]
        if len(off):
            print(f'      non-dominated among the displayed but dominated in the field: {", ".join(lab[a] for a in off.agent)}')
    print('\n## roster dependence: equal-weight ranks, bests over the field vs over the displayed 15')
    for c, g in roster.groupby('scenario', sort=False):
        ch = g[g.rank_field != g.rank_shown]
        lead_f = ', '.join(lab[a] for a in g.agent[g.rank_field == 1])
        lead_s = ', '.join(lab[a] for a in g.agent[g.rank_shown == 1])
        print(f'  {c:8s} rank changes {len(ch)}/15, max shift {int((g.rank_field - g.rank_shown).abs().max())}, '
              f'max |score change| {float((g.score_field - g.score_shown).abs().max()):.2f}; leader {lead_f} -> {lead_s}'
              + ('' if ch.empty else '; ' + ', '.join(f'{lab[a]} {f}->{s_}' for a, f, s_ in zip(ch.agent, ch.rank_field, ch.rank_shown))))
    drops = changes[changes.change.str.startswith('drop:')]
    lead_moves = changes[changes.direct_leader_change]
    print(f'\n## roster changes ({changes.change.nunique()}): direct effect (original vs changed roster, own bests)')
    print(f'  changes that move a leader: {len(lead_moves)}')
    if len(lead_moves):
        print(lead_moves[['change', 'scenario', 'direct_leader_original', 'direct_leader_changed', 'agent_rank']].to_string(index=False))
    print(f'  max rank shift of a common policy over all changes: {int(changes.direct_max_rank_shift.max())}')
    print('\n## reference-value effect (order of the common policies, original vs new bests)')
    print(f'  single removals: max order changes {int(drops.ref_n_order_changes.max())}, max shift {int(drops.ref_max_order_shift.max())}, '
          f'max score change {drops.ref_max_score_change.max():.2f}, max rescale {drops.ref_max_rescale.max():.4f}, '
          f'removals with any order change {drops[drops.ref_n_order_changes > 0].change.nunique()}, '
          f'with a common-policy leader change {drops[drops.ref_leader_before != drops.ref_leader_after].change.nunique()}')
    other = changes[~changes.change.str.startswith('drop:')]
    cols = ['change', 'scenario', 'direct_leader_original', 'direct_leader_changed', 'agent_rank', 'direct_n_rank_changes',
            'ref_n_order_changes', 'ref_max_order_shift', 'ref_max_score_change', 'ref_max_rescale']
    print(other[cols].round(4).to_string(index=False))
    moved = drops[drops.ref_n_order_changes > 0]
    if len(moved):
        print(moved[cols].round(4).to_string(index=False))
    print(f'\nwritten: {args.out}/{{gaps,rank_stats,corners,pairwise,overall_diff,rank1_weights,pareto,roster,roster_changes}}.csv, appendix_table.tex')


if __name__ == '__main__':
    main()

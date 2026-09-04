"""Summary scores (mean % distance to the best value per KPI) for the v6w
benchmark field, under selectable KPI sets.

KPI sets (all add final fleet knowledge at the two career-horizon scenarios):
  published    products, episode-mean MTTR, availability, disruptions/10^3 products
  noavail_mean products, episode-mean MTTR, disruptions/10^3 products
  noavail_final products, final-window MTTR, disruptions/10^3 products
  noavail_both products, episode-mean MTTR, final-window MTTR, disruptions/10^3 products

final-window MTTR = mean of the harness's 50-repair rolling MTTR over the last
TAIL of the horizon (the right-hand end of the scenario figures' panel (a)).
Fields: deployable (oracles excluded, best taken over deployables) and full
(oracles included).  The PO twin is an ablation, not a field member.

Usage: uv run --no-sync python scripts/summary_scores.py [--validate] [--md]
"""
import argparse, glob, re
import numpy as np, pandas as pd

ROOT = 'reports/hvp_eval_v6w'
PARTS = 'reports/hvp_v6w_parts'
TAIL = 0.03
SCEN = [('small_scale', 'Small'), ('baseline', 'Base'), ('massive_scale', 'Indust.'),
        ('very_long', 'V-long'), ('lifecycle', 'Lifec.')]
KNOW_SCEN = {'very_long', 'lifecycle'}
ORACLES = {'greedy_reward', 'topsis', 'shortest_processing', 'optimal_assignment', 'reserve_specialist', 'evo_topsis_inf'}
EXCLUDE = {'po_v6', 'po_v6_last', 'hc_v6_ext', 'hc_v6_ext_last', 'hc_v6_wr', 'hc_v6_wr_last'}
TEX = {  # key -> label in tab:results_dist_full
    'greedy_reward': r'\textsc{GreedyReward}$^{*}$', 'empirical_topsis': r'\textsc{Emp-Topsis}',
    'ft_protect_last': 'ft-protect-last (ours)', 'topsis': r'\textsc{Topsis}$^{*}$',
    'hc_v6': 'HTT-RL (ours)', 'shortest_processing': r'\textsc{Spt}$^{*}$',
    'ft_protect': 'ft-protect (ours)', 'ft_fatigue_last': 'ft-fatigue-last (ours)',
    'hc_v6_last': 'HTT-RL-last (ours)', 'ft_fatigue': 'ft-fatigue (ours)',
    'ft_quality': 'ft-quality (ours)', 'ft_gini': 'ft-gini (ours)',
    'ft_quality_last': 'ft-quality-last (ours)', 'batch_milp': r'\textsc{BatchMilp}',
    'optimal_assignment': r'\textsc{Hungarian}$^{*}$', 'empirical_spt': r'\textsc{Emp-Spt}',
    'ft_gini_last': 'ft-gini-last (ours)', 'reserve_specialist': r'\textsc{ReserveSpec}$^{*}$',
    'shortest_queue': r'\textsc{ShortestQueue}', 'least_fatigued': r'\textsc{LeastFatigued}',
    'round_robin': r'\textsc{RoundRobin}', 'random': r'\textsc{Random}',
    'dql_mlp': 'DDQN-MLP (anchor)', 'a2c_mlp': 'A2C-MLP (anchor)',
    'dql_mlp_last': 'DDQN-MLP-last (anchor)', 'grpo_mlp': 'GRPO-MLP (anchor)',
    'grpo_mlp_last': 'GRPO-MLP-last (anchor)', 'least_busy': r'\textsc{LeastBusy}',
    'a2c_mlp_last': 'A2C-MLP-last (anchor)', 'train_weakest': r'\textsc{TrainWeakest}',
    'evo_topsis': r'\textsc{Evo-Topsis}', 'evo_topsis_inf': r'\textsc{Evo-Topsis}$^{*}$'}
PLAIN = {k: (re.sub(r'\\textsc\{([^}]*)\}', r'\1', v)
             .replace(r'$^{*}$', '*').replace(' (ours)', '').replace(' (anchor)', ''))
         for k, v in TEX.items()}
# KPI: (direction, pretty)
KPIS = {'prod': (+1, 'Products'), 'mttr_mean': (-1, 'MTTR (episode mean)'),
        'mttr_final': (-1, 'MTTR (final window)'), 'avail': (+1, 'Availability'),
        'disr': (-1, 'Disr./10^3 products'), 'know': (+1, 'Final knowledge')}
VARIANTS = {'published': ['prod', 'mttr_mean', 'avail', 'disr'],
            'noavail_mean': ['prod', 'mttr_mean', 'disr'],
            'noavail_final': ['prod', 'mttr_final', 'disr'],
            'noavail_both': ['prod', 'mttr_mean', 'mttr_final', 'disr']}
STEP_COLS = ['agent', 'episode', 'step', 'sim_time', 'mttr_rolling', 'fleet_knowledge']


def load_steps(scenario, agents):
    df = pd.read_csv(f'{ROOT}/{scenario}/steps.csv.gz', usecols=STEP_COLS)
    df = df[df.agent.isin(agents)]
    missing = set(agents) - set(df.agent.unique())
    extra = []
    for path in sorted(glob.glob(f'{PARTS}/*/{scenario}/steps.csv.gz')):
        if not missing:
            break
        p = pd.read_csv(path, usecols=STEP_COLS)
        p = p[p.agent.isin(missing)]
        if len(p):
            extra.append(p); missing -= set(p.agent.unique())
    if extra:
        df = pd.concat([df] + extra, ignore_index=True)
    return df, missing


def terminal_mttr(g):
    g = g[(g.mttr_rolling > 0) & (g.step >= 52)]
    hi = g.sim_time.max()
    return float(g[g.sim_time >= hi * (1 - TAIL)].mttr_rolling.mean())


def scenario_metrics(scenario):
    ep = pd.read_csv(f'{ROOT}/{scenario}/episodes.csv')
    agents = sorted(set(ep.agent.unique()) - EXCLUDE)
    ep = ep[ep.agent.isin(agents)]
    m = ep.groupby('agent').agg(prod=('finished_products', 'mean'), mttr_mean=('mttr', 'mean'),
                                avail=('fleet_availability_rate', 'mean'),
                                ill=('ill_technician_count', 'mean'), n=('episode', 'nunique'))
    m['disr'] = m.ill / m['prod'] * 1000.0
    st, missing = load_steps(scenario, agents)
    rows = {}
    for a, g in st.groupby('agent'):
        per_ep = [(terminal_mttr(h), float(h.sort_values('step').fleet_knowledge.iloc[-1]))
                  for _, h in g.groupby('episode')]
        rows[a] = (np.mean([x for x, _ in per_ep]), np.mean([y for _, y in per_ep]) / 1e3)
    m['mttr_final'] = [rows.get(a, (np.nan, np.nan))[0] for a in m.index]
    m['know'] = [rows.get(a, (np.nan, np.nan))[1] for a in m.index]
    return m, missing


def score(m, kpis, field):
    """Mean % distance to the field's best value, per agent (field = index subset)."""
    sub = m.loc[field]
    out = pd.Series(0.0, index=sub.index)
    for k in kpis:
        d, _ = KPIS[k]
        best = sub[k].max() if d > 0 else sub[k].min()
        out += (best - sub[k]) / best * 100.0 if d > 0 else (sub[k] - best) / best * 100.0
    return out / len(kpis)


def main():
    global TAIL
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true', help='compare "published" with tab:results_dist_full')
    ap.add_argument('--md', action='store_true', help='markdown tables')
    ap.add_argument('--tail', type=float, default=TAIL, help='final-window fraction of the horizon')
    ap.add_argument('--exclude', default='', help='comma-separated agent keys dropped from the field (rows AND best-value computation)')
    ap.add_argument('--compare', action='store_true', help='overall score + rank per variant, side by side')
    ap.add_argument('--extra', default='reports/hvp_eval_v6w/mlp_last_step_metrics.csv', help='CSV of scenario,agent,episode,mttr_final,know_final for agents without local step records')
    args = ap.parse_args()
    TAIL = args.tail
    extra = pd.read_csv(args.extra) if args.extra else None
    metrics = {}
    for s, _ in SCEN:
        m, missing = scenario_metrics(s)
        if extra is not None and missing:
            e = extra[extra.scenario == s].groupby('agent').agg(mttr_final=('mttr_final', 'mean'), know_final=('know_final', 'mean'))
            for a in list(missing):
                if a in e.index:
                    m.loc[a, 'mttr_final'] = e.loc[a, 'mttr_final']; m.loc[a, 'know'] = e.loc[a, 'know_final'] / 1e3; missing.discard(a)
        if missing:
            print(f'# {s}: no step records for {sorted(missing)} (final-window MTTR / final knowledge = NaN)')
        metrics[s] = m
    drop = {a for a in args.exclude.split(',') if a}
    full = sorted(set.intersection(*[set(metrics[s].index) for s, _ in SCEN]) - drop)
    if drop:
        print(f'# excluded from the field: {sorted(drop)}')
    deploy = [a for a in full if a not in ORACLES]
    print(f'# field: {len(full)} agents full, {len(deploy)} deployable')
    tables = {}
    for vname, base_kpis in VARIANTS.items():
        for fname, field in (('deployable', deploy), ('full', full)):
            cols = {}
            for s, short in SCEN:
                kp = base_kpis + (['know'] if s in KNOW_SCEN else [])
                cols[short] = score(metrics[s], kp, field)
            t = pd.DataFrame(cols)
            t['Overall'] = t.mean(axis=1)
            t = t.sort_values('Overall')
            t.insert(0, 'rank', range(1, len(t) + 1))
            tables[(vname, fname)] = t
    if args.validate:
        tex = open('paper/Manuscript.tex', encoding='utf-8').read()
        body = tex[tex.index(r'\label{tab:results_dist_full}'):]
        body = body[:body.index(r'\end{table}')]
        pub = {}
        for line in body.splitlines():
            if ' & ' not in line or line.startswith('Agent'):
                continue
            cells = [c.strip() for c in line.rstrip('\\').split('&')]
            lab = cells[0]
            key = next((k for k, v in TEX.items() if v == lab), None)
            assert key, lab
            pub[key] = [float(re.sub(r'\\textbf\{([^}]*)\}', r'\1', c)) for c in cells[1:]]
        t = tables[('published', 'full')]
        dev = pd.DataFrame({k: np.array(v) - t.loc[k, ['Small', 'Base', 'Indust.', 'V-long', 'Lifec.', 'Overall']].to_numpy()
                            for k, v in pub.items()}).T
        dev.columns = ['Small', 'Base', 'Indust.', 'V-long', 'Lifec.', 'Overall']
        print('# validation vs tab:results_dist_full (published - recomputed), max |dev| per column:')
        print(dev.abs().max().round(2).to_string())
        bad = dev[(dev.abs() > 0.06).any(axis=1)]
        if len(bad):
            print('# rows deviating > 0.06:'); print(bad.round(2).to_string())
    if args.compare:
        for fname in ('deployable', 'full'):
            cmp = pd.DataFrame({v: tables[(v, fname)]['Overall'] for v in VARIANTS})
            rk = pd.DataFrame({v: tables[(v, fname)]['rank'] for v in VARIANTS})
            cmp = cmp.sort_values('published').rename(index=PLAIN); rk = rk.loc[[k for k in cmp.index.map({v: k for k, v in PLAIN.items()})]]
            print(f'\n## overall score (rank) per variant, {fname} field, tail={TAIL:.0%}')
            print('| Agent | ' + ' | '.join(VARIANTS) + ' |'); print('|---|' + '|'.join(['--:'] * len(VARIANTS)) + '|')
            for (a, r), (_, k) in zip(cmp.iterrows(), rk.iterrows()):
                print(f'| {a} | ' + ' | '.join(f'{r[v]:.1f} ({int(k[v])})' for v in VARIANTS) + ' |')
        return
    for (vname, fname), t in tables.items():
        print(f'\n## {vname} / {fname} field  (KPIs: ' +
              ', '.join(KPIS[k][1] for k in VARIANTS[vname]) + ' + final knowledge at S4/S5)')
        t = t.rename(index=PLAIN)
        if args.md:
            print('| # | Agent | ' + ' | '.join(t.columns[1:]) + ' |')
            print('|--:|---|' + '|'.join(['--:'] * (len(t.columns) - 1)) + '|')
            for a, r in t.iterrows():
                print(f'| {int(r["rank"])} | {a} | ' + ' | '.join(f'{v:.1f}' for v in r.iloc[1:]) + ' |')
        else:
            print(t.round(1).to_string())


if __name__ == '__main__':
    main()

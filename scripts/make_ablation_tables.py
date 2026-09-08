"""Pivoted ablation tables for the paper, in the layout of tab:results_levers:
agents as columns, scenario x KPI multirow blocks as rows, all five scenarios.

Emits
  reports/hvp_eval_v6w/po_table.tex       tab:results_po      (human-centred vs
                                          performance-only twin)
  reports/hvp_eval_v6w/anchors_table.tex  tab:results_anchors (same information,
                                          traditional networks)

KPIs per scenario, episode-averaged where the scenario has more than one episode:
  MTTR   terminal level of the harness's 50-repair rolling MTTR = mean of the
         recorded rolling value over the last TAIL of the horizon (the
         right-hand end of the curve in the scenario figures), NOT the
         episode-mean MTTR of the summary score
  Know.  final mean per-technician fleet knowledge (x10^3)
  Fat.   episode-mean technician fatigue
  Disr.  technician disruptions per 10^3 finished products (ratio of episode
         means -- the summary-score metric)
  Prod.  final cumulative finished products

Best value per row in bold, second best underlined, as everywhere else in the
paper.  Ties at display precision produce several marks, as they do in the
other tables.
"""
import numpy as np, pandas as pd

ROOT = 'reports/hvp_eval_v6w'
TAIL = 0.03
SCEN = [('small_scale', 'S1', 'Small'), ('baseline', 'S2', 'Baseline'),
        ('massive_scale', 'S3', 'Industrial'), ('very_long', 'S4', 'Very-long'),
        ('lifecycle', 'S5', 'Lifecycle')]
# The four KPIs of tab:results_levers, so the three ablation tables share one
# format.  'fat' (episode-mean technician fatigue) is computed as well and can
# be added per table via the spec's 'kpis' key.
ALL_KPIS = {
    'mttr': (r'MTTR $\downarrow$', -1, '{:.1f}'),
    'know': (r'Know.\ $\uparrow$', +1, '{:.1f}'),
    'fat':  (r'Fat.\ $\downarrow$', -1, '{:.3f}'),
    'disr': (r'Disr.\ $\downarrow$', -1, '{:.0f}'),
    'prod': (r'Prod.\ $\uparrow$', +1, '{:.0f}'),
}
LEVERS_KPIS = ['mttr', 'know', 'disr', 'prod']

TABLES = {
    'po_table': dict(
        agents=['hc_v6', 'ft_quality', 'po_v6'],
        head=['HTT-RL', r'HTT-RL\textsuperscript{quality}', 'PO-HTT-RL'],
        groups=[(2, 'human-centric'), (1, 'performance-only')]),
    'anchors_table': dict(
        agents=['hc_v6', 'ft_quality', 'a2c_mlp', 'grpo_mlp', 'dql_mlp',
                'random', 'train_weakest'],
        head=['HTT-RL', r'HTT-RL\textsuperscript{quality}', 'A2C', 'GRPO', 'DDQN',
              r'\textsc{Random}', r'\textsc{TrainW}'],
        groups=[(2, 'HTT-RL'), (3, 'MLP anchors'), (2, 'reference')]),
}


def terminal_mttr(g):
    g = g[(g.mttr_rolling > 0) & (g.step >= 52)]
    hi = g.sim_time.max()
    return float(g[g.sim_time >= hi * (1 - TAIL)].mttr_rolling.mean())


# The MLP anchors were evaluated on dgy and their step records are not in the
# local tree; EXTRA carries the same per-episode aggregates, extracted there and
# verified against the earlier mlp_last_step_metrics.csv to rounding.
EXTRA = f'{ROOT}/mlp_step_metrics.csv'


def scenario_values(scenario, agents):
    ep = pd.read_csv(f'{ROOT}/{scenario}/episodes.csv')
    st = pd.read_csv(f'{ROOT}/{scenario}/steps.csv.gz',
                     usecols=['agent', 'episode', 'step', 'sim_time',
                              'mttr_rolling', 'fleet_knowledge', 'fatigue_mean'])
    ep = ep[ep.agent.isin(agents)]
    st = st[st.agent.isin(agents)]
    try:
        xt = pd.read_csv(EXTRA)
        xt = xt[xt.scenario == scenario]
    except FileNotFoundError:
        xt = pd.DataFrame(columns=['agent', 'mttr_final', 'know_final', 'fat_mean'])
    out = {}
    for a in agents:
        e, s = ep[ep.agent == a], st[st.agent == a]
        n_ep = e.episode.nunique()
        assert n_ep, (scenario, a, 'no episodes')
        if s.episode.nunique() == n_ep:
            mttr = np.mean([terminal_mttr(g) for _, g in s.groupby('episode')])
            know = np.mean([g.sort_values('step').fleet_knowledge.iloc[-1]
                            for _, g in s.groupby('episode')]) / 1e3
            fat = float(s.fatigue_mean.mean())
        else:
            x = xt[xt.agent == a]
            assert len(x) == n_ep, (scenario, a, 'no step records and no extras',
                                    n_ep, len(x))
            mttr = float(x.mttr_final.mean())
            know = float(x.know_final.mean()) / 1e3
            fat = float(x.fat_mean.mean())
        out[a] = {
            'mttr': mttr, 'know': know, 'fat': fat,
            'disr': e.ill_technician_count.mean() / e.finished_products.mean() * 1000,
            'prod': e.finished_products.mean(),
            'n': n_ep,
        }
    return out


def render(name, spec):
    agents, head, groups = spec['agents'], spec['head'], spec['groups']
    kpis = [(k, *ALL_KPIS[k]) for k in spec.get('kpis', LEVERS_KPIS)]
    n = len(agents)
    lines = [r'\begin{tabular}{@{}ll' + 'r' * n + r'@{}}', r'\toprule']
    if groups:
        cells, rules, col = [], [], 3
        for span, title in groups:
            cells.append(r'\multicolumn{%d}{c}{%s}' % (span, title) if span > 1 else title)
            rules.append(r'\cmidrule(lr){%d-%d}' % (col, col + span - 1))
            col += span
        lines.append('& & ' + ' & '.join(cells) + r' \\')
        lines.append(''.join(rules))
    lines.append(r'Scenario & KPI & ' + ' & '.join(head) + r' \\')
    for scenario, tag, label in SCEN:
        vals = scenario_values(scenario, agents)
        lines.append(r'\midrule')
        for j, (key, klabel, direction, fmt) in enumerate(kpis):
            row = [vals[a][key] for a in agents]
            txts = [fmt.format(v) for v in row]
            uniq = sorted({float(t) for t in txts}, reverse=direction > 0)
            best = fmt.format(uniq[0])
            second = fmt.format(uniq[1]) if len(uniq) > 1 else None
            cells = []
            for t in txts:
                if t == best:
                    t = r'\textbf{' + t + '}'
                elif second is not None and t == second:
                    t = r'\underline{' + t + '}'
                cells.append(t)
            first = (r'\multirow{%d}{*}{\makecell[l]{%s\\%s}}' % (len(kpis), tag, label)
                     if j == 0 else '')
            lines.append(f'{first} & {klabel} & ' + ' & '.join(cells) + r' \\')
        eps = {vals[a]['n'] for a in agents}
        print(f'% {name} {scenario}: {sorted(eps)} episode(s) per agent')
    lines += [r'\bottomrule', r'\end{tabular}']
    body = '\n'.join(lines)
    open(f'{ROOT}/{name}.tex', 'w').write(body + '\n')
    return body


if __name__ == '__main__':
    for name, spec in TABLES.items():
        print(f'\n%%% {name}')
        print(render(name, spec))

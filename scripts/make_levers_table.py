"""Pivoted one-lever fine-tune table for the paper (tab:results_levers):
fine-tunes as columns, scenario x KPI multirow blocks as rows.

KPIs per scenario (episode-averaged where the scenario has >1 episode):
  MTTR   terminal level of the harness's 50-repair rolling MTTR = mean of
         the recorded rolling value over the last TAIL of the horizon
         (the right-hand end of the curve in the scenario figures)
  Know.  final mean per-technician fleet knowledge (x10^3)
  Disr.  technician disruptions per 10^3 finished products (episode totals,
         ratio of episode means -- the summary-score metric)
  Prod.  final cumulative finished products

Writes reports/hvp_eval_v6w/levers_table.tex and prints it."""
import numpy as np, pandas as pd

ROOT = 'reports/hvp_eval_v6w'
TAIL = 0.03
SCEN = [('small_scale', 'S1', 'Small'), ('baseline', 'S2', 'Baseline'),
        ('massive_scale', 'S3', 'Industrial'), ('very_long', 'S4', 'Very-long'),
        ('lifecycle', 'S5', 'Lifecycle')]
# Each fine-tune is represented by the checkpoint twin with the better
# overall summary score (full field, appendix): final checkpoint for the
# fatigue and protect levers, best-evaluation checkpoint otherwise --
# the same selection as the pre-pivot table and the S3/S5 prose numbers.
AGENTS = ['hc_v6', 'ft_quality', 'ft_fatigue_last', 'ft_protect_last', 'ft_gini']
HEAD = ['base', r'\textsuperscript{quality}', r'\textsuperscript{fatigue}',
        r'\textsuperscript{protect}', r'\textsuperscript{gini}']
# (key, row label, direction, format)
KPIS = [('mttr', r'MTTR $\downarrow$', -1, '{:.1f}'),
        ('know', r'Know.\ $\uparrow$', +1, '{:.1f}'),
        ('disr', r'Disr.\ $\downarrow$', -1, '{:.0f}'),
        ('prod', r'Prod.\ $\uparrow$', +1, '{:.0f}')]


def terminal_mttr(g):
    g = g[(g.mttr_rolling > 0) & (g.step >= 52)]
    hi = g.sim_time.max()
    tail = g[g.sim_time >= hi * (1 - TAIL)]
    return float(tail.mttr_rolling.mean())


def scenario_values(scenario):
    ep = pd.read_csv(f'{ROOT}/{scenario}/episodes.csv')
    st = pd.read_csv(f'{ROOT}/{scenario}/steps.csv.gz')
    ep = ep[ep.agent.isin(AGENTS)]
    st = st[st.agent.isin(AGENTS)]
    out = {}
    for a in AGENTS:
        e = ep[ep.agent == a]
        s = st[st.agent == a]
        n_ep = e.episode.nunique()
        assert n_ep == s.episode.nunique(), (scenario, a, n_ep, s.episode.nunique())
        mttr = np.mean([terminal_mttr(g) for _, g in s.groupby('episode')])
        know = np.mean([g.sort_values('step').fleet_knowledge.iloc[-1]
                        for _, g in s.groupby('episode')]) / 1e3
        disr = e.ill_technician_count.mean() / e.finished_products.mean() * 1000
        prod = e.finished_products.mean()
        out[a] = {'mttr': mttr, 'know': know, 'disr': disr, 'prod': prod, 'n': n_ep}
    return out


lines = []
lines.append(r'\begin{tabular}{@{}llrrrrr@{}}')
lines.append(r'\toprule')
lines.append(r'& & \multicolumn{5}{c}{HTT-RL} \\')
lines.append(r'\cmidrule(lr){3-7}')
lines.append(r'Scenario & KPI & ' + ' & '.join(HEAD) + r' \\')
for i, (scenario, tag, name) in enumerate(SCEN):
    vals = scenario_values(scenario)
    lines.append(r'\midrule')
    for j, (key, label, direction, fmt) in enumerate(KPIS):
        row = [vals[a][key] for a in AGENTS]
        best = max(row) if direction > 0 else min(row)
        cells = []
        for v in row:
            txt = fmt.format(v)
            if fmt.format(best) == txt:
                txt = r'\textbf{' + txt + '}'
            cells.append(txt)
        first = (r'\multirow{4}{*}{\makecell[l]{' + tag + r'\\' + name + '}}'
                 if j == 0 else '')
        lines.append(f'{first} & {label} & ' + ' & '.join(cells) + r' \\')
    n = {vals[a]['n'] for a in AGENTS}
    assert len(n) == 1, (scenario, n)
    print(f'% {scenario}: {n.pop()} episode(s) per agent')
    base = vals[AGENTS[0]]
    for a in AGENTS[1:]:
        d = {k: 100 * (vals[a][k] / base[k] - 1) for k in ('mttr', 'know', 'disr', 'prod')}
        print(f"%   {a:16s} vs base: " + ' '.join(f"{k} {d[k]:+.1f}%" for k in d))
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
body = '\n'.join(lines)
open(f'{ROOT}/levers_table.tex', 'w').write(body + '\n')
print(body)

"""Full per-scenario KPI table for the appendix: agents as columns grouped
by archetype (multicolumn headers), scenarios as multirow blocks, one
metric per row.  Field = the 30-agent benchmark field + the PO twin.
Values are means over the scenario's episodes; final fleet knowledge and
final-window MTTR come from the step records (MLP final-checkpoint
anchors: reports/hvp_eval_v6w/mlp_last_step_metrics.csv).

Writes reports/hvp_eval_v6w/full_metrics_table.tex (tabular body only)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, 'scripts')
import summary_scores as ss

GROUPS = [
    ('HTT-RL (ours)', [('hc_v6', 'HTT-RL'), ('hc_v6_last', 'HTT-RL-last'),
                       ('ft_quality', r'\textsuperscript{quality}'), ('ft_quality_last', r'\textsuperscript{quality}-last'),
                       ('ft_fatigue', r'\textsuperscript{fatigue}'), ('ft_fatigue_last', r'\textsuperscript{fatigue}-last'),
                       ('ft_protect', r'\textsuperscript{protect}'), ('ft_protect_last', r'\textsuperscript{protect}-last'),
                       ('ft_gini', r'\textsuperscript{gini}'), ('ft_gini_last', r'\textsuperscript{gini}-last'),
                       ('po_v6', 'PO-HTT-RL')]),
    ('Multicriteria', [('topsis', r'\textsc{Topsis}$^{*}$'), ('empirical_topsis', r'\textsc{Emp-Topsis}')]),
    ('Rule-based', [('shortest_processing', r'\textsc{Spt}$^{*}$'), ('empirical_spt', r'\textsc{Emp-Spt}'),
                    ('reserve_specialist', r'\textsc{ReserveSpec}$^{*}$'), ('least_fatigued', r'\textsc{LeastFat}'),
                    ('train_weakest', r'\textsc{TrainW}'), ('shortest_queue', r'\textsc{ShortQ}'),
                    ('round_robin', r'\textsc{RoundR}'), ('least_busy', r'\textsc{LeastBusy}'),
                    ('random', r'\textsc{Random}'), ('greedy_reward', r'\textsc{GreedyReward}$^{*}$')]),
    ('Optimization', [('optimal_assignment', r'\textsc{Hungarian}$^{*}$'), ('batch_milp', r'\textsc{BatchMilp}')]),
    ('Learning-based', [('a2c_mlp', 'A2C'), ('a2c_mlp_last', 'A2C-last'), ('dql_mlp', 'DDQN'),
                        ('dql_mlp_last', 'DDQN-last'), ('grpo_mlp', 'GRPO'), ('grpo_mlp_last', 'GRPO-last')]),
]
# (key, label, direction (+1 higher better / -1 lower / 0 report-only), fmt, scale)
ROWS = [('prod', 'Products', +1, '{:.0f}', 1), ('thrpt', 'Thrpt.\\ rate', +1, '{:.2f}', 1),
        ('mttr_mean', 'MTTR', -1, '{:.1f}', 1), ('mttr_final', 'MTTR (final window)', -1, '{:.1f}', 1),
        ('mtbf', 'MTBF', 0, '{:.0f}', 1), ('avail', 'Availability', +1, '{:.3f}', 1),
        ('balance', 'Balance', +1, '{:.3f}', 1), ('ill', 'Disruptions', -1, '{:.0f}', 1),
        ('disr', r'Disr./$10^3$ prod.', -1, '{:.0f}', 1), ('breakd', 'Breakdowns', 0, '{:.0f}', 1),
        ('know', r'Final know.\ ($\times 10^3$)', +1, '{:.1f}', 1)]
SCEN = [('small_scale', 'S1 Small'), ('baseline', 'S2 Baseline'), ('massive_scale', 'S3 Industrial'),
        ('very_long', 'S4 Very-long'), ('lifecycle', 'S5 Lifecycle')]
extra = pd.read_csv('reports/hvp_eval_v6w/mlp_last_step_metrics.csv')
ss.EXCLUDE = ss.EXCLUDE - {'po_v6'}
keys = [k for _, g in GROUPS for k, _ in g]
lines = []
ncol = len(keys)
lines.append(r'\begin{tabular}{@{}ll' + 'r' * ncol + '@{}}')
lines.append(r'\toprule')
heads, cmids, pos = [], [], 3
for name, g in GROUPS:
    heads.append(r'\multicolumn{%d}{c}{%s}' % (len(g), name))
    cmids.append(r'\cmidrule(lr){%d-%d}' % (pos, pos + len(g) - 1)); pos += len(g)
lines.append('& & ' + ' & '.join(heads) + r' \\')
lines.append(''.join(cmids))
lines.append('Scenario & Metric & ' + ' & '.join(r'\rot{' + lab + '}' for _, g in GROUPS for _, lab in g) + r' \\')
for scen, sname in SCEN:
    ep = pd.read_csv(f'reports/hvp_eval_v6w/{scen}/episodes.csv')
    m, missing = ss.scenario_metrics(scen)
    e = extra[extra.scenario == scen].groupby('agent').agg(mttr_final=('mttr_final', 'mean'), know_final=('know_final', 'mean'))
    for a in list(missing):
        if a in e.index:
            m.loc[a, 'mttr_final'] = e.loc[a, 'mttr_final']; m.loc[a, 'know'] = e.loc[a, 'know_final'] / 1e3
    g = ep.groupby('agent').mean(numeric_only=True)
    m['thrpt'] = g.throughput_rate; m['mtbf'] = g.mtbf; m['balance'] = g.workload_balance; m['breakd'] = g.total_breakdowns
    lines.append(r'\midrule')
    for i, (key, label, d, fmt, scale) in enumerate(ROWS):
        vals = [m.loc[k, key] if k in m.index else np.nan for k in keys]
        arr = np.array(vals, dtype=float)
        best = (np.nanmax(arr) if d > 0 else np.nanmin(arr)) if d else None
        cells = []
        for v in vals:
            if not np.isfinite(v): cells.append('---'); continue
            txt = fmt.format(v / scale)
            if d and fmt.format(best / scale) == txt: txt = r'\textbf{' + txt + '}'
            cells.append(txt)
        first = r'\multirow{%d}{*}{\makecell[l]{%s}}' % (len(ROWS), sname.replace(' ', r'\\', 1)) if i == 0 else ''
        lines.append(f'{first} & {label} & ' + ' & '.join(cells) + r' \\')
lines.append(r'\bottomrule'); lines.append(r'\end{tabular}')
Path('reports/hvp_eval_v6w/full_metrics_table.tex').write_text('\n'.join(lines) + '\n')
print(f'{ncol} agent columns x {len(ROWS)} metrics x {len(SCEN)} scenarios written')

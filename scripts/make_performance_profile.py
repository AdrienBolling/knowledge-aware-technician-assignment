"""Dolan-More performance profiles for the benchmark's summary-score data.

Treats each (scenario, scored KPI) pair as a "problem" (5 x 4 = 20) and each
agent as a "solver".  The performance ratio of agent s on problem p is

    r_{p,s} = v_{p,s} / min_s v_{p,s}     for a cost KPI  (MTTR, disruptions)
    r_{p,s} = max_s v_{p,s} / v_{p,s}     for a benefit KPI (products, knowledge)

so r >= 1 with r = 1 at the frontier, and for a cost KPI (r - 1) * 100 is
exactly the percentage distance the summary score averages.  The profile is

    rho_s(tau) = |{p : r_{p,s} <= tau}| / |P|

Emits standalone panels at a fixed physical size, assembled by the manuscript
as LaTeX subfigures (same pattern as make_scenario_figures.py):

    paper/figures/panels/profile_deployable.pdf   25-agent deployable field
    paper/figures/panels/profile_full.pdf         30-agent field, informed included
    paper/figures/panels/profile_legend.pdf       shared legend strip (full width)
"""
import importlib.util, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

spec = importlib.util.spec_from_file_location('ss', os.path.join(os.path.dirname(__file__), 'summary_scores.py'))
ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)

OUT = 'paper/figures/panels'
PANEL = (3.2, 2.35)          # inches; included at 0.49\textwidth
KPIS = [('prod', +1), ('mttr_mean', -1), ('disr', -1), ('know', +1)]
TAU_MAX = 2.5
# key: (label, colour, linestyle, linewidth)
ACCENT = {'hc_v6':            ('HTT-RL',            '#0072B2', '-',   1.8),
          'ft_quality':       (r'HTT-RL$^{quality}$','#009E73', '-',   1.8),
          'empirical_topsis': ('Emp-Topsis',        '#D55E00', '-',   1.3),
          'empirical_spt':    ('Emp-Spt',           '#CC79A7', '-.',  1.3),
          'batch_milp':       ('BatchMilp',         '#E69F00', '--',  1.3),
          'random':           ('Random',            '#4D4D4D', ':',   1.1)}
INFORMED = {'greedy_reward': ('GreedyReward*', '#000000', (0, (4, 1.5)), 1.4),
            'topsis':        ('Topsis*',       '#56B4E9', (0, (4, 1.5)), 1.4)}
MLPS = ['a2c_mlp', 'grpo_mlp', 'dql_mlp']

plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42})


def load():
    extra = pd.read_csv(f'{ss.ROOT}/mlp_last_step_metrics.csv')
    out = {}
    for s, _ in ss.SCEN:
        m, missing = ss.scenario_metrics(s)
        e = extra[extra.scenario == s].groupby('agent').agg(know_final=('know_final', 'mean'))
        for a in list(missing):
            if a in e.index:
                m.loc[a, 'know'] = e.loc[a, 'know_final'] / 1e3; missing.discard(a)
        if missing:
            print(f'# {s}: no knowledge record for {sorted(missing)}')
        out[s] = m
    return out


def ratios(metrics, field):
    """-> DataFrame agent x problem of performance ratios (>= 1)."""
    cols = {}
    for s, short in ss.SCEN:
        sub = metrics[s].loc[field]
        for k, d in KPIS:
            best = sub[k].max() if d > 0 else sub[k].min()
            cols[f'{short}:{k}'] = (best / sub[k]) if d > 0 else (sub[k] / best)
    return pd.DataFrame(cols)


def profile(r, taus):
    return np.array([(r <= t + 1e-12).mean() for t in taus])


def draw(ax, R, accents, title):
    taus = np.concatenate([np.linspace(1.0, 1.5, 400), np.linspace(1.5, TAU_MAX, 200)])
    for a in R.index:
        if a in accents: continue
        style = '--' if a in MLPS else '-'
        ax.plot(taus, profile(R.loc[a].to_numpy(), taus), style, color='0.72', lw=0.7, zorder=1)
    for a, (lab, c, ls, lw) in accents.items():
        if a not in R.index: continue
        ax.plot(taus, profile(R.loc[a].to_numpy(), taus), linestyle=ls, color=c, lw=lw, zorder=3)
    ax.set_xscale('log')
    ax.set_xticks([1, 1.1, 1.25, 1.5, 2, 2.5])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(1, TAU_MAX); ax.set_ylim(0, 1.02)
    ax.set_xlabel(r'$\tau$ (factor of the best value)')
    ax.set_ylabel(r'$\rho(\tau)$')
    ax.set_title(title)
    ax.grid(alpha=0.25, lw=0.5)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)


def main():
    os.makedirs(OUT, exist_ok=True)
    metrics = load()
    full = sorted(set.intersection(*[set(metrics[s].index) for s, _ in ss.SCEN]))
    deploy = [a for a in full if a not in ss.ORACLES]
    for name, field, accents, title in (
            ('profile_deployable', deploy, ACCENT, f'Deployable field ({len(deploy)} agents)'),
            ('profile_full', full, {**ACCENT, **INFORMED}, f'Full field ({len(full)} agents)')):
        R = ratios(metrics, field)
        fig, ax = plt.subplots(figsize=PANEL)
        draw(ax, R, accents, title)
        fig.subplots_adjust(left=0.145, right=0.965, top=0.885, bottom=0.20)
        fig.savefig(f'{OUT}/{name}.pdf'); plt.close(fig)
        best = (R <= 1 + 1e-9).mean(axis=1).sort_values(ascending=False)
        print(f'\n## {name}: rho(1) top rows'); print((best[best > 0]).round(3).to_string())
        print(f'   worst-cell ratio: ' + ', '.join(
            f'{a}={R.loc[a].max():.2f}' for a in accents if a in R.index))
        for t in (1.05, 1.10, 1.25):
            row = {a: profile(R.loc[a].to_numpy(), [t])[0] for a in accents if a in R.index}
            print(f'   rho({t:.2f}): ' + ', '.join(f'{a}={v:.2f}' for a, v in row.items()))

    handles = [Line2D([], [], color=c, linestyle=ls, lw=lw, label=lab)
               for lab, c, ls, lw in list(ACCENT.values()) + list(INFORMED.values())]
    handles += [Line2D([], [], color='0.72', ls='-', lw=0.7, label='other rules'),
                Line2D([], [], color='0.72', ls='--', lw=0.7, label='MLP anchors')]
    fig = plt.figure(figsize=(6.42, 0.34))  # full-width strip, as make_scenario_figures.py
    fig.legend(handles=handles, loc='center', ncol=8, frameon=False,
               handlelength=1.8, columnspacing=1.0, fontsize=6.5)
    fig.savefig(f'{OUT}/profile_legend.pdf'); plt.close(fig)
    print(f'\nwritten: {OUT}/profile_{{deployable,full,legend}}.pdf')


if __name__ == '__main__':
    main()

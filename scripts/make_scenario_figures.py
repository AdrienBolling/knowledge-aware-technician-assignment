"""Per-scenario MTTR(100-repair rolling) + fleet-knowledge figures,
restricted to the summary-table roster (deployable field, no oracles)."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = 'reports/hvp_eval_v6w'
PARTS = 'reports/hvp_v6w_parts'
OUT = 'paper/figures'
SCEN = {'small_scale':'S1 Small','baseline':'S2 Baseline','massive_scale':'S3 Industrial',
        'very_long':'S4 Very-long','lifecycle':'S5 Lifecycle'}
ACCENT = {'hc_v6':('HTT-RL','#0072B2','-',1.8),
          'ft_quality':(r'HTT-RL$^{quality}$','#009E73','-',1.8),
          'empirical_topsis':('Emp-Topsis','#D55E00','-',1.3),
          'empirical_spt':('Emp-Spt','#CC79A7','-.',1.3),
          'random':('Random','#4D4D4D',':',1.1)}
RULES = ['batch_milp','shortest_queue','least_fatigued','round_robin','least_busy','train_weakest']
MLPS  = ['a2c_mlp','grpo_mlp','dql_mlp']
ORDER = RULES + MLPS + ['random','empirical_spt','empirical_topsis','ft_quality','hc_v6']
RET = [0.8e6, 2.5e6, 4.2e6]

plt.rcParams.update({'font.size':7.5,'axes.titlesize':8,'axes.labelsize':7.5,
                     'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})

def style(agent):
    if agent in ACCENT:
        lab,c,ls,lw = ACCENT[agent]; return c,ls,lw,1.0
    if agent in MLPS: return '#8C8C8C','--',0.9,0.9
    return '#BFBFBF','-',0.9,0.9

def load(scenario):
    df = pd.read_csv(f'{ROOT}/{scenario}/steps.csv.gz')
    parts = [pd.read_csv(f'{PARTS}/{a}/{scenario}/steps.csv.gz') for a in MLPS]
    df = pd.concat([df] + parts, ignore_index=True)
    return df[df.agent.isin(ORDER)]

def r50(g):
    """The harness's recorded rolling MTTR (last 50 completed repairs),
    plotted as-is; rows before the first completed repair are dropped
    and the curve starts once the window is full (step >= 52)."""
    g = g[(g.mttr_rolling > 0) & (g.step >= 52)].sort_values('step')
    if len(g) < 3: return None
    return g.sim_time.to_numpy(float), g.mttr_rolling.to_numpy(float)

def mean_curve(sub, series):
    """Average per-episode curves on a common sim-time grid."""
    curves = []
    for _, g in sub.groupby('episode'):
        r = series(g)
        if r is not None: curves.append(r)
    if not curves: return None
    lo = max(c[0][0] for c in curves); hi = min(c[0][-1] for c in curves)
    grid = np.linspace(lo, hi, 500)
    return grid, np.mean([np.interp(grid, *c) for c in curves], axis=0)

for scenario, title in SCEN.items():
    df = load(scenario)
    xs = 1e6 if df.sim_time.max() > 1e6 else 1e3
    xl = 'simulation time (M t.u.)' if xs == 1e6 else 'simulation time (k t.u.)'
    fig, axes2 = plt.subplots(2, 2, figsize=(7.0, 4.5))
    axes = axes2.ravel()
    pscale = 1e3 if df.finished_products.max() > 5000 else 1.0
    plab = (r'cumulative products ($\times 10^3$)' if pscale == 1e3
            else 'cumulative products')
    for ax, series, ylab, tt in (
        (axes[0], r50, 'rolling MTTR (t.u.)', 'Rolling MTTR, 50-repair window'),
        (axes[1], lambda g: (g.sort_values('step').sim_time.to_numpy(float),
                             g.sort_values('step').fleet_knowledge.to_numpy(float)/1e3),
         r'mean fleet knowledge ($\times 10^3$)', 'Fleet knowledge'),
        (axes[3], lambda g: (g.sort_values('step').sim_time.to_numpy(float),
                             g.sort_values('step').finished_products.to_numpy(float)/pscale),
         plab, 'Cumulative finished products')):
        for a in ORDER:
            sub = df[df.agent == a]
            if sub.empty: continue
            r = mean_curve(sub, series)
            if r is None: continue
            c, ls, lw, alpha = style(a)
            z = 3 if a in ACCENT else 1
            y = r[1]
            if ax is axes[0]:
                # centered rolling median (kills isolated spikes, no
                # edge padding artifacts) then a light mean: ~3% of
                # the horizon combined
                y = (pd.Series(y).rolling(15, center=True, min_periods=1).median()
                       .rolling(7, center=True, min_periods=1).mean().to_numpy())
            ax.plot(r[0]/xs, y, color=c, ls=ls, lw=lw, alpha=alpha, zorder=z)
        if scenario == 'lifecycle':
            for t in RET: ax.axvline(t/1e6, color='#999999', ls=(0,(2,2)), lw=0.7, zorder=0)
        if ax is axes[0]:
            p90s = [np.percentile(l.get_ydata(), 90) for l in ax.get_lines() if len(l.get_ydata())]
            if p90s:
                lo = min(np.percentile(l.get_ydata(), 10) for l in ax.get_lines() if len(l.get_ydata()))
                ax.set_ylim(lo*0.95, max(p90s)*1.15)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=0.22, lw=0.5)
        if ax in (axes[2], axes[3]): ax.set_xlabel(xl)
        ax.set_ylabel(ylab); ax.set_title(tt, fontsize=8)
    # bottom-left: the summary score's output-normalised absence metric
    ep = pd.read_csv(f'{ROOT}/{scenario}/episodes.csv')
    ep = ep[ep.agent.isin(ORDER)].groupby('agent').mean(numeric_only=True)
    ipk = (ep.ill_technician_count / ep.finished_products * 1000).reindex(ORDER).dropna()
    ipk = ipk.sort_values(ascending=False)  # best (lowest) ends on top
    SHORT = {'hc_v6':'HTT-RL','ft_quality':'HTT-RL$^{quality}$','empirical_topsis':'Emp-Topsis',
             'empirical_spt':'Emp-Spt','batch_milp':'Milp','shortest_queue':'ShortQ',
             'least_fatigued':'LeastFat','round_robin':'RoundR','least_busy':'LeastBusy',
             'train_weakest':'TrainW','random':'Random','a2c_mlp':'A2C','grpo_mlp':'GRPO','dql_mlp':'DDQN'}
    ax = axes[2]
    cols = [style(a)[0] for a in ipk.index]
    ax.barh(range(len(ipk)), ipk.values, color=cols, height=0.72)
    ax.set_yticks(range(len(ipk)), [SHORT[a] for a in ipk.index], fontsize=6.3)
    ax.set_xlabel(r'technician disruptions / $10^3$ products')
    ax.set_title('Disruptions per output (episode total)', fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=0.22, lw=0.5, axis='x')
    handles = [Line2D([],[], color=ACCENT[a][1], ls=ACCENT[a][2], lw=ACCENT[a][3], label=ACCENT[a][0])
               for a in ('hc_v6','ft_quality','empirical_topsis','empirical_spt','random')]
    handles += [Line2D([],[], color='#BFBFBF', ls='-', lw=0.9, label='other rules (6)'),
                Line2D([],[], color='#8C8C8C', ls='--', lw=0.9, label='MLP anchors (3)')]
    fig.legend(handles=handles, ncol=7, loc='upper center', frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, 1.02), columnspacing=1.2, handlelength=1.9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f'{OUT}/mttr_knowledge_{scenario}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(scenario, 'done')

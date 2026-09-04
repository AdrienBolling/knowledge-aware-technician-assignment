"""Per-scenario benchmark panels for the paper's Results section.

Every scenario yields four standalone panel PDFs that the manuscript
assembles as LaTeX subfigures (one 2x2 figure* per scenario), plus one
legend strip shared by all scenarios:

    paper/figures/panels/<scenario>_mttr.pdf         rolling MTTR (50-repair window)
    paper/figures/panels/<scenario>_knowledge.pdf    mean per-technician fleet knowledge
    paper/figures/panels/<scenario>_disruptions.pdf  technician disruptions / 10^3 products, stacked by type
    paper/figures/panels/<scenario>_products.pdf     cumulative finished products
    paper/figures/panels/scenario_legend.pdf         shared legend strip

Panels are emitted at a fixed physical size (PANEL, inches) with no
tight-bbox cropping, so every panel scales identically when included at
0.49\textwidth (elsarticle 3p: \textwidth = 468 pt = 6.5 in).  Roster =
the summary-table deployable field (no oracles)."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = 'reports/hvp_eval_v6w'
PARTS = 'reports/hvp_v6w_parts'
DISR = 'reports/hvp_eval_disr'   # per-type disruption counts (same episodes, instrumented re-run)
OUT = 'paper/figures/panels'
PANEL = (3.2, 2.15)   # inches; 0.49 * 6.5 in = 3.19 in in the manuscript
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
# (key suffix, legend label, hatch): injuries solid, exhaustion hatched, vacations dotted
TYPES = [('injury', 'injury', ''), ('exhaustion', 'exhaustion', '////'), ('vacation', 'vacation', '....')]
plt.rcParams['hatch.linewidth'] = 0.5
SHORT = {'hc_v6':'HTT-RL','ft_quality':'HTT-RL$^{quality}$','empirical_topsis':'Emp-Topsis',
         'empirical_spt':'Emp-Spt','batch_milp':'Milp','shortest_queue':'ShortQ',
         'least_fatigued':'LeastFat','round_robin':'RoundR','least_busy':'LeastBusy',
         'train_weakest':'TrainW','random':'Random','a2c_mlp':'A2C','grpo_mlp':'GRPO','dql_mlp':'DDQN'}

plt.rcParams.update({'font.size':7.5,'axes.titlesize':8,'axes.labelsize':7.5,
                     'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})


MIN_PER_MONTH = 43830.0   # 30.44 d x 1440 min, 1 t.u. ~ 1 minute
MIN_PER_YEAR = 525960.0

def add_calendar_axis(ax, xs, tmax_tu):
    """Top axis with approximate calendar-time ticks (1 t.u. ~ 1 min)."""
    if tmax_tu > 1.5e6:
        vals = [y * MIN_PER_YEAR for y in (2, 4, 6, 8)]
        labs = ['2 y', '4 y', '6 y', '8 y']
    elif tmax_tu > 1.2e5:
        vals = [m * MIN_PER_MONTH for m in (1, 2, 3, 4)]
        labs = ['1 mo', '2 mo', '3 mo', '4 mo']
    else:
        vals = [m * MIN_PER_MONTH for m in (1, 2)]
        labs = ['1 mo', '2 mo']
    vals, labs = zip(*[(v, l) for v, l in zip(vals, labs) if v <= tmax_tu * 1.02])
    sec = ax.secondary_xaxis('top')
    sec.set_xticks([v / xs for v in vals], labs)
    sec.tick_params(labelsize=6.3, colors='#777777', length=2.5)
    for sp in sec.spines.values():
        sp.set_visible(False)
    return sec

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

def new_panel():
    fig, ax = plt.subplots(figsize=PANEL, constrained_layout=True)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=0.22, lw=0.5)
    return fig, ax

def time_panel(df, scenario, series, ylab, smooth_mttr=False):
    """One time-series panel over the roster (mean curve per agent)."""
    xs = 1e6 if df.sim_time.max() > 1e6 else 1e3
    xl = 'simulation time (M t.u.)' if xs == 1e6 else 'simulation time (k t.u.)'
    fig, ax = new_panel()
    for a in ORDER:
        sub = df[df.agent == a]
        if sub.empty: continue
        r = mean_curve(sub, series)
        if r is None: continue
        c, ls, lw, alpha = style(a)
        z = 3 if a in ACCENT else 1
        y = r[1]
        if smooth_mttr:
            # centered rolling median (kills isolated spikes, no edge
            # padding artifacts) then a light mean: ~3% of the horizon
            y = (pd.Series(y).rolling(15, center=True, min_periods=1).median()
                   .rolling(7, center=True, min_periods=1).mean().to_numpy())
        ax.plot(r[0]/xs, y, color=c, ls=ls, lw=lw, alpha=alpha, zorder=z)
    if scenario == 'lifecycle':
        for t in RET: ax.axvline(t/1e6, color='#999999', ls=(0,(2,2)), lw=0.7, zorder=0)
    if smooth_mttr:
        # data curves only -- axvline markers carry 2-point (0,1) ydata
        curves = [l.get_ydata() for l in ax.get_lines() if len(l.get_ydata()) > 10]
        if curves:
            lo = min(np.median(y) for y in curves)
            ax.set_ylim(lo*0.85, max(np.percentile(y, 90) for y in curves)*1.15)
    add_calendar_axis(ax, xs, df.sim_time.max())
    ax.set_xlabel(xl); ax.set_ylabel(ylab)
    return fig

def disruption_panel(scenario):
    """Bar panel: the summary score's output-normalised absence metric,
    each bar split into the three disruption types.  Totals come from the
    published tree (table-consistent); the type proportions come from the
    disruption-instrumented re-run of the same episodes (reports/
    hvp_eval_disr), whose learned-agent totals differ by up to ~3% at the
    30-technician scenarios (GPU-kernel nondeterminism)."""
    ep = pd.read_csv(f'{ROOT}/{scenario}/episodes.csv')
    ep = ep[ep.agent.isin(ORDER)].groupby('agent').mean(numeric_only=True)
    ipk = (ep.ill_technician_count / ep.finished_products * 1000).reindex(ORDER).dropna()
    ipk = ipk.sort_values(ascending=False)  # best (lowest) ends on top
    dz = pd.read_csv(f'{DISR}/{scenario}/episodes.csv')
    dz = dz[dz.agent.isin(ORDER)].groupby('agent').mean(numeric_only=True)
    shares = dz[[f'disruptions_{t}' for t, _, _ in TYPES]]
    shares = shares.div(shares.sum(axis=1), axis=0).reindex(ipk.index)
    fig, ax = new_panel()
    ax.grid(False); ax.grid(alpha=0.22, lw=0.5, axis='x')
    cols = [style(a)[0] for a in ipk.index]
    left = np.zeros(len(ipk))
    for t, label, hatch in TYPES:
        seg = ipk.values * shares[f'disruptions_{t}'].to_numpy()
        ax.barh(range(len(ipk)), seg, left=left, color=cols, height=0.72,
                hatch=hatch, edgecolor='white', linewidth=0.4)
        left += seg
    ax.set_yticks(range(len(ipk)), [SHORT[a] for a in ipk.index], fontsize=6.3)
    ax.set_xlabel(r'technician disruptions / $10^3$ products')
    handles = [Patch(facecolor='#9A9A9A', edgecolor='white', hatch=h, label=label)
               for _, label, h in TYPES]
    ax.legend(handles=handles, loc='upper right', frameon=False, fontsize=6.3,
              handlelength=1.6, handleheight=1.1, borderaxespad=0.2)
    return fig

def legend_strip():
    handles = [Line2D([],[], color=ACCENT[a][1], ls=ACCENT[a][2], lw=ACCENT[a][3], label=ACCENT[a][0])
               for a in ('hc_v6','ft_quality','empirical_topsis','empirical_spt','random')]
    handles += [Line2D([],[], color='#BFBFBF', ls='-', lw=0.9, label='other rules (6)'),
                Line2D([],[], color='#8C8C8C', ls='--', lw=0.9, label='MLP anchors (3)')]
    fig = plt.figure(figsize=(6.5, 0.26))
    fig.legend(handles=handles, ncol=7, loc='center', frameon=False,
               fontsize=7, columnspacing=1.2, handlelength=1.9)
    fig.savefig(f'{OUT}/scenario_legend.pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)

os.makedirs(OUT, exist_ok=True)
legend_strip()
for scenario, title in SCEN.items():
    df = load(scenario)
    pscale = 1e3 if df.finished_products.max() > 5000 else 1.0
    plab = (r'cumulative products ($\times 10^3$)' if pscale == 1e3
            else 'cumulative products')
    by_step = lambda g: g.sort_values('step')
    panels = {
        'mttr': time_panel(df, scenario, r50, 'rolling MTTR (t.u.)', smooth_mttr=True),
        'knowledge': time_panel(
            df, scenario,
            lambda g: (by_step(g).sim_time.to_numpy(float),
                       by_step(g).fleet_knowledge.to_numpy(float)/1e3),
            r'mean fleet knowledge ($\times 10^3$)'),
        'disruptions': disruption_panel(scenario),
        'products': time_panel(
            df, scenario,
            lambda g: (by_step(g).sim_time.to_numpy(float),
                       by_step(g).finished_products.to_numpy(float)/pscale),
            plab),
    }
    for name, fig in panels.items():
        fig.savefig(f'{OUT}/{scenario}_{name}.pdf')
        plt.close(fig)
    print(scenario, 'done')

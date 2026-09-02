"""Deep-dive figure: HTT-RL vs PO-HTT-RL vs the two empirical rules at
the industrial scale, with mean +/- std bands.

Band semantics differ by panel (stated in the caption): the rolling-MTTR
and repair-quality bands are the std WITHIN the rolling window; the
throughput band is the std ACROSS MACHINES of the per-machine
processing rate; fleet-knowledge and fatigue bands are the std ACROSS
TECHNICIANS.  All curves and bands are means over the 3 episodes.

Data: reports/deepdive_parts/<agent>/massive_scale/steps.csv.gz —
re-evaluated locally with per-decision records incl. fleet_knowledge_std
(not part of the published generation; industrial rows there are means
over the same seeds, but 30-technician evals are not bit-reproducible).
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = 'reports/deepdive_parts'
OUT = 'paper/figures/deepdive_industrial.pdf'
AGENTS = {
    'hc_v6':            ('HTT-RL',        '#0072B2', '-',  2.0),
    'po_v6':            ('PO-HTT-RL',     '#E69F00', '--', 2.0),
    'empirical_topsis': ('Emp-Topsis',    '#D55E00', '-',  1.5),
    'empirical_spt':    ('Emp-Spt',       '#CC79A7', '-.', 1.5),
}
GRID = np.linspace(2000, 100000, 400)
W = 5000.0  # throughput-rate window (t.u.)

plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42})


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

def smooth(y, k1=15, k2=7):
    return (pd.Series(y).rolling(k1, center=True, min_periods=1).median()
              .rolling(k2, center=True, min_periods=1).mean().to_numpy())

def per_episode(df, fn):
    """fn(g) -> (t, y); interpolated onto GRID, one row per episode."""
    rows = []
    for _, g in df.groupby('episode'):
        t, y = fn(g.sort_values('step'))
        rows.append(np.interp(GRID, t, y))
    return np.array(rows)

fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.3))
data = {a: pd.read_csv(f'{ROOT}/{a}/massive_scale/steps.csv.gz') for a in AGENTS}

def panel(ax, mean_fn, band_fn, ylab, title, do_smooth=False):
    for a, (lab, c, ls, lw) in AGENTS.items():
        df = data[a]
        m = per_episode(df, mean_fn)
        b = per_episode(df, band_fn)
        mu, sd = m.mean(axis=0), b.mean(axis=0)
        if do_smooth:
            mu, sd = smooth(mu), smooth(sd)
        ax.plot(GRID/1e3, mu, color=c, ls=ls, lw=lw, zorder=3)
        ax.fill_between(GRID/1e3, np.maximum(mu-sd, 0.0), mu+sd, color=c, alpha=0.14, lw=0, zorder=1)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=0.22, lw=0.5)
    if ax in (axes[0][0], axes[0][1], axes[0][2]):
        add_calendar_axis(ax, 1e3, GRID[-1])
    ax.set_ylabel(ylab); ax.set_title(title, fontsize=8, pad=14)

def mttr(g):
    g = g[(g.mttr_rolling > 0) & (g.step >= 52)]
    return g.sim_time.to_numpy(float), g.mttr_rolling.to_numpy(float)

def mttr_sd(g):
    g = g[(g.mttr_rolling > 0) & (g.step >= 52)]
    return g.sim_time.to_numpy(float), g.mttr_rolling_std.to_numpy(float)

QW = 52  # decisions ~ 50 repairs, matches the MTTR window

def quality(g):
    g = g.sort_values('step')
    q = pd.Series(g.repair_quality.to_numpy(float)).rolling(QW, min_periods=QW).mean()
    return g.sim_time.to_numpy(float), q.to_numpy()

def quality_sd(g):
    g = g.sort_values('step')
    q = pd.Series(g.repair_quality.to_numpy(float)).rolling(QW, min_periods=QW).std()
    return g.sim_time.to_numpy(float), q.to_numpy()

def mrate(g):    return g.sim_time.to_numpy(float), g.machine_rate_mean.to_numpy(float)
def mrate_sd(g): return g.sim_time.to_numpy(float), g.machine_rate_std.to_numpy(float)

def knowledge(g):   return g.sim_time.to_numpy(float), g.fleet_knowledge.to_numpy(float)/1e3
def knowledge_sd(g):return g.sim_time.to_numpy(float), g.fleet_knowledge_std.to_numpy(float)/1e3
def fatigue(g):     return g.sim_time.to_numpy(float), g.fatigue_mean.to_numpy(float)
def fatigue_sd(g):  return g.sim_time.to_numpy(float), g.fatigue_std.to_numpy(float)

def thr_rate(g):
    t, p = g.sim_time.to_numpy(float), g.finished_products.to_numpy(float)
    prev = np.interp(t - W, t, p, left=0.0)
    return t, (p - prev) / W * 1e3   # products per 10^3 t.u.

panel(axes[0][0], mttr, mttr_sd, 'rolling MTTR (t.u.)',
      'Rolling MTTR (band: window std)', do_smooth=True)
panel(axes[0][1], quality, quality_sd, 'repair quality',
      'Rolling repair quality (band: window std)', do_smooth=True)
panel(axes[0][2], mrate, mrate_sd, r'items / $10^3$ t.u. per machine',
      'Per-machine rate (band: machines)', do_smooth=True)
panel(axes[1][0], knowledge, knowledge_sd, r'fleet knowledge ($\times 10^3$)',
      'Fleet knowledge (band: technicians)')
panel(axes[1][1], fatigue, fatigue_sd, 'technician fatigue',
      'Fatigue (band: technicians)', do_smooth=True)
for ax in axes[1][:2]:
    ax.set_xlabel('simulation time (k t.u.)')
axes[0][2].set_xlabel('simulation time (k t.u.)')

axes[1][2].axis('off')
handles = [Line2D([], [], color=c, ls=ls, lw=lw, label=lab)
           for lab, c, ls, lw in AGENTS.values()]
axes[1][2].legend(handles=handles, loc='center', frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT, bbox_inches='tight')
print('saved', OUT)

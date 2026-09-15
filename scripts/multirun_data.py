"""Per-episode KPI records for every agent and scenario of the benchmark.

Sources
  S1-S3  the published generation, reports/hvp_eval_v6w (3 / 5 / 3 episodes
         per agent, identical seeds across agents).
  S4-S5  the five-run generation, reports/hvp_s45/run<r>/<agent>/<scenario>
         (one episode per run, eval seed 20260908 + r, identical across
         agents); the run index is the episode id.  Final-window MTTR and
         final fleet knowledge come from reports/hvp_s45/step_metrics.csv,
         computed from the step records on the evaluation host with the same
         definitions as terminal_mttr() below.

Statistics
  Means follow the published definitions: per-KPI episode means, and
  disruptions per 10^3 products = mean disruptions / mean products.
  Standard deviations are sample standard deviations (ddof=1) over episodes;
  for disruptions per 10^3 products they are taken over per-episode ratios.
"""
import glob

import numpy as np
import pandas as pd

V6W = 'reports/hvp_eval_v6w'
V6W_PARTS = 'reports/hvp_v6w_parts'
MLP_EXTRA = 'reports/hvp_eval_v6w/mlp_last_step_metrics.csv'
S45 = 'reports/hvp_s45'
TAIL = 0.03
MULTIRUN = ('very_long', 'lifecycle')
SCENARIOS = ('small_scale', 'baseline', 'massive_scale', 'very_long', 'lifecycle')
EP_COLS = {'finished_products': 'prod', 'mttr': 'mttr_mean', 'fleet_availability_rate': 'avail',
           'ill_technician_count': 'ill', 'throughput_rate': 'thrpt', 'mtbf': 'mtbf',
           'workload_balance': 'balance', 'total_breakdowns': 'breakdowns'}
STEP_COLS = ['agent', 'episode', 'step', 'sim_time', 'mttr_rolling', 'fleet_knowledge']
KPI_COLS = ['prod', 'mttr_mean', 'mttr_final', 'avail', 'ill', 'disr', 'know',
            'thrpt', 'mtbf', 'balance', 'breakdowns']


def terminal_mttr(g):
    """Mean rolling MTTR over the last TAIL of the horizon (published definition)."""
    g = g[(g.mttr_rolling > 0) & (g.step >= 52)]
    hi = g.sim_time.max()
    return float(g[g.sim_time >= hi * (1 - TAIL)].mttr_rolling.mean())


def _v6w_steps(scenario, agents):
    """Step records for `agents`, from the merged tree, then from the parts."""
    df = pd.read_csv(f'{V6W}/{scenario}/steps.csv.gz', usecols=STEP_COLS)
    df = df[df.agent.isin(agents)]
    missing = set(agents) - set(df.agent.unique())
    extra = []
    for path in sorted(glob.glob(f'{V6W_PARTS}/*/{scenario}/steps.csv.gz')):
        if not missing:
            break
        p = pd.read_csv(path, usecols=STEP_COLS)
        p = p[p.agent.isin(missing)]
        if len(p):
            extra.append(p)
            missing -= set(p.agent.unique())
    return (pd.concat([df] + extra, ignore_index=True) if extra else df), missing


def _step_metrics_v6w(scenario, agents):
    st, missing = _v6w_steps(scenario, agents)
    rows = [(a, e, terminal_mttr(h), float(h.sort_values('step').fleet_knowledge.iloc[-1]))
            for (a, e), h in st.groupby(['agent', 'episode'])]
    sm = pd.DataFrame(rows, columns=['agent', 'episode', 'mttr_final', 'know_final'])
    if missing:
        xt = pd.read_csv(MLP_EXTRA)
        xt = xt[(xt.scenario == scenario) & xt.agent.isin(missing)]
        sm = pd.concat([sm, xt[sm.columns]], ignore_index=True)
    return sm


def episodes(scenario, source='auto'):
    """One row per (agent, episode) with every KPI of KPI_COLS.

    source='auto' reads S4/S5 from the five-run generation and S1-S3 from the
    published one; source='v6w' forces the published generation everywhere.
    """
    if scenario in MULTIRUN and source == 'auto':
        frames = []
        for path in sorted(glob.glob(f'{S45}/run*/*/{scenario}/episodes.csv')):
            e = pd.read_csv(path)
            e['episode'] = int(path.split('/')[-4][3:])
            frames.append(e)
        ep = pd.concat(frames, ignore_index=True)
        sm = pd.read_csv(f'{S45}/step_metrics.csv')
        sm = sm[sm.scenario == scenario].drop(columns=['episode', 'scenario'])
        sm = sm.rename(columns={'run': 'episode'})
    else:
        ep = pd.read_csv(f'{V6W}/{scenario}/episodes.csv')
        sm = _step_metrics_v6w(scenario, sorted(ep.agent.unique()))
    out = ep[['agent', 'episode'] + list(EP_COLS)].rename(columns=EP_COLS)
    out = out.merge(sm, on=['agent', 'episode'], how='left')
    out['know'] = out.pop('know_final') / 1e3
    out['disr'] = out.ill / out['prod'] * 1000.0
    return out[['agent', 'episode'] + KPI_COLS]


def agent_stats(ep):
    """Per-agent mean and std of every KPI; index = agent, columns = (kpi, stat)."""
    g = ep.groupby('agent')
    mean = g[KPI_COLS].mean()
    mean['disr'] = mean.ill / mean['prod'] * 1000.0
    std = g[KPI_COLS].std(ddof=1)
    out = pd.concat({'mean': mean, 'std': std}, axis=1).swaplevel(axis=1).sort_index(axis=1)
    out[('n', '')] = g.size()
    return out

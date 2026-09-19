"""Model-based sequential dispatching baselines on the human-state model.

Two non-learned baselines that optimise HTT-RL's own TRAINING objective
(the human-centric v5 reward stack of
``run_configs/benchmark_suite/train_multiscale_v5.json``) with an explicit
model of the simulator's human dynamics, instead of a learned policy:

* :class:`GreedyTrainingRewardAgent` scores the current assignment only:
  the v5 reward components that this assignment moves, as far as the model
  predicts them at decision time.  No look-ahead over other tickets and no
  terminal value.
* :class:`RollingHorizonMPCAgent` plans the current ticket together with
  ``horizon_k`` further tickets under the same model, adds a terminal value
  for the post-window human state, executes the first assignment and
  re-plans at the next decision (rolling horizon / model predictive
  control).  With ``horizon_k=0`` and ``terminal_weight=0`` it is the
  greedy (tested).

They separate the value of long-horizon credit assignment (HTT-RL) from the
value of a good state-dependent scoring rule on the same human-state model.

Information
-----------
Both agents receive the env handle through :meth:`Agent.attach_env`, like
every baseline of the benchmark.  They read the quantities that HTT-RL's
set observation carries (technician fatigue, knowledge grids and match,
busy and disruption flags, the ticket type and its base repair time), the
static technician parameters of the human-state model (fatigue rates,
Wright exponent and diffusion width of the knowledge grid, travel time,
knowledge and fatigue multiplier parameters, disruption rates) and the v5
reward coefficients.  They never clone or step the simulator (a
``copy.deepcopy`` of a SimPy world is not possible) and never read the
future: breakdown times, disruption draws and future ticket types are
unknown to them.  Rates (decisions and products per time unit) and
ticket-type frequencies are estimated online from the episode so far.

Human-state model (technician ``i``)
------------------------------------
* Repair on a ticket with base time ``B``: travel ``T`` then
  ``B * m_k * m_f`` on the tool, with
  ``m_k = f* + (1 - f*) * exp(-alpha* * n**b)`` (``n`` = experience of the
  ticket's knowledge-grid cell, ``b`` = the grid's Wright exponent,
  ``(f*, alpha*)`` = the per-component overrides when
  ``failure_wise_knowledge_parameters`` is on) and ``m_f = exp(a F)``
  (exponential) or ``1 + F`` (linear), ``F`` = fatigue at the job start.
  This is ``GymTechnician.compute_repair_time``.
* Fatigue (Jaber): exponential recovery ``F exp(-mu t)`` while idle or
  absent; frozen from the job start (before travel) to the completion; at
  the completion ``F + (1 - F) (1 - exp(-lambda floor(B)))`` (the simulator
  accumulates on the BASE time).  A technician who is busy at the job start
  starts at its predicted free time with the predicted post-job fatigue
  (no idle gap, so no recovery).
* Knowledge: a completed repair adds the exact Gaussian bump of
  ``KnowledgeGrid.add_ticket_knowledge`` (scipy kernel, truncate 3, cropped
  at the border) to the grid.  Inside the planning window the bump of an
  earlier planned repair lowers ``m_k`` of a later planned repair by the
  same technician (that repair always starts after the earlier one ends).
* Availability: free technicians start now.  A busy technician is free at
  the predicted end of the jobs this agent queued on it (the env queues a
  job behind the current one).  The bookkeeping is re-anchored at every
  decision on observed events: a completion in ``env.repair_log()`` starts
  the next queued job, a re-queued ticket leaves the technician, and the
  elapsed time of an absence delays the queue.  An absent technician is
  free at the expected end of the absence (mean duration of the detected
  disruption type minus the time already seen, at least 20 % of it).
  Queued work is stretched by the expected absence time per time unit
  (``sum rate * mean duration`` of the disruption types, the exhaustion
  rate at the technician's fatigue).
* Preemption risk: a preemptive disruption during travel and repair
  (exhaustion at rate ``coefficient * F``, injury at its Poisson rate)
  re-queues the ticket; the completion time includes the expected delay
  ``1.5 p (T + d)`` (half the lost work plus a new travel and repair).

Objective
---------
Components ``c`` of the v5 stack with coefficients ``lambda_c``; HTT-RL
standardises each component with running statistics, so the weight of a
component is ``C_c = lambda_c / sigma_c``.  ``sigma_c`` are fixed scales
calibrated on the training worlds (``scripts/calibrate_seqbase.py``,
because the normaliser state of HTT-RL was never checkpointed).  The
running means shift every plan by the same amount and are dropped.
Discount: semi-MDP ``gamma = 0.9999`` per time unit, ``beta = -ln gamma``.
Decisions arrive at the measured rate ``rho``; a component paid at every
decision with value ``x(t)`` is integrated as ``rho * int gamma**t x(t)``.
For one assignment (ticket arrival ``r``, start ``s``, completion ``c``):

* ``repair_quality``, ``fatigue_cost``, ``busy_technician`` (paid at the
  decision): ``gamma**r (C_q (1 - m_k) - C_f F_i(r) - C_b busy_i(r))``.
* ``fleet_availability`` and ``throughput_delta``: the machine is down
  from ``r`` to ``c``; cost ``kappa int_r^c gamma**t dt`` with
  ``kappa = C_av rho / M + C_tp rho phi / M_type``, where
  ``phi = x exp(-x)``, ``x = pi / rho`` (products per decision) is the
  sensitivity of the "at least one product since the last decision"
  indicator, and one machine is ``1 / M_type`` of the product rate ``pi``
  (every stage treated as binding).
* ``knowledge_increment`` (potential-based, ``Phi`` = mean fleet knowledge
  volume): the shaping telescopes, so a repair that adds ``dV_i`` to
  technician ``i``'s volume pays ``C_ki gamma**c dV_i / N`` (the window's
  potential term, with the window end at the completion).
* ``workload_balance`` (``-std`` of fleet fatigue): linearised,
  ``d std / d F_i = (F_i - mean F) / (N std)``, integrated over the exact
  fatigue difference between the plan and the no-assignment path (frozen
  during the job, jump at the completion, exponential recovery).
* ``terminal_finished_products`` and ``terminal_fleet_knowledge``: the
  episode-end terms, discounted by ``gamma**(T_end - now)``; downtime
  costs ``pi / M_type`` products per time unit.
* Terminal value (weight ``terminal_weight``; MPC only) for the rewards of
  decisions after the window: (a) knowledge: the bump lowers ``m_k`` of
  technician ``i`` on the frequent ticket types ``c'``; future type-``c'``
  tickets served by ``i`` arrive at ``rho p_c' share_ic'`` and each gains
  ``C_q dm + kappa_c' B_c' m_f dm``; integrated with discount over
  ``terminal_horizon`` time units after the window.  (b) Fatigue: the
  fatigue increase decays at ``mu_i``; future decisions that choose ``i``
  (rate ``rho share_i``) pay ``C_f dF`` and a slower repair
  (``d m_f / dF`` times the expected downtime cost of one repair by ``i``).
  ``p`` = ticket-type frequencies observed in the episode (exponentially
  weighted), ``share`` = this agent's own assignment shares (exponentially
  weighted, with a uniform prior).  The potential term of the knowledge
  shaping already credits the knowledge stock at the window end; the
  terminal value adds the modelled effect of that stock on later quality
  and downtime rewards.  In the exact objective the two overlap (the
  shaping telescopes), which ``terminal_weight`` absorbs.

Search (MPC)
------------
The env pops a ticket as soon as it arrives, so the pending queue is empty
at nearly every decision and queued tickets alone give no look-ahead.  The
window holds the current ticket, the queued tickets (at most
``horizon_k``), and anticipated tickets for the remaining look-ahead
slots.  Anticipated ticket ``k`` arrives at ``k / rho`` with a type drawn
from the observed type frequencies: ``scenarios`` stratified quantile
draws with fixed Latin-hypercube permutations (deterministic, and common
random numbers for every first action).  For each first action and each
scenario the assignments of the later slots are searched: exact
enumeration when ``n_active**(K-1) <= exact_limit``, else a beam of width
``beam_width`` per (first action, scenario).  A later slot can use only
technicians that the model predicts free at its arrival (all active
technicians when none is free, as the env mask does).  The value of a
first action is its own score plus the mean over scenarios of the best
continuation.  The first actions are the env mask (exactly), reduced to
the ``root_beam`` best one-step scores when larger.  The best first action
is executed (lowest index on ties) and the plan is discarded.

Approximations (deliberate)
---------------------------
* Anticipated tickets use the mean base time and knowledge parameters of
  their type; their arrivals are deterministic at ``k / rho``.
* Knowledge bumps inside the window change ``m_k`` of later planned
  repairs by the same technician, but not their volume gain ``dV`` (that
  uses the grid at decision time).
* Knowledge decay (every ``knowledge_decay_interval``) and lifecycle
  events are not predicted; the model re-reads the state at every decision.
* The duration of an unknown job (a busy technician that this agent did
  not assign, e.g. after a reset) is half a typical job.
* Exhaustion risk enters only through the preemption delay; the capacity
  lost when an idle technician becomes absent is not priced.
* The throughput model treats every stage as the bottleneck.
* ``workload_balance`` uses the active fleet (the env includes retired
  technicians) and a linearised standard deviation.
* The fatigue recovery at a job start is continuous (the simulator
  truncates the idle time to an integer).
* A queued job keeps the ``m_k`` of its assignment time (the knowledge of
  the jobs before it in the queue is not added); the fatigue chain of a
  queue does not recover during a predicted absence.
* In a later window slot, a technician that is still busy at the ticket
  arrival pays the fatigue of its planned post-job state (only reachable
  when no technician is free, as in the env mask fallback).
* The scales ``sigma_c`` come from the Topsis rule (HTT-RL's
  behaviour-cloning teacher) on the training worlds, not from HTT-RL's own
  trajectories.
"""

from __future__ import annotations

import json
import math
import time
import weakref
from array import array
from pathlib import Path
from typing import Any

import numpy as np

from agents.base import Agent
from agents.baselines.heuristics import _available

_REPO = Path(__file__).resolve().parents[3]
V5_REWARD_CONFIG = _REPO / "run_configs/benchmark_suite/train_multiscale_v5.json"
SEQBASE_PARAMS_PATH = _REPO / "run_configs/agents/seqbase_mpc.json"
GAMMA_PER_TU = 0.9999  # conf/train.yaml time_based_discount gamma (v5/v6)

V5_COMPONENTS = (
    "repair_quality",
    "fatigue_cost",
    "busy_technician",
    "throughput_delta",
    "workload_balance",
    "fleet_availability",
    "knowledge_increment",
    "terminal_finished_products",
    "terminal_fleet_knowledge",
)

DEFAULT_PARAMS: dict[str, Any] = {
    "horizon_k": 2,
    "terminal_weight": 1.0,
    "scenarios": 8,
    "beam_width": 4,
    "exact_limit": 16,
    "root_beam": 8,
    "top_keys": 8,
    "terminal_horizon": 5000.0,
    "sigma": {name: 1.0 for name in V5_COMPONENTS},
}

_PLANNER_KEYS = tuple(k for k in DEFAULT_PARAMS if k != "sigma")


def _merged(params: dict[str, Any] | None) -> dict[str, Any]:
    out = json.loads(json.dumps(DEFAULT_PARAMS))
    for key, value in (params or {}).items():
        if key == "sigma" and isinstance(value, dict):
            out["sigma"].update({k: float(v) for k, v in value.items()})
        elif key in DEFAULT_PARAMS:
            out[key] = value
    return out


def load_params(path: Path | str | None = SEQBASE_PARAMS_PATH) -> dict[str, Any]:
    """Planner parameters: defaults updated by the JSON at ``path`` if any."""
    if path is not None and Path(path).is_file():
        return _merged(json.loads(Path(path).read_text()))
    return _merged(None)


def v5_weights(sigma: dict[str, float],
               reward_config: Path | str = V5_REWARD_CONFIG) -> dict[str, Any]:
    """Objective weights ``C_c = lambda_c / sigma_c`` of the v5 stack.

    Disabled components get weight 0.  Also returns the
    ``fleet_knowledge_scale`` of the terminal knowledge term.
    """
    cfg = json.loads(Path(reward_config).read_text())["gym"]
    rew = cfg["reward"]
    weights = {}
    for name in V5_COMPONENTS:
        comp = rew.get(name, {})
        lam = float(comp.get("coefficient", 0.0)) if comp.get("enabled") else 0.0
        s = float(sigma.get(name, 1.0)) if rew.get("normalize_components") else 1.0
        weights[name] = lam / max(s, 1e-8)
    return {
        "C": weights,
        "pbrs": bool(rew.get("knowledge_increment_potential_based", False)),
        "fleet_knowledge_scale": float(cfg.get("fleet_knowledge_scale", 10.0)),
    }


# ---------------------------------------------------------------------------
# Knowledge-grid bump table (exact scipy kernel)
# ---------------------------------------------------------------------------

_BUMP_CACHE: dict[tuple[float, tuple[int, ...]], np.ndarray] = {}


def bump_table(sigma: float, shape: tuple[int, ...]) -> np.ndarray:
    """``table[c, j]``: experience added to flat cell ``j`` by one repair
    centred on flat cell ``c`` (``KnowledgeGrid.add_ticket_knowledge``)."""
    key = (float(sigma), tuple(int(s) for s in shape))
    table = _BUMP_CACHE.get(key)
    if table is None:
        from scipy.ndimage import gaussian_filter

        size = int(np.prod(shape))
        table = np.empty((size, size), dtype=np.float64)
        for c in range(size):
            imp = np.zeros(shape, dtype=float)
            imp[np.unravel_index(c, shape)] = 1.0
            table[c] = gaussian_filter(
                imp, sigma=float(sigma), mode="constant", truncate=3.0
            ).ravel()
        _BUMP_CACHE[key] = table
    return table


# ---------------------------------------------------------------------------
# Pure planning core
# ---------------------------------------------------------------------------


def _fatigue_multiplier(mode: str, alpha: float, F):
    """``m_f`` of ``GymTechnician.get_fatigue_multiplier``."""
    if mode == "exponential":
        return np.exp(alpha * F)
    if mode == "linear":
        return 1.0 + F
    return np.ones_like(F) if isinstance(F, np.ndarray) else 1.0


def _fatigue_slope(mode: str, alpha: float, F):
    """``d m_f / d F`` at ``F``."""
    if mode == "exponential":
        return alpha * np.exp(alpha * F)
    if mode == "linear":
        return np.ones_like(F) if isinstance(F, np.ndarray) else 1.0
    return np.zeros_like(F) if isinstance(F, np.ndarray) else 0.0


def score_assignment(ctx: dict[str, Any], slot: dict[str, Any],
                     free: np.ndarray, Ff: np.ndarray, terminal_weight: float,
                     mk: np.ndarray | None = None):
    """Score the assignment of one ticket to every technician, for a batch
    of model states.

    ``free`` and ``Ff`` have shape ``(R, N)``: the time (relative to now) at
    which each technician is free and its fatigue at that time.  Ticket
    quantities in ``slot`` are ``(R, N)`` / ``(N,)`` per technician and
    ``(R, 1)`` / scalar per ticket.  ``mk`` overrides ``slot["mk"]``.
    Returns ``(value, completion, fatigue_after)``, each ``(R, N)``.
    """
    beta = ctx["beta"]
    tau_end = ctx["tau_end"]
    C = ctx["C"]
    mu = ctx["mu"]
    r = slot["r"]
    if mk is None:
        mk = slot["mk"]
    base = slot["base"]

    s = np.maximum(r, free)
    Fs = Ff * np.exp(-mu * (s - free))
    mf = _fatigue_multiplier(ctx["fatigue_mode"], ctx["fatigue_alpha"], Fs)
    dur = ctx["travel"] + base * mk * mf
    p_int = np.minimum(1.0, (ctx["preempt_fatigue_rate"] * Fs + ctx["preempt_rate"]) * dur)
    c = s + dur * (1.0 + 1.5 * p_int)
    Fp = np.minimum(1.0, Fs + (1.0 - Fs) * (1.0 - np.exp(-ctx["lam"] * slot["floor_base"])))

    if "F_r" in slot:  # current decision: exact fatigue and busy flags
        F_r = slot["F_r"]
        busy_r = slot["busy_r"]
    else:
        idle = free <= r
        F_r = np.where(idle, Ff * np.exp(-mu * np.maximum(r - free, 0.0)), Ff)
        busy_r = (~idle).astype(np.float64)

    e_r = math.exp(-beta * min(r, tau_end)) if np.isscalar(r) else np.exp(-beta * np.minimum(r, tau_end))
    e_cu = np.exp(-beta * c)
    e_c = np.maximum(e_cu, math.exp(-beta * tau_end))  # = exp(-beta min(c, tau_end))
    value = e_r * (C["repair_quality"] * (1.0 - mk)
                   - C["fatigue_cost"] * F_r
                   - C["busy_technician"] * busy_r)
    # Machine down from arrival to completion (availability, throughput).
    value = value - slot["kappa"] * (e_r - e_c) / beta
    # Episode-end products lost to that downtime.
    value = value - ctx["C_term_products"] * slot["prod_per_tu"] * (
        np.minimum(c, tau_end) - min(float(r), tau_end))
    # Knowledge credit when the repair completes inside the episode.
    inside = c < tau_end
    value = value + inside * (C["knowledge_increment"] * e_cu
                              + ctx["C_term_knowledge"]) * slot["dV"] / ctx["n_active"]
    # Fatigue balance: exact fatigue difference to the no-assignment path.
    bm = beta + mu
    e_s = np.exp(-beta * s)
    I1 = Fs * ((e_s - e_cu) / beta - e_s * (1.0 - np.exp(-bm * (c - s))) / bm)
    dF_c = Fp - Fs * np.exp(-mu * (c - s))
    I2 = dF_c * e_cu / bm
    value = value - C["workload_balance"] * ctx["rho"] * ctx["std_grad"] * (I1 + I2)
    if terminal_weight:
        t_w = np.maximum(c, ctx["window_end"])
        span = np.clip(tau_end - t_w, 0.0, ctx["terminal_horizon"])
        vk = slot["VK"] * np.exp(-beta * t_w) * (-np.expm1(-beta * span)) / beta
        vf = -ctx["fatigue_future"] * dF_c * e_cu / bm
        value = value + terminal_weight * (vk + vf)
    return value, c, Fp


def _rows(slot: dict[str, Any], scen_rows: np.ndarray) -> dict[str, Any]:
    """Per-row view of a slot: ``dist`` slots index their per-scenario
    arrays by the scenario of each row."""
    if slot["kind"] != "dist":
        return slot
    out = {"kind": "dist", "r": slot["r"]}
    for key, val in slot.items():
        if key in ("kind", "r"):
            continue
        out[key] = val[scen_rows]
    return out


def _within_window_mk(ctx, sl, prior_t, prior_c, n_rows):
    """``m_k`` per (row, technician) of a later slot, with the knowledge
    bumps of the earlier planned repairs of the same row applied."""
    N = ctx["n"]
    mk = np.broadcast_to(sl["mk"], (n_rows, N)).copy()
    if not ctx["knowledge_enabled"] or prior_t.shape[1] == 0:
        return mk
    k = prior_t.shape[1]
    rows = np.repeat(np.arange(n_rows), k)
    techs = prior_t.ravel()
    cells_now = np.broadcast_to(np.asarray(sl["cell"]).reshape(-1), (n_rows,)) \
        if np.ndim(sl["cell"]) else np.full(n_rows, int(sl["cell"]))
    extra = np.zeros((n_rows, N))
    np.add.at(extra, (rows, techs), ctx["bumps"][techs, prior_c.ravel(), cells_now[rows]])
    Gc = np.broadcast_to(sl["Gc"], (n_rows, N))[rows, techs]
    fstar = np.broadcast_to(np.asarray(sl["fstar"]).reshape(-1), (n_rows,))[rows] \
        if np.ndim(sl["fstar"]) else sl["fstar"]
    alpha = np.broadcast_to(np.asarray(sl["alpha"]).reshape(-1), (n_rows,))[rows] \
        if np.ndim(sl["alpha"]) else sl["alpha"]
    kn = (Gc + extra[rows, techs]) ** ctx["b"][techs]
    mk[rows, techs] = fstar + (1.0 - fstar) * np.exp(-alpha * kn)
    return mk


_TECH_KEYS_CTX = ("mu", "lam", "std_grad", "fatigue_future")
_TECH_KEYS_SLOT = ("mk", "dV", "Gc", "VK", "F_r", "busy_r")


def _columns(ctx: dict[str, Any], slot: dict[str, Any], cols: np.ndarray):
    """Views of ``ctx`` and ``slot`` restricted to the technicians ``cols``
    (last axis of the per-technician arrays)."""
    N = ctx["n"]
    ctx2, slot2 = dict(ctx), dict(slot)
    for src, dst, keys in ((ctx, ctx2, _TECH_KEYS_CTX), (slot, slot2, _TECH_KEYS_SLOT)):
        for key in keys:
            v = src.get(key)
            if isinstance(v, np.ndarray) and v.ndim and v.shape[-1] == N:
                dst[key] = v[..., cols]
    return ctx2, slot2


def plan_first_action(ctx: dict[str, Any], slots: list[dict[str, Any]],
                      candidates: np.ndarray, params: dict[str, Any]) -> tuple[int, np.ndarray, np.ndarray]:
    """Rolling-horizon search.  Returns ``(action, roots, value per root)``.

    ``slots[0]`` is the current ticket; later slots are ``fixed`` (a queued
    ticket, the same in every scenario) or ``dist`` (an anticipated ticket
    whose arrays have a leading scenario axis of length
    ``ctx["n_scenarios"]``).
    """
    wT = float(params["terminal_weight"])
    N = ctx["n"]
    active = ctx["active"]
    free0 = ctx["free"][None, :]
    Ff0 = ctx["Ff"][None, :]
    v0, c0, Fp0 = score_assignment(ctx, slots[0], free0, Ff0, wT)
    v0, c0, Fp0 = v0[0], c0[0], Fp0[0]
    cand = np.asarray(candidates, dtype=np.int64)
    if len(slots) == 1 or len(cand) == 1:
        return int(cand[int(np.argmax(v0[cand]))]), cand, v0[cand]

    root_beam = int(params["root_beam"])
    if len(cand) > root_beam:
        order = np.argsort(-v0[cand], kind="stable")[:root_beam]
        roots = np.sort(cand[order])
    else:
        roots = cand
    J = len(roots)
    S = int(ctx["n_scenarios"])
    n_act = max(1, int(active.sum()))
    K = len(slots) - 1
    n_branch = n_act ** (K - 1)
    W = n_branch if n_branch <= int(params["exact_limit"]) else int(params["beam_width"])
    W = max(1, W)

    # Root rows: one per (first action, scenario); group id = j * S + s.
    n_groups = J * S
    first = np.repeat(roots, S)
    rows_idx = np.arange(n_groups)
    free = np.repeat(free0, n_groups, axis=0)
    Ff = np.repeat(Ff0, n_groups, axis=0)
    free[rows_idx, first] = c0[first]
    Ff[rows_idx, first] = Fp0[first]
    beam_val = np.zeros(n_groups)
    group = rows_idx.copy()
    prior_t = first[:, None]
    prior_c = np.full((n_groups, 1), int(slots[0]["cell"]), dtype=np.int64)

    for k, slot in enumerate(slots[1:], start=1):
        last = k == K
        n_rows = len(group)
        ok = (free <= slot["r"]) & active[None, :]
        none = ~ok.any(axis=1)
        if none.any():
            ok[none] = active[None, :]
        # Score only the technicians that can take this ticket in some row
        # (the others would be -inf everywhere).
        cols = np.flatnonzero(ok.any(axis=0))
        Cn = len(cols)
        scen_rows = group % S
        sl = _rows(slot, scen_rows)
        mk = _within_window_mk(ctx, sl, prior_t, prior_c, n_rows)
        if Cn < N:
            ctx_c, sl_c = _columns(ctx, sl, cols)
            vals, cc, Fp = score_assignment(ctx_c, sl_c, free[:, cols], Ff[:, cols], wT, mk=mk[:, cols])
            ok = ok[:, cols]
        else:
            vals, cc, Fp = score_assignment(ctx, sl, free, Ff, wT, mk=mk)
        total = np.where(ok, beam_val[:, None] + vals, -np.inf)
        if last:
            beam_val = total.max(axis=1)
            break
        width = n_rows // n_groups
        flat = total.reshape(n_groups, width * Cn)
        keep = min(W, flat.shape[1])
        if keep < flat.shape[1]:
            top = np.argpartition(-flat, keep - 1, axis=1)[:, :keep]
        else:
            top = np.broadcast_to(np.arange(flat.shape[1]), (n_groups, keep))
        top_val = np.take_along_axis(flat, top, axis=1)
        parent = (np.arange(n_groups)[:, None] * width + top // Cn).ravel()
        t_col = (top % Cn).ravel()
        tech = cols[t_col]
        r_idx = np.arange(len(parent))
        new_cc = cc[parent, t_col]
        new_Fp = Fp[parent, t_col]
        free = free[parent]
        Ff = Ff[parent]
        free[r_idx, tech] = new_cc
        Ff[r_idx, tech] = new_Fp
        cell_rows = np.broadcast_to(np.asarray(sl["cell"]).reshape(-1), (n_rows,)) \
            if np.ndim(sl["cell"]) else np.full(n_rows, int(sl["cell"]))
        prior_t = np.concatenate([prior_t[parent], tech[:, None]], axis=1)
        prior_c = np.concatenate([prior_c[parent], cell_rows[parent][:, None]], axis=1)
        beam_val = top_val.ravel()
        group = np.repeat(np.arange(n_groups), keep)

    best_per_group = np.full(n_groups, -np.inf)
    np.maximum.at(best_per_group, group, beam_val)
    recourse = best_per_group.reshape(J, S).mean(axis=1)
    value = v0[roots] + recourse
    return int(roots[int(np.argmax(value))]), roots, value


# ---------------------------------------------------------------------------
# Environment reader: human-state snapshot + own bookkeeping
# ---------------------------------------------------------------------------


class _HumanStateReader:
    """Reads the env state into planner arrays and keeps per-episode
    statistics (decision and product rates, ticket-type frequencies,
    assignment shares, predicted busy-until times)."""

    EW_RATE = 1.0 / 500.0        # rate estimators (decisions)
    EW_FREQ_HALF_LIFE = 2000.0   # ticket-type frequencies (decisions)
    EW_SHARE_HALF_LIFE = 5000.0  # assignment shares (decisions)
    SHARE_PRIOR = 2.0            # prior assignments, spread uniformly
    MIN_DECISION_GAP = 0.5       # floor of the mean decision interval (t.u.)

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        weights = v5_weights(params["sigma"])
        self.C = weights["C"]
        self.fk_scale = weights["fleet_knowledge_scale"]
        self.beta = -math.log(GAMMA_PER_TU)
        self._env = None
        self._world = None

    # -- episode lifecycle -------------------------------------------------
    def reset(self, env) -> None:
        self._env = env
        self._world = getattr(env, "sim_env", None)
        sim = env.dispatcher.sim_cfg
        self.travel = float(sim.technicians.travel_time)
        self.fatigue_alpha = float(sim.technicians.fatigue_alpha)
        self.fatigue_mode = (str(sim.technicians.fatigue_model)
                             if bool(sim.repair.fatigue_enabled) else "off")
        self.knowledge_enabled = bool(sim.repair.knowledge_enabled)
        self.min_floor = float(sim.repair.min_repair_fraction)
        self.sensitivity = float(sim.repair.knowledge_sensitivity)
        self.failure_wise = bool(getattr(sim.repair, "failure_wise_knowledge_parameters", False))
        self.dis_mu: dict[str, float] = {}
        self.preempt_fatigue_rate = 0.0
        self.preempt_rate = 0.0
        # Expected absence time per time unit (queue delay): constant part
        # and part proportional to fatigue.
        self.absence_rate0 = 0.0
        self.absence_rate_f = 0.0
        for name, dcfg in sim.disruptions.dis_dict.items():
            mu_d = float(dcfg.duration_mu)
            self.dis_mu[name] = mu_d
            if dcfg.trigger == "fatigue":
                self.absence_rate_f += float(dcfg.fatigue_coefficient or 0.0) * mu_d
            elif dcfg.trigger == "random":
                self.absence_rate0 += float(dcfg.rate or 0.0) * mu_d
            elif dcfg.trigger == "periodic" and float(dcfg.interval or 0.0) > 0.0:
                self.absence_rate0 += mu_d / float(dcfg.interval)
            if not dcfg.preemptive:
                continue
            if dcfg.trigger == "fatigue":
                self.preempt_fatigue_rate += float(dcfg.fatigue_coefficient or 0.0)
            elif dcfg.trigger == "random":
                self.preempt_rate += float(dcfg.rate or 0.0)
        self.mean_dis_mu = float(np.mean(list(self.dis_mu.values()))) if self.dis_mu else 0.0
        self.n = 0
        self.lam = np.zeros(0)
        self.mu = np.zeros(0)
        self.b = np.zeros(0)
        self.shape: tuple[int, ...] | None = None
        self.cell_vol = 1.0
        self.size = 0
        self.bumps = np.zeros((0, 0, 0))
        self.G = np.zeros((0, 0))
        self.Gb = np.zeros((0, 0))
        self.vol = np.zeros(0)
        self.grid_ver: list[tuple[int, int]] = []
        self.decay_mark = None
        # Own bookkeeping per technician: the jobs this agent queued on it
        # (FIFO, head = running or next), the predicted start of the head
        # job, and the absence tracking.
        self.jobs: list[list[tuple]] = []
        self.anchor = np.zeros(0)
        self.abs_total = np.zeros(0)
        self.abs_seen = np.zeros(0)
        self.abs_hold = np.zeros(0)
        self.abs_count = np.zeros(0, dtype=np.int64)
        self.was_absent = np.zeros(0, dtype=bool)
        self.last_counts: list[dict[str, int]] = []
        self.log_pos = 0
        self.seen: weakref.WeakSet = weakref.WeakSet()
        self.keys: dict[str, int] = {}
        self.key_cell = np.zeros(0, dtype=np.int64)
        self.key_mtype: list[str] = []
        self.key_stats = np.zeros((0, 4))  # count, mean base, mean f*, mean alpha*
        self.key_freq = np.zeros(0)
        self.dV = np.zeros((0, 0))
        self.share = np.zeros((0, 0))
        self.decisions = 0
        self.last_now = float(env._sim_time())
        self.start_now = self.last_now
        self.last_products = self._products()
        self.start_products = self.last_products
        self.ew_dt: float | None = None
        self.ew_prod = 0.0
        self.machine_counts: dict[str, int] = {}
        self.n_machines = 1
        self.freq_scale = 1.0
        self.share_scale = 1.0

    def ensure(self, env) -> None:
        if (env is not self._env or getattr(env, "sim_env", None) is not self._world
                or float(env._sim_time()) < self.last_now):
            self.reset(env)

    # -- helpers -------------------------------------------------------------
    def _products(self) -> int:
        sinks = getattr(self._env.dispatcher, "sinks", []) or []
        return sum(int(getattr(s, "completed", 0)) for s in sinks)

    def _grow(self, techs) -> None:
        n_new = len(techs)
        if n_new == self.n:
            return
        new = techs[self.n:]
        if self.shape is None:
            grid0 = techs[0].knowledge_grid
            self.shape = tuple(int(s) for s in grid0._shape)
            self.cell_vol = float(np.prod(grid0._cell_spacing))
            self.size = int(np.prod(self.shape))
        size = self.size
        add = n_new - self.n
        for t in new:
            if tuple(int(s) for s in t.knowledge_grid._shape) != self.shape:
                raise ValueError("sequential baselines need one knowledge-grid shape for the fleet")
        self.lam = np.concatenate([self.lam, [float(t.fatigue_lambda) for t in new]])
        self.mu = np.concatenate([self.mu, [float(t.fatigue_mu) for t in new]])
        self.b = np.concatenate([self.b, [float(t.knowledge_grid.b) for t in new]])
        new_bumps = np.stack([bump_table(float(t.knowledge_grid._propagation_sigma), self.shape)
                              for t in new])
        self.bumps = np.concatenate([self.bumps.reshape(self.n, size, size), new_bumps], axis=0)
        self.G = np.vstack([self.G.reshape(self.n, size), np.zeros((add, size))])
        self.Gb = np.vstack([self.Gb.reshape(self.n, size), np.zeros((add, size))])
        self.vol = np.concatenate([self.vol, np.zeros(add)])
        self.grid_ver += [(-1, -1)] * add
        now = float(self._env._sim_time())
        self.jobs += [[] for _ in new]
        self.anchor = np.concatenate([self.anchor, np.full(add, now)])
        self.abs_total = np.concatenate(
            [self.abs_total, [float(sum(getattr(t, "disruption_time_by_type", {}).values())) for t in new]])
        self.abs_seen = np.concatenate([self.abs_seen, np.full(add, np.nan)])
        self.abs_hold = np.concatenate([self.abs_hold, np.zeros(add)])
        self.abs_count = np.concatenate(
            [self.abs_count, np.asarray([int(getattr(t, "disruption_count", 0)) for t in new], dtype=np.int64)])
        self.was_absent = np.concatenate([self.was_absent, np.zeros(add, dtype=bool)])
        self.last_counts += [dict(getattr(t, "disruption_counts_by_type", {})) for t in new]
        n_keys = len(self.keys)
        self.dV = np.vstack([self.dV.reshape(self.n, n_keys), np.full((add, n_keys), np.nan)])
        self.share = np.vstack([self.share.reshape(self.n, n_keys), np.zeros((add, n_keys))])
        self.n = n_new

    def _refresh_grids(self, techs) -> None:
        # Decay mutates grids in place; completions replace the array and
        # bump the experience count.
        mark = getattr(self._env, "_last_knowledge_decay", None)
        all_stale = mark != self.decay_mark
        self.decay_mark = mark
        stale = []
        for i, t in enumerate(techs):
            g = t.knowledge_grid
            ver = (id(g._grid), int(g._total_num_experiences))
            if all_stale or ver != self.grid_ver[i]:
                self.grid_ver[i] = ver
                stale.append(i)
        if not stale:
            return
        idx = np.asarray(stale)
        self.G[idx] = np.stack([techs[i].knowledge_grid._grid.ravel() for i in stale])
        self.Gb[idx] = np.where(self.G[idx] > 0, self.G[idx], 0.0) ** self.b[idx, None]
        self.vol[idx] = self.cell_vol * self.Gb[idx].sum(axis=1)
        self.dV[idx] = np.nan

    def key_index(self, request, *, update: bool = True) -> int:
        """Register the ticket type of ``request``; update its running
        means once per request."""
        env = self._env
        key = env.failure_key(request)
        k = self.keys.get(key)
        if k is None:
            k = len(self.keys)
            self.keys[key] = k
            techs = env.dispatcher.techs
            grid = techs[0].knowledge_grid
            coords = grid.embedding_to_coords(techs[0].encoder.encode(request))
            self.key_cell = np.append(self.key_cell, int(np.ravel_multi_index(coords, self.shape)))
            self.key_mtype.append(str(getattr(request.machine, "mtype", "unknown")))
            self.key_stats = np.vstack([self.key_stats, np.zeros(4)])
            self.key_freq = np.append(self.key_freq, 0.0)
            self.dV = np.hstack([self.dV, np.full((self.n, 1), np.nan)])
            self.share = np.hstack([self.share, np.zeros((self.n, 1))])
            update = True  # a new type needs its statistics
        if update:
            try:
                fresh = request not in self.seen
                if fresh:
                    self.seen.add(request)
            except TypeError:
                fresh = True
            if fresh or self.key_stats[k, 0] == 0:
                base = float(request.get_repair_time())
                fstar, alpha = self.request_knowledge_params(request)
                st = self.key_stats[k]
                st[0] += 1.0
                st[1:] += (np.array([base, fstar, alpha]) - st[1:]) / st[0]
        return k

    def request_knowledge_params(self, request) -> tuple[float, float]:
        fstar, alpha = self.min_floor, self.sensitivity
        if self.failure_wise:
            getter = getattr(request, "get_knowledge_parameters", None)
            kp = getter() if callable(getter) else None
            if kp is not None:
                if kp[0] is not None:
                    fstar = float(kp[0])
                if kp[1] is not None:
                    alpha = float(kp[1])
        return fstar, alpha

    def m_k(self, cells: np.ndarray, fstar, alpha) -> np.ndarray:
        """Knowledge multiplier ``(N, len(cells))`` at the current grids."""
        if not self.knowledge_enabled:
            return np.ones((self.n, len(cells)))
        kn = self.Gb[:, cells]
        return fstar + (1.0 - fstar) * np.exp(-alpha * kn)

    def dV_for(self, keys: np.ndarray) -> np.ndarray:
        """Knowledge-volume gain ``(N, len(keys))`` of one repair per type."""
        sub = self.dV[:, keys]
        miss = np.isnan(sub)
        if miss.any():
            for i in np.where(miss.any(axis=1))[0]:
                cols = keys[miss[i]]
                new = (self.G[i][None, :] + self.bumps[i][self.key_cell[cols]]) ** self.b[i]
                self.dV[i, cols] = self.cell_vol * new.sum(axis=1) - self.vol[i]
            sub = self.dV[:, keys]
        return sub

    def shares(self) -> np.ndarray:
        """Exponentially weighted assignment counts in natural units."""
        return self.share / self.share_scale

    # -- per-decision snapshot -------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        env = self._env
        techs = env.dispatcher.techs
        self._grow(techs)
        self._refresh_grids(techs)
        now = float(env._sim_time())
        n = self.n
        F0 = np.fromiter((t.fatigue for t in techs), dtype=np.float64, count=n)
        busy = np.fromiter((bool(t.busy) for t in techs), dtype=bool, count=n)
        disr = np.fromiter((bool(getattr(t, "_in_disruption", False)) for t in techs), dtype=bool, count=n)
        active = np.fromiter((not getattr(t, "retired", False) for t in techs), dtype=bool, count=n)

        # Rates: decision interval and products per decision.
        dt = now - self.last_now
        products = self._products()
        dp = products - self.last_products
        self.last_products = products
        self.last_now = now
        self.decisions += 1
        if self.decisions > 1:
            if self.ew_dt is None or self.decisions <= 1.0 / self.EW_RATE:
                self.ew_dt = (now - self.start_now) / (self.decisions - 1)
                self.ew_prod = (products - self.start_products) / (self.decisions - 1)
            else:
                self.ew_dt += self.EW_RATE * (dt - self.ew_dt)
                self.ew_prod += self.EW_RATE * (dp - self.ew_prod)
        if self.decisions % 256 == 1:
            counts: dict[str, int] = {}
            try:
                machines = env._factory_machines()
            except Exception:  # noqa: BLE001
                machines = []
            for m in machines:
                if getattr(m, "retired", False):
                    continue
                mt = str(getattr(m, "mtype", "unknown"))
                counts[mt] = counts.get(mt, 0) + 1
            self.machine_counts = counts
            self.n_machines = max(1, sum(counts.values()))
        if self.ew_dt is not None and self.decisions > 2:
            rho = 1.0 / max(self.ew_dt, self.MIN_DECISION_GAP)
            pi = max(self.ew_prod, 0.0) * rho
        else:
            rho = 0.002 * self.n_machines
            pi = 0.0

        free, Ff = self._free_times(techs, now, F0, busy, disr)

        n_act = max(1, int(active.sum()))
        Fa = F0[active]
        sd = math.sqrt(float(Fa.var()) + 1e-4) if len(Fa) else 1.0
        mean_F = float(Fa.mean()) if len(Fa) else 0.0
        std_grad = np.where(active, (F0 - mean_F) / (n_act * sd), 0.0)
        tau_end = max(0.0, float(env._episode_max_sim_time) - now)
        C = self.C
        return {
            "now": now, "n": n, "F0": F0, "busy": busy, "disr": disr, "active": active,
            "free": free - now, "Ff": Ff, "rho": rho, "pi": pi, "n_active": n_act,
            "std_grad": std_grad, "tau_end": tau_end, "beta": self.beta, "C": C,
            "mu": self.mu, "lam": self.lam, "b": self.b, "bumps": self.bumps,
            "knowledge_enabled": self.knowledge_enabled, "travel": self.travel,
            "fatigue_alpha": self.fatigue_alpha, "fatigue_mode": self.fatigue_mode,
            "preempt_fatigue_rate": self.preempt_fatigue_rate, "preempt_rate": self.preempt_rate,
            "C_term_products": C["terminal_finished_products"] * math.exp(-self.beta * tau_end),
            "C_term_knowledge": (C["terminal_fleet_knowledge"] * math.exp(-self.beta * tau_end)
                                 / self.fk_scale),
            "window_end": 0.0, "terminal_horizon": 0.0, "fatigue_future": 0.0,
            "n_scenarios": 1,
        }

    def kappa(self, mtype: str, rho: float, pi: float) -> tuple[float, float]:
        """Downtime cost per time unit and products lost per time unit for a
        machine of type ``mtype``."""
        m_type = max(1, self.machine_counts.get(mtype, 1))
        x = pi / rho if rho > 0 else 0.0
        phi = math.exp(-x) * x
        kap = (self.C["fleet_availability"] * rho / self.n_machines
               + self.C["throughput_delta"] * rho * phi / m_type)
        return kap, pi / m_type

    # -- own bookkeeping -------------------------------------------------------
    def _free_times(self, techs, now: float, F0: np.ndarray, busy: np.ndarray,
                    disr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicted time at which each technician is free for a new job,
        and its fatigue at that time.

        Own bookkeeping, re-anchored on observations at every decision:

        * completions (``env.repair_log()``) pop the head job of the
          technician (the SimPy resource serves repair jobs first in, first
          out) and set the start of the next job to the completion time;
        * a job whose ticket left the technician (a preemptive disruption
          re-queued it; the dispatcher clears ``chosen_technician_id``) is
          dropped;
        * absences: the elapsed hold time of an absence (from
          ``disruption_time_by_type``, exact once it ends) delays the queued
          jobs; an ongoing absence adds its expected residual (mean duration
          of the detected type minus the time seen, at least 20 % of it);
        * the queued work is stretched by the expected absence time per
          time unit (``sum rate * mean duration`` over the disruption
          types, the fatigue-driven rate at the current fatigue).

        Fatigue along a job chain: frozen during a job, Jaber accumulation
        on the floored base time at the completion, no recovery between
        consecutive jobs, exponential recovery while idle or absent.
        """
        n = self.n
        completed = set()
        log = self._env.repair_log()
        for rec in log[self.log_pos:]:
            i = int(rec["tech"])
            if i < n and self.jobs[i]:
                self.jobs[i].pop(0)
                self.anchor[i] = float(rec["completed_at"])
                completed.add(i)
        self.log_pos = len(log)

        free = np.full(n, now)
        Ff = F0.copy()
        travel = self.travel
        mode, a_f = self.fatigue_mode, self.fatigue_alpha
        for i in range(n):
            t = techs[i]
            q = self.jobs[i]
            if q:
                tid = t.id
                kept = [j for j in q if getattr(j[0], "chosen_technician_id", tid) == tid]
                if len(kept) != len(q):
                    if not kept or kept[0] is not q[0]:
                        self.anchor[i] = now  # the running job was preempted
                    self.jobs[i] = q = kept
            cnt = int(getattr(t, "disruption_count", 0))
            if cnt != self.abs_count[i] or self.was_absent[i]:
                tot = float(sum(getattr(t, "disruption_time_by_type", {}).values()))
                d_tot = tot - self.abs_total[i]
                self.abs_total[i] = tot
                if d_tot > 0.0 and q and i not in completed:
                    self.anchor[i] += d_tot
            if disr[i]:
                if cnt != self.abs_count[i] or np.isnan(self.abs_seen[i]):
                    cur = dict(getattr(t, "disruption_counts_by_type", {}))
                    prev = self.last_counts[i]
                    inc = [name for name, v in cur.items() if v > prev.get(name, 0)]
                    self.abs_hold[i] = self.dis_mu.get(inc[-1], self.mean_dis_mu) if inc else self.mean_dis_mu
                    self.abs_seen[i] = now
                    self.last_counts[i] = cur
                hold = self.abs_hold[i]
                back = now + max(0.2 * hold, hold - (now - self.abs_seen[i]))
            else:
                if cnt != self.abs_count[i]:
                    self.last_counts[i] = dict(getattr(t, "disruption_counts_by_type", {}))
                self.abs_seen[i] = np.nan
                back = now
            self.abs_count[i] = cnt
            self.was_absent[i] = bool(disr[i])

            mu = float(self.mu[i])
            if not q:
                self.anchor[i] = now
                if disr[i]:
                    free[i] = back
                    Ff[i] = F0[i] * math.exp(-mu * (back - now))
                elif busy[i]:  # a job this agent did not assign
                    typical = travel + (float(np.mean(self.key_stats[:, 1])) if len(self.keys) else travel)
                    free[i] = now + 0.5 * typical
                continue
            lam = float(self.lam[i])
            F = float(F0[i])
            if busy[i]:
                # Head job running since the anchor, fatigue frozen at F0.
                _, base, fb, mk = q[0]
                rep_t = base * mk * float(_fatigue_multiplier(mode, a_f, F))
                end = min(float(self.anchor[i]), now) + travel + rep_t
                if end < now + 1.0:  # overrun: the model cannot say how long
                    end = now + max(1.0, 0.25 * rep_t)
                F = F + (1.0 - F) * (1.0 - math.exp(-lam * fb))
                tt = end
                work = end - now
                rest = q[1:]
            else:
                tt = max(float(self.anchor[i]), back)
                F = F * math.exp(-mu * (tt - now))
                work = 0.0
                rest = q
            for _, base, fb, mk in rest:
                dur = travel + base * mk * float(_fatigue_multiplier(mode, a_f, F))
                tt += dur
                work += dur
                F = F + (1.0 - F) * (1.0 - math.exp(-lam * fb))
            ovh = self.absence_rate0 + self.absence_rate_f * 0.5 * (float(F0[i]) + F)
            free[i] = tt + work * ovh
            Ff[i] = min(1.0, F)
        return free, Ff

    def commit(self, snap: dict[str, Any], request, action: int, key: int,
               mk_i: float, base: float) -> None:
        """Record the chosen assignment: the queued job, assignment share,
        type frequency."""
        i = int(action)
        now = snap["now"]
        if not self.jobs[i]:
            if snap["disr"][i] and not np.isnan(self.abs_seen[i]):
                self.anchor[i] = self.abs_seen[i]  # start of the absence (as seen)
            else:
                self.anchor[i] = now
        self.jobs[i].append((request, float(base), float(math.floor(base)), float(mk_i)))
        start = now + max(0.0, float(snap["free"][i]))
        Fs = float(snap["Ff"][i])
        mf = float(_fatigue_multiplier(self.fatigue_mode, self.fatigue_alpha, Fs))
        self.last_prediction = (i, start, base * float(mk_i) * mf, Fs)
        self.share_scale /= 0.5 ** (1.0 / self.EW_SHARE_HALF_LIFE)
        self.share[i, key] += self.share_scale
        self.freq_scale /= 0.5 ** (1.0 / self.EW_FREQ_HALF_LIFE)
        self.key_freq[key] += self.freq_scale
        if self.share_scale > 1e100 or self.freq_scale > 1e100:
            self.share /= self.share_scale
            self.key_freq /= self.freq_scale
            self.share_scale = 1.0
            self.freq_scale = 1.0


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class _ModelBasedDispatcher(Agent):
    """Shared state reading, slot construction and search of the two
    baselines."""

    def __init__(self, n_actions: int, *, name: str, params: dict[str, Any] | None = None,
                 horizon_k: int = 0, terminal_weight: float = 0.0) -> None:
        super().__init__(n_actions, name=name)
        self.params = load_params() if params is None else _merged(params)
        self.horizon_k = int(horizon_k)
        self.terminal_weight = float(terminal_weight)
        self._reader = _HumanStateReader(self.params)
        self._latin: np.ndarray | None = None
        self._times = array("d")

    def on_episode_start(self) -> None:
        self._reader._env = None  # re-read everything at the first decision
        self._times = array("d")

    def on_episode_end(self, episode_reward: float) -> None:
        if len(self._times):
            t = np.frombuffer(self._times, dtype=np.float64) * 1e3
            print(f"[seqbase] {self.name} K={self.horizon_k} w={self.terminal_weight:g}: "
                  f"{len(t)} decisions, planner ms/decision median {np.median(t):.3f} "
                  f"mean {t.mean():.3f} p90 {np.percentile(t, 90):.3f}", flush=True)

    def planner_ms(self) -> np.ndarray:
        """Planner wall time per decision of the current episode (ms)."""
        return np.frombuffer(self._times, dtype=np.float64) * 1e3

    def select_action(self, obs: dict[str, Any], *, deterministic: bool = False) -> int:
        t0 = time.perf_counter()
        action = self._decide(obs)
        self._times.append(time.perf_counter() - t0)
        return action

    # -- slot builders ---------------------------------------------------------
    def _ticket_slot(self, snap, request, key) -> dict[str, Any]:
        rd = self._reader
        fstar, alpha = rd.request_knowledge_params(request)
        base = float(request.get_repair_time())
        cell = int(rd.key_cell[key])
        mk = rd.m_k(np.asarray([cell]), fstar, alpha)[:, 0]
        kap, ppt = rd.kappa(rd.key_mtype[key], snap["rho"], snap["pi"])
        return {
            "kind": "fixed", "r": 0.0, "mk": mk, "base": base, "floor_base": float(math.floor(base)),
            "dV": rd.dV_for(np.asarray([key]))[:, 0], "kappa": kap, "prod_per_tu": ppt,
            "cell": cell, "Gc": rd.G[:, cell].copy(), "fstar": fstar, "alpha": alpha, "key": key,
        }

    def _decide(self, obs: dict[str, Any]) -> int:
        env = self._env
        avail = _available(obs, self.n_actions)
        request = getattr(env, "current_request", None) if env is not None else None
        if env is None or request is None:
            return int(avail[0])
        rd = self._reader
        rd.ensure(env)
        snap = rd.snapshot()
        cand = avail[avail < rd.n]
        if len(cand):
            cand = cand[snap["active"][cand]]
        if len(cand) == 0:
            return int(avail[0])
        key = rd.key_index(request)
        slot0 = self._ticket_slot(snap, request, key)
        slot0["F_r"] = snap["F0"]
        slot0["busy_r"] = snap["busy"].astype(np.float64)
        K = self.horizon_k
        wT = self.terminal_weight
        ctx = snap
        slots = [slot0]
        if K > 0 or wT:
            ctx["terminal_horizon"] = float(self.params["terminal_horizon"])
            rho = snap["rho"]
            if K > 0:
                queue = env._queue()
                for qreq in list(getattr(queue, "items", queue) or [])[:K]:
                    qs = self._ticket_slot(snap, qreq, rd.key_index(qreq, update=False))
                    slots.append(qs)
            n_anticipated = K - (len(slots) - 1)
            # Scenarios only differ in anticipated tickets.
            ctx["n_scenarios"] = int(self.params["scenarios"]) if n_anticipated > 0 else 1
            ctx["window_end"] = (n_anticipated + 0.5) / rho if (K > 0 and rho > 0) else 0.0
            scen_keys = (self._scenario_keys(n_anticipated) if n_anticipated > 0
                         else np.zeros((ctx["n_scenarios"], 0), dtype=np.int64))
            plan_keys = np.asarray([s["key"] for s in slots], dtype=np.int64)
            all_keys = np.unique(np.concatenate([plan_keys, scen_keys.ravel()]))
            col = {int(k): j for j, k in enumerate(all_keys)}
            if wT:
                top, p_top = self._top_keys(key)
                VK, ctx["fatigue_future"] = self._terminal_coefficients(snap, all_keys, top, p_top)
                for s in slots:
                    s["VK"] = VK[:, col[int(s["key"])]]
            if n_anticipated > 0:
                cells = rd.key_cell[all_keys]
                st = rd.key_stats[all_keys]
                mk_u = rd.m_k(cells, st[:, 2], st[:, 3])  # (N, U)
                dV_u = rd.dV_for(all_keys)
                G_u = rd.G[:, cells]
                kp = [rd.kappa(rd.key_mtype[int(k)], rho, snap["pi"]) for k in all_keys]
                kap_u = np.asarray([x[0] for x in kp])[:, None]
                ppt_u = np.asarray([x[1] for x in kp])[:, None]
                for a in range(n_anticipated):
                    u = np.asarray([col[int(k)] for k in scen_keys[:, a]])
                    slot = {
                        "kind": "dist", "r": (a + 1) / rho if rho > 0 else 0.0,
                        "mk": mk_u[:, u].T, "base": st[u, 1][:, None],
                        "floor_base": np.floor(st[u, 1])[:, None],
                        "dV": dV_u[:, u].T, "kappa": kap_u[u], "prod_per_tu": ppt_u[u],
                        "cell": cells[u], "Gc": G_u[:, u].T,
                        "fstar": st[u, 2][:, None], "alpha": st[u, 3][:, None],
                    }
                    if wT:
                        slot["VK"] = VK[:, u].T
                    slots.append(slot)
        params = {**self.params, "terminal_weight": wT}
        action, _, _ = plan_first_action(ctx, slots, cand, params)
        rd.commit(snap, request, action, key, float(slot0["mk"][action]), slot0["base"])
        return int(action)

    def _top_keys(self, current_key: int) -> tuple[np.ndarray, np.ndarray]:
        rd = self._reader
        freq = rd.key_freq.copy()
        if freq.sum() <= 0:
            freq[current_key] = 1.0
        m = min(int(self.params["top_keys"]), len(freq))
        top = np.argsort(-freq, kind="stable")[:m]
        return top, freq[top] / freq.sum()

    def _scenario_keys(self, n_slots: int) -> np.ndarray:
        """``(S, n_slots)`` ticket types by stratified quantiles of the
        observed type frequencies (fixed Latin-hypercube permutations)."""
        rd = self._reader
        S = int(self.params["scenarios"])
        if self._latin is None or self._latin.shape[1] < n_slots or self._latin.shape[0] != S:
            rng = np.random.default_rng(20260916)
            self._latin = np.stack([rng.permutation(S) for _ in range(max(n_slots, 1))], axis=1)
        freq = rd.key_freq
        order = np.argsort(-freq, kind="stable")
        cdf = np.cumsum(freq[order])
        if len(cdf) == 0 or cdf[-1] <= 0:
            return np.zeros((S, n_slots), dtype=np.int64)
        u = (self._latin[:, :n_slots] + 0.5) / S * cdf[-1]
        pos = np.minimum(np.searchsorted(cdf, u, side="right"), len(order) - 1)
        return order[pos]

    def _terminal_coefficients(self, snap, keys: np.ndarray, top: np.ndarray, p: np.ndarray):
        """``VK[i, k]``: value per unit of discounted time after the window
        of the knowledge bump of one type-``k`` repair by technician ``i``;
        and the per-technician fatigue future-cost coefficient."""
        rd = self._reader
        rho, pi = snap["rho"], snap["pi"]
        n_act = snap["n_active"]
        active = snap["active"]
        C = rd.C
        cells_top = rd.key_cell[top]
        st_top = rd.key_stats[top]
        f_top, a_top, B_top = st_top[:, 2], st_top[:, 3], st_top[:, 1]
        mk_top = rd.m_k(cells_top, f_top, a_top)  # (N, M)
        kap_top = np.asarray([rd.kappa(rd.key_mtype[int(k)], rho, pi)[0] for k in top])
        a0 = rd.SHARE_PRIOR
        shares = rd.shares()
        sh = shares[:, top] * active[:, None]
        share_ic = (sh + a0 / n_act * active[:, None]) / (sh.sum(axis=0, keepdims=True) + a0)
        F0 = snap["F0"]
        mf0 = np.asarray(_fatigue_multiplier(snap["fatigue_mode"], snap["fatigue_alpha"], F0)) * np.ones(rd.n)
        slope0 = np.asarray(_fatigue_slope(snap["fatigue_mode"], snap["fatigue_alpha"], F0)) * np.ones(rd.n)
        # (N, M): value rate of one unit of m_k decrease on type c'.
        per = rho * p[None, :] * share_ic * (C["repair_quality"] + kap_top[None, :] * B_top[None, :] * mf0[:, None])
        VK = np.zeros((rd.n, len(keys)))
        act = np.where(active)[0]
        if rd.knowledge_enabled and len(act):
            cells_k = rd.key_cell[keys]
            bump = rd.bumps[act[:, None, None], cells_k[None, :, None], cells_top[None, None, :]]  # (A, K, M)
            n_new = rd.G[act][:, cells_top][:, None, :] + bump
            mk_new = f_top + (1.0 - f_top) * np.exp(-a_top * n_new ** rd.b[act][:, None, None])
            VK[act] = ((mk_top[act][:, None, :] - mk_new) * per[act][:, None, :]).sum(axis=2)
        tot = shares.sum(axis=1) * active
        share_i = (tot + a0 / n_act * active) / (tot.sum() + a0)
        downtime_per_repair = (p[None, :] * B_top[None, :] * mk_top * kap_top[None, :]).sum(axis=1)
        fatigue_future = rho * share_i * (C["fatigue_cost"] + slope0 * downtime_per_repair)
        return VK, fatigue_future


class GreedyTrainingRewardAgent(_ModelBasedDispatcher):
    """One-step greedy on HTT-RL's human-centric v5 TRAINING reward.

    Picks the available technician whose assignment of the current ticket
    has the highest predicted v5 score: the quality, fatigue and busy terms
    paid at the decision; the machine downtime through availability and
    throughput; the potential-based knowledge credit and the fatigue-balance
    effect of this repair; the episode-end terms.  The prediction uses the
    human-state model of this module and the v5 coefficients divided by the
    calibrated normalisation scales.  No look-ahead over other tickets and
    no terminal value: it is :class:`RollingHorizonMPCAgent` with
    ``horizon_k=0`` and ``terminal_weight=0``.  Deterministic (ties go to
    the lowest technician index).

    Not the same baseline as
    :class:`~agents.baselines.heuristics.GreedyRewardAgent`.  That agent
    probes ``env.assignment_reward_estimates()``: the reward stack of the
    EVALUATION scenario config (busy, fatigue, quality, throughput and the
    floored knowledge increment; no workload balance, no fleet
    availability, no potential-based shaping), computed at the decision
    instant.  There only the busy, fatigue and quality terms depend on the
    action, and the benchmark harness freezes the reward normaliser before
    any update, so GreedyReward maximises ``(1 - m_k) - F - busy`` with
    unit weights.  This agent uses the v5 TRAINING stack with its
    coefficients and calibrated scales, and it also predicts the terms
    that the assignment moves at later decisions: machine downtime
    (availability and throughput), the knowledge credit at the completion,
    the fatigue-balance path and the episode-end terms.
    """

    def __init__(self, n_actions: int, *, params: dict[str, Any] | None = None) -> None:
        super().__init__(n_actions, name="GreedyTrainReward", params=params,
                         horizon_k=0, terminal_weight=0.0)


class RollingHorizonMPCAgent(_ModelBasedDispatcher):
    """Rolling-horizon look-ahead dispatcher (model predictive control).

    Plans the current ticket and ``horizon_k`` further tickets (queued
    tickets first, then anticipated ones) under the human-state model of
    this module, scores each plan with the v5 training objective plus a
    terminal value of the post-window knowledge and fatigue state, executes
    the first assignment and re-plans at the next decision.  See the module
    docstring for the model, the objective, the search and every
    approximation.  ``horizon_k`` and ``terminal_weight`` default to
    ``run_configs/agents/seqbase_mpc.json`` (with the other planner
    parameters and the calibrated ``sigma`` scales).  Deterministic.
    """

    def __init__(self, n_actions: int, *, params: dict[str, Any] | None = None,
                 horizon_k: int | None = None, terminal_weight: float | None = None) -> None:
        merged = load_params() if params is None else _merged(params)
        k = int(merged["horizon_k"] if horizon_k is None else horizon_k)
        w = float(merged["terminal_weight"] if terminal_weight is None else terminal_weight)
        super().__init__(n_actions, name="RollingMPC", params=merged, horizon_k=k, terminal_weight=w)

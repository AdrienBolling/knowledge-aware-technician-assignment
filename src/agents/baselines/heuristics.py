"""Decision-rule baseline agents for technician dispatching.

These agents implement fixed heuristic policies that require no training.
They serve as lower bounds for evaluating learned policies.

All heuristics are *availability-aware*: they restrict their choice to the
technicians flagged valid by ``obs["action_mask"]`` whenever the
environment exposes it.  The mask marks technicians that are neither busy
nor absorbed by a disruption (injury / vacation / exhaustion) --- the same
information a masked RL policy consumes --- so rule-based and learned
agents compete on an equal observational footing.  Without the mask, the
heuristics fall back to the raw ``technician_busy`` flags, which do NOT
reflect disruptions: a disruption-blind rule will happily queue tickets
behind an absent technician, which collapses at scale (see the results
section of the paper).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.base import Agent


def _available(obs: dict[str, Any], n_actions: int) -> np.ndarray:
    """Indices of currently-assignable technicians.

    Preference order: ``action_mask`` (busy + disruption aware) >
    ``technician_busy`` (busy only) > all technicians.  The environment's
    mask falls back to all-ones when the whole fleet is unavailable, so
    the returned array is never empty when the mask is present.
    """
    mask = obs.get("action_mask")
    if mask is not None:
        idx = np.where(np.asarray(mask) == 1)[0]
        if len(idx):
            return idx
    busy = obs.get("technician_busy")
    if busy is not None:
        idx = np.where(np.asarray(busy) == 0)[0]
        if len(idx):
            return idx
    return np.arange(n_actions)


class RandomAgent(Agent):
    """Uniform random selection among available technicians."""

    def __init__(self, n_actions: int) -> None:
        super().__init__(n_actions, name="Random")

    def select_action(self, obs: dict[str, Any], *, deterministic: bool = False) -> int:
        return int(np.random.choice(_available(obs, self.n_actions)))


class RoundRobinAgent(Agent):
    """Cycle through technicians in order, skipping unavailable ones."""

    def __init__(self, n_actions: int) -> None:
        super().__init__(n_actions, name="RoundRobin")
        self._next = 0

    def select_action(self, obs: dict[str, Any], *, deterministic: bool = False) -> int:
        avail = set(_available(obs, self.n_actions).tolist())
        for _ in range(self.n_actions):
            action = self._next
            self._next = (self._next + 1) % self.n_actions
            if action in avail:
                return action
        return int(self._next)  # unreachable: avail is never empty

    def on_episode_start(self) -> None:
        self._next = 0


class LeastBusyAgent(Agent):
    """Pick an available technician.  Random tiebreak unless deterministic.

    With the action mask exposed, "available" means neither busy nor
    disrupted; the deterministic variant picks the lowest index.
    """

    def __init__(self, n_actions: int) -> None:
        super().__init__(n_actions, name="LeastBusy")

    def select_action(self, obs: dict[str, Any], *, deterministic: bool = False) -> int:
        avail = _available(obs, self.n_actions)
        return int(avail[0]) if deterministic else int(np.random.choice(avail))


class LeastFatiguedAgent(Agent):
    """Pick the available technician with the lowest fatigue.

    Falls back to the global fatigue argmin when fatigue info is
    unavailable for the valid set (never happens with the standard
    structured observation).
    """

    def __init__(self, n_actions: int) -> None:
        super().__init__(n_actions, name="LeastFatigued")

    def select_action(self, obs: dict[str, Any], *, deterministic: bool = False) -> int:
        avail = _available(obs, self.n_actions)
        fatigue = obs.get("technician_fatigue")
        if fatigue is None:
            return int(avail[0]) if deterministic else int(np.random.choice(avail))
        fatigue = np.asarray(fatigue, dtype=np.float32)
        return int(avail[np.argmin(fatigue[avail])])


class ShortestQueueAgent(Agent):
    """Least-fatigued available technician; proxy for finishing soonest.

    Same information as :class:`LeastFatiguedAgent` but keeps its own
    identity so ablations against historical results remain possible;
    when every technician is unavailable the mask collapses to all-ones
    and the choice degrades to the global fatigue argmin.
    """

    def __init__(self, n_actions: int) -> None:
        super().__init__(n_actions, name="ShortestQueue")

    def select_action(self, obs: dict[str, Any], *, deterministic: bool = False) -> int:
        avail = _available(obs, self.n_actions)
        fatigue = obs.get("technician_fatigue")
        if fatigue is None:
            return int(avail[0]) if deterministic else int(np.random.choice(avail))
        fatigue = np.asarray(fatigue, dtype=np.float32)
        return int(avail[np.argmin(fatigue[avail])])


class ShortestProcessingTimeAgent(Agent):
    """Assign the available technician with the shortest expected repair.

    The classic *shortest-processing-time* (SPT) dispatching rule, adapted
    to a skill-heterogeneous fleet: for the open ticket, pick the available
    technician whose estimated repair time is smallest given their current
    knowledge.  This is a purely *exploitative* skill-greedy policy --- it
    always sends the fastest qualified technician and never develops weaker
    ones --- which makes it the natural foil for an upskilling agent: any
    long-horizon edge over SPT is the payoff of investing in fleet
    knowledge rather than exploiting it.

    Reads ``technician_expected_repair`` from the observation (exposed by
    ``include_repair_estimate_in_observation``); if absent, it queries the
    attached environment's :meth:`expected_repair_times`.  Falls back to a
    least-busy pick when no repair estimate is available at all.
    """

    def __init__(self, n_actions: int) -> None:
        super().__init__(n_actions, name="ShortestProcessingTime")

    def _repair_estimates(self, obs: dict[str, Any]) -> np.ndarray | None:
        eta = obs.get("technician_expected_repair")
        if eta is None and self._env is not None:
            try:
                eta = self._env.expected_repair_times()
            except Exception:
                eta = None
        return None if eta is None else np.asarray(eta, dtype=np.float64)

    def select_action(self, obs: dict[str, Any], *, deterministic: bool = False) -> int:
        avail = _available(obs, self.n_actions)
        eta = self._repair_estimates(obs)
        if eta is None:
            return int(avail[0]) if deterministic else int(np.random.choice(avail))
        # argmin of expected repair time restricted to available techs.
        return int(avail[np.argmin(eta[avail])])


class OptimalAssignmentAgent(Agent):
    """Myopic optimal (Hungarian) assignment over the open ticket set.

    Solves the linear-sum assignment problem that minimises the total
    expected repair time of all currently-open tickets (the one to dispatch
    now plus the pending queue) against the available technicians, then
    commits the technician matched to the current ticket.  Unlike
    shortest-processing-time it avoids the greedy trap of handing the single
    fastest technician to a ticket that a scarcer specialist should take ---
    the classic exact-assignment baseline of the HRAP literature applied
    online, one decision at a time.

    Requires the attached environment (for the batch cost matrix via
    :meth:`assignment_cost_matrix`).  Degrades gracefully to SPT for the
    current ticket if the matrix or an optimal match is unavailable.
    """

    def __init__(self, n_actions: int) -> None:
        super().__init__(n_actions, name="OptimalAssignment")

    def select_action(self, obs: dict[str, Any], *, deterministic: bool = False) -> int:
        avail = _available(obs, self.n_actions)
        matrix = None
        if self._env is not None:
            try:
                matrix = self._env.assignment_cost_matrix()
            except Exception:
                matrix = None
        if matrix is None:
            return self._spt_fallback(obs, avail, deterministic)

        cost, _tickets = matrix
        cost = np.asarray(cost, dtype=np.float64)
        if cost.ndim != 2 or cost.shape[0] == 0:
            return self._spt_fallback(obs, avail, deterministic)

        # linear_sum_assignment cannot take inf; replace unavailable pairs
        # with a big-M dominating any real assignment so they are chosen
        # only when nothing else can cover a ticket.
        finite = cost[np.isfinite(cost)]
        big_m = (float(finite.max()) + 1.0) * (cost.size + 1.0) if finite.size else 1.0
        solvable = np.where(np.isfinite(cost), cost, big_m)

        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(solvable)
        # Technician matched to the current ticket (row 0).
        match = {int(r): int(c) for r, c in zip(rows, cols)}
        chosen = match.get(0)
        # Accept the match only if it is a genuinely available technician
        # (finite original cost); otherwise fall back to SPT on row 0.
        if chosen is not None and np.isfinite(cost[0, chosen]):
            return chosen
        return self._spt_fallback(obs, avail, deterministic, row0=cost[0])

    def _spt_fallback(
        self,
        obs: dict[str, Any],
        avail: np.ndarray,
        deterministic: bool,
        row0: np.ndarray | None = None,
    ) -> int:
        if row0 is None:
            eta = obs.get("technician_expected_repair")
            if eta is None and self._env is not None:
                try:
                    eta = self._env.expected_repair_times()
                except Exception:
                    eta = None
            row0 = None if eta is None else np.asarray(eta, dtype=np.float64)
        if row0 is None:
            return int(avail[0]) if deterministic else int(np.random.choice(avail))
        return int(avail[np.argmin(row0[avail])])

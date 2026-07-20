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


class TopsisAgent(Agent):
    """Multi-criteria (TOPSIS) dispatching rule.

    Ranks the available technicians by their closeness to the ideal
    solution across three cost criteria --- expected repair time (skill),
    fatigue, and workload (assignment count so far) --- and picks the
    closest.  This is the method of Ferjani et al. (2017), the nearest
    online multi-skilled-with-fatigue assignment work in the survey, whose
    TOPSIS rule trades processing time against congestion; here it is
    extended with an explicit fatigue criterion, making it the strongest
    "considers-everything" non-learned competitor --- it weighs skill *and*
    fatigue *and* balance at once, so it patches shortest-processing-time's
    fatigue blindness.

    Criterion weights default to favouring repair time (skill dominates,
    fatigue and balance temper it); pass ``weights`` to retune.
    """

    def __init__(
        self, n_actions: int, *, weights: tuple[float, float, float] = (0.5, 0.3, 0.2)
    ) -> None:
        super().__init__(n_actions, name="TOPSIS")
        self.weights = np.asarray(weights, dtype=np.float64)

    def _criteria(self, obs: dict[str, Any], avail: np.ndarray) -> np.ndarray | None:
        """Decision matrix (n_avail x 3) of cost criteria, or None."""
        if self._env is None:
            return None
        try:
            repair = np.asarray(self._env.expected_repair_times(), dtype=np.float64)
            counts = np.asarray(self._env.assignment_counts(), dtype=np.float64)
        except Exception:
            return None
        fatigue = obs.get("technician_fatigue")
        if fatigue is None:
            fatigue = np.zeros(self.n_actions, dtype=np.float64)
        fatigue = np.asarray(fatigue, dtype=np.float64)
        # Non-finite repair estimates (no ticket) make the rule ill-defined.
        if not np.all(np.isfinite(repair[avail])):
            return None
        return np.column_stack([repair[avail], fatigue[avail], counts[avail]])

    def select_action(self, obs: dict[str, Any], *, deterministic: bool = False) -> int:
        avail = _available(obs, self.n_actions)
        if len(avail) == 1:
            return int(avail[0])
        m = self._criteria(obs, avail)
        if m is None:
            return int(avail[0]) if deterministic else int(np.random.choice(avail))
        # Vector-normalise each criterion, weight, then rank by relative
        # closeness to the ideal-best (all criteria are costs, so the
        # ideal-best is the column min and the ideal-worst the column max).
        norms = np.sqrt((m ** 2).sum(axis=0))
        norms[norms == 0.0] = 1.0
        v = (m / norms) * self.weights
        best, worst = v.min(axis=0), v.max(axis=0)
        d_best = np.sqrt(((v - best) ** 2).sum(axis=1))
        d_worst = np.sqrt(((v - worst) ** 2).sum(axis=1))
        denom = d_best + d_worst
        denom[denom == 0.0] = 1.0
        closeness = d_worst / denom  # higher == closer to ideal-best
        return int(avail[int(np.argmax(closeness))])


class GreedyRewardAgent(Agent):
    """Myopically maximise the environment's per-assignment reward.

    Picks the available technician yielding the highest *immediate* reward
    under the env's configured (human-centric) reward stack --- the greedy,
    zero-lookahead version of the learned policy's own objective.  It
    isolates the contribution of long-horizon planning from the reward
    design: if this matches the RL agent, the value is in the reward
    shaping; if the RL agent beats it, the value is in temporal credit
    assignment.  Requires the attached environment
    (:meth:`assignment_reward_estimates`).
    """

    def __init__(self, n_actions: int) -> None:
        super().__init__(n_actions, name="GreedyReward")

    def select_action(self, obs: dict[str, Any], *, deterministic: bool = False) -> int:
        avail = _available(obs, self.n_actions)
        rewards = None
        if self._env is not None:
            try:
                rewards = np.asarray(
                    self._env.assignment_reward_estimates(), dtype=np.float64
                )
            except Exception:
                rewards = None
        if rewards is None or not np.all(np.isfinite(rewards[avail])):
            return int(avail[0]) if deterministic else int(np.random.choice(avail))
        return int(avail[int(np.argmax(rewards[avail]))])


class TrainWeakestAgent(Agent):
    """Assign the ticket to the least-skilled available technician.

    The deliberate-upskilling foil: always develops the weakest qualified
    technician (lowest current skill match for the ticket), the opposite of
    shortest-processing-time.  Naively always investing should hurt
    short-term throughput and repair speed; testing the learned policy
    against it isolates whether the policy's value is learning *when* to
    invest versus exploit, rather than simply always developing the fleet.
    Reads :meth:`skill_match_scores` from the attached environment.
    """

    def __init__(self, n_actions: int) -> None:
        super().__init__(n_actions, name="TrainWeakest")

    def select_action(self, obs: dict[str, Any], *, deterministic: bool = False) -> int:
        avail = _available(obs, self.n_actions)
        scores = None
        if self._env is not None:
            try:
                scores = np.asarray(self._env.skill_match_scores(), dtype=np.float64)
            except Exception:
                scores = None
        if scores is None:
            return int(avail[0]) if deterministic else int(np.random.choice(avail))
        # Lowest skill match == most room to learn on this ticket.
        return int(avail[int(np.argmin(scores[avail]))])

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

"""Decision-rule baseline agents for technician dispatching.

These agents implement fixed heuristic policies that require no training.
They serve as lower bounds for evaluating learned policies.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.base import Agent


class RandomAgent(Agent):
    """Uniform random technician selection."""

    def __init__(self, n_actions: int) -> None:
        super().__init__(n_actions, name="Random")

    def select_action(self, obs: dict[str, Any], *, deterministic: bool = False) -> int:
        return int(np.random.randint(self.n_actions))


class RoundRobinAgent(Agent):
    """Cycle through technicians in order."""

    def __init__(self, n_actions: int) -> None:
        super().__init__(n_actions, name="RoundRobin")
        self._next = 0

    def select_action(self, obs: dict[str, Any], *, deterministic: bool = False) -> int:
        action = self._next
        self._next = (self._next + 1) % self.n_actions
        return action

    def on_episode_start(self) -> None:
        self._next = 0


class LeastBusyAgent(Agent):
    """Pick a non-busy technician.  Random tiebreak among free techs.

    Works with both structured observations (``technician_busy`` key)
    and token observations (falls back to random).
    """

    def __init__(self, n_actions: int) -> None:
        super().__init__(n_actions, name="LeastBusy")

    def select_action(self, obs: dict[str, Any], *, deterministic: bool = False) -> int:
        busy = obs.get("technician_busy")
        if busy is None:
            return int(np.random.randint(self.n_actions))
        busy = np.asarray(busy)
        free = np.where(busy == 0)[0]
        if len(free) > 0:
            return int(np.random.choice(free)) if not deterministic else int(free[0])
        return int(np.random.randint(self.n_actions))


class LeastFatiguedAgent(Agent):
    """Pick the non-busy technician with the lowest fatigue.

    Falls back to least-busy when fatigue info is unavailable.
    """

    def __init__(self, n_actions: int) -> None:
        super().__init__(n_actions, name="LeastFatigued")

    def select_action(self, obs: dict[str, Any], *, deterministic: bool = False) -> int:
        busy = obs.get("technician_busy")
        fatigue = obs.get("technician_fatigue")

        if busy is None:
            return int(np.random.randint(self.n_actions))

        busy = np.asarray(busy, dtype=np.float32)
        if fatigue is not None:
            fatigue = np.asarray(fatigue, dtype=np.float32)
        else:
            fatigue = np.zeros_like(busy)

        # Score: fatigue + large penalty for busy techs
        scores = fatigue + busy * 100.0
        return int(np.argmin(scores))


class ShortestQueueAgent(Agent):
    """Assign to least-busy tech, breaking ties by queue awareness.

    Prefers technicians that are free.  When all are busy, picks the one
    with the lowest fatigue (proxy for finishing soonest).  This is a
    slightly smarter version of LeastBusy.
    """

    def __init__(self, n_actions: int) -> None:
        super().__init__(n_actions, name="ShortestQueue")

    def select_action(self, obs: dict[str, Any], *, deterministic: bool = False) -> int:
        busy = obs.get("technician_busy")
        fatigue = obs.get("technician_fatigue")

        if busy is None:
            return int(np.random.randint(self.n_actions))

        busy = np.asarray(busy, dtype=np.float32)
        fatigue = (
            np.asarray(fatigue, dtype=np.float32)
            if fatigue is not None
            else np.zeros_like(busy)
        )

        free = np.where(busy == 0)[0]
        if len(free) > 0:
            # Among free techs, pick the least fatigued
            best = free[np.argmin(fatigue[free])]
            return int(best)
        # All busy: pick least fatigued (will finish sooner)
        return int(np.argmin(fatigue))

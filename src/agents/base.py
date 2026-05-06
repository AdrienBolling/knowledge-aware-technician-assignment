"""Abstract base class for all KATA agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Agent(ABC):
    """Base class for agents interacting with KataEnv.

    Every agent must implement :meth:`select_action`.  Learning agents
    should also override :meth:`update`.  The :meth:`save` / :meth:`load`
    pair allows checkpointing.

    Parameters
    ----------
    n_actions:
        Size of the discrete action space (number of technicians).
    name:
        Human-readable identifier for logging.

    """

    def __init__(self, n_actions: int, *, name: str = "Agent") -> None:
        self.n_actions = n_actions
        self.name = name

    @abstractmethod
    def select_action(self, obs: dict[str, Any], *, deterministic: bool = False) -> int:
        """Choose an action given the current observation.

        Parameters
        ----------
        obs:
            Observation dict from ``KataEnv`` (keys depend on the
            observation representation).
        deterministic:
            When ``True``, use a greedy / deterministic policy (for
            evaluation).  When ``False``, the agent may explore.

        Returns
        -------
        int
            Index of the technician to assign.

        """

    def update(self, **kwargs: Any) -> dict[str, float]:
        """Perform one learning step.  No-op for non-learning agents.

        Returns a dict of scalar metrics (loss, etc.) for logging.
        """
        return {}

    def observe_transition(
        self,
        obs: dict[str, Any],
        action: int,
        reward: float,
        next_obs: dict[str, Any],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        """Store a transition for later learning (e.g. replay buffer).

        Non-learning agents can ignore this.
        """

    def on_episode_start(self) -> None:
        """Called at the beginning of each episode."""

    def on_episode_end(self, episode_reward: float) -> None:
        """Called at the end of each episode."""

    def save(self, path: str | Path) -> None:
        """Persist agent state to *path*."""

    def load(self, path: str | Path) -> None:
        """Restore agent state from *path*."""

    def __repr__(self) -> str:
        return f"{self.name}(n_actions={self.n_actions})"

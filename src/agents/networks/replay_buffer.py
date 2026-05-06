"""Experience replay buffers for off-policy learning."""

from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import numpy as np
import torch


class Transition(NamedTuple):
    """A single environment transition."""

    obs: np.ndarray  # token_ids or flattened obs
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    """Uniform-sampling replay buffer.

    Parameters
    ----------
    capacity:
        Maximum number of transitions stored.

    """

    def __init__(self, capacity: int = 100_000) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._buffer.append(Transition(obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a random batch and return it as stacked tensors."""
        batch = random.sample(self._buffer, batch_size)
        return {
            "obs": torch.from_numpy(np.stack([t.obs for t in batch])),
            "action": torch.tensor([t.action for t in batch], dtype=torch.long),
            "reward": torch.tensor([t.reward for t in batch], dtype=torch.float32),
            "next_obs": torch.from_numpy(np.stack([t.next_obs for t in batch])),
            "done": torch.tensor([t.done for t in batch], dtype=torch.float32),
        }

    def __len__(self) -> int:
        return len(self._buffer)


class PrioritizedReplayBuffer:
    """Proportional prioritized experience replay.

    Uses a simple array-based implementation with stochastic
    prioritized sampling.

    Parameters
    ----------
    capacity:
        Maximum number of transitions.
    alpha:
        Prioritization exponent (0 = uniform, 1 = full prioritization).
    beta_start:
        Initial importance-sampling correction exponent.
    beta_frames:
        Number of frames over which beta anneals to 1.0.

    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
    ) -> None:
        self._capacity = capacity
        self._alpha = alpha
        self._beta_start = beta_start
        self._beta_frames = beta_frames
        self._frame = 0

        self._buffer: list[Transition | None] = [None] * capacity
        self._priorities = np.zeros(capacity, dtype=np.float64)
        self._pos = 0
        self._size = 0

    @property
    def _beta(self) -> float:
        fraction = min(1.0, self._frame / max(1, self._beta_frames))
        return self._beta_start + (1.0 - self._beta_start) * fraction

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        max_prio = self._priorities[: self._size].max() if self._size > 0 else 1.0
        self._buffer[self._pos] = Transition(obs, action, reward, next_obs, done)
        self._priorities[self._pos] = max_prio
        self._pos = (self._pos + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """Sample a prioritized batch.

        Returns
        -------
        batch:
            Dict of tensors (obs, action, reward, next_obs, done).
        indices:
            Indices of sampled transitions (for updating priorities).
        weights:
            Importance-sampling weights.

        """
        self._frame += 1
        prios = self._priorities[: self._size] ** self._alpha
        probs = prios / prios.sum()

        indices = np.random.choice(self._size, batch_size, p=probs, replace=False)
        transitions = [self._buffer[i] for i in indices]

        # Importance-sampling weights
        weights = (self._size * probs[indices]) ** (-self._beta)
        weights /= weights.max()

        batch = {
            "obs": torch.from_numpy(np.stack([t.obs for t in transitions])),
            "action": torch.tensor([t.action for t in transitions], dtype=torch.long),
            "reward": torch.tensor(
                [t.reward for t in transitions], dtype=torch.float32
            ),
            "next_obs": torch.from_numpy(np.stack([t.next_obs for t in transitions])),
            "done": torch.tensor([t.done for t in transitions], dtype=torch.float32),
        }
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled transitions."""
        for idx, prio in zip(indices, priorities):
            self._priorities[idx] = max(prio, 1e-6)

    def __len__(self) -> int:
        return self._size

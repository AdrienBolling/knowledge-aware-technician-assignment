"""Lightweight rollout collection for encoder pretraining.

The MLM pretrainer only needs token sequences (and, in hybrid mode, the
parallel continuous channels emitted by the env).  This module provides
a tiny ring-buffer and a ``collect_token_rollouts`` helper that runs a
Gym env with any ``Agent`` and stashes one row of token observations
per step.

The buffer auto-detects from the first observation whether to store a
single ``token_ids`` channel (legacy ``token_ids`` mode) or the three
hybrid channels (``token_ids``, ``cont_values``, ``cont_kinds``).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class TokenObsBuffer:
    """Fixed-capacity ring buffer of token observations.

    Stores observations as ``(N, S)`` int64 in the non-hybrid case, or
    a dict of ``(N, S)`` arrays in the hybrid case.  Old rows are
    overwritten once ``capacity`` is reached, so memory stays bounded
    even on very long rollout horizons.
    """

    def __init__(
        self,
        capacity: int,
        seq_length: int,
        *,
        hybrid: bool = False,
    ) -> None:
        if capacity <= 0:
            msg = "capacity must be positive"
            raise ValueError(msg)
        if seq_length <= 0:
            msg = "seq_length must be positive"
            raise ValueError(msg)
        self.capacity = int(capacity)
        self.seq_length = int(seq_length)
        self.hybrid = bool(hybrid)
        self._token_ids = np.zeros((self.capacity, self.seq_length), dtype=np.int64)
        if self.hybrid:
            self._cont_values = np.zeros(
                (self.capacity, self.seq_length), dtype=np.float32
            )
            self._cont_kinds = np.zeros(
                (self.capacity, self.seq_length), dtype=np.int8
            )
        self._next = 0
        self._filled = 0

    def __len__(self) -> int:
        return self._filled

    def add(self, obs: np.ndarray | dict[str, np.ndarray]) -> None:
        """Add a single observation.

        ``obs`` is either a ``(S,)`` int64 array (non-hybrid) or a dict
        with keys ``token_ids`` / ``cont_values`` / ``cont_kinds`` each
        of shape ``(S,)`` (hybrid).
        """
        if self.hybrid:
            if not isinstance(obs, dict):
                msg = (
                    "hybrid buffer expects a dict observation with "
                    "token_ids / cont_values / cont_kinds"
                )
                raise TypeError(msg)
            tid = np.asarray(obs["token_ids"], dtype=np.int64)
            cv = np.asarray(obs["cont_values"], dtype=np.float32)
            ck = np.asarray(obs["cont_kinds"], dtype=np.int8)
            for name, arr in (("token_ids", tid), ("cont_values", cv), ("cont_kinds", ck)):
                if arr.shape != (self.seq_length,):
                    msg = (
                        f"{name} shape {arr.shape} does not match buffer "
                        f"seq_length {self.seq_length}"
                    )
                    raise ValueError(msg)
            self._token_ids[self._next] = tid
            self._cont_values[self._next] = cv
            self._cont_kinds[self._next] = ck
        else:
            arr = np.asarray(obs, dtype=np.int64)
            if arr.shape != (self.seq_length,):
                msg = (
                    f"token_ids shape {arr.shape} does not match buffer "
                    f"seq_length {self.seq_length}"
                )
                raise ValueError(msg)
            self._token_ids[self._next] = arr
        self._next = (self._next + 1) % self.capacity
        self._filled = min(self._filled + 1, self.capacity)

    def as_array(self) -> np.ndarray | dict[str, np.ndarray]:
        """Return stored sequences.

        Returns ``(N, S)`` token-id array in non-hybrid mode, or a dict
        of three ``(N, S)`` arrays in hybrid mode.
        """
        n = self._filled
        if self.hybrid:
            return {
                "token_ids": self._token_ids[:n].copy(),
                "cont_values": self._cont_values[:n].copy(),
                "cont_kinds": self._cont_kinds[:n].copy(),
            }
        return self._token_ids[:n].copy()


def collect_token_rollouts(
    env: Any,
    agent: Any,
    n_steps: int,
    *,
    buffer: TokenObsBuffer | None = None,
    seed: int | None = None,
) -> TokenObsBuffer:
    """Roll out ``agent`` in ``env`` and stash token observations.

    Auto-detects from the first observation whether the env is in
    ``hybrid`` mode (presence of ``cont_values`` key) or plain
    ``token_ids`` mode and allocates the buffer accordingly.

    Parameters
    ----------
    env:
        ``KataEnv`` (or any Gym env emitting ``obs["token_ids"]``, plus
        ``obs["cont_values"]`` / ``obs["cont_kinds"]`` in hybrid mode).
    agent:
        Any object with ``select_action(obs, deterministic=False)``.
    n_steps:
        Total environment steps to collect.
    buffer:
        Existing buffer to extend.  When ``None``, a new buffer is
        created with capacity ``n_steps``.

    Returns
    -------
    buffer:
        The (now-populated) ``TokenObsBuffer``.
    """

    def _to_record(o: dict[str, Any]) -> np.ndarray | dict[str, np.ndarray]:
        if "cont_values" in o and "cont_kinds" in o:
            return {
                "token_ids": np.asarray(o["token_ids"]),
                "cont_values": np.asarray(o["cont_values"]),
                "cont_kinds": np.asarray(o["cont_kinds"]),
            }
        return np.asarray(o["token_ids"])

    obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
    first_record = _to_record(obs)

    if buffer is None:
        if isinstance(first_record, dict):
            seq_len = int(first_record["token_ids"].shape[-1])
            buffer = TokenObsBuffer(capacity=n_steps, seq_length=seq_len, hybrid=True)
        else:
            seq_len = int(first_record.shape[-1])
            buffer = TokenObsBuffer(capacity=n_steps, seq_length=seq_len, hybrid=False)
        buffer.add(first_record)
        steps_taken = 1
    else:
        steps_taken = 0

    while steps_taken < n_steps:
        action = agent.select_action(obs, deterministic=False)
        obs, _, terminated, truncated, _ = env.step(action)
        buffer.add(_to_record(obs))
        steps_taken += 1
        if terminated or truncated:
            obs, _ = env.reset()
    return buffer

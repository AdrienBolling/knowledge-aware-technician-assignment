"""Warmup environment for fitting embedders (e.g. MCA) before training.

``WarmupEnv`` is a thin subclass of :class:`KataEnv` that runs the simulation
under a heuristic policy, collects repair requests, and uses them to fit an
encoder and pre-populate a tokenizer vocabulary.

Typical usage inside ``KataEnv.reset()``::

    warmup = WarmupEnv(scenario_factory=self._scenario_factory, config=self.config)
    encoder, tokenizer = warmup.run()
"""

from __future__ import annotations

from typing import Any

from kata.core.tokenizer import StateTokenizer
from kata.entities.encoder.mca_encoder import MCAEncoder
from kata.env import KataEnv


class WarmupEnv(KataEnv):
    """Runs a heuristic policy to collect repair data for encoder fitting.

    Parameters
    ----------
    n_warmup_steps:
        Maximum number of assignment steps to execute during warmup.
    mca_grid_shape:
        Grid shape passed to the MCAEncoder.
    mca_n_components:
        Number of MCA components.
    tokenizer_seq_length:
        Sequence length for the StateTokenizer.

    """

    def __init__(
        self,
        *,
        n_warmup_steps: int = 200,
        mca_grid_shape: tuple[int, ...] = (10, 10),
        mca_n_components: int = 2,
        tokenizer_seq_length: int = 64,
        **kwargs: Any,
    ) -> None:
        self._n_warmup_steps = n_warmup_steps
        self._mca_grid_shape = mca_grid_shape
        self._mca_n_components = mca_n_components
        self._tokenizer_seq_length = tokenizer_seq_length
        self._collected_requests: list[Any] = []
        self._collected_token_obs: list[tuple[str, ...]] = []
        super().__init__(**kwargs)

    def _heuristic_action(self) -> int:
        """Select the least-busy, least-fatigued technician."""
        techs = self.dispatcher.techs
        best_idx = 0
        best_score = float("inf")
        for i, tech in enumerate(techs):
            busy_penalty = 100.0 if getattr(tech, "busy", False) else 0.0
            fatigue = float(getattr(tech, "fatigue", 0.0))
            score = busy_penalty + fatigue
            if score < best_score:
                best_score = score
                best_idx = i
        return best_idx

    def run(self) -> tuple[MCAEncoder, StateTokenizer]:
        """Execute warmup and return a fitted (encoder, tokenizer) pair."""
        obs, _info = self.reset()
        self._collect_obs(obs)

        for _ in range(self._n_warmup_steps):
            if self.current_request is not None:
                self._collected_requests.append(self.current_request)

            action = self._heuristic_action()
            obs, _reward, terminated, truncated, _info = self.step(action)
            self._collect_obs(obs)

            if terminated or truncated:
                obs, _info = self.reset()
                self._collect_obs(obs)

        # Fit MCA encoder
        encoder = MCAEncoder(
            grid_shape=self._mca_grid_shape,
            n_components=self._mca_n_components,
        )
        if self._collected_requests:
            encoder.fit(self._collected_requests)

        # Build tokenizer vocabulary from collected observations
        tokenizer = StateTokenizer(seq_length=self._tokenizer_seq_length)
        for token_seq in self._collected_token_obs:
            tokenizer.encode(token_seq)  # adds unseen tokens to vocab
        tokenizer.freeze()

        return encoder, tokenizer

    def _collect_obs(self, obs: dict[str, Any]) -> None:
        """Store token observations for vocabulary building."""
        tokens = obs.get("tokens")
        if tokens is not None:
            self._collected_token_obs.append(tuple(tokens))

"""Vectorised parallel KataEnv construction for PPO training.

Provides picklable worker factories so ``gymnasium.vector.AsyncVectorEnv``
can rebuild a fully-independent simulator stack (config, scenario
sampler, tokenizer, SimPy environment) inside each subprocess.  Nothing
is shared between workers except the immutable JSON config and the
frozen vocabulary, so SimPy state cannot collide across environments;
``scripts/sanity_vec_env.py`` verifies this empirically (solo-vs-vector
bit-identical episodes, seed independence, wall-clock scaling).

Each worker's :class:`RandomScenarioSampler` is seeded with
``base_seed + 1000 * worker_idx`` so parallel rollouts traverse
*different* factory layouts, mirroring the diversity a single
sequential env would see across episodes.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import gymnasium as gym
import numpy as np


class SeededResetWrapper(gym.Wrapper):
    """Seed the *process-global* RNGs on seeded resets.

    Parts of the simulator (component failure draws, heuristic
    tie-breaks) consume ``np.random`` / ``random`` directly rather than
    an env-local generator.  A worker process therefore starts from OS
    entropy and two identically-seeded workers would diverge.  Seeding
    the globals at reset — exactly what the benchmark harness does —
    makes every worker's trajectory a pure function of its seeds.
    Unseeded (auto)resets continue the process-local stream, which
    keeps successive episodes diverse but still reproducible end-to-end.
    """

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return self.env.reset(seed=seed, options=options)


def _build_worker_env(
    env_cfg_json: str,
    agent_type: str,
    worker_idx: int,
    base_seed: int,
    vocab: dict[str, int] | None,
):
    """Build one independent KataEnv (runs inside the worker process)."""
    os.environ.setdefault("KATA_CONF_PATH", "/dev/null/__no_file__")
    from kata.core.config import KATAConfig
    from kata.core.tokenizer import StateTokenizer
    from kata.env import KataEnv
    from kata.EntityFactories import RandomScenarioSampler
    from kata.scenario import ScenarioBuilder

    cfg = KATAConfig(**json.loads(env_cfg_json))
    gym_cfg = cfg.gym
    rcfg = cfg.randomized_scenario

    if rcfg.enabled:
        sampler_seed = (rcfg.seed or 0) + 1000 * worker_idx + base_seed
        factory = RandomScenarioSampler(cfg, rcfg, seed=sampler_seed)
    else:
        factory = lambda: ScenarioBuilder(cfg).build()  # noqa: E731

    tokenizer = None
    if vocab is not None:
        tokenizer = StateTokenizer(seq_length=gym_cfg.tokenizer_seq_length)
        tokenizer.load_vocab(vocab)
        tokenizer.freeze()

    representation = "set" if agent_type == "set_transformer" else (
        gym_cfg.observation_representation
    )
    gym_cfg = gym_cfg.model_copy(
        update={"observation_representation": representation}
    )
    env = KataEnv(scenario_factory=factory, config=gym_cfg, tokenizer=tokenizer)
    return SeededResetWrapper(env)


class _WorkerFactory:
    """Picklable callable wrapper (closures don't always survive spawn)."""

    def __init__(self, env_cfg_json, agent_type, worker_idx, base_seed, vocab):
        self.args = (env_cfg_json, agent_type, worker_idx, base_seed, vocab)

    def __call__(self):
        return _build_worker_env(*self.args)


def build_vector_env(
    env_cfg: Any,
    agent_type: str,
    n_envs: int,
    *,
    base_seed: int = 0,
    vocab: dict[str, int] | None = None,
    use_async: bool = True,
) -> gym.vector.VectorEnv:
    """Create an ``n_envs``-way vector env over independent simulators.

    Parameters
    ----------
    env_cfg:
        The ``KATAConfig`` (pydantic model) of the training environment.
    agent_type:
        Agent key — decides the observation representation.
    vocab:
        Frozen tokenizer vocabulary shared by every worker (dict is
        immutable-by-convention here; each worker builds its own
        ``StateTokenizer`` from it).
    use_async:
        Subprocess workers (true parallelism) vs in-process stepping
        (deterministic debugging).
    """
    env_cfg_json = env_cfg.model_dump_json()
    fns = [
        _WorkerFactory(env_cfg_json, agent_type, i, base_seed, vocab)
        for i in range(n_envs)
    ]
    if use_async:
        return gym.vector.AsyncVectorEnv(fns, shared_memory=False)
    return gym.vector.SyncVectorEnv(fns)


def unbatch_obs(obs: dict[str, Any], n_envs: int) -> list[dict[str, Any]]:
    """Split a vector env's dict-of-stacked-arrays obs into per-env dicts."""
    return [{k: v[i] for k, v in obs.items()} for i in range(n_envs)]

"""Vectorised parallel KataEnv construction for PPO training.

Provides picklable worker factories so ``gymnasium.vector.AsyncVectorEnv``
can rebuild a fully-independent simulator stack (config, scenario
sampler, tokenizer, SimPy environment) inside each subprocess.  Nothing
is shared between workers except the immutable JSON config and the
frozen vocabulary, so SimPy state cannot collide across environments;
``scripts/sanity_vec_env.py`` verifies this empirically (solo-vs-vector
bit-identical episodes over the first *and* the autoreset episode,
seed independence, wall-clock scaling).

Reproducibility holds per worker process: every episode of a worker —
including the ones started by autoreset — is a pure function of that
worker's initial reset seed (see :class:`SeededResetWrapper`).  The
*interleaving* of workers is not controlled, so a vectorised run
reproduces per-env trajectories, not the order they arrive in.

Each worker's :class:`RandomScenarioSampler` is seeded with
``base_seed + 1000 * worker_idx`` so parallel rollouts traverse
*different* factory layouts, mirroring the diversity a single
sequential env would see across episodes.
"""

from __future__ import annotations

import json
import os
import random
import warnings
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode

from experiment.config import SET_OBS_AGENT_TYPES
from kata.funcs import seed_numba_rng


class SeededResetWrapper(gym.Wrapper):
    """Seed the *process-global* RNGs on every reset.

    Parts of the simulator (component failure draws, heuristic
    tie-breaks) consume ``np.random`` / ``random`` directly rather than
    an env-local generator.  A worker process therefore starts from OS
    entropy and two identically-seeded workers would diverge.  Seeding
    the globals at reset — exactly what the benchmark harness does —
    makes every worker's trajectory a pure function of its seeds.

    Autoresets arrive with ``seed=None``; passing that through would
    leave the per-technician disruption RNGs (rebuilt from OS entropy on
    every scenario bootstrap, and only re-seeded by ``KataEnv.reset``
    when a seed is given) uncontrolled from episode 2 onwards.  A seed is
    therefore *derived* from the already-seeded process-local stream, so
    successive episodes stay diverse while the whole worker trajectory
    remains a pure function of its first seed.
    """

    def reset(self, *, seed: int | None = None, options=None):
        if seed is None:
            seed = int(np.random.randint(0, 2**31))
        np.random.seed(seed)
        random.seed(seed)
        # numba keeps its own RNG state: the machine-failure draws in
        # kata.funcs.step_degrade ignore the global seeds above.
        seed_numba_rng(seed & 0xFFFFFFFF)
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

    representation = "set" if agent_type in SET_OBS_AGENT_TYPES else (
        gym_cfg.observation_representation
    )
    gym_cfg = gym_cfg.model_copy(
        update={"observation_representation": representation}
    )
    env = KataEnv(scenario_factory=factory, config=gym_cfg, tokenizer=tokenizer)
    # KataEnv.__init__ consumed one sampler build to bootstrap the
    # observation/action spaces; forget it so the worker's first
    # TRAINING episode opens a fresh episodes_per_scenario block
    # (mirrors Experiment._build_env — otherwise every k-block is
    # offset by one and straddles two sampled factories).
    reset_scenario_cache = getattr(factory, "reset_scenario_cache", None)
    if callable(reset_scenario_cache):
        reset_scenario_cache()
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
        (debugging only — see the warning below).
    """
    env_cfg_json = env_cfg.model_dump_json()
    fns = [
        _WorkerFactory(env_cfg_json, agent_type, i, base_seed, vocab)
        for i in range(n_envs)
    ]
    if use_async:
        # "fork" pinned explicitly: the interpreter default flips to
        # forkserver on 3.14, whose children would need the `src` layout
        # re-plumbed onto PYTHONPATH.
        venv = gym.vector.AsyncVectorEnv(
            fns,
            shared_memory=False,
            context="fork",
            autoreset_mode=AutoresetMode.NEXT_STEP,
        )
    else:
        if n_envs > 1:
            warnings.warn(
                "SyncVectorEnv with n_envs > 1 steps every simulator in one "
                "process: they share np.random/random, so seeded resets "
                "overwrite each other and the envs' draws interleave. "
                "Trajectories are cross-coupled and not reproducible — "
                "debug use only, never for training runs.",
                RuntimeWarning,
                stacklevel=2,
            )
        venv = gym.vector.SyncVectorEnv(fns, autoreset_mode=AutoresetMode.NEXT_STEP)
    if venv.autoreset_mode is not AutoresetMode.NEXT_STEP:
        raise RuntimeError(
            "Vector env autoreset mode is "
            f"{venv.autoreset_mode!r}, expected {AutoresetMode.NEXT_STEP!r}: "
            "the training loops assume the terminal step returns the final "
            "observation and the *next* step call is the reset one."
        )
    return venv


def unbatch_obs(obs: dict[str, Any], n_envs: int) -> list[dict[str, Any]]:
    """Split a vector env's dict-of-stacked-arrays obs into per-env dicts."""
    return [{k: v[i] for k, v in obs.items()} for i in range(n_envs)]

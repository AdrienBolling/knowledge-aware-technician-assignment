"""Vectorised-training regression tests.

Pins the contracts of the parallel (``parallel_envs > 1``) training path
in :meth:`Experiment._train_loop_vec` and of the seeding it shares with
the serial loop:

* the final checkpoint is written by BOTH loops (the vec loop used to
  return without one, so a clean run left only round checkpoints);
* inline-eval results reach ``_log_wandb`` from the vec loop;
* the NEXT_STEP autoreset step is never recorded as a transition;
* every episode of a worker --- including the ones started by autoreset
  --- is a pure function of that worker's first reset seed
  (``vec_env.SeededResetWrapper`` derives a seed for unseeded resets);
* ``_run_episode`` seeds the process-global RNGs the simulator draws
  from, so re-running a seed replays the episode;
* the stream snapshot taken around an inline eval carries the semi-MDP
  sim-time anchor.

Real simulators throughout (baseline.json world with shrunk horizons);
only the harness-side W&B sink and the vector-env constructor are
wrapped.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from agents.ppo.ppo_set_transformer import SetTransformerAgent
from experiment.config import (
    AgentConfig,
    CheckpointConfig,
    EvalConfig,
    ExperimentConfig,
    ReportsConfig,
    WandbConfig,
)
from experiment.runner import Experiment
from experiment.vec_env import build_vector_env, unbatch_obs
from kata.core.config import KATAConfig
from kata.core.tokenizer import StateTokenizer

BASELINE = Path("run_configs/benchmark_suite/baseline.json")

# The step cap binds long before the sim horizon, so every episode is a
# fixed, small number of decisions.  ``ROLLOUT_STEPS`` exceeds the cap so
# every PPO round crosses at least one episode boundary (and hence one
# autoreset).
MAX_EPISODE_STEPS = 40
MAX_SIM_TIME = 50_000.0
ROLLOUT_STEPS = 48

AGENT_PARAMS: dict[str, object] = {
    "d_model": 32,
    "n_heads": 2,
    "n_layers": 1,
    "dropout": 0.0,
    "value_hidden": 32,
    "pointer_d_attn": 16,
    "rollout_steps": ROLLOUT_STEPS,
    "n_epochs": 1,
    "minibatch_size": 32,
    "total_updates": 8,
    "warmup_updates": 1,
    "seed": 3,
    "device": "cpu",
}


@pytest.fixture(scope="module")
def env_cfg() -> KATAConfig:
    """baseline.json in set mode, shrunk to a seconds-long episode."""
    raw = json.loads(BASELINE.read_text())
    raw["gym"].update(
        {
            "observation_representation": "set",
            "max_episode_steps": MAX_EPISODE_STEPS,
            "max_sim_time": MAX_SIM_TIME,
            # Caps sized to the sampled world (4 techs, 12-22 machines):
            # every action index is then a real technician slot.
            "max_techs": 4,
            "max_machines": 24,
        }
    )
    return KATAConfig(**raw)


@pytest.fixture(scope="module")
def set_vocab(env_cfg) -> dict[str, int]:
    """Canonical set vocabulary (the workers' frozen token alphabet)."""
    path = Path(env_cfg.gym.set_vocab_path or "")
    if not path.is_file():
        pytest.skip(f"canonical set vocab not built: {path}")
    return StateTokenizer.from_json(
        path, seq_length=env_cfg.gym.tokenizer_seq_length
    ).get_vocab()


def _experiment(
    env_cfg: KATAConfig,
    ckpt_dir: Path,
    *,
    parallel_envs: int,
    n_episodes: int,
    eval_enabled: bool = False,
) -> Experiment:
    exp_cfg = ExperimentConfig(
        mode="train",
        seed=7,
        n_episodes=n_episodes,
        log_interval=100,
        parallel_envs=parallel_envs,
        eval=EvalConfig(enabled=eval_enabled, interval=1, n_episodes=1),
        checkpoint=CheckpointConfig(
            enabled=True, interval=1, dir=str(ckpt_dir), save_best=False
        ),
        wandb=WandbConfig(enabled=False),
        reports=ReportsConfig(enabled=False),
    )
    return Experiment(
        env_cfg,
        AgentConfig(agent_type="set_transformer", params=dict(AGENT_PARAMS)),
        exp_cfg,
        quiet=True,
    )


def _first_valid_action(obs: dict[str, np.ndarray]) -> int:
    """Scripted policy: the lowest unmasked technician slot."""
    valid = np.flatnonzero(np.asarray(obs["action_mask"]).reshape(-1))
    return int(valid[0]) if valid.size else 0


class _DoneRecorder:
    """Delegating vector-env proxy that logs each step's done flags."""

    def __init__(self, venv, log: list[np.ndarray]) -> None:
        self._venv = venv
        self._log = log

    def step(self, actions):
        obs, rewards, terms, truncs, infos = self._venv.step(actions)
        self._log.append(
            np.asarray(terms, dtype=bool) | np.asarray(truncs, dtype=bool)
        )
        return obs, rewards, terms, truncs, infos

    def __getattr__(self, name):
        return getattr(self._venv, name)


def test_vec_final_checkpoint(env_cfg, tmp_path):
    """Both loops must leave a ``*_final.pt`` behind, plus the periodic
    round/episode checkpoints."""
    vec_dir = tmp_path / "vec"
    _experiment(env_cfg, vec_dir, parallel_envs=2, n_episodes=4).run()
    assert (vec_dir / "set_transformer_final.pt").is_file()
    assert sorted(vec_dir.glob("set_transformer_round*.pt")), (
        f"no periodic round checkpoints in {sorted(vec_dir.iterdir())}"
    )

    serial_dir = tmp_path / "serial"
    _experiment(env_cfg, serial_dir, parallel_envs=1, n_episodes=2).run()
    assert (serial_dir / "set_transformer_final.pt").is_file()


def test_vec_eval_logged(env_cfg, tmp_path, monkeypatch):
    """The vec loop's inline eval must be logged, not silently dropped."""
    calls: list[dict] = []

    def _record(self, data, step):
        calls.append(dict(data))

    monkeypatch.setattr(Experiment, "_log_wandb", _record)
    _experiment(
        env_cfg, tmp_path, parallel_envs=2, n_episodes=2, eval_enabled=True
    ).run()

    eval_calls = [c for c in calls if any(k.startswith("eval/") for k in c)]
    assert eval_calls, f"no eval/* keys logged; saw {[sorted(c) for c in calls]}"
    assert any("eval/return_mean" in c for c in eval_calls)


def test_vec_autoreset_no_phantom_transition(env_cfg, tmp_path, monkeypatch):
    """Under NEXT_STEP autoreset the step following a done returns the new
    episode's first observation and ignores the action: it must never
    enter the rollout buffer."""
    import experiment.vec_env as vec_env

    dones: list[np.ndarray] = []
    real_build = vec_env.build_vector_env

    def _recording_build(*args, **kwargs):
        return _DoneRecorder(real_build(*args, **kwargs), dones)

    monkeypatch.setattr(vec_env, "build_vector_env", _recording_build)

    exp = _experiment(env_cfg, tmp_path, parallel_envs=2, n_episodes=4)
    recorded: list[int] = []
    real_observe = exp.agent.observe_transition

    def _spy(*args, **kwargs):
        recorded.append(int(kwargs["env_id"]))
        return real_observe(*args, **kwargs)

    exp.agent.observe_transition = _spy
    exp.run()

    flags = np.asarray(dones)  # (n_steps, n_envs)
    assert flags.size, "the vec loop never stepped the vector env"
    # Every done but one landing on the very last step costs exactly one
    # ignored (autoreset) step in the following iteration.
    n_autoreset = int(flags[:-1].sum())
    assert n_autoreset > 0, "the run never crossed an autoreset"
    assert len(recorded) == flags.size - n_autoreset


def test_vec_reproducible_end_to_end(env_cfg, set_vocab):
    """Two identically-seeded scripted pool rollouts must agree through
    episode 2 — the episode started by an unseeded autoreset."""
    episodes = 2

    def rollout() -> tuple[list[list[float]], np.ndarray]:
        venv = build_vector_env(
            env_cfg,
            "set_transformer",
            2,
            base_seed=13,
            vocab=set_vocab,
            use_async=True,
        )
        rewards: list[list[float]] = []
        finished = np.zeros(2, dtype=int)
        try:
            obs, _ = venv.reset(seed=[101, 202])
            for _ in range(4 * MAX_EPISODE_STEPS):
                actions = np.asarray(
                    [_first_valid_action(o) for o in unbatch_obs(obs, 2)],
                    dtype=np.int64,
                )
                obs, r, term, trunc, _info = venv.step(actions)
                rewards.append([float(x) for x in r])
                finished += (
                    np.asarray(term, dtype=bool) | np.asarray(trunc, dtype=bool)
                ).astype(int)
                if (finished >= episodes).all():
                    break
        finally:
            venv.close()
        return rewards, finished

    rewards_a, finished_a = rollout()
    rewards_b, finished_b = rollout()

    assert (finished_a >= episodes).all(), (
        f"rollout stopped before episode {episodes}: {finished_a}"
    )
    assert np.array_equal(finished_a, finished_b)
    assert rewards_a == rewards_b


def test_serial_seeding_parity(env_cfg, tmp_path):
    """``_run_episode`` seeds the process-global RNGs the simulator draws
    from: two identically-configured experiments replay the same episode
    whatever happened to those RNGs in between.  (A single experiment
    cannot replay itself — its scenario sampler is deliberately stateful
    and deals a fresh world on every reset.)"""
    exp = _experiment(env_cfg, tmp_path / "a", parallel_envs=1, n_episodes=1)
    first = exp._run_episode(training=False, seed=555)

    np.random.seed(2718)
    random.seed(2718)
    np.random.random(64)
    for _ in range(64):
        random.random()

    exp2 = _experiment(env_cfg, tmp_path / "b", parallel_envs=1, n_episodes=1)
    second = exp2._run_episode(training=False, seed=555)

    assert second["length"] == first["length"]
    assert second["return"] == first["return"]


def test_stream_snapshot_restores_sim_time_anchor():
    """The inline eval's ``reset_stream(0)`` clears stream 0's semi-MDP
    dt anchor; the snapshot taken around it must put the anchor back, or
    the first transition recorded after the eval gets dt=0."""
    agent = SetTransformerAgent(
        n_actions=4,
        vocab_size=32,
        d_model=16,
        n_heads=2,
        n_layers=1,
        max_techs=4,
        max_machines=8,
        env_length=8,
        device="cpu",
        seed=0,
    )
    agent._last_sim_time[0] = 1234.5
    snap = agent.snapshot_stream_state()

    agent.reset_stream(0)  # what the inline eval does via on_episode_start
    assert agent._last_sim_time[0] is None

    agent.restore_stream_state(snap)
    assert agent._last_sim_time[0] == 1234.5

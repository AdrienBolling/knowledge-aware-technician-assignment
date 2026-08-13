"""Scenario-rotation regression tests for the training environment.

``RandomizedScenarioConfig.episodes_per_scenario`` (``k``) is meant to
hold one sampled world fixed for ``k`` consecutive *episodes*.  The
rotation counter, however, lives in the sampler and advances on every
factory call --- and ``KataEnv.__init__`` calls the factory once to
bootstrap the observation / action spaces before a single episode has
run.  Left uncorrected, that bootstrap draw shifts the whole rotation by
one build: training episode 1 is already the sampler's SECOND build and
each ``k``-block straddles two worlds (``k-1`` episodes of one scenario
plus 1 of the next).  For ``k = 8`` this is the empirically observed
7+1 pattern, which breaks any group-per-scenario assumption (GRPO) and
skews the layout mix every agent sees.

These tests pin the integration --- the env is built through
:meth:`Experiment._build_env`, exactly as training does --- and the
defect itself (a KataEnv built straight from a fresh sampler, i.e. the
pre-fix construction, still straddles).

Real sampler + real ScenarioBuilder throughout; the worlds are shrunk so
each reset is a handful of machines.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from experiment.config import (
    AgentConfig,
    CheckpointConfig,
    EvalConfig,
    ExperimentConfig,
    ReportsConfig,
    WandbConfig,
)
from experiment.runner import Experiment
from kata.core.config import GymEnvConfig, KATAConfig, RandomizedScenarioConfig
from kata.env import KataEnv

EPISODES_PER_SCENARIO = 4


def _env_cfg(*, episodes_per_scenario: int = EPISODES_PER_SCENARIO) -> KATAConfig:
    """A tiny randomised world with a wide machine-count spread.

    The spread matters: two consecutive draws must be distinguishable by
    the observable signature, not only by cache identity.
    """
    os.environ["KATA_CONF_PATH"] = "/dev/null/__no_file__"
    return KATAConfig(
        gym=GymEnvConfig(
            observation_representation="structured",
            max_episode_steps=8,
            max_sim_time=2_000.0,
            max_techs=16,
            max_machines=32,
        ),
        randomized_scenario=RandomizedScenarioConfig(
            enabled=True,
            seed=2026,
            n_technicians=3,
            episodes_per_scenario=episodes_per_scenario,
            technician_templates=["expert", "generalist", "junior"],
            n_machines_min=3,
            n_machines_max=9,
            machine_templates=[
                "cnc_weibull", "assembly_mixed", "conveyor", "welder",
            ],
            route_min_length=2,
            route_max_length=3,
        ),
    )


def _experiment(env_cfg: KATAConfig, ckpt_dir: Path) -> Experiment:
    """Experiment with the cheapest possible agent — only the env build
    (and hence the training env's scenario factory) is under test."""
    exp_cfg = ExperimentConfig(
        mode="train",
        seed=7,
        n_episodes=1,
        log_interval=100,
        parallel_envs=1,
        eval=EvalConfig(enabled=False, interval=1, n_episodes=1),
        checkpoint=CheckpointConfig(enabled=False, interval=1, dir=str(ckpt_dir)),
        wandb=WandbConfig(enabled=False),
        reports=ReportsConfig(enabled=False),
    )
    return Experiment(
        env_cfg,
        AgentConfig(agent_type="random", params={}),
        exp_cfg,
        quiet=True,
    )


def _signature(env: KataEnv) -> tuple[Any, ...]:
    """Observable identity of the world the env currently holds.

    Fleet size + technician names + machine ids: all pure functions of
    the sampled ``KATAConfig``, so two builds of one cached scenario
    agree and two different draws (almost surely) do not.
    """
    dispatcher = env.dispatcher
    return (
        len(dispatcher.techs),
        tuple(t.name for t in dispatcher.techs),
        tuple(sorted(dispatcher.machines)),
    )


def _cached_scenario(env: KataEnv) -> Any:
    """The sampler's currently cached ``KATAConfig`` (identity marker)."""
    return env._scenario_factory._cached_config


def _reset_signatures(env: KataEnv, n_resets: int) -> list[tuple[Any, ...]]:
    """Reset ``n_resets`` times, recording (signature, cached-config id)."""
    out: list[tuple[Any, ...]] = []
    for i in range(n_resets):
        env.reset(seed=1000 + i)
        out.append((_signature(env), id(_cached_scenario(env))))
    return out


@pytest.fixture(scope="module")
def env_cfg() -> KATAConfig:
    return _env_cfg()


def test_build_env_drops_the_bootstrap_scenario(env_cfg, tmp_path):
    """``_build_env`` must leave the sampler's rotation at age 0.

    The bootstrap build is space setup, not an episode — if its scenario
    stays cached, episode 1 inherits a partly-consumed block.
    """
    exp = _experiment(env_cfg, tmp_path)
    sampler = exp.env._scenario_factory

    assert sampler._cached_config is None, (
        "the bootstrap scenario is still cached — episode 1 would run in a "
        "block that is already one build old"
    )
    assert sampler._scenario_age == 0
    # The draw itself still happened (spaces had to be sized).
    assert sampler._call_count == 1


def test_first_block_spans_k_full_resets(env_cfg, tmp_path):
    """Resets 1..k share one scenario; the boundary lands AFTER k."""
    k = env_cfg.randomized_scenario.episodes_per_scenario
    exp = _experiment(env_cfg, tmp_path)

    seen = _reset_signatures(exp.env, k + 1)
    block, boundary = seen[:k], seen[k]

    assert all(entry == block[0] for entry in block), (
        f"episodes 1..{k} did not share one scenario: {block}"
    )
    # Rotation happened exactly once, and at reset k+1 — the sharp,
    # RNG-independent statement of "the boundary lands after k".
    assert boundary[1] != block[0][1], (
        f"reset {k + 1} reused the first block's scenario — rotation is late"
    )
    assert boundary[0] != block[0][0], (
        "the new block is observationally identical to the previous one; "
        "re-seed the sampler in this test so the two draws differ"
    )


def test_second_block_also_spans_k_resets(env_cfg, tmp_path):
    """Alignment must hold for every block, not just the first."""
    k = env_cfg.randomized_scenario.episodes_per_scenario
    exp = _experiment(env_cfg, tmp_path)

    seen = _reset_signatures(exp.env, 2 * k + 1)
    first, second, third = seen[:k], seen[k : 2 * k], seen[2 * k]

    assert all(entry == second[0] for entry in second), (
        f"episodes {k + 1}..{2 * k} did not share one scenario: {second}"
    )
    assert second[0][1] != first[0][1]
    assert third[1] != second[0][1]


def test_bootstrap_draw_straddles_blocks_without_the_reset(env_cfg):
    """Pin the defect: a KataEnv built straight from a fresh sampler (the
    pre-fix construction) rotates one reset EARLY — k-1 episodes of one
    world plus 1 of the next."""
    from kata.EntityFactories import RandomScenarioSampler

    k = env_cfg.randomized_scenario.episodes_per_scenario
    sampler = RandomScenarioSampler(
        env_cfg, env_cfg.randomized_scenario, seed=env_cfg.randomized_scenario.seed
    )
    env = KataEnv(scenario_factory=sampler, config=env_cfg.gym)
    # NOTE: deliberately NO ``sampler.reset_scenario_cache()`` here.

    seen = _reset_signatures(env, k)
    assert all(entry == seen[0] for entry in seen[: k - 1])
    assert seen[k - 1][1] != seen[0][1], (
        "expected the pre-fix straddle (rotation at reset k, not k+1); if this "
        "no longer holds, KataEnv stopped consuming a factory call at init"
    )

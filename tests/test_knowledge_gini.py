"""Tests for the knowledge_gini reward component and step metric.

Pins the Gini computation (level fleet, full concentration, a
hand-checked two-tech case), the retired-technician exclusion, the
reward wiring (sign + coefficient), the default-off config, and the
metric/reward identity.
"""

import pytest
from conftest import FakeDispatcher, FakeRequest

from kata.core.config import GymEnvConfig
from kata.metrics import STEP_METRICS, KnowledgeGini

from test_new_rewards import _make_env


def _env_with_knowledge(volumes, reward_overrides=None):
    dispatcher = FakeDispatcher(tech_count=len(volumes))
    for tech, v in zip(dispatcher.techs, volumes):
        tech.knowledge = v
    dispatcher.repair_queue.items.append(
        FakeRequest(machine_id=1, created_at=0.0)
    )
    return _make_env(dispatcher=dispatcher, reward_overrides=reward_overrides)


class TestGiniValue:
    def test_level_fleet_is_zero(self):
        env = _env_with_knowledge([5.0, 5.0, 5.0, 5.0])
        assert env._knowledge_gini() == pytest.approx(0.0)

    def test_full_concentration_is_n_minus_1_over_n(self):
        env = _env_with_knowledge([0.0, 0.0, 0.0, 12.0])
        assert env._knowledge_gini() == pytest.approx(0.75)

    def test_two_tech_hand_case(self):
        # [1, 3]: G = sum|xi-xj| / (2 n^2 mu) = 4 / (2*4*2) = 0.25
        env = _env_with_knowledge([1.0, 3.0])
        assert env._knowledge_gini() == pytest.approx(0.25)

    def test_degenerate_fleets_are_zero(self):
        assert _env_with_knowledge([7.0])._knowledge_gini() == 0.0
        assert _env_with_knowledge([0.0, 0.0])._knowledge_gini() == 0.0

    def test_retired_technicians_excluded(self):
        env = _env_with_knowledge([10.0, 10.0, 1000.0])
        env.dispatcher.techs[2].retired = True
        assert env._knowledge_gini() == pytest.approx(0.0)


class TestRewardWiring:
    def test_component_pays_minus_gini_times_coefficient(self):
        env = _env_with_knowledge(
            [0.0, 0.0, 0.0, 12.0],
            reward_overrides={
                "knowledge_gini": {"enabled": True, "coefficient": 2.0}
            },
        )
        env.reset()
        _, _, _, _, info = env.step(0)
        assert info["reward_breakdown"]["knowledge_gini"] == pytest.approx(
            -1.5  # 2.0 * -(0.75)
        )

    def test_level_fleet_pays_zero(self):
        env = _env_with_knowledge(
            [5.0, 5.0],
            reward_overrides={
                "knowledge_gini": {"enabled": True, "coefficient": 5.0}
            },
        )
        env.reset()
        _, _, _, _, info = env.step(0)
        assert info["reward_breakdown"]["knowledge_gini"] == pytest.approx(0.0)

    def test_disabled_by_default(self):
        assert GymEnvConfig(
            max_episode_steps=10, max_sim_time=100.0
        ).reward.knowledge_gini.enabled is False
        env = _env_with_knowledge([0.0, 12.0])
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "knowledge_gini" not in info["reward_breakdown"]


class TestMetric:
    def test_registered_in_step_metrics(self):
        assert "knowledge_gini" in {m.name for m in STEP_METRICS}

    def test_metric_matches_reward_quantity(self):
        env = _env_with_knowledge([0.0, 0.0, 0.0, 12.0])
        assert KnowledgeGini().compute(None, None, env) == pytest.approx(
            env._knowledge_gini()
        )

    def test_metric_in_step_info(self):
        env = _env_with_knowledge([1.0, 3.0])
        env.reset()
        _, _, _, _, info = env.step(0)
        assert info["metrics"]["knowledge_gini"] == pytest.approx(0.25)

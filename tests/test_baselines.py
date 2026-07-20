"""Tests for the skill (SPT) and optimisation (Hungarian) dispatching
baselines and the environment decision-support API they consume."""

from __future__ import annotations

import numpy as np
from conftest import FakeDispatcher, FakeRequest, FakeSimEnv

from agents import (
    GreedyRewardAgent,
    OptimalAssignmentAgent,
    ShortestProcessingTimeAgent,
    TopsisAgent,
    TrainWeakestAgent,
)
from kata.core.config import GymEnvConfig
from kata.env import KataEnv


def _set_skills(techs, skills):
    """Give each fake tech a ``get_knowledge_multiplier`` so its skill match
    (``1 - multiplier``) equals the requested value."""
    for t, s in zip(techs, skills):
        t.get_knowledge_multiplier = lambda request, _m=1.0 - s: _m


def _make_env(dispatcher, **cfg):
    return KataEnv(
        sim_env=FakeSimEnv(),
        dispatcher=dispatcher,
        config=GymEnvConfig(max_episode_steps=10, max_sim_time=100.0, **cfg),
    )


def _set_repair_times(techs, table):
    """Give each fake tech a ``compute_repair_time`` keyed by (tech id,
    machine id) so the decision-support API returns distinct estimates."""
    for t in techs:
        def compute(base, ticket, _tid=t.id):
            return float(table[(_tid, int(ticket.machine.machine_id))])
        t.compute_repair_time = compute


def test_expected_repair_field_present_and_shape():
    d = FakeDispatcher(tech_count=3)
    d.repair_queue.items.append(FakeRequest(machine_id=1))
    env = _make_env(d)
    obs, _ = env.reset()
    assert "technician_expected_repair" in obs
    assert obs["technician_expected_repair"].shape == (3,)
    assert "technician_expected_repair" in env.observation_space.spaces


def test_expected_repair_field_absent_when_flag_off():
    d = FakeDispatcher(tech_count=2)
    d.repair_queue.items.append(FakeRequest(machine_id=1))
    env = _make_env(d, include_repair_estimate_in_observation=False)
    obs, _ = env.reset()
    assert "technician_expected_repair" not in obs
    assert "technician_expected_repair" not in env.observation_space.spaces


def test_spt_picks_fastest_available_technician():
    d = FakeDispatcher(tech_count=3)
    _set_repair_times(d.techs, {(0, 1): 3.0, (1, 1): 9.0, (2, 1): 5.0})
    d.techs[0].busy = True  # the globally fastest tech is unavailable
    d.repair_queue.items.append(FakeRequest(machine_id=1))
    env = _make_env(d)
    obs, _ = env.reset()

    assert list(np.round(env.expected_repair_times(), 1)) == [3.0, 9.0, 5.0]
    spt = ShortestProcessingTimeAgent(3)
    spt.attach_env(env)
    # tech 0 (3.0) is busy → the fastest *available* is tech 2 (5.0)
    assert spt.select_action(obs, deterministic=True) == 2


def test_optimal_assignment_avoids_greedy_trap():
    d = FakeDispatcher(tech_count=2)
    # tech 0 is good at both machines; tech 1 is only decent at machine 1.
    _set_repair_times(
        d.techs, {(0, 1): 5.0, (1, 1): 6.0, (0, 2): 5.0, (1, 2): 100.0}
    )
    d.repair_queue.items.append(FakeRequest(machine_id=1))  # current
    d.repair_queue.items.append(FakeRequest(machine_id=2))  # queued
    env = _make_env(d)
    obs, _ = env.reset()
    assert int(env.current_request.machine.machine_id) == 1

    cost, tickets = env.assignment_cost_matrix()
    assert cost.shape == (2, 2)
    assert len(tickets) == 2

    # SPT greedily takes tech 0 for the current ticket (5 < 6)...
    spt = ShortestProcessingTimeAgent(2)
    spt.attach_env(env)
    assert spt.select_action(obs, deterministic=True) == 0

    # ...but the joint optimum saves tech 0 for the queued ticket only it can
    # serve cheaply, so Hungarian assigns tech 1 now (total 11 vs 105).
    opt = OptimalAssignmentAgent(2)
    opt.attach_env(env)
    assert opt.select_action(obs, deterministic=True) == 1


def test_assignment_cost_matrix_masks_busy_techs():
    d = FakeDispatcher(tech_count=3)
    for t in d.techs:
        t.compute_repair_time = lambda base, ticket: 7.0
    d.techs[1].busy = True
    d.repair_queue.items.append(FakeRequest(machine_id=1))
    env = _make_env(d)
    env.reset()

    cost, tickets = env.assignment_cost_matrix()
    assert cost.shape == (1, 3)
    assert np.isinf(cost[0, 1])  # busy technician's column is +inf
    assert cost[0, 0] == 7.0 and cost[0, 2] == 7.0


def test_baselines_fall_back_without_env_handle():
    # No attached env and no obs field → both degrade to an available pick.
    d = FakeDispatcher(tech_count=2)
    d.repair_queue.items.append(FakeRequest(machine_id=1))
    env = _make_env(d, include_repair_estimate_in_observation=False)
    obs, _ = env.reset()
    for agent_cls in (
        ShortestProcessingTimeAgent,
        OptimalAssignmentAgent,
        TopsisAgent,
        GreedyRewardAgent,
        TrainWeakestAgent,
    ):
        agent = agent_cls(2)  # no attach_env
        action = agent.select_action(obs, deterministic=True)
        assert action in (0, 1)


def test_topsis_prefers_pareto_best_technician():
    d = FakeDispatcher(tech_count=3)
    _set_repair_times(
        d.techs, {(0, 1): 5.0, (1, 1): 5.0, (2, 1): 20.0}
    )
    # tech 0: fast + rested; tech 1: fast + exhausted; tech 2: slow + rested.
    d.techs[0].fatigue = 0.1
    d.techs[1].fatigue = 0.9
    d.techs[2].fatigue = 0.1
    d.repair_queue.items.append(FakeRequest(machine_id=1))
    env = _make_env(d)
    obs, _ = env.reset()

    topsis = TopsisAgent(3)
    topsis.attach_env(env)
    # tech 0 dominates on both repair time and fatigue → chosen.
    assert topsis.select_action(obs, deterministic=True) == 0


def test_train_weakest_picks_least_skilled_available():
    d = FakeDispatcher(tech_count=3)
    _set_skills(d.techs, [0.8, 0.2, 0.5])  # tech 1 is least skilled
    d.repair_queue.items.append(FakeRequest(machine_id=1))
    env = _make_env(d)
    obs, _ = env.reset()
    assert list(np.round(env.skill_match_scores(), 1)) == [0.8, 0.2, 0.5]

    tw = TrainWeakestAgent(3)
    tw.attach_env(env)
    assert tw.select_action(obs, deterministic=True) == 1


def test_assignment_reward_estimates_is_side_effect_free():
    d = FakeDispatcher(tech_count=3)
    _set_skills(d.techs, [0.9, 0.3, 0.6])
    d.repair_queue.items.append(FakeRequest(machine_id=1))
    env = _make_env(d)
    env.reset()
    env.freeze_reward_normalizer()

    before = (env._prev_finished_products, env._prev_fleet_knowledge)
    r1 = env.assignment_reward_estimates()
    r2 = env.assignment_reward_estimates()
    after = (env._prev_finished_products, env._prev_fleet_knowledge)
    assert np.allclose(r1, r2)           # idempotent probe
    assert before == after               # no per-step delta mutation
    assert r1.shape == (3,)


def test_greedy_reward_matches_argmax_estimate():
    d = FakeDispatcher(tech_count=3)
    _set_skills(d.techs, [0.9, 0.3, 0.6])
    d.techs[0].busy = True  # highest-reward tech unavailable → mask respected
    d.repair_queue.items.append(FakeRequest(machine_id=1))
    env = _make_env(d)
    obs, _ = env.reset()
    env.freeze_reward_normalizer()

    gr = GreedyRewardAgent(3)
    gr.attach_env(env)
    est = env.assignment_reward_estimates()
    avail = np.array([1, 2])  # tech 0 busy
    assert gr.select_action(obs, deterministic=True) == int(avail[np.argmax(est[avail])])

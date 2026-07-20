"""Tests for the skill (SPT) and optimisation (Hungarian) dispatching
baselines and the environment decision-support API they consume."""

from __future__ import annotations

import numpy as np
from conftest import FakeDispatcher, FakeRequest, FakeSimEnv

from agents import (
    BatchMILPAgent,
    EmpiricalSPTAgent,
    EmpiricalTopsisAgent,
    GreedyRewardAgent,
    OptimalAssignmentAgent,
    ReserveSpecialistAgent,
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


def test_repair_log_records_tech_key_duration_and_resets():
    d = FakeDispatcher(tech_count=3)
    req = FakeRequest(machine_id=1)
    d.repair_queue.items.append(req)
    env = _make_env(d)
    env.reset()

    env._on_repair_completed(req, 7.5, d.techs[1])
    env._on_repair_completed(req, 3.0, d.techs[2])
    env._on_repair_completed(req, 4.0)  # legacy caller: counted, not logged
    log = env.repair_log()
    assert [rec["tech"] for rec in log] == [1, 2]
    assert log[0]["duration"] == 7.5
    assert log[0]["key"] == env.failure_key(req) == env.current_failure_key()
    assert env._completed_repair_counter == 3

    env.reset()
    assert env.repair_log() == []


def test_empirical_spt_learns_from_observed_completions():
    d = FakeDispatcher(tech_count=3)
    req = FakeRequest(machine_id=1)
    d.repair_queue.items.append(req)
    env = _make_env(d)
    obs, _ = env.reset()

    agent = EmpiricalSPTAgent(3)
    agent.attach_env(env)
    agent.on_episode_start()
    # Cold start: no completions observed yet → deterministic fallback.
    assert agent.select_action(obs, deterministic=True) == 0

    # Observed history: tech 2 is fast (3.0), tech 0 slow (9.0) on this
    # failure type; tech 1 never seen on it (backs off to its own mean 20).
    env._on_repair_completed(req, 9.0, d.techs[0])
    env._on_repair_completed(req, 3.0, d.techs[2])
    env._on_repair_completed(req, 20.0, d.techs[1])
    assert agent.select_action(obs, deterministic=True) == 2

    # The fastest tech becoming unavailable redirects to the next-best.
    d.techs[2].busy = True
    obs2 = env._structured_obs()
    assert agent.select_action(obs2, deterministic=True) == 0


def test_empirical_topsis_uses_learned_repair_criterion():
    d = FakeDispatcher(tech_count=3)
    req = FakeRequest(machine_id=1)
    d.repair_queue.items.append(req)
    env = _make_env(d)
    obs, _ = env.reset()
    # tech 0: observed fast + rested → dominates on every criterion.
    d.techs[0].fatigue = 0.1
    d.techs[1].fatigue = 0.9
    d.techs[2].fatigue = 0.1
    env._on_repair_completed(req, 5.0, d.techs[0])
    env._on_repair_completed(req, 5.0, d.techs[1])
    env._on_repair_completed(req, 20.0, d.techs[2])

    topsis = EmpiricalTopsisAgent(3)
    topsis.attach_env(env)
    topsis.on_episode_start()
    assert topsis.select_action(obs, deterministic=True) == 0


def test_reserve_specialist_picks_weakest_fast_enough():
    d = FakeDispatcher(tech_count=3)
    # Repair times: tech 0 fastest (5), tech 1 within tau*min (6 <= 7.5),
    # tech 2 too slow (20).  Skills: tech 1 weakest of the eligible pair.
    _set_repair_times(d.techs, {(0, 1): 5.0, (1, 1): 6.0, (2, 1): 20.0})
    _set_skills(d.techs, [0.8, 0.2, 0.5])
    d.repair_queue.items.append(FakeRequest(machine_id=1))
    env = _make_env(d)
    obs, _ = env.reset()

    agent = ReserveSpecialistAgent(3)  # default tau=1.5
    agent.attach_env(env)
    assert agent.select_action(obs, deterministic=True) == 1

    # tau=1.0 collapses to the SPT choice (only the fastest is eligible).
    strict = ReserveSpecialistAgent(3, tau=1.0)
    strict.attach_env(env)
    assert strict.select_action(obs, deterministic=True) == 0


def test_batch_milp_spreads_load_under_capacity_cap():
    d = FakeDispatcher(tech_count=2)
    # 3 open tickets, 2 techs → balance cap ceil(3/2)=2.  Tech 0 is
    # cheaper on every ticket, but capacity forces one ticket to tech 1;
    # the joint optimum gives tech 1 the CURRENT ticket (6 + 1 + 1 = 8
    # beats 5 + 1 + 100 = 106).
    _set_repair_times(
        d.techs,
        {(0, 1): 5.0, (1, 1): 6.0, (0, 2): 1.0, (1, 2): 100.0, (0, 3): 1.0, (1, 3): 100.0},
    )
    d.repair_queue.items.append(FakeRequest(machine_id=1))  # current
    d.repair_queue.items.append(FakeRequest(machine_id=2))  # queued
    d.repair_queue.items.append(FakeRequest(machine_id=3))  # queued
    env = _make_env(d)
    obs, _ = env.reset()

    milp = BatchMILPAgent(2)
    milp.attach_env(env)
    assert milp.select_action(obs, deterministic=True) == 1

    # Sanity: with a fleet-sized queue (cap=1) it matches plain Hungarian.
    d2 = FakeDispatcher(tech_count=2)
    _set_repair_times(d2.techs, {(0, 1): 5.0, (1, 1): 6.0, (0, 2): 5.0, (1, 2): 100.0})
    d2.repair_queue.items.append(FakeRequest(machine_id=1))
    d2.repair_queue.items.append(FakeRequest(machine_id=2))
    env2 = _make_env(d2)
    obs2, _ = env2.reset()
    milp2 = BatchMILPAgent(2)
    milp2.attach_env(env2)
    hung = OptimalAssignmentAgent(2)
    hung.attach_env(env2)
    assert milp2.select_action(obs2, deterministic=True) == hung.select_action(
        obs2, deterministic=True
    ) == 1

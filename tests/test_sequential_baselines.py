"""Tests for the model-based sequential dispatching baselines
(``agents.baselines.sequential``): the rolling-horizon look-ahead planner
and the greedy on the v5 training reward.

They read the live human state of real technicians (knowledge grids,
Jaber fatigue, disruptions), so every test builds a real ScenarioBuilder
world from a benchmark config.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agents.baselines.sequential import (
    V5_COMPONENTS,
    GreedyTrainingRewardAgent,
    RollingHorizonMPCAgent,
    load_params,
)
from kata.core.config import KATAConfig
from kata.EntityFactories.scenario_sampler import RandomScenarioSampler
from kata.entities.technicians.GymTechnician import GymTechnician
from kata.env import KataEnv
from kata.scenario import ScenarioBuilder

SUITE = Path("run_configs/benchmark_suite")

# Realistic scales (order of the calibration on the training worlds); fixed
# here so the tests do not depend on the committed parameter file.
SIGMA = {
    "repair_quality": 0.16, "fatigue_cost": 0.12, "busy_technician": 0.12,
    "throughput_delta": 0.33, "workload_balance": 0.07, "fleet_availability": 0.05,
    "knowledge_increment": 60.0, "terminal_finished_products": 280.0,
    "terminal_fleet_knowledge": 190.0,
}
PARAMS = {"sigma": SIGMA, "horizon_k": 2, "terminal_weight": 1.0}


def _env(config: str, *, sim_time: float, steps: int = 20_000, seed: int = 4321) -> KataEnv:
    cfg = KATAConfig(**json.loads((SUITE / config).read_text()))
    fixed = RandomScenarioSampler(cfg, cfg.randomized_scenario, seed=seed).sample_config()
    gym_cfg = cfg.gym.model_copy(update={
        "max_episode_steps": steps, "max_sim_time": sim_time,
        "observation_representation": "structured",
    })
    return KataEnv(scenario_factory=lambda c=fixed: ScenarioBuilder(c).build(), config=gym_cfg)


def _rollout(agent, env, *, seed: int = 7, check_mask: bool = True):
    """Run one episode; returns (actions, number of fallback decisions)."""
    import random

    np.random.seed(seed)
    random.seed(seed)
    agent.attach_env(env)
    agent.on_episode_start()
    obs, _ = env.reset(seed=seed)
    actions, fallbacks = [], 0
    while True:
        mask = np.asarray(obs["action_mask"])
        techs = env.dispatcher.techs
        if all(t.busy or t._in_disruption or t.retired for t in techs):
            fallbacks += 1
        action = agent.select_action(obs, deterministic=True)
        if check_mask:
            assert 0 <= action < len(mask) and mask[action] == 1, (action, mask)
            assert not techs[action].retired
        actions.append(int(action))
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            return actions, fallbacks


def _uniform_fleet(env, *, fatigue=None) -> None:
    """Give every technician the same parameters and an empty grid, so a
    test controls the only difference between them."""
    now = float(env._sim_time())
    for i, t in enumerate(env.dispatcher.techs):
        t.fatigue_lambda = 0.01
        t.fatigue_mu = 0.05
        g = t.knowledge_grid
        g._grid = np.zeros(g._shape)
        g._propagation_sigma = 1.0
        g.b = -np.log(0.7) / np.log(2)
        t._invalidate_knowledge_cache()
        t._fatigue = 0.0 if fatigue is None else float(fatigue[i])
        t._last_idle_since = now


def _cell(env, request) -> tuple[int, ...]:
    t = env.dispatcher.techs[0]
    return t.knowledge_grid.embedding_to_coords(t.encoder.encode(request))


def _set_experience(env, i: int, request, amount: float) -> None:
    t = env.dispatcher.techs[i]
    g = t.knowledge_grid
    grid = g._grid.copy()
    grid[_cell(env, request)] += amount
    g._grid = grid
    t._invalidate_knowledge_cache()


def _first_decision(env, seed: int = 3):
    obs, _ = env.reset(seed=seed)
    assert env.current_request is not None
    return obs


def _obs(env):
    return env._obs()


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def test_params_file_has_every_scale():
    params = load_params()
    assert set(V5_COMPONENTS) <= set(params["sigma"])
    assert all(v > 0 for v in params["sigma"].values())
    assert int(params["horizon_k"]) >= 0


# ---------------------------------------------------------------------------
# Action mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [GreedyTrainingRewardAgent, RollingHorizonMPCAgent])
def test_mask_respected_in_contended_episode(cls):
    # 2-tech small-scale world: the all-unavailable fallback happens often.
    env = _env("small_scale.json", sim_time=15_000.0)
    agent = cls(len(env.dispatcher.techs) if env.dispatcher else 30, params=PARAMS)
    actions, fallbacks = _rollout(agent, env)
    assert len(actions) > 50
    assert fallbacks > 0  # the fallback branch was exercised


@pytest.mark.parametrize("cls", [GreedyTrainingRewardAgent, RollingHorizonMPCAgent])
def test_fallback_never_picks_a_retired_technician(cls):
    env = _env("baseline.json", sim_time=5_000.0)
    obs = _first_decision(env)
    techs = env.dispatcher.techs
    techs[0].retired = True
    for t in techs[1:]:
        t.busy = True  # everybody unavailable -> mask = the active fleet
    obs = _obs(env)
    mask = np.asarray(obs["action_mask"])
    assert mask[0] == 0 and mask[1:].all()
    agent = cls(len(techs), params=PARAMS)
    agent.attach_env(env)
    agent.on_episode_start()
    action = agent.select_action(obs, deterministic=True)
    assert action != 0 and mask[action] == 1


# ---------------------------------------------------------------------------
# Determinism and the K=0 reduction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [GreedyTrainingRewardAgent, RollingHorizonMPCAgent])
def test_deterministic(cls):
    runs = []
    for _ in range(2):
        env = _env("baseline.json", sim_time=12_000.0)
        agent = cls(4, params=PARAMS)
        runs.append(_rollout(agent, env, seed=11)[0])
    assert runs[0] == runs[1]


def test_k0_without_terminal_value_equals_greedy():
    env_a = _env("baseline.json", sim_time=12_000.0)
    env_b = _env("baseline.json", sim_time=12_000.0)
    mpc = RollingHorizonMPCAgent(4, params=PARAMS, horizon_k=0, terminal_weight=0.0)
    greedy = GreedyTrainingRewardAgent(4, params=PARAMS)
    a, _ = _rollout(mpc, env_a, seed=5)
    b, _ = _rollout(greedy, env_b, seed=5)
    assert a == b


# ---------------------------------------------------------------------------
# Toy optima
# ---------------------------------------------------------------------------


class _StubRequest:
    """A queued ticket of a chosen failure type (planner inputs only)."""

    def __init__(self, mtype: str, ctype: str, base: float) -> None:
        self.machine = SimpleNamespace(mtype=mtype, machine_id=-1)
        self.created_at = 0
        self.chosen_technician_id = None
        self._ctype, self._base = ctype, base

    def get_repair_time(self) -> float:
        return self._base

    def get_failed_component_info(self) -> dict:
        return {"component_id": -1, "component_type": self._ctype, "repair_time": self._base}

    def get_knowledge_parameters(self):
        return (None, None)


def _far_failure_type(env, request):
    """A failure type of the embedding whose grid cell is far from ``request``."""
    enc = env.dispatcher.techs[0].encoder
    here = np.asarray(_cell(env, request))
    g = env.dispatcher.techs[0].knowledge_grid
    for key, coords in sorted(enc._table.items()):
        cell = np.asarray(g.embedding_to_coords(np.asarray(coords)))
        if np.abs(cell - here).max() >= 6:
            mtype, ctype = key.split(":", 1)
            return mtype, ctype
    pytest.skip("no distant failure type in the embedding")


@pytest.mark.parametrize("cls", [GreedyTrainingRewardAgent, RollingHorizonMPCAgent])
def test_prefers_knowledge_matched_technician(cls):
    env = _env("baseline.json", sim_time=5_000.0)
    _first_decision(env)
    _uniform_fleet(env)
    current = env.current_request
    # Two queued tickets of a distant failure type that every technician
    # knows equally well: the window holds no second ticket of the current
    # type, so reserving the specialist for later cannot pay.
    far = _far_failure_type(env, current)
    queued = [_StubRequest(*far, base=float(current.get_repair_time())) for _ in range(2)]
    env.dispatcher.repair_queue.items.extend(queued)
    matched = 2
    for i in range(len(env.dispatcher.techs)):
        _set_experience(env, i, current, 400.0 if i == matched else 30.0)
        _set_experience(env, i, queued[0], 50.0)
    agent = cls(4, params=PARAMS)  # MPC: K=2, terminal value on
    agent.attach_env(env)
    agent.on_episode_start()
    assert agent.select_action(_obs(env), deterministic=True) == matched
    for q in queued:
        env.dispatcher.repair_queue.items.remove(q)


@pytest.mark.parametrize("cls", [GreedyTrainingRewardAgent, RollingHorizonMPCAgent])
def test_avoids_fatigued_technician(cls):
    env = _env("baseline.json", sim_time=5_000.0)
    _first_decision(env)
    _uniform_fleet(env, fatigue=[0.9, 0.8, 0.05, 0.85])
    for i in range(4):  # same experience everywhere: only fatigue differs
        _set_experience(env, i, env.current_request, 20.0)
    agent = cls(4, params=PARAMS)
    agent.attach_env(env)
    agent.on_episode_start()
    assert agent.select_action(_obs(env), deterministic=True) == 2


def test_lookahead_keeps_the_specialist_for_the_queued_ticket():
    env = _env("baseline.json", sim_time=5_000.0)
    _first_decision(env)
    _uniform_fleet(env)
    current = env.current_request
    base = float(current.get_repair_time())
    queued = _StubRequest(*_far_failure_type(env, current), base=base)
    env.dispatcher.repair_queue.items.append(queued)
    techs = env.dispatcher.techs
    for t in techs[2:]:
        t._in_disruption = True  # only technicians 0 and 1 can work now
    # Tech 0: slightly better on the current ticket, the only one who
    # knows the queued ticket.  Tech 1: slightly worse on the current one.
    _set_experience(env, 0, current, 12.0)
    _set_experience(env, 1, current, 10.0)
    _set_experience(env, 0, queued, 200.0)
    # Knowledge credit and terminal value off: the test isolates the search.
    params = {"sigma": {**SIGMA, "knowledge_increment": 1e12,
                        "terminal_fleet_knowledge": 1e12}}
    obs = _obs(env)
    greedy = GreedyTrainingRewardAgent(4, params=params)
    greedy.attach_env(env)
    greedy.on_episode_start()
    assert greedy.select_action(obs, deterministic=True) == 0
    mpc = RollingHorizonMPCAgent(4, params=params, horizon_k=1, terminal_weight=0.0)
    mpc.attach_env(env)
    mpc.on_episode_start()
    assert mpc.select_action(obs, deterministic=True) == 1
    env.dispatcher.repair_queue.items.remove(queued)


# ---------------------------------------------------------------------------
# Human-state model against the simulator
# ---------------------------------------------------------------------------


def test_predicted_repair_matches_simulator_for_idle_starts(monkeypatch):
    env = _env("massive_scale.json", sim_time=6_000.0)
    agent = RollingHorizonMPCAgent(30, params=PARAMS)
    pred, actual = {}, {}
    reader = agent._reader
    orig_commit = type(reader).commit

    def commit(self, snap, request, action, key, mk_i, base):
        orig_commit(self, snap, request, action, key, mk_i, base)
        i, start, rep, Fs = self.last_prediction
        if snap["free"][i] <= 0.0:  # idle start
            request._seq_id = len(pred)
            pred[request._seq_id] = (start, rep, Fs)

    orig_start = GymTechnician.start_repair
    orig_done = GymTechnician.repair_finished

    def start_repair(self, when):
        orig_start(self, when)
        self._seq_start = (float(when), float(self._fatigue))

    def repair_finished(self, request, when):
        rid = getattr(request, "_seq_id", None)
        if rid is not None:
            s, F = self._seq_start
            actual[rid] = (s, float(when) - s - self.travel_time(request.machine), F)
        orig_done(self, request, when)

    monkeypatch.setattr(type(reader), "commit", commit)
    monkeypatch.setattr(GymTechnician, "start_repair", start_repair)
    monkeypatch.setattr(GymTechnician, "repair_finished", repair_finished)
    _rollout(agent, env, seed=2, check_mask=False)
    common = [k for k in pred if k in actual]
    assert len(common) > 200
    err_start = np.asarray([pred[k][0] - actual[k][0] for k in common])
    err_rep = np.asarray([pred[k][1] - actual[k][1] for k in common])
    err_F = np.asarray([pred[k][2] - actual[k][2] for k in common])
    assert np.median(np.abs(err_start)) < 1e-6
    assert np.median(np.abs(err_rep)) < 1.0  # integer idle truncation only
    assert np.median(np.abs(err_F)) < 0.02


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------


def test_planner_speed_at_thirty_technicians():
    env = _env("massive_scale.json", sim_time=4_000.0)
    assert len(env.dispatcher.techs) == 30
    agent = RollingHorizonMPCAgent(30, params=PARAMS)  # K=2, terminal value on
    _rollout(agent, env, seed=4, check_mask=False)
    ms = agent.planner_ms()
    assert len(ms) > 300
    assert float(np.median(ms[20:])) <= 5.0, float(np.median(ms[20:]))
    assert math.isfinite(float(ms.mean()))


# ---------------------------------------------------------------------------
# Lifecycle: hires and retirements during the episode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [GreedyTrainingRewardAgent, RollingHorizonMPCAgent])
def test_lifecycle_hires_and_retirements(cls):
    from kata.core.config import LifecycleEventConfig

    cfg = KATAConfig(**json.loads((SUITE / "baseline.json").read_text()))
    fixed = RandomScenarioSampler(cfg, cfg.randomized_scenario, seed=4321).sample_config()
    events = [
        LifecycleEventConfig(time=1500.0, kind="retire_technician", count=1, select="highest_knowledge"),
        LifecycleEventConfig(time=2500.0, kind="add_technician", count=2, template="trainee"),
    ]
    gym_cfg = cfg.gym.model_copy(update={
        "max_episode_steps": 20_000, "max_sim_time": 8_000.0,
        "observation_representation": "structured", "lifecycle_events": events,
    })
    env = KataEnv(scenario_factory=lambda c=fixed: ScenarioBuilder(c).build(), config=gym_cfg)
    agent = cls(int(gym_cfg.max_techs), params=PARAMS)
    actions, _ = _rollout(agent, env, seed=9)  # checks the mask and the tombstone
    techs = env.dispatcher.techs
    assert len(techs) == 6 and sum(t.retired for t in techs) == 1
    assert agent._reader.n == 6
    assert any(a >= 4 for a in actions)  # the hires were used

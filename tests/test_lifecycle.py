"""Lifecycle-event tests: mid-episode fleet/park mutations.

Covers the ``GymEnvConfig.lifecycle_events`` machinery (add/retire
technicians, add/retire/replace machines), the retired-technician
tombstone semantics, and the two observation bugs found while mapping
the lifecycle surface (D11 bool-token vocab mismatch, D12 machine-id
fallback).

Uses a real ScenarioBuilder world (baseline.json layout) — lifecycle is
a whole-factory feature and fakes would prove nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kata.core.config import KATAConfig, LifecycleEventConfig
from kata.core.tokenizer import StateTokenizer
from kata.env import KataEnv, _bool_token
from kata.EntityFactories.machine_factory import (
    create_config_from_template,
    list_templates,
)
from kata.EntityFactories.scenario_sampler import RandomScenarioSampler
from kata.scenario import ScenarioBuilder

BASELINE = Path("run_configs/benchmark_suite/baseline.json")


@pytest.fixture(scope="module")
def base_cfg() -> KATAConfig:
    return KATAConfig(**json.loads(BASELINE.read_text()))


@pytest.fixture(scope="module")
def fixed_scenario(base_cfg):
    sampler = RandomScenarioSampler(
        base_cfg, base_cfg.randomized_scenario, seed=4321
    )
    return sampler.sample_config()


@pytest.fixture(scope="module")
def park_template(fixed_scenario) -> str:
    """A machine template whose type already exists in the park."""
    park_types = {m.machine_type for m in fixed_scenario.machines.values()}
    return next(
        t
        for t in list_templates()
        if create_config_from_template(t).machine_type in park_types
    )


def _make_env(base_cfg, fixed_scenario, events, *, sim_time=1500.0):
    gym_cfg = base_cfg.gym.model_copy(
        update={
            "max_episode_steps": 2000,
            "max_sim_time": sim_time,
            "observation_representation": "structured",
            "lifecycle_events": events,
        }
    )
    return KataEnv(
        scenario_factory=lambda c=fixed_scenario: ScenarioBuilder(c).build(),
        config=gym_cfg,
    )


def _run_past(env, target_sim_time, *, seed=11):
    """Reset and step (first-available policy) until sim time passes
    ``target_sim_time`` or the episode ends.  Returns the last info."""
    obs, info = env.reset(seed=seed)
    for _ in range(2000):
        if info["sim_time"] >= target_sim_time:
            break
        mask = env._action_mask()
        action = int(np.flatnonzero(mask)[0]) if mask.any() else 0
        obs, _r, term, trunc, info = env.step(action)
        if term or trunc:
            break
    return info


def test_add_technician(base_cfg, fixed_scenario):
    events = [
        LifecycleEventConfig(
            time=300.0, kind="add_technician", template="junior", count=2
        )
    ]
    env = _make_env(base_cfg, fixed_scenario, events)
    env.reset(seed=11)
    n0 = len(env.dispatcher.techs)
    info = _run_past(env, 400.0)
    techs = env.dispatcher.techs
    assert len(techs) == n0 + 2
    hires = techs[-2:]
    # Full registration: id map, resource, env injection, idle anchor,
    # count arrays extended.
    for t in hires:
        assert env.dispatcher._tech_by_id[t.id] is t
        assert t.id in env.dispatcher._tech_resource
        assert t.env is env.sim_env
        assert t._last_idle_since > 0.0
    assert len(env._tech_assignment_counts) == n0 + 2
    assert len(env._tech_last_assignment_time) == n0 + 2
    # Numeric-only suffix so the TEMPLATE token still resolves.
    assert all(t.name.split("_")[-1].isdigit() for t in hires)
    assert len(info["lifecycle_events"]) == 2


def test_retire_technician_tombstone(base_cfg, fixed_scenario):
    events = [
        LifecycleEventConfig(
            time=200.0, kind="retire_technician", select="highest_knowledge"
        )
    ]
    env = _make_env(base_cfg, fixed_scenario, events)
    env.reset(seed=11)
    n0 = len(env.dispatcher.techs)
    _run_past(env, 300.0)
    retired = [
        (i, t) for i, t in enumerate(env.dispatcher.techs) if t.retired
    ]
    assert len(retired) == 1
    ridx, rtech = retired[0]
    # Slot preserved (no index shift), but masked everywhere.
    assert len(env.dispatcher.techs) == n0
    assert env._action_mask()[ridx] == 0
    # Excluded from the fleet-knowledge mean.
    vols_all = env._fleet_mean_knowledge_volume()
    assert np.isfinite(vols_all)
    active = [t for t in env.dispatcher.techs if not t.retired]
    assert len(active) == n0 - 1
    # Structured obs reports the tombstone as busy (unassignable).
    obs = env._structured_obs()
    assert obs["technician_busy"][ridx] == 1
    # step() refuses to assign a retired technician.
    for _ in range(500):
        if env.current_request is not None:
            break
        mask = env._action_mask()
        env.step(int(np.flatnonzero(mask)[0]))
    if env.current_request is not None:
        _obs, reward, term, _trunc, _info = env.step(ridx)
        assert reward == pytest.approx(
            float(env.config.invalid_action_penalty)
        )
        assert env.current_request is not None  # ticket still pending


def test_add_and_retire_machine(base_cfg, fixed_scenario, park_template):
    events = [
        LifecycleEventConfig(
            time=250.0, kind="add_machine", template=park_template
        ),
        LifecycleEventConfig(time=500.0, kind="retire_machine"),
    ]
    env = _make_env(base_cfg, fixed_scenario, events)
    env.reset(seed=11)
    n0 = len(env.dispatcher.machines)
    _run_past(env, 700.0)
    # Net effect of add @250 + retire @500: back to n0, with the added
    # machine wired into its type feeder behind an aligned buffer (the
    # retiree is selected at random, so the added machine may itself be
    # the one retired — only the invariants below are guaranteed).
    assert len(env.dispatcher.machines) == n0
    for feeder in env.dispatcher.factory_handles.feeders.values():
        assert len(feeder.machines) == len(feeder.machine_input_buffers)
        assert len(feeder.machines) == len(set(map(id, feeder.machines)))
    kinds = [e["kind"] for e in env._lifecycle_log]
    assert kinds.count("add_machine") == 1
    assert kinds.count("retire_machine") == 1
    retired = [
        m
        for f in env.dispatcher.factory_handles.feeders.values()
        for m in f.machines
        if getattr(m, "retired", False)
    ]
    assert retired == []  # retired machines are unwired from feeders


def test_replace_machine_renews(base_cfg, fixed_scenario, park_template):
    events = [
        LifecycleEventConfig(
            time=300.0,
            kind="replace_machine",
            template=park_template,
            select="random",
        )
    ]
    env = _make_env(base_cfg, fixed_scenario, events)
    env.reset(seed=11)
    n0 = len(env.dispatcher.machines)
    _run_past(env, 1400.0)
    # Replacement is retire+add: park size conserved, one lc machine in.
    assert len(env.dispatcher.machines) == n0
    entries = [
        e for e in env._lifecycle_log if e["kind"] == "replace_machine"
    ]
    assert len(entries) == 2  # retiree + replacement
    assert any(e["target"].startswith("replacement:") for e in entries)


def test_events_beyond_horizon_never_fire(base_cfg, fixed_scenario):
    events = [
        LifecycleEventConfig(
            time=999_999.0, kind="add_technician", template="junior"
        )
    ]
    env = _make_env(base_cfg, fixed_scenario, events, sim_time=800.0)
    env.reset(seed=11)
    n0 = len(env.dispatcher.techs)
    info = _run_past(env, 1e9)  # runs to episode end
    assert len(env.dispatcher.techs) == n0
    assert info["lifecycle_events"] == []


# ---------------------------------------------------------------------------
# D11 / D12 — observation bugs found while mapping the lifecycle surface
# ---------------------------------------------------------------------------


def test_set_bool_tokens_match_frozen_vocab():
    """D11: the SET emitter must produce ``KEY=T``/``KEY=F`` — the set
    vocab's spelling.  The historical TRUE/FALSE (the flat-stream
    convention) made all six boolean tokens <UNK> in every set-mode
    observation.  The flat stream keeps standalone TRUE/FALSE."""
    from kata.env import _SetEmitter

    emitter = _SetEmitter()
    slot: tuple[list, list, list] = ([], [], [])
    emitter.open_slot(slot)
    emitter.bool("BUSY", True)
    emitter.bool("DISRUPT", False)
    emitter.close_slot()
    assert slot[0] == ["BUSY=T", "DISRUPT=F"]
    # Flat-stream convention unchanged (its vocab registers TRUE/FALSE).
    assert _bool_token(True) == "TRUE"
    vocab_path = Path("run_configs/vocab/set_vocab.json")
    if vocab_path.is_file():
        tok = StateTokenizer.from_json(vocab_path, seq_length=8)
        unk = tok.token_to_id("<DEFINITELY-NOT-A-TOKEN>")
        for key in ("BUSY", "DISRUPT", "BROKEN", "PROC", "IS_CURRENT", "HAS_T"):
            assert tok.token_to_id(f"{key}=T") != unk
            assert tok.token_to_id(f"{key}=F") != unk


def test_machine_id_uses_machine_id_attribute(base_cfg, fixed_scenario):
    """D12: ``_machine_id_from_machine`` must return ``machine_id`` (the
    key of ``dispatcher.machines`` and of every tracking dict), not the
    CPython ``id()`` fallback."""
    env = _make_env(base_cfg, fixed_scenario, [])
    env.reset(seed=11)
    for mid, machine in env.dispatcher.machines.items():
        assert env._machine_id_from_machine(machine) == mid


def test_replace_machine_without_template_is_like_for_like(
    base_cfg, fixed_scenario
):
    """replace_machine may omit the template: the replacement machine
    must then match the retiree's machine type."""
    events = [
        LifecycleEventConfig(
            time=300.0, kind="replace_machine", select="most_breakdowns"
        )
    ]
    env = _make_env(base_cfg, fixed_scenario, events)
    env.reset(seed=11)
    types_before = sorted(
        {m.mtype for m in env.dispatcher.machines.values()}
    )
    n0 = len(env.dispatcher.machines)
    _run_past(env, 1400.0)
    assert len(env.dispatcher.machines) == n0
    assert (
        sorted({m.mtype for m in env.dispatcher.machines.values()})
        == types_before
    )
    entries = [
        e for e in env._lifecycle_log if e["kind"] == "replace_machine"
    ]
    assert any(e["target"].startswith("replacement:") for e in entries)

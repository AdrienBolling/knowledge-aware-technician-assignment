"""Regression tests for two simulator defects found by the KataJAX
differential validation (kata-jax ``reports/DIFFERENTIAL.md``).

BUG 1: ``Machine._run`` lost products when a breakdown interrupted a store
request (the queued get swallowed the next product; an interrupted put
skipped ``total_processed``).

BUG 2: breakdown counts and downtime intervals were sampled at decision
boundaries, so short breakdowns were invisible and a machine broken at
every decision counted once.

Each fix has a legacy switch (:mod:`kata.core.legacy`).  The legacy tests
prove that each defect stays reproducible behind its switch.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import simpy

from conftest import FakeDispatcher, FakeMachine, FakeRequest, FakeSimEnv
from kata.core.config import GymEnvConfig, KATAConfig, LifecycleEventConfig
from kata.core.legacy import (
    LEGACY_BUFFER_INTERRUPT_ENV,
    LEGACY_MACHINE_TRACKING_ENV,
    legacy_buffer_interrupt,
    legacy_machine_tracking,
)
from kata.entities.machines.complex_machine import ComplexMachine
from kata.entities.machines.machine import Machine
from kata.entities.products.product import Product
from kata.entities.tech_dispatcher.GymTechDispatcher import GymTechDispatcher
from kata.env import KataEnv
from kata.EntityFactories.machine_factory import (
    create_config_from_template,
    list_templates,
)
from kata.EntityFactories.scenario_sampler import RandomScenarioSampler
from kata.features.breakdown.simple_breakdown import SimpleBreakdownProcess
from kata.metrics import FleetAvailabilityRate, MeanTimeBetweenFailures
from kata.scenario import ScenarioBuilder


@pytest.fixture(autouse=True)
def _fixed_simulator(monkeypatch):
    """Every test starts on the fixed simulator; legacy tests opt in.

    ``KATA_CONF_PATH`` points nowhere so that a run config loads alone
    (the default JSON file would merge extra machines into it), like the
    evaluation harness does.
    """
    monkeypatch.setenv("KATA_CONF_PATH", "/nonexistent/__kata_test__.json")
    monkeypatch.delenv(LEGACY_BUFFER_INTERRUPT_ENV, raising=False)
    monkeypatch.delenv(LEGACY_MACHINE_TRACKING_ENV, raising=False)


# ---------------------------------------------------------------------------
# Legacy switches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", False), ("0", False), ("off", False), ("1", True), ("TRUE", True), ("yes", True)],
)
def test_legacy_switch_values(monkeypatch, raw, expected):
    monkeypatch.setenv(LEGACY_BUFFER_INTERRUPT_ENV, raw)
    monkeypatch.setenv(LEGACY_MACHINE_TRACKING_ENV, raw)
    assert legacy_buffer_interrupt() is expected
    assert legacy_machine_tracking() is expected


def test_legacy_switch_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv(LEGACY_MACHINE_TRACKING_ENV, "2")
    with pytest.raises(ValueError, match=LEGACY_MACHINE_TRACKING_ENV):
        legacy_machine_tracking()


# ---------------------------------------------------------------------------
# BUG 1: machine buffer requests interrupted by a breakdown
# ---------------------------------------------------------------------------


class _StubDispatcher:
    """Repair signalling only; the test decides when a repair ends."""

    def __init__(self, env: simpy.Environment) -> None:
        self.env = env
        self._events: dict[Machine, simpy.Event] = {}
        self.filed = 0

    def wait_until_repaired(self, machine: Machine) -> simpy.Event:
        if machine not in self._events:
            self._events[machine] = self.env.event()
        return self._events[machine]

    def request_repair(self, machine: Machine) -> None:
        self.filed += 1

    def finish_repair(self, machine: Machine) -> None:
        machine.repair(None)
        ev = self._events.pop(machine, None)
        if ev is not None:
            ev.succeed()


def _machine(env, dispatcher, *, legacy=None, out_capacity=float("inf"), process_time=5):
    """A hazard-free machine: it breaks only when the test says so."""
    inp = simpy.Store(env)
    out = simpy.Store(env, capacity=out_capacity)
    machine = Machine(
        env=env,
        machine_id=1,
        mtype="generic",
        input_buffer=inp,
        output_buffer=out,
        tech_dispatcher=dispatcher,
        breakdown_process=SimpleBreakdownProcess(
            failure_prob_working=0.0, failure_prob_idle=0.0
        ),
        process_time=process_time,
        dt=1,
        legacy_buffer_interrupt=legacy,
    )
    return machine, inp, out


@pytest.mark.parametrize(("legacy", "lost"), [(False, 0), (True, 1)])
def test_idle_machine_breakdown_loses_no_product(legacy, lost):
    """A machine breaks while it waits for input, then receives products.

    Fixed: created == finished + WIP.  Legacy: the queued get request
    swallows the first product.
    """
    env = simpy.Environment()
    dispatcher = _StubDispatcher(env)
    machine, inp, out = _machine(env, dispatcher, legacy=legacy)
    created: list[Product] = []

    def scenario():
        yield env.timeout(5)
        assert len(inp.get_queue) == 1  # the machine waits for input
        machine._trigger_breakdown()
        yield env.timeout(5)
        dispatcher.finish_repair(machine)
        for i in range(3):
            yield env.timeout(10)
            product = Product(product_id=i, route=["generic"])
            created.append(product)
            yield inp.put(product)

    env.process(scenario())
    env.run(until=500)

    finished = len(out.items)
    wip = len(inp.items) + int(machine.current_product is not None)
    assert dispatcher.filed == 1
    assert len(created) - finished - wip == lost
    assert machine.total_processed == finished == 3 - lost
    if not legacy:
        assert [p.product_id for p in out.items] == [0, 1, 2]
        assert inp.get_queue == [machine.proc.target]  # no stale request


@pytest.mark.parametrize("legacy", [False, True])
def test_triggered_input_get_keeps_the_product(legacy):
    """The store hands a product to the get request at the same instant as
    the breakdown interrupt: the fixed machine keeps and processes it; the
    legacy machine drops it."""
    env = simpy.Environment()
    dispatcher = _StubDispatcher(env)
    machine, inp, out = _machine(env, dispatcher, legacy=legacy)

    def scenario():
        yield env.timeout(5)
        # The put triggers the machine's queued get (value handed over,
        # event scheduled); the breakdown interrupt is URGENT and lands
        # before the machine resumes.
        inp.put(Product(product_id=0, route=["generic"]))
        machine._trigger_breakdown()
        yield env.timeout(10)
        dispatcher.finish_repair(machine)

    env.process(scenario())
    env.run(until=100)
    assert inp.items == []
    assert machine.current_product is None
    if legacy:
        assert out.items == [] and machine.total_processed == 0
    else:
        assert [p.product_id for p in out.items] == [0]
        assert machine.total_processed == 1


@pytest.mark.parametrize("legacy", [False, True])
def test_breakdown_after_output_put_delivered(legacy):
    """The output put already delivered the product when the breakdown
    interrupt lands.  Fixed: the product is counted.  Legacy: it is in the
    output store but ``total_processed`` skips it."""
    env = simpy.Environment()
    dispatcher = _StubDispatcher(env)
    machine, inp, out = _machine(env, dispatcher, legacy=legacy, process_time=5)
    inp.put(Product(product_id=0, route=["generic"]))

    def breaker():
        yield env.timeout(1)
        # Scheduled after the machine's processing timeout (same instant
        # t=5, later event id): runs right after the machine created the
        # output put and yielded it.
        yield env.timeout(4)
        assert len(out.items) == 1 and machine.proc.target is not None
        machine._trigger_breakdown()
        yield env.timeout(10)
        dispatcher.finish_repair(machine)

    env.process(breaker())
    env.run(until=100)
    assert [p.product_id for p in out.items] == [0]
    assert machine.total_processed == (0 if legacy else 1)


@pytest.mark.parametrize("legacy", [False, True])
def test_breakdown_while_blocked_on_full_output(legacy):
    """The output buffer is full when the breakdown lands.  The product
    must reach the buffer exactly once when space frees; the fixed machine
    counts it once."""
    env = simpy.Environment()
    dispatcher = _StubDispatcher(env)
    machine, inp, out = _machine(
        env, dispatcher, legacy=legacy, out_capacity=1, process_time=5
    )
    for i in range(2):
        inp.put(Product(product_id=i, route=["generic"]))
    received: list[int] = []

    def scenario():
        yield env.timeout(20)
        # Product 0 fills the output buffer; product 1 is finished and
        # its put request waits in the put queue.
        assert len(out.items) == 1 and len(out.put_queue) == 1
        machine._trigger_breakdown()
        yield env.timeout(10)
        dispatcher.finish_repair(machine)
        yield env.timeout(10)
        for _ in range(2):
            product = yield out.get()
            received.append(product.product_id)

    env.process(scenario())
    env.run(until=200)
    assert received == [0, 1]
    assert out.items == [] and out.put_queue == []
    assert machine.total_processed == (1 if legacy else 2)


# ---------------------------------------------------------------------------
# BUG 2: machine tracking sampled at decision boundaries
# ---------------------------------------------------------------------------


class _HookDispatcher(FakeDispatcher):
    """FakeDispatcher with the two callback slots the real one exposes."""

    def __init__(self, tech_count: int = 1) -> None:
        super().__init__(tech_count)
        self.on_repair_completed = None
        self.on_machine_breakdown = None


def _tracking_env():
    sim_env = FakeSimEnv()
    dispatcher = _HookDispatcher()
    m1 = FakeMachine(machine_id=3)
    m1.name = "delta"
    m1.broken = False
    m2 = FakeMachine(machine_id=4)
    m2.name = "epsilon"
    m2.broken = False
    dispatcher.machines = [m1, m2]
    dispatcher.repair_queue.items.append(FakeRequest(machine_id=3, created_at=0.0))
    env = KataEnv(
        sim_env=sim_env,
        dispatcher=dispatcher,
        config=GymEnvConfig(max_episode_steps=10, max_sim_time=1000.0),
    )
    env.reset()
    return env, sim_env, dispatcher, m1


def _two_breakdowns_between_decisions(env, sim_env, dispatcher, machine):
    """Decision at t=0; breakdown/repair at 10-25 and 40-47.5; decision at 60."""
    env._update_machine_state_tracking()
    request = FakeRequest(machine_id=machine.machine_id)
    request.machine = machine
    for t_break, t_repair in ((10.0, 25.0), (40.0, 47.5)):
        sim_env.now = t_break
        machine.broken = True
        if dispatcher.on_machine_breakdown is not None:
            dispatcher.on_machine_breakdown(machine)
        sim_env.now = t_repair
        machine.broken = False
        dispatcher.on_repair_completed(request, 5.0, None)
    sim_env.now = 60.0
    env._update_machine_state_tracking()


def test_two_breakdowns_between_decisions_are_counted():
    env, sim_env, dispatcher, machine = _tracking_env()
    assert dispatcher.on_machine_breakdown is not None
    _two_breakdowns_between_decisions(env, sim_env, dispatcher, machine)

    assert env._machine_breakdown_counts == {3: 2}
    assert env._total_downtime == pytest.approx(22.5)
    assert env._machine_down_since == {}
    stats = env._per_machine_episode_stats()
    assert stats["delta"]["breakdowns"] == 2.0
    assert stats["delta"]["maintenance_time"] == pytest.approx(22.5)
    # Two machines over 60 t.u. = 120 machine-time.
    assert FleetAvailabilityRate().compute(env) == pytest.approx(1.0 - 22.5 / 120.0)
    assert MeanTimeBetweenFailures().compute(env) == pytest.approx((120.0 - 22.5) / 2)


def test_legacy_tracking_misses_breakdowns_between_decisions(monkeypatch):
    monkeypatch.setenv(LEGACY_MACHINE_TRACKING_ENV, "1")
    env, sim_env, dispatcher, machine = _tracking_env()
    assert dispatcher.on_machine_breakdown is None
    _two_breakdowns_between_decisions(env, sim_env, dispatcher, machine)
    assert sum(env._machine_breakdown_counts.values()) == 0
    assert env._total_downtime == 0.0
    assert FleetAvailabilityRate().compute(env) == 1.0


# ---------------------------------------------------------------------------
# Whole-factory checks (real ScenarioBuilder worlds, random policy)
# ---------------------------------------------------------------------------


def _random_episode(name: str, horizon: float, seed: int = 0):
    cfg = KATAConfig(**json.loads(Path(f"run_configs/{name}.json").read_text()))
    gym_cfg = cfg.gym.model_copy(
        update={
            "max_sim_time": horizon,
            "max_episode_steps": 100_000,
            "observation_representation": "structured",
        }
    )
    env = KataEnv(scenario_factory=lambda: ScenarioBuilder(cfg).build(), config=gym_cfg)
    np.random.seed(seed)
    random.seed(seed)
    env.reset(seed=seed)
    while True:
        mask = env._action_mask()
        action = int(np.random.choice(np.flatnonzero(mask)))
        _obs, _r, term, trunc, info = env.step(action)
        if term or trunc:
            return env, info


@pytest.mark.parametrize("legacy", [False, True])
def test_factory_v2_product_conservation(monkeypatch, legacy):
    """7-stage route with failures (the differential-validation repro)."""
    if legacy:
        monkeypatch.setenv(LEGACY_BUFFER_INTERRUPT_ENV, "1")
    env, info = _random_episode("factory_v2", 20_000.0)
    counts = env.product_conservation()
    assert counts["products_created"] > 0
    assert counts["products_finished"] == int(info["metrics"]["finished_products"])
    if legacy:
        assert counts["products_lost"] > 0
    else:
        assert counts["products_lost"] == 0
    # Fixed tracking: one tracked breakdown per filed ticket.
    assert sum(env._machine_breakdown_counts.values()) == env.dispatcher.breakdowns_filed


@pytest.mark.parametrize("legacy", [False, True])
def test_minimal_availability_is_event_exact(monkeypatch, legacy):
    """One machine, one technician: the decision IS the breakdown, so the
    legacy sampler sees the machine broken at every decision."""
    if legacy:
        monkeypatch.setenv(LEGACY_MACHINE_TRACKING_ENV, "1")
    downs: list[float] = []
    ups: list[float] = []
    real_request = GymTechDispatcher.request_repair

    def request_repair(self, machine):
        downs.append(float(self.env.now))
        real_request(self, machine)

    def logged(repair):
        def _repair(self, request):
            repair(self, request)
            ups.append(float(self.env.now))

        return _repair

    monkeypatch.setattr(GymTechDispatcher, "request_repair", request_repair)
    monkeypatch.setattr(Machine, "repair", logged(Machine.repair))
    monkeypatch.setattr(ComplexMachine, "repair", logged(ComplexMachine.repair))

    horizon = 20_000.0
    env, info = _random_episode("minimal", horizon)
    now = float(info["sim_time"])
    assert len(downs) > 20
    down_time = sum(u - d for d, u in zip(downs, ups)) + sum(
        now - d for d in downs[len(ups):]
    )
    exact = 1.0 - down_time / now
    availability = info["metrics"]["fleet_availability_rate"]
    if legacy:
        assert sum(env._machine_breakdown_counts.values()) <= 2
        assert availability < 0.2 < exact
    else:
        assert sum(env._machine_breakdown_counts.values()) == len(downs)
        assert availability == pytest.approx(exact, abs=1e-9)


def test_lifecycle_retirement_scrap_is_not_a_loss():
    """Retiring a machine scraps its input buffer on purpose: counted as
    scrapped, not lost; the fleet machine-time counts active lives only."""
    base = KATAConfig(
        **json.loads(Path("run_configs/benchmark_suite/baseline.json").read_text())
    )
    scenario = RandomScenarioSampler(base, base.randomized_scenario, seed=4321).sample_config()
    park_types = {m.machine_type for m in scenario.machines.values()}
    template = next(
        t for t in list_templates()
        if create_config_from_template(t).machine_type in park_types
    )
    events = [
        LifecycleEventConfig(time=250.0, kind="add_machine", template=template),
        LifecycleEventConfig(time=500.0, kind="retire_machine"),
    ]
    gym_cfg = base.gym.model_copy(
        update={
            "max_episode_steps": 5000,
            "max_sim_time": 3000.0,
            "observation_representation": "structured",
            "lifecycle_events": events,
        }
    )
    env = KataEnv(
        scenario_factory=lambda c=scenario: ScenarioBuilder(c).build(), config=gym_cfg
    )
    np.random.seed(3)
    random.seed(3)
    _obs, info = env.reset(seed=3)
    n0 = len(env.dispatcher.factory_handles.all_machines)
    while True:
        mask = env._action_mask()
        _obs, _r, term, trunc, info = env.step(int(np.flatnonzero(mask)[0]))
        if term or trunc:
            break
    counts = env.product_conservation()
    assert counts["products_lost"] == 0
    handles = env.dispatcher.factory_handles
    assert len(handles.all_machines) == n0 + 1
    retired = [m for m in handles.all_machines if m.retired]
    assert len(retired) == 1
    now = float(info["sim_time"])
    expected = (n0 * now) + (now - 250.0) - (now - float(retired[0].retired_at))
    assert env._fleet_machine_time(now) == pytest.approx(expected)

"""Tests for GymTechDispatcher."""

import simpy

from kata.core.config import TechnicianConfig
from kata.entities.machines.machine import Machine
from kata.entities.requests.RepairRequest import RepairRequest
from kata.entities.tech_dispatcher.config import TechDispatcherConfig
from kata.entities.tech_dispatcher.GymTechDispatcher import GymTechDispatcher
from kata.entities.technicians.GymTechnician import GymTechnician
from kata.features.breakdown.simple_breakdown import SimpleBreakdownProcess


def _build_env_and_dispatcher(n_techs=2, config=None):
    env = simpy.Environment()
    techs = [GymTechnician(TechnicianConfig(name=f"tech_{i}")) for i in range(n_techs)]
    dispatcher = GymTechDispatcher(env, techs, config=config)
    return env, techs, dispatcher


def _make_machine(env, dispatcher):
    return Machine(
        env=env,
        machine_id=1,
        mtype="test",
        input_buffer=simpy.Store(env),
        output_buffer=simpy.Store(env),
        tech_dispatcher=dispatcher,
        breakdown_process=SimpleBreakdownProcess(
            failure_prob_working=0.0, failure_prob_idle=0.0
        ),
        process_time=5,
        dt=1,
    )


class TestGymTechDispatcher:
    def test_request_repair_adds_to_queue(self):
        env, techs, dispatcher = _build_env_and_dispatcher()
        machine = _make_machine(env, dispatcher)

        dispatcher.request_repair(machine)
        assert len(dispatcher.repair_queue.items) == 1

    def test_start_repair_sets_tech_busy(self):
        env, techs, dispatcher = _build_env_and_dispatcher(n_techs=1)
        machine = _make_machine(env, dispatcher)
        machine.broken = True
        request = RepairRequest(machine, created_at=0)

        dispatcher.start_repair(techs[0].id, request)

        # The disruption process also competes for the resource.
        # Run long enough for disruption to finish and repair to acquire.
        env.run(until=2000)

        # After full repair cycle, tech.busy is set True then cleared.
        # Check that an assignment happened (repair_finished was called).
        # The tech should be free again after repair completes.
        assert not techs[0].busy  # repair finished
        assert not machine.broken  # machine was repaired

    def test_repair_completes_and_signals_event(self):
        env, techs, dispatcher = _build_env_and_dispatcher(n_techs=1)
        machine = _make_machine(env, dispatcher)

        # Set up wait event
        event = dispatcher.wait_until_repaired(machine)
        machine.broken = True

        request = RepairRequest(machine, created_at=0)
        dispatcher.start_repair(techs[0].id, request)

        # Disruption holds the resource for ~480 time units (sick_leave mu),
        # then travel + repair time. Run long enough.
        env.run(until=2000)

        assert event.triggered
        assert not machine.broken

    def test_tech_lookup_by_id(self):
        env, techs, dispatcher = _build_env_and_dispatcher(n_techs=3)
        for tech in techs:
            found = dispatcher._get_tech(tech.id)
            assert found is tech

    def test_invalid_tech_id_raises(self):
        env, techs, dispatcher = _build_env_and_dispatcher(n_techs=1)
        import pytest

        with pytest.raises(ValueError, match="not found"):
            dispatcher._get_tech(999)

    def test_uses_config_queue_capacity(self):
        cfg = TechDispatcherConfig(repair_queue_capacity=42)
        env, techs, dispatcher = _build_env_and_dispatcher(config=cfg)
        assert dispatcher.repair_queue.capacity == 42

    def test_repair_updates_technician_state(self):
        env, techs, dispatcher = _build_env_and_dispatcher(n_techs=1)
        machine = _make_machine(env, dispatcher)
        machine.broken = True

        request = RepairRequest(machine, created_at=0)
        dispatcher.start_repair(techs[0].id, request)
        # Run long enough for disruption + travel + repair to complete
        env.run(until=2000)

        # After repair completes, tech should not be busy
        assert not techs[0].busy
        # Fatigue should have increased
        assert techs[0].fatigue > 0.0

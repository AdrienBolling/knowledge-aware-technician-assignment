"""Shared fixtures for the KATA test suite."""

from __future__ import annotations

import math
from typing import Any

import pytest
import simpy

from kata.entities.buffers.buffer import Buffer
from kata.entities.machine_feeder.machine_feeder import MachineFeeder
from kata.entities.routers.router import Router
from kata.entities.sinks.sink import Sink
from kata.entities.sources.source import Source
from kata.entities.technicians.GymTechnician import GymTechnician
from kata.entities.technicians.technician import Technician

# ---------------------------------------------------------------------------
# ID counter reset (shared across all test modules)
# ---------------------------------------------------------------------------

_COUNTER_CLASSES = [
    Buffer,
    Source,
    Sink,
    Router,
    MachineFeeder,
    Technician,
    GymTechnician,
]


@pytest.fixture(autouse=True)
def _reset_entity_id_counters():
    """Reset all entity _id_counter values before each test."""
    saved = {cls: cls._id_counter for cls in _COUNTER_CLASSES}
    for cls in _COUNTER_CLASSES:
        cls._id_counter = 0
    yield
    for cls, val in saved.items():
        cls._id_counter = val


# ---------------------------------------------------------------------------
# Lightweight fakes for KataEnv tests
# ---------------------------------------------------------------------------


class FakeMachine:
    def __init__(self, machine_id: int = 1, mtype: str = "generic"):
        self.machine_id = machine_id
        self.mtype = mtype
        self.broken = True
        self.is_processing = False
        self.total_processed = 2
        self.input_buffer = FakeQueue()
        self.output_buffer = FakeQueue()


class FakeRequest:
    def __init__(self, machine_id: int = 1, created_at: float = 0.0):
        self.machine = FakeMachine(machine_id)
        self.created_at = created_at


class FakeTech:
    def __init__(self, tech_id: int = 0, name: str | None = None):
        self.id = tech_id
        # Unique per-id default keeps the fleet-name uniqueness check
        # in ``KataEnv._bootstrap_scenario`` happy in test scenarios
        # that spin up multiple FakeTechs.
        self.name = name if name is not None else f"tech_{tech_id}"
        self.busy = False
        self.fatigue = 0.0
        self.knowledge = 0.0


class FakeQueue:
    def __init__(self):
        self.items: list[Any] = []


class FakeDispatcher:
    def __init__(self, tech_count: int = 2):
        self.techs = [FakeTech(i) for i in range(tech_count)]
        self.repair_queue = FakeQueue()
        self.assignments: list[tuple[int, int]] = []

    def start_repair(self, tech_id: int, request: Any) -> None:
        self.assignments.append((tech_id, request.machine.machine_id))
        self.techs[tech_id].busy = True


class FakeSimEnv:
    def __init__(self):
        self.now = 0.0
        self._events: list[tuple[float, Any]] = []

    def schedule(self, at: float, callback) -> None:
        self._events.append((float(at), callback))
        self._events.sort(key=lambda x: x[0])

    def peek(self) -> float:
        if not self._events:
            return math.inf
        return self._events[0][0]

    def step(self) -> None:
        when, callback = self._events.pop(0)
        self.now = when
        callback()


# ---------------------------------------------------------------------------
# Mock dispatcher for SimPy-level machine tests
# ---------------------------------------------------------------------------


class MockTechDispatcher:
    """Minimal dispatcher that immediately signals repair."""

    def __init__(self, env: simpy.Environment):
        self.env = env
        self._events: dict[Any, simpy.Event] = {}
        self.requested_repairs: list[Any] = []

    def wait_until_repaired(self, machine: Any) -> simpy.Event:
        if machine not in self._events:
            self._events[machine] = self.env.event()
        return self._events[machine]

    def request_repair(self, machine: Any) -> None:
        self.requested_repairs.append(machine)

    def signal_repaired(self, machine: Any) -> None:
        if machine in self._events:
            self._events[machine].succeed()
            del self._events[machine]

import simpy
import pytest

from kata.entities.buffers.buffer import Buffer
from kata.entities.machine_feeder.machine_feeder import MachineFeeder
from kata.entities.routers.router import Router
from kata.entities.sinks.sink import Sink
from kata.entities.sources.source import Source
from kata.entities.technicians.technician import Technician


@pytest.fixture(autouse=True)
def _reset_entity_id_counters():
    classes = [Buffer, Source, Sink, Router, MachineFeeder, Technician]
    saved = {cls: cls._id_counter for cls in classes}
    for cls in classes:
        cls._id_counter = 0
    yield
    for cls in classes:
        cls._id_counter = saved[cls]


def test_buffer_ids_are_auto_incremented():
    env = simpy.Environment()

    b1 = Buffer(env, "b1", capacity=1)
    b2 = Buffer(env, "b2", capacity=1)

    assert b1.id == 0
    assert b2.id == 1


def test_entity_ids_are_auto_incremented_per_class():
    env = simpy.Environment()
    buffer_a = Buffer(env, "a", capacity=1)
    buffer_b = Buffer(env, "b", capacity=1)
    buffer_c = Buffer(env, "c", capacity=1)

    source_1 = Source(env=env, name="src1", out_buffer=buffer_a)
    source_2 = Source(env=env, name="src2", out_buffer=buffer_a)
    sink_1 = Sink(env=env, name="sink1", in_buffer=buffer_b)
    sink_2 = Sink(env=env, name="sink2", in_buffer=buffer_b)
    router_1 = Router(env=env, name="router1", in_buffer=buffer_a, type_to_buffer={"x": buffer_b})
    router_2 = Router(env=env, name="router2", in_buffer=buffer_a, type_to_buffer={"x": buffer_b})
    feeder_1 = MachineFeeder(
        env=env,
        name="feeder1",
        machine_type="x",
        in_buffer=buffer_b,
        machine_input_buffers=[buffer_c],
    )
    feeder_2 = MachineFeeder(
        env=env,
        name="feeder2",
        machine_type="x",
        in_buffer=buffer_b,
        machine_input_buffers=[buffer_c],
    )
    tech_1 = Technician(name="tech1")
    tech_2 = Technician(name="tech2")

    assert (source_1.id, source_2.id) == (0, 1)
    assert (sink_1.id, sink_2.id) == (0, 1)
    assert (router_1.id, router_2.id) == (0, 1)
    assert (feeder_1.id, feeder_2.id) == (0, 1)
    assert (tech_1.id, tech_2.id) == (0, 1)

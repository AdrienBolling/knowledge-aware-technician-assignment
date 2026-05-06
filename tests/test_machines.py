"""Tests for Machine and ComplexMachine SimPy entities."""

import simpy
from conftest import MockTechDispatcher

from kata.entities.components.component import MachineComponent
from kata.entities.machines.complex_machine import ComplexMachine
from kata.entities.machines.machine import Machine
from kata.entities.products.product import Product
from kata.entities.requests.RepairRequest import RepairRequest
from kata.features.breakdown.simple_breakdown import SimpleBreakdownProcess


def _make_simple_machine(env, dispatcher, **kwargs):
    defaults = {
        "env": env,
        "machine_id": 1,
        "mtype": "test",
        "input_buffer": simpy.Store(env),
        "output_buffer": simpy.Store(env),
        "tech_dispatcher": dispatcher,
        "breakdown_process": SimpleBreakdownProcess(failure_prob_working=0.0),
        "process_time": 5,
        "dt": 1,
    }
    defaults.update(kwargs)
    return defaults["input_buffer"], defaults["output_buffer"], Machine(**defaults)


class TestMachine:
    def test_processes_product_and_increments_counter(self):
        env = simpy.Environment()
        disp = MockTechDispatcher(env)
        in_buf, out_buf, machine = _make_simple_machine(env, disp)

        product = Product(product_id=0, route=["test"])
        in_buf.put(product)
        env.run(until=10)

        assert machine.total_processed == 1
        assert len(out_buf.items) == 1

    def test_product_route_is_advanced(self):
        env = simpy.Environment()
        disp = MockTechDispatcher(env)
        in_buf, out_buf, machine = _make_simple_machine(env, disp)

        product = Product(product_id=0, route=["test", "next"])
        in_buf.put(product)
        env.run(until=10)

        finished = out_buf.items[0]
        assert finished.step == 1
        assert finished.next_machine_type() == "next"

    def test_breakdown_triggers_repair_request(self):
        env = simpy.Environment()
        disp = MockTechDispatcher(env)
        # High failure probability to guarantee breakdown
        bp = SimpleBreakdownProcess(failure_prob_working=1.0, failure_prob_idle=1.0)
        in_buf, out_buf, machine = _make_simple_machine(
            env,
            disp,
            breakdown_process=bp,
        )
        in_buf.put(Product(product_id=0, route=["test"]))
        env.run(until=5)

        assert machine.broken
        assert len(disp.requested_repairs) > 0

    def test_repair_resets_broken_flag(self):
        env = simpy.Environment()
        disp = MockTechDispatcher(env)
        in_buf, out_buf, machine = _make_simple_machine(env, disp)

        machine.broken = True
        request = RepairRequest(machine, created_at=0)
        machine.repair(request)

        assert not machine.broken


class TestComplexMachine:
    def _make_complex_machine(self, env, dispatcher, failure_prob=0.0):
        in_buf = simpy.Store(env)
        out_buf = simpy.Store(env)
        components = [
            MachineComponent(
                "motor_0",
                "motor",
                SimpleBreakdownProcess(failure_prob_working=failure_prob),
                base_repair_time=50.0,
            ),
            MachineComponent(
                "bearing_0",
                "bearing",
                SimpleBreakdownProcess(failure_prob_working=failure_prob),
                base_repair_time=30.0,
            ),
        ]
        machine = ComplexMachine(
            env=env,
            machine_id=10,
            mtype="complex_test",
            input_buffer=in_buf,
            output_buffer=out_buf,
            tech_dispatcher=dispatcher,
            components=components,
            process_time=5,
            dt=1,
        )
        return in_buf, out_buf, machine

    def test_creation_with_components(self):
        env = simpy.Environment()
        disp = MockTechDispatcher(env)
        _, _, machine = self._make_complex_machine(env, disp)
        assert len(machine.components) == 2
        assert machine.failed_component is None

    def test_processes_product_without_breakdown(self):
        env = simpy.Environment()
        disp = MockTechDispatcher(env)
        in_buf, out_buf, machine = self._make_complex_machine(env, disp)

        in_buf.put(Product(product_id=0, route=["complex_test"]))
        env.run(until=10)

        assert machine.total_processed == 1

    def test_component_failure_triggers_breakdown(self):
        env = simpy.Environment()
        disp = MockTechDispatcher(env)
        in_buf, out_buf, machine = self._make_complex_machine(
            env,
            disp,
            failure_prob=1.0,
        )
        in_buf.put(Product(product_id=0, route=["complex_test"]))
        env.run(until=5)

        assert machine.broken
        assert machine.failed_component is not None

    def test_repair_resets_failed_component(self):
        env = simpy.Environment()
        disp = MockTechDispatcher(env)
        _, _, machine = self._make_complex_machine(env, disp)

        # Simulate component failure
        machine.failed_component = machine.components[0]
        machine.broken = True

        request = RepairRequest(machine, created_at=0)
        machine.repair(request)

        assert not machine.broken
        assert machine.failed_component is None

    def test_repair_request_gets_component_info(self):
        env = simpy.Environment()
        disp = MockTechDispatcher(env)
        _, _, machine = self._make_complex_machine(env, disp)

        machine.failed_component = machine.components[0]
        request = RepairRequest(machine, created_at=100)

        info = request.get_failed_component_info()
        assert info is not None
        assert info["component_id"] == "motor_0"
        assert info["component_type"] == "motor"
        assert info["repair_time"] == 50.0

    def test_repair_request_repair_time_from_component(self):
        env = simpy.Environment()
        disp = MockTechDispatcher(env)
        _, _, machine = self._make_complex_machine(env, disp)

        machine.failed_component = machine.components[1]  # bearing
        request = RepairRequest(machine, created_at=0)

        assert request.get_repair_time() == 30.0

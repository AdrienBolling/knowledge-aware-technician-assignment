"""Tests for MachineComponent and component configuration."""

from kata.entities.components.component import MachineComponent
from kata.features.breakdown.simple_breakdown import (
    SimpleBreakdownProcess,
    WeibullBreakdownProcess,
)


class TestMachineComponent:
    def _make_component(self, **kwargs):
        defaults = {
            "component_id": "motor_1",
            "component_type": "motor",
            "breakdown_process": SimpleBreakdownProcess(
                failure_prob_working=0.1,
                failure_prob_idle=0.01,
            ),
            "base_repair_time": 50.0,
        }
        defaults.update(kwargs)
        return MachineComponent(**defaults)

    def test_getters(self):
        comp = self._make_component()
        assert comp.get_id() == "motor_1"
        assert comp.get_type() == "motor"
        assert comp.get_repair_time() == 50.0

    def test_failure_prob_when_processing(self):
        comp = self._make_component()
        prob = comp.step_and_get_failure_prob(is_processing=True)
        assert prob == 0.1

    def test_failure_prob_when_idle(self):
        comp = self._make_component()
        prob = comp.step_and_get_failure_prob(is_processing=False)
        assert prob == 0.01

    def test_repair_resets_breakdown_process(self):
        bp = SimpleBreakdownProcess(failure_prob_working=0.1)
        comp = MachineComponent("c1", "type", bp, 10.0)
        comp.step_and_get_failure_prob(True)
        assert bp.time_since_repair == 1
        comp.repair()
        assert bp.time_since_repair == 0

    def test_with_weibull_breakdown(self):
        bp = WeibullBreakdownProcess(shape=2.0, scale=1000.0, dt=1)
        comp = MachineComponent("w1", "pump", bp, 75.0)
        p1 = comp.step_and_get_failure_prob(True)
        p2 = comp.step_and_get_failure_prob(True)
        # Weibull with shape>1 has increasing hazard
        assert p2 > p1

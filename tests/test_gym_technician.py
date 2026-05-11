"""Tests for GymTechnician fatigue, knowledge, and repair flow."""

import pytest

from kata.core.config import TechnicianConfig
from kata.entities.technicians.GymTechnician import GymTechnician


class _FakeMachine:
    def __init__(self):
        self.mtype = "test"
        self.machine_id = 1


class _FakeRequest:
    def __init__(self, repair_time: float = 10.0, component_type: str | None = None):
        self.machine = _FakeMachine()
        self.created_at = 0
        self._repair_time = repair_time
        self._comp_type = component_type

    def get_repair_time(self) -> float:
        return self._repair_time

    def get_failed_component_info(self):
        if self._comp_type is None:
            return None
        return {"component_type": self._comp_type}


class TestFatigue:
    def test_initial_fatigue_is_zero(self):
        tech = GymTechnician(TechnicianConfig())
        assert tech.fatigue == 0.0

    def test_increase_fatigue_grows(self):
        tech = GymTechnician(TechnicianConfig(fatigue_lambda=0.1))
        tech._increase_fatigue(10)
        assert tech.fatigue > 0.0

    def test_fatigue_clamped_to_one(self):
        tech = GymTechnician(TechnicianConfig(fatigue_lambda=10.0))
        tech._increase_fatigue(1000)
        assert tech.fatigue <= 1.0

    def test_fatigue_clamped_to_zero(self):
        tech = GymTechnician(TechnicianConfig())
        tech.fatigue = 0.5
        tech._recover_fatigue(100000)
        assert tech.fatigue >= 0.0

    def test_recover_fatigue_decreases(self):
        tech = GymTechnician(TechnicianConfig(fatigue_mu=0.1))
        tech.fatigue = 0.8
        tech._recover_fatigue(10)
        assert tech.fatigue < 0.8

    def test_negative_work_time_raises(self):
        tech = GymTechnician(TechnicianConfig())
        with pytest.raises(ValueError, match="non-negative"):
            tech._increase_fatigue(-1)

    def test_negative_idle_time_raises(self):
        tech = GymTechnician(TechnicianConfig())
        with pytest.raises(ValueError, match="non-negative"):
            tech._recover_fatigue(-1)

    def test_exponential_fatigue_multiplier_is_slowdown(self):
        # Fresh tech: multiplier == 1 (no slowdown)
        tech = GymTechnician(TechnicianConfig())
        assert tech.get_fatigue_multiplier() == 1.0
        # Tired tech: multiplier > 1 (repairs take longer)
        tech.fatigue = 0.5
        assert tech.get_fatigue_multiplier() > 1.0
        # Fully exhausted: multiplier reaches exp(alpha) (alpha=0.5 -> ~1.65)
        tech.fatigue = 1.0
        assert tech.get_fatigue_multiplier() > 1.5


class TestStartRepairAndRecovery:
    """``start_repair`` must drive idle-time fatigue recovery."""

    def test_first_repair_recovers_from_episode_start(self):
        tech = GymTechnician(TechnicianConfig(fatigue_mu=0.1))
        tech.fatigue = 0.5  # forced fatigue carryover
        # No prior repair — _last_idle_since defaults to 0.0
        tech.start_repair(when=10.0)
        assert tech.fatigue < 0.5
        assert tech.busy is True

    def test_recovery_between_two_repairs(self):
        tech = GymTechnician(TechnicianConfig(fatigue_mu=0.1, fatigue_lambda=0.05))
        # First repair finishes at t=100 → fatigue increases, idle clock starts
        req = _FakeRequest(repair_time=20.0)
        tech.repair_finished(req, when=100.0)
        f_after_first = tech.fatigue
        assert f_after_first > 0.0
        assert tech._last_idle_since == 100.0

        # Second repair starts at t=300 → 200 units of idle → recovery
        tech.start_repair(when=300.0)
        assert tech.fatigue < f_after_first
        assert tech.busy is True

    def test_back_to_back_repairs_have_no_recovery(self):
        tech = GymTechnician(TechnicianConfig(fatigue_mu=0.1, fatigue_lambda=0.05))
        req = _FakeRequest(repair_time=20.0)
        tech.repair_finished(req, when=100.0)
        f_after_first = tech.fatigue
        # Queued repair starts immediately after the first finishes
        tech.start_repair(when=100.0)
        assert tech.fatigue == f_after_first
        assert tech.busy is True

    def test_long_idle_drives_fatigue_close_to_zero(self):
        tech = GymTechnician(TechnicianConfig(fatigue_mu=0.1))
        tech.fatigue = 0.9
        tech._last_idle_since = 0.0
        tech.start_repair(when=1000.0)
        # exp(-0.1 * 1000) ≈ 0  → fatigue ~ 0
        assert tech.fatigue < 1e-6


class TestRepairFinished:
    def test_sets_busy_false(self):
        tech = GymTechnician(TechnicianConfig())
        tech.busy = True
        req = _FakeRequest(repair_time=10.0)
        tech.repair_finished(req, when=100.0)
        assert tech.busy is False

    def test_increases_fatigue_on_repair(self):
        tech = GymTechnician(TechnicianConfig(fatigue_lambda=0.1))
        req = _FakeRequest(repair_time=50.0)
        tech.repair_finished(req, when=100.0)
        assert tech.fatigue > 0.0

    def test_increases_knowledge_on_repair(self):
        tech = GymTechnician(TechnicianConfig())
        initial_knowledge = tech.knowledge_grid.get_max_knowledge()
        req = _FakeRequest(repair_time=10.0, component_type="motor")
        tech.repair_finished(req, when=100.0)
        final_knowledge = tech.knowledge_grid.get_max_knowledge()
        assert final_knowledge > initial_knowledge


class TestComputeRepairTime:
    def test_returns_non_negative_float(self):
        tech = GymTechnician(TechnicianConfig())
        req = _FakeRequest(repair_time=1.0)
        result = tech.compute_repair_time(1.0, req)
        # The function returns a non-negative float (no integer floor).
        assert isinstance(result, float)
        assert result >= 0.0

    def test_fatigue_increases_repair_time(self):
        # Fresh tech with no knowledge: base time, no slowdown.
        tech = GymTechnician(TechnicianConfig())
        req = _FakeRequest(repair_time=100.0)
        t_fresh = tech.compute_repair_time(100.0, req)

        # Tired tech with no knowledge gain: fatigue multiplier > 1 ⇒
        # repair takes *longer* (slowdown semantics).
        tech.fatigue = 0.9
        t_tired = tech.compute_repair_time(100.0, req)
        assert t_tired > t_fresh


class TestAutoId:
    def test_ids_auto_increment(self):
        t1 = GymTechnician(TechnicianConfig())
        t2 = GymTechnician(TechnicianConfig())
        assert t2.id == t1.id + 1

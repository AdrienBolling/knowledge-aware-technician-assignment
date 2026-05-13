"""Tests for GymTechnician fatigue, knowledge, and repair flow."""

import math

import pytest

from kata.core.config import TechnicianConfig
from kata.entities.technicians.GymTechnician import CONFIG, GymTechnician


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


class _FakeRequestWithOverrides:
    """A RepairRequest stand-in that exposes per-failure knowledge params.

    Mirrors the contract surfaced by ``RepairRequest.get_knowledge_parameters``
    so we can exercise the override path without spinning up a SimPy
    environment + ComplexMachine + breakdown process.
    """

    def __init__(
        self,
        repair_time: float = 10.0,
        component_type: str | None = None,
        knowledge_params: tuple[float | None, float | None] | None = None,
    ):
        self.machine = _FakeMachine()
        self.created_at = 0
        self._repair_time = repair_time
        self._comp_type = component_type
        self._knowledge_params = knowledge_params

    def get_repair_time(self) -> float:
        return self._repair_time

    def get_failed_component_info(self):
        if self._comp_type is None:
            return None
        return {"component_type": self._comp_type}

    def get_knowledge_parameters(self):
        return self._knowledge_params


@pytest.fixture
def repair_cfg_snapshot():
    """Snapshot & restore ``CONFIG.sim.repair`` around a test.

    GymTechnician reads ``sim.repair.{min_repair_fraction,
    knowledge_sensitivity, failure_wise_knowledge_parameters}`` from the
    cached singleton at *call* time, so tests that toggle these fields
    must restore them or they leak across tests.
    """
    cfg = CONFIG.sim.repair
    saved = (
        cfg.min_repair_fraction,
        cfg.knowledge_sensitivity,
        cfg.failure_wise_knowledge_parameters,
    )
    yield cfg
    (
        cfg.min_repair_fraction,
        cfg.knowledge_sensitivity,
        cfg.failure_wise_knowledge_parameters,
    ) = saved


class TestFailureWiseKnowledgeMultiplier:
    """``GymTechnician.get_knowledge_multiplier`` flag-gated override path."""

    def test_flag_off_ignores_per_component_overrides(self, repair_cfg_snapshot):
        # Global params: floor=0.3, alpha=0.002.
        repair_cfg_snapshot.min_repair_fraction = 0.3
        repair_cfg_snapshot.knowledge_sensitivity = 0.002
        repair_cfg_snapshot.failure_wise_knowledge_parameters = False

        tech = GymTechnician(TechnicianConfig())
        req_no_override = _FakeRequestWithOverrides(component_type="motor")
        req_with_override = _FakeRequestWithOverrides(
            component_type="motor",
            knowledge_params=(0.05, 5.0),  # would be very fast if honoured
        )

        # With no knowledge experience k=0, multiplier is identically 1
        # regardless of params — so add some experience first.
        tech.repair_finished(req_no_override, when=10.0)

        # Both requests must land on the *same* multiplier because the
        # flag is off — per-component overrides are ignored.
        assert tech.get_knowledge_multiplier(
            req_no_override
        ) == tech.get_knowledge_multiplier(req_with_override)

    def test_flag_on_uses_per_component_overrides(self, repair_cfg_snapshot):
        # Global params: relatively gentle (high floor, low alpha).
        repair_cfg_snapshot.min_repair_fraction = 0.7
        repair_cfg_snapshot.knowledge_sensitivity = 0.01
        repair_cfg_snapshot.failure_wise_knowledge_parameters = True

        tech = GymTechnician(TechnicianConfig())
        # Earn some knowledge so the multiplier isn't pinned at 1.
        warmup = _FakeRequestWithOverrides(component_type="motor")
        for _ in range(5):
            tech.repair_finished(warmup, when=10.0)

        global_req = _FakeRequestWithOverrides(
            component_type="motor",
            knowledge_params=None,  # falls back to global
        )
        override_req = _FakeRequestWithOverrides(
            component_type="motor",
            knowledge_params=(0.2, 0.5),  # lower floor, much higher alpha
        )

        m_global = tech.get_knowledge_multiplier(global_req)
        m_override = tech.get_knowledge_multiplier(override_req)

        # Override has a steeper alpha *and* lower floor, so for any
        # nonzero knowledge the override multiplier is strictly smaller
        # (faster repairs).
        assert m_override < m_global
        # Sanity: floors are respected.
        assert m_override >= 0.2
        assert m_global >= 0.7

    def test_flag_on_partial_override_only_floor(self, repair_cfg_snapshot):
        # Component overrides the floor but defers alpha to global.
        repair_cfg_snapshot.min_repair_fraction = 0.5
        repair_cfg_snapshot.knowledge_sensitivity = 0.3
        repair_cfg_snapshot.failure_wise_knowledge_parameters = True

        tech = GymTechnician(TechnicianConfig())
        warmup = _FakeRequestWithOverrides(component_type="motor")
        for _ in range(10):
            tech.repair_finished(warmup, when=10.0)

        global_req = _FakeRequestWithOverrides(
            component_type="motor",
            knowledge_params=None,  # falls back to global (floor=0.5)
        )
        override_req = _FakeRequestWithOverrides(
            component_type="motor",
            knowledge_params=(0.1, None),  # lower floor, global alpha
        )
        m_global = tech.get_knowledge_multiplier(global_req)
        m_override = tech.get_knowledge_multiplier(override_req)

        # Same alpha → identical exponential decay shape, but the
        # override sits ``0.4 * (1 - exp(-alpha*k))`` *below* the global
        # multiplier whenever k > 0.  Strictly less is guaranteed.
        assert m_override < m_global
        assert m_override >= 0.1  # respects per-component floor
        assert m_global >= 0.5  # respects global floor

    def test_flag_on_partial_override_only_alpha(self, repair_cfg_snapshot):
        # Component overrides alpha but defers the floor to global.
        repair_cfg_snapshot.min_repair_fraction = 0.4
        repair_cfg_snapshot.knowledge_sensitivity = 0.01
        repair_cfg_snapshot.failure_wise_knowledge_parameters = True

        tech = GymTechnician(TechnicianConfig())
        warmup = _FakeRequestWithOverrides(component_type="motor")
        for _ in range(5):
            tech.repair_finished(warmup, when=10.0)

        req_slow_learn = _FakeRequestWithOverrides(
            component_type="motor",
            knowledge_params=(None, 0.01),  # global alpha
        )
        req_fast_learn = _FakeRequestWithOverrides(
            component_type="motor",
            knowledge_params=(None, 1.0),  # much steeper
        )
        m_slow = tech.get_knowledge_multiplier(req_slow_learn)
        m_fast = tech.get_knowledge_multiplier(req_fast_learn)

        # Higher alpha → faster descent toward the (shared) floor.
        assert m_fast < m_slow
        # Both bounded by the global floor.
        assert m_slow >= 0.4
        assert m_fast >= 0.4

    def test_flag_on_no_failed_component_falls_back_to_global(
        self, repair_cfg_snapshot
    ):
        # Simulates a simple-machine request where get_knowledge_parameters
        # returns None outright (no failed component).
        repair_cfg_snapshot.min_repair_fraction = 0.3
        repair_cfg_snapshot.knowledge_sensitivity = 0.5
        repair_cfg_snapshot.failure_wise_knowledge_parameters = True

        tech = GymTechnician(TechnicianConfig())
        for _ in range(5):
            tech.repair_finished(
                _FakeRequestWithOverrides(component_type="motor"), when=10.0
            )

        no_comp_req = _FakeRequestWithOverrides(
            component_type=None,
            knowledge_params=None,  # mimics "no failed component"
        )
        m = tech.get_knowledge_multiplier(no_comp_req)
        # No override → uses global (floor=0.3, alpha=0.5).
        assert m >= 0.3
        # And matches the global formula exactly.
        k = float(tech.knowledge_grid.get_knowledge(tech.encoder.encode(no_comp_req)))
        expected = 0.3 + (1.0 - 0.3) * math.exp(-0.5 * k)
        assert math.isclose(m, expected, rel_tol=1e-9)

    def test_flag_on_request_missing_method_falls_back_to_global(
        self, repair_cfg_snapshot
    ):
        # A request that doesn't expose get_knowledge_parameters at all
        # must not crash — the multiplier should silently fall back to
        # the global params.
        repair_cfg_snapshot.min_repair_fraction = 0.25
        repair_cfg_snapshot.knowledge_sensitivity = 0.4
        repair_cfg_snapshot.failure_wise_knowledge_parameters = True

        tech = GymTechnician(TechnicianConfig())

        legacy = _FakeRequest(repair_time=10.0, component_type="motor")
        # Earn some knowledge first.
        tech.repair_finished(legacy, when=10.0)
        m = tech.get_knowledge_multiplier(legacy)
        assert m >= 0.25
        assert m <= 1.0


class TestFailureWiseKnowledgeIntegration:
    """End-to-end test of MachineComponent → RepairRequest → multiplier."""

    def test_repair_request_surfaces_component_overrides(self, repair_cfg_snapshot):
        from kata.entities.components.component import MachineComponent
        from kata.entities.requests.RepairRequest import RepairRequest
        from kata.features.breakdown.simple_breakdown import SimpleBreakdownProcess

        repair_cfg_snapshot.min_repair_fraction = 0.6
        repair_cfg_snapshot.knowledge_sensitivity = 0.01
        repair_cfg_snapshot.failure_wise_knowledge_parameters = True

        comp = MachineComponent(
            component_id="belt_0",
            component_type="mechanical",
            breakdown_process=SimpleBreakdownProcess(failure_prob_working=0.001),
            base_repair_time=30.0,
            min_repair_fraction=0.2,
            knowledge_sensitivity=0.8,
        )

        class _MachineStub:
            def __init__(self, c):
                self._c = c
                self.mtype = "Conveyor"
                self.machine_id = 42

            def get_failed_component(self):
                return self._c

        req = RepairRequest(machine=_MachineStub(comp), created_at=0)

        # RepairRequest must expose the component's overrides verbatim.
        assert req.get_knowledge_parameters() == (0.2, 0.8)

        # And the technician's multiplier honours them: with the
        # override's much steeper alpha=0.8, even a small amount of
        # knowledge should push the multiplier well below the global
        # floor (0.6) and toward the component-level floor (0.2).
        tech = GymTechnician(TechnicianConfig())
        for _ in range(10):
            tech.repair_finished(req, when=10.0)
        m = tech.get_knowledge_multiplier(req)
        assert m < 0.6
        assert m >= 0.2

    def test_simple_machine_request_returns_none_for_overrides(self):
        # A RepairRequest built from a machine that doesn't expose
        # get_failed_component (a plain Machine) must return None from
        # get_knowledge_parameters so the global params are used.
        from kata.entities.requests.RepairRequest import RepairRequest

        class _SimpleMachineStub:
            def __init__(self):
                self.mtype = "Generic"
                self.machine_id = 1

        req = RepairRequest(machine=_SimpleMachineStub(), created_at=0)
        assert req.get_knowledge_parameters() is None

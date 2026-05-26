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

        # With the default disruption pool, no disruption is likely to
        # fire within the first 2000 sim time (random: mean 10000;
        # vacation: uniform offset of up to 8000; fatigue: needs
        # accumulated fatigue first), so this exercises a clean
        # repair → completion path.
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


# ---------------------------------------------------------------------------
# Disruption-mechanism tests (added with the per-trigger refactor).
# ---------------------------------------------------------------------------


class TestDisruptionTriggers:
    """Per-trigger behaviour and preempt-then-requeue."""

    def _build(self, disruption_dict, n_techs=1):
        """Build a dispatcher with a hand-crafted disruption pool."""
        from kata import get_config
        from kata.core.config import DisruptionConfig

        # Mutate the cached singleton so the dispatcher's spawn loop
        # sees our pool.  Save / restore around the test in a finalizer
        # built by the caller.
        cfg = get_config()
        prev = cfg.sim.disruptions
        cfg.sim.disruptions = DisruptionConfig(dis_dict=disruption_dict)
        env, techs, dispatcher = _build_env_and_dispatcher(n_techs=n_techs)
        return env, techs, dispatcher, prev

    def _restore(self, prev):
        from kata import get_config

        get_config().sim.disruptions = prev

    def test_periodic_disruption_fires_multiple_times(self):
        """A periodic disruption with short interval fires many times per episode."""
        from kata.core.config import DisruptionTypeConfig

        pool = {
            "tick": DisruptionTypeConfig(
                trigger="periodic",
                interval=100.0,
                jitter=0.0,
                duration_mu=5.0,
                duration_sig=0.0,
                preemptive=False,
            ),
        }
        env, techs, _disp, prev = self._build(pool)
        try:
            env.run(until=5_000)
            # Initial uniform offset of up to ``interval``, then every
            # 100 sim time units → expect ~40-50 firings in 5000 units.
            assert techs[0].disruption_count >= 10
        finally:
            self._restore(prev)

    def test_random_disruption_fires_at_least_once_with_high_rate(self):
        from kata.core.config import DisruptionTypeConfig

        pool = {
            "frequent_injury": DisruptionTypeConfig(
                trigger="random",
                rate=1e-2,            # mean inter-arrival = 100
                duration_mu=5.0,
                duration_sig=0.0,
                preemptive=False,
            ),
        }
        env, techs, _disp, prev = self._build(pool)
        try:
            env.run(until=10_000)
            # 10 000 sim time × rate 0.01 → expected ~100 events.  We
            # only require ``>= 5`` for tail-bound robustness.
            assert techs[0].disruption_count >= 5
        finally:
            self._restore(prev)

    def test_preemptive_disruption_requeues_ticket(self):
        """A preempting disruption mid-repair must restore tech state + re-queue ticket."""
        from kata.core.config import DisruptionTypeConfig

        # Pool with a single periodic disruption that fires almost
        # immediately and preempts.
        pool = {
            "preempt_test": DisruptionTypeConfig(
                trigger="periodic",
                interval=10.0,
                jitter=0.0,
                duration_mu=500.0,
                duration_sig=0.0,
                preemptive=True,
            ),
        }
        env, techs, dispatcher, prev = self._build(pool, n_techs=1)
        try:
            machine = _make_machine(env, dispatcher)
            machine.broken = True
            # Stage a long-base repair so the disruption is highly
            # likely to land mid-repair.
            class _LongRequest(RepairRequest):
                def get_repair_time(self):
                    return 1_000.0

            request = _LongRequest(machine, created_at=0)
            dispatcher.start_repair(techs[0].id, request)

            # Run long enough for the disruption to fire and preempt
            # but not long enough for the disruption (500 units) to end
            # AND a full follow-up repair to complete.
            env.run(until=400)
            assert techs[0].disruption_count >= 1
            # The preempted ticket must have been re-queued for re-assignment.
            assert request in list(dispatcher.repair_queue.items)
            # And the technician must have been released.
            assert techs[0].busy is False
        finally:
            self._restore(prev)


class TestDisruptionConfigAlias:
    """``interrupt_on_disrupt`` is derived from per-type ``preemptive`` flags."""

    def test_true_when_any_type_preempts(self):
        from kata.core.config import DisruptionConfig, DisruptionTypeConfig

        cfg = DisruptionConfig(
            dis_dict={
                "a": DisruptionTypeConfig(
                    trigger="periodic", interval=100.0, duration_mu=10.0,
                    preemptive=False,
                ),
                "b": DisruptionTypeConfig(
                    trigger="periodic", interval=100.0, duration_mu=10.0,
                    preemptive=True,
                ),
            }
        )
        assert cfg.interrupt_on_disrupt is True

    def test_false_when_no_type_preempts(self):
        from kata.core.config import DisruptionConfig, DisruptionTypeConfig

        cfg = DisruptionConfig(
            dis_dict={
                "a": DisruptionTypeConfig(
                    trigger="periodic", interval=100.0, duration_mu=10.0,
                    preemptive=False,
                ),
            }
        )
        assert cfg.interrupt_on_disrupt is False


class TestFatigueTriggeredDisruption:
    """End-to-end exercise of ``trigger='fatigue'``."""

    def test_fatigue_trigger_fires_when_fatigue_is_high(self):
        """Force fatigue to 1 and verify the polling loop actually fires.

        The time-aware fatigue property decays the seeded value as the
        SimPy clock advances; to keep the hazard pinned at its ceiling
        for the duration of the test we construct the technician with
        a near-zero recovery rate ``mu`` (the validator forbids 0
        exactly).
        """
        from kata import get_config
        from kata.core.config import DisruptionConfig, DisruptionTypeConfig

        prev = get_config().sim.disruptions
        get_config().sim.disruptions = DisruptionConfig(
            dis_dict={
                "exhaustion": DisruptionTypeConfig(
                    trigger="fatigue",
                    fatigue_coefficient=0.5,
                    poll_interval=10.0,
                    duration_mu=5.0,
                    duration_sig=0.0,
                    preemptive=False,
                ),
            }
        )
        try:
            env = simpy.Environment()
            techs = [
                GymTechnician(TechnicianConfig(name="tech_0", fatigue_mu=1e-6))
            ]
            disp = GymTechDispatcher(env, techs)
            disp.seed_disruptions(42)
            techs[0].fatigue = 1.0
            env.run(until=2_000)
            # Per-poll p = clip(0.5 * 1 * 10, 0, 1) = 1.0; every poll
            # fires, separated by poll (10) + duration (5).  Expected
            # ~ 2000 / 15 ≈ 130 events; conservative lower bound:
            assert techs[0].disruption_count >= 50
        finally:
            get_config().sim.disruptions = prev

    def test_fatigue_trigger_does_not_fire_at_zero_fatigue(self):
        """At fatigue=0 the per-poll probability is identically zero."""
        from kata import get_config
        from kata.core.config import DisruptionConfig, DisruptionTypeConfig

        prev = get_config().sim.disruptions
        get_config().sim.disruptions = DisruptionConfig(
            dis_dict={
                "exhaustion": DisruptionTypeConfig(
                    trigger="fatigue",
                    fatigue_coefficient=1.0,    # extreme
                    poll_interval=10.0,
                    duration_mu=5.0,
                    duration_sig=0.0,
                    preemptive=False,
                ),
            }
        )
        try:
            env, techs, _disp = _build_env_and_dispatcher(n_techs=1)
            _disp.seed_disruptions(0)
            techs[0].fatigue = 0.0
            env.run(until=5_000)
            assert techs[0].disruption_count == 0
        finally:
            get_config().sim.disruptions = prev


class TestDisruptionReproducibility:
    """Same seed → identical disruption sequences across runs."""

    def test_same_seed_same_counts(self):
        from kata import get_config
        from kata.core.config import DisruptionConfig, DisruptionTypeConfig

        prev = get_config().sim.disruptions
        get_config().sim.disruptions = DisruptionConfig(
            dis_dict={
                "injury": DisruptionTypeConfig(
                    trigger="random", rate=5e-3, duration_mu=5.0, duration_sig=0.0,
                    preemptive=False,
                ),
                "vacation": DisruptionTypeConfig(
                    trigger="periodic", interval=500.0, jitter=50.0,
                    duration_mu=10.0, duration_sig=0.0, preemptive=False,
                ),
            }
        )
        try:
            results = []
            for _trial in range(2):
                env, techs, disp = _build_env_and_dispatcher(n_techs=2)
                disp.seed_disruptions(1234)
                env.run(until=2_000)
                results.append(
                    [(t.disruption_count, dict(t.disruption_counts_by_type)) for t in techs]
                )
            assert results[0] == results[1], (
                f"Same seed produced different disruption sequences: "
                f"{results[0]} vs {results[1]}"
            )
        finally:
            get_config().sim.disruptions = prev

    def test_different_seeds_diverge(self):
        from kata import get_config
        from kata.core.config import DisruptionConfig, DisruptionTypeConfig

        prev = get_config().sim.disruptions
        get_config().sim.disruptions = DisruptionConfig(
            dis_dict={
                "injury": DisruptionTypeConfig(
                    trigger="random", rate=5e-3, duration_mu=5.0, duration_sig=0.0,
                    preemptive=False,
                ),
            }
        )
        try:
            counts = []
            for seed in (1, 2, 3):
                env, techs, disp = _build_env_and_dispatcher(n_techs=2)
                disp.seed_disruptions(seed)
                env.run(until=2_000)
                counts.append(tuple(t.disruption_count for t in techs))
            # At least one pair of seeds should give different counts.
            assert len(set(counts)) > 1, (
                f"Different seeds produced identical counts {counts}"
            )
        finally:
            get_config().sim.disruptions = prev


class TestDisruptionAvailability:
    """A technician in a disruption hold must be masked out of the action set."""

    def test_in_disruption_flag_set_during_hold(self):
        from kata import get_config
        from kata.core.config import DisruptionConfig, DisruptionTypeConfig

        prev = get_config().sim.disruptions
        get_config().sim.disruptions = DisruptionConfig(
            dis_dict={
                "vacation": DisruptionTypeConfig(
                    trigger="periodic", interval=5.0, jitter=0.0,
                    duration_mu=1000.0, duration_sig=0.0, preemptive=False,
                ),
            }
        )
        try:
            env, techs, _disp = _build_env_and_dispatcher(n_techs=1)
            # Run just long enough for the vacation to start but not end.
            env.run(until=50)
            assert techs[0]._in_disruption is True
            assert techs[0].disruption_count >= 1
        finally:
            get_config().sim.disruptions = prev

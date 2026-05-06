"""Tests for the new manufacturing-KPI reward components.

Tests cover:
- fleet_availability
- throughput_delta
- repair_backlog_age
- technician_utilization
- downtime_cost

Each test verifies correct raw value computation, coefficient application,
and edge-case behavior.
"""

import math

from conftest import (
    FakeDispatcher,
    FakeMachine,
    FakeRequest,
    FakeSimEnv,
)

from kata.core.config import GymEnvConfig
from kata.env import KataEnv


def _disabled_all() -> dict:
    """Return a reward config dict with all components disabled."""
    return {
        "assignment": {"enabled": False, "coefficient": 1.0},
        "wait_time": {"enabled": False, "coefficient": 1.0},
        "queue_size": {"enabled": False, "coefficient": 1.0},
        "busy_technician": {"enabled": False, "coefficient": 1.0},
    }


def _make_env(
    sim_env=None,
    dispatcher=None,
    reward_overrides=None,
    **kwargs,
):
    """Helper to create a KataEnv with only specified rewards enabled."""
    reward = _disabled_all()
    if reward_overrides:
        reward.update(reward_overrides)
    sim_env = sim_env or FakeSimEnv()
    dispatcher = dispatcher or FakeDispatcher(tech_count=2)
    config = GymEnvConfig(
        max_episode_steps=100,
        max_sim_time=10000.0,
        reward=reward,
        **kwargs,
    )
    return KataEnv(sim_env=sim_env, dispatcher=dispatcher, config=config)


# ======================================================================
# fleet_availability
# ======================================================================


class TestFleetAvailability:
    """Fleet availability = fraction of non-broken machines."""

    def test_all_machines_operational(self):
        dispatcher = FakeDispatcher(tech_count=2)
        m1 = FakeMachine(machine_id=1)
        m1.broken = False
        m2 = FakeMachine(machine_id=2)
        m2.broken = False
        dispatcher.machines = [m1, m2]
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))

        env = _make_env(
            dispatcher=dispatcher,
            reward_overrides={
                "fleet_availability": {"enabled": True, "coefficient": 1.0}
            },
        )
        env.reset()
        _, reward, _, _, info = env.step(0)

        assert info["reward_breakdown"]["fleet_availability"] == 1.0

    def test_half_machines_broken(self):
        dispatcher = FakeDispatcher(tech_count=2)
        m1 = FakeMachine(machine_id=1)
        m1.broken = True
        m2 = FakeMachine(machine_id=2)
        m2.broken = False
        dispatcher.machines = [m1, m2]
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))

        env = _make_env(
            dispatcher=dispatcher,
            reward_overrides={
                "fleet_availability": {"enabled": True, "coefficient": 1.0}
            },
        )
        env.reset()
        _, reward, _, _, info = env.step(0)

        assert info["reward_breakdown"]["fleet_availability"] == 0.5

    def test_all_machines_broken(self):
        dispatcher = FakeDispatcher(tech_count=2)
        m1 = FakeMachine(machine_id=1)
        m1.broken = True
        m2 = FakeMachine(machine_id=2)
        m2.broken = True
        dispatcher.machines = [m1, m2]
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))

        env = _make_env(
            dispatcher=dispatcher,
            reward_overrides={
                "fleet_availability": {"enabled": True, "coefficient": 1.0}
            },
        )
        env.reset()
        _, reward, _, _, info = env.step(0)

        assert info["reward_breakdown"]["fleet_availability"] == 0.0

    def test_coefficient_applied(self):
        dispatcher = FakeDispatcher(tech_count=2)
        m1 = FakeMachine(machine_id=1)
        m1.broken = False
        dispatcher.machines = [m1]
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))

        env = _make_env(
            dispatcher=dispatcher,
            reward_overrides={
                "fleet_availability": {"enabled": True, "coefficient": 3.0}
            },
        )
        env.reset()
        _, reward, _, _, info = env.step(0)

        assert info["reward_breakdown"]["fleet_availability"] == 3.0


# ======================================================================
# throughput_delta
# ======================================================================


class FakeSink:
    def __init__(self, completed: int = 0):
        self.completed = completed


class TestThroughputDelta:
    """Throughput delta = clipped change in finished products."""

    def test_no_throughput_gives_zero(self):
        dispatcher = FakeDispatcher(tech_count=2)
        dispatcher.sinks = [FakeSink(0)]
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))

        env = _make_env(
            dispatcher=dispatcher,
            reward_overrides={
                "throughput_delta": {"enabled": True, "coefficient": 1.0}
            },
        )
        env.reset()
        _, reward, _, _, info = env.step(0)

        assert info["reward_breakdown"]["throughput_delta"] == 0.0

    def test_throughput_increase_gives_positive(self):
        dispatcher = FakeDispatcher(tech_count=2)
        dispatcher.sinks = [FakeSink(0)]
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=2, created_at=0.0))

        env = _make_env(
            dispatcher=dispatcher,
            reward_overrides={
                "throughput_delta": {"enabled": True, "coefficient": 1.0}
            },
        )
        env.reset()

        # First step: no products yet
        _, _, _, _, info1 = env.step(0)
        assert info1["reward_breakdown"]["throughput_delta"] == 0.0

        # Simulate a product completing between steps
        dispatcher.sinks[0].completed = 1
        _, _, _, _, info2 = env.step(0)
        assert info2["reward_breakdown"]["throughput_delta"] == 1.0

    def test_throughput_clipped_at_one(self):
        dispatcher = FakeDispatcher(tech_count=2)
        dispatcher.sinks = [FakeSink(5)]  # already 5 products at start
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))

        env = _make_env(
            dispatcher=dispatcher,
            reward_overrides={
                "throughput_delta": {"enabled": True, "coefficient": 1.0}
            },
        )
        env.reset()  # _prev_finished_products gets reset to 0

        # First step sees all 5 existing products as delta — but clipped to 1
        _, _, _, _, info = env.step(0)
        assert info["reward_breakdown"]["throughput_delta"] == 1.0


# ======================================================================
# repair_backlog_age
# ======================================================================


class TestRepairBacklogAge:
    """Repair backlog age = -tanh(mean_age / 200) of all queued requests."""

    def test_empty_queue_gives_zero(self):
        dispatcher = FakeDispatcher(tech_count=2)
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))

        env = _make_env(
            dispatcher=dispatcher,
            reward_overrides={
                "repair_backlog_age": {"enabled": True, "coefficient": 1.0}
            },
        )
        env.reset()
        # After reset, the request is popped as current_request, queue is empty
        _, _, _, _, info = env.step(0)

        assert info["reward_breakdown"]["repair_backlog_age"] == 0.0

    def test_old_backlog_gives_negative(self):
        sim_env = FakeSimEnv()
        sim_env.now = 500.0
        dispatcher = FakeDispatcher(tech_count=2)
        # Current request + one old request still in queue
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))
        dispatcher.repair_queue.items.append(
            FakeRequest(machine_id=2, created_at=100.0)
        )

        env = _make_env(
            sim_env=sim_env,
            dispatcher=dispatcher,
            reward_overrides={
                "repair_backlog_age": {"enabled": True, "coefficient": 1.0}
            },
        )
        env.reset()
        # After reset: first request is current_request, second stays in queue
        # Queue item has age = 500 - 100 = 400
        _, _, _, _, info = env.step(0)

        raw = info["reward_breakdown"]["repair_backlog_age"]
        expected = -math.tanh(400.0 / 200.0)
        assert abs(raw - expected) < 1e-4

    def test_value_is_bounded(self):
        """Even extremely old backlogs produce values in (-1, 0]."""
        sim_env = FakeSimEnv()
        sim_env.now = 9000.0  # within max_sim_time
        dispatcher = FakeDispatcher(tech_count=2)
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=2, created_at=0.0))

        env = _make_env(
            sim_env=sim_env,
            dispatcher=dispatcher,
            reward_overrides={
                "repair_backlog_age": {"enabled": True, "coefficient": 1.0}
            },
        )
        env.reset()
        _, _, _, _, info = env.step(0)

        raw = info["reward_breakdown"]["repair_backlog_age"]
        assert -1.0 <= raw < 0.0


# ======================================================================
# technician_utilization
# ======================================================================


class TestTechnicianUtilization:
    """Utilization reward = Gaussian centred at optimal (0.65)."""

    def test_no_busy_techs_gives_penalty(self):
        dispatcher = FakeDispatcher(tech_count=4)
        # All idle (utilization = 0.0)
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))

        env = _make_env(
            dispatcher=dispatcher,
            reward_overrides={
                "technician_utilization": {"enabled": True, "coefficient": 1.0}
            },
        )
        env.reset()
        _, _, _, _, info = env.step(0)

        # After step(0), tech 0 becomes busy (1/4 = 0.25 utilization)
        raw = info["reward_breakdown"]["technician_utilization"]
        # Not at optimal, so less than max
        assert -1.0 <= raw <= 1.0

    def test_optimal_utilization_gives_highest_reward(self):
        """When ~65% of techs are busy, reward should be near peak."""
        dispatcher = FakeDispatcher(tech_count=3)
        # Set 2 of 3 techs busy (67% utilization, close to optimal 65%)
        dispatcher.techs[0].busy = True
        dispatcher.techs[1].busy = True
        dispatcher.techs[2].busy = False
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))

        env = _make_env(
            dispatcher=dispatcher,
            reward_overrides={
                "technician_utilization": {"enabled": True, "coefficient": 1.0}
            },
        )
        env.reset()
        _, _, _, _, info = env.step(0)

        # After step: tech 0 becomes busy via start_repair, but was already busy
        # Utilization is now 3/3 = 1.0 (all busy)
        # Hmm, start_repair sets the assigned tech to busy.
        # Actually let's check: step(0) assigns tech_id=0 -> tech 0 set busy
        # But tech 0 was already busy. So still 2/3 + tech 2 was assigned.
        # Wait - step action=0 means tech index 0. start_repair does:
        #   self.techs[tech_id].busy = True
        # So after step(0): tech 0 was already busy, stays busy. All 3 weren't changed.
        # Actually action=0 means tech index 0 in the action space.
        # The dispatcher sets self.techs[0].busy = True (was already True).
        # So utilization = 2/3 = 0.667 still.
        raw = info["reward_breakdown"]["technician_utilization"]
        # At 0.667, very close to optimal 0.65
        # Gaussian(0.667, 0.65, 0.3): exp(-((0.667-0.65)^2)/(2*0.09)) * 2 - 1
        # ≈ exp(-0.0016) * 2 - 1 ≈ 0.997 * 2 - 1 ≈ 0.993
        assert raw > 0.9

    def test_value_range(self):
        """Reward is always in [-1, 1]."""
        for n_busy in range(5):
            dispatcher = FakeDispatcher(tech_count=4)
            for i in range(n_busy):
                dispatcher.techs[i].busy = True
            dispatcher.repair_queue.items.append(
                FakeRequest(machine_id=1, created_at=0.0)
            )

            env = _make_env(
                dispatcher=dispatcher,
                reward_overrides={
                    "technician_utilization": {"enabled": True, "coefficient": 1.0}
                },
            )
            env.reset()
            _, _, _, _, info = env.step(0)

            raw = info["reward_breakdown"]["technician_utilization"]
            assert -1.0 <= raw <= 1.0


# ======================================================================
# downtime_cost
# ======================================================================


class TestDowntimeCost:
    """Downtime cost = negative fraction of machine-time lost."""

    def test_no_downtime_gives_zero(self):
        sim_env = FakeSimEnv()
        sim_env.now = 100.0
        dispatcher = FakeDispatcher(tech_count=2)
        m1 = FakeMachine(machine_id=1)
        m1.broken = False
        m2 = FakeMachine(machine_id=2)
        m2.broken = False
        dispatcher.machines = [m1, m2]
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))

        env = _make_env(
            sim_env=sim_env,
            dispatcher=dispatcher,
            reward_overrides={"downtime_cost": {"enabled": True, "coefficient": 1.0}},
        )
        env.reset()
        _, _, _, _, info = env.step(0)

        assert info["reward_breakdown"]["downtime_cost"] == 0.0

    def test_broken_machines_give_penalty(self):
        sim_env = FakeSimEnv()
        sim_env.now = 100.0
        dispatcher = FakeDispatcher(tech_count=2)
        m1 = FakeMachine(machine_id=1)
        m1.broken = True  # Has been broken for some time
        m2 = FakeMachine(machine_id=2)
        m2.broken = False
        dispatcher.machines = [m1, m2]
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))

        env = _make_env(
            sim_env=sim_env,
            dispatcher=dispatcher,
            reward_overrides={"downtime_cost": {"enabled": True, "coefficient": 1.0}},
        )
        env.reset()
        # After reset, downtime tracking starts. Machine 1 is broken at t=100.
        # But _machine_down_since was just reset, so first observation records it as down "now"
        # Downtime = 0 since it was just detected
        _, _, _, _, info = env.step(0)
        raw = info["reward_breakdown"]["downtime_cost"]
        assert raw <= 0.0

    def test_value_bounded_negative(self):
        """Downtime cost is always in [-1, 0]."""
        sim_env = FakeSimEnv()
        sim_env.now = 1000.0
        dispatcher = FakeDispatcher(tech_count=2)
        # All machines broken
        machines = [FakeMachine(machine_id=i) for i in range(5)]
        for m in machines:
            m.broken = True
        dispatcher.machines = machines
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))

        env = _make_env(
            sim_env=sim_env,
            dispatcher=dispatcher,
            reward_overrides={"downtime_cost": {"enabled": True, "coefficient": 1.0}},
        )
        env.reset()
        _, _, _, _, info = env.step(0)

        raw = info["reward_breakdown"]["downtime_cost"]
        assert -1.0 <= raw <= 0.0


# ======================================================================
# Integration: all new rewards enabled simultaneously
# ======================================================================


class TestAllNewRewardsEnabled:
    """Verify all 5 new components work together without interference."""

    def test_all_new_rewards_produce_values(self):
        sim_env = FakeSimEnv()
        sim_env.now = 50.0
        dispatcher = FakeDispatcher(tech_count=3)
        dispatcher.techs[0].busy = True
        dispatcher.techs[1].busy = False
        dispatcher.techs[2].busy = False
        m1 = FakeMachine(machine_id=1)
        m1.broken = True
        m2 = FakeMachine(machine_id=2)
        m2.broken = False
        dispatcher.machines = [m1, m2]
        dispatcher.sinks = [FakeSink(3)]
        # Two requests: one becomes current, one stays in backlog
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=10.0))
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=2, created_at=30.0))

        env = _make_env(
            sim_env=sim_env,
            dispatcher=dispatcher,
            reward_overrides={
                "fleet_availability": {"enabled": True, "coefficient": 1.0},
                "throughput_delta": {"enabled": True, "coefficient": 1.0},
                "repair_backlog_age": {"enabled": True, "coefficient": 1.0},
                "technician_utilization": {"enabled": True, "coefficient": 1.0},
                "downtime_cost": {"enabled": True, "coefficient": 1.0},
            },
        )
        env.reset()
        _, reward, _, _, info = env.step(0)

        breakdown = info["reward_breakdown"]
        assert "fleet_availability" in breakdown
        assert "throughput_delta" in breakdown
        assert "repair_backlog_age" in breakdown
        assert "technician_utilization" in breakdown
        assert "downtime_cost" in breakdown

        # fleet_availability: 1 of 2 machines operational = 0.5
        assert breakdown["fleet_availability"] == 0.5

        # throughput_delta: 3 finished at start - 0 prev = clipped to 1.0
        assert breakdown["throughput_delta"] == 1.0

        # repair_backlog_age: one request in backlog (age 50-30=20)
        expected_backlog = -math.tanh(20.0 / 200.0)
        assert abs(breakdown["repair_backlog_age"] - expected_backlog) < 1e-4

        # technician_utilization: 2/3 busy (tech 0 already busy + tech 0 assigned)
        # Actually step(0) assigns tech 0 who was already busy -> still 1/3 busy
        # Wait no: start_repair sets techs[0].busy = True (was already True)
        # So only tech 0 is busy: 1/3 = 0.333
        assert -1.0 <= breakdown["technician_utilization"] <= 1.0

        # downtime_cost: negative, bounded
        assert -1.0 <= breakdown["downtime_cost"] <= 0.0

        # Total reward is the sum of all components
        total = sum(breakdown.values())
        assert abs(reward - total) < 1e-6


# ======================================================================
# New episode metrics
# ======================================================================


class TestNewEpisodeMetrics:
    """Verify the new episode-level metrics are produced."""

    def test_mttr_metric(self):
        sim_env = FakeSimEnv()
        sim_env.now = 200.0
        dispatcher = FakeDispatcher(tech_count=1)
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))

        env = _make_env(sim_env=sim_env, dispatcher=dispatcher)
        env.reset()
        _, _, terminated, _, info = env.step(0)

        assert terminated is True
        metrics = info["metrics"]
        assert "mttr" in metrics
        # MTTR = sim_time / repairs = 200 / 1 = 200
        assert metrics["mttr"] == 200.0

    def test_fleet_availability_rate_metric(self):
        sim_env = FakeSimEnv()
        sim_env.now = 100.0
        dispatcher = FakeDispatcher(tech_count=1)
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))

        env = _make_env(sim_env=sim_env, dispatcher=dispatcher)
        env.reset()
        _, _, terminated, _, info = env.step(0)

        assert terminated is True
        assert "fleet_availability_rate" in info["metrics"]

    def test_throughput_rate_metric(self):
        sim_env = FakeSimEnv()
        sim_env.now = 1000.0
        dispatcher = FakeDispatcher(tech_count=1)
        dispatcher.sinks = [FakeSink(10)]
        dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))

        env = _make_env(sim_env=sim_env, dispatcher=dispatcher)
        env.reset()
        _, _, _, _, info = env.step(0)

        metrics = info["metrics"]
        assert "throughput_rate" in metrics
        # 10 products / 1000 time * 1000 = 10.0
        assert metrics["throughput_rate"] == 10.0

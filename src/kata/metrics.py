"""Environment metrics for monitoring decision quality beyond rewards.

Metrics are divided into two categories:

- **Step metrics** — computed after each assignment action, capturing the
  quality of that specific decision (e.g. repair time delta, technician
  skill match).
- **Episode metrics** — computed once at episode end from cumulative
  simulation state (e.g. total breakdowns, completed products).

Both follow the same pattern as reward components: each metric is a
small class with a ``compute`` method that receives the environment
state.  New metrics can be added by subclassing ``StepMetric`` or
``EpisodeMetric`` and appending an instance to :data:`STEP_METRICS`
or :data:`EPISODE_METRICS`.

Metrics are returned in the ``info["metrics"]`` dict and are **never**
added to the reward signal — they are purely observational.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# ======================================================================
# Base classes
# ======================================================================


class StepMetric(ABC):
    """A metric computed after each technician-assignment action."""

    name: str  # short identifier used as dict key

    @abstractmethod
    def compute(
        self,
        tech: Any,
        request: Any,
        env: Any,
    ) -> float:
        """Return the metric value for this assignment.

        Parameters
        ----------
        tech:
            The technician that was assigned (``GymTechnician``).
        request:
            The repair request being serviced (``RepairRequest``).
        env:
            The ``KataEnv`` instance (for accessing factory state).

        """


class EpisodeMetric(ABC):
    """A metric computed once at episode end."""

    name: str

    @abstractmethod
    def compute(self, env: Any) -> float:
        """Return the metric value for the completed episode.

        Parameters
        ----------
        env:
            The ``KataEnv`` instance.

        """


# ======================================================================
# Step metrics
# ======================================================================


class RepairTimeDelta(StepMetric):
    """Difference between base and effective repair time.

    Positive means the chosen technician is *slower* than the base time
    (bad); negative means faster (good).  The raw value is in simulation
    time units so it is directly interpretable.
    """

    name = "repair_time_delta"

    def compute(self, tech: Any, request: Any, env: Any) -> float:
        base = (
            float(request.get_repair_time())
            if hasattr(request, "get_repair_time")
            else 10.0
        )
        compute_fn = getattr(tech, "compute_repair_time", None)
        if compute_fn is not None and callable(compute_fn):
            effective = float(compute_fn(int(base), request))
        else:
            effective = base
        return effective - base


class RepairQuality(StepMetric):
    """Skill-based repair quality of the chosen technician.

    Returns a score in ``[0, 1]`` where 1.0 means the technician has
    full expertise for this repair type and 0.0 means none.

    In the current simulation all repairs restore the component to full
    health ("perfect repair"), so this metric captures the *competence*
    dimension of quality — how well the tech's knowledge matches the
    failure — rather than a partial-health outcome.
    """

    name = "repair_quality"

    def compute(self, tech: Any, request: Any, env: Any) -> float:
        get_km = getattr(tech, "get_knowledge_multiplier", None)
        if get_km is None or not callable(get_km):
            return 0.0
        # multiplier ∈ (0, 1]: 1 = no knowledge, near 0 = full expertise
        multiplier = float(get_km(request))
        return max(0.0, min(1.0, 1.0 - multiplier))


# ======================================================================
# Episode metrics
# ======================================================================


class TotalBreakdowns(EpisodeMetric):
    """Count of machine breakdowns during the episode."""

    name = "total_breakdowns"

    def compute(self, env: Any) -> float:
        return float(getattr(env, "_breakdown_counter", 0))


class TotalRepairs(EpisodeMetric):
    """Count of completed repairs during the episode."""

    name = "total_repairs"

    def compute(self, env: Any) -> float:
        return float(getattr(env, "_repair_counter", 0))


class FinishedProducts(EpisodeMetric):
    """Count of products that reached the sink during the episode."""

    name = "finished_products"

    def compute(self, env: Any) -> float:
        sinks = getattr(env.dispatcher, "sinks", [])
        return float(sum(getattr(s, "completed", 0) for s in sinks))


class MeanTimeToRepair(EpisodeMetric):
    """Average repair duration across all completed repairs (MTTR).

    Computed as ``total_sim_time / total_repairs``.  Lower is better —
    indicates the assignment policy is selecting skilled, rested
    technicians.  Returns 0 if no repairs were completed.
    """

    name = "mttr"

    def compute(self, env: Any) -> float:
        repairs = getattr(env, "_repair_counter", 0)
        if repairs == 0:
            return 0.0
        sim_time = float(getattr(env.sim_env, "now", 0.0))
        return sim_time / repairs


class FleetAvailabilityRate(EpisodeMetric):
    """Fraction of machine-time spent operational (OEE availability).

    Computed as ``1 - (total_downtime / total_machine_time)``.  Higher
    is better.  Returns 1.0 at episode start.
    """

    name = "fleet_availability_rate"

    def compute(self, env: Any) -> float:
        machines = env._factory_machines()
        now = float(getattr(env.sim_env, "now", 0.0))
        total_available = max(now * len(machines), 1.0)

        # Accumulate finished + still-broken downtime
        total_down = getattr(env, "_total_downtime", 0.0)
        down_since = getattr(env, "_machine_down_since", {})
        active_down = sum(now - t0 for t0 in down_since.values())
        return max(0.0, 1.0 - (total_down + active_down) / total_available)


class ThroughputRate(EpisodeMetric):
    """Products completed per 1000 simulation time units.

    Normalised by sim time so the metric is comparable across episodes
    with different ``max_sim_time`` settings.
    """

    name = "throughput_rate"

    def compute(self, env: Any) -> float:
        sinks = getattr(env.dispatcher, "sinks", [])
        finished = sum(getattr(s, "completed", 0) for s in sinks)
        sim_time = float(getattr(env.sim_env, "now", 0.0))
        if sim_time <= 0:
            return 0.0
        return float(finished) / sim_time * 1000.0


# ======================================================================
# Metric registries — append new instances to extend
# ======================================================================

STEP_METRICS: list[StepMetric] = [
    RepairTimeDelta(),
    RepairQuality(),
]

EPISODE_METRICS: list[EpisodeMetric] = [
    TotalBreakdowns(),
    TotalRepairs(),
    FinishedProducts(),
    MeanTimeToRepair(),
    FleetAvailabilityRate(),
    ThroughputRate(),
]

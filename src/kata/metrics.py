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
    """Time saved by the chosen technician versus the base repair time.

    Computed as ``base - effective`` (in simulation time units): the
    knowledge multiplier and the fatigue multiplier are both in
    ``(0, 1]`` so ``effective <= base`` and the metric is non-negative.
    Higher is better — the chosen technician is faster than a baseline
    one with no knowledge and no fatigue would be.

    The raw value (rather than a ratio) is reported so it is directly
    interpretable in time units; divide by the request's base repair
    time if a normalised speed-up is needed.
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
            effective = float(compute_fn(base, request))
        else:
            effective = base
        return max(0.0, base - effective)


class RepairTimeDeltaPercent(StepMetric):
    """Time saved by the chosen technician as a percentage of the base.

    Same idea as :class:`RepairTimeDelta` but normalised by the
    request's base repair time so values are comparable across
    components with different absolute repair durations:

    ``(base - effective) / base * 100`` clipped to ``[0, 100]``.

    A value of ``0`` means no speed-up over a fresh, no-knowledge
    technician; ``100`` means the chosen technician would resolve the
    repair instantaneously.
    """

    name = "repair_time_delta_per"

    def compute(self, tech: Any, request: Any, env: Any) -> float:
        base = (
            float(request.get_repair_time())
            if hasattr(request, "get_repair_time")
            else 10.0
        )
        if base <= 0.0:
            return 0.0
        compute_fn = getattr(tech, "compute_repair_time", None)
        if compute_fn is not None and callable(compute_fn):
            effective = float(compute_fn(base, request))
        else:
            effective = base
        pct = (base - effective) / base * 100.0
        return max(0.0, min(100.0, pct))


def _fleet_labels(techs: list[Any]) -> dict[Any, str]:
    """Return a ``{tech: tech.name}`` mapping.

    Technician name uniqueness is enforced at environment initialisation
    (see ``KataEnv._bootstrap_scenario``), so labels are simply the
    technician's ``name`` attribute.  Duplicate names would silently
    overwrite each other in the resulting plot dict — the env-level
    check fails fast instead.
    """
    return {t: str(getattr(t, "name", f"tech_{getattr(t, 'id', '?')}")) for t in techs}


class TechnicianKnowledge(StepMetric):
    """Per-technician knowledge level.

    Returns a dict ``{label: max_knowledge}`` with one entry per
    technician in the dispatcher.  The env unpacks this into individual
    step series ``tech_knowledge/<label>`` so the evolution of each
    technician's expertise can be plotted at episode end.

    ``max_knowledge`` is the peak value over the knowledge grid — a
    monotonically non-decreasing scalar in ``[0, +inf)`` in this
    knowledge model.
    """

    name = "tech_knowledge"

    def compute(self, tech: Any, request: Any, env: Any) -> dict[str, float]:
        _ = tech, request
        techs = getattr(env.dispatcher, "techs", [])
        labels = _fleet_labels(techs)
        out: dict[str, float] = {}
        for t in techs:
            grid = getattr(t, "knowledge_grid", None)
            if grid is not None and hasattr(grid, "knowledge_volume"):
                value = float(grid.knowledge_volume())
            else:
                # Fallback for technicians without a knowledge grid
                # (e.g. the lightweight ``FakeTech`` used in tests).
                value = float(getattr(t, "knowledge", 0.0))
            out[labels[t]] = value
        return out


class TechnicianSpecializationIndex(StepMetric):
    """Per-technician specialization index.

    Returns a dict ``{label: specialization_index}`` with one entry per
    technician in the dispatcher.  The env unpacks this into individual
    step series ``tech_specialization/<label>`` so the evolution of each
    technician's specialization can be plotted at episode end.

    The specialization index is computed as the normalized entropy of the
    technician's knowledge distribution across repair types, giving a
    value in ``[0, 1]`` where 0 means fully specialized (all knowledge
    concentrated in one repair type) and 1 means fully generalized
    (knowledge evenly distributed across all repair types).
    """

    name = "tech_specialization"

    def compute(self, tech: Any, request: Any, env: Any) -> dict[str, float]:
        _ = tech, request
        techs = getattr(env.dispatcher, "techs", [])
        labels = _fleet_labels(techs)
        out: dict[str, float] = {}
        for t in techs:
            grid = getattr(t, "knowledge_grid", None)
            if grid is not None and hasattr(grid, "specialization_index"):
                value = float(grid.specialization_index())
            else:
                # Fallback for technicians without a knowledge grid
                value = 1.0  # Assume fully generalized if no knowledge info
            out[labels[t]] = value
        return out


class TechnicianFatigue(StepMetric):
    """Per-technician fatigue level, sampled at every assignment step.

    Returns a dict ``{label: fatigue}`` with one entry per technician.
    The env unpacks this into individual step series
    ``tech_fatigue/<label>`` and the runner renders them as a single
    multi-line matplotlib figure per episode (``train/metrics/
    tech_fatigue_episode``).

    Fatigue is in ``[0, 1]`` — 0 fresh, 1 fully exhausted.  The value
    is updated by the simulator at repair completion, so what is
    sampled here is the most recent post-repair fatigue.
    """

    name = "tech_fatigue"

    def compute(self, tech: Any, request: Any, env: Any) -> dict[str, float]:
        _ = tech, request
        techs = getattr(env.dispatcher, "techs", [])
        labels = _fleet_labels(techs)
        return {labels[t]: float(getattr(t, "fatigue", 0.0)) for t in techs}


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


class TotalAssignments(EpisodeMetric):
    """Count of repair assignments dispatched during the episode.

    Increments at every ``env.step()`` that picked a technician. Some
    assignments may still be in-flight at episode end — see
    ``total_completed_repairs`` for the count of repairs that actually
    finished within the episode.
    """

    name = "total_assignments"

    def compute(self, env: Any) -> float:
        return float(getattr(env, "_repair_counter", 0))


class TotalRepairs(EpisodeMetric):
    """Count of repairs that actually completed during the episode.

    Driven by the dispatcher's ``on_repair_completed`` callback, so
    only counts assignments whose SimPy job ran to completion before
    the episode terminated.
    """

    name = "total_repairs"

    def compute(self, env: Any) -> float:
        return float(getattr(env, "_completed_repair_counter", 0))


class IllTechnicianCount(EpisodeMetric):
    """Number of technicians that experienced at least one disruption.

    Counts the distinct technicians whose ``disruption_count`` is
    non-zero — i.e. technicians that went on sick leave (or any other
    stochastic disruption configured under
    ``sim.disruptions.dis_dict``) at least once during the episode.

    With the current dispatcher, each technician runs a single one-shot
    disruption process per episode so this metric is bounded above by
    ``len(dispatcher.techs)``.
    """

    name = "ill_technician_count"

    def compute(self, env: Any) -> float:
        techs = getattr(env.dispatcher, "techs", [])
        return float(
            sum(
                t.disruption_count
                for t in techs
                if int(getattr(t, "disruption_count", 0)) > 0
            )
        )


class FinishedProducts(EpisodeMetric):
    """Count of products that reached the sink during the episode."""

    name = "finished_products"

    def compute(self, env: Any) -> float:
        sinks = getattr(env.dispatcher, "sinks", [])
        return float(sum(getattr(s, "completed", 0) for s in sinks))


class MeanTimeToRepair(EpisodeMetric):
    """Mean duration of one completed repair (MTTR).

    Computed as ``total_repair_time / total_completed_repairs`` where
    ``total_repair_time`` is the sum of actual on-tool repair durations
    reported by the dispatcher (excluding queueing and travel).  Lower
    is better — a skilled, rested fleet brings this down.  Returns 0
    when no repair has completed.

    NOTE: this is the textbook MTTR definition, not
    ``sim_time / repair_count`` (which is mean *time between* repairs
    and conflates demand intensity with repair speed).
    """

    name = "mttr"

    def compute(self, env: Any) -> float:
        repairs = getattr(env, "_completed_repair_counter", 0)
        if repairs == 0:
            return 0.0
        total = float(getattr(env, "_total_repair_time", 0.0))
        return total / repairs


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
    RepairTimeDeltaPercent(),
    RepairQuality(),
    TechnicianKnowledge(),
    TechnicianFatigue(),
    TechnicianSpecializationIndex(),
]

EPISODE_METRICS: list[EpisodeMetric] = [
    TotalBreakdowns(),
    TotalAssignments(),
    TotalRepairs(),
    IllTechnicianCount(),
    FinishedProducts(),
    MeanTimeToRepair(),
    FleetAvailabilityRate(),
    ThroughputRate(),
]

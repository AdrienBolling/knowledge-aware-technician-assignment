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
    """Signed time delta of the chosen technician versus the base repair time.

    Computed as ``base - effective`` (in simulation time units):

    * **positive** — chosen technician is faster than a baseline tech
      with no knowledge and no fatigue would be (a speed-up);
    * **negative** — chosen technician is slower (a fatigue-dominated
      slowdown).

    The knowledge multiplier sits in ``[min_repair_fraction, 1]`` so it
    can only speed up the repair; the fatigue multiplier sits in
    ``[1, +∞)`` and can push ``effective`` above ``base``.  The signed
    delta is reported so the episode-mean is unbiased — clamping at 0
    would hide slowdowns and inflate the apparent benefit of the chosen
    tech.

    Divide by ``request.get_repair_time()`` if you need a normalised
    speed-up; see :class:`RepairTimeDeltaPercent`.
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
        return base - effective


class RepairTimeDeltaPercent(StepMetric):
    """Signed time delta as a percentage of the request's base repair time.

    Same idea as :class:`RepairTimeDelta` but normalised so values are
    comparable across components with different absolute durations:

    ``(base - effective) / base * 100``.

    * ``0``        — chosen technician matches the no-knowledge,
                     no-fatigue baseline.
    * **positive** — a speed-up.  Bounded above by
                     ``(1 - min_repair_fraction) * 100`` because the
                     knowledge multiplier cannot drop below
                     ``min_repair_fraction`` (the per-component or
                     global floor).  With the default floor of 0.3 the
                     practical maximum is ~70%, NOT 100% — 100% would
                     require an instantaneous repair, which the floor
                     forbids.
    * **negative** — a slowdown, driven by the fatigue multiplier
                     (which is unbounded above) overpowering whatever
                     knowledge speed-up was available.

    Reported as a signed value (no lower clamp).  Clamping slowdowns to
    0 would silently hide them and bias the episode-mean upward — making
    speedup-poor agents look better than they are.
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
        return (base - effective) / base * 100.0


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


class MeanTimeToRepairRolling(StepMetric):
    """Rolling MTTR over the most recent completed repairs (per assignment step).

    Mean of ``env._recent_repair_durations`` — a deque bounded by
    ``GymEnvConfig.mttr_rolling_window`` and appended to by
    :meth:`KataEnv._on_repair_completed`.  At each assignment step this
    returns the average repair duration over the last ``window`` (or
    fewer, before the window is full) completed repairs, so the runner
    plots a within-episode trend of how repair speed evolves.

    Lower is better — a fleet that's getting better at its repairs
    drives this down.  Returns ``0.0`` until the first repair completes.

    Distinct from :class:`MeanTimeToRepair` (the cumulative episode-end
    metric).  The two agree by construction at the very start
    (single sample) and drift apart as the window slides.
    """

    name = "mttr_rolling"

    def compute(self, tech: Any, request: Any, env: Any) -> float:
        _ = tech, request
        deque_ = getattr(env, "_recent_repair_durations", None)
        if deque_ is None or len(deque_) == 0:
            return 0.0
        return float(sum(deque_)) / len(deque_)


class RepairQuality(StepMetric):
    """Skill-based repair quality of the chosen technician.

    Defined as ``1 - knowledge_multiplier`` where the multiplier sits in
    ``[min_repair_fraction, 1]``.  Reported range is therefore
    ``[0, 1 - min_repair_fraction]``:

    * ``0``                       — chosen technician has no knowledge
                                    of this failure type.
    * ``1 - min_repair_fraction`` — chosen technician has saturated
                                    knowledge (e.g. ~0.7 at the default
                                    floor of 0.3).  This is the
                                    practical maximum; a value of 1.0
                                    would mean an instantaneous repair,
                                    which the floor forbids.

    Uses :meth:`GymTechnician.get_knowledge_multiplier`, so per-component
    overrides of ``min_repair_fraction`` and ``knowledge_sensitivity``
    are honoured when
    ``sim.repair.failure_wise_knowledge_parameters=True``.

    Fatigue is intentionally *not* in this metric — quality measures
    competence (knowledge-failure match), not speed; the speed cost of
    fatigue is captured by :class:`RepairTimeDelta(Percent)` instead.

    In the current simulation all repairs restore the component to full
    health, so this metric captures the *competence* dimension of
    quality rather than a partial-health outcome.
    """

    name = "repair_quality"

    def compute(self, tech: Any, request: Any, env: Any) -> float:
        get_km = getattr(tech, "get_knowledge_multiplier", None)
        if get_km is None or not callable(get_km):
            return 0.0
        # multiplier ∈ [min_floor, 1]: 1 = no knowledge, min_floor = saturated.
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


class MeanTimeBetweenFailures(EpisodeMetric):
    """Mean fleet uptime per breakdown (MTBF).

    Computed as ``total_uptime / total_breakdowns`` where
    ``total_uptime = sim_time * n_machines - total_downtime`` aggregates
    the operational machine-time across the fleet, and
    ``total_breakdowns`` sums per-machine breakdown counts captured at
    the moment each machine transitions to broken (so pending tickets
    still queued at the agent are included).  Higher is better — a
    well-maintained fleet sees longer stretches between failures.
    Returns 0 when no breakdown has occurred yet.
    """

    name = "mtbf"

    def compute(self, env: Any) -> float:
        counts = getattr(env, "_machine_breakdown_counts", {})
        total_breakdowns = float(sum(counts.values()))
        if total_breakdowns <= 0.0:
            return 0.0
        machines = env._factory_machines()
        now = float(getattr(env.sim_env, "now", 0.0))
        total_machine_time = now * len(machines)
        total_down = float(getattr(env, "_total_downtime", 0.0))
        down_since = getattr(env, "_machine_down_since", {})
        active_down = sum(now - t0 for t0 in down_since.values())
        uptime = max(0.0, total_machine_time - total_down - active_down)
        return uptime / total_breakdowns


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


class WorkloadBalance(EpisodeMetric):
    """How evenly assignments were spread across the technician fleet.

    Reports Jain's fairness index on ``env._tech_assignment_counts``:

    ``J(x) = (sum x)^2 / (n * sum x^2)``

    Returns a value in ``[1/n, 1]``: ``1.0`` when every technician got
    the same number of assignments, ``1/n`` when one technician got
    them all.  Returns ``1.0`` when no assignment has been made yet
    (vacuously balanced).  Independent of total assignment volume so
    it is comparable across episodes of different lengths.
    """

    name = "workload_balance"

    def compute(self, env: Any) -> float:
        counts = list(getattr(env, "_tech_assignment_counts", []))
        n = len(counts)
        if n == 0:
            return 1.0
        total = float(sum(counts))
        if total <= 0.0:
            return 1.0
        sq = float(sum(c * c for c in counts))
        if sq <= 0.0:
            return 1.0
        return (total * total) / (n * sq)


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
    MeanTimeToRepairRolling(),
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
    MeanTimeBetweenFailures(),
    FleetAvailabilityRate(),
    WorkloadBalance(),
    ThroughputRate(),
]

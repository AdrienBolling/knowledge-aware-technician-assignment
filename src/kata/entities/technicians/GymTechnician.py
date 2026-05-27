"""GymTechnician with fatigue, knowledge, and disruption modelling."""  # noqa: N999

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import simpy
from ongoing import KnowledgeGrid

from kata.core.config import TechnicianConfig, get_config
from kata.entities.technicians.base import Technician

if TYPE_CHECKING:
    from kata.entities.encoder.base import RequestEncoder
    from kata.entities.machines.base import Machine
    from kata.entities.requests.RepairRequest import RepairRequest

CONFIG = get_config()


class GymTechnician(Technician):
    """Technician for Gym environments with knowledge and fatigue modelling."""

    _id_counter = 0

    def __init__(
        self,
        tech_conf: TechnicianConfig,
        encoder: RequestEncoder | None = None,
    ) -> None:
        """Initialise from a TechnicianConfig, optionally injecting an encoder."""
        self.id = GymTechnician._id_counter
        GymTechnician._id_counter += 1
        self.name = tech_conf.name
        self.busy: bool = False

        # Fatigue parameters.  ``self._fatigue`` is the raw, event-driven
        # base value, written by ``_increase_fatigue`` / ``_recover_fatigue``
        # at repair-finished / start-of-next-repair boundaries.  External
        # consumers (observations, metrics, rewards, exhaustion loop) read
        # ``self.fatigue`` --- a property that adds *continuous* recovery
        # for any idle time elapsed since the last event, using the env
        # clock injected by the dispatcher via ``self.env``.
        self._fatigue: float = 0.0
        self.fatigue_lambda: float = tech_conf.fatigue_lambda
        self.fatigue_mu: float = tech_conf.fatigue_mu
        # SimPy environment reference --- set by the dispatcher once it
        # has the technician in its fleet.  Until then the fatigue
        # property falls back to the raw base value (back-compat for
        # unit tests that exercise the technician in isolation).
        self.env: simpy.Environment | None = None

        # Set True for the duration of a stochastic disruption (vacation,
        # injury, exhaustion).  Distinct from ``busy`` --- which flips
        # only during a *repair* --- because the fatigue property
        # *should* keep applying recovery during a disruption (you do
        # rest while on vacation) while the action mask *should* still
        # treat the tech as unavailable for new assignments.
        self._in_disruption: bool = False

        # Per-technician RNG used by the stochastic-disruption loops.
        # Seeded by the dispatcher via ``seed_disruptions`` so that
        # ``env.reset(seed=…)`` produces deterministic disruption
        # timings.  Falls back to a fresh non-seeded generator when no
        # seed is provided.
        self._rng: np.random.Generator = np.random.default_rng()

        # Number of stochastic disruptions (sick leaves, …) this
        # technician has experienced.  Reset to 0 on each scenario build
        # since technicians are re-instantiated by the scenario builder
        # at every ``env.reset``.  ``disruption_counts_by_type`` carries
        # the same total broken out by the named type (``injury``,
        # ``exhaustion``, ``vacation``, …) — used by diagnostic scripts
        # and metrics that want a per-trigger view.
        self.disruption_count: int = 0
        self.disruption_counts_by_type: dict[str, int] = {}

        # Simulation time at which the technician last became idle.
        # Used by ``start_repair`` to drive fatigue recovery: the gap
        # between this timestamp and the moment the tech actually
        # starts the next repair is the idle interval.  Initialised
        # to 0.0 (the scenario start) so the first repair correctly
        # accounts for any pre-work idle time, but with fatigue at 0
        # the recovery is a no-op then anyway.
        self._last_idle_since: float = 0.0

        # Knowledge parameters
        self.k_shape = tech_conf.knowledge_k_shape
        self.k_propagation_sigma = tech_conf.knowledge_propagation_sigma
        self.k_transmission_factor = tech_conf.knowledge_transmission_factor
        self.k_learning_rate = tech_conf.knowledge_learning_rate
        self.k_methods = ["propagation"]

        self.knowledge_grid: KnowledgeGrid = KnowledgeGrid(
            shape=self.k_shape,
            propagation_sigma=self.k_propagation_sigma,
            transmission_factor=self.k_transmission_factor,
            learning_rate=self.k_learning_rate,
            methods=self.k_methods,
        )

        # Optionally seed the grid from a saved profile so the
        # technician starts the episode with realistic career
        # experience.  The legacy behaviour (empty grid) is preserved
        # when ``initial_knowledge_grid_path`` is ``None`` or when the
        # file does not exist --- we warn rather than crash so a
        # missing artefact does not break training entirely.
        init_path = getattr(tech_conf, "initial_knowledge_grid_path", None)
        if init_path:
            from pathlib import Path as _Path

            resolved = _Path(init_path)
            if resolved.is_file():
                loaded = KnowledgeGrid.load(resolved)
                # Sanity-check: the loaded grid must agree with the
                # configured shape AND with the other knowledge
                # hyperparameters --- otherwise future
                # ``add_ticket_knowledge`` calls (which run with the
                # *loaded* grid's parameters) would silently disagree
                # with the technician config.  We fail loudly so a
                # mismatch is caught at scenario-build time rather
                # than producing subtly wrong dynamics throughout
                # an episode.
                checks = [
                    ("shape", tuple(loaded._shape), tuple(self.k_shape)),
                    (
                        "propagation_sigma",
                        float(loaded._propagation_sigma),
                        float(self.k_propagation_sigma),
                    ),
                    (
                        "transmission_factor",
                        float(loaded._transmission_factor),
                        float(self.k_transmission_factor),
                    ),
                    (
                        "learning_rate",
                        float(loaded._learning_rate),
                        float(self.k_learning_rate),
                    ),
                ]
                for label, file_val, cfg_val in checks:
                    if file_val != cfg_val:
                        msg = (
                            f"initial_knowledge_grid_path {init_path!r} "
                            f"has {label}={file_val} but technician is "
                            f"configured for {label}={cfg_val}. "
                            f"Rebuild the profile (see "
                            f"``kata.EntityFactories.technician_profile_builder``) "
                            f"or align the config."
                        )
                        raise ValueError(msg)
                self.knowledge_grid = loaded
            else:
                import warnings as _warnings

                _warnings.warn(
                    f"initial_knowledge_grid_path={init_path!r} does not "
                    f"exist; starting with an empty grid instead.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Encoder (lazy import of default if not provided)
        if encoder is not None:
            self._encoder: RequestEncoder = encoder
        else:
            self._encoder = None  # type: ignore[assignment]

    @property
    def encoder(self) -> RequestEncoder:
        """Return the encoder, lazily importing the default on first use."""
        if self._encoder is None:
            from kata.entities.encoder.base import ENCODER

            self._encoder = ENCODER
        return self._encoder

    def travel_time(self, machine: Machine) -> int:
        """Return the travel time to the given machine."""
        _ = machine
        return CONFIG.sim.technicians.travel_time

    # ------------------------------------------------------------------
    # Stochastic disruptions
    # ------------------------------------------------------------------
    #
    # Each named disruption type configured in ``CONFIG.sim.disruptions.dis_dict``
    # is realised as its own long-running SimPy process per technician.
    # The dispatcher is responsible for spawning these processes (one
    # per (tech, type) pair) at construction time; here we expose the
    # loop bodies.  All loops:
    #
    #   * run for the entire episode (``while True``),
    #   * yield to the env until their trigger fires,
    #   * acquire the technician's resource at priority 0 (above repairs
    #     at priority 1) with ``preempt=cfg.preemptive``,
    #   * hold the resource for a Normal(``duration_mu``, ``duration_sig``)
    #     sample, then release it and loop.
    #
    # If a disruption with ``preempt=True`` claims the resource while a
    # repair is in progress, the repair's ``yield`` raises ``simpy.Interrupt``
    # and the dispatcher re-queues the partially-completed ticket.

    def run_disruption_process(
        self,
        env: simpy.Environment,
        tech_resource: simpy.PreemptiveResource,
        dis_name: str,
        dis_cfg: Any,
    ):  # SimPy generator
        """Drive ``dis_cfg.trigger``-typed disruptions for this technician."""
        trigger = dis_cfg.trigger
        if trigger == "random":
            yield from self._random_disruption_loop(
                env, tech_resource, dis_name, dis_cfg
            )
        elif trigger == "fatigue":
            yield from self._fatigue_disruption_loop(
                env, tech_resource, dis_name, dis_cfg
            )
        elif trigger == "periodic":
            yield from self._periodic_disruption_loop(
                env, tech_resource, dis_name, dis_cfg
            )
        else:
            msg = f"Unknown disruption trigger: {trigger}"
            raise ValueError(msg)

    def _take_disruption(
        self,
        env: simpy.Environment,
        tech_resource: simpy.PreemptiveResource,
        dis_name: str,
        dis_cfg: Any,
    ):
        """Claim the technician resource and hold it for a sampled duration.

        Sets ``self._in_disruption = True`` for the duration of the hold
        so the env's action mask correctly treats the technician as
        unavailable.  Fatigue continues to recover during the hold
        because the fatigue property only checks ``busy`` (which is
        repair-only), not ``_in_disruption``.
        """
        with tech_resource.request(priority=0, preempt=bool(dis_cfg.preemptive)) as req:
            yield req
            self._in_disruption = True
            try:
                self.disruption_count += 1
                self.disruption_counts_by_type[dis_name] = (
                    self.disruption_counts_by_type.get(dis_name, 0) + 1
                )
                duration = self._sample_duration(dis_cfg)
                yield env.timeout(duration)
            finally:
                self._in_disruption = False

    def _sample_duration(self, dis_cfg: Any) -> float:
        """Sample a strictly-positive duration from the configured Normal."""
        sample = float(
            self._rng.normal(float(dis_cfg.duration_mu), float(dis_cfg.duration_sig))
        )
        return sample if sample > 0 else float(dis_cfg.duration_mu)

    def _random_disruption_loop(
        self,
        env: simpy.Environment,
        tech_resource: simpy.PreemptiveResource,
        dis_name: str,
        dis_cfg: Any,
    ):
        """Poisson process: exponential inter-arrival times with mean ``1/rate``."""
        rate = float(dis_cfg.rate or 0.0)
        if rate <= 0.0:
            return
        scale = 1.0 / rate
        while True:
            yield env.timeout(float(self._rng.exponential(scale)))
            yield from self._take_disruption(env, tech_resource, dis_name, dis_cfg)

    def _fatigue_disruption_loop(
        self,
        env: simpy.Environment,
        tech_resource: simpy.PreemptiveResource,
        dis_name: str,
        dis_cfg: Any,
    ):
        """Polling Bernoulli: fires with probability ``coef * fatigue * poll`` per tick."""
        coef = float(dis_cfg.fatigue_coefficient or 0.0)
        poll = float(dis_cfg.poll_interval)
        if coef <= 0.0:
            return
        while True:
            yield env.timeout(poll)
            p = min(1.0, coef * float(self.fatigue) * poll)
            if p > 0.0 and self._rng.random() < p:
                yield from self._take_disruption(env, tech_resource, dis_name, dis_cfg)

    def _periodic_disruption_loop(
        self,
        env: simpy.Environment,
        tech_resource: simpy.PreemptiveResource,
        dis_name: str,
        dis_cfg: Any,
    ):
        """Strictly-periodic schedule with optional uniform jitter."""
        interval = float(dis_cfg.interval or 0.0)
        jitter = float(dis_cfg.jitter)
        if interval <= 0.0:
            return
        # Stagger initial offset by up to one full interval so that
        # multiple technicians don't all vacation at sim time = interval.
        yield env.timeout(float(self._rng.uniform(0.0, interval)))
        while True:
            yield from self._take_disruption(env, tech_resource, dis_name, dis_cfg)
            wait = interval
            if jitter > 0.0:
                wait += float(self._rng.uniform(-jitter, jitter))
            yield env.timeout(max(1.0, wait))

    def compute_repair_time(
        self,
        base_repair_time: float,
        request: RepairRequest,
    ) -> float:
        """Compute the effective repair duration in simulation time units.

        Applies two multipliers on top of ``base_repair_time``:

        * **knowledge multiplier** ``m_k ∈ [min_repair_fraction, 1]`` —
          lower means more skilled, so the repair finishes faster;
        * **fatigue multiplier** ``m_f ∈ [1, +∞)`` — higher means
          more tired, so the repair *takes longer*.

        Effective time ``= base × m_k × m_f``.  Note that the floor
        ``min_repair_fraction`` only constrains ``m_k``; a fatigued
        technician can still push the effective time above ``base``.

        Returns a non-negative ``float`` — no integer truncation, no
        artificial floor.  SimPy's ``timeout`` accepts floats so the
        dispatcher consumes this directly.
        """
        base: float = float(base_repair_time)
        if CONFIG.sim.repair.knowledge_enabled:
            base *= self.get_knowledge_multiplier(request)
        if CONFIG.sim.repair.fatigue_enabled:
            base *= self.get_fatigue_multiplier()
        return max(0.0, base)

    def increase_knowledge(self, request: RepairRequest) -> None:
        """Increase knowledge based on the completed repair."""
        embedding = self.encoder.encode(request)
        self.knowledge_grid.add_ticket_knowledge(embedding)

    def get_knowledge_multiplier(self, request: RepairRequest) -> float:
        """Return knowledge-based repair time multiplier in ``[min_floor, 1]``.

        Saturating-exponential response::

            m_k = min_floor + (1 - min_floor) * exp(-alpha * k)

        where ``min_floor`` and ``alpha`` are by default taken from
        ``sim.repair.min_repair_fraction`` / ``sim.repair.knowledge_sensitivity``.

        When ``sim.repair.failure_wise_knowledge_parameters`` is True the
        simulator looks at the *failed component* first and uses any
        per-component override it carries (``ComponentConfig.min_repair_fraction`` /
        ``ComponentConfig.knowledge_sensitivity``), falling back to the
        global value for whichever parameter is ``None``.

        - No experience (k = 0) → multiplier = 1 (full base repair time).
        - High experience (k → ∞) → multiplier → ``min_floor``.
        """
        embedding = self.encoder.encode(request)
        knowledge = float(self.knowledge_grid.get_knowledge(embedding))

        cfg = CONFIG.sim.repair
        min_floor = float(cfg.min_repair_fraction)
        alpha = float(cfg.knowledge_sensitivity)

        if getattr(cfg, "failure_wise_knowledge_parameters", False):
            getter = getattr(request, "get_knowledge_parameters", None)
            params = getter() if callable(getter) else None
            if params is not None:
                per_floor, per_alpha = params
                if per_floor is not None:
                    min_floor = float(per_floor)
                if per_alpha is not None:
                    alpha = float(per_alpha)

        return min_floor + (1.0 - min_floor) * math.exp(-alpha * knowledge)

    def decay_knowledge(self) -> None:
        """Decay knowledge over time."""
        self.knowledge_grid.decay_knowledge()

    def get_fatigue_multiplier(self) -> float:
        """Return fatigue-based repair time multiplier.

        Fatigue *slows* repairs — the multiplier is ``>= 1`` and grows
        as the technician gets tired:

        - ``linear``      : ``multiplier = 1 + fatigue``
                            (range ``[1, 2]``)
        - ``exponential`` : ``multiplier = exp(fatigue_alpha * fatigue)``
                            (range ``[1, exp(alpha)]``)

        ``fatigue`` is a unit-interval scalar in ``[0, 1]`` (0 = fresh,
        1 = exhausted).  At ``fatigue = 0`` both models give ``1`` so a
        fresh technician's repair time is unchanged from the base.
        """
        model = CONFIG.sim.technicians.fatigue_model
        alpha = CONFIG.sim.technicians.fatigue_alpha

        if model == "linear":
            return 1.0 + self.fatigue
        if model == "exponential":
            return math.exp(alpha * self.fatigue)
        msg = f"Unknown fatigue model: {model}"
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # Fatigue: event-driven base value + continuous-recovery property
    # ------------------------------------------------------------------

    @property
    def fatigue(self) -> float:
        """Current fatigue level with continuous-time recovery applied.

        Returns the raw event-driven base value while the technician
        is performing a repair (``self.busy``), and otherwise decays
        it exponentially according to the Jaber recovery model:

            F(now) = F_base * exp(-mu * (now - last_idle_since))

        Disruption holds (vacation, injury, exhaustion) do *not*
        suspend recovery --- a technician on holiday is still
        resting --- so the recovery formula applies through them.
        Availability (whether the action mask considers the
        technician selectable) is tracked separately via
        ``self._in_disruption``; see ``KataEnv._action_mask``.

        Every external read --- observations, metrics, rewards, the
        fatigue-driven exhaustion disruption loop --- goes through
        this property so they see a fresh value rather than the
        stale snapshot last written by ``_increase_fatigue`` or
        ``_recover_fatigue``.
        """
        base = self._fatigue
        if self.busy or self.env is None:
            return base
        idle = float(self.env.now) - float(self._last_idle_since)
        if idle <= 0.0 or self.fatigue_mu <= 0.0:
            return base
        return float(base * math.exp(-self.fatigue_mu * idle))

    @fatigue.setter
    def fatigue(self, value: float) -> None:
        """Allow tests / external callers to seed the raw base value."""
        self._fatigue = float(value)

    def _increase_fatigue(self, work_time: int) -> None:
        """Accumulate fatigue after a repair.

        Reads / writes ``self._fatigue`` directly to bypass the
        continuous-recovery property: by the time this method runs,
        ``repair_finished`` has set ``busy = False`` but
        ``_last_idle_since`` is still the *previous* idle anchor, so
        going through the property would double-apply recovery.
        """
        if work_time < 0:
            msg = "Work time must be non-negative."
            raise ValueError(msg)
        base = self._fatigue
        base = base + (1.0 - base) * (
            1.0 - math.exp(-self.fatigue_lambda * work_time)
        )
        self._fatigue = min(1.0, max(0.0, base))

    def _recover_fatigue(self, idle_time: int) -> None:
        """Recover fatigue during idle time (event-driven snapshot)."""
        if idle_time < 0:
            msg = "Idle time must be non-negative."
            raise ValueError(msg)
        base = self._fatigue * math.exp(-self.fatigue_mu * idle_time)
        self._fatigue = min(1.0, max(0.0, base))

    def start_repair(self, when: float) -> None:
        """Mark the technician as starting a repair at simulation time ``when``.

        Recovers fatigue based on the idle time since this technician's
        last completed repair (or the start of the episode) before
        flipping ``busy = True``.  Called by the dispatcher's
        ``_repair_job`` once the SimPy resource has been acquired.
        """
        idle_time = max(0.0, float(when) - float(self._last_idle_since))
        if idle_time > 0:
            self._recover_fatigue(int(idle_time))
        self.busy = True

    def repair_finished(self, request: RepairRequest, when: float) -> None:
        """Update technician state after a repair is completed."""
        self.busy = False
        repair_time = request.get_repair_time()
        self._increase_fatigue(int(repair_time))
        self.increase_knowledge(request)
        # Bookkeeping for the next call to ``start_repair`` — anything
        # past this timestamp counts as idle and contributes to the
        # next recovery interval.
        self._last_idle_since = float(when)

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
        self._interrupt_on_disrupt: bool = CONFIG.sim.disruptions.interrupt_on_disrupt

        # Fatigue parameters
        self.fatigue: float = 0.0
        self.fatigue_lambda: float = tech_conf.fatigue_lambda
        self.fatigue_mu: float = tech_conf.fatigue_mu
        self.exhausted: bool = False

        # Number of stochastic disruptions (sick leaves, …) this
        # technician has experienced.  Reset to 0 on each scenario build
        # since technicians are re-instantiated by the scenario builder
        # at every ``env.reset``.
        self.disruption_count: int = 0

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

    def stochastic_disruptions_process(
        self,
        env: simpy.Environment,
        tech_resource: simpy.PreemptiveResource,
    ):  # SimPy generator
        """Stochastically require the technician resource for the disruption duration."""
        with tech_resource.request(
            priority=1, preempt=self._interrupt_on_disrupt
        ) as req:
            yield req
            # The technician has just gone on disruption (e.g. sick
            # leave) — record it for the ``ill_technician_count`` metric.
            self.disruption_count += 1
            yield env.timeout(self._get_stochastic_disruption_duration())

    def _get_stochastic_disruption_duration(self) -> int:
        """Return disruption duration sampled from configured distribution."""
        dis_type = self._get_stochastic_disruption_type()
        mu = CONFIG.sim.disruptions.dis_dict[dis_type]["mu"]
        sig = CONFIG.sim.disruptions.dis_dict[dis_type]["sig"]
        time: float = np.random.normal(mu, sig)
        return int(time if time > 0 else mu)

    def _get_stochastic_disruption_type(self) -> str:
        """Sample disruption type according to configured probabilities."""
        disruptions: dict[str, Any] = CONFIG.sim.disruptions.dis_dict
        types: list[str] = list(disruptions.keys())
        probs: list[float] = [disruptions[t]["prob"] for t in types]
        chosen_type: str = np.random.choice(a=types, p=probs)
        return chosen_type

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

        where ``min_floor = sim.repair.min_repair_fraction`` and
        ``alpha = sim.repair.knowledge_sensitivity``.

        - No experience (k = 0) → multiplier = 1 (full base repair time).
        - High experience (k → ∞) → multiplier → ``min_floor`` (bounded
          speed-up, never instant).
        """
        embedding = self.encoder.encode(request)
        knowledge = float(self.knowledge_grid.get_knowledge(embedding))

        min_floor = float(CONFIG.sim.repair.min_repair_fraction)
        alpha = float(CONFIG.sim.repair.knowledge_sensitivity)
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

    def _increase_fatigue(self, work_time: int) -> None:
        """Accumulate fatigue after a repair."""
        if work_time < 0:
            msg = "Work time must be non-negative."
            raise ValueError(msg)
        self.fatigue = self.fatigue + (1.0 - self.fatigue) * (
            1.0 - math.exp(-self.fatigue_lambda * work_time)
        )
        self.fatigue = min(1.0, max(0.0, self.fatigue))

    def _recover_fatigue(self, idle_time: int) -> None:
        """Recover fatigue during idle time."""
        if idle_time < 0:
            msg = "Idle time must be non-negative."
            raise ValueError(msg)
        self.fatigue = self.fatigue * math.exp(-self.fatigue_mu * idle_time)
        self.fatigue = min(1.0, max(0.0, self.fatigue))

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

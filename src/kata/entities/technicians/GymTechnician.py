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

    def compute_repair_time(self, base_repair_time: int, request: RepairRequest) -> int:
        """Compute repair time applying knowledge and fatigue multipliers."""
        base: float = float(base_repair_time)
        if CONFIG.sim.repair.knowledge_enabled:
            base *= self.get_knowledge_multiplier(request)
        if CONFIG.sim.repair.fatigue_enabled:
            base *= self.get_fatigue_multiplier()
        return max(1, int(base))

    def increase_knowledge(self, request: RepairRequest) -> None:
        """Increase knowledge based on the completed repair."""
        embedding = self.encoder.encode(request)
        self.knowledge_grid.add_ticket_knowledge(embedding)

    def get_knowledge_multiplier(self, request: RepairRequest) -> float:
        """Return knowledge-based repair time multiplier in (0, 1].

        Higher knowledge → lower multiplier → faster repair.
        ``get_knowledge()`` returns ``experiences^b`` (>= 0, unbounded).
        We map this to (0, 1] via ``1 / (1 + knowledge)`` so that:
        - No experience → multiplier = 1 (base repair time).
        - High experience → multiplier → 0 (near-instant repair).
        """
        embedding = self.encoder.encode(request)
        knowledge = self.knowledge_grid.get_knowledge(embedding)
        return 1.0 / (1.0 + knowledge)

    def decay_knowledge(self) -> None:
        """Decay knowledge over time."""
        self.knowledge_grid.decay_knowledge()

    def get_fatigue_multiplier(self) -> float:
        """Return fatigue-based repair time multiplier.

        - linear: multiplier = max(0, 1 - fatigue)
        - exponential: multiplier = exp(-fatigue_alpha * fatigue)
        """
        model = CONFIG.sim.technicians.fatigue_model
        alpha = CONFIG.sim.technicians.fatigue_alpha

        if model == "linear":
            return max(0.0, 1.0 - self.fatigue)
        if model == "exponential":
            return math.exp(-alpha * self.fatigue)
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

    def repair_finished(self, request: RepairRequest, when: float) -> None:
        """Update technician state after a repair is completed."""
        _ = when
        self.busy = False
        repair_time = request.get_repair_time()
        self._increase_fatigue(int(repair_time))
        self.increase_knowledge(request)

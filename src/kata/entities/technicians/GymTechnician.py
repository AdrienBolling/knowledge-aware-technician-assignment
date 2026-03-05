"""GymTechnician is the file for a Technician for a GymEnvironment with a lot of features."""  # noqa: N999

import math
from typing import Any

import numpy as np
import simpy
from ongoing import KnowledgeGrid

from kata.core.config import TechnicianConfig, get_config
from kata.entities.machines.base import Machine
from kata.entities.requests.RepairRequest import RepairRequest
from kata.entities.technicians.base import Technician

from kata.entities.encoder.base import ENCODER

CONFIG = get_config()


class GymTechnician(Technician):
    """Technician for Gym environments. Fitting for use with the GymTechDispatcher class."""
    _id_counter = 0

    def __init__(
        self,
        tech_conf: TechnicianConfig,
        fatigue_lambda: float = 0.01,
        fatigue_mu: float = 0.05,
    ) -> None:
        _ = fatigue_lambda, fatigue_mu  # kept for API compatibility
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

    def travel_time(self, machine: Machine) -> int:
        """Return the travel time to the given machine. Here, we just return a constant value."""
        _ = machine  # Unused
        return CONFIG.sim.technicians.travel_time

    def stochastic_disruptions_process(
        self,
        env: simpy.Environment,
        tech_resource: simpy.PreemptiveResource,
    ):
        """Stochastically requires the technician ressource and frees it up at the end of the disruption."""
        with tech_resource.request(
            priority=1, preempt=self._interrupt_on_disrupt
        ) as req:
            yield req
            yield env.timeout(self._get_stochastic_disruption_duration())

    def _get_stochastic_disruption_duration(self) -> int:
        """Return the disruption duration sampled from the configured normal distribution according to the disruption type."""
        dis_type = self._get_stochastic_disruption_type()
        mu, sig = (
            CONFIG.sim.disruptions.dis_dict[dis_type]["mu"],
            CONFIG.sim.disruptions.dis_dict[dis_type]["sig"],
        )
        time: float = np.random.normal(mu, sig)
        return int(time if time > 0 else mu)  # Avoid negative times

    def _get_stochastic_disruption_type(self) -> str:
        """Sample from the different types of disruptions according to their configured probabilities."""
        disruptions: dict[str, Any] = CONFIG.sim.disruptions.dis_dict
        types: list[str] = list(disruptions.keys())
        probs: list[float] = [disruptions[t]["prob"] for t in types]
        chosen_type: str = np.random.choice(a=types, p=probs)
        return chosen_type

    def compute_repair_time(self, base_repair_time: int, request: RepairRequest) -> int:
        """Compute the repair time for the given request. Here, we just return the base repair time."""
        base: float = float(base_repair_time)
        if CONFIG.sim.repair.knowledge_enabled:
            base *= self.get_knowledge_multiplier(request)
        if CONFIG.sim.repair.fatigue_enabled:
            base *= self.get_fatigue_multiplier()
        return int(base)

    def increase_knowledge(self, request: RepairRequest) -> None:
        """Increase the knowledge of the technician based on the given request."""
        embedding = ENCODER.encode(request)
        self.knowledge_grid.add_ticket_knowledge(embedding)

    def get_knowledge_multiplier(self, request: RepairRequest) -> float:
        """Return the knowledge factor for the given request."""
        embedding = ENCODER.encode(request)
        return 1 + self.knowledge_grid.get_knowledge(embedding)

    def decay_knowledge(self) -> None:
        """Decay the knowledge of the technician over time."""
        self.knowledge_grid.decay_knowledge()

    def get_fatigue_multiplier(self) -> float:
        """Return the fatigue factor for the technician.

        Maps the fatigue of the technician to a multiplier using the configured fatigue model.
        There are two models available: 'linear' and 'exponential'.

        - linear: multiplier = max(0, 1 - fatigue)
        - exponential: multiplier = exp(-fatigue_alpha * fatigue)
        """
        model = CONFIG.sim.technicians.fatigue_model
        alpha = CONFIG.sim.technicians.fatigue_alpha

        if model == "linear":
            return max(0.0, 1.0 - self.fatigue)
        if model == "exponential":
            return math.exp(-alpha * self.fatigue)
        error_string = f"Unknown fatigue model: {model}"
        raise ValueError(error_string)

    def _increase_fatigue(self, work_time: int) -> None:
        """Fatigue accumulation after a repair."""
        if not (0.0 <= self.fatigue <= 1.0):
            raise ValueError("Fatigue must be between 0 and 1.")
        if work_time < 0:
            raise ValueError("Work time must be non-negative.")
        self.fatigue = self.fatigue + (1.0 - self.fatigue) * (
            1.0 - math.exp(-self.fatigue_lambda * work_time)
        )

    def _recover_fatigue(self, idle_time: int) -> None:
        """Fatigue recovery during idle time."""
        if not (0.0 <= self.fatigue <= 1.0):
            error_string = "Fatigue must be between 0 and 1."
            raise ValueError(error_string)
        if idle_time < 0:
            error_string = "Idle time must be non-negative."
            raise ValueError(error_string)
        self.fatigue = self.fatigue * math.exp(-self.fatigue_mu * idle_time)

    def repair_finished(self, request: RepairRequest, when: float) -> None:
        """Mark the technician as not busy."""
        self.busy = False

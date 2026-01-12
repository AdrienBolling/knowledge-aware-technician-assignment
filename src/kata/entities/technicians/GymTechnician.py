from typing import Any

import numpy as np
import simpy

from kata import get_config
from kata.entities.machines.base import Machine
from kata.entities.requests.RepairRequest import RepairRequest
from kata.entities.technicians.base import Technician

from ongoing import KnowledgeGrid

CONFIG = get_config()


class GymTechnician(Technician):
    """Technician for Gym environments. Fitting for use with the GymTechDispatcher class."""

    def __init__(
        self,
    ) -> None:
        self.busy: bool = False
        self._interupt_on_disrupt: bool = CONFIG.sim.disruptions.interupt_on_disrupt

        # Knowledge

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
            priority=1, preempt=self._interupt_on_disrupt
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
        if CONFIG.sim.repair.knowledge_enabled:
            base_repair_time *= self.get_knowledge_multiplier(request)
        if CONFIG.sim.repair.fatigue_enabled:
            base_repair_time *= self.get_fatigue_multiplier()
        return int(base_repair_time)

    def repair_finished(self, request: RepairRequest, when: int | float) -> None:
        """Mark the technician as not busy."""
        self.busy = False

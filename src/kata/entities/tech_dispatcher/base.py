from abc import ABC, abstractmethod

import simpy

from kata.entities.machines.base import Machine
from kata.entities.requests.RepairRequest import RepairRequest
from kata.entities.technicians.base import Technician


class TechDispatcher(ABC):
    """Base class for technician dispatchers."""

    env: simpy.Environment
    technicians: list[Technician]

    @abstractmethod
    def request_repair(self, machine: Machine):
        """Request a technician to repair the given machine."""
        raise NotImplementedError

    def start_repair(self, tech_id: int, request: RepairRequest):
        """Start the repair process for the given machine by the specified technician.
        Called by a Gym wrapper env after choosing an action.
        """
        raise NotImplementedError

    def wait_until_repaired(self, machine: Machine) -> simpy.Event:
        """Return an event that is triggered when the machine is repaired."""
        raise NotImplementedError

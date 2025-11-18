import simpy
from kata.entities import Technician
from kata.entities import Request
from kata.entities import Machine


class TechDispatcher:
    env: simpy.Environment
    technicians: list[Technician]

    def request_repair(self, machine):
        """Request a technician to repair the given machine."""
        raise NotImplementedError

    def start_repair(self, machine, tech_id: int, request: Request):
        """
        Start the repair process for the given machine by the specified technician.
        Called by a Gym wrapper env after choosing an action.
        """
        raise NotImplementedError

    def wait_until_repaired(self, machine: Machine) -> simpy.Event:
        """Return an event that is triggered when the machine is repaired."""
        raise NotImplementedError

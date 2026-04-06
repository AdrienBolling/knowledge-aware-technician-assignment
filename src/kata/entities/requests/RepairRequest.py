from typing import TYPE_CHECKING

from kata.core.tokenizer import PAD_TOKEN, PRIO_TOKEN
from kata.entities.components import component
from kata.entities.tickets.synthetic_ticket import SyntheticTicket

if TYPE_CHECKING:
    from kata.entities.components.component import MachineComponent
    from kata.entities.machines.machine import Machine

from kata.entities.machines.complex_machine import ComplexMachine
from kata.entities.requests.base import Request


class RepairRequest(Request):
    """Stores information related to the repair needed."""

    chosen_technician_id: int | None = None

    def __init__(self, machine: "Machine", created_at: int) -> None:
        """Initialize the RepairRequest."""
        self.machine = machine
        self.created_at = created_at
        self._failed_component: MachineComponent | None = None

        # Try to get the failed component if machine is a ComplexMachine
        if isinstance(machine, ComplexMachine):
            self._failed_component = machine.get_failed_component()

    def get_repair_time(self) -> float:
        """Get the base repair time for this request.
        
        For ComplexMachine instances, returns component-specific repair time.
        For regular machines, returns default repair time.
        
        Returns:
            Base repair time in time units (can be modified by technician efficiency)

        """
        # If we have a failed component, use its repair time
        if self._failed_component is not None:
            return self._failed_component.get_repair_time()

        # Default base repair time for machines without components
        return 10.0

    def get_failed_component_info(self) -> dict | None:
        """Get information about the failed component.
        
        Returns:
            Dictionary with component ID and type, or None if no component info available

        """
        if self._failed_component is None:
            return None

        return {
            "component_id": self._failed_component.get_id(),
            "component_type": self._failed_component.get_type(),
            "repair_time": self._failed_component.get_repair_time(),
        }

    def create_ticket(self) -> SyntheticTicket:

        machine = self.machine

        # Pick random priority for now (can be improved with a more sophisticated logic)
        p_priority = np.random.choice(["low", "medium", "high"], p=[0.7, 0.2, 0.1])  # 1=High, 2=Medium, 3=Low

        machine_id = machine.machine_id
        machine_type = machine.mtype
        component_type = self._failed_component.get_type() if self._failed_component else PAD_TOKEN
        base_repair_time = self.get_repair_time()
        priority = PRIO_TOKEN[p_priority]
        
        # Create the ticket with all relevant information
        ticket = SyntheticTicket(
            machine=machine_id,
            machine_type=machine_type,
            priority=priority,
            component_type=component_type,
            repair_time_estimate=base_repair_time,
        )

        return ticket
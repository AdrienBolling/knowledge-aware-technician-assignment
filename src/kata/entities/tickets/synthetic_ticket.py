"""
SyntheticTicket implementation for maintenance tickets based on machine failures.
"""
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from kata.entities.machines.machine import Machine

from kata.entities.tickets.base import Ticket


class SyntheticTicket(Ticket):
    """
    A synthetic maintenance ticket generated from machine failure events.
    
    This ticket captures all relevant information about a machine failure including
    the machine details, component failure information, operational context, and
    priority for maintenance scheduling.
    """
    
    def __init__(
        self,
        machine: "Machine",
        machine_type: str,
        failure_type: str,
        priority: int,
        nb_in_buffer: int,
        created_at: int,
        ticket_id: Optional[int] = None,
        component_id: Optional[str] = None,
        component_type: Optional[str] = None,
        repair_time_estimate: Optional[float] = None,
    ) -> None:
        """
        Initialize a SyntheticTicket.
        
        Args:
            machine: Reference to the failed machine
            machine_type: Type/category of the machine
            failure_type: Type of failure (derived from component or general failure)
            priority: Priority level for maintenance (higher = more urgent)
            nb_in_buffer: Number of items in the machine's input buffer
            created_at: Simulation time when the ticket was created
            ticket_id: Optional unique identifier for the ticket
            component_id: ID of the failed component (if applicable)
            component_type: Type of the failed component (if applicable)
            repair_time_estimate: Estimated repair time (if available)
        """
        self.machine = machine
        self.machine_type = machine_type
        self.failure_type = failure_type
        self.priority = priority
        self.nb_in_buffer = nb_in_buffer
        self.created_at = created_at
        self.ticket_id = ticket_id
        self.component_id = component_id
        self.component_type = component_type
        self.repair_time_estimate = repair_time_estimate
    
    def get_machine_id(self) -> int:
        """Get the ID of the failed machine."""
        return self.machine.machine_id
    
    def get_machine_type(self) -> str:
        """Get the type of the failed machine."""
        return self.machine_type
    
    def get_failure_type(self) -> str:
        """Get the type of failure that occurred."""
        return self.failure_type
    
    def get_priority(self) -> int:
        """Get the priority level of this ticket."""
        return self.priority
    
    def get_buffer_level(self) -> int:
        """Get the number of items in the machine's input buffer."""
        return self.nb_in_buffer
    
    def get_component_info(self) -> Optional[dict]:
        """
        Get information about the failed component.
        
        Returns:
            Dictionary with component details, or None if not a component failure
        """
        if self.component_id is None:
            return None
        
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "repair_time_estimate": self.repair_time_estimate,
        }
    
    def __repr__(self) -> str:
        """String representation of the ticket."""
        comp_info = f" ({self.component_type}:{self.component_id})" if self.component_id else ""
        return (
            f"SyntheticTicket(id={self.ticket_id}, machine={self.get_machine_id()}, "
            f"type={self.machine_type}, failure={self.failure_type}{comp_info}, "
            f"priority={self.priority}, buffer={self.nb_in_buffer}, created_at={self.created_at})"
        )

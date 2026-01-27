from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from kata.entities.machines.machine import Machine
    from kata.entities.components.component import MachineComponent

from kata.entities.requests.base import Request


class RepairRequest(Request):
    chosen_technician_id: int | None = None

    def __init__(self, machine: "Machine", created_at: int) -> None:
        """Initialize the RepairRequest."""
        self.machine = machine
        self.created_at = created_at
        self._failed_component: Optional["MachineComponent"] = None
        
        # Try to get the failed component if machine is a ComplexMachine
        if hasattr(machine, 'get_failed_component'):
            self._failed_component = machine.get_failed_component()

    def get_repair_time(self) -> float:
        """
        Get the base repair time for this request.
        
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
    
    def get_failed_component_info(self) -> Optional[dict]:
        """
        Get information about the failed component.
        
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

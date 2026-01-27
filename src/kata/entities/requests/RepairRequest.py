from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kata.entities.machines.machine import Machine

from kata.entities import Request


class RepairRequest(Request):
    chosen_technician_id: int | None = None

    def __init__(self, machine: "Machine", created_at: int) -> None:
        """Initialize the RepairRequest."""
        self.machine = machine
        self.created_at = created_at

    def get_repair_time(self) -> int:
        """
        Get the base repair time for this request.
        
        Returns:
            Base repair time in time units (can be modified by technician efficiency)
        """
        # Get the base repair time from the machine's breakdown process
        # This is a simplified implementation - could be more complex based on failure type
        return 10  # Default base repair time

from kata.entities import Request
from kata.entities.machines.machine import Machine


class RepairRequest(Request):
    chosen_technician_id: int | None = None

    def __init__(self, machine: Machine, created_at: int) -> None:
        """Initialize the RepairRequest."""
        self.machine = machine
        self.created_at = created_at

    def get_repair_time(self) -> int:
        """Get the base repair time for this request."""
        ...

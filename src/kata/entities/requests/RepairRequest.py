from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kata.entities.components.component import MachineComponent
    from kata.entities.machines.machine import Machine

from kata.entities.requests.base import Request


class RepairRequest(Request):
    """Stores information related to the repair needed."""

    chosen_technician_id: int | None = None

    def __init__(self, machine: Machine, created_at: int) -> None:
        """Initialize the RepairRequest."""
        self.machine = machine
        self.created_at = created_at
        self._failed_component: MachineComponent | None = None

        # Try to get the failed component if machine supports components
        getter: Any = getattr(machine, "get_failed_component", None)
        if callable(getter):
            self._failed_component = getter()

    def get_repair_time(self) -> float:
        """Get the base repair time for this request.

        For machines with components, returns component-specific repair time.
        For regular machines, returns default repair time.

        Returns:
            Base repair time in time units (can be modified by technician efficiency)

        """
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

    def get_knowledge_parameters(self) -> tuple[float | None, float | None] | None:
        """Return the failed component's per-failure knowledge overrides.

        Returns ``(min_repair_fraction, knowledge_sensitivity)`` — either
        entry may be ``None`` to fall back to the global value.  Returns
        ``None`` outright when there is no failed component (simple
        machine breakdown) or the component does not expose
        ``get_knowledge_parameters``.
        """
        comp = self._failed_component
        if comp is None or not hasattr(comp, "get_knowledge_parameters"):
            return None
        return comp.get_knowledge_parameters()

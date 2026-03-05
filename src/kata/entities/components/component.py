"""
Machine Component implementation with degradation tracking.
"""
from kata.entities.components.base import Component
from kata.features.breakdown.base import BreakdownProcess


class MachineComponent(Component):
    """
    A machine component with individual degradation process.
    """
    
    def __init__(
        self,
        component_id: str,
        component_type: str,
        breakdown_process: BreakdownProcess,
        base_repair_time: float = 10.0,
        idle_degradation_factor: float = 0.1,
    ):
        """
        Initialize a MachineComponent.
        
        Args:
            component_id: Unique identifier for this component
            component_type: Type/category of the component (e.g., "motor", "bearing", "sensor")
            breakdown_process: The breakdown process governing this component's degradation
            base_repair_time: Base time required to repair this component
            idle_degradation_factor: Factor to reduce degradation when machine is idle (default 0.1)
        """
        self._component_id = component_id
        self._component_type = component_type
        self._breakdown_process = breakdown_process
        self._base_repair_time = base_repair_time
        self._idle_degradation_factor = idle_degradation_factor
    
    def get_id(self) -> str:
        """Get the unique identifier for this component."""
        return self._component_id
    
    def get_type(self) -> str:
        """Get the type of this component."""
        return self._component_type
    
    def step_and_get_failure_prob(self, is_processing: bool) -> float:
        """
        Step the component's degradation and return failure probability.
        
        Args:
            is_processing: Whether the machine is currently processing
            
        Returns:
            Probability of component failure (0.0 to 1.0)
        """
        if is_processing:
            return self._breakdown_process.step_and_get_proba()
        else:
            return self._breakdown_process.step_and_get_idle_proba()
    
    def repair(self) -> None:
        """Reset the component state after repair."""
        self._breakdown_process.repair()
    
    def get_repair_time(self) -> float:
        """Get the base repair time for this component."""
        return self._base_repair_time

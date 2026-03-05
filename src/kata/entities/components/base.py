from abc import ABC, abstractmethod


class Component(ABC):
    """Base class for machine components with individual degradation."""
    
    @abstractmethod
    def get_id(self) -> str:
        """Get the unique identifier for this component."""
        pass
    
    @abstractmethod
    def get_type(self) -> str:
        """Get the type of this component."""
        pass
    
    @abstractmethod
    def step_and_get_failure_prob(self, is_processing: bool) -> float:
        """
        Step the component's degradation and return failure probability.
        
        Args:
            is_processing: Whether the machine is currently processing
            
        Returns:
            Probability of component failure (0.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def repair(self) -> None:
        """Reset the component state after repair."""
        pass
    
    @abstractmethod
    def get_repair_time(self) -> float:
        """Get the base repair time for this component."""
        pass

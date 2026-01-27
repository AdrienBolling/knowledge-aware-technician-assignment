class BreakdownProcess:
    """Base class for breakdown processes that model machine failures."""
    
    def step_and_get_proba(self) -> float:
        """
        Step the breakdown process and get the probability of failure while working.
        
        Returns:
            Probability of breakdown (0.0 to 1.0)
        """
        raise NotImplementedError()

    def step_and_get_idle_proba(self) -> float:
        """
        Step the breakdown process and get the probability of failure while idle.
        
        Returns:
            Probability of breakdown while idle (0.0 to 1.0)
        """
        raise NotImplementedError()
    
    def repair(self) -> None:
        """
        Reset the breakdown process after a repair.
        Should be called when the machine is repaired.
        """
        raise NotImplementedError()

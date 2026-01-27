import random

from kata.features.breakdown.base import BreakdownProcess


class SimpleBreakdownProcess(BreakdownProcess):
    """
    A simple breakdown process with constant failure probabilities.
    """
    
    def __init__(
        self,
        failure_prob_working: float = 0.001,
        failure_prob_idle: float = 0.0001,
    ):
        """
        Initialize a SimpleBreakdownProcess.
        
        Args:
            failure_prob_working: Probability of failure per time step while working
            failure_prob_idle: Probability of failure per time step while idle
        """
        self.failure_prob_working = failure_prob_working
        self.failure_prob_idle = failure_prob_idle
        self.time_since_repair = 0
    
    def step_and_get_proba(self) -> float:
        """Get probability of breakdown while working."""
        self.time_since_repair += 1
        return self.failure_prob_working
    
    def step_and_get_idle_proba(self) -> float:
        """Get probability of breakdown while idle."""
        self.time_since_repair += 1
        return self.failure_prob_idle
    
    def repair(self) -> None:
        """Reset the breakdown process after repair."""
        self.time_since_repair = 0


class WeibullBreakdownProcess(BreakdownProcess):
    """
    A breakdown process based on Weibull distribution for more realistic aging.
    """
    
    def __init__(
        self,
        shape: float = 2.0,
        scale: float = 1000.0,
        dt: int = 1,
    ):
        """
        Initialize a WeibullBreakdownProcess.
        
        Args:
            shape: Weibull shape parameter (k)
            scale: Weibull scale parameter (lambda)
            dt: Time step size
        """
        self.shape = shape
        self.scale = scale
        self.dt = dt
        self.age = 0
    
    def step_and_get_proba(self) -> float:
        """Calculate failure probability based on Weibull hazard function."""
        self.age += self.dt
        # Weibull hazard function: h(t) = (k/lambda) * (t/lambda)^(k-1)
        if self.age <= 0:
            return 0.0
        hazard = (self.shape / self.scale) * ((self.age / self.scale) ** (self.shape - 1))
        # Convert hazard to probability over dt: p = 1 - exp(-h * dt)
        prob = 1.0 - (2.71828 ** (-hazard * self.dt))
        return min(1.0, prob)
    
    def step_and_get_idle_proba(self) -> float:
        """Idle failure probability (much lower)."""
        return self.step_and_get_proba() * 0.1
    
    def repair(self) -> None:
        """Reset age after repair (perfect repair)."""
        self.age = 0

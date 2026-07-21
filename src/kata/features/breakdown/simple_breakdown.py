import math
import random

from kata.features.breakdown.base import BreakdownProcess


class SimpleBreakdownProcess(BreakdownProcess):
    """A simple breakdown process with constant failure probabilities."""

    def __init__(
        self,
        failure_prob_working: float = 0.001,
        failure_prob_idle: float = 0.0001,
        restoration_alpha: float = 0.0,
    ):
        """Initialize a SimpleBreakdownProcess.

        Args:
            failure_prob_working: Probability of failure per time step while working
            failure_prob_idle: Probability of failure per time step while idle
            restoration_alpha: Kijima type-1 restoration factor in [0, 1].
                0 (default) = perfect repair (as-good-as-new, historical
                behaviour); 1 = minimal repair (as-bad-as-old).  Memoryless
                process, so this only affects the diagnostic counter.

        """
        self.failure_prob_working = failure_prob_working
        self.failure_prob_idle = failure_prob_idle
        self.restoration_alpha = float(restoration_alpha)
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
        """Reset the breakdown process after repair (Kijima type-1)."""
        self.time_since_repair = int(self.restoration_alpha * self.time_since_repair)

    # -- event-driven sampling (see Machine._breakdown_driver_event) ----
    supports_event_driven = True

    def _rates(self, poll_dt: float) -> tuple[float, float]:
        """Continuous hazard rates equivalent to the per-poll Bernoullis."""
        cap = 1.0 - 1e-12
        rw = -math.log(1.0 - min(self.failure_prob_working, cap)) / poll_dt
        ri = -math.log(1.0 - min(self.failure_prob_idle, cap)) / poll_dt
        return rw, ri

    def sample_envelope_wait(self, poll_dt: float) -> float | None:
        """Time to the next failure *candidate* at the envelope (max) rate,
        or ``None`` when the process is hazard-free.  Candidates must be
        thinned with :meth:`accept_fraction` at fire time."""
        rw, ri = self._rates(poll_dt)
        env_rate = max(rw, ri)
        if env_rate <= 0.0:
            return None
        return random.expovariate(env_rate)

    def accept_fraction(self, is_processing: bool, poll_dt: float) -> float:
        rw, ri = self._rates(poll_dt)
        env_rate = max(rw, ri)
        if env_rate <= 0.0:
            return 0.0
        return (rw if is_processing else ri) / env_rate

    def advance_age(self, elapsed: float) -> None:
        self.time_since_repair += elapsed


class WeibullBreakdownProcess(BreakdownProcess):
    """A breakdown process based on Weibull distribution for more realistic aging."""

    def __init__(
        self,
        shape: float = 2.0,
        scale: float = 1000.0,
        dt: int = 1,
        restoration_alpha: float = 0.0,
    ):
        """Initialize a WeibullBreakdownProcess.

        Args:
            shape: Weibull shape parameter (k)
            scale: Weibull scale parameter (lambda)
            dt: Time step size
            restoration_alpha: Kijima type-1 restoration factor in [0, 1].
                After a repair the component keeps ``alpha * age`` as
                residual (virtual) age: 0 (default) = perfect repair
                (as-good-as-new, historical behaviour); 1 = minimal repair
                (as-bad-as-old); intermediate values model imperfect
                maintenance, so repeatedly-repaired components fail faster.

        """
        self.shape = shape
        self.scale = scale
        self.dt = dt
        self.restoration_alpha = float(restoration_alpha)
        self.age = 0

    def step_and_get_proba(self) -> float:
        """Calculate failure probability based on Weibull hazard function."""
        self.age += self.dt
        # Weibull hazard function: h(t) = (k/lambda) * (t/lambda)^(k-1)
        if self.age <= 0:
            return 0.0
        hazard = (self.shape / self.scale) * (
            (self.age / self.scale) ** (self.shape - 1)
        )
        # Convert hazard to probability over dt: p = 1 - exp(-h * dt)
        import math

        prob = 1.0 - math.exp(-hazard * self.dt)
        return min(1.0, prob)

    def step_and_get_idle_proba(self) -> float:
        """Idle failure probability (much lower)."""
        return self.step_and_get_proba() * 0.1

    def repair(self) -> None:
        """Kijima type-1 repair: keep ``restoration_alpha * age`` as
        residual virtual age (0 = perfect repair, the historical default)."""
        self.age = self.restoration_alpha * self.age

    # -- event-driven sampling (see Machine._breakdown_driver_event) ----
    supports_event_driven = True
    IDLE_HAZARD_FACTOR = 0.1  # mirrors step_and_get_idle_proba

    def sample_envelope_wait(self, poll_dt: float) -> float | None:
        """Exact inverse-CDF sample of the time to the next failure
        candidate at the working hazard (the envelope), conditioned on the
        current virtual age: with cumulative hazard ``H(t) = (t/scale)^shape``
        and ``E ~ Exp(1)``, the next-failure age solves
        ``H(a + w) = H(a) + E``."""
        _ = poll_dt
        if self.scale <= 0.0:
            return None
        e = random.expovariate(1.0)
        a = max(0.0, float(self.age))
        h_a = (a / self.scale) ** self.shape
        t_next = self.scale * (h_a + e) ** (1.0 / self.shape)
        return max(t_next - a, 1e-9)

    def accept_fraction(self, is_processing: bool, poll_dt: float) -> float:
        _ = poll_dt
        return 1.0 if is_processing else self.IDLE_HAZARD_FACTOR

    def advance_age(self, elapsed: float) -> None:
        self.age += elapsed

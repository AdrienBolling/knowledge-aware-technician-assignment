"""ComplexMachine implementation with component-based degradation system."""

import random

import simpy as sp

from kata.entities.components.component import MachineComponent
from kata.entities.machines.machine import Machine
from kata.entities.tech_dispatcher.GymTechDispatcher import GymTechDispatcher
from kata.features.breakdown.simple_breakdown import SimpleBreakdownProcess


class ComplexMachine(Machine):
    """A machine composed of multiple components, each with individual degradation.

    When any component fails, the entire machine breaks down. The repair process
    is specific to the failed component.
    """

    def __init__(
        self,
        env: sp.Environment,
        machine_id: int,
        mtype: str,
        input_buffer: sp.Store,
        output_buffer: sp.Store,
        tech_dispatcher: GymTechDispatcher,
        components: list[MachineComponent],
        process_time: int,
        dt: int,
        event_driven: bool | None = None,
    ) -> None:
        """Initialize a ComplexMachine with multiple components.

        Args:
            env: SimPy environment
            machine_id: Unique machine identifier
            mtype: Machine type
            input_buffer: Input buffer for products
            output_buffer: Output buffer for products
            tech_dispatcher: Technician dispatcher for repairs
            components: List of MachineComponent instances
            process_time: Time to process a product
            dt: Time step for degradation checking

        """
        self.components = components
        self.failed_component: MachineComponent | None = None

        # Initialize parent with a dummy breakdown process
        # We'll override the breakdown behavior with component-based logic
        dummy_breakdown = SimpleBreakdownProcess(failure_prob_working=0.0)

        self._candidate_component: MachineComponent | None = None

        super().__init__(
            env=env,
            machine_id=machine_id,
            mtype=mtype,
            input_buffer=input_buffer,
            output_buffer=output_buffer,
            tech_dispatcher=tech_dispatcher,
            breakdown_process=dummy_breakdown,
            process_time=process_time,
            dt=dt,
            event_driven=event_driven,
        )

    # -- event-driven hooks: the candidate is the min over components --
    def _next_failure_candidate(self) -> float | None:
        best, best_c = None, None
        for c in self.components:
            bp = c.breakdown_process
            if not getattr(bp, "supports_event_driven", False):
                return None  # mixed setups fall back to polling upstream
            w = bp.sample_envelope_wait(float(self.dt))
            if w is not None and (best is None or w < best):
                best, best_c = w, c
        self._candidate_component = best_c
        return best

    def _advance_failure_clocks(self, elapsed: float) -> None:
        # All component clocks advance together while the machine is up.
        for c in self.components:
            c.breakdown_process.advance_age(elapsed)

    def _accept_candidate(self) -> bool:
        c = self._candidate_component
        if c is None:
            return False
        frac = c.breakdown_process.accept_fraction(
            self.is_processing, float(self.dt)
        )
        if frac > 0.0 and random.uniform(0, 1) < frac:
            self.failed_component = c
            self._log(
                f"Component '{c.get_id()}' ({c.get_type()}) FAILED!"
            )
            return True
        return False

    def _breakdown_driver(self):
        """Override parent's breakdown driver to check each component.

        On each time step, check all components for failure. If any component fails,
        trigger a breakdown and record which component caused the failure.
        """
        if self.event_driven and all(
            getattr(c.breakdown_process, "supports_event_driven", False)
            for c in self.components
        ):
            yield from self._breakdown_driver_event()
            return
        while True:
            yield self.env.timeout(self.dt)
            if self.broken:
                continue

            # Check each component for failure
            for component in self.components:
                p_break = component.step_and_get_failure_prob(self.is_processing)
                if p_break >= random.uniform(0, 1) and p_break > 0:
                    # This component has failed
                    self.failed_component = component
                    self._log(
                        f"Component '{component.get_id()}' ({component.get_type()}) FAILED!"
                    )
                    self._trigger_breakdown()
                    break  # Only one component fails at a time

    def repair(self, request) -> None:
        """Repair the machine by fixing the failed component.

        Args:
            request: The repair request that triggered this repair

        """
        if self.failed_component:
            component_id = self.failed_component.get_id()
            component_type = self.failed_component.get_type()
            self._log(f"Repairing component '{component_id}' ({component_type})")
            self.failed_component.repair()
            self.failed_component = None

        self.broken = False
        self._log(f"Successfully repaired! Total processed: {self.total_processed}")

    def get_failed_component(self) -> MachineComponent | None:
        """Get the component that caused the current breakdown.

        Returns:
            The failed component, or None if no failure

        """
        return self.failed_component

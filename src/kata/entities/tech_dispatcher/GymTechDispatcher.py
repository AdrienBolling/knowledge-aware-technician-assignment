"""Gym-compatible technician dispatcher."""

from __future__ import annotations

from typing import TYPE_CHECKING

import simpy

from kata import get_config
from kata.entities.requests.RepairRequest import RepairRequest
from kata.entities.tech_dispatcher.config import TechDispatcherConfig
from kata.entities.technicians.GymTechnician import GymTechnician as Technician

if TYPE_CHECKING:
    from kata.entities.machines.machine import Machine

CONFIG = get_config()


class GymTechDispatcher:
    """Technician dispatcher for Gym environments."""

    def __init__(
        self,
        env: simpy.Environment,
        technicians: list[Technician],
        config: TechDispatcherConfig | None = None,
    ) -> None:
        """Initialise the dispatcher with technicians and optional config."""
        self.env: simpy.Environment = env
        self.techs: list[Technician] = technicians
        self._tech_by_id: dict[int, Technician] = {t.id: t for t in technicians}

        cfg = config or TechDispatcherConfig()

        # SimPy events to signal when a machine is repaired
        self._repair_events: dict[Machine, simpy.Event] = {}
        # SimPy resource to model technician availability
        self._tech_resource: dict[int, simpy.PreemptiveResource] = {
            t.id: simpy.PreemptiveResource(env, capacity=1) for t in technicians
        }
        # Queue of repair requests for RL agent
        self.repair_queue: simpy.Store = simpy.Store(
            env,
            capacity=cfg.repair_queue_capacity,
        )

        # Start stochastic disruption processes
        for t in technicians:
            _ = env.process(
                generator=t.stochastic_disruptions_process(
                    env,
                    tech_resource=self._tech_resource[t.id],
                )
            )

    # -- External API used by machines ----------------------------------------

    def request_repair(self, machine: Machine) -> None:
        """Create a repair request and add it to the queue."""
        req = RepairRequest(machine=machine, created_at=int(self.env.now))
        _ = self.repair_queue.put(req)

    def start_repair(self, tech_id: int, request: RepairRequest) -> None:
        """Start a repair job. Called by the Gym wrapper after choosing an action."""
        request.chosen_technician_id = tech_id
        tech = self._get_tech(tech_id)
        _ = self.env.process(self._repair_job(tech, request))

    def wait_until_repaired(self, machine: Machine) -> simpy.Event:
        """Return an event that succeeds when *machine* is repaired."""
        if machine not in self._repair_events:
            self._repair_events[machine] = self.env.event()
        return self._repair_events[machine]

    # -- Internal helpers -----------------------------------------------------

    def _get_tech(self, tech_id: int) -> Technician:
        tech = self._tech_by_id.get(tech_id)
        if tech is None:
            msg = f"Technician with id {tech_id} not found"
            raise ValueError(msg)
        return tech

    def _repair_job(self, tech: Technician, request: RepairRequest):  # SimPy generator
        """SimPy process: travel -> repair -> signal completion."""
        machine = request.machine
        tech_res = self._tech_resource[tech.id]
        machine._log(f"Requesting repair by Tech {tech.id}")
        with tech_res.request(priority=0, preempt=False) as req:
            yield req  # Wait for technician to be available
            tech.busy = True

            # Travel time
            t_travel = tech.travel_time(machine)
            machine._log(f"Technician {tech.id} traveling for {t_travel}")
            yield self.env.timeout(t_travel)

            # Repair time (knowledge + fatigue modulated)
            base = request.get_repair_time()
            final_repair_time = tech.compute_repair_time(base, request)
            yield self.env.timeout(final_repair_time)

            # Update machine and technician states
            machine.repair(request)
            if machine in self._repair_events:
                self._repair_events[machine].succeed()
                del self._repair_events[machine]
            machine._log(
                f"Repaired by Tech {tech.id} in {final_repair_time} time units"
            )

            tech.repair_finished(request, self.env.now)

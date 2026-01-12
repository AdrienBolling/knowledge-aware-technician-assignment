import simpy

from kata.entities.machines.base import Machine
from kata.entities.tech_dispatcher.base import TechDispatcher
from kata.entities.requests.RepairRequest import RepairRequest
from kata.entities.requests.base import Request
from kata.entities.technicians.GymTechnician import GymTechnician as Technician

from kata import get_config

from typing import override

CONFIG = get_config()


class GymTechDispatcher(TechDispatcher):
    """Technician dispatcher for Gym environments."""

    def __init__(self, env: simpy.Environment, technicians: list[Technician]) -> None:
        """Initialize the GymTechDispatcher."""
        self.env: simpy.Environment = env
        self.techs: list[Technician] = technicians

        # Simpy events to signal when a machine is repaired
        self._repair_events: dict[Machine, simpy.Event] = {}
        # Simpy ressource to model technician availability
        self._tech_resource: dict[int, simpy.PreemptiveResource] = {
            t.id: simpy.PreemptiveResource(env, capacity=1) for t in technicians
        }
        # Queue of tickets for RL agent (Gym wrapper will query this to notice when a breakdown occurs)
        self.repair_queue: simpy.Store = simpy.Store(env, capacity=9999)

        # Start all stochastic disruptions process from the technicians
        for t in technicians:
            _ = env.process(
                generator=t.stochastic_disruptions_process(
                    env,
                    tech_resource=self._tech_resource[t.id],
                )
            )

    # External API used by the machines
    @override
    def request_repair(self, machine: Machine) -> None:
        req = RepairRequest(machine=machine, created_at=int(self.env.now))
        _ = self.repair_queue.put(req)

    @override
    def start_repair(self, tech_id: int, request: RepairRequest) -> None:
        """Must be called by the Gym wrapper after choosing an action."""
        request.chosen_technician_id = tech_id
        tech = self._get_tech(tech_id)
        self.env.process(self._repair_job(tech, request))

    @override
    def wait_until_repaired(self, machine: Machine) -> simpy.Event:
        if machine not in self._repair_events:
            self._repair_events[machine] = self.env.event()
        return self._repair_events[machine]

    def _get_tech(self, tech_id: int) -> Technician:
        for t in self.techs:
            if t.id == tech_id:
                return t
        message = f"Technician with id {tech_id} not found"
        raise ValueError(message)

    def _repair_job(self, tech: Technician, request: RepairRequest):
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

            # Basic repair time
            base = request.get_repair_time()

            # Final computation of repair time through the technician (knowledge and fatigue models)
            final_repair_time = tech.compute_repair_time(base, request)
            yield self.env.timeout(final_repair_time)

            # Update machine and technician states
            machine.repair(request)
            if machine in self._repair_events:
                _ = self._repair_events[machine].succeed()
                del self._repair_events[machine]
            machine._log(
                f"Repaired by Tech {tech.id} in {final_repair_time} time units"
            )

            tech.repair_finished(request, self.env.now)

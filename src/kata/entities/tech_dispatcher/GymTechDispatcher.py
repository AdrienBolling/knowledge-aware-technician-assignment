from kata.entities import TechDispatcher, Technician, Machine
import simpy


class GymTechDispatcher(TechDispatcher):
    """
    Technician dispatcher for Gym environments.
    """

    def __init__(self, env: simpy.Environment, technicians: list[Technician]) -> None:
        self.env = env
        self.techs = technicians

        # Simpy events to signal when a machine is repaired
        self._repair_events: dict[Machine, simpy.Event] = {}
        # Simpy ressource to model technician availability
        self._tech_resource = {
            t.id: simpy.Resource(env, capacity=1) for t in technicians
        }
        # Queue of tickets for RL agent (Gym wrapper will query this to notice when a breakdown occurs)
        self.repair_queue = simpy.Store(env, capacity=9999)

    # External API used by the machines
    def request_repair(self, machine: Machine):
        req = RepairRequest(machine=machine, created_at=self.env.now)
        self.repair_queue.put(req)

    def start_repair(self, tech_id: int, request: RepairRequest):
        """Called by the Gym wrapper after choosing an action"""
        request.chosen_technician_id = tech_id
        tech = self._get_tech(tech_id)
        self.env.process(self._repair_job(tech, request))

    def wait_until_repaired(self, machine: Machine) -> simpy.Event:
        if machine not in self._repair_events:
            self._repair_events[machine] = self.env.event()
        return self._repair_events[machine]

    def _get_tech(self, tech_id: int) -> Technician:
        for t in self.techs:
            if t.id == tech_id:
                return t
        raise KeyError(f"Technician with id {tech_id} not found")

    def _repair_job(self, tech: Technician, request: RepairRequest):
        machine = request.machine
        res = self._tech_resource[tech.id]
        machine._log(f"Requesting repair by Tech {tech.id}")
        with res.request() as req:
            yield req  # Wait for technician to be available
            tech.busy = True
            t_travel = tech.travel_time() if hasattr(tech, "travel_time") else 0
            machine._log(
                f"Technician {tech.id} traveling to machine (time: {t_travel})"
            )
            yield self.env.timeout(t_travel)
            # Repair time
            final_repair_time = tech.compute_repair_time(request)
            machine._log(
                f"Technician {tech.id} repairing machine (time: {final_repair_time})"
            )
            yield self.env.timeout(final_repair_time)
            # Flag machine as repaired
            machine.repair(request)
            if machine in self._repair_events:
                self._repair_events[machine].succeed()
                del self._repair_events[machine]
            machine._log(f"Technician {tech.id} completed repair")
            tech.busy = False

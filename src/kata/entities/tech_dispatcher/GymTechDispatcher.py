"""Gym-compatible technician dispatcher."""

from __future__ import annotations

from typing import TYPE_CHECKING

import simpy

from kata.core.config import SimEnvConfig
from kata.entities.requests.RepairRequest import RepairRequest
from kata.entities.tech_dispatcher.config import TechDispatcherConfig
from kata.entities.technicians.GymTechnician import GymTechnician as Technician

if TYPE_CHECKING:
    from kata.entities.machines.machine import Machine


class GymTechDispatcher:
    """Technician dispatcher for Gym environments."""

    def __init__(
        self,
        env: simpy.Environment,
        technicians: list[Technician],
        sim_cfg: SimEnvConfig,
        config: TechDispatcherConfig | None = None,
    ) -> None:
        """Initialise the dispatcher with technicians and the run's ``sim`` config.

        ``sim_cfg`` is **required** and kept public: the env's
        lifecycle-hire path reads it back off the dispatcher when it
        builds a mid-episode technician.  It replaces a module-level
        ``CONFIG = get_config()`` snapshot that was captured at import
        time and therefore never carried the per-run JSON's ``sim.*``
        values (see :class:`~kata.entities.technicians.GymTechnician`).
        """
        self.env: simpy.Environment = env
        self.techs: list[Technician] = technicians
        self.sim_cfg: SimEnvConfig = sim_cfg
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

        # Optional callback invoked when a repair finishes — used by the
        # Gym wrapper to track completed repairs and actual repair
        # durations.  Signature: ``(request, repair_duration)``.
        self.on_repair_completed: callable | None = None

        # Inject the env reference so each technician's time-aware
        # ``fatigue`` property can resolve the current simulation clock
        # without an extra plumbing layer.  Without this the property
        # falls back to its raw event-driven base value, which is the
        # right behaviour for unit-tested technicians but wrong inside
        # a live simulation.
        for t in technicians:
            t.env = env

        # Start one long-running disruption process per (technician,
        # disruption type) pair.  Each process runs for the entire
        # episode and may fire its disruption many times, with
        # inter-arrival behaviour governed by the type's trigger
        # (random Poisson, fatigue-driven Bernoulli, or strict periodic).
        for t in technicians:
            for dis_name, dis_cfg in self.sim_cfg.disruptions.dis_dict.items():
                _ = env.process(
                    generator=t.run_disruption_process(
                        env=env,
                        tech_resource=self._tech_resource[t.id],
                        dis_name=dis_name,
                        dis_cfg=dis_cfg,
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

    def seed_disruptions(self, root_seed: int) -> None:
        """Seed each technician's disruption RNG from a single root seed.

        Spawns one independent ``np.random.Generator`` per technician via
        ``np.random.SeedSequence`` so that calling
        ``env.reset(seed=seed)`` reproduces the same disruption timing
        sequence across runs.  Without this, the disruption loops fall
        back to a fresh non-seeded generator at technician
        construction.
        """
        import numpy as _np

        seed_seq = _np.random.SeedSequence(int(root_seed))
        for tech, child in zip(self.techs, seed_seq.spawn(len(self.techs))):
            tech._rng = _np.random.default_rng(child)

    def add_technician(
        self, tech: Technician, *, rng_seed: int | None = None
    ) -> Technician:
        """Register a technician mid-episode (lifecycle 'hire').

        Performs every registration step the constructor does for the
        initial fleet: env injection, idle-anchor stamping, RNG
        seeding, id/resource maps, and one disruption process per
        configured disruption type.
        """
        import numpy as _np

        tech.env = self.env
        # Fresh hire: anchor fatigue-recovery bookkeeping at 'now', not
        # at t=0 (otherwise the first repair sees a huge bogus idle
        # interval).
        tech._last_idle_since = float(self.env.now)
        if rng_seed is not None:
            tech._rng = _np.random.default_rng(int(rng_seed))
        self.techs.append(tech)
        self._tech_by_id[tech.id] = tech
        self._tech_resource[tech.id] = simpy.PreemptiveResource(
            self.env, capacity=1
        )
        for dis_name, dis_cfg in self.sim_cfg.disruptions.dis_dict.items():
            _ = self.env.process(
                generator=tech.run_disruption_process(
                    env=self.env,
                    tech_resource=self._tech_resource[tech.id],
                    dis_name=dis_name,
                    dis_cfg=dis_cfg,
                )
            )
        return tech

    # -- Internal helpers -----------------------------------------------------

    def _get_tech(self, tech_id: int) -> Technician:
        tech = self._tech_by_id.get(tech_id)
        if tech is None:
            msg = f"Technician with id {tech_id} not found"
            raise ValueError(msg)
        return tech

    def _repair_job(self, tech: Technician, request: RepairRequest):  # SimPy generator
        """SimPy process: travel -> repair -> signal completion.

        Holds the technician's resource at priority 1 (lower precedence
        than disruptions at priority 0) with ``preempt=False`` so a
        preempting disruption (one with ``preemptive=True``) can
        interrupt this process mid-repair.  When that happens the
        partially-completed ticket is restored to the dispatcher's
        repair queue so the agent can re-assign it.
        """
        machine = request.machine
        tech_res = self._tech_resource[tech.id]
        machine._log(f"Requesting repair by Tech {tech.id}")
        with tech_res.request(priority=1, preempt=False) as req:
            try:
                yield req  # Wait for technician to be available
                # Hand off to the technician so it can recover fatigue
                # based on the time elapsed since its previous
                # ``repair_finished`` before flipping ``busy = True``.
                if hasattr(tech, "start_repair"):
                    tech.start_repair(self.env.now)
                else:
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

                # Notify any listener (e.g. KataEnv) that a repair just
                # completed.  ``final_repair_time`` excludes travel and
                # queue waiting so it is exactly the repair duration; the
                # technician identity feeds the env's public repair log.
                if self.on_repair_completed is not None:
                    self.on_repair_completed(
                        request, float(final_repair_time), tech
                    )
            except simpy.Interrupt:
                # A higher-priority disruption preempted this repair.
                # Restore technician state and re-queue the ticket; the
                # agent will see it again on the next decision step.
                machine._log(
                    f"Repair by Tech {tech.id} preempted; re-queueing ticket"
                )
                tech.busy = False
                # Fatigue is updated only at ``repair_finished``, so the
                # partial work done here intentionally does not count
                # against the technician.
                request.chosen_technician_id = None
                _ = self.repair_queue.put(request)

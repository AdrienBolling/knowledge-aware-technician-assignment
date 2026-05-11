"""Config-driven scenario builder.

``ScenarioBuilder`` constructs a complete SimPy factory simulation from a
``KATAConfig``, wiring up machines, technicians, buffers, routers, feeders,
sources, sinks, and the tech dispatcher.  This is the recommended way to
create reproducible, configurable simulation scenarios.

Usage
-----
>>> from kata.scenario import ScenarioBuilder
>>> builder = ScenarioBuilder(config)
>>> env, dispatcher = builder.build()
>>> # Pass to KataEnv as scenario_factory or directly
"""

from __future__ import annotations

from typing import Any

import simpy

from kata.core.config import KATAConfig, get_config
from kata.entities.buffers.buffer import Buffer
from kata.entities.components.component import MachineComponent
from kata.entities.machine_feeder.machine_feeder import MachineFeeder
from kata.entities.machines.complex_machine import ComplexMachine
from kata.entities.machines.machine import Machine
from kata.entities.routers.router import Router
from kata.entities.sinks.sink import Sink
from kata.entities.sources.source import Source
from kata.entities.tech_dispatcher.GymTechDispatcher import GymTechDispatcher
from kata.entities.technicians.GymTechnician import GymTechnician
from kata.features.breakdown.simple_breakdown import (
    SimpleBreakdownProcess,
    WeibullBreakdownProcess,
)


class ScenarioBuilder:
    """Build a complete factory simulation from a KATAConfig.

    The builder creates all entities, wires them together, and returns the
    SimPy environment + dispatcher pair that KataEnv expects.
    """

    def __init__(self, config: KATAConfig | None = None) -> None:
        """Initialise with a configuration (defaults to ``get_config()``)."""
        self.config = config or get_config()

    def build(self) -> tuple[simpy.Environment, GymTechDispatcher]:
        """Construct and return ``(simpy.Environment, GymTechDispatcher)``."""
        env = simpy.Environment()

        # -- Technicians ------------------------------------------------------
        technicians = self._build_technicians()

        # -- Dispatcher -------------------------------------------------------
        dispatcher = GymTechDispatcher(env, technicians)

        # -- Buffers & Machines -----------------------------------------------
        machines_by_type: dict[str, list[Machine]] = {}
        machine_input_buffers: dict[str, list[Buffer]] = {}
        all_output_buffers: list[tuple[str, Buffer]] = []

        for name, mcfg in self.config.machines.items():
            mtype = mcfg.machine_type
            in_buf = Buffer(env, f"BUF_{name}_IN", capacity=50)
            out_buf = Buffer(env, f"BUF_{name}_OUT", capacity=50)

            if mcfg.components:
                machine = self._build_complex_machine(
                    env,
                    name,
                    mcfg,
                    in_buf,
                    out_buf,
                    dispatcher,
                )
            else:
                machine = self._build_simple_machine(
                    env,
                    name,
                    mcfg,
                    in_buf,
                    out_buf,
                    dispatcher,
                )

            # Tag the simulator-side machine with its config-side name
            # so observability tooling can label per-machine plots with
            # human-readable identifiers ("cnc_1") instead of the
            # hashed numeric machine_id.
            machine.name = name  # type: ignore[attr-defined]

            machines_by_type.setdefault(mtype, []).append(machine)
            machine_input_buffers.setdefault(mtype, []).append(in_buf)
            all_output_buffers.append((mtype, out_buf))

        # -- Routing infrastructure -------------------------------------------
        route_buffer = Buffer(env, "BUF_ROUTE", capacity=200)
        sink_buffer = Buffer(env, "BUF_SINK", capacity=200)

        # Type-specific queues for feeders
        type_queues: dict[str, Buffer] = {}
        for mtype in machines_by_type:
            type_queues[mtype] = Buffer(env, f"BUF_{mtype}_Q", capacity=100)

        # Router: route_buffer -> type queues (and sink)
        type_to_buffer: dict[str, Buffer] = {**type_queues, "__SINK__": sink_buffer}
        Router(env, "MainRouter", route_buffer, type_to_buffer)

        # Feeders: type queue -> machine input buffers
        for mtype, bufs in machine_input_buffers.items():
            MachineFeeder(env, f"{mtype}Feeder", mtype, type_queues[mtype], bufs)

        # Conveyors: machine output -> route buffer (for multi-step routes)
        for mtype, out_buf in all_output_buffers:
            self._create_conveyor(env, out_buf, route_buffer)

        # -- Source -----------------------------------------------------------
        routes = [pcfg.route for pcfg in self.config.products.values() if pcfg.route]
        default_route = routes[0] if routes else list(machines_by_type.keys())

        for pname, pcfg in self.config.products.items():
            Source(
                env,
                name=f"Source_{pname}",
                out_buffer=route_buffer,
                interarrival_time=10.0,
                route=pcfg.route or default_route,
            )

        # -- Sink -------------------------------------------------------------
        main_sink = Sink(env, "MainSink", sink_buffer)

        # Expose machines and sinks on dispatcher for observations / metrics
        dispatcher.machines = {  # type: ignore[attr-defined]
            m.machine_id: m for machines in machines_by_type.values() for m in machines
        }
        dispatcher.sinks = [main_sink]  # type: ignore[attr-defined]

        return env, dispatcher

    # -- Private helpers ------------------------------------------------------

    def _build_technicians(self) -> list[GymTechnician]:
        """Create GymTechnician instances from config."""
        techs: list[GymTechnician] = []
        for _name, tcfg in self.config.technicians.items():
            techs.append(GymTechnician(tech_conf=tcfg))
        return techs

    def _build_simple_machine(
        self,
        env: simpy.Environment,
        name: str,
        mcfg: Any,
        in_buf: Buffer,
        out_buf: Buffer,
        dispatcher: GymTechDispatcher,
    ) -> Machine:
        """Create a simple Machine with a default breakdown process."""
        return Machine(
            env=env,
            machine_id=hash(name) % 10000,
            mtype=mcfg.machine_type,
            input_buffer=in_buf,
            output_buffer=out_buf,
            tech_dispatcher=dispatcher,
            breakdown_process=SimpleBreakdownProcess(
                failure_prob_working=0.005,
                failure_prob_idle=0.0005,
            ),
            process_time=mcfg.process_time,
            dt=mcfg.dt,
        )

    def _build_complex_machine(
        self,
        env: simpy.Environment,
        name: str,
        mcfg: Any,
        in_buf: Buffer,
        out_buf: Buffer,
        dispatcher: GymTechDispatcher,
    ) -> ComplexMachine:
        """Create a ComplexMachine with components from config."""
        components: list[MachineComponent] = []
        for _cname, ccfg in mcfg.components.items():
            if ccfg.breakdown_model == "weibull":
                bp = WeibullBreakdownProcess(
                    shape=ccfg.weibull_breakdown.shape,
                    scale=ccfg.weibull_breakdown.scale,
                    dt=mcfg.dt,
                )
            else:
                bp = SimpleBreakdownProcess(
                    failure_prob_working=ccfg.simple_breakdown.failure_prob_working,
                    failure_prob_idle=ccfg.simple_breakdown.failure_prob_idle,
                )
            components.append(
                MachineComponent(
                    component_id=ccfg.component_id,
                    component_type=ccfg.component_type,
                    breakdown_process=bp,
                    base_repair_time=ccfg.base_repair_time,
                    idle_degradation_factor=ccfg.idle_degradation_factor,
                )
            )

        return ComplexMachine(
            env=env,
            machine_id=hash(name) % 10000,
            mtype=mcfg.machine_type,
            input_buffer=in_buf,
            output_buffer=out_buf,
            tech_dispatcher=dispatcher,
            components=components,
            process_time=mcfg.process_time,
            dt=mcfg.dt,
        )

    @staticmethod
    def _create_conveyor(
        env: simpy.Environment,
        src: Buffer,
        dst: Buffer,
    ) -> simpy.Process:
        """Create a simple pass-through conveyor process."""

        def _process():
            while True:
                item = yield src.get()
                yield dst.put(item)

        return env.process(_process())

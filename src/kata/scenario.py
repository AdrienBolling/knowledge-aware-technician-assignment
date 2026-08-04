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

from dataclasses import dataclass, field
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


@dataclass
class FactoryHandles:
    """Live references to the routing infrastructure of a built factory.

    Historically every intermediate (router, feeders, route buffer, …)
    was a discarded local of ``ScenarioBuilder.build`` — mid-episode
    lifecycle events (add/retire machines) need these handles, so the
    builder now attaches them to the dispatcher as
    ``dispatcher.factory_handles``.
    """

    builder: "ScenarioBuilder"
    route_buffer: Buffer
    sink_buffer: Buffer
    type_queues: dict[str, Buffer]
    router: Router
    feeders: dict[str, MachineFeeder]
    machines_by_type: dict[str, list[Machine]] = field(default_factory=dict)


def next_free_machine_id(dispatcher: GymTechDispatcher) -> int:
    """Smallest id strictly above every registered machine id.

    The builder derives ids as ``hash(name) % 10000``; lifecycle adds
    must not collide with those (or each other), so they allocate above
    the current maximum.
    """
    machines = getattr(dispatcher, "machines", {}) or {}
    return (max(machines.keys()) + 1) if machines else 10_000


def add_machine_to_factory(
    env: simpy.Environment,
    dispatcher: GymTechDispatcher,
    template: str,
    label: str,
) -> Machine:
    """Build a machine from *template* and wire it into the live factory.

    Restricted to machine types already present in the park: the
    router/feeder chain for a brand-new type does not exist and no
    product route would visit it.
    """
    handles: FactoryHandles | None = getattr(
        dispatcher, "factory_handles", None
    )
    if handles is None:
        msg = (
            "dispatcher has no factory_handles — the scenario predates "
            "lifecycle support (rebuild via ScenarioBuilder.build)"
        )
        raise RuntimeError(msg)
    from kata.EntityFactories.machine_factory import (
        create_config_from_template,
    )

    mcfg = create_config_from_template(template)
    mtype = mcfg.machine_type
    feeder = handles.feeders.get(mtype)
    if feeder is None:
        msg = (
            f"lifecycle add_machine: type {mtype!r} (template "
            f"{template!r}) is not present in the park — adds are "
            "restricted to existing machine types"
        )
        raise ValueError(msg)

    in_buf = Buffer(env, f"BUF_{label}_IN", capacity=50)
    out_buf = Buffer(env, f"BUF_{label}_OUT", capacity=50)
    b = handles.builder
    if mcfg.components:
        machine = b._build_complex_machine(
            env, label, mcfg, in_buf, out_buf, dispatcher
        )
    else:
        machine = b._build_simple_machine(
            env, label, mcfg, in_buf, out_buf, dispatcher
        )
    # Replace the hash-derived id with a guaranteed-unique one before
    # any registry sees it.
    machine.machine_id = next_free_machine_id(dispatcher)
    machine.name = label  # type: ignore[attr-defined]

    feeder.machines.append(machine)
    feeder.machine_input_buffers.append(in_buf)
    ScenarioBuilder._create_conveyor(env, out_buf, handles.route_buffer)
    dispatcher.machines[machine.machine_id] = machine  # type: ignore[attr-defined]
    # ``feeder.machines`` IS ``machines_by_type[mtype]`` (the builder
    # hands the list by reference) — only append separately if a copy
    # was made somewhere upstream.
    by_type = handles.machines_by_type.setdefault(mtype, [])
    if by_type is not feeder.machines:
        by_type.append(machine)
    return machine


def retire_machine_from_factory(
    dispatcher: GymTechDispatcher, machine: Machine
) -> None:
    """Permanently stop *machine* and unwire it from the live factory.

    The caller must ensure the machine is not broken / mid-repair (the
    env's lifecycle scheduler defers retirement until repaired).
    Products already in its input buffer are stranded — deliberate:
    scrapping WIP is part of a machine swap.
    """
    handles: FactoryHandles | None = getattr(
        dispatcher, "factory_handles", None
    )
    machine.retired = True
    in_buf = None
    if handles is not None:
        feeder = handles.feeders.get(machine.mtype)
        if feeder is not None and machine in feeder.machines:
            i = feeder.machines.index(machine)
            feeder.machines.pop(i)
            in_buf = feeder.machine_input_buffers.pop(i)
        by_type = handles.machines_by_type.get(machine.mtype)
        if by_type and machine in by_type:
            by_type.remove(machine)
    machines = getattr(dispatcher, "machines", None)
    if machines is not None:
        machines.pop(machine.machine_id, None)
    if in_buf is not None:
        # Drain the retired machine's input buffer: a feeder blocked on
        # put() into this (full) buffer would otherwise starve its whole
        # type forever.  Consuming items completes any pending put and
        # scraps the WIP (deliberate — part of a machine swap).
        def _drain(buf=in_buf):
            store = getattr(buf, "store", buf)
            while store.items or store.put_queue:
                yield store.get()

        dispatcher.env.process(_drain())


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
        # Route / type / sink buffers are intentionally unbounded.  The
        # router and the per-type feeders are single-process serial pipes,
        # so any bounded capacity on these buffers creates a hard deadlock
        # the moment one stage runs slower than upstream: the router blocks
        # on a full type queue, back-pressure cascades to the conveyors,
        # and the whole factory grinds to a halt with zero products
        # finishing.  Per-machine input / output buffers stay bounded
        # because they model real WIP capacity and machines genuinely
        # idle when their output is full.
        route_buffer = Buffer(env, "BUF_ROUTE")
        sink_buffer = Buffer(env, "BUF_SINK")

        # Type-specific queues for feeders
        type_queues: dict[str, Buffer] = {}
        for mtype in machines_by_type:
            type_queues[mtype] = Buffer(env, f"BUF_{mtype}_Q")

        # Router: route_buffer -> type queues (and sink)
        type_to_buffer: dict[str, Buffer] = {**type_queues, "__SINK__": sink_buffer}
        router = Router(env, "MainRouter", route_buffer, type_to_buffer)

        # Feeders: type queue -> machine input buffers
        feeders: dict[str, MachineFeeder] = {}
        for mtype, bufs in machine_input_buffers.items():
            feeders[mtype] = MachineFeeder(
                env,
                f"{mtype}Feeder",
                mtype,
                type_queues[mtype],
                bufs,
                machines=machines_by_type[mtype],
            )

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
        # Live routing handles for mid-episode lifecycle events.
        dispatcher.factory_handles = FactoryHandles(  # type: ignore[attr-defined]
            builder=self,
            route_buffer=route_buffer,
            sink_buffer=sink_buffer,
            type_queues=type_queues,
            router=router,
            feeders=feeders,
            machines_by_type=machines_by_type,
        )

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
                restoration_alpha=float(
                    getattr(
                        self.config.sim.repair,
                        "default_restoration_alpha",
                        0.0,
                    )
                ),
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
        # Per-component restoration_alpha wins; components that leave it
        # unset (0.0) inherit the global repair-physics default.
        global_alpha = float(
            getattr(self.config.sim.repair, "default_restoration_alpha", 0.0)
        )
        components: list[MachineComponent] = []
        for _cname, ccfg in mcfg.components.items():
            alpha = getattr(ccfg, "restoration_alpha", 0.0) or global_alpha
            if ccfg.breakdown_model == "weibull":
                bp = WeibullBreakdownProcess(
                    shape=ccfg.weibull_breakdown.shape,
                    scale=ccfg.weibull_breakdown.scale,
                    dt=mcfg.dt,
                    restoration_alpha=alpha,
                )
            else:
                bp = SimpleBreakdownProcess(
                    failure_prob_working=ccfg.simple_breakdown.failure_prob_working,
                    failure_prob_idle=ccfg.simple_breakdown.failure_prob_idle,
                    restoration_alpha=alpha,
                )
            components.append(
                MachineComponent(
                    component_id=ccfg.component_id,
                    component_type=ccfg.component_type,
                    breakdown_process=bp,
                    base_repair_time=ccfg.base_repair_time,
                    idle_degradation_factor=ccfg.idle_degradation_factor,
                    min_repair_fraction=ccfg.min_repair_fraction,
                    knowledge_sensitivity=ccfg.knowledge_sensitivity,
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

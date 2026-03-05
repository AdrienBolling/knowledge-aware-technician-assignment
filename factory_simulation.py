"""Complete SimPy Factory Simulation Example

This demonstrates the complete factory simulation using all the SimPy entities:
- Products flowing through production lines
- Machines with breakdown processes
- Buffers managing product flow
- Technicians responding to breakdowns
- Routers directing products to appropriate machines
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import simpy

from kata.core.config import TechnicianConfig
from kata.entities.buffers.buffer import Buffer
from kata.entities.machine_feeder.machine_feeder import MachineFeeder
from kata.entities.machines.machine import Machine
from kata.entities.routers.router import Router
from kata.entities.sinks.sink import Sink
from kata.entities.sources.source import Source
from kata.entities.tech_dispatcher.GymTechDispatcher import GymTechDispatcher
from kata.entities.technicians.GymTechnician import GymTechnician
from kata.features.breakdown.simple_breakdown import SimpleBreakdownProcess


class SimpleAgentPolicy:
    """Simple policy that assigns repairs to random available technicians."""

    def __init__(self, dispatcher: GymTechDispatcher):
        self.dispatcher = dispatcher

    def run(self):
        """Process repair requests from the queue."""
        while True:
            # Wait for a repair request
            request = yield self.dispatcher.repair_queue.get()

            # Find available technician (simplified: just pick first available)
            available_techs = [t for t in self.dispatcher.techs if not t.busy]

            if available_techs:
                chosen = available_techs[0]
            else:
                # If all busy, just pick the first one
                chosen = self.dispatcher.techs[0]

            # Assign repair
            self.dispatcher.start_repair(chosen.id, request)


def create_conveyor(env: simpy.Environment, src: Buffer, dst: Buffer, name: str = "conv", delay: float = 0.0):
    """Create a conveyor process that moves products between buffers."""
    def conveyor_process():
        while True:
            item = yield src.get()
            if delay > 0:
                yield env.timeout(delay)
            print(f"[{env.now:8.1f}] [CNV:{name}] Moving product {item.product_id} -> {dst.name}")
            yield dst.put(item)

    return env.process(conveyor_process())


def build_factory_simulation(seed: int = 42) -> simpy.Environment:
    """Build a complete factory simulation with:
    - 1 Source generating products
    - 1 Router distributing products by type
    - 2 Drill machines and 1 Paint machine
    - Feeders for load balancing
    - 3 Technicians for repairs
    - 1 Sink collecting finished products
    """
    # Initialize environment
    env = simpy.Environment()
    np.random.seed(seed)

    # Create technicians
    techs = [
        GymTechnician(
            tech_conf=TechnicianConfig(),
        )
        for i in range(3)
    ]
    for i, t in enumerate(techs):
        t.id = i

    # Create tech dispatcher
    dispatcher = GymTechDispatcher(env, techs)

    # Start simple agent policy
    agent_policy = SimpleAgentPolicy(dispatcher)
    env.process(agent_policy.run())

    # Create buffers
    buffers = {}
    buffers["src_out"] = Buffer(env, 0, "BUF_SRC", capacity=50)
    buffers["route_out"] = Buffer(env, 1, "BUF_ROUTE", capacity=100)
    buffers["drill_q"] = Buffer(env, 2, "BUF_DRILL_IN", capacity=30)
    buffers["paint_q"] = Buffer(env, 3, "BUF_PAINT_IN", capacity=30)
    buffers["m1_in"] = Buffer(env, 4, "BUF_M1_IN", capacity=5)
    buffers["m1_out"] = Buffer(env, 5, "BUF_M1_OUT", capacity=5)
    buffers["m2_in"] = Buffer(env, 6, "BUF_M2_IN", capacity=5)
    buffers["m2_out"] = Buffer(env, 7, "BUF_M2_OUT", capacity=5)
    buffers["p1_in"] = Buffer(env, 8, "BUF_P1_IN", capacity=5)
    buffers["p1_out"] = Buffer(env, 9, "BUF_P1_OUT", capacity=5)
    buffers["sink_in"] = Buffer(env, 10, "BUF_SINK", capacity=100)

    # Create machines with breakdown processes
    m1 = Machine(
        env=env,
        machine_id=1,
        mtype="Drill",
        input_buffer=buffers["m1_in"],
        output_buffer=buffers["m1_out"],
        tech_dispatcher=dispatcher,
        breakdown_process=SimpleBreakdownProcess(failure_prob_working=0.01),
        process_time=15,
        dt=1,
    )

    m2 = Machine(
        env=env,
        machine_id=2,
        mtype="Drill",
        input_buffer=buffers["m2_in"],
        output_buffer=buffers["m2_out"],
        tech_dispatcher=dispatcher,
        breakdown_process=SimpleBreakdownProcess(failure_prob_working=0.01),
        process_time=15,
        dt=1,
    )

    p1 = Machine(
        env=env,
        machine_id=3,
        mtype="Paint",
        input_buffer=buffers["p1_in"],
        output_buffer=buffers["p1_out"],
        tech_dispatcher=dispatcher,
        breakdown_process=SimpleBreakdownProcess(failure_prob_working=0.008),
        process_time=20,
        dt=1,
    )

    # Create source
    source = Source(
        env=env,
        source_id=0,
        name="MainSource",
        out_buffer=buffers["route_out"],
        interarrival_time=7.0,
        route=["Drill", "Paint"],
        max_products=20,
    )

    # Create router
    router = Router(
        env=env,
        router_id=0,
        name="MainRouter",
        in_buffer=buffers["route_out"],
        type_to_buffer={
            "Drill": buffers["drill_q"],
            "Paint": buffers["paint_q"],
            "__SINK__": buffers["sink_in"],
        },
    )

    # Create feeders
    drill_feeder = MachineFeeder(
        env=env,
        feeder_id=0,
        name="DrillFeeder",
        machine_type="Drill",
        in_buffer=buffers["drill_q"],
        machine_input_buffers=[buffers["m1_in"], buffers["m2_in"]],
    )

    paint_feeder = MachineFeeder(
        env=env,
        feeder_id=1,
        name="PaintFeeder",
        machine_type="Paint",
        in_buffer=buffers["paint_q"],
        machine_input_buffers=[buffers["p1_in"]],
    )

    # Create conveyors to route products back to router after each stage
    create_conveyor(env, buffers["m1_out"], buffers["route_out"], name="M1->Route")
    create_conveyor(env, buffers["m2_out"], buffers["route_out"], name="M2->Route")
    create_conveyor(env, buffers["p1_out"], buffers["sink_in"], name="P1->Sink")

    # Create sink
    sink = Sink(
        env=env,
        sink_id=0,
        name="MainSink",
        in_buffer=buffers["sink_in"],
    )

    # Print initial state
    print("="*80)
    print("Factory Simulation Initialized")
    print("="*80)
    print("Machines: 2 Drills, 1 Paint")
    print(f"Technicians: {len(techs)}")
    print(f"Product route: {source.route}")
    print(f"Max products: {source.max_products}")
    print("="*80)
    print()

    return env


def run_simulation(duration: int = 300, seed: int = 42):
    """Run the factory simulation for the specified duration."""
    env = build_factory_simulation(seed=seed)

    # Run simulation
    print(f"Starting simulation for {duration} time units...")
    print()
    env.run(until=duration)

    print()
    print("="*80)
    print("Simulation Complete")
    print("="*80)


if __name__ == "__main__":
    run_simulation(duration=300, seed=42)

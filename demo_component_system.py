"""Demonstration of the component-based degradation system.
Shows a ComplexMachine with multiple components running in a simulation.
"""

import simpy as sp

from kata.entities.components.component import MachineComponent
from kata.entities.machines.complex_machine import ComplexMachine
from kata.EntityFactories import (
    create_complex_machine_from_template,
    create_machine_config,
    list_machine_templates,
)
from kata.features.breakdown.simple_breakdown import SimpleBreakdownProcess


class SimpleTechDispatcher:
    """Simple technician dispatcher for demo purposes."""

    def __init__(self, env):
        self.env = env
        self.repair_events = {}

    def wait_until_repaired(self, machine):
        """Return an event that will be triggered when repair is complete."""
        if machine.machine_id not in self.repair_events:
            self.repair_events[machine.machine_id] = self.env.event()
        return self.repair_events[machine.machine_id]

    def request_repair(self, machine):
        """Simulate technician dispatching and repair."""
        from kata.entities.requests.RepairRequest import RepairRequest

        request = RepairRequest(machine, created_at=self.env.now)

        # Get repair time (component-specific if ComplexMachine)
        repair_time = request.get_repair_time()

        # Get component info if available
        comp_info = request.get_failed_component_info()
        if comp_info:
            print(f"[{self.env.now:05}] - [Dispatcher] Component failure detected!")
            print(
                f"           Component: {comp_info['component_id']} ({comp_info['component_type']})"
            )
            print(f"           Estimated repair time: {repair_time}")
        else:
            print(
                f"[{self.env.now:05}] - [Dispatcher] Machine failure, repair time: {repair_time}"
            )

        # Schedule repair
        self.env.process(self._perform_repair(machine, repair_time))

    def _perform_repair(self, machine, repair_time):
        """Perform the repair after the specified time."""
        yield self.env.timeout(repair_time)
        machine.repair(None)
        if machine.machine_id in self.repair_events:
            self.repair_events[machine.machine_id].succeed()
            del self.repair_events[machine.machine_id]


class SimpleProduct:
    """Simple product for demo."""

    def __init__(self, product_id):
        self.product_id = product_id

    def advance(self):
        """Advance to next step in route."""


def product_generator(env, output_buffer, num_products=5):
    """Generate products to process."""
    for i in range(num_products):
        product = SimpleProduct(i)
        yield env.timeout(50)  # Generate a product every 50 time units
        print(f"[{env.now:05}] - [Generator] Created product {i}")
        yield output_buffer.put(product)


def demo_complex_machine_simple():
    """Demo 1: ComplexMachine with simple component configuration."""
    print("\n" + "=" * 70)
    print("  DEMO 1: ComplexMachine with Simple Components")
    print("=" * 70)

    env = sp.Environment()

    # Create buffers
    input_buffer = sp.Store(env, capacity=10)
    output_buffer = sp.Store(env, capacity=10)

    # Create tech dispatcher
    tech_dispatcher = SimpleTechDispatcher(env)

    # Create components with varying failure rates
    components = [
        MachineComponent(
            component_id="motor_main",
            component_type="motor",
            breakdown_process=SimpleBreakdownProcess(
                failure_prob_working=0.02,  # 2% chance per time unit
                failure_prob_idle=0.001,
            ),
            base_repair_time=80.0,
        ),
        MachineComponent(
            component_id="sensor_temp",
            component_type="sensor",
            breakdown_process=SimpleBreakdownProcess(
                failure_prob_working=0.05,  # 5% chance - sensors fail more often
                failure_prob_idle=0.002,
            ),
            base_repair_time=30.0,  # But quicker to fix
        ),
        MachineComponent(
            component_id="bearing_spindle",
            component_type="bearing",
            breakdown_process=SimpleBreakdownProcess(
                failure_prob_working=0.015,  # 1.5% chance
                failure_prob_idle=0.0005,
            ),
            base_repair_time=120.0,  # Takes longer to replace
        ),
    ]

    # Create ComplexMachine
    machine = ComplexMachine(
        env=env,
        machine_id=1,
        mtype="demo_machine",
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        tech_dispatcher=tech_dispatcher,
        components=components,
        process_time=20,
        dt=1,
    )

    print(f"\nMachine created with {len(components)} components:")
    for comp in components:
        print(
            f"  - {comp.get_id()} ({comp.get_type()}): "
            f"repair_time={comp.get_repair_time()}"
        )

    # Start product generator
    env.process(product_generator(env, input_buffer, num_products=8))

    # Run simulation
    print("\nStarting simulation...\n")
    env.run(until=1000)

    print("\nSimulation complete!")
    print(f"Total products processed: {machine.total_processed}")


def demo_complex_machine_from_json():
    """Demo 2: ComplexMachine created from a packaged template."""
    print("\n" + "=" * 70)
    print("  DEMO 2: ComplexMachine from Packaged Template")
    print("=" * 70)

    env = sp.Environment()

    template_name = "assembly_robot"
    print(f"\nLoading template: {template_name}")

    # Create buffers and dispatcher
    input_buffer = sp.Store(env, capacity=10)
    output_buffer = sp.Store(env, capacity=10)
    tech_dispatcher = SimpleTechDispatcher(env)

    machine = create_complex_machine_from_template(
        env=env,
        machine_id=100,
        template_name=template_name,
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        tech_dispatcher=tech_dispatcher,
        dt=1,
    )

    print("\nMachine components:")
    for comp in machine.components:
        print(
            f"  - {comp.get_id()} ({comp.get_type()}): "
            f"repair_time={comp.get_repair_time()}"
        )

    # Start product generator
    env.process(product_generator(env, input_buffer, num_products=5))

    # Run simulation
    print("\nStarting simulation...\n")
    env.run(until=800)

    print("\nSimulation complete!")
    print(f"Total products processed: {machine.total_processed}")


def demo_comparison():
    """Demo 3: Compare different machine templates."""
    print("\n" + "=" * 70)
    print("  DEMO 3: Comparing Different Machine Templates")
    print("=" * 70)

    print("\nAvailable machine templates:")
    for name in list_machine_templates():
        cfg = create_machine_config(name)
        print(f"\n  {name}:")
        print(f"    machine_type: {cfg.machine_type}")
        print(f"    process_time: {cfg.process_time}")
        print(f"    components: {len(cfg.components)}")
        for comp_id, comp in cfg.components.items():
            print(
                f"      - {comp_id} ({comp.component_type}): "
                f"repair={comp.base_repair_time}, "
                f"model={comp.breakdown_model}"
            )


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  Component-Based Degradation System - Demonstrations")
    print("=" * 70)

    # Run demos
    demo_complex_machine_simple()
    demo_complex_machine_from_json()
    demo_comparison()

    print("\n" + "=" * 70)
    print("  All demonstrations completed successfully!")
    print("=" * 70 + "\n")

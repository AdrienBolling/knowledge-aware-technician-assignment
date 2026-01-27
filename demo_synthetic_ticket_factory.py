"""
Demonstration of SyntheticTicketFactory usage.

This example shows how to use the SyntheticTicketFactory to generate
maintenance tickets from machine failures in a SimPy simulation.
"""
import simpy as sp
from kata.entities.components.component import MachineComponent
from kata.entities.machines.complex_machine import ComplexMachine
from kata.entities.machines.machine import Machine
from kata.entities.requests.RepairRequest import RepairRequest
from kata.features.breakdown.simple_breakdown import SimpleBreakdownProcess
from kata.EntityFactories.synthetic_ticket_factory import SyntheticTicketFactory


def demo_basic_usage():
    """Demonstrate basic ticket generation from a repair request."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Ticket Generation from RepairRequest")
    print("="*70)
    
    env = sp.Environment()
    input_buffer = sp.Store(env)
    output_buffer = sp.Store(env)
    
    # Add some items to the buffer to simulate workload
    for i in range(8):
        input_buffer.put(f"product_{i}")
    
    class MockTechDispatcher:
        def wait_until_repaired(self, machine):
            return env.event().succeed()
        def request_repair(self, machine):
            pass
    
    # Create a ComplexMachine with multiple components
    components = [
        MachineComponent(
            component_id="spindle_motor",
            component_type="motor",
            breakdown_process=SimpleBreakdownProcess(failure_prob_working=0.1),
            base_repair_time=120.0,
        ),
        MachineComponent(
            component_id="coolant_pump",
            component_type="pump",
            breakdown_process=SimpleBreakdownProcess(failure_prob_working=0.05),
            base_repair_time=60.0,
        ),
    ]
    
    machine = ComplexMachine(
        env=env,
        machine_id=101,
        mtype="cnc_mill",
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        tech_dispatcher=MockTechDispatcher(),
        components=components,
        process_time=50,
        dt=1,
    )
    
    # Simulate a component failure
    machine.failed_component = components[0]  # Motor fails
    machine.total_processed = 75
    
    # Create a repair request as would happen in real simulation
    repair_request = RepairRequest(machine=machine, created_at=1500)
    
    # Create factory and generate ticket
    factory = SyntheticTicketFactory()
    ticket = factory.create_ticket_from_repair_request(repair_request)
    
    print(f"\n✓ Generated Ticket:")
    print(f"  Ticket ID: {ticket.ticket_id}")
    print(f"  Machine: {ticket.get_machine_id()} (Type: {ticket.get_machine_type()})")
    print(f"  Failure Type: {ticket.get_failure_type()}")
    print(f"  Component: {ticket.component_type}:{ticket.component_id}")
    print(f"  Priority: {ticket.get_priority()}")
    print(f"  Buffer Level: {ticket.get_buffer_level()} items")
    print(f"  Created At: {ticket.created_at}")
    print(f"  Repair Estimate: {ticket.repair_time_estimate} time units")
    print(f"\n  String representation: {ticket}")


def demo_priority_rules():
    """Demonstrate custom priority rules."""
    print("\n" + "="*70)
    print("DEMO 2: Custom Priority Rules")
    print("="*70)
    
    env = sp.Environment()
    
    class MockTechDispatcher:
        def wait_until_repaired(self, machine):
            return env.event().succeed()
        def request_repair(self, machine):
            pass
    
    # Define custom priority rules for different component types
    priority_rules = {
        "motor": 10,      # Motors are critical
        "sensor": 3,      # Sensors are less critical
        "pump": 7,        # Pumps are moderately critical
        "bearing": 8,     # Bearings are important
    }
    
    print(f"\n✓ Priority Rules:")
    for comp_type, priority in priority_rules.items():
        print(f"  {comp_type}: base priority = {priority}")
    
    # Create factory with custom rules
    factory = SyntheticTicketFactory(priority_rules=priority_rules)
    
    # Create multiple machines with different component failures
    test_cases = [
        ("motor", 15, 150),      # Critical component, high buffer, high productivity
        ("sensor", 2, 20),       # Non-critical, low buffer, low productivity
        ("pump", 8, 80),         # Moderate across the board
    ]
    
    print(f"\n✓ Generated Tickets with Different Priorities:")
    
    for idx, (comp_type, buffer_items, productivity) in enumerate(test_cases):
        input_buffer = sp.Store(env)
        output_buffer = sp.Store(env)
        
        # Fill buffer
        for i in range(buffer_items):
            input_buffer.put(f"item_{i}")
        
        component = MachineComponent(
            component_id=f"{comp_type}_1",
            component_type=comp_type,
            breakdown_process=SimpleBreakdownProcess(failure_prob_working=0.1),
            base_repair_time=50.0,
        )
        
        machine = ComplexMachine(
            env=env,
            machine_id=200 + idx,
            mtype="test_machine",
            input_buffer=input_buffer,
            output_buffer=output_buffer,
            tech_dispatcher=MockTechDispatcher(),
            components=[component],
            process_time=10,
            dt=1,
        )
        machine.failed_component = component
        machine.total_processed = productivity
        
        request = RepairRequest(machine=machine, created_at=2000 + idx * 100)
        ticket = factory.create_ticket_from_repair_request(request)
        
        print(f"\n  Machine {ticket.get_machine_id()}:")
        print(f"    Component: {ticket.component_type}")
        print(f"    Buffer: {ticket.get_buffer_level()} items")
        print(f"    Productivity: {productivity} items")
        print(f"    → Priority: {ticket.get_priority()}")


def demo_batch_generation():
    """Demonstrate batch ticket generation."""
    print("\n" + "="*70)
    print("DEMO 3: Batch Ticket Generation")
    print("="*70)
    
    env = sp.Environment()
    
    class MockTechDispatcher:
        def wait_until_repaired(self, machine):
            return env.event().succeed()
        def request_repair(self, machine):
            pass
    
    # Simulate multiple machine failures
    repair_requests = []
    machine_configs = [
        ("cnc_mill", 10),
        ("lathe", 5),
        ("grinder", 3),
        ("drill_press", 7),
        ("milling_machine", 12),
    ]
    
    for idx, (machine_type, buffer_count) in enumerate(machine_configs):
        input_buffer = sp.Store(env)
        output_buffer = sp.Store(env)
        
        # Fill buffer
        for i in range(buffer_count):
            input_buffer.put(f"product_{i}")
        
        breakdown = SimpleBreakdownProcess(failure_prob_working=0.1)
        machine = Machine(
            env=env,
            machine_id=300 + idx,
            mtype=machine_type,
            input_buffer=input_buffer,
            output_buffer=output_buffer,
            tech_dispatcher=MockTechDispatcher(),
            breakdown_process=breakdown,
            process_time=20,
            dt=1,
        )
        
        request = RepairRequest(machine=machine, created_at=3000 + idx * 50)
        repair_requests.append(request)
    
    # Generate batch of tickets
    factory = SyntheticTicketFactory()
    tickets = factory.create_batch_tickets(repair_requests)
    
    print(f"\n✓ Generated {len(tickets)} tickets in batch:")
    for ticket in tickets:
        print(f"  • Ticket {ticket.ticket_id}: {ticket.get_machine_type()} "
              f"(Machine {ticket.get_machine_id()}, Priority: {ticket.get_priority()}, "
              f"Buffer: {ticket.get_buffer_level()})")


def demo_randomness():
    """Demonstrate randomness in ticket generation."""
    print("\n" + "="*70)
    print("DEMO 4: Randomness in Priority Calculation")
    print("="*70)
    
    env = sp.Environment()
    input_buffer = sp.Store(env)
    output_buffer = sp.Store(env)
    
    class MockTechDispatcher:
        def wait_until_repaired(self, machine):
            return env.event().succeed()
        def request_repair(self, machine):
            pass
    
    # Create factory with randomness enabled
    factory_with_randomness = SyntheticTicketFactory(
        add_randomness=True,
        random_priority_variance=2,  # +/- 2 priority points
    )
    
    # Create one machine
    breakdown = SimpleBreakdownProcess(failure_prob_working=0.1)
    machine = Machine(
        env=env,
        machine_id=400,
        mtype="test_machine",
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        tech_dispatcher=MockTechDispatcher(),
        breakdown_process=breakdown,
        process_time=10,
        dt=1,
    )
    
    # Generate multiple tickets from the same machine to show variance
    print(f"\n✓ Generating 10 tickets from the same machine with randomness:")
    priorities = []
    for i in range(10):
        ticket = factory_with_randomness.create_ticket_from_machine(
            machine, created_at=4000 + i * 10
        )
        priorities.append(ticket.get_priority())
        print(f"  Ticket {ticket.ticket_id}: Priority = {ticket.get_priority()}")
    
    print(f"\n  Priority range: {min(priorities)} - {max(priorities)}")
    print(f"  Average priority: {sum(priorities) / len(priorities):.1f}")
    print(f"  Unique priorities: {sorted(set(priorities))}")


def demo_integration_with_complex_machine():
    """Demonstrate full integration with ComplexMachine failures."""
    print("\n" + "="*70)
    print("DEMO 5: Full Integration with ComplexMachine Component Failures")
    print("="*70)
    
    env = sp.Environment()
    input_buffer = sp.Store(env)
    output_buffer = sp.Store(env)
    
    # Add work items
    for i in range(20):
        input_buffer.put(f"part_{i}")
    
    class MockTechDispatcher:
        def wait_until_repaired(self, machine):
            return env.event().succeed()
        def request_repair(self, machine):
            pass
    
    # Create a realistic machine with multiple components
    components = [
        MachineComponent(
            component_id="main_spindle",
            component_type="motor",
            breakdown_process=SimpleBreakdownProcess(failure_prob_working=0.01),
            base_repair_time=150.0,
        ),
        MachineComponent(
            component_id="hydraulic_pump",
            component_type="pump",
            breakdown_process=SimpleBreakdownProcess(failure_prob_working=0.02),
            base_repair_time=90.0,
        ),
        MachineComponent(
            component_id="temperature_sensor",
            component_type="sensor",
            breakdown_process=SimpleBreakdownProcess(failure_prob_working=0.03),
            base_repair_time=30.0,
        ),
        MachineComponent(
            component_id="main_bearing",
            component_type="bearing",
            breakdown_process=SimpleBreakdownProcess(failure_prob_working=0.015),
            base_repair_time=120.0,
        ),
    ]
    
    machine = ComplexMachine(
        env=env,
        machine_id=500,
        mtype="advanced_cnc_mill",
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        tech_dispatcher=MockTechDispatcher(),
        components=components,
        process_time=100,
        dt=1,
    )
    machine.total_processed = 250  # High productivity machine
    
    # Define priority rules that make sense for this machine
    priority_rules = {
        "motor": 10,     # Main spindle is most critical
        "bearing": 9,    # Bearing failure is serious
        "pump": 7,       # Hydraulic system is important
        "sensor": 4,     # Sensor can be temporarily ignored
    }
    
    factory = SyntheticTicketFactory(priority_rules=priority_rules)
    
    print(f"\n✓ Machine Configuration:")
    print(f"  Machine ID: {machine.machine_id}")
    print(f"  Type: {machine.mtype}")
    print(f"  Components: {len(components)}")
    print(f"  Buffer: {len(input_buffer.items)} items waiting")
    print(f"  Total Processed: {machine.total_processed}")
    
    # Simulate different component failures
    print(f"\n✓ Simulating Component Failures:")
    
    for component in components:
        # Simulate this component failing
        machine.failed_component = component
        
        # Create repair request
        request = RepairRequest(machine=machine, created_at=5000)
        
        # Generate ticket
        ticket = factory.create_ticket_from_repair_request(request)
        
        print(f"\n  Component Failure: {component.get_type()} ({component.get_id()})")
        print(f"    → Ticket ID: {ticket.ticket_id}")
        print(f"    → Failure Type: {ticket.get_failure_type()}")
        print(f"    → Priority: {ticket.get_priority()}")
        print(f"    → Repair Estimate: {ticket.repair_time_estimate} time units")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("  SyntheticTicketFactory - Usage Examples")
    print("="*70)
    
    demo_basic_usage()
    demo_priority_rules()
    demo_batch_generation()
    demo_randomness()
    demo_integration_with_complex_machine()
    
    print("\n" + "="*70)
    print("  All Demonstrations Complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

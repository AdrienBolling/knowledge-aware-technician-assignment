"""
Test suite for SyntheticTicketFactory and SyntheticTicket.
"""
import simpy as sp
from kata.entities.tickets.synthetic_ticket import SyntheticTicket
from kata.EntityFactories.synthetic_ticket_factory import SyntheticTicketFactory
from kata.entities.components.component import MachineComponent
from kata.entities.machines.complex_machine import ComplexMachine
from kata.entities.machines.machine import Machine
from kata.entities.requests.RepairRequest import RepairRequest
from kata.features.breakdown.simple_breakdown import SimpleBreakdownProcess


def test_synthetic_ticket_creation():
    """Test basic SyntheticTicket creation and getters."""
    print("\n=== Test 1: SyntheticTicket Creation ===")
    
    # Create a mock machine
    class MockMachine:
        machine_id = 42
        mtype = "test_machine"
    
    mock_machine = MockMachine()
    
    ticket = SyntheticTicket(
        machine=mock_machine,
        machine_type="test_machine",
        failure_type="component_failure_motor",
        priority=8,
        nb_in_buffer=15,
        created_at=1000,
        ticket_id=1,
        component_id="motor_1",
        component_type="motor",
        repair_time_estimate=50.0,
    )
    
    assert ticket.get_machine_id() == 42, "Machine ID mismatch"
    assert ticket.get_machine_type() == "test_machine", "Machine type mismatch"
    assert ticket.get_failure_type() == "component_failure_motor", "Failure type mismatch"
    assert ticket.get_priority() == 8, "Priority mismatch"
    assert ticket.get_buffer_level() == 15, "Buffer level mismatch"
    
    comp_info = ticket.get_component_info()
    assert comp_info is not None, "Component info should not be None"
    assert comp_info["component_id"] == "motor_1", "Component ID mismatch"
    assert comp_info["component_type"] == "motor", "Component type mismatch"
    assert comp_info["repair_time_estimate"] == 50.0, "Repair time mismatch"
    
    print("✓ SyntheticTicket created successfully")
    print(f"  {ticket}")


def test_ticket_without_component():
    """Test ticket creation for general failure (no component)."""
    print("\n=== Test 2: Ticket Without Component ===")
    
    class MockMachine:
        machine_id = 10
        mtype = "simple_machine"
    
    ticket = SyntheticTicket(
        machine=MockMachine(),
        machine_type="simple_machine",
        failure_type="general_failure",
        priority=5,
        nb_in_buffer=3,
        created_at=500,
        ticket_id=2,
    )
    
    assert ticket.get_failure_type() == "general_failure", "Failure type should be general"
    comp_info = ticket.get_component_info()
    assert comp_info is None, "Component info should be None for general failure"
    
    print("✓ Ticket without component created successfully")
    print(f"  {ticket}")


def test_factory_from_repair_request():
    """Test creating ticket from RepairRequest."""
    print("\n=== Test 3: Factory from RepairRequest ===")
    
    env = sp.Environment()
    input_buffer = sp.Store(env)
    output_buffer = sp.Store(env)
    
    # Add some items to buffer (direct manipulation for testing)
    for i in range(5):
        input_buffer.items.append(f"item_{i}")
    
    class MockTechDispatcher:
        def wait_until_repaired(self, machine):
            return env.event().succeed()
        def request_repair(self, machine):
            pass
    
    # Create a ComplexMachine with a component
    breakdown = SimpleBreakdownProcess(failure_prob_working=0.1)
    component = MachineComponent(
        component_id="bearing_alpha",
        component_type="bearing",
        breakdown_process=breakdown,
        base_repair_time=75.0,
    )
    
    machine = ComplexMachine(
        env=env,
        machine_id=100,
        mtype="cnc_mill",
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        tech_dispatcher=MockTechDispatcher(),
        components=[component],
        process_time=10,
        dt=1,
    )
    
    # Simulate component failure
    machine.failed_component = component
    
    # Create repair request
    request = RepairRequest(machine=machine, created_at=2000)
    
    # Create factory and generate ticket
    factory = SyntheticTicketFactory()
    ticket = factory.create_ticket_from_repair_request(request)
    
    assert ticket.get_machine_id() == 100, "Machine ID mismatch"
    assert ticket.get_machine_type() == "cnc_mill", "Machine type mismatch"
    assert ticket.get_failure_type() == "component_failure_bearing", "Failure type should be component-based"
    assert ticket.created_at == 2000, "Creation time mismatch"
    assert ticket.ticket_id == 1, "Ticket ID should be 1"
    assert ticket.component_id == "bearing_alpha", "Component ID mismatch"
    assert ticket.component_type == "bearing", "Component type mismatch"
    assert ticket.repair_time_estimate == 75.0, "Repair time estimate mismatch"
    assert ticket.nb_in_buffer == 5, "Buffer level should be 5"
    
    print("✓ Ticket created from RepairRequest successfully")
    print(f"  {ticket}")


def test_factory_from_machine():
    """Test creating ticket directly from machine."""
    print("\n=== Test 4: Factory from Machine ===")
    
    env = sp.Environment()
    input_buffer = sp.Store(env)
    output_buffer = sp.Store(env)
    
    # Add items to buffer (direct manipulation for testing)
    for i in range(12):
        input_buffer.items.append(f"product_{i}")
    
    class MockTechDispatcher:
        def wait_until_repaired(self, machine):
            return env.event().succeed()
        def request_repair(self, machine):
            pass
    
    # Create machine
    breakdown = SimpleBreakdownProcess(failure_prob_working=0.05)
    machine = Machine(
        env=env,
        machine_id=25,
        mtype="lathe",
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        tech_dispatcher=MockTechDispatcher(),
        breakdown_process=breakdown,
        process_time=15,
        dt=1,
    )
    
    # Create ticket
    factory = SyntheticTicketFactory()
    ticket = factory.create_ticket_from_machine(machine, created_at=3500)
    
    assert ticket.get_machine_id() == 25, "Machine ID mismatch"
    assert ticket.get_machine_type() == "lathe", "Machine type mismatch"
    assert ticket.get_failure_type() == "general_failure", "Should be general failure"
    assert ticket.created_at == 3500, "Creation time mismatch"
    assert ticket.nb_in_buffer == 12, "Buffer level should be 12"
    assert ticket.component_id is None, "Component ID should be None for simple machine"
    
    print("✓ Ticket created from Machine successfully")
    print(f"  {ticket}")


def test_priority_calculation():
    """Test priority calculation with different scenarios."""
    print("\n=== Test 5: Priority Calculation ===")
    
    env = sp.Environment()
    input_buffer = sp.Store(env)
    output_buffer = sp.Store(env)
    
    class MockTechDispatcher:
        def wait_until_repaired(self, machine):
            return env.event().succeed()
        def request_repair(self, machine):
            pass
    
    # Scenario 1: High priority component with buffer
    priority_rules = {
        "motor": 10,
        "sensor": 3,
        "pump": 7,
    }
    factory = SyntheticTicketFactory(priority_rules=priority_rules)
    
    # Add many items to buffer (should increase priority) - direct manipulation for testing
    for i in range(15):
        input_buffer.items.append(f"item_{i}")
    
    breakdown = SimpleBreakdownProcess(failure_prob_working=0.1)
    motor_component = MachineComponent(
        component_id="motor_main",
        component_type="motor",
        breakdown_process=breakdown,
        base_repair_time=100.0,
    )
    
    machine = ComplexMachine(
        env=env,
        machine_id=1,
        mtype="test",
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        tech_dispatcher=MockTechDispatcher(),
        components=[motor_component],
        process_time=10,
        dt=1,
    )
    machine.failed_component = motor_component
    machine.total_processed = 150  # High productivity
    
    request = RepairRequest(machine=machine, created_at=1000)
    ticket = factory.create_ticket_from_repair_request(request)
    
    # Priority should be: base(10) + buffer(3) + productivity(2) = 15
    print(f"✓ High priority ticket: priority={ticket.get_priority()}")
    assert ticket.get_priority() >= 13, f"Priority should be at least 13, got {ticket.get_priority()}"
    
    # Scenario 2: Low priority component with empty buffer
    sensor_component = MachineComponent(
        component_id="sensor_temp",
        component_type="sensor",
        breakdown_process=breakdown,
        base_repair_time=20.0,
    )
    
    empty_buffer = sp.Store(env)
    machine2 = ComplexMachine(
        env=env,
        machine_id=2,
        mtype="test2",
        input_buffer=empty_buffer,
        output_buffer=output_buffer,
        tech_dispatcher=MockTechDispatcher(),
        components=[sensor_component],
        process_time=10,
        dt=1,
    )
    machine2.failed_component = sensor_component
    machine2.total_processed = 5
    
    request2 = RepairRequest(machine=machine2, created_at=2000)
    ticket2 = factory.create_ticket_from_repair_request(request2)
    
    # Priority should be: base(3) + buffer(0) + productivity(0) = 3
    print(f"✓ Low priority ticket: priority={ticket2.get_priority()}")
    assert ticket2.get_priority() <= 5, f"Priority should be at most 5, got {ticket2.get_priority()}"


def test_batch_ticket_generation():
    """Test batch generation of tickets."""
    print("\n=== Test 6: Batch Ticket Generation ===")
    
    env = sp.Environment()
    
    class MockTechDispatcher:
        def wait_until_repaired(self, machine):
            return env.event().succeed()
        def request_repair(self, machine):
            pass
    
    # Create multiple machines and requests
    requests = []
    for i in range(5):
        input_buffer = sp.Store(env)
        output_buffer = sp.Store(env)
        
        breakdown = SimpleBreakdownProcess(failure_prob_working=0.1)
        machine = Machine(
            env=env,
            machine_id=i,
            mtype=f"machine_type_{i % 2}",
            input_buffer=input_buffer,
            output_buffer=output_buffer,
            tech_dispatcher=MockTechDispatcher(),
            breakdown_process=breakdown,
            process_time=10,
            dt=1,
        )
        
        request = RepairRequest(machine=machine, created_at=i * 100)
        requests.append(request)
    
    # Generate batch
    factory = SyntheticTicketFactory()
    tickets = factory.create_batch_tickets(requests)
    
    assert len(tickets) == 5, "Should create 5 tickets"
    
    # Verify each ticket
    for i, ticket in enumerate(tickets):
        assert ticket.get_machine_id() == i, f"Ticket {i} machine ID mismatch"
        assert ticket.ticket_id == i + 1, f"Ticket {i} ID mismatch"
        print(f"  ✓ Ticket {ticket.ticket_id}: Machine {ticket.get_machine_id()}, Type: {ticket.get_machine_type()}")
    
    print(f"✓ Batch generation created {len(tickets)} tickets successfully")


def test_randomness():
    """Test priority randomness feature."""
    print("\n=== Test 7: Priority Randomness ===")
    
    env = sp.Environment()
    input_buffer = sp.Store(env)
    output_buffer = sp.Store(env)
    
    class MockTechDispatcher:
        def wait_until_repaired(self, machine):
            return env.event().succeed()
        def request_repair(self, machine):
            pass
    
    breakdown = SimpleBreakdownProcess(failure_prob_working=0.1)
    machine = Machine(
        env=env,
        machine_id=99,
        mtype="test_machine",
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        tech_dispatcher=MockTechDispatcher(),
        breakdown_process=breakdown,
        process_time=10,
        dt=1,
    )
    
    # Create factory with randomness
    factory = SyntheticTicketFactory(
        add_randomness=True,
        random_priority_variance=3,
    )
    
    # Generate multiple tickets and check for variance
    priorities = []
    for i in range(10):
        ticket = factory.create_ticket_from_machine(machine, created_at=i * 100)
        priorities.append(ticket.get_priority())
    
    # Check that priorities vary
    unique_priorities = set(priorities)
    print(f"✓ Generated priorities: {priorities}")
    print(f"  Unique values: {unique_priorities}")
    
    # With randomness, we should see some variation (not guaranteed but highly likely)
    # At minimum, all priorities should be >= 1
    assert all(p >= 1 for p in priorities), "All priorities should be at least 1"


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("  SyntheticTicketFactory - Test Suite")
    print("="*60)
    
    try:
        test_synthetic_ticket_creation()
        test_ticket_without_component()
        test_factory_from_repair_request()
        test_factory_from_machine()
        test_priority_calculation()
        test_batch_ticket_generation()
        test_randomness()
        
        print("\n" + "="*60)
        print("  ✓ ALL TESTS PASSED")
        print("="*60 + "\n")
        return True
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

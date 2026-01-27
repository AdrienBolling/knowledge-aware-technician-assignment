"""
Manual test script for Component and ComplexMachine functionality.
This script validates the component-based degradation system.
"""
import simpy as sp
from kata.entities.components.component import MachineComponent
from kata.entities.machines.complex_machine import ComplexMachine
from kata.features.breakdown.simple_breakdown import SimpleBreakdownProcess
from kata.features.breakdown.simple_breakdown import WeibullBreakdownProcess
from kata.entities.tech_dispatcher.GymTechDispatcher import GymTechDispatcher
from kata.entities.requests.RepairRequest import RepairRequest


def test_component_creation():
    """Test creating a MachineComponent."""
    print("\n=== Test 1: Component Creation ===")
    
    breakdown = SimpleBreakdownProcess(failure_prob_working=0.1, failure_prob_idle=0.01)
    component = MachineComponent(
        component_id="motor_1",
        component_type="motor",
        breakdown_process=breakdown,
        base_repair_time=50.0,
    )
    
    assert component.get_id() == "motor_1", "Component ID mismatch"
    assert component.get_type() == "motor", "Component type mismatch"
    assert component.get_repair_time() == 50.0, "Component repair time mismatch"
    
    print("✓ Component created successfully")
    print(f"  ID: {component.get_id()}")
    print(f"  Type: {component.get_type()}")
    print(f"  Repair time: {component.get_repair_time()}")


def test_component_degradation():
    """Test component degradation probabilities."""
    print("\n=== Test 2: Component Degradation ===")
    
    breakdown = SimpleBreakdownProcess(failure_prob_working=0.5, failure_prob_idle=0.05)
    component = MachineComponent(
        component_id="bearing_1",
        component_type="bearing",
        breakdown_process=breakdown,
        base_repair_time=30.0,
    )
    
    # Test failure probability when processing
    prob_working = component.step_and_get_failure_prob(is_processing=True)
    print(f"✓ Failure probability (working): {prob_working}")
    assert prob_working == 0.5, "Working failure probability mismatch"
    
    # Test failure probability when idle
    prob_idle = component.step_and_get_failure_prob(is_processing=False)
    print(f"✓ Failure probability (idle): {prob_idle}")
    assert prob_idle == 0.05, "Idle failure probability mismatch"


def test_repair_request_with_component():
    """Test RepairRequest with component information."""
    print("\n=== Test 3: RepairRequest with Component ===")
    
    # Create a mock machine with failed component
    class MockComplexMachine:
        def __init__(self, failed_comp):
            self._failed_component = failed_comp
        
        def get_failed_component(self):
            return self._failed_component
    
    breakdown = SimpleBreakdownProcess(failure_prob_working=0.1)
    component = MachineComponent(
        component_id="pump_1",
        component_type="pump",
        breakdown_process=breakdown,
        base_repair_time=75.0,
    )
    
    mock_machine = MockComplexMachine(component)
    request = RepairRequest(mock_machine, created_at=100)
    
    # Test repair time from component
    repair_time = request.get_repair_time()
    print(f"✓ Repair time from component: {repair_time}")
    assert repair_time == 75.0, "Component repair time not used"
    
    # Test component info
    comp_info = request.get_failed_component_info()
    print(f"✓ Component info: {comp_info}")
    assert comp_info is not None, "Component info is None"
    assert comp_info["component_id"] == "pump_1", "Component ID mismatch"
    assert comp_info["component_type"] == "pump", "Component type mismatch"
    assert comp_info["repair_time"] == 75.0, "Component repair time mismatch"


def test_complex_machine_with_weibull():
    """Test ComplexMachine with Weibull breakdown process."""
    print("\n=== Test 4: ComplexMachine with Weibull Components ===")
    
    env = sp.Environment()
    input_buffer = sp.Store(env)
    output_buffer = sp.Store(env)
    
    # Mock tech dispatcher
    class MockTechDispatcher:
        def wait_until_repaired(self, machine):
            return env.event().succeed()
        
        def request_repair(self, machine):
            pass
    
    tech_dispatcher = MockTechDispatcher()
    
    # Create components with Weibull degradation
    components = [
        MachineComponent(
            component_id="motor_weibull",
            component_type="motor",
            breakdown_process=WeibullBreakdownProcess(shape=2.0, scale=1000.0, dt=1),
            base_repair_time=100.0,
        ),
        MachineComponent(
            component_id="sensor_weibull",
            component_type="sensor",
            breakdown_process=WeibullBreakdownProcess(shape=1.5, scale=500.0, dt=1),
            base_repair_time=25.0,
        ),
    ]
    
    machine = ComplexMachine(
        env=env,
        machine_id=1,
        mtype="test_machine",
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        tech_dispatcher=tech_dispatcher,
        components=components,
        process_time=10,
        dt=1,
    )
    
    print(f"✓ ComplexMachine created with {len(machine.components)} components")
    for comp in machine.components:
        print(f"  - {comp.get_id()} ({comp.get_type()}): repair time = {comp.get_repair_time()}")
    
    assert len(machine.components) == 2, "Component count mismatch"
    assert machine.failed_component is None, "Machine should not be failed initially"


def test_factory_from_json_config():
    """Test creating ComplexMachine from JSON configuration."""
    print("\n=== Test 5: Factory from JSON Config ===")
    
    import json
    from kata.EntityFactories.complex_machine_factory import create_complex_machine_from_config
    
    # Load a template from the JSON file
    with open("src/kata/resources/machine_templates.json", "r") as f:
        templates = json.load(f)
    
    # Get the basic_cnc_machine template
    config = templates["basic_cnc_machine"]
    
    env = sp.Environment()
    input_buffer = sp.Store(env)
    output_buffer = sp.Store(env)
    
    class MockTechDispatcher:
        def wait_until_repaired(self, machine):
            return env.event().succeed()
        def request_repair(self, machine):
            pass
    
    tech_dispatcher = MockTechDispatcher()
    
    machine = create_complex_machine_from_config(
        env=env,
        machine_id=100,
        machine_config=config,
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        tech_dispatcher=tech_dispatcher,
        dt=1,
    )
    
    print(f"✓ ComplexMachine created from JSON config")
    print(f"  Type: {machine.mtype}")
    print(f"  Components: {len(machine.components)}")
    for comp in machine.components:
        print(f"    - {comp.get_id()} ({comp.get_type()}): repair={comp.get_repair_time()}")
    
    assert len(machine.components) == 3, "Expected 3 components from basic_cnc_machine"
    assert machine.components[0].get_id() == "spindle_motor", "First component ID mismatch"


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("  Component-Based Degradation System - Test Suite")
    print("="*60)
    
    try:
        test_component_creation()
        test_component_degradation()
        test_repair_request_with_component()
        test_complex_machine_with_weibull()
        test_factory_from_json_config()
        
        print("\n" + "="*60)
        print("  ✓ ALL TESTS PASSED")
        print("="*60 + "\n")
        return True
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

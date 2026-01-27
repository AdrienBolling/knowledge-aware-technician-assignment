# Component-Based Machine Degradation System

## Overview

This enhancement introduces a sophisticated component-based degradation system for machines in the Knowledge-Aware Technician Assignment simulation. The system allows modeling machines as aggregates of individual components, each with its own degradation process, failure probabilities, and repair characteristics.

## Key Features

### 1. Component Class
- **Base Class**: `Component` (abstract base in `src/kata/entities/components/base.py`)
- **Implementation**: `MachineComponent` (concrete implementation in `src/kata/entities/components/component.py`)
- **Capabilities**:
  - Individual degradation tracking per component
  - Component-specific failure probabilities
  - Configurable repair times
  - Support for different breakdown models (Simple, Weibull)

### 2. ComplexMachine Class
- **Location**: `src/kata/entities/machines/complex_machine.py`
- **Extends**: `Machine` class (maintains backward compatibility)
- **Features**:
  - Aggregates multiple components
  - Tracks which component caused failure
  - Component-level degradation monitoring
  - Seamless SimPy integration

### 3. Enhanced RepairRequest
- **Location**: `src/kata/entities/requests/RepairRequest.py`
- **Enhancements**:
  - Tracks failed component information
  - Returns component-specific repair times
  - Provides detailed failure diagnostics
  - Backward compatible with regular machines

### 4. Configuration Support
- **ComponentConfig**: Pydantic model for component parameters
- **MachineConfig**: Extended to support component definitions
- **JSON Templates**: Example templates in `src/kata/resources/machine_templates.json`
- **Factory**: `create_complex_machine_from_config()` in `src/kata/EntityFactories/complex_machine_factory.py`

## Usage Examples

### Creating a ComplexMachine with Components

```python
import simpy as sp
from kata.entities.components.component import MachineComponent
from kata.entities.machines.complex_machine import ComplexMachine
from kata.features.breakdown.simple_breakdown import SimpleBreakdownProcess

env = sp.Environment()

# Create components
components = [
    MachineComponent(
        component_id="motor_1",
        component_type="motor",
        breakdown_process=SimpleBreakdownProcess(
            failure_prob_working=0.002,
            failure_prob_idle=0.0001
        ),
        base_repair_time=120.0,
    ),
    MachineComponent(
        component_id="sensor_1",
        component_type="sensor",
        breakdown_process=SimpleBreakdownProcess(
            failure_prob_working=0.005,
            failure_prob_idle=0.0002
        ),
        base_repair_time=30.0,
    ),
]

# Create ComplexMachine
machine = ComplexMachine(
    env=env,
    machine_id=1,
    mtype="cnc_mill",
    input_buffer=input_buffer,
    output_buffer=output_buffer,
    tech_dispatcher=tech_dispatcher,
    components=components,
    process_time=100,
    dt=1,
)
```

### Creating from JSON Configuration

```python
import json
from kata.EntityFactories.complex_machine_factory import create_complex_machine_from_config

# Load configuration
with open("src/kata/resources/machine_templates.json", "r") as f:
    templates = json.load(f)

config = templates["basic_cnc_machine"]

# Create machine from config
machine = create_complex_machine_from_config(
    env=env,
    machine_id=1,
    machine_config=config,
    input_buffer=input_buffer,
    output_buffer=output_buffer,
    tech_dispatcher=tech_dispatcher,
    dt=1,
)
```

### Accessing Failure Information

```python
from kata.entities.requests.RepairRequest import RepairRequest

# When a failure occurs
request = RepairRequest(machine, created_at=env.now)

# Get component-specific repair time
repair_time = request.get_repair_time()

# Get detailed component information
comp_info = request.get_failed_component_info()
if comp_info:
    print(f"Failed component: {comp_info['component_id']}")
    print(f"Component type: {comp_info['component_type']}")
    print(f"Repair time: {comp_info['repair_time']}")
```

## Configuration Format

### JSON Template Structure

```json
{
  "machine_name": {
    "prod_rate": 50,
    "name": "Machine Name",
    "brand": "Brand Name",
    "type": "machine_type",
    "weibull_k": 2.5,
    "weibull_lambda": 2000.0,
    "components": [
      {
        "component_id": "motor_1",
        "component_type": "motor",
        "degradation_rate": 0.002,
        "base_repair_time": 120.0,
        "breakdown_model": "simple",
        "failure_prob_working": 0.002,
        "failure_prob_idle": 0.0002
      },
      {
        "component_id": "sensor_1",
        "component_type": "sensor",
        "degradation_rate": 0.005,
        "base_repair_time": 30.0,
        "breakdown_model": "weibull",
        "weibull_shape": 2.0,
        "weibull_scale": 1000.0
      }
    ]
  }
}
```

### ComponentConfig Fields

- **component_id**: Unique identifier for the component
- **component_type**: Type/category (e.g., "motor", "bearing", "sensor")
- **degradation_rate**: Base degradation rate (for simple model)
- **base_repair_time**: Time required to repair this component
- **breakdown_model**: "simple" or "weibull"
- **failure_prob_working**: Failure probability when machine is working (simple model)
- **failure_prob_idle**: Failure probability when machine is idle (simple model)
- **weibull_shape**: Shape parameter k (Weibull model)
- **weibull_scale**: Scale parameter λ (Weibull model)
- **idle_degradation_factor**: Factor to reduce degradation when idle (default: 0.1)

## Breakdown Models

### Simple Breakdown Model
- Constant failure probabilities
- Separate rates for working and idle states
- Good for components with predictable failure patterns

### Weibull Breakdown Model
- Age-based degradation
- More realistic for mechanical components
- Hazard function: h(t) = (k/λ) * (t/λ)^(k-1)
- Perfect repair resets age to zero

## Testing

### Running Unit Tests

```bash
cd /path/to/knowledge-aware-technician-assignment
PYTHONPATH=src:$PYTHONPATH python tests/test_component_system.py
```

### Running Demonstrations

```bash
PYTHONPATH=src:$PYTHONPATH python demo_component_system.py
```

## Integration with Existing Code

The component-based system is **fully backward compatible**:

- Existing `Machine` class remains unchanged
- `ComplexMachine` extends `Machine` with component functionality
- `RepairRequest` works with both `Machine` and `ComplexMachine`
- No changes required to existing simulation code

## Architecture Notes

### SimPy Integration

The component system integrates seamlessly with SimPy's event simulation:

1. **Degradation Process**: Each component's degradation is checked at regular intervals (dt)
2. **Failure Events**: When a component fails, the machine's main process is interrupted
3. **Repair Events**: Repairs are scheduled through the tech dispatcher with component-specific durations
4. **State Management**: Component states are properly synchronized with the machine's broken/operational status

### Design Principles

1. **Separation of Concerns**: Each component manages its own degradation
2. **Composition over Inheritance**: Machines are composed of components
3. **Extensibility**: Easy to add new component types and breakdown models
4. **Configuration-Driven**: Component parameters defined in JSON for flexibility
5. **Type Safety**: Pydantic models ensure configuration validation

## Security

All code has been scanned with CodeQL and no security vulnerabilities were found.

## Future Enhancements

Potential areas for extension:

1. **Component Dependencies**: Model cascading failures
2. **Preventive Maintenance**: Schedule maintenance based on component age
3. **Part Inventory**: Track spare parts for different component types
4. **Learning Curves**: Technician efficiency improves with component familiarity
5. **Component Wear Visualization**: Real-time degradation dashboards

## Authors

- Implementation: GitHub Copilot
- Co-authored-by: AdrienBolling <97266153+AdrienBolling@users.noreply.github.com>

## License

Follows the same license as the parent repository.

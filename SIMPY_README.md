# SimPy Factory Simulation

This repository contains a complete discrete event simulation of a manufacturing factory using SimPy, designed for knowledge-aware technician assignment research.

## Overview

The simulation models a production line with:
- **Products** flowing through multiple processing stages
- **Machines** that can break down stochastically
- **Technicians** who repair broken machines
- **Buffers** for managing product flow
- **Routers** for directing products to appropriate machines
- **Breakdown processes** modeling machine degradation and failures

## Architecture

### Core Entities

#### Products (`src/kata/entities/products/product.py`)
- Track their route through the production line
- Advance through processing stages
- Support multi-stage routing (e.g., Drill → Paint → Assembly)

#### Machines (`src/kata/entities/machines/machine.py`)
- Process products from input to output buffers
- Implement stochastic breakdown processes
- Can be interrupted during processing for repairs
- Track total processed products and failure history

#### Buffers (`src/kata/entities/buffers/buffer.py`)
- Wrap SimPy's Store with capacity limits
- Manage product queues between entities
- Support blocking put/get operations

#### Technicians (`src/kata/entities/technicians/`)
- Respond to repair requests
- Model travel time, efficiency, fatigue, and knowledge
- Can be assigned through various policies (random, RL-based, etc.)

#### Tech Dispatcher (`src/kata/entities/tech_dispatcher/GymTechDispatcher.py`)
- Manages repair request queue
- Coordinates technician assignments
- Integrates with RL agents for decision-making

#### Source (`src/kata/entities/sources/source.py`)
- Generates products at specified intervals
- Assigns routes to products
- Can limit total production

#### Sink (`src/kata/entities/sinks/sink.py`)
- Collects completed products
- Tracks throughput metrics

#### Router (`src/kata/entities/routers/router.py`)
- Directs products to appropriate machine type buffers
- Routes completed products to sink

#### MachineFeeder (`src/kata/entities/machine_feeder/machine_feeder.py`)
- Load balances products across multiple machines of same type
- Uses round-robin distribution

### Breakdown Processes (`src/kata/features/breakdown/`)

Two implementations provided:

1. **SimpleBreakdownProcess**: Constant failure probability per time step
2. **WeibullBreakdownProcess**: Age-based failure using Weibull distribution

Both support:
- Different failure rates for working vs idle states
- Reset on repair

## Running the Simulation

### Installation

```bash
pip install simpy numpy pydantic pydantic-settings
```

### Basic Usage

```python
import simpy
from kata.entities.products.product import Product
from kata.entities.buffers.buffer import Buffer
from kata.entities.machines.machine import Machine
# ... import other entities

# Create SimPy environment
env = simpy.Environment()

# Create entities
buffer_in = Buffer(env, 0, "INPUT", capacity=10)
buffer_out = Buffer(env, 1, "OUTPUT", capacity=10)

machine = Machine(
    env=env,
    machine_id=1,
    mtype="Drill",
    input_buffer=buffer_in,
    output_buffer=buffer_out,
    tech_dispatcher=dispatcher,
    breakdown_process=SimpleBreakdownProcess(failure_prob_working=0.01),
    process_time=15,
    dt=1,
)

# Run simulation
env.run(until=100)
```

### Complete Example

Run the provided factory simulation:

```bash
python factory_simulation.py
```

This demonstrates a complete production line with:
- 1 Source generating 20 products
- 2 Drill machines
- 1 Paint machine
- 3 Technicians
- Product route: Drill → Paint
- Duration: 300 time units

Expected output:
```
[     7.0] [SRC:MainSource] Creating product 0 with route ['Drill', 'Paint']
[022.0] - [M: 1]  finished processing product 0
[    22.0] [RTR:MainRouter] Routing product 0 to Paint buffer
[042.0] - [M: 3]  finished processing product 0
[    42.0] [SINK:MainSink] Received completed product 0 (total: 1)
[00148] - [M: 2]  BREAKDOWN occurred!
[00148] - [M: 2]  Requesting repair by Tech 0
...
```

## Configuration

The simulation can be configured through:

1. **Machine parameters**:
   - `process_time`: Time to process one product
   - `dt`: Time step for breakdown checks
   - `breakdown_process`: Failure model

2. **Breakdown process parameters**:
   - `failure_prob_working`: Probability of failure while working
   - `failure_prob_idle`: Probability of failure while idle
   - Or Weibull parameters (`shape`, `scale`)

3. **Technician parameters**:
   - `travel_time`: Time to reach machine
   - `efficiency`: Repair speed multiplier
   - `fatigue_lambda`: Fatigue accumulation rate
   - `fatigue_mu`: Fatigue recovery rate

4. **Production parameters**:
   - `interarrival_time`: Time between product arrivals
   - `route`: List of machine types for products
   - `max_products`: Maximum products to generate

## Integration with RL

The simulation is designed to integrate with Reinforcement Learning for technician assignment:

1. **Observation**: Current machine states, technician states, pending tickets
2. **Action**: Select which technician to assign to repair request
3. **Reward**: Based on downtime, throughput, efficiency

The `GymTechDispatcher` provides hooks for RL integration:
- `repair_queue`: SimPy Store of pending repair requests
- `start_repair(tech_id, request)`: Assign technician to request
- `wait_until_repaired(machine)`: Event for repair completion

## Key Features

- **Discrete Event Simulation**: Accurate time-based modeling
- **Stochastic Failures**: Realistic machine breakdown patterns
- **Load Balancing**: Distribute work across multiple machines
- **Repair Coordination**: Queue-based technician assignment
- **Modular Design**: Easy to extend with new entities
- **Type Safety**: Uses type hints throughout
- **Logging**: Detailed event logging for debugging

## File Structure

```
src/kata/
├── entities/
│   ├── products/product.py       # Product with routing
│   ├── machines/machine.py       # Machine with breakdowns
│   ├── buffers/buffer.py         # SimPy Store wrapper
│   ├── technicians/
│   │   ├── technician.py         # Basic technician
│   │   └── GymTechnician.py      # RL-compatible technician
│   ├── tech_dispatcher/
│   │   └── GymTechDispatcher.py  # Repair coordination
│   ├── sources/source.py         # Product generator
│   ├── sinks/sink.py             # Product consumer
│   ├── routers/router.py         # Product routing
│   └── machine_feeder/           # Load balancing
└── features/
    └── breakdown/
        ├── base.py               # Breakdown interface
        └── simple_breakdown.py   # Implementations

factory_simulation.py              # Complete example
```

## Next Steps

Potential extensions:
1. **RL Integration**: Connect with Stable-Baselines3 or Ray RLlib
2. **Knowledge Modeling**: Track technician experience per machine/component
3. **Complex Routing**: Multi-stage products with branching
4. **Batch Processing**: Multiple products per machine cycle
5. **Preventive Maintenance**: Schedule maintenance before failures
6. **Metrics Dashboard**: Real-time KPI tracking
7. **Visualization**: Animated production line view

## License

(Inherit from repository)

## References

- SimPy Documentation: https://simpy.readthedocs.io/
- Discrete Event Simulation: Banks et al., "Discrete-Event System Simulation"
- Maintenance Optimization: Jardine & Tsang, "Maintenance, Replacement, and Reliability"

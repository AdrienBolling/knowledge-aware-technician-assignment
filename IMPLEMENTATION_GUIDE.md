# KataEnv Implementation Guide

## Overview

This document describes the completed implementation of the KataEnv reinforcement learning environment for knowledge-aware technician assignment in manufacturing production lines.

## What Was Implemented

### 1. Core Environment (`src/kata/env.py`)

The `KataEnv` class is a Gymnasium-compatible RL environment that simulates:
- **Production Lines**: Multiple production lines with machines that can fail according to Weibull distributions
- **Technicians**: A pool of technicians who can be assigned to repair broken machines
- **Ticket System**: Repair requests generated when machines fail, with priorities and estimated completion times
- **Machine Degradation**: Uses Kijima Type-I imperfect repair model for realistic aging

#### Key Methods Implemented

1. **`__init__(config: dict)`**
   - Initializes the environment with configuration
   - Sets up observation and action spaces
   - Defines internal state variables

2. **`reset(seed=None, options=None)`**
   - Resets environment to initial state
   - Initializes production lines
   - Steps simulation until first ticket is generated
   - Returns initial observation and info dict

3. **`step(action: int)`**
   - Executes one environment step
   - Assigns current ticket to selected technician
   - Steps production lines forward in time
   - Handles repair completions and machine recoveries
   - Generates new tickets for machine failures
   - Returns (observation, reward, terminated, truncated, info)

4. **`_get_next_obs(current_ticket=None)`**
   - Constructs observation vector from current state
   - Includes ticket features, technician states, and production line states
   - Returns numpy array matching observation space shape

5. **`_get_info()`**
   - Returns diagnostic information about current state
   - Includes episode step, pending tickets, assigned tickets count

6. **`_get_reward()`**
   - Calculates reward based on:
     - Penalty for pending tickets (encourages fast assignment)
     - Penalty for machines in maintenance (-1.0 per broken machine)
     - Small time penalty (-0.01 per step)

7. **`_get_done()`**
   - Checks termination condition (all machines broken)

8. **`_get_truncated()`**
   - Checks if max episode steps reached

#### Utility Function

- **`create_ticket(...)`**: Creates ticket dictionaries for repair requests

### 2. Production Line Functions (`src/kata/funcs.py`)

Fixed critical bug in `step_prod_line` function:
- **Issue**: Return statement was incorrectly indented inside the for loop
- **Fix**: Moved return statement outside the loop to properly return state after all machines are processed

### 3. Testing

Created comprehensive tests (`test_kataenv.py`) covering:
- Ticket creation
- Environment initialization
- Reset functionality
- Step execution
- Invalid action handling
- Full episode simulation

### 4. Example Usage

Created example script (`example_usage.py`) demonstrating:
- Environment configuration
- Random policy implementation
- Episode execution
- Statistics tracking
- Multiple episode runs

## How It Works

### State Management

The environment maintains several state arrays for each production line:

```python
self.status[i]           # Machine status (-1: maintenance, 0: idle, 1: running)
self.in_buff[i]          # Input buffer levels
self.out_buff[i]         # Output buffer levels
self.prod_completions[i] # Production progress
self.absolute_times[i]   # Absolute time counters
self.s_since_repairs[i]  # Time since last repair
self.virtual_ages[i]     # Virtual age (Kijima model)
```

### Ticket Assignment Flow

1. Agent receives observation with current ticket to assign
2. Agent selects a technician (action)
3. Ticket is assigned to technician
4. Production lines step forward
5. Ongoing repairs progress (time_to_complete decrements)
6. Completed repairs restore machines to operational state
7. New tickets generated for machine failures
8. Next ticket presented to agent

### Machine Failure Model

Uses Weibull distribution for failure probability:
- Shape parameter (k): Controls failure rate curve
- Scale parameter (λ): Controls average time to failure
- Kijima Type-I model for imperfect repairs (maintains virtual age)

## Configuration Example

```python
config = {
    "technicians": [
        {"id": 0, "name": "Tech_A"},
        {"id": 1, "name": "Tech_B"},
        {"id": 2, "name": "Tech_C"},
    ],
    "production_lines": [
        {
            "prod_rates": np.array([10, 15, 20]),
            "prod_costs": np.array([100, 150, 200]),
            "in_max_cap": np.array([50, 50, 50]),
            "out_max_cap": np.array([50, 50, 50]),
            "weibull_ks": np.array([2.0, 2.5, 3.0]),
            "weibull_inv_lambdas": np.array([0.01, 0.01, 0.01]),
            "initial_in_buff": 10,
        }
    ],
    "max_episode_steps": 500,
}
```

## Usage

### Basic Usage

```python
from kata.env import KataEnv

# Create environment
env = KataEnv(config)

# Reset and get initial observation
obs, info = env.reset(seed=42)

# Run episode
for step in range(100):
    # Select action (technician to assign ticket)
    action = env.action_space.sample()
    
    # Take step
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break
```

### With RL Agent

```python
import gymnasium as gym
from stable_baselines3 import PPO

# Create environment
env = KataEnv(config)

# Train agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Evaluate
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

## Key Design Decisions

1. **Observation Space**: Flattened vector containing all relevant state information
   - Allows compatibility with standard RL algorithms
   - Includes ticket, technician, and production line features

2. **Action Space**: Discrete space for technician selection
   - Simple and interpretable
   - Invalid actions (assigning to busy technician) result in termination with penalty

3. **Reward Structure**: Penalty-based to encourage efficient operation
   - Penalizes downtime and delays
   - Encourages quick ticket assignment and machine repair

4. **Time Stepping**: Production lines step until next decision point
   - Efficient simulation (skips idle time)
   - Always presents agent with valid decision state

5. **Ticket Generation**: On-demand based on machine failures
   - Realistic simulation of production environment
   - Priority and completion time randomly assigned

## Testing Results

All tests pass successfully:
- ✓ Ticket creation works correctly
- ✓ Environment initialization works correctly  
- ✓ Reset functionality works correctly
- ✓ Step execution works correctly
- ✓ Invalid action handling works correctly
- ✓ Full episode simulation works correctly

Example episode statistics (10 episodes, random policy):
- Average steps per episode: 7.5
- Average total reward: -9.28
- Average tickets assigned: 7.5

## Future Enhancements

Potential improvements for future versions:

1. **Technician Skills**: Add skill matching between technicians and repair types
2. **Knowledge Modeling**: Track and reward knowledge transfer between repairs
3. **Travel Time**: Add physical location and travel time between machines
4. **Priorities**: More sophisticated priority schemes
5. **Batch Assignment**: Allow assigning multiple tickets simultaneously
6. **Rendering**: Add visualization of production line state
7. **Metrics**: Track additional KPIs (throughput, MTTR, MTBF)
8. **Advanced Rewards**: Shaped rewards for better learning

## Dependencies

- `gymnasium>=1.2.0`: RL environment interface
- `numpy>=2.2.6`: Numerical operations
- `numba>=0.61.2`: JIT compilation for performance
- `simpy>=4.1.1`: Discrete event simulation (used in alternative implementation)

## File Structure

```
src/kata/
├── env.py              # Main KataEnv implementation
├── funcs.py            # Production line stepping functions
├── EntityFactories/
│   └── machine_factory.py  # Machine creation utilities
├── core/
│   └── config.py       # Configuration models
└── resources/
    └── machine_templates.json  # Machine templates

tests/
└── simpy_scratch.py    # SimPy-based reference implementation

Root:
├── test_kataenv.py     # Unit tests for KataEnv
└── example_usage.py    # Example usage script
```

## Compatibility

The implementation follows the Gymnasium interface and is compatible with:
- Stable-Baselines3
- Ray RLlib
- Custom RL implementations
- Any Gymnasium-compatible RL framework

## License

(Inherit from repository license)

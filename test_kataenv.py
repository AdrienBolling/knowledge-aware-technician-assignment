"""
Simple test script to verify the KataEnv implementation.
"""

import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from kata.env import KataEnv, create_ticket

def test_create_ticket():
    """Test the create_ticket utility function."""
    print("Testing create_ticket...")
    ticket = create_ticket(
        ticket_id=1,
        priority=5,
        time_to_complete=10,
        machine_id=42,
        component="motor",
        type_of_failure="overheating"
    )
    
    assert ticket["id"] == 1
    assert ticket["priority"] == 5
    assert ticket["time_to_complete"] == 10
    assert ticket["machine_id"] == 42
    assert ticket["component"] == "motor"
    assert ticket["type_of_failure"] == "overheating"
    print("✓ create_ticket works correctly")


def test_kataenv_init():
    """Test KataEnv initialization."""
    print("\nTesting KataEnv initialization...")
    
    # Create a simple config
    config = {
        "technicians": [{"id": 0}, {"id": 1}, {"id": 2}],
        "production_lines": [
            {
                "prod_rates": np.array([10, 15, 20]),
                "prod_costs": np.array([100, 150, 200]),
                "in_max_cap": np.array([50, 50, 50]),
                "out_max_cap": np.array([50, 50, 50]),
                "weibull_ks": np.array([2.0, 2.5, 3.0]),
                "weibull_inv_lambdas": np.array([0.01, 0.01, 0.01]),
                "initial_in_buff": 10,
            },
            {
                "prod_rates": np.array([12, 18]),
                "prod_costs": np.array([120, 180]),
                "in_max_cap": np.array([40, 40]),
                "out_max_cap": np.array([40, 40]),
                "weibull_ks": np.array([2.0, 2.5]),
                "weibull_inv_lambdas": np.array([0.01, 0.01]),
                "initial_in_buff": 10,
            }
        ],
        "max_episode_steps": 100,
    }
    
    env = KataEnv(config)
    
    # Check basic attributes
    assert env.n_technicians == 3
    assert env.n_production_lines == 2
    assert env.max_episode_steps == 100
    assert env.action_space.n == 3
    
    print("✓ KataEnv initialization works correctly")
    return env


def test_kataenv_reset(env):
    """Test KataEnv reset."""
    print("\nTesting KataEnv reset...")
    
    obs, info = env.reset(seed=42)
    
    # Check observation shape
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    
    # Check info
    assert isinstance(info, dict)
    assert "episode_step" in info
    assert "pending_tickets" in info
    
    # Check state initialization
    assert env.episode_step >= 0
    assert len(env.status) == env.n_production_lines
    assert env.current_ticket is not None
    
    print(f"  Episode step: {env.episode_step}")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Current ticket: {env.current_ticket}")
    print("✓ KataEnv reset works correctly")
    return obs, info


def test_kataenv_step(env):
    """Test KataEnv step."""
    print("\nTesting KataEnv step...")
    
    # Reset first
    obs, info = env.reset(seed=42)
    
    # Take a step with a valid action
    action = 0  # Assign to first technician
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    # Check return types
    assert isinstance(next_obs, np.ndarray)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    # Check that observation shape matches
    assert next_obs.shape == env.observation_space.shape
    
    # Check that the ticket was assigned
    assert env.assigned_tickets["ids"][action] != -1
    
    print(f"  Reward: {reward}")
    print(f"  Terminated: {terminated}")
    print(f"  Truncated: {truncated}")
    print(f"  Assigned tickets: {np.sum(env.assigned_tickets['ids'] != -1)}")
    print("✓ KataEnv step works correctly")


def test_kataenv_invalid_action(env):
    """Test KataEnv with invalid action."""
    print("\nTesting KataEnv with invalid action...")
    
    # Reset first
    obs, info = env.reset(seed=42)
    
    # Assign to first technician
    action = 0
    env.step(action)
    
    # Try to assign to the same technician (invalid)
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    # Check that we got a penalty
    assert reward < 0
    assert terminated  # Should terminate on invalid action
    assert "invalid_action" in info
    assert info["invalid_action"] is True
    
    print(f"  Invalid action penalty: {reward}")
    print("✓ KataEnv handles invalid actions correctly")


def test_kataenv_episode():
    """Test a short episode."""
    print("\nTesting a short episode...")
    
    config = {
        "technicians": [{"id": 0}, {"id": 1}],
        "production_lines": [
            {
                "prod_rates": np.array([10, 15]),
                "prod_costs": np.array([100, 150]),
                "in_max_cap": np.array([50, 50]),
                "out_max_cap": np.array([50, 50]),
                "weibull_ks": np.array([2.0, 2.5]),
                "weibull_inv_lambdas": np.array([0.01, 0.01]),
                "initial_in_buff": 10,
            }
        ],
        "max_episode_steps": 50,
    }
    
    env = KataEnv(config)
    obs, info = env.reset(seed=42)
    
    total_reward = 0
    steps = 0
    max_steps_to_test = 10
    
    while steps < max_steps_to_test:
        # Choose a random available technician
        available = np.where(env.assigned_tickets["ids"] == -1)[0]
        if len(available) == 0:
            print(f"  No available technicians at step {steps}")
            break
        
        action = np.random.choice(available)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            print(f"  Episode ended at step {steps}")
            break
    
    print(f"  Completed {steps} steps")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final episode step: {env.episode_step}")
    print("✓ Episode test completed successfully")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing KataEnv Implementation")
    print("=" * 60)
    
    try:
        # Test individual components
        test_create_ticket()
        env = test_kataenv_init()
        test_kataenv_reset(env)
        test_kataenv_step(env)
        test_kataenv_invalid_action(env)
        test_kataenv_episode()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

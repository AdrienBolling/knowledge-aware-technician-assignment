"""
Example usage of the KataEnv reinforcement learning environment.

This script demonstrates:
1. Environment initialization with a simple configuration
2. Using the gymnasium interface for RL
3. Running a simple agent (random policy)
4. Tracking key metrics during the episode
"""

import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from kata.env import KataEnv


def create_simple_config():
    """
    Create a simple configuration for the KataEnv.
    
    This configuration includes:
    - 3 technicians available for repairs
    - 2 production lines with multiple machines each
    - Each machine has production rates, costs, and failure characteristics
    """
    config = {
        # Technicians available for assignment
        "technicians": [
            {"id": 0, "name": "Tech_A"},
            {"id": 1, "name": "Tech_B"},
            {"id": 2, "name": "Tech_C"},
        ],
        
        # Production lines configuration
        "production_lines": [
            {
                # Line 1: 3 machines with different characteristics
                "prod_rates": np.array([10, 15, 20]),  # Production steps per timestep
                "prod_costs": np.array([100, 150, 200]),  # Steps required to complete a product
                "in_max_cap": np.array([50, 50, 50]),  # Max input buffer capacity
                "out_max_cap": np.array([50, 50, 50]),  # Max output buffer capacity
                "weibull_ks": np.array([2.0, 2.5, 3.0]),  # Weibull shape (failure distribution)
                "weibull_inv_lambdas": np.array([0.01, 0.01, 0.01]),  # Weibull scale (1/lambda)
                "initial_in_buff": 10,  # Initial input buffer level
            },
            {
                # Line 2: 2 machines
                "prod_rates": np.array([12, 18]),
                "prod_costs": np.array([120, 180]),
                "in_max_cap": np.array([40, 40]),
                "out_max_cap": np.array([40, 40]),
                "weibull_ks": np.array([2.0, 2.5]),
                "weibull_inv_lambdas": np.array([0.01, 0.01]),
                "initial_in_buff": 10,
            }
        ],
        
        # Episode configuration
        "max_episode_steps": 500,
    }
    
    return config


def run_random_policy(env, num_steps=50, verbose=True):
    """
    Run a simple random policy that assigns tickets to random available technicians.
    
    Args:
        env: The KataEnv environment
        num_steps: Maximum number of steps to run
        verbose: Whether to print detailed information
    
    Returns:
        Dictionary with episode statistics
    """
    # Reset the environment
    obs, info = env.reset(seed=42)
    
    # Statistics tracking
    total_reward = 0
    steps_taken = 0
    tickets_assigned = 0
    invalid_actions = 0
    
    if verbose:
        print("\n" + "="*70)
        print("Starting Episode with Random Policy")
        print("="*70)
        print(f"Initial state:")
        print(f"  Available technicians: {info['available_technicians']}")
        print(f"  Pending tickets: {info['pending_tickets']}")
        print(f"  Episode step: {info['episode_step']}")
    
    # Run the episode
    for step in range(num_steps):
        # Find available technicians
        available = np.where(env.assigned_tickets["ids"] == -1)[0]
        
        if len(available) == 0:
            if verbose:
                print(f"\nStep {step}: No technicians available, ending episode")
            break
        
        # Random policy: choose a random available technician
        action = np.random.choice(available)
        
        if verbose and step < 10:  # Only print first 10 steps to avoid clutter
            print(f"\nStep {step}:")
            print(f"  Current ticket: ID={env.current_ticket['id']}, "
                  f"Priority={env.current_ticket['priority']}, "
                  f"Time={env.current_ticket['time_to_complete']}, "
                  f"Machine={env.current_ticket['machine_id']}")
            print(f"  Action: Assign to technician {action}")
        
        # Take the action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Update statistics
        total_reward += reward
        steps_taken += 1
        
        if "invalid_action" in info and info["invalid_action"]:
            invalid_actions += 1
        else:
            tickets_assigned += 1
        
        if verbose and step < 10:
            print(f"  Reward: {reward:.3f}")
            print(f"  Assigned tickets: {info['assigned_tickets_count']}")
            print(f"  Pending tickets: {info['pending_tickets']}")
            print(f"  Episode step: {info['episode_step']}")
        
        # Check if episode ended
        if terminated or truncated:
            if verbose:
                reason = "Terminated" if terminated else "Truncated"
                print(f"\nEpisode ended ({reason}) at step {step}")
            break
    
    # Print summary
    if verbose:
        print("\n" + "="*70)
        print("Episode Summary")
        print("="*70)
        print(f"Steps taken: {steps_taken}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Average reward per step: {total_reward / steps_taken:.3f}")
        print(f"Tickets assigned: {tickets_assigned}")
        print(f"Invalid actions: {invalid_actions}")
        print(f"Final environment step: {env.episode_step}")
        print("="*70 + "\n")
    
    return {
        "steps_taken": steps_taken,
        "total_reward": total_reward,
        "avg_reward": total_reward / steps_taken if steps_taken > 0 else 0,
        "tickets_assigned": tickets_assigned,
        "invalid_actions": invalid_actions,
    }


def demonstrate_environment():
    """Main demonstration function."""
    print("\n" + "="*70)
    print("KataEnv - Knowledge-Aware Technician Assignment Environment")
    print("="*70)
    
    # Create configuration
    print("\n1. Creating environment configuration...")
    config = create_simple_config()
    print(f"   ✓ Configuration created:")
    print(f"     - Technicians: {len(config['technicians'])}")
    print(f"     - Production lines: {len(config['production_lines'])}")
    print(f"     - Max episode steps: {config['max_episode_steps']}")
    
    # Initialize environment
    print("\n2. Initializing KataEnv...")
    env = KataEnv(config)
    print(f"   ✓ Environment initialized:")
    print(f"     - Observation space shape: {env.observation_space.shape}")
    print(f"     - Action space: Discrete({env.action_space.n})")
    
    # Test reset
    print("\n3. Testing environment reset...")
    obs, info = env.reset(seed=42)
    print(f"   ✓ Environment reset successfully:")
    print(f"     - Observation shape: {obs.shape}")
    print(f"     - Info keys: {list(info.keys())}")
    
    # Run episodes with random policy
    print("\n4. Running episode with random policy...")
    stats = run_random_policy(env, num_steps=20, verbose=True)
    
    # Run multiple episodes for statistics
    print("\n5. Running 10 episodes for statistics...")
    all_stats = []
    for episode in range(10):
        episode_stats = run_random_policy(env, num_steps=30, verbose=False)
        all_stats.append(episode_stats)
    
    # Print aggregate statistics
    print("="*70)
    print("Aggregate Statistics (10 episodes)")
    print("="*70)
    avg_steps = np.mean([s["steps_taken"] for s in all_stats])
    avg_reward = np.mean([s["total_reward"] for s in all_stats])
    avg_tickets = np.mean([s["tickets_assigned"] for s in all_stats])
    print(f"Average steps per episode: {avg_steps:.1f}")
    print(f"Average total reward: {avg_reward:.2f}")
    print(f"Average tickets assigned: {avg_tickets:.1f}")
    print("="*70)
    
    print("\n✓ Demonstration complete!")


def main():
    """Entry point for the example script."""
    try:
        demonstrate_environment()
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

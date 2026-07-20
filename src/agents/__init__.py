"""Agents for the KATA environment."""

from agents.base import Agent
from agents.baselines.heuristics import (
    LeastBusyAgent,
    LeastFatiguedAgent,
    OptimalAssignmentAgent,
    RandomAgent,
    RoundRobinAgent,
    ShortestProcessingTimeAgent,
    ShortestQueueAgent,
)
from agents.dqn.rainbow import RainbowDQNAgent
from agents.grpo.grpo import GRPOAgent
from agents.ppo.ppo_latent import PPOLatentAgent
from agents.ppo.ppo_set_transformer import SetTransformerAgent
from agents.ppo.ppo_transformer import PPOTransformerAgent

__all__ = [
    "Agent",
    "GRPOAgent",
    "LeastBusyAgent",
    "LeastFatiguedAgent",
    "OptimalAssignmentAgent",
    "PPOLatentAgent",
    "PPOTransformerAgent",
    "RainbowDQNAgent",
    "RandomAgent",
    "RoundRobinAgent",
    "SetTransformerAgent",
    "ShortestProcessingTimeAgent",
    "ShortestQueueAgent",
]

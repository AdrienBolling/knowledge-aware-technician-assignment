"""Agents for the KATA environment."""

from agents.base import Agent
from agents.baselines.heuristics import (
    BatchMILPAgent,
    EmpiricalSPTAgent,
    EmpiricalTopsisAgent,
    GreedyRewardAgent,
    LeastBusyAgent,
    LeastFatiguedAgent,
    OptimalAssignmentAgent,
    RandomAgent,
    ReserveSpecialistAgent,
    RoundRobinAgent,
    ShortestProcessingTimeAgent,
    ShortestQueueAgent,
    TopsisAgent,
    TrainWeakestAgent,
)
from agents.dqn.rainbow import RainbowDQNAgent
from agents.grpo.grpo import GRPOAgent
from agents.ppo.ppo_latent import PPOLatentAgent
from agents.ppo.ppo_set_transformer import SetTransformerAgent
from agents.ppo.ppo_transformer import PPOTransformerAgent

__all__ = [
    "Agent",
    "BatchMILPAgent",
    "EmpiricalSPTAgent",
    "EmpiricalTopsisAgent",
    "GRPOAgent",
    "GreedyRewardAgent",
    "LeastBusyAgent",
    "LeastFatiguedAgent",
    "OptimalAssignmentAgent",
    "PPOLatentAgent",
    "PPOTransformerAgent",
    "RainbowDQNAgent",
    "RandomAgent",
    "ReserveSpecialistAgent",
    "RoundRobinAgent",
    "SetTransformerAgent",
    "ShortestProcessingTimeAgent",
    "ShortestQueueAgent",
    "TopsisAgent",
    "TrainWeakestAgent",
]

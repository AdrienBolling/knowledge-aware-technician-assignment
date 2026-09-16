"""Agents for the KATA environment."""

from agents.base import Agent
from agents.baselines.heuristics import (
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
from agents.a2c.a2c_mlp import A2CMLPAgent
from agents.dqn.dql_mlp import DQLMLPAgent
from agents.grpo.grpo_mlp import GRPOMLPAgent
from agents.ppo.ppo_set_transformer import SetTransformerAgent
from agents.ppo.ppo_transformer import PPOTransformerAgent

__all__ = [
    "A2CMLPAgent",
    "Agent",
    "DQLMLPAgent",
    "GRPOMLPAgent",
    "GreedyRewardAgent",
    "LeastBusyAgent",
    "LeastFatiguedAgent",
    "OptimalAssignmentAgent",
    "PPOTransformerAgent",
    "RandomAgent",
    "ReserveSpecialistAgent",
    "RoundRobinAgent",
    "SetTransformerAgent",
    "ShortestProcessingTimeAgent",
    "ShortestQueueAgent",
    "TopsisAgent",
    "TrainWeakestAgent",
]

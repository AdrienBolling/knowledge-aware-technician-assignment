"""Agents for the KATA environment."""

from agents.base import Agent
from agents.baselines.heuristics import (
    BatchMILPAgent,
    EmpiricalSPTAgent,
    EmpiricalTopsisAgent,
    EvoTopsisAgent,
    EvoTopsisInformedAgent,
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
from agents.dqn.rainbow import RainbowDQNAgent
from agents.grpo.grpo import GRPOAgent
from agents.grpo.grpo_mlp import GRPOMLPAgent
from agents.ppo.ppo_latent import PPOLatentAgent
from agents.ppo.ppo_set_transformer import SetTransformerAgent
from agents.ppo.ppo_transformer import PPOTransformerAgent

__all__ = [
    "A2CMLPAgent",
    "Agent",
    "BatchMILPAgent",
    "DQLMLPAgent",
    "EmpiricalSPTAgent",
    "EmpiricalTopsisAgent",
    "EvoTopsisAgent",
    "EvoTopsisInformedAgent",
    "GRPOAgent",
    "GRPOMLPAgent",
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

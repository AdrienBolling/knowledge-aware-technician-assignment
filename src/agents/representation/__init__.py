"""Representation-learning utilities (encoder pretraining via masked-token modelling).

This package separates *state representation* from *control*: a transformer
encoder is trained on rollout data with a BERT-style masked-token objective,
then frozen and reused by RL agents (``PPOLatentAgent``).  The MLM accuracy
during pretraining doubles as a probe for *token learnability* — if even a
well-sized encoder cannot recover masked tokens from context, the
observation alphabet is too noisy / under-specified.
"""

from agents.representation.data import TokenObsBuffer, collect_token_rollouts
from agents.representation.mtm import MaskedTokenModel, MTMTrainer, mlm_collate

__all__ = [
    "MaskedTokenModel",
    "MTMTrainer",
    "TokenObsBuffer",
    "collect_token_rollouts",
    "mlm_collate",
]

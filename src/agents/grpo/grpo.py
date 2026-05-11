"""Group Relative Policy Optimization (GRPO) agent.

GRPO is an on-policy algorithm that:
1. Samples a *group* of actions from the current policy for each state.
2. Evaluates each action by executing it in the environment (or via a
   critic estimate) to obtain rewards.
3. Computes group-relative advantages by normalizing rewards within
   the group (zero-mean, unit-variance).
4. Updates the policy with a clipped surrogate objective (similar to
   PPO) using the group-relative advantages.

This implementation uses a Transformer encoder backbone to process
token-ID observations and outputs a categorical policy over technicians.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from agents.base import Agent, resolve_device
from agents.networks.transformer import TransformerEncoder

# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------


class PolicyNetwork(nn.Module):
    """Categorical policy on top of a Transformer encoder.

    Parameters
    ----------
    encoder:
        Transformer backbone.
    n_actions:
        Number of discrete actions.
    hidden_dim:
        Hidden layer size between encoder output and action logits.

    """

    def __init__(
        self, encoder: TransformerEncoder, n_actions: int, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return action logits.

        Parameters
        ----------
        token_ids:
            ``(batch, seq_len)`` integer token IDs.

        Returns
        -------
        torch.Tensor
            ``(batch, n_actions)`` unnormalized logits.

        """
        features = self.encoder(token_ids)
        return self.head(features)

    def get_distribution(self, token_ids: torch.Tensor) -> Categorical:
        logits = self.forward(token_ids)
        return Categorical(logits=logits)


# ---------------------------------------------------------------------------
# GRPO Agent
# ---------------------------------------------------------------------------


class GRPOAgent(Agent):
    """Group Relative Policy Optimization agent with Transformer encoder.

    Parameters
    ----------
    n_actions:
        Number of technicians.
    vocab_size:
        Token vocabulary size.
    group_size:
        Number of actions sampled per state for group-relative advantage.
    clip_eps:
        PPO-style clipping epsilon.
    entropy_coef:
        Entropy bonus coefficient.
    d_model:
        Transformer embedding dimension.
    n_heads:
        Number of attention heads.
    n_layers:
        Number of Transformer encoder layers.
    hidden_dim:
        Hidden dimension for the policy head.
    lr:
        Learning rate.
    max_seq_len:
        Maximum token sequence length.
    device:
        Torch device string.

    """

    def __init__(
        self,
        n_actions: int,
        vocab_size: int,
        *,
        group_size: int = 8,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        max_seq_len: int = 128,
        device: str = "auto",
    ) -> None:
        super().__init__(n_actions, name="GRPO")
        self.device = torch.device(resolve_device(device))
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.max_seq_len = max_seq_len

        encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
        )
        self.policy = PolicyNetwork(encoder, n_actions, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Rollout buffer for on-policy collection
        self._obs_buffer: list[np.ndarray] = []
        self._action_buffer: list[int] = []
        self._reward_buffer: list[float] = []
        self._logprob_buffer: list[float] = []
        self._done_buffer: list[bool] = []

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _extract_token_ids(self, obs: dict[str, Any]) -> np.ndarray:
        if "token_ids" in obs:
            ids = np.asarray(obs["token_ids"], dtype=np.int64)
        else:
            ids = np.zeros(self.max_seq_len, dtype=np.int64)
        if len(ids) < self.max_seq_len:
            padded = np.zeros(self.max_seq_len, dtype=np.int64)
            padded[: len(ids)] = ids
            return padded
        return ids[: self.max_seq_len]

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: dict[str, Any], *, deterministic: bool = False) -> int:
        token_ids = self._extract_token_ids(obs)
        token_tensor = torch.from_numpy(token_ids).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist = self.policy.get_distribution(token_tensor)

        if deterministic:
            action = dist.probs.argmax(dim=-1).item()
            return int(action)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        self._logprob_buffer.append(log_prob.item())
        return int(action.item())

    def observe_transition(
        self,
        obs: dict[str, Any],
        action: int,
        reward: float,
        next_obs: dict[str, Any],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        self._obs_buffer.append(self._extract_token_ids(obs))
        self._action_buffer.append(action)
        self._reward_buffer.append(reward)
        self._done_buffer.append(terminated or truncated)

    def sample_group_actions(self, obs: dict[str, Any]) -> list[int]:
        """Sample a group of actions for GRPO advantage estimation.

        Returns a list of ``group_size`` actions sampled from the current
        policy.  The caller is expected to evaluate each and call
        :meth:`update_from_group`.
        """
        token_ids = self._extract_token_ids(obs)
        token_tensor = torch.from_numpy(token_ids).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist = self.policy.get_distribution(token_tensor)
            actions = dist.sample((self.group_size,))  # (G, 1)

        return actions.squeeze(-1).tolist()

    def update_from_group(
        self,
        obs: dict[str, Any],
        actions: list[int],
        rewards: list[float],
    ) -> dict[str, float]:
        """Update policy from a single group evaluation.

        Parameters
        ----------
        obs:
            The observation for which the group was sampled.
        actions:
            List of ``group_size`` actions.
        rewards:
            Corresponding rewards for each action.

        Returns
        -------
        dict
            Training metrics.

        """
        token_ids = self._extract_token_ids(obs)
        token_tensor = torch.from_numpy(token_ids).unsqueeze(0).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_arr = np.array(rewards, dtype=np.float32)

        # Group-relative advantage: normalize within the group
        if rewards_arr.std() > 1e-8:
            advantages = (rewards_arr - rewards_arr.mean()) / rewards_arr.std()
        else:
            advantages = np.zeros_like(rewards_arr)
        advantages_t = torch.from_numpy(advantages).to(self.device)

        # Compute old log-probs (detached)
        with torch.no_grad():
            old_dist = self.policy.get_distribution(token_tensor)
            old_log_probs = old_dist.log_prob(actions_t)  # (G,)

        # Policy update with clipped objective
        dist = self.policy.get_distribution(token_tensor.expand(len(actions), -1))
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        ratio = (log_probs - old_log_probs).exp()
        clipped_ratio = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)
        surrogate = torch.min(ratio * advantages_t, clipped_ratio * advantages_t)

        loss = -surrogate.mean() - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "entropy": entropy.item(),
            "mean_advantage": advantages.mean().item(),
        }

    def update(self, **kwargs: Any) -> dict[str, float]:
        """Update from the collected rollout buffer.

        This performs a single epoch of PPO-style updates using
        group-relative advantages computed across the entire buffer.
        """
        if len(self._obs_buffer) == 0:
            return {}

        obs_t = torch.from_numpy(np.stack(self._obs_buffer)).to(self.device)
        actions_t = torch.tensor(
            self._action_buffer, dtype=torch.long, device=self.device
        )
        rewards = np.array(self._reward_buffer, dtype=np.float32)

        # Group-relative advantage over the collected buffer
        if rewards.std() > 1e-8:
            advantages = (rewards - rewards.mean()) / rewards.std()
        else:
            advantages = np.zeros_like(rewards)
        advantages_t = torch.from_numpy(advantages).to(self.device)

        # Old log-probs
        with torch.no_grad():
            old_dist = self.policy.get_distribution(obs_t)
            old_log_probs = old_dist.log_prob(actions_t)

        # Policy gradient with clipping
        dist = self.policy.get_distribution(obs_t)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        ratio = (log_probs - old_log_probs).exp()
        clipped_ratio = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)
        surrogate = torch.min(ratio * advantages_t, clipped_ratio * advantages_t)

        loss = -surrogate.mean() - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        metrics = {
            "loss": loss.item(),
            "entropy": entropy.item(),
            "mean_reward": rewards.mean().item(),
        }

        # Clear buffer after update
        self._obs_buffer.clear()
        self._action_buffer.clear()
        self._reward_buffer.clear()
        self._logprob_buffer.clear()
        self._done_buffer.clear()

        return metrics

    def on_episode_start(self) -> None:
        pass

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

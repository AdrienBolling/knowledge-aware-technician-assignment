"""Rainbow DQN agent for technician dispatching.

Combines several DQN improvements:
- Double DQN (van Hasselt et al., 2016)
- Dueling architecture (Wang et al., 2016)
- Prioritized experience replay (Schaul et al., 2016)
- N-step returns (Sutton, 1988)
- Noisy linear layers (Fortunato et al., 2018)

The agent consumes token-ID observations via a TransformerEncoder backbone.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from agents.base import Agent
from agents.networks.replay_buffer import PrioritizedReplayBuffer
from agents.networks.transformer import TransformerEncoder

# ---------------------------------------------------------------------------
# Noisy Linear layer
# ---------------------------------------------------------------------------


class NoisyLinear(nn.Module):
    """Factorized Gaussian noisy linear layer."""

    def __init__(
        self, in_features: int, out_features: int, sigma_init: float = 0.5
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self._reset_parameters()
        self.reset_noise()

    def _reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _factorized_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        eps_in = self._factorized_noise(self.in_features)
        eps_out = self._factorized_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ---------------------------------------------------------------------------
# Dueling Q-Network
# ---------------------------------------------------------------------------


class DuelingQNetwork(nn.Module):
    """Dueling architecture with noisy heads on top of a Transformer encoder.

    Parameters
    ----------
    encoder:
        Transformer backbone producing ``(batch, d_model)`` vectors.
    n_actions:
        Number of discrete actions.
    hidden_dim:
        Hidden size of the value / advantage streams.

    """

    def __init__(
        self, encoder: TransformerEncoder, n_actions: int, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.encoder = encoder
        d = encoder.output_dim

        # Value stream
        self.value_hidden = NoisyLinear(d, hidden_dim)
        self.value_out = NoisyLinear(hidden_dim, 1)

        # Advantage stream
        self.adv_hidden = NoisyLinear(d, hidden_dim)
        self.adv_out = NoisyLinear(hidden_dim, n_actions)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return Q-values for each action.

        Parameters
        ----------
        token_ids:
            ``(batch, seq_len)`` integer token IDs.

        Returns
        -------
        torch.Tensor
            ``(batch, n_actions)`` Q-values.

        """
        features = self.encoder(token_ids)  # (B, D)

        value = F.relu(self.value_hidden(features))
        value = self.value_out(value)  # (B, 1)

        advantage = F.relu(self.adv_hidden(features))
        advantage = self.adv_out(advantage)  # (B, A)

        # Combine: Q = V + A - mean(A)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q

    def reset_noise(self) -> None:
        for module in [
            self.value_hidden,
            self.value_out,
            self.adv_hidden,
            self.adv_out,
        ]:
            module.reset_noise()


# ---------------------------------------------------------------------------
# Rainbow DQN Agent
# ---------------------------------------------------------------------------


class RainbowDQNAgent(Agent):
    """Rainbow DQN agent with Transformer encoder.

    Parameters
    ----------
    n_actions:
        Number of technicians (discrete actions).
    vocab_size:
        Token vocabulary size for the Transformer encoder.
    d_model:
        Transformer embedding dimension.
    n_heads:
        Number of attention heads.
    n_layers:
        Number of Transformer encoder layers.
    hidden_dim:
        Hidden size for the dueling heads.
    lr:
        Learning rate.
    gamma:
        Discount factor.
    n_step:
        N-step return horizon.
    batch_size:
        Mini-batch size for learning.
    buffer_capacity:
        Replay buffer capacity.
    target_update_freq:
        Steps between target network updates.
    min_replay_size:
        Minimum buffer size before learning starts.
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
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        hidden_dim: int = 128,
        lr: float = 1e-4,
        gamma: float = 0.99,
        n_step: int = 3,
        batch_size: int = 32,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 1_000,
        min_replay_size: int = 1_000,
        max_seq_len: int = 128,
        device: str = "cpu",
    ) -> None:
        super().__init__(n_actions, name="RainbowDQN")
        self.device = torch.device(device)
        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.min_replay_size = min_replay_size
        self.max_seq_len = max_seq_len

        # Networks
        encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
        )
        self.online_net = DuelingQNetwork(encoder, n_actions, hidden_dim).to(
            self.device
        )

        target_encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
        )
        self.target_net = DuelingQNetwork(target_encoder, n_actions, hidden_dim).to(
            self.device
        )
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        # Replay
        self.replay = PrioritizedReplayBuffer(capacity=buffer_capacity)

        # N-step buffer
        self._n_step_buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = (
            deque(maxlen=n_step)
        )

        self._step_count = 0

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _extract_token_ids(self, obs: dict[str, Any]) -> np.ndarray:
        """Get padded token-ID array from observation dict."""
        if "token_ids" in obs:
            ids = np.asarray(obs["token_ids"], dtype=np.int64)
        else:
            ids = np.zeros(self.max_seq_len, dtype=np.int64)
        # Ensure fixed length
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

        self.online_net.reset_noise()
        with torch.no_grad():
            q_values = self.online_net(token_tensor)  # (1, A)

        return int(q_values.argmax(dim=-1).item())

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
        obs_ids = self._extract_token_ids(obs)
        next_obs_ids = self._extract_token_ids(next_obs)
        done = terminated or truncated

        self._n_step_buffer.append((obs_ids, action, reward, next_obs_ids, done))

        # Only push when we have a full n-step transition
        if len(self._n_step_buffer) == self.n_step:
            self._push_n_step()

        # If episode ended, flush remaining transitions
        if done:
            while len(self._n_step_buffer) > 0:
                self._push_n_step()

    def _push_n_step(self) -> None:
        """Compute n-step return and push to replay buffer."""
        obs, action, _, _, _ = self._n_step_buffer[0]
        _, _, _, next_obs, done = self._n_step_buffer[-1]

        # Compute discounted n-step reward
        n_step_reward = 0.0
        for i, (_, _, r, _, d) in enumerate(self._n_step_buffer):
            n_step_reward += (self.gamma**i) * r
            if d:
                done = True
                break

        self.replay.push(obs, action, n_step_reward, next_obs, done)
        self._n_step_buffer.popleft()

    def update(self, **kwargs: Any) -> dict[str, float]:
        if len(self.replay) < self.min_replay_size:
            return {}

        self._step_count += 1

        # Sample from prioritized replay
        batch, indices, weights = self.replay.sample(self.batch_size)
        obs = batch["obs"].to(self.device)
        actions = batch["action"].to(self.device)
        rewards = batch["reward"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        dones = batch["done"].to(self.device)
        weights_t = torch.from_numpy(weights).to(self.device)

        # Reset noise for both networks
        self.online_net.reset_noise()
        self.target_net.reset_noise()

        # Current Q-values
        q_values = self.online_net(obs)  # (B, A)
        q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # Double DQN target
        with torch.no_grad():
            next_q_online = self.online_net(next_obs)  # (B, A)
            best_actions = next_q_online.argmax(dim=-1, keepdim=True)  # (B, 1)
            next_q_target = self.target_net(next_obs)  # (B, A)
            next_q = next_q_target.gather(1, best_actions).squeeze(1)  # (B,)
            target = rewards + (self.gamma**self.n_step) * next_q * (1 - dones)

        # Weighted Huber loss
        td_errors = q - target
        loss = (weights_t * F.smooth_l1_loss(q, target, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update priorities
        new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
        self.replay.update_priorities(indices, new_priorities)

        # Periodic target update
        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return {"loss": loss.item(), "mean_q": q.mean().item()}

    def on_episode_start(self) -> None:
        self._n_step_buffer.clear()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step_count": self._step_count,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step_count = ckpt["step_count"]

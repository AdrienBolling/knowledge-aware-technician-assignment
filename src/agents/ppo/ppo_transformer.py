"""PPO + Transformer agent — the SOTA-grade learner for KATA.

Implements Proximal Policy Optimisation on top of the modern
encoder defined in :mod:`agents.networks.modern_transformer`.  This
is a "kitchen-sink" implementation: every commonly-used PPO trick is
in the box and either active by default or controllable from the
constructor.

What's inside
-------------
* **Backbone**: shared Transformer encoder (RMSNorm + SwiGLU + RoPE +
  CLS pooling).  At default sizes (``d_model=192, n_layers=6``) the
  full actor-critic network reaches ~1M parameters.
* **Heads**: a 2-layer MLP policy head producing logits over
  technicians and a 2-layer MLP value head producing a scalar V(s).
  Both heads share the encoder.
* **Algorithm**: clipped PPO objective with:

  - **GAE-λ** advantage estimation;
  - **Advantage normalisation** per minibatch;
  - **Value-function clipping** (matches Stable-Baselines3's
    ``clip_range_vf`` mode);
  - **Entropy bonus** for exploration;
  - **Adaptive early-stopping** when mean approximate KL exceeds
    ``target_kl`` (PPO best practice);
  - **Cosine LR schedule with linear warmup** over the planned number
    of updates;
  - **Gradient clipping** (max-norm).
* **Rollout**: a fixed-length on-policy buffer that auto-bootstraps
  the truncated final value via the critic.
* **Mixed precision**: optional ``torch.amp.autocast`` + ``GradScaler``
  on CUDA, transparent fall-back on CPU/MPS.
* **Checkpointing**: ``save`` / ``load`` persist model, optimizer,
  and scheduler state.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from agents.base import Agent, resolve_device
from agents.networks.hybrid_encoder import HybridTokenEncoder
from agents.networks.modern_transformer import ModernTransformerEncoder
from agents.networks.running_stats import RunningMeanStd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Actor-critic network
# ---------------------------------------------------------------------------


class _MLPHead(nn.Module):
    """Tiny 2-layer MLP head shared by the policy and value paths."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """Shared-encoder actor-critic.

    Forward returns ``(logits, value)`` where ``value`` is a
    ``(batch,)`` tensor (squeezed scalar critic output).

    The forward signature is **dict-typed** so the same module can
    drive a plain :class:`ModernTransformerEncoder` (single ``token_ids``
    field) or a :class:`HybridTokenEncoder` (``token_ids`` +
    ``cont_values`` + ``cont_kinds``) — the encoder picks what it needs.
    """

    def __init__(
        self,
        encoder: ModernTransformerEncoder | HybridTokenEncoder,
        n_actions: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self._hybrid = isinstance(encoder, HybridTokenEncoder)
        self.policy_head = _MLPHead(encoder.output_dim, hidden_dim, n_actions)
        self.value_head = _MLPHead(encoder.output_dim, hidden_dim, 1)

        # Per recommendations from the PPO paper / "What Matters in
        # On-Policy RL" — initialise the policy output with very small
        # weights so the initial policy is near-uniform.
        nn.init.orthogonal_(self.policy_head.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.policy_head.net[-1].bias)
        nn.init.orthogonal_(self.value_head.net[-1].weight, gain=1.0)
        nn.init.zeros_(self.value_head.net[-1].bias)

    def _encode(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self._hybrid:
            return self.encoder(
                obs["token_ids"], obs["cont_values"], obs["cont_kinds"]
            )
        return self.encoder(obs["token_ids"])

    def forward(
        self,
        obs: dict[str, torch.Tensor] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Back-compat: callers passing a bare token-id tensor still work.
        if not isinstance(obs, dict):
            obs = {"token_ids": obs}
        features = self._encode(obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value


# ---------------------------------------------------------------------------
# Cosine schedule with linear warmup
# ---------------------------------------------------------------------------


def _cosine_warmup_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_factor: float = 0.0,
) -> float:
    """Multiplier ∈ [``min_factor``, 1] applied to the base LR.

    ``min_factor`` floors the post-warmup multiplier so that a run whose
    true update count exceeds ``total_steps`` (the schedule length is an
    *estimate* — see the launchers' decisions-per-t.u. heuristic) keeps
    learning instead of silently freezing at lr=0.
    """
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return max(float(min_factor), 0.5 * (1.0 + math.cos(math.pi * progress)))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class PPOAgentInfraMixin:
    """Infra shared by :class:`PPOTransformerAgent` and the set-transformer
    agent (which does NOT subclass it — keep single-source helpers here so
    fixes cannot strand in one copy, as the 2026-07 GAE fix did).

    Expects ``self.net`` (nn.Module), ``self.optimizer``,
    ``self.lr_scheduler`` (LambdaLR), ``self.total_updates``,
    ``self.warmup_updates`` and ``self._lr_lambda``.
    """

    @contextmanager
    def _eval_mode_if(self, flag: bool):
        """Temporarily switch the net to eval mode (disables dropout).

        Deterministic acting — inline eval and the benchmark harness —
        must not sample dropout masks: an "argmax" over dropout-noisy
        logits is stochastic across calls (defect D2).
        """
        was_training = self.net.training
        if flag and was_training:
            self.net.eval()
        try:
            yield
        finally:
            if flag and was_training:
                self.net.train()

    def _rearm_lr_schedule_if_exhausted(self) -> None:
        """Re-arm the LR schedule when a restored checkpoint outlives it.

        A checkpoint resumed past its schedule end (e.g. a plateau
        extension restarting from a late round) would otherwise spend
        its entire budget at the cosine tail.  Rewind the schedule to
        just past warmup and re-apply the corresponding LR to the
        optimizer's param groups (the restored optimizer state carries
        the tail LR).
        """
        if self.lr_scheduler.last_epoch < self.total_updates:
            return
        logger.warning(
            "Restored LR schedule is exhausted (step %d >= total_updates %d)"
            " — re-arming a fresh post-warmup schedule.",
            self.lr_scheduler.last_epoch,
            self.total_updates,
        )
        self.lr_scheduler.last_epoch = int(self.warmup_updates)
        for group, base in zip(
            self.optimizer.param_groups, self.lr_scheduler.base_lrs
        ):
            group["lr"] = base * self._lr_lambda(int(self.warmup_updates))


class PPOTransformerAgent(PPOAgentInfraMixin, Agent):
    """PPO agent with a Transformer encoder backbone."""

    def __init__(
        self,
        n_actions: int,
        vocab_size: int,
        *,
        # Transformer architecture
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 6,
        d_ff: int | None = None,
        head_hidden_dim: int = 256,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        # PPO hyperparameters
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        clip_eps_vf: float | None = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        target_kl: float | None = 0.02,
        normalize_advantages: bool = True,
        # Optimiser
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        # Rollout / update
        rollout_steps: int = 2048,
        n_epochs: int = 4,
        minibatch_size: int = 256,
        total_updates: int = 200,
        warmup_updates: int = 10,
        lr_min_factor: float = 0.05,
        # Action masking
        use_action_mask: bool = True,
        # Reward normalization (Schulman/SB3 VecNormalize-style)
        normalize_rewards: bool = False,
        # Mixed precision
        use_amp: bool = False,
        # Hybrid observations (PLE / Time2Vec / Fourier continuous channels)
        hybrid_obs: bool = False,
        sim_time_scale: float = 200_000.0,
        # Misc
        seed: int | None = None,
        device: str = "auto",
    ) -> None:
        super().__init__(n_actions, name="PPOTransformer")

        # -- Resolve device (cuda → mps → cpu when "auto") --
        self.device = torch.device(resolve_device(device))
        if seed is not None:
            torch.manual_seed(int(seed))

        # -- Network --
        self.hybrid_obs = bool(hybrid_obs)
        if self.hybrid_obs:
            encoder: ModernTransformerEncoder | HybridTokenEncoder = HybridTokenEncoder(
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                dropout=dropout,
                sim_time_scale=sim_time_scale,
            )
        else:
            encoder = ModernTransformerEncoder(
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
        self.net = ActorCritic(encoder, n_actions, hidden_dim=head_hidden_dim).to(
            self.device
        )

        # -- Optimiser & scheduler --
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=1e-5,
            betas=(0.9, 0.999),
        )
        self.total_updates = int(total_updates)
        self.warmup_updates = int(warmup_updates)
        self.lr_min_factor = float(lr_min_factor)
        self._lr_lambda = lambda step: _cosine_warmup_lr(
            step,
            warmup_steps=self.warmup_updates,
            total_steps=self.total_updates,
            min_factor=self.lr_min_factor,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._lr_lambda
        )

        # -- Mixed precision --
        self._amp_enabled = bool(use_amp and self.device.type == "cuda")
        self._scaler = (
            torch.amp.GradScaler("cuda") if self._amp_enabled else None
        )

        # -- Hyperparameters --
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.clip_eps_vf = clip_eps_vf
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.target_kl = target_kl
        self.normalize_advantages = bool(normalize_advantages)
        self.max_grad_norm = float(max_grad_norm)
        self.rollout_steps = int(rollout_steps)
        self.n_epochs = int(n_epochs)
        self.minibatch_size = int(minibatch_size)
        self.max_seq_len = int(max_seq_len)
        self.vocab_size = int(vocab_size)
        self.use_action_mask = bool(use_action_mask)
        self.normalize_rewards = bool(normalize_rewards)

        # -- Running stats for reward normalisation (VecNormalize-style)
        # Tracks the std of the *discounted return* stream, which is
        # what value learning is sensitive to.  Active only when
        # ``normalize_rewards`` is True.
        self._return_rms = RunningMeanStd()
        self._return_running: float = 0.0

        # -- Rollout buffers --
        self._obs_buffer: list[dict[str, np.ndarray]] = []
        self._action_buffer: list[int] = []
        self._reward_buffer: list[float] = []
        self._done_buffer: list[bool] = []
        self._logprob_buffer: list[float] = []
        self._value_buffer: list[float] = []
        self._mask_buffer: list[np.ndarray] = []  # action masks at sample time
        self._last_obs: dict[str, np.ndarray] | None = None  # for value bootstrap
        self._last_mask: np.ndarray | None = None  # for final-step bootstrap

    # ------------------------------------------------------------------
    # Convenience: parameter counter for runtime sanity checks
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

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

    def _extract_obs(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Return the dict-of-arrays representation consumed by the network.

        Non-hybrid: ``{"token_ids": ...}`` only.
        Hybrid: ``{"token_ids": ..., "cont_values": ..., "cont_kinds": ...}``.
        """
        out: dict[str, np.ndarray] = {"token_ids": self._extract_token_ids(obs)}
        if self.hybrid_obs:
            cv = np.asarray(
                obs.get("cont_values", np.zeros(self.max_seq_len, dtype=np.float32)),
                dtype=np.float32,
            )
            ck = np.asarray(
                obs.get("cont_kinds", np.zeros(self.max_seq_len, dtype=np.int8)),
                dtype=np.int8,
            )
            # Trim / pad to max_seq_len in lockstep with token_ids
            if len(cv) < self.max_seq_len:
                pad_cv = np.zeros(self.max_seq_len, dtype=np.float32)
                pad_cv[: len(cv)] = cv
                cv = pad_cv
            else:
                cv = cv[: self.max_seq_len]
            if len(ck) < self.max_seq_len:
                pad_ck = np.zeros(self.max_seq_len, dtype=np.int8)
                pad_ck[: len(ck)] = ck
                ck = pad_ck
            else:
                ck = ck[: self.max_seq_len]
            out["cont_values"] = cv
            out["cont_kinds"] = ck
        return out

    def _to_tensor_batch(
        self, obs_list: list[dict[str, np.ndarray]] | dict[str, np.ndarray]
    ) -> dict[str, torch.Tensor]:
        """Stack a list of per-step obs dicts into a batched tensor dict."""
        if isinstance(obs_list, dict):
            # Already batched (single step) — just promote to tensors.
            return {
                k: torch.from_numpy(np.asarray(v)).to(self.device)
                for k, v in obs_list.items()
            }
        stacked = {k: np.stack([o[k] for o in obs_list], axis=0) for k in obs_list[0]}
        return {k: torch.from_numpy(v).to(self.device) for k, v in stacked.items()}

    def _extract_action_mask(self, obs: dict[str, Any]) -> np.ndarray:
        """Return a boolean mask of length ``n_actions``.

        Falls back to all-True when masking is disabled, the obs has no
        mask field, or every entry is zero (no valid action would
        otherwise exist).
        """
        if not self.use_action_mask:
            return np.ones(self.n_actions, dtype=bool)
        raw = obs.get("action_mask") if isinstance(obs, dict) else None
        if raw is None:
            return np.ones(self.n_actions, dtype=bool)
        mask = np.asarray(raw, dtype=bool).reshape(-1)
        if mask.shape[0] != self.n_actions:
            return np.ones(self.n_actions, dtype=bool)
        if not mask.any():
            return np.ones(self.n_actions, dtype=bool)
        return mask

    @property
    def _autocast_ctx(self):
        return (
            torch.amp.autocast("cuda")
            if self._amp_enabled
            else nullcontext()
        )

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: dict[str, Any], *, deterministic: bool = False) -> int:
        obs_dict = self._extract_obs(obs)
        mask = self._extract_action_mask(obs)
        obs_batch = {
            k: torch.from_numpy(v).unsqueeze(0).to(self.device)
            for k, v in obs_dict.items()
        }
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        with self._eval_mode_if(deterministic), torch.no_grad(), self._autocast_ctx:
            logits, value = self.net(obs_batch)
        # Apply mask in fp32 so masked positions stay -inf under softmax
        masked_logits = logits.float().masked_fill(~mask_tensor, float("-inf"))
        dist = Categorical(logits=masked_logits)

        if deterministic:
            action = int(dist.probs.argmax(dim=-1).item())
            log_prob = float(dist.log_prob(torch.tensor([action], device=self.device)).item())
        else:
            action_t = dist.sample()
            action = int(action_t.item())
            log_prob = float(dist.log_prob(action_t).item())

        # Stash the latest scalar critic estimate / log-prob / mask for
        # the buffer (observe_transition fires next).
        self._pending_logprob = log_prob
        self._pending_value = float(value.item())
        self._pending_mask = mask
        return action

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
        _ = info
        done = bool(terminated or truncated)

        # Update the discounted-return tracker and normalise the
        # reward by the running std of returns (SB3 VecNormalize-style).
        # This keeps value targets numerically well-conditioned without
        # changing the *sign* or *relative magnitude* of the rewards.
        r_buf = float(reward)
        if self.normalize_rewards:
            self._return_running = self._return_running * self.gamma + r_buf
            self._return_rms.update(self._return_running)
            std = max(self._return_rms.std, 1e-8)
            r_buf = r_buf / std
            if done:
                self._return_running = 0.0

        self._obs_buffer.append(self._extract_obs(obs))
        self._action_buffer.append(int(action))
        self._reward_buffer.append(r_buf)
        self._done_buffer.append(done)
        self._logprob_buffer.append(float(getattr(self, "_pending_logprob", 0.0)))
        self._value_buffer.append(float(getattr(self, "_pending_value", 0.0)))
        pending_mask = getattr(self, "_pending_mask", None)
        if pending_mask is None:
            pending_mask = np.ones(self.n_actions, dtype=bool)
        self._mask_buffer.append(np.asarray(pending_mask, dtype=bool))

        # Track last observation / mask so we can bootstrap a truncated
        # final value at the boundary of a non-terminal rollout chunk.
        self._last_obs = self._extract_obs(next_obs)
        self._last_mask = self._extract_action_mask(next_obs)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, **kwargs: Any) -> dict[str, float]:
        """Run one PPO update from the current rollout buffer."""
        _ = kwargs
        if not self._obs_buffer:
            return {}

        # Stack each obs-dict channel into a (T, ...) tensor.
        keys = list(self._obs_buffer[0].keys())
        obs_t: dict[str, torch.Tensor] = {
            k: torch.from_numpy(
                np.stack([o[k] for o in self._obs_buffer], axis=0)
            ).to(self.device)
            for k in keys
        }
        actions = np.asarray(self._action_buffer, dtype=np.int64)
        rewards = np.asarray(self._reward_buffer, dtype=np.float32)
        dones = np.asarray(self._done_buffer, dtype=bool)
        old_log_probs = np.asarray(self._logprob_buffer, dtype=np.float32)
        values = np.asarray(self._value_buffer, dtype=np.float32)
        if self._mask_buffer:
            masks = np.stack(self._mask_buffer, axis=0).astype(bool)
        else:
            masks = np.ones((len(actions), self.n_actions), dtype=bool)

        # Bootstrap value for the final state — 0 if it terminated, else V(s_T)
        if dones[-1] or self._last_obs is None:
            last_value = 0.0
        else:
            last_batch = {
                k: torch.from_numpy(v).unsqueeze(0).to(self.device)
                for k, v in self._last_obs.items()
            }
            with torch.no_grad(), self._autocast_ctx:
                _, v = self.net(last_batch)
            last_value = float(v.item())

        advantages, returns = self._compute_gae(rewards, values, dones, last_value)

        # Move to tensors for training
        actions_t = torch.from_numpy(actions).to(self.device)
        old_log_probs_t = torch.from_numpy(old_log_probs).to(self.device)
        old_values_t = torch.from_numpy(values).to(self.device)
        advantages_t = torch.from_numpy(advantages).to(self.device)
        returns_t = torch.from_numpy(returns).to(self.device)
        masks_t = torch.from_numpy(masks).to(self.device)

        n = next(iter(obs_t.values())).shape[0]
        idx = np.arange(n)

        losses, pg_losses, vf_losses, ent_losses, kls, clip_fracs = (
            [], [], [], [], [], []
        )
        early_stop = False
        for epoch in range(self.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.minibatch_size):
                mb = idx[start : start + self.minibatch_size]
                if len(mb) < 2:
                    continue
                mb_t = torch.as_tensor(mb, device=self.device, dtype=torch.long)
                mb_obs = {k: v.index_select(0, mb_t) for k, v in obs_t.items()}
                mb_actions = actions_t.index_select(0, mb_t)
                mb_old_logp = old_log_probs_t.index_select(0, mb_t)
                mb_old_values = old_values_t.index_select(0, mb_t)
                mb_adv = advantages_t.index_select(0, mb_t)
                mb_ret = returns_t.index_select(0, mb_t)
                mb_masks = masks_t.index_select(0, mb_t)

                if self.normalize_advantages and mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                with self._autocast_ctx:
                    logits, value = self.net(mb_obs)
                    # Apply the same mask the sampler saw so log-probs
                    # / entropy match the distribution actions were
                    # drawn from.
                    masked_logits = logits.float().masked_fill(~mb_masks, float("-inf"))
                    dist = Categorical(logits=masked_logits)
                    new_logp = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()

                    # PPO policy loss
                    ratio = torch.exp(new_logp - mb_old_logp)
                    clipped_ratio = torch.clamp(
                        ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                    )
                    pg_loss = -torch.min(ratio * mb_adv, clipped_ratio * mb_adv).mean()

                    # Value loss (optionally clipped)
                    if self.clip_eps_vf is not None:
                        vf_clipped = mb_old_values + torch.clamp(
                            value - mb_old_values,
                            -self.clip_eps_vf,
                            self.clip_eps_vf,
                        )
                        vf_loss_unclipped = (value - mb_ret).pow(2)
                        vf_loss_clipped = (vf_clipped - mb_ret).pow(2)
                        vf_loss = 0.5 * torch.max(vf_loss_unclipped, vf_loss_clipped).mean()
                    else:
                        vf_loss = 0.5 * (value - mb_ret).pow(2).mean()

                    loss = (
                        pg_loss
                        + self.value_coef * vf_loss
                        - self.entropy_coef * entropy
                    )

                self.optimizer.zero_grad(set_to_none=True)
                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.net.parameters(), self.max_grad_norm
                    )
                    self._scaler.step(self.optimizer)
                    self._scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.net.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                with torch.no_grad():
                    log_ratio = new_logp - mb_old_logp
                    approx_kl = float(((torch.exp(log_ratio) - 1) - log_ratio).mean().item())
                    clip_frac = float(
                        ((ratio - 1).abs() > self.clip_eps).float().mean().item()
                    )

                losses.append(float(loss.item()))
                pg_losses.append(float(pg_loss.item()))
                vf_losses.append(float(vf_loss.item()))
                ent_losses.append(float(entropy.item()))
                kls.append(approx_kl)
                clip_fracs.append(clip_frac)

            # Adaptive early stopping (per-epoch, on mean-KL)
            if self.target_kl is not None and kls:
                mean_kl = float(np.mean(kls[-(n // self.minibatch_size + 1) :]))
                if mean_kl > 1.5 * self.target_kl:
                    early_stop = True
                    break

        # Step the LR scheduler once per ``update`` call
        self.lr_scheduler.step()

        # Reset rollout buffers
        self._obs_buffer.clear()
        self._action_buffer.clear()
        self._reward_buffer.clear()
        self._done_buffer.clear()
        self._logprob_buffer.clear()
        self._value_buffer.clear()
        self._mask_buffer.clear()

        return {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "pg_loss": float(np.mean(pg_losses)) if pg_losses else float("nan"),
            "vf_loss": float(np.mean(vf_losses)) if vf_losses else float("nan"),
            "entropy": float(np.mean(ent_losses)) if ent_losses else float("nan"),
            "approx_kl": float(np.mean(kls)) if kls else float("nan"),
            "clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "early_stop": float(bool(early_stop)),
            "rollout_size": float(n),
        }

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generalised Advantage Estimation.

        ``dones[t]`` flags that transition *t* ended its episode, i.e.
        ``s_{t+1}`` is terminal: it must mask transition t's OWN
        bootstrap and sever its λ-chain.  The historical variant
        deferred the mask by one step (transition t consumed
        ``dones[t+1]``) — the off-by-one fixed in the set-transformer
        subclass on 2026-07-20 and here on 2026-08-04 (defect D6:
        terminal credit severed at n−2, cross-episode λ-leakage).
        """
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        next_value = float(last_value)
        for t in reversed(range(n)):
            non_terminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "return_rms": self._return_rms.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except (ValueError, KeyError):
                pass
        if "lr_scheduler" in ckpt:
            try:
                self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            except (ValueError, KeyError):
                pass
            else:
                self._rearm_lr_schedule_if_exhausted()
        if "return_rms" in ckpt:
            try:
                self._return_rms.load_state_dict(ckpt["return_rms"])
            except (KeyError, TypeError):
                pass


__all__ = ["PPOTransformerAgent", "ActorCritic"]

"""PPO with the SetTransformer encoder + pointer action head.

This agent consumes the ``set`` observation mode emitted by
:class:`kata.env.KataEnv` — three grouped streams (technicians,
machines, env) padded to fixed max-slot sizes — and produces one logit
per technician slot via a pointer-network attention head.

The PPO update, GAE, clipping, KL early-stop and rollout-buffer
machinery are inherited from :class:`PPOTransformerAgent` unchanged.
Only the network construction and the observation extraction differ.

Why it transfers across environments
------------------------------------

The action head is a *pointer* over candidate technicians: its
parameter count is independent of how many technicians exist in the
environment.  The encoder is sized by the *cap* (``max_techs`` /
``max_machines``), not the real fleet — so the same network can be
loaded into an env with 2 or 30 technicians.  Pretrained weights stay
useful as long as the token vocabulary fits the new env, which holds
when the tokenizer is fit on a warmup sweep covering the target
scenarios.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from agents.base import Agent, resolve_device
from agents.networks.running_stats import RunningMeanStd
from agents.networks.set_transformer import (
    SetTransformerActorCritic,
    SetTransformerEncoder,
)
from agents.ppo.ppo_transformer import _cosine_warmup_lr


_SET_OBS_KEYS: tuple[str, ...] = (
    "tech_token_ids",
    "tech_cont_values",
    "tech_cont_kinds",
    "tech_mask",
    "machine_token_ids",
    "machine_cont_values",
    "machine_cont_kinds",
    "machine_mask",
    "env_token_ids",
    "env_cont_values",
    "env_cont_kinds",
)


class SetTransformerAgent(Agent):
    """PPO over the three-stream Set observation.

    The class is parallel to :class:`PPOTransformerAgent` but with a
    different network and observation schema.  The PPO update loop is
    re-implemented here verbatim to avoid the parent's
    encoder-construction tax.
    """

    def __init__(
        self,
        n_actions: int,                     # = max_techs (the encoder cap)
        vocab_size: int,
        *,
        # Network architecture
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int | None = None,
        dropout: float = 0.1,
        value_hidden: int = 256,
        pointer_d_attn: int = 64,
        max_techs: int = 30,
        max_machines: int = 100,
        env_length: int = 8,
        sim_time_scale: float = 200_000.0,
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
        # Reward normalisation
        normalize_rewards: bool = False,
        # Action masking
        use_action_mask: bool = True,
        # Mixed precision (CUDA only)
        use_amp: bool = False,
        # Misc
        seed: int | None = None,
        device: str = "auto",
    ) -> None:
        super().__init__(n_actions, name="SetTransformer")

        self.device = torch.device(resolve_device(device))
        if seed is not None:
            torch.manual_seed(int(seed))

        if n_actions != max_techs:
            msg = (
                f"SetTransformerAgent expects n_actions == max_techs "
                f"(got n_actions={n_actions}, max_techs={max_techs}).  "
                "The pointer head produces one logit per tech slot."
            )
            raise ValueError(msg)

        encoder = SetTransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_techs=max_techs,
            max_machines=max_machines,
            env_length=env_length,
            sim_time_scale=sim_time_scale,
        )
        self.net = SetTransformerActorCritic(
            encoder,
            value_hidden=value_hidden,
            pointer_d_attn=pointer_d_attn,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=1e-5,
            betas=(0.9, 0.999),
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _cosine_warmup_lr(
                step, warmup_steps=warmup_updates, total_steps=total_updates
            ),
        )

        self._amp_enabled = bool(use_amp and self.device.type == "cuda")
        self._scaler = (
            torch.amp.GradScaler("cuda") if self._amp_enabled else None
        )

        # PPO knobs
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
        self.vocab_size = int(vocab_size)
        self.max_techs = int(max_techs)
        self.max_machines = int(max_machines)
        self.use_action_mask = bool(use_action_mask)
        self.normalize_rewards = bool(normalize_rewards)

        # Return-normalisation running stats (off by default)
        self._return_rms = RunningMeanStd()
        self._return_running: float = 0.0

        # Rollout buffers
        self._obs_buffer: list[dict[str, np.ndarray]] = []
        self._action_buffer: list[int] = []
        self._reward_buffer: list[float] = []
        self._done_buffer: list[bool] = []
        self._logprob_buffer: list[float] = []
        self._value_buffer: list[float] = []
        self._mask_buffer: list[np.ndarray] = []
        self._last_obs: dict[str, np.ndarray] | None = None
        self._last_mask: np.ndarray | None = None

    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    # ------------------------------------------------------------------
    @property
    def _autocast_ctx(self):
        return (
            torch.amp.autocast("cuda") if self._amp_enabled else nullcontext()
        )

    # ------------------------------------------------------------------
    def _extract_obs(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Pull the set-mode keys into a numpy dict.

        Casts each field to the dtype expected by the model and fills
        any missing key with zeros of the expected shape.
        """
        out: dict[str, np.ndarray] = {}
        for k in _SET_OBS_KEYS:
            v = obs.get(k)
            if v is None:
                continue
            if k.endswith("_token_ids"):
                out[k] = np.asarray(v, dtype=np.int64)
            elif k.endswith("_cont_values"):
                out[k] = np.asarray(v, dtype=np.float32)
            elif k.endswith("_cont_kinds"):
                out[k] = np.asarray(v, dtype=np.int8)
            elif k.endswith("_mask"):
                out[k] = np.asarray(v, dtype=np.int8)
            else:
                out[k] = np.asarray(v)
        return out

    def _extract_action_mask(self, obs: dict[str, Any]) -> np.ndarray:
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

    # ------------------------------------------------------------------
    def select_action(
        self, obs: dict[str, Any], *, deterministic: bool = False
    ) -> int:
        obs_dict = self._extract_obs(obs)
        mask = self._extract_action_mask(obs)
        obs_batch = {
            k: torch.from_numpy(v).unsqueeze(0).to(self.device)
            for k, v in obs_dict.items()
        }
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with torch.no_grad(), self._autocast_ctx:
            logits, value = self.net(obs_batch)
        masked_logits = logits.float().masked_fill(~mask_tensor, float("-inf"))
        dist = Categorical(logits=masked_logits)
        if deterministic:
            action = int(dist.probs.argmax(dim=-1).item())
            log_prob = float(
                dist.log_prob(torch.tensor([action], device=self.device)).item()
            )
        else:
            at = dist.sample()
            action = int(at.item())
            log_prob = float(dist.log_prob(at).item())
        self._pending_logprob = log_prob
        self._pending_value = float(value.item())
        self._pending_mask = mask
        return action

    # ------------------------------------------------------------------
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
        self._last_obs = self._extract_obs(next_obs)
        self._last_mask = self._extract_action_mask(next_obs)

    # ------------------------------------------------------------------
    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        next_value = float(last_value)
        next_non_terminal = 0.0 if dones[-1] else 1.0
        for t in reversed(range(n)):
            delta = (
                rewards[t]
                + self.gamma * next_value * next_non_terminal
                - values[t]
            )
            gae = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            )
            advantages[t] = gae
            next_value = values[t]
            next_non_terminal = 0.0 if dones[t] else 1.0
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    def update(self, **kwargs: Any) -> dict[str, float]:
        _ = kwargs
        if not self._obs_buffer:
            return {}

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

        # Bootstrap value at the final state
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

        advantages, returns = self._compute_gae(
            rewards, values, dones, last_value
        )

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
                    masked_logits = logits.float().masked_fill(
                        ~mb_masks, float("-inf")
                    )
                    dist = Categorical(logits=masked_logits)
                    new_logp = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_logp - mb_old_logp)
                    clipped_ratio = torch.clamp(
                        ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                    )
                    pg_loss = -torch.min(
                        ratio * mb_adv, clipped_ratio * mb_adv
                    ).mean()

                    if self.clip_eps_vf is not None:
                        vf_clipped = mb_old_values + torch.clamp(
                            value - mb_old_values,
                            -self.clip_eps_vf,
                            self.clip_eps_vf,
                        )
                        vf_loss_unclipped = (value - mb_ret).pow(2)
                        vf_loss_clipped = (vf_clipped - mb_ret).pow(2)
                        vf_loss = 0.5 * torch.max(
                            vf_loss_unclipped, vf_loss_clipped
                        ).mean()
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
                    approx_kl = float(
                        ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    )
                    clip_frac = float(
                        ((ratio - 1).abs() > self.clip_eps).float().mean().item()
                    )

                losses.append(float(loss.item()))
                pg_losses.append(float(pg_loss.item()))
                vf_losses.append(float(vf_loss.item()))
                ent_losses.append(float(entropy.item()))
                kls.append(approx_kl)
                clip_fracs.append(clip_frac)

            if self.target_kl is not None and kls:
                mean_kl = float(
                    np.mean(kls[-(n // self.minibatch_size + 1) :])
                )
                if mean_kl > 1.5 * self.target_kl:
                    early_stop = True
                    break

        self.lr_scheduler.step()

        # Reset buffers
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
    # Checkpointing — encoder + heads + optimiser
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the agent.

        Also persists the tokenizer's vocabulary (if attached via
        :meth:`attach_vocab`) so eval-time loads do not depend on
        rebuilding the same vocab from configs on a different machine.
        The vocab is the source of truth for token IDs — without it,
        the embedding rows have no portable meaning.
        """
        ckpt = {
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "return_rms": self._return_rms.state_dict(),
            "max_techs": self.max_techs,
            "max_machines": self.max_machines,
            "vocab_size": self.vocab_size,
        }
        vocab = getattr(self, "_vocab", None)
        if vocab is not None:
            ckpt["vocab"] = dict(vocab)
        torch.save(ckpt, path)

    def attach_vocab(self, vocab: dict[str, int]) -> None:
        """Attach a tokenizer vocabulary so it is saved with the checkpoint.

        Called by the experiment runner immediately after agent
        construction — passes ``self.tokenizer.get_vocab()`` so the
        token-id mapping travels with the weights.
        """
        self._vocab = dict(vocab)

    def load(self, path: str | Path) -> None:
        """Restore agent state from a checkpoint.

        Supports **append-only vocabulary growth**: if the agent's
        embedding table is larger than the checkpoint's (e.g. the
        canonical vocab grew between training runs), the checkpoint's
        rows are copied into the first ``ckpt_vocab`` positions of the
        live embedding and the trailing positions retain their
        fresh-init values.  The reverse — checkpoint larger than the
        live embedding — is rejected, since silently dropping rows
        could discard learned representations the policy depends on.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt["net"]
        # Detect and reconcile a row-count mismatch on the token
        # embedding before delegating to load_state_dict.
        state = self._resize_token_embedding(state)
        self.net.load_state_dict(state)
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
        if "return_rms" in ckpt:
            try:
                self._return_rms.load_state_dict(ckpt["return_rms"])
            except (KeyError, TypeError):
                pass
        if "vocab" in ckpt:
            self._vocab = dict(ckpt["vocab"])

    def _resize_token_embedding(self, state: dict) -> dict:
        """Pad or refuse to load the token-embedding row count.

        Looks up every state-dict key whose tail matches
        ``token_embedding.weight``.  If the checkpoint's row count is
        smaller than the live embedding's, copies the checkpoint rows
        into the leading positions of the live tensor and uses that as
        the replacement state-dict entry.  Larger-than-live mismatches
        raise — silently dropping trained rows is the kind of behaviour
        you want to know about, not paper over.
        """
        live = self.net.state_dict()
        out = dict(state)
        for k in list(out.keys()):
            if not k.endswith("token_embedding.weight"):
                continue
            ckpt_w = out[k]
            live_w = live.get(k)
            if live_w is None or ckpt_w.shape == live_w.shape:
                continue
            n_ckpt, d_ckpt = int(ckpt_w.shape[0]), int(ckpt_w.shape[1])
            n_live, d_live = int(live_w.shape[0]), int(live_w.shape[1])
            if d_ckpt != d_live:
                msg = (
                    f"Cannot load checkpoint: token_embedding d_model "
                    f"differs ({d_ckpt} vs {d_live}).  This is a hard "
                    f"architecture mismatch — re-instantiate the agent "
                    f"with the checkpoint's d_model."
                )
                raise RuntimeError(msg)
            if n_ckpt > n_live:
                msg = (
                    f"Cannot load checkpoint: it has {n_ckpt} vocab "
                    f"rows but the live agent has only {n_live}.  The "
                    f"canonical vocab appears to have shrunk; refusing "
                    f"to drop learned rows."
                )
                raise RuntimeError(msg)
            # n_ckpt < n_live → pad with the live fresh-init rows.
            merged = live_w.clone()
            merged[:n_ckpt] = ckpt_w.to(merged.device, dtype=merged.dtype)
            out[k] = merged
            import logging as _logging
            _logging.getLogger(__name__).info(
                "Token embedding resized at load: %d (ckpt) → %d (live); "
                "trailing %d rows fresh-init.",
                n_ckpt, n_live, n_live - n_ckpt,
            )
        return out

    @staticmethod
    def peek_vocab(path: str | Path) -> dict[str, int] | None:
        """Return the vocabulary stored alongside a checkpoint, if any.

        Eval-time callers can use this to build a tokenizer that
        matches the training-time id mapping exactly — see the
        ``benchmark_agent.ipynb`` setup for a usage example.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        v = ckpt.get("vocab") if isinstance(ckpt, dict) else None
        return dict(v) if isinstance(v, dict) else None


__all__ = ["SetTransformerAgent"]

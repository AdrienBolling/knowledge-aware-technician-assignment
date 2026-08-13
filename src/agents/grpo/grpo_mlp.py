"""GRPO over the flattened ``set`` observation — traditional MLP baseline.

Group Relative Policy Optimization is *critic-free*: instead of learning
a value function it scores a **group** of samples drawn from the same
policy and uses their within-group z-score as the advantage.  In the
LLM setting a group is ``G`` completions of one prompt; the episodic-RL
adaptation used here is:

* a **group = ``group_size`` complete episodes** collected under the
  current policy (the serial training loop calls :meth:`update` once per
  episode — ``experiment/runner.py:1489-1492`` — so this class buffers
  episodes across calls and returns ``{}`` until the group is full);
* the **outcome** of episode ``i`` is its raw (undiscounted) return
  ``R_i``;
* every step of episode ``i`` gets the same advantage
  ``A_i = (R_i - mean(R)) / (std(R) + eps)`` — outcome supervision
  broadcast over the trajectory, no per-step credit assignment, no
  critic, no GAE;
* the policy is then improved with the PPO clipped surrogate over
  ``n_epochs`` passes of minibatches.

Why this is a rewrite rather than a patch of ``agents/grpo/grpo.py``
-------------------------------------------------------------------

The legacy :class:`~agents.grpo.grpo.GRPOAgent` is kept untouched for
provenance but is not usable as a baseline; it has four defects that are
designed out here:

1. **The ratio is identically 1** (``grpo.py:310-321``): ``old_log_probs``
   are computed from the *same* network, in the same call, immediately
   before ``log_probs`` — no weight update separates them, so
   ``exp(new - old) == 1`` for every sample and the clipped surrogate
   degenerates to a plain (unclipped, unweighted) policy gradient.  Here
   the old log-probs are computed **once**, in a batched ``no_grad``
   forward, *before* the K-epoch loop, so from the second minibatch
   onwards the ratio genuinely measures policy movement (pinned by
   ``tests/test_grpo_mlp.py::test_ratio_moves_across_epochs``).
2. **Advantages were per-step reward z-scores** (``grpo.py:303-307``):
   normalising the raw per-step reward stream is neither group-relative
   nor a return — it discards everything that happens after the step.
   Here the z-score is taken over *episode outcomes*, which is what
   "group relative" means.
3. **Dead group API**: ``sample_group_actions`` / ``update_from_group``
   have no call site anywhere in the runner, so the "group" never
   existed at training time.  Groups here are formed by the normal
   ``observe_transition`` / ``update`` flow.
4. **No action mask**: the legacy agent sampled over all technician
   slots including absent / busy ones.  Every distribution here is
   masked (``SetObsFlattener.extract_action_mask``), with no all-ones
   fallback.

Observation handling
--------------------

The env's ``set`` observation is flattened to a constant-width vector by
:class:`~agents.networks.mlp_encoder.SetObsFlattener` (one-hot per
categorical slot position from the frozen vocab, symlog per wide-range
scalar) and consumed by a plain MLP policy head.  The flat width is
fixed by the *caps* (``max_techs`` / ``max_machines``), so the same
network handles any fleet size **up to** the cap — but, unlike the
pointer-head SetTransformer, it cannot transfer across different caps.
That limitation is the point of the baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.distributions import Categorical

from agents.base import Agent, resolve_device
from agents.networks.mlp_encoder import MLPPolicy, SetObsFlattener
from agents.ppo.ppo_transformer import PPOAgentInfraMixin, _cosine_warmup_lr

logger = logging.getLogger(__name__)

# Canonical frozen set vocabulary — the same artefact the runner loads
# into the tokenizer (``GymEnvConfig.set_vocab_path``, config.py:722).
DEFAULT_SET_VOCAB_PATH = "run_configs/vocab/set_vocab.json"

# Warn once when a buffered group grows past this many bytes: a group of
# eight 20k-step episodes at the default caps is ~1.7 GB in float16.
_BUFFER_WARN_BYTES = 4 * 1024**3


def _resolve_vocab_path() -> Path:
    """Locate the canonical vocab JSON from cwd or the repo root."""
    p = Path(DEFAULT_SET_VOCAB_PATH)
    if p.is_file():
        return p
    # src/agents/grpo/grpo_mlp.py -> repo root
    root = Path(__file__).resolve().parents[3]
    return root / DEFAULT_SET_VOCAB_PATH


@dataclass
class _Episode:
    """One completed trajectory — the unit GRPO groups over."""

    obs: np.ndarray       # (N, D) flattened observations
    actions: np.ndarray   # (N,)  int64
    masks: np.ndarray     # (N, A) bool — action masks at sample time
    ret: float            # scalar outcome R_i

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    @property
    def nbytes(self) -> int:
        return int(self.obs.nbytes + self.actions.nbytes + self.masks.nbytes)


class GRPOMLPAgent(PPOAgentInfraMixin, Agent):
    """Critic-free GRPO with a plain-MLP policy over the flattened set obs.

    Parameters
    ----------
    n_actions:
        Size of the (padded) action space.  Must equal ``max_techs``:
        set-mode envs expose ``Discrete(max_techs)`` and the flat
        observation is sized by the same cap.
    vocab:
        Frozen set vocabulary — a ``{token: id}`` mapping or a path to a
        vocab JSON.  Defaults to the canonical
        ``run_configs/vocab/set_vocab.json``; the runner overrides it via
        :meth:`attach_vocab` with the tokenizer's own mapping.
    vocab_size:
        Accepted for parity with the runner's token-agent injection
        (``runner.py:363``) — the flattener keys one-hots off the vocab
        *mapping*, so this is only cross-checked, never used to size a
        table.  Deliberately absent from the shipped JSON config: a
        pinned ``vocab_size`` is what left ``rainbow_dqn.json`` at 128
        against a 152-token vocab.
    hidden_sizes:
        Widths of the MLP trunk (Linear-LayerNorm-ReLU per entry).
    max_techs, max_machines, env_length, tech_slot_length,
    machine_slot_length:
        Slot geometry of the set observation.  The first four names
        match what the runner injects for set-obs agents
        (``runner.py:344-359``); ``machine_slot_length`` is not injected
        and defaults to the env's own default (12).
    sim_time_scale:
        Accepted for runner parity and unused: the flattener squashes
        sim time with a stateless ``symlog`` instead of a fitted Fourier
        scale (that unpersisted scale was defect D4).
    group_size:
        ``G`` — number of complete episodes per group update.
    gamma:
        Discount applied when accumulating an episode's outcome.  The
        specified algorithm uses the **raw undiscounted** return, i.e.
        the default ``1.0``; values < 1 are an ablation knob.
    clip_eps, entropy_coef, n_epochs, minibatch_size, target_kl:
        PPO-style surrogate knobs.  ``target_kl`` (default ``None`` =
        off) early-stops the epoch loop when the approximate KL exceeds
        ``1.5 * target_kl``.
    advantage_eps:
        Floor added to the group's standard deviation.  Advantages are
        **not** renormalised per minibatch — that would undo the group
        statistic, which is the entire signal here.
    lr, weight_decay, max_grad_norm, total_updates, warmup_updates,
    lr_min_factor:
        Optimiser and LR-schedule settings.  One schedule step happens
        per *group* update, so ``total_updates`` must be sized in groups
        (``episodes / group_size``), not episodes.
    use_action_mask:
        When ``False`` the policy is unmasked (tests / ablations only).
        When ``True`` an observation without a usable mask raises rather
        than falling back to all-ones.
    store_float16:
        Buffer flattened observations as float16 (default) instead of
        float32, halving the group's memory.  Old log-probs are computed
        from the *stored* features, so the update stays self-consistent;
        the only effect is a ~1e-3 relative rounding between acting-time
        and update-time inputs.
    """

    def __init__(
        self,
        n_actions: int,
        *,
        # Observation encoding
        vocab: Mapping[str, int] | str | Path | None = None,
        vocab_size: int | None = None,
        hidden_sizes: Sequence[int] = (512, 512),
        max_techs: int = 30,
        max_machines: int = 100,
        env_length: int = 16,
        tech_slot_length: int = 16,
        machine_slot_length: int = 12,
        sim_time_scale: float | None = None,
        # GRPO
        group_size: int = 8,
        gamma: float = 1.0,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        n_epochs: int = 4,
        minibatch_size: int = 256,
        advantage_eps: float = 1e-8,
        target_kl: float | None = None,
        # Optimiser
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        total_updates: int = 200,
        warmup_updates: int = 10,
        lr_min_factor: float = 0.05,
        # Misc
        use_action_mask: bool = True,
        store_float16: bool = True,
        seed: int | None = None,
        device: str = "auto",
    ) -> None:
        super().__init__(n_actions, name="GRPOMLP")

        self.device = torch.device(resolve_device(device))
        if seed is not None:
            torch.manual_seed(int(seed))

        if n_actions != max_techs:
            msg = (
                f"GRPOMLPAgent expects n_actions == max_techs (got "
                f"n_actions={n_actions}, max_techs={max_techs}).  The flat "
                "observation and the policy head are both sized by the "
                "technician-slot cap."
            )
            raise ValueError(msg)

        # -- Observation flattener (stateless: no learnable parameters) --
        self._slot_geometry = {
            "max_techs": int(max_techs),
            "max_machines": int(max_machines),
            "tech_slot_len": int(tech_slot_length),
            "machine_slot_len": int(machine_slot_length),
            "env_len": int(env_length),
        }
        if vocab is None:
            vocab = _resolve_vocab_path()
        if not isinstance(vocab, Mapping):
            vocab = SetObsFlattener.load_vocab(vocab)
        # Kept as ``_vocab`` (the name the runner's vocab cross-check and
        # the checkpoint writer both look for).
        self._vocab: dict[str, int] = {str(k): int(v) for k, v in vocab.items()}
        self.flattener = self._build_flattener(self._vocab)
        self.obs_dim = int(self.flattener.out_dim)
        if vocab_size is not None and int(vocab_size) != len(self._vocab):
            logger.warning(
                "vocab_size=%d was injected but the attached vocab holds %d "
                "tokens — the flattener keys its one-hots off the mapping, "
                "so the mapping wins.  Check for vocab drift.",
                int(vocab_size), len(self._vocab),
            )
        self.vocab_size = len(self._vocab)

        # -- Policy network (no value head: GRPO is critic-free) --
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.net = MLPPolicy(self.obs_dim, n_actions, self.hidden_sizes).to(
            self.device
        )

        # -- Optimiser & LR schedule (shared infra with the PPO agents) --
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
        self._ctor_lr = float(lr)
        self._lr_lambda = lambda step: _cosine_warmup_lr(
            step,
            warmup_steps=self.warmup_updates,
            total_steps=self.total_updates,
            min_factor=self.lr_min_factor,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._lr_lambda
        )

        # -- Hyperparameters --
        self.group_size = int(group_size)
        if self.group_size < 2:
            msg = (
                f"group_size must be >= 2 (got {self.group_size}): a "
                "one-episode group has zero variance and therefore no "
                "group-relative signal."
            )
            raise ValueError(msg)
        self.gamma = float(gamma)
        self.clip_eps = float(clip_eps)
        self.entropy_coef = float(entropy_coef)
        self.n_epochs = int(n_epochs)
        self.minibatch_size = int(minibatch_size)
        self.advantage_eps = float(advantage_eps)
        self.target_kl = target_kl
        self.max_grad_norm = float(max_grad_norm)
        self.use_action_mask = bool(use_action_mask)
        self.store_float16 = bool(store_float16)
        self._store_dtype = np.float16 if self.store_float16 else np.float32

        # -- Buffers: one in-progress episode + the accumulating group --
        self._group: list[_Episode] = []
        self._cur_obs: list[np.ndarray] = []
        self._cur_actions: list[int] = []
        self._cur_masks: list[np.ndarray] = []
        self._cur_rewards: list[float] = []
        self._buffer_warned = False

        # -- Diagnostics from the most recent group update --
        self._epoch_ratio_dev: list[float] = []
        self._last_group_returns: np.ndarray = np.zeros(0, dtype=np.float64)
        self._last_advantages: np.ndarray = np.zeros(0, dtype=np.float32)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_flattener(self, vocab: Mapping[str, int]) -> SetObsFlattener:
        return SetObsFlattener(vocab, **self._slot_geometry).to(self.device)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _action_mask(self, obs: dict[str, Any]) -> np.ndarray:
        """Boolean ``(n_actions,)`` mask for ``obs``.

        Delegates to the flattener, which refuses the all-ones fallback:
        a missing or empty mask is a bug worth crashing on, not worth
        papering over with an unmasked policy.
        """
        if not self.use_action_mask:
            return np.ones(self.n_actions, dtype=bool)
        return self.flattener.extract_action_mask(obs)

    def _flatten(self, obs: dict[str, Any]) -> torch.Tensor:
        with torch.no_grad():
            return self.flattener(obs)

    def _masked_dist(
        self, logits: torch.Tensor, mask: torch.Tensor
    ) -> Categorical:
        return Categorical(logits=logits.float().masked_fill(~mask, float("-inf")))

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(
        self, obs: dict[str, Any], *, deterministic: bool = False
    ) -> int:
        """Sample (or argmax) a technician slot.  **Side-effect free.**

        Nothing is buffered here: the transition — including the action
        mask and the flattened features — is recorded by
        :meth:`observe_transition`, and the behaviour log-probs are
        recomputed in :meth:`update` from the stored features.  That
        keeps evaluation rollouts (which never call
        ``observe_transition``) from contaminating a training group.
        """
        mask = self._action_mask(obs)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with self._eval_mode_if(deterministic), torch.no_grad():
            logits = self.net(self._flatten(obs).unsqueeze(0))
        dist = self._masked_dist(logits, mask_t)
        if deterministic:
            return int(dist.probs.argmax(dim=-1).item())
        return int(dist.sample().item())

    # ------------------------------------------------------------------
    # Collection
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
        _ = (next_obs, info)  # critic-free: no bootstrap value is needed
        flat = self._flatten(obs).detach().cpu().numpy()
        self._cur_obs.append(flat.astype(self._store_dtype, copy=False))
        self._cur_actions.append(int(action))
        self._cur_masks.append(self._action_mask(obs))
        self._cur_rewards.append(float(reward))
        if bool(terminated or truncated):
            self._flush_episode()

    def on_episode_start(self) -> None:
        """Close out any still-open episode before a new one begins.

        The serial loop always ends an episode on a ``done`` flag, so
        this is normally a no-op; it exists so a rollout that ended
        without one can never splice two trajectories into a single
        "outcome".
        """
        self._flush_episode()

    def _flush_episode(self) -> None:
        """Move the in-progress episode into the group with its outcome."""
        if not self._cur_obs:
            return
        rewards = np.asarray(self._cur_rewards, dtype=np.float64)
        if self.gamma >= 1.0:
            ret = float(rewards.sum())
        else:
            discounts = np.power(
                self.gamma, np.arange(rewards.shape[0], dtype=np.float64)
            )
            ret = float((rewards * discounts).sum())
        self._group.append(
            _Episode(
                obs=np.stack(self._cur_obs, axis=0),
                actions=np.asarray(self._cur_actions, dtype=np.int64),
                masks=np.stack(self._cur_masks, axis=0).astype(bool),
                ret=ret,
            )
        )
        self._cur_obs.clear()
        self._cur_actions.clear()
        self._cur_masks.clear()
        self._cur_rewards.clear()
        self._warn_if_buffer_large()

    def _warn_if_buffer_large(self) -> None:
        if self._buffer_warned:
            return
        total = sum(ep.nbytes for ep in self._group)
        if total > _BUFFER_WARN_BYTES:
            self._buffer_warned = True
            logger.warning(
                "GRPO group buffer holds %.1f GB across %d/%d episodes "
                "(flat obs dim %d).  Consider a smaller group_size, "
                "store_float16=True, or shorter episodes.",
                total / 1024**3, len(self._group), self.group_size,
                self.obs_dim,
            )

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _group_advantages(self, returns: np.ndarray) -> np.ndarray:
        """Within-group z-score of the episode outcomes.

        Population standard deviation (``ddof=0``) matches the group
        statistic of the GRPO formulation; ``advantage_eps`` keeps a
        zero-variance group (all outcomes equal) at exactly zero
        advantage instead of dividing by zero.
        """
        r = np.asarray(returns, dtype=np.float64)
        return ((r - r.mean()) / (r.std() + self.advantage_eps)).astype(np.float32)

    def update(self, **kwargs: Any) -> dict[str, float]:
        """Run one group update once ``group_size`` episodes are buffered.

        The serial training loop calls this once per episode
        (``runner.py:1489-1492``); it returns ``{}`` — i.e. "nothing to
        log" — for the first ``group_size - 1`` calls and performs the
        clipped-surrogate update on the call that completes the group.
        Surplus episodes (only possible if a call was skipped) stay
        buffered for the next group instead of being silently merged
        into this one.
        """
        _ = kwargs
        self._flush_episode()
        if len(self._group) < self.group_size:
            return {}

        group, self._group = (
            self._group[: self.group_size],
            self._group[self.group_size :],
        )
        self._buffer_warned = False

        # --- Group-relative advantages, broadcast over each episode ---
        returns = np.asarray([ep.ret for ep in group], dtype=np.float64)
        lengths = np.asarray([len(ep) for ep in group], dtype=np.int64)
        adv_per_episode = self._group_advantages(returns)
        advantages = np.repeat(adv_per_episode, lengths)
        self._last_group_returns = returns
        self._last_advantages = advantages

        # Observations stay on the CPU (a group of long episodes is
        # gigabytes wide); minibatches are moved and up-cast on demand.
        obs_all = torch.from_numpy(np.concatenate([ep.obs for ep in group], axis=0))
        actions_t = torch.from_numpy(
            np.concatenate([ep.actions for ep in group])
        ).to(self.device)
        masks_t = torch.from_numpy(
            np.concatenate([ep.masks for ep in group], axis=0)
        ).to(self.device)
        adv_t = torch.from_numpy(advantages).to(self.device)
        n = int(obs_all.shape[0])

        # --- Behaviour log-probs: ONE batched pass BEFORE the epochs ---
        # This is the fix of the legacy agent's identically-1 ratio: the
        # reference distribution must be the policy that *collected* the
        # group, frozen before any gradient step touches the weights.
        old_logp_t = self._behaviour_log_probs(obs_all, actions_t, masks_t)

        idx = np.arange(n)
        losses, pg_losses, entropies, kls, clip_fracs, ratio_devs = (
            [], [], [], [], [], []
        )
        self._epoch_ratio_dev = []
        early_stop = False
        for _epoch in range(self.n_epochs):
            np.random.shuffle(idx)
            epoch_devs: list[float] = []
            epoch_kls: list[float] = []
            for start in range(0, n, self.minibatch_size):
                mb = idx[start : start + self.minibatch_size]
                if len(mb) < 2:
                    continue
                mb_cpu = torch.from_numpy(mb)
                mb_t = mb_cpu.to(self.device)
                mb_obs = obs_all.index_select(0, mb_cpu).to(
                    self.device, torch.float32
                )
                mb_actions = actions_t.index_select(0, mb_t)
                mb_masks = masks_t.index_select(0, mb_t)
                mb_adv = adv_t.index_select(0, mb_t)
                mb_old_logp = old_logp_t.index_select(0, mb_t)

                logits = self.net(mb_obs)
                dist = self._masked_dist(logits, mb_masks)
                new_logp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - mb_old_logp)
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                )
                # No per-minibatch advantage renormalisation: the group
                # z-score IS the advantage, rescaling it inside a
                # minibatch would destroy the between-episode ordering.
                pg_loss = -torch.min(ratio * mb_adv, clipped * mb_adv).mean()
                loss = pg_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
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
                    dev = float((ratio - 1.0).abs().mean().item())
                    clip_frac = float(
                        ((ratio - 1).abs() > self.clip_eps).float().mean().item()
                    )

                losses.append(float(loss.item()))
                pg_losses.append(float(pg_loss.item()))
                entropies.append(float(entropy.item()))
                kls.append(approx_kl)
                epoch_kls.append(approx_kl)
                clip_fracs.append(clip_frac)
                ratio_devs.append(dev)
                epoch_devs.append(dev)

            self._epoch_ratio_dev.append(
                float(np.mean(epoch_devs)) if epoch_devs else 0.0
            )
            if (
                self.target_kl is not None
                and epoch_kls
                and float(np.mean(epoch_kls)) > 1.5 * float(self.target_kl)
            ):
                early_stop = True
                break

        # One schedule step per GROUP update — size ``total_updates`` in
        # groups (episodes / group_size), not episodes.
        self.lr_scheduler.step()

        return {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "pg_loss": float(np.mean(pg_losses)) if pg_losses else float("nan"),
            "entropy": float(np.mean(entropies)) if entropies else float("nan"),
            "approx_kl": float(np.mean(kls)) if kls else float("nan"),
            "clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "ratio_dev": float(np.mean(ratio_devs)) if ratio_devs else 0.0,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "group_size": float(len(group)),
            "group_steps": float(n),
            "group_return_mean": float(returns.mean()),
            "group_return_std": float(returns.std()),
            "advantage_abs_mean": float(np.abs(adv_per_episode).mean()),
            "early_stop": float(bool(early_stop)),
        }

    def _behaviour_log_probs(
        self,
        obs_all: torch.Tensor,
        actions_t: torch.Tensor,
        masks_t: torch.Tensor,
    ) -> torch.Tensor:
        """Log-probs of the taken actions under the pre-update policy."""
        n = int(obs_all.shape[0])
        chunk = max(self.minibatch_size, 256)
        out = torch.empty(n, dtype=torch.float32, device=self.device)
        with self._eval_mode_if(True), torch.no_grad():
            for start in range(0, n, chunk):
                stop = min(start + chunk, n)
                logits = self.net(
                    obs_all[start:stop].to(self.device, torch.float32)
                )
                dist = self._masked_dist(logits, masks_t[start:stop])
                out[start:stop] = dist.log_prob(actions_t[start:stop])
        return out

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def attach_vocab(self, vocab: dict[str, int]) -> None:
        """Adopt the runner's tokenizer vocabulary.

        Called right after construction (``runner.py:393``).  The
        flattener's one-hot tables are keyed off this mapping, so a
        vocabulary that differs from the constructor's forces a rebuild;
        a rebuild that would change the flat width is rejected — the
        network is already sized and silently re-encoding features under
        it is exactly the kind of drift that produced defect D11.
        """
        vocab = {str(k): int(v) for k, v in vocab.items()}
        if vocab != self._vocab:
            rebuilt = self._build_flattener(vocab)
            if int(rebuilt.out_dim) != self.obs_dim:
                msg = (
                    f"attach_vocab: the attached vocabulary changes the flat "
                    f"observation width ({self.obs_dim} -> {rebuilt.out_dim}); "
                    "rebuild the agent with vocab=<this vocabulary> instead."
                )
                raise RuntimeError(msg)
            logger.warning(
                "attach_vocab: rebuilding the flattener from the runner's "
                "vocabulary (%d tokens, was %d) — same flat width.",
                len(vocab), len(self._vocab),
            )
            self.flattener = rebuilt
        self._vocab = vocab
        self.vocab_size = len(vocab)

    @staticmethod
    def peek_vocab(path: str | Path) -> dict[str, int] | None:
        """Return the vocabulary stored alongside a checkpoint, if any."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        v = ckpt.get("vocab") if isinstance(ckpt, dict) else None
        return dict(v) if isinstance(v, dict) else None

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist weights, optimiser, LR schedule and the vocabulary.

        A **partially accumulated group is deliberately not saved**.  The
        buffered episodes are on-policy data whose behaviour policy is
        the pre-checkpoint network; after a restore (possibly with a
        different LR, a different world, or a re-sampled scenario) they
        would enter a group whose z-score mixes outcomes from two
        different policies and two different factories.  Resumed runs
        therefore start a fresh group, losing at most ``group_size - 1``
        episodes of collection.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "agent_type": "grpo_mlp",
            "n_actions": int(self.n_actions),
            "obs_dim": int(self.obs_dim),
            "hidden_sizes": list(self.hidden_sizes),
            "vocab": dict(self._vocab),
            # Rebuild recipe for eval-time loaders (mirrors the
            # SetTransformer's ``improvements`` block).
            "improvements": {
                "group_size": int(self.group_size),
                "gamma": float(self.gamma),
                "clip_eps": float(self.clip_eps),
                "entropy_coef": float(self.entropy_coef),
                "store_float16": bool(self.store_float16),
                "use_action_mask": bool(self.use_action_mask),
                **{k: int(v) for k, v in self._slot_geometry.items()},
            },
        }
        torch.save(ckpt, path)

    def load(self, path: str | Path) -> None:
        """Restore weights, optimiser and LR schedule from a checkpoint.

        Any buffered episodes are dropped (see :meth:`save`): mixing
        pre-load and post-load episodes inside one group would compare
        outcomes collected under two different policies.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        try:
            self.net.load_state_dict(ckpt["net"])
        except RuntimeError as exc:
            msg = (
                f"Cannot load {path}: checkpoint network shape does not match "
                f"this agent (flat obs dim {self.obs_dim}, "
                f"{self.n_actions} actions, hidden {list(self.hidden_sizes)}; "
                f"checkpoint reports obs_dim={ckpt.get('obs_dim')}, "
                f"n_actions={ckpt.get('n_actions')}, "
                f"hidden={ckpt.get('hidden_sizes')}).  Slot caps, hidden "
                "sizes and vocabulary must match — the flat encoding has no "
                "size-agnostic head to fall back on."
            )
            raise RuntimeError(msg) from exc
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
        if isinstance(ckpt.get("vocab"), dict):
            self.attach_vocab(ckpt["vocab"])
        # Drop stale on-policy data: the behaviour policy just changed.
        self._group.clear()
        self._cur_obs.clear()
        self._cur_actions.clear()
        self._cur_masks.clear()
        self._cur_rewards.clear()


__all__ = ["GRPOMLPAgent"]

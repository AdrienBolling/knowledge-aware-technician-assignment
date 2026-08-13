"""A2C over a flattened set observation — the traditional-MLP baseline.

Synchronous advantage actor-critic (Mnih et al. 2016, the synchronous
"A2C" variant) on the plain :class:`MLPActorCritic` fed by
:class:`SetObsFlattener`.  It is the *deliberately conventional*
counterpart to the paper agent: the same observation content, the same
GAE-λ advantages, the same masked categorical head — but a fixed-width
one-hot/scalar encoding instead of learned tokens, and a plain
policy-gradient update instead of PPO's clipped surrogate.

What separates it from :class:`SetTransformerAgent` (the A2C-vs-PPO line)
-------------------------------------------------------------------------
* **No importance-ratio clipping.**  The policy loss is the textbook
  ``-log pi(a|s) * A``; there is no surrogate objective, no ``clip_eps``
  and no KL early-stop.
* **One epoch, one full batch.**  The rollout is consumed exactly once,
  as a single gradient step — no minibatching, no data reuse.  This is
  what makes A2C on-policy in the strict sense and what PPO relaxes.
* **Fixed-width encoder.**  ``SetObsFlattener`` is stateless: categorical
  and boolean slot positions become one-hots over the frozen set vocab,
  continuous positions become symlog scalars.  Fleet-size independence
  therefore comes from *padding + masking* (constant width, padded rows
  zeroed), not from a pointer head — the same checkpoint still acts in a
  2-technician and a 30-technician factory.

What it keeps from the paper agent (so the comparison is about the
algorithm, not the infrastructure): the corrected ``dones[t]`` GAE
semantics, optional semi-MDP ``gamma**dt`` discounting, entropy bonus,
value MSE, gradient-norm clipping, the cosine-with-warmup LR schedule
and its re-arm-on-resume behaviour (:class:`PPOAgentInfraMixin`), the
vectorised-collection contract, and vocab-carrying checkpoints.

Acting is **side-effect free**: ``select_action`` writes no per-stream
cache, so an inline evaluation borrowing stream 0 cannot corrupt a live
training stream.  The log-probabilities and values the update needs are
recomputed from the buffered observations in one batched ``no_grad``
forward at the top of :meth:`update` — for a single-epoch on-policy
update these are exactly the acting-time quantities (same weights, same
masks), only paid for once per round instead of once per decision.

Memory note: buffered observations are stored flattened (~5 kB per
decision at the 30/100 slot caps), so the vectorised loop's
``rollout_steps`` is the knob that bounds the update's batch.  The
episode-based loop updates once per episode and therefore holds a whole
episode of vectors.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.distributions import Categorical

from agents.base import Agent, resolve_device
from agents.networks.continuous_features import ContKind
from agents.networks.mlp_encoder import (
    ENV_SLOT_LAYOUT,
    MACHINE_SLOT_LAYOUT,
    MLPActorCritic,
    SetObsFlattener,
    TECH_SLOT_LAYOUT,
)
from agents.networks.running_stats import RunningMeanStd
from agents.ppo.ppo_transformer import PPOAgentInfraMixin, _cosine_warmup_lr

logger = logging.getLogger(__name__)


# The canonical frozen set vocabulary.  The flattener's one-hot tables
# are derived from it, so it is an *architecture* input (it fixes the
# input width) — resolved here relative to the repo root so the agent is
# constructible without a runner or a tokenizer.
_DEFAULT_VOCAB_PATH = (
    Path(__file__).resolve().parents[3] / "run_configs" / "vocab" / "set_vocab.json"
)

_CATEGORICAL_LAYOUTS: tuple[tuple[str, tuple], ...] = (
    ("tech", TECH_SLOT_LAYOUT),
    ("machine", MACHINE_SLOT_LAYOUT),
    ("env", ENV_SLOT_LAYOUT),
)


class A2CMLPAgent(PPOAgentInfraMixin, Agent):
    """Synchronous advantage actor-critic on the flattened set observation.

    Parameters
    ----------
    n_actions:
        Size of the action space — must equal ``max_techs`` (actions
        index technician *slots*, the mask filters absent ones).
    vocab:
        Frozen set vocabulary: a ``{token: id}`` mapping or a path to a
        vocab JSON.  ``None`` loads the canonical
        ``run_configs/vocab/set_vocab.json``.  It fixes the one-hot
        widths and hence the network's input dimension.
    hidden_sizes:
        Widths of the shared Linear-LayerNorm-ReLU trunk.
    max_techs / max_machines / tech_slot_length / machine_slot_length /
    env_length:
        Set-observation shape — must match the env config (the runner
        injects all but ``machine_slot_length``).
    gamma, time_based_discount, gae_lambda:
        Discounting.  With ``time_based_discount`` ``gamma`` is read per
        sim-TIME-UNIT and each transition is discounted by
        ``gamma ** dt`` (semi-MDP), as in the paper agent.
    entropy_coef, value_coef, normalize_advantages:
        A2C loss weights.  ``loss = pg + value_coef * vf - entropy_coef * H``.
    lr, weight_decay, max_grad_norm, total_updates, warmup_updates,
    lr_min_factor:
        AdamW + the cosine-with-warmup schedule shared with the PPO agents.
    rollout_steps:
        Decisions per worker between updates in the vectorised loop
        (the episode-based loop updates once per episode and ignores it).
    normalize_rewards:
        Divide rewards by the running std of the discounted return
        (SB3-style), as used by the paper agent's training configs.
    vocab_size, sim_time_scale:
        Accepted for runner-injection compatibility.  ``vocab_size`` is
        only cross-checked against ``vocab``; ``sim_time_scale`` is
        unused — the flattener's symlog squash carries no learned scale
        (which is also why it cannot suffer defect D4's rebuilt-scale
        mismatch).
    use_popart, rnn_type:
        Accepted because the shared Hydra launcher writes them into every
        agent's params; a non-default value raises rather than being
        silently ignored — this baseline implements neither.
    """

    def __init__(
        self,
        n_actions: int,
        *,
        # Observation encoding
        vocab: Mapping[str, int] | str | Path | None = None,
        max_techs: int = 30,
        max_machines: int = 100,
        tech_slot_length: int = 16,
        machine_slot_length: int = 12,
        env_length: int = 16,
        # Network
        hidden_sizes: Sequence[int] = (512, 512),
        # A2C hyperparameters
        gamma: float = 0.997,
        time_based_discount: bool = False,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        normalize_advantages: bool = True,
        # Optimiser
        lr: float = 7e-4,
        weight_decay: float = 0.0,
        max_grad_norm: float = 0.5,
        # Rollout / schedule
        rollout_steps: int = 128,
        total_updates: int = 200,
        warmup_updates: int = 10,
        lr_min_factor: float = 0.05,
        # Reward normalisation
        normalize_rewards: bool = False,
        # Accepted for launcher / runner compatibility
        vocab_size: int | None = None,
        sim_time_scale: float = 200_000.0,
        use_popart: bool = False,
        rnn_type: str = "none",
        # Misc
        seed: int | None = None,
        device: str = "auto",
    ) -> None:
        super().__init__(int(n_actions), name="A2CMLP")

        self.device = torch.device(resolve_device(device))
        if seed is not None:
            torch.manual_seed(int(seed))

        if int(n_actions) != int(max_techs):
            msg = (
                f"A2CMLPAgent expects n_actions == max_techs (got "
                f"n_actions={n_actions}, max_techs={max_techs}).  Actions "
                "index technician slots of the padded set observation."
            )
            raise ValueError(msg)
        if bool(use_popart):
            msg = (
                "A2CMLPAgent does not implement PopArt value normalisation "
                "(use normalize_rewards instead) — refusing to ignore the "
                "flag silently."
            )
            raise ValueError(msg)
        if str(rnn_type) != "none":
            msg = (
                f"A2CMLPAgent is feed-forward by design (rnn_type={rnn_type!r} "
                "requested).  Recurrence is a paper-agent ablation, not part "
                "of the traditional baseline."
            )
            raise ValueError(msg)
        _ = sim_time_scale  # stateless symlog encoding — nothing to scale

        # --- observation encoder (no learnable parameters) -------------
        vocab_src: Any = _DEFAULT_VOCAB_PATH if vocab is None else vocab
        if not isinstance(vocab_src, Mapping):
            vocab_src = SetObsFlattener.load_vocab(vocab_src)
        self._vocab: dict[str, int] = {str(k): int(v) for k, v in vocab_src.items()}
        if vocab_size is not None and int(vocab_size) != len(self._vocab):
            logger.warning(
                "vocab_size=%d disagrees with the %d-token vocab the "
                "flattener was built from; the observation encoding follows "
                "the vocab, not the size.  Attach the runner's tokenizer "
                "vocab (attach_vocab) to have the mismatch checked exactly.",
                int(vocab_size), len(self._vocab),
            )
        # Kept so the one-hot tables can be REBUILT (identical geometry)
        # if a vocabulary with the same layout but different token ids is
        # attached later — see :meth:`_adopt_vocab`.
        self._slot_geometry = {
            "max_techs": int(max_techs),
            "max_machines": int(max_machines),
            "tech_slot_len": int(tech_slot_length),
            "machine_slot_len": int(machine_slot_length),
            "env_len": int(env_length),
        }
        self.flattener = self._build_flattener(self._vocab)
        self._cat_signature = self._categorical_signature(self._vocab)

        # --- network + optimiser ---------------------------------------
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.net = MLPActorCritic(
            self.flattener.out_dim, int(n_actions), self.hidden_sizes
        ).to(self.device)

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

        # --- A2C knobs --------------------------------------------------
        self.gamma = float(gamma)
        self.time_based_discount = bool(time_based_discount)
        self.gae_lambda = float(gae_lambda)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.normalize_advantages = bool(normalize_advantages)
        self.max_grad_norm = float(max_grad_norm)
        self.rollout_steps = int(rollout_steps)
        self.normalize_rewards = bool(normalize_rewards)
        self.max_techs = int(max_techs)
        self.max_machines = int(max_machines)
        self.tech_slot_length = int(tech_slot_length)
        self.machine_slot_length = int(machine_slot_length)
        self.env_length = int(env_length)
        self.vocab_size = len(self._vocab)

        # Return-normalisation running stats (off by default)
        self._return_rms = RunningMeanStd()
        self._return_running: dict[int, float] = defaultdict(float)

        # --- rollout buffers, one stream per (vectorised) env -----------
        # Observations are stored already flattened; actions/masks are the
        # acting-time quantities, log-probs and values are NOT stored --
        # they are recomputed in update()'s single batched forward.
        self._streams: dict[int, dict[str, list]] = defaultdict(
            lambda: {
                "obs": [], "action": [], "reward": [], "done": [],
                "mask": [], "dt": [],
            }
        )
        # Flattened newest next_obs per stream — the truncation bootstrap.
        self._last: dict[int, np.ndarray] = {}
        # Sim-time of each stream's newest observation (semi-MDP dt
        # bookkeeping).  None = fresh episode: the first transition falls
        # back to dt=0, a one-transition approximation.
        self._last_sim_time: dict[int, float | None] = {}

    # ------------------------------------------------------------------
    def _build_flattener(self, vocab: Mapping[str, int]) -> SetObsFlattener:
        return SetObsFlattener(vocab, **self._slot_geometry).to(self.device)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    # ------------------------------------------------------------------
    # Observation plumbing
    # ------------------------------------------------------------------
    def _flatten(self, obs: dict[str, Any]) -> torch.Tensor:
        """Flatten one observation to a ``(out_dim,)`` device tensor."""
        return self.flattener(obs)

    def _flatten_np(self, obs: dict[str, Any]) -> np.ndarray:
        return self.flattener(obs).detach().cpu().numpy().astype(np.float32)

    def _action_mask(self, obs: dict[str, Any]) -> np.ndarray:
        """Boolean ``(n_actions,)`` mask; never an all-ones fallback.

        Deterministic in ``obs``, which is what lets acting stay
        side-effect free: :meth:`observe_transition` recomputes the very
        mask :meth:`select_action` used instead of reading a cache.
        """
        return self.flattener.extract_action_mask(obs)

    def _sample(
        self, logits: torch.Tensor, mask: torch.Tensor, deterministic: bool
    ) -> np.ndarray:
        masked = logits.float().masked_fill(~mask, float("-inf"))
        dist = Categorical(logits=masked)
        acts = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return acts.detach().cpu().numpy()

    # ------------------------------------------------------------------
    def select_action(
        self, obs: dict[str, Any], *, deterministic: bool = False, env_id: int = 0
    ) -> int:
        # No per-stream bookkeeping here by design (see module docstring):
        # env_id is accepted for signature parity with the PPO agents.
        _ = env_id
        x = self._flatten(obs).unsqueeze(0)
        mask = torch.from_numpy(self._action_mask(obs)).unsqueeze(0).to(self.device)
        with self._eval_mode_if(deterministic), torch.no_grad():
            logits, _value = self.net(x)
        return int(self._sample(logits, mask, deterministic)[0])

    def select_actions(
        self,
        obs_list: list[dict[str, Any]],
        *,
        deterministic: bool = False,
        env_ids: list[int] | None = None,
    ) -> list[int]:
        """Batched :meth:`select_action` for vectorised collection.

        One network forward for ``len(obs_list)`` observations.  Like the
        single-env path it caches nothing, so ``env_ids`` is only a
        signature-compatibility argument.
        """
        _ = env_ids
        x = torch.stack([self._flatten(o) for o in obs_list], dim=0)
        mask = torch.from_numpy(
            np.stack([self._action_mask(o) for o in obs_list], axis=0)
        ).to(self.device)
        with self._eval_mode_if(deterministic), torch.no_grad():
            logits, _values = self.net(x)
        return [int(a) for a in self._sample(logits, mask, deterministic)]

    # ------------------------------------------------------------------
    # Per-stream episode state
    # ------------------------------------------------------------------
    def reset_stream(self, env_id: int = 0) -> None:
        """Reset per-episode state of a stream (running return, dt anchor).

        The rollout buffer is deliberately untouched: in the vectorised
        loop this fires on autoreset steps, mid-round, and the buffered
        transitions still have to reach the round-boundary update.
        """
        self._return_running[env_id] = 0.0
        self._last_sim_time[env_id] = None

    def on_episode_start(self) -> None:
        self.reset_stream(0)

    def snapshot_stream_state(self) -> tuple:
        """Shallow snapshot of the per-stream episode state.

        Pairs with :meth:`restore_stream_state` so an inline evaluation —
        which borrows stream 0 via ``on_episode_start`` — cannot clobber a
        live training episode's bootstrap anchor or dt stamp.
        """
        return (
            dict(self._last),
            dict(self._return_running),
            dict(self._last_sim_time),
        )

    def restore_stream_state(self, snap: tuple) -> None:
        """Restore the state captured by :meth:`snapshot_stream_state`."""
        last, running, sim_time = snap
        for live, saved in (
            (self._last, last),
            (self._return_running, running),
            (self._last_sim_time, sim_time),
        ):
            live.clear()
            live.update(saved)

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
        env_id: int = 0,
    ) -> None:
        done = bool(terminated or truncated)
        # Semi-MDP bookkeeping: dt = sim time elapsed across this
        # transition.  ``info["sim_time"]`` stamps next_obs; the previous
        # transition's stamp is the time of obs.
        sim_time = info.get("sim_time") if isinstance(info, dict) else None
        prev_time = self._last_sim_time.get(env_id)
        if sim_time is not None and prev_time is not None:
            dt = max(0.0, float(sim_time) - prev_time)
        else:
            dt = 0.0
        self._last_sim_time[env_id] = (
            None if done else (float(sim_time) if sim_time is not None else None)
        )

        r_buf = float(reward)
        if self.normalize_rewards:
            self._return_running[env_id] = (
                self._return_running[env_id] * self.gamma + r_buf
            )
            self._return_rms.update(self._return_running[env_id])
            r_buf = r_buf / max(self._return_rms.std, 1e-8)
            if done:
                self._return_running[env_id] = 0.0

        stream = self._streams[env_id]
        stream["obs"].append(self._flatten_np(obs))
        stream["action"].append(int(action))
        stream["reward"].append(r_buf)
        stream["done"].append(done)
        stream["mask"].append(self._action_mask(obs))
        stream["dt"].append(dt)
        # Flatten eagerly: vectorised workers hand out views into buffers
        # they overwrite on the next step, so keeping the raw dict would
        # bootstrap from whatever lands there later.
        self._last[env_id] = self._flatten_np(next_obs)

    # ------------------------------------------------------------------
    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
        dts: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generalised Advantage Estimation (Schulman et al. 2016).

        ``dones[t]`` flags that transition *t* ended its episode, so it
        masks transition t's OWN bootstrap and severs its λ-chain — the
        semantics fixed in the PPO agents on 2026-07-20 / 2026-08-04
        (defect D6).  A rollout buffer holding interior episode
        boundaries must produce, for each episode, exactly the advantages
        that episode would have produced alone.

        With ``time_based_discount`` the per-transition discount is
        ``gamma ** dt`` (gamma per sim-time-unit); λ stays per-decision —
        it trades bias for variance over *decisions*, it is not a time
        preference.
        """
        n = len(rewards)
        if self.time_based_discount and dts is not None:
            gammas = np.power(self.gamma, dts.astype(np.float64))
        else:
            gammas = np.full(n, self.gamma, dtype=np.float64)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        next_value = float(last_value)
        for t in reversed(range(n)):
            non_terminal = 0.0 if dones[t] else 1.0
            delta = (
                rewards[t] + gammas[t] * next_value * non_terminal - values[t]
            )
            gae = delta + gammas[t] * self.gae_lambda * non_terminal * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    def update(self, **kwargs: Any) -> dict[str, float]:
        """One A2C step: single epoch, single full-batch gradient step.

        No ratio, no clipping, no minibatch shuffling — the whole rollout
        contributes to exactly one update, then the buffers are dropped.
        The reported ``approx_kl`` measures how far the step moved the
        policy on its own batch (pre → post), not a within-update ratio:
        this algorithm has none.
        """
        _ = kwargs
        active = {i: s for i, s in self._streams.items() if s["obs"]}
        if not active:
            return {}
        order = sorted(active)

        # --- one batched no_grad forward: buffered obs + bootstrap obs --
        obs_blocks = [np.stack(active[i]["obs"], axis=0) for i in order]
        lengths = [b.shape[0] for b in obs_blocks]
        boot_slot: dict[int, int] = {}
        boot_rows: list[np.ndarray] = []
        for env_id in order:
            # A stream cut mid-episode bootstraps from its newest obs; a
            # stream ending on a terminal transition has no successor.
            if not active[env_id]["done"][-1] and env_id in self._last:
                boot_slot[env_id] = len(boot_rows)
                boot_rows.append(self._last[env_id])
        blocks = obs_blocks + ([np.stack(boot_rows, axis=0)] if boot_rows else [])
        x = torch.from_numpy(np.concatenate(blocks, axis=0)).to(self.device)
        n = int(sum(lengths))

        actions = np.concatenate(
            [np.asarray(active[i]["action"], dtype=np.int64) for i in order]
        )
        masks = np.concatenate(
            [np.stack(active[i]["mask"], axis=0).astype(bool) for i in order], axis=0
        )
        actions_t = torch.from_numpy(actions).to(self.device)
        masks_t = torch.from_numpy(masks).to(self.device)

        with torch.no_grad():
            logits, values_t = self.net(x)
            old_log_probs_t = Categorical(
                logits=logits[:n].float().masked_fill(~masks_t, float("-inf"))
            ).log_prob(actions_t)
        values_all = values_t.float().detach().cpu().numpy()

        # --- per-stream GAE: each stream is one contiguous trajectory ---
        adv_parts, ret_parts = [], []
        offset = 0
        for k, env_id in enumerate(order):
            s = active[env_id]
            m = lengths[k]
            last_value = (
                float(values_all[n + boot_slot[env_id]])
                if env_id in boot_slot
                else 0.0
            )
            adv, ret = self._compute_gae(
                np.asarray(s["reward"], dtype=np.float32),
                values_all[offset : offset + m].astype(np.float32),
                np.asarray(s["done"], dtype=bool),
                last_value,
                np.asarray(s["dt"], dtype=np.float64),
            )
            adv_parts.append(adv)
            ret_parts.append(ret)
            offset += m
        advantages = np.concatenate(adv_parts)
        returns = np.concatenate(ret_parts)

        advantages_t = torch.from_numpy(advantages).to(self.device)
        returns_t = torch.from_numpy(returns.astype(np.float32)).to(self.device)
        if self.normalize_advantages and advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (
                advantages_t.std() + 1e-8
            )

        # --- the single gradient step ----------------------------------
        logits, value = self.net(x[:n])
        dist = Categorical(
            logits=logits.float().masked_fill(~masks_t, float("-inf"))
        )
        new_log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()
        # Vanilla policy gradient — the PPO surrogate's ratio is exactly
        # what this baseline drops.
        pg_loss = -(new_log_probs * advantages_t).mean()
        vf_loss = 0.5 * (value - returns_t).pow(2).mean()
        loss = pg_loss + self.value_coef * vf_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.net.parameters(), self.max_grad_norm
        )
        self.optimizer.step()
        self.lr_scheduler.step()

        # Policy movement caused by THIS step.  A2C has no within-update
        # drift to measure -- one epoch, one gradient step, so the acting
        # log-probs and the loss's log-probs come from the same weights
        # and their ratio is identically 1.  The one-step analogue of
        # PPO's approx_kl is therefore the PRE->POST comparison: a third
        # forward on the same batch, after the optimiser moved the
        # weights.  Same no_grad path as ``old_log_probs_t`` so the only
        # difference between the two is the step itself.
        with torch.no_grad():
            post_logits, _post_value = self.net(x[:n])
            post_log_probs = Categorical(
                logits=post_logits.float().masked_fill(~masks_t, float("-inf"))
            ).log_prob(actions_t)
            log_ratio = post_log_probs - old_log_probs_t
            approx_kl = float(
                ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
            )
        # How much of the return variance the critic explains — the A2C
        # diagnostic that replaces PPO's clip fraction.
        ret_var = float(np.var(returns))
        explained_var = (
            float(1.0 - np.var(returns - values_all[:n]) / ret_var)
            if ret_var > 1e-12
            else float("nan")
        )

        # Drop the rollout (strictly on-policy).  The dt anchors survive:
        # they are acting-time continuity, not rollout data.
        self._streams.clear()
        self._last.clear()

        return {
            "loss": float(loss.item()),
            "pg_loss": float(pg_loss.item()),
            "vf_loss": float(vf_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": approx_kl,
            "explained_variance": explained_var,
            "grad_norm": float(grad_norm),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "rollout_size": float(n),
        }

    # ------------------------------------------------------------------
    # Vocabulary guards
    # ------------------------------------------------------------------
    @staticmethod
    def _categorical_signature(
        vocab: Mapping[str, int]
    ) -> dict[str, tuple[str, ...]]:
        """Per-position one-hot bin order implied by ``vocab``.

        The flattener's input width and column semantics are a pure
        function of these lists (mlp_encoder ``_build_stream``), so two
        vocabs with the same signature are interchangeable for a trained
        checkpoint and any other pair is not — vocab drift re-indexes
        shared keys rather than appending (the 2026-08-05 vocab-guard
        lesson).
        """
        sig: dict[str, tuple[str, ...]] = {}
        for stream, layout in _CATEGORICAL_LAYOUTS:
            for sp in layout:
                if sp.kind != ContKind.CATEGORICAL:
                    continue
                prefix = f"{sp.key}="
                sig[f"{stream}/{sp.key}"] = tuple(
                    tok
                    for tok, _tid in sorted(
                        (
                            (t, i)
                            for t, i in vocab.items()
                            if t.startswith(prefix)
                        ),
                        key=lambda kv: kv[1],
                    )
                )
        return sig

    def _assert_vocab_compatible(self, vocab: Mapping[str, int], src: str) -> None:
        sig = self._categorical_signature(vocab)
        bad = [k for k in sorted(set(sig) | set(self._cat_signature))
               if sig.get(k) != self._cat_signature.get(k)]
        if not bad:
            return
        msg = (
            f"{src} vocabulary is incompatible with this agent's flattener: "
            f"the one-hot layout differs at {bad[:5]}"
            f"{' …' if len(bad) > 5 else ''}.  The input width and column "
            "meanings are baked into the trained weights — rebuild the agent "
            "with the matching vocab instead of remapping."
        )
        raise RuntimeError(msg)

    def _adopt_vocab(self, vocab: Mapping[str, int], src: str) -> None:
        """Adopt ``vocab``, rebuilding the one-hot tables if the ids moved.

        Two distinct failure modes, two distinct answers:

        * a different one-hot **layout** (bin order or membership) is
          unfixable — the trained weights read those columns positionally,
          so :meth:`_assert_vocab_compatible` refuses;
        * the same layout at different **absolute ids** is fixable and
          must be fixed: the flattener's lookup tables are indexed by
          absolute token id (``mlp_encoder._build_stream``), so keeping
          the constructor's tables would send every categorical straight
          to the OTHER bin — silent feature erasure, the D11 failure
          mode.  Rebuild the tables from the incoming mapping instead.
        """
        vocab = {str(k): int(v) for k, v in vocab.items()}
        self._assert_vocab_compatible(vocab, src)
        if vocab != self._vocab:
            rebuilt = self._build_flattener(vocab)
            if int(rebuilt.out_dim) != int(self.flattener.out_dim):
                msg = (
                    f"{src} vocabulary changes the flat observation width "
                    f"({self.flattener.out_dim} -> {rebuilt.out_dim}); the "
                    "network is already sized, so rebuild the agent with "
                    "vocab=<this vocabulary> instead of remapping."
                )
                raise RuntimeError(msg)
            logger.warning(
                "%s vocabulary re-indexes the token ids (%d tokens, was %d) "
                "— rebuilding the flattener's one-hot tables; the layout and "
                "the flat width are unchanged.",
                src, len(vocab), len(self._vocab),
            )
            self.flattener = rebuilt
        self._vocab = vocab
        self.vocab_size = len(self._vocab)
        self._cat_signature = self._categorical_signature(self._vocab)

    def attach_vocab(self, vocab: dict[str, int]) -> None:
        """Attach the runner's tokenizer vocabulary (saved with the weights).

        Called right after construction.  The flattener's one-hot tables
        were already built from ``vocab=`` (default: the canonical
        artefact), so an attached mapping that would change their *layout*
        is a configuration error; one that merely re-indexes the tokens
        forces a rebuild rather than being absorbed silently.
        """
        self._adopt_vocab(vocab, "Attached")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Persist weights, optimiser, schedule, reward stats and vocab.

        The vocab travels with the weights for the same reason as in the
        set-transformer agent: it is the source of truth for the input
        encoding, here for the one-hot column layout.
        """
        torch.save(
            {
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "return_rms": self._return_rms.state_dict(),
                "vocab": dict(self._vocab),
                "max_techs": self.max_techs,
                "max_machines": self.max_machines,
                "vocab_size": self.vocab_size,
                # Everything an eval-time loader needs to rebuild an
                # identically-shaped agent from the checkpoint alone.
                "improvements": {
                    "hidden_sizes": list(self.hidden_sizes),
                    "tech_slot_length": self.tech_slot_length,
                    "machine_slot_length": self.machine_slot_length,
                    "env_length": self.env_length,
                    "in_dim": int(self.flattener.out_dim),
                    "time_based_discount": self.time_based_discount,
                    "normalize_rewards": self.normalize_rewards,
                },
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Restore agent state from a checkpoint.

        The observation encoding is fixed by the vocab and the slot caps,
        so a mismatch is a hard architecture error: there is no partial
        load that would keep the columns meaning what they meant during
        training.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        ckpt_vocab = ckpt.get("vocab")
        if isinstance(ckpt_vocab, dict):
            self._assert_vocab_compatible(ckpt_vocab, "Checkpoint")
        try:
            self.net.load_state_dict(ckpt["net"])
        except RuntimeError as exc:
            imp = ckpt.get("improvements") or {}
            msg = (
                f"Checkpoint architecture does not match this agent "
                f"(checkpoint: in_dim={imp.get('in_dim')}, "
                f"hidden_sizes={imp.get('hidden_sizes')}, "
                f"max_techs={ckpt.get('max_techs')}; live: "
                f"in_dim={self.flattener.out_dim}, "
                f"hidden_sizes={list(self.hidden_sizes)}, "
                f"max_techs={self.max_techs}).  Rebuild the agent from the "
                f"checkpoint's own configuration."
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
                # A resumed run whose schedule already ended would spend
                # its whole budget at the floor (defect D1).
                self._rearm_lr_schedule_if_exhausted()
        if "return_rms" in ckpt:
            try:
                self._return_rms.load_state_dict(ckpt["return_rms"])
            except (KeyError, TypeError):
                pass
        if isinstance(ckpt_vocab, dict):
            # Same adoption path as attach_vocab: the checkpoint's ids are
            # the ones its weights were trained under, so if they moved the
            # one-hot tables have to follow them.
            self._adopt_vocab(ckpt_vocab, "Checkpoint")

    @staticmethod
    def peek_vocab(path: str | Path) -> dict[str, int] | None:
        """Return the vocabulary stored alongside a checkpoint, if any."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        v = ckpt.get("vocab") if isinstance(ckpt, dict) else None
        return dict(v) if isinstance(v, dict) else None


__all__ = ["A2CMLPAgent"]

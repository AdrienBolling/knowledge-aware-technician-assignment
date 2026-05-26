"""PPO with a pretrained, optionally-frozen encoder.

Identical to :class:`agents.ppo.ppo_transformer.PPOTransformerAgent` in
algorithm and optimiser surface — the only difference is that the
encoder backbone is **loaded from an MTM (masked-token-modelling)
checkpoint** instead of being randomly initialised, and its parameters
can be frozen so only the MLP actor-critic heads are trained.

This makes the split between *state representation* and *control*
explicit: the encoder produces a continuous summary of the token
observation, and PPO only ever sees that summary.  Useful as
(a) a representation-quality probe (training the heads from a frozen
encoder isolates "how informative is the latent?") and (b) a faster
specialisation regime when good representations are already known.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from agents.ppo.ppo_transformer import PPOTransformerAgent
from agents.representation.mtm import MTMTrainer


class PPOLatentAgent(PPOTransformerAgent):
    """PPO with a frozen-encoder + MLP-only actor-critic.

    Parameters
    ----------
    n_actions:
        Number of technicians.
    vocab_size:
        Vocabulary size of the tokenizer.  Must match the vocab the
        encoder was pretrained with — a mismatch is caught and raised
        at load time.
    encoder_path:
        Path to a ``MTMTrainer.save(...)`` checkpoint.  Required.
    freeze_encoder:
        When ``True`` (default), encoder parameters do not receive
        gradients — only the policy / value heads are optimised.  Set
        ``False`` for a "warm-start" regime where the encoder
        fine-tunes alongside the heads.
    encoder_lr_scale:
        When ``freeze_encoder=False``, this multiplier scales the
        encoder's learning rate relative to the heads.  A value below 1
        (e.g. 0.1) is a common choice to slow encoder drift.
    **ppo_kwargs:
        Any keyword forwarded to :class:`PPOTransformerAgent` — PPO
        hyperparameters, head sizes, etc.  Encoder architecture
        kwargs (``d_model``, ``n_layers``, ``max_seq_len``) are
        **overridden** from the saved encoder so they always match the
        checkpoint exactly.
    """

    def __init__(
        self,
        n_actions: int,
        vocab_size: int,
        *,
        encoder_path: str | Path,
        freeze_encoder: bool = True,
        encoder_lr_scale: float = 0.1,
        **ppo_kwargs: Any,
    ) -> None:
        encoder_path = Path(encoder_path)
        if not encoder_path.exists():
            msg = f"Encoder checkpoint not found at {encoder_path}"
            raise FileNotFoundError(msg)

        # Peek at the saved metadata so we build the parent with a
        # geometry that matches the pretrained encoder exactly.
        ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
        ckpt_vocab = int(ckpt["vocab_size"])
        if ckpt_vocab != int(vocab_size):
            msg = (
                f"vocab_size mismatch: encoder checkpoint was trained "
                f"with {ckpt_vocab} tokens but PPOLatentAgent received "
                f"vocab_size={vocab_size}.  Rebuild the tokenizer with "
                f"the same machine/component pool used for pretraining."
            )
            raise ValueError(msg)

        enc_kw = ckpt["encoder_kwargs"]
        ppo_kwargs.update(
            d_model=int(enc_kw["d_model"]),
            n_layers=int(enc_kw["n_layers"]),
            n_heads=int(enc_kw.get("n_heads", 6)),
            max_seq_len=int(enc_kw["max_seq_len"]),
        )

        # Hybrid encoders are detected by the presence of their
        # continuous-feature submodules in the checkpoint state-dict.
        # Forcing this here means the user does not need to set
        # ``hybrid_obs`` manually — the agent always matches the
        # geometry of whichever encoder it loads.
        ckpt_keys: set[str] = set(ckpt.get("model", {}).keys())
        ckpt_hybrid = any(
            k.startswith("encoder.ratio_ple.")
            or k.startswith("encoder.count_ple.")
            or k.startswith("encoder.time2vec.")
            or k.startswith("encoder.fourier.")
            for k in ckpt_keys
        )
        if ckpt_hybrid:
            ppo_kwargs["hybrid_obs"] = True
            # Carry through the sim_time_scale used at pretraining if the
            # extras saved it; otherwise fall back to the user override
            # or the parent default.
            extras = ckpt.get("extras") or {}
            if "sim_time_scale" in extras and "sim_time_scale" not in ppo_kwargs:
                ppo_kwargs["sim_time_scale"] = float(extras["sim_time_scale"])

        # Capture the scheduler hyperparameters *before* super() consumes
        # them so we can rebuild a matching LambdaLR after rewiring the
        # parameter groups.
        self._warmup_updates = int(ppo_kwargs.get("warmup_updates", 10))
        self._total_updates = int(ppo_kwargs.get("total_updates", 200))

        super().__init__(n_actions=n_actions, vocab_size=vocab_size, **ppo_kwargs)
        self.name = "PPOLatent"

        # Swap the freshly-initialised encoder weights for the
        # pretrained ones.  The parent's ``ModernTransformerEncoder`` was
        # constructed with matching geometry above so this is a clean
        # state-dict load.
        encoder = MTMTrainer.load_encoder(encoder_path, device=self.device)
        self.net.encoder.load_state_dict(encoder.state_dict())

        # Freeze / unfreeze + rebuild the optimiser so the parameter
        # groups respect the chosen schedule.
        self.freeze_encoder = bool(freeze_encoder)
        self.encoder_lr_scale = float(encoder_lr_scale)
        self._reconfigure_optimizer()

    # ------------------------------------------------------------------

    def _reconfigure_optimizer(self) -> None:
        """Rebuild the optimiser with two param groups (encoder + heads)."""
        base_lr = self.optimizer.param_groups[0]["lr"]
        weight_decay = self.optimizer.param_groups[0].get("weight_decay", 0.0)
        eps = self.optimizer.param_groups[0].get("eps", 1e-5)
        betas = self.optimizer.param_groups[0].get("betas", (0.9, 0.999))

        encoder_params = list(self.net.encoder.parameters())
        head_params = [
            p
            for n, p in self.net.named_parameters()
            if not n.startswith("encoder.")
        ]
        if self.freeze_encoder:
            for p in encoder_params:
                p.requires_grad_(False)
            # Critically, also disable train-mode side effects (dropout
            # etc.) on the encoder.  ``requires_grad_(False)`` only
            # blocks parameter updates --- it leaves dropout active,
            # which would inject noise both during rollout collection
            # (breaking determinism of greedy ``select_action``) and
            # during PPO updates (gradient signal into the heads
            # depends on stochastic encoder outputs).
            self.net.encoder.eval()
            # Pin in eval: rebind ``train`` so a future call (e.g. the
            # nightly habit of ``self.net.train()`` at the top of the
            # update loop) cannot accidentally flip the encoder back.
            _encoder = self.net.encoder
            _encoder.train = lambda mode=True, _e=_encoder: _e  # type: ignore[method-assign]
            param_groups: list[dict[str, Any]] = [
                {"params": head_params, "lr": base_lr}
            ]
        else:
            param_groups = [
                {"params": head_params, "lr": base_lr},
                {"params": encoder_params, "lr": base_lr * self.encoder_lr_scale},
            ]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=weight_decay,
            eps=eps,
            betas=betas,
        )

        # Replace the scheduler so the new optimiser has a matching
        # LambdaLR.  Reuses the same warmup/cosine recipe.
        from agents.ppo.ppo_transformer import _cosine_warmup_lr

        warmup = getattr(self, "_warmup_updates", 10)
        total = getattr(self, "_total_updates", 200)
        # One lr_lambda per param group so each group gets the same
        # warmup+cosine shape on top of its own initial lr.
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=[
                (lambda step, w=warmup, t=total: _cosine_warmup_lr(step, w, t))
                for _ in self.optimizer.param_groups
            ],
        )

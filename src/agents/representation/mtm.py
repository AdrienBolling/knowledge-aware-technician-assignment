"""Masked-token-modelling head + trainer.

Wraps a :class:`agents.networks.modern_transformer.ModernTransformerEncoder`
with a linear "predict the masked token" head and a BERT-style training
loop.  Two artefacts come out of this:

* a **pretrained encoder** (saved as ``encoder.pt``) that can be plugged
  into ``PPOLatentAgent`` as a frozen state encoder;
* a **per-step accuracy curve** that doubles as a *token learnability*
  probe — a stuck MLM accuracy is direct evidence the observation
  alphabet is too noisy or too aliased to support useful representations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from agents.networks.modern_transformer import ModernTransformerEncoder

# Special-token IDs come from StateTokenizer.  We deliberately do NOT
# introduce a new ``<MASK>`` entry: a pre-built vocab in this project
# never emits ``UNK`` in practice (every observable token is registered
# at construction time), so we reuse ``UNK_ID`` as the BERT mask token.
# This avoids re-numbering every vocab entry just to make room for MASK.
from kata.core.tokenizer import BOS_ID, EOS_ID, PAD_ID, UNK_ID  # noqa: F401

MASK_ID = UNK_ID
NUM_RESERVED = 4  # PAD, UNK, BOS, EOS — masking never targets these.


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MaskedTokenModel(nn.Module):
    """Encoder + linear MLM head.

    The MLM head is tied to the input *categorical* embedding weights
    (Press & Wolf 2017) so the parameter count is unchanged and the
    head stays consistent with the encoder's token representation.

    Supports two encoder backbones depending on ``hybrid``:

    * ``hybrid=False`` (default) — :class:`ModernTransformerEncoder`,
      input is ``token_ids`` only.  Legacy bucket-token pretraining.
    * ``hybrid=True`` — :class:`HybridTokenEncoder`, input is
      ``(token_ids, cont_values, cont_kinds)``.  Continuous channels
      feed PLE / Time2Vec / Fourier and are *always visible* to the
      encoder; the MLM objective masks only categorical positions
      (since cross-entropy on continuous values doesn't fit a
      discrete-token head).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 6,
        d_ff: int | None = None,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        hybrid: bool = False,
        sim_time_scale: float = 200_000.0,
    ) -> None:
        super().__init__()
        self.hybrid = bool(hybrid)
        if self.hybrid:
            from agents.networks.hybrid_encoder import HybridTokenEncoder

            self.encoder = HybridTokenEncoder(
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                dropout=dropout,
                pad_token_id=PAD_ID,
                sim_time_scale=sim_time_scale,
            )
        else:
            self.encoder = ModernTransformerEncoder(
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                dropout=dropout,
                pad_token_id=PAD_ID,
            )
        # Weight-tied head: logits = h @ E.T
        self.head_bias = nn.Parameter(torch.zeros(vocab_size))

    @property
    def output_dim(self) -> int:
        return self.encoder.output_dim

    def forward(
        self,
        token_ids: torch.Tensor,
        cont_values: torch.Tensor | None = None,
        cont_kinds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, cls)``.

        ``cont_values`` / ``cont_kinds`` are required when the underlying
        encoder is :class:`HybridTokenEncoder` and ignored otherwise.
        """
        if self.hybrid:
            assert cont_values is not None and cont_kinds is not None, (
                "Hybrid MaskedTokenModel requires cont_values and cont_kinds"
            )
            cls, tokens = self.encoder.encode(token_ids, cont_values, cont_kinds)
        else:
            cls, tokens = self.encoder.encode(token_ids)
        # Tied embedding: logits = tokens @ E.T + b
        logits = tokens @ self.encoder.token_embedding.weight.T + self.head_bias
        return logits, cls


# ---------------------------------------------------------------------------
# Masking utility
# ---------------------------------------------------------------------------


def mlm_collate(
    batch_seqs: list[np.ndarray] | np.ndarray,
    *,
    vocab_size: int,
    mask_prob: float = 0.15,
    random_prob: float = 0.10,
    keep_prob: float = 0.10,
    rng: np.random.Generator | None = None,
    cont_kinds: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply BERT-style masking to a batch of token sequences.

    Each non-special token is independently selected for prediction
    with probability ``mask_prob``.  Selected positions are replaced
    with ``MASK_ID`` (``1 - random_prob - keep_prob`` of the time),
    a random token (``random_prob``), or left unchanged (``keep_prob``)
    — the classic BERT 80/10/10 split when defaults are used.

    When ``cont_kinds`` is provided (hybrid mode), positions where
    ``cont_kinds != 0`` (i.e. continuous-value positions carrying the
    ``<NUM>`` placeholder) are *excluded* from masking — categorical
    cross-entropy on those positions would be uninformative since the
    real signal lives on the parallel ``cont_values`` channel.
    """
    rng = rng or np.random.default_rng()
    batch_seqs = np.asarray(batch_seqs, dtype=np.int64)
    inputs = batch_seqs.copy()
    targets = np.full_like(batch_seqs, -100)

    candidate = batch_seqs >= NUM_RESERVED
    if cont_kinds is not None:
        candidate = candidate & (np.asarray(cont_kinds) == 0)
    pick = candidate & (rng.random(batch_seqs.shape) < mask_prob)
    targets[pick] = batch_seqs[pick]

    rand_draw = rng.random(batch_seqs.shape)
    p_mask = 1.0 - random_prob - keep_prob
    do_mask = pick & (rand_draw < p_mask)
    do_random = pick & (rand_draw >= p_mask) & (rand_draw < p_mask + random_prob)

    inputs[do_mask] = MASK_ID
    if do_random.any():
        rand_ids = rng.integers(
            NUM_RESERVED, vocab_size, size=int(do_random.sum()), dtype=np.int64
        )
        inputs[do_random] = rand_ids

    return (
        torch.from_numpy(inputs),
        torch.from_numpy(targets),
        torch.from_numpy(pick),
    )


class _TokenSequenceDataset(Dataset):
    """Sequence dataset.

    Holds either a single ``(N, S)`` int64 token-id array (legacy) or
    a dict ``{"token_ids": ..., "cont_values": ..., "cont_kinds": ...}``
    of ``(N, S)`` arrays (hybrid).  ``__getitem__`` returns either an
    ndarray or a dict to mirror.
    """

    def __init__(
        self, sequences: np.ndarray | dict[str, np.ndarray]
    ) -> None:
        if isinstance(sequences, dict):
            self.hybrid = True
            self.token_ids = np.asarray(sequences["token_ids"], dtype=np.int64)
            self.cont_values = np.asarray(
                sequences["cont_values"], dtype=np.float32
            )
            self.cont_kinds = np.asarray(sequences["cont_kinds"], dtype=np.int8)
        else:
            self.hybrid = False
            self.token_ids = np.asarray(sequences, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, idx: int):
        if self.hybrid:
            return {
                "token_ids": self.token_ids[idx],
                "cont_values": self.cont_values[idx],
                "cont_kinds": self.cont_kinds[idx],
            }
        return self.token_ids[idx]


def _hybrid_collate(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """DataLoader collate fn for hybrid sequences — stack each channel."""
    return {k: np.stack([b[k] for b in batch]) for k in batch[0]}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class MTMTrainConfig:
    """Hyper-parameters for the MLM pretraining loop."""

    n_epochs: int = 20
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    # In hybrid mode the effective per-step supervision rate is
    # ``mask_prob * (fraction_of_categorical_positions)``: continuous
    # positions (cont_kinds != 0) are excluded from masking because
    # predicting "<NUM>" everywhere would be uninformative.  If a tech-
    # aware obs is ~40% continuous, set ``mask_prob`` higher than you
    # would in the categorical-only regime to keep the realised
    # supervision rate constant.
    mask_prob: float = 0.15
    warmup_steps: int = 200
    log_every: int = 50
    device: str = "auto"
    seed: int = 0


class MTMTrainer:
    """BERT-style MLM training loop for a :class:`MaskedTokenModel`."""

    def __init__(
        self,
        model: MaskedTokenModel,
        vocab_size: int,
        cfg: MTMTrainConfig | None = None,
    ) -> None:
        from agents.base import resolve_device

        self.model = model
        self.vocab_size = vocab_size
        self.cfg = cfg or MTMTrainConfig()
        self.device = torch.device(resolve_device(self.cfg.device))
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.rng = np.random.default_rng(self.cfg.seed)

        # History so callers can plot / log the learnability curve
        self.history: list[dict[str, float]] = []
        self._step = 0

    def _lr_at(self, step: int, total_steps: int) -> float:
        """Linear-warmup + cosine-decay schedule."""
        warmup = max(1, self.cfg.warmup_steps)
        if step < warmup:
            return self.cfg.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return self.cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def fit(
        self, sequences: np.ndarray | dict[str, np.ndarray]
    ) -> list[dict[str, float]]:
        """Train ``self.model`` on ``sequences``.

        ``sequences`` is either a 2-D ``(N, S)`` int64 array (legacy
        token-only pretraining) or a dict of three ``(N, S)`` arrays
        (hybrid).  Returns the per-log-step history.
        """
        hybrid = isinstance(sequences, dict)
        if hybrid:
            if not getattr(self.model, "hybrid", False):
                msg = "Hybrid sequences require a MaskedTokenModel(hybrid=True)"
                raise ValueError(msg)
            tids = sequences["token_ids"]
            if tids.ndim != 2:
                msg = f"token_ids must be 2-D; got shape {tids.shape}"
                raise ValueError(msg)
            collate = _hybrid_collate
        else:
            if sequences.ndim != 2:
                msg = f"sequences must be 2-D (N, S); got shape {sequences.shape}"
                raise ValueError(msg)
            collate = lambda batch: np.stack(batch)  # noqa: E731 — concise + local

        ds = _TokenSequenceDataset(sequences)
        loader = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=collate,
        )
        total_steps = self.cfg.n_epochs * max(1, len(loader))

        self.model.train()
        for epoch in range(self.cfg.n_epochs):
            for batch in loader:
                if hybrid:
                    batch_seqs = batch["token_ids"]
                    cont_values_np = batch["cont_values"]
                    cont_kinds_np = batch["cont_kinds"]
                    inputs, targets, mask = mlm_collate(
                        batch_seqs,
                        vocab_size=self.vocab_size,
                        mask_prob=self.cfg.mask_prob,
                        rng=self.rng,
                        cont_kinds=cont_kinds_np,
                    )
                else:
                    cont_values_np = None
                    cont_kinds_np = None
                    inputs, targets, mask = mlm_collate(
                        batch,
                        vocab_size=self.vocab_size,
                        mask_prob=self.cfg.mask_prob,
                        rng=self.rng,
                    )
                if mask.sum().item() == 0:
                    continue
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                if hybrid:
                    cv = torch.from_numpy(cont_values_np).to(self.device)
                    ck = torch.from_numpy(cont_kinds_np).to(self.device)
                    logits, _ = self.model(inputs, cv, ck)
                else:
                    logits, _ = self.model(inputs)
                loss = self.loss_fn(
                    logits.reshape(-1, self.vocab_size), targets.reshape(-1)
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )
                for g in self.optimizer.param_groups:
                    g["lr"] = self._lr_at(self._step, total_steps)
                self.optimizer.step()

                with torch.no_grad():
                    valid = targets != -100
                    preds = logits.argmax(dim=-1)
                    correct = ((preds == targets) & valid).sum().item()
                    total = valid.sum().item()
                    acc = correct / max(1, total)

                if self._step % self.cfg.log_every == 0:
                    self.history.append(
                        {
                            "step": float(self._step),
                            "epoch": float(epoch),
                            "loss": float(loss.item()),
                            "mlm_accuracy": float(acc),
                            "lr": float(self.optimizer.param_groups[0]["lr"]),
                        }
                    )
                self._step += 1

        self.model.eval()
        return self.history

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path, extras: dict[str, Any] | None = None) -> None:
        """Persist the MLM model + a metadata payload to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Hybrid encoders wrap the backbone in ``HybridTokenEncoder``, so
        # reach through the wrapper when introspecting transformer-block
        # parameters.
        if getattr(self.model, "hybrid", False):
            backbone = self.model.encoder.backbone
        else:
            backbone = self.model.encoder
        first_block = backbone.blocks[0]
        n_heads = first_block.attn.n_heads

        encoder_kwargs: dict[str, Any] = {
            "d_model": backbone.d_model,
            "max_seq_len": backbone.max_seq_len,
            "pad_token_id": backbone.pad_token_id,
            "n_layers": len(backbone.blocks),
            "n_heads": int(n_heads),
        }
        extras = dict(extras or {})
        if getattr(self.model, "hybrid", False):
            # Carry the Fourier input scale through so consumers
            # (PPOLatent, the script's eval helper) can rebuild the
            # encoder with the same dynamic range.
            extras.setdefault(
                "sim_time_scale", float(self.model.encoder.fourier.input_scale)
            )
            extras.setdefault("hybrid", True)

        payload: dict[str, Any] = {
            "model": self.model.state_dict(),
            "vocab_size": self.vocab_size,
            "encoder_kwargs": encoder_kwargs,
            "history": self.history,
            "extras": extras,
        }
        torch.save(payload, path)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        sequences: np.ndarray | dict[str, np.ndarray],
        *,
        n_passes: int = 1,
        mask_prob: float | None = None,
    ) -> dict[str, Any]:
        """Score the trained model on held-out sequences.

        Accepts either a 2-D ``(N, S)`` int64 array (legacy mode) or a
        dict of three ``(N, S)`` arrays (hybrid mode).  Returns
        per-token-id accuracy + overall MLM accuracy averaged over
        ``n_passes`` independent maskings.
        """
        hybrid = isinstance(sequences, dict)
        if not hybrid:
            if sequences.ndim != 2:
                msg = f"sequences must be 2-D (N, S); got shape {sequences.shape}"
                raise ValueError(msg)
            sequences = sequences.astype(np.int64)
        mask_prob_eff = self.cfg.mask_prob if mask_prob is None else float(mask_prob)
        ds = _TokenSequenceDataset(sequences)
        collate = _hybrid_collate if hybrid else (lambda batch: np.stack(batch))
        loader = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate,
        )

        per_id_correct: dict[int, int] = {}
        per_id_total: dict[int, int] = {}
        total_correct = 0
        total_seen = 0

        self.model.eval()
        for _ in range(max(1, int(n_passes))):
            for batch in loader:
                if hybrid:
                    batch_seqs = batch["token_ids"]
                    cont_values_np = batch["cont_values"]
                    cont_kinds_np = batch["cont_kinds"]
                    inputs, targets, _ = mlm_collate(
                        batch_seqs,
                        vocab_size=self.vocab_size,
                        mask_prob=mask_prob_eff,
                        rng=self.rng,
                        cont_kinds=cont_kinds_np,
                    )
                else:
                    inputs, targets, _ = mlm_collate(
                        batch,
                        vocab_size=self.vocab_size,
                        mask_prob=mask_prob_eff,
                        rng=self.rng,
                    )
                inputs = inputs.to(self.device)
                targets_np = targets.numpy()
                if hybrid:
                    cv = torch.from_numpy(cont_values_np).to(self.device)
                    ck = torch.from_numpy(cont_kinds_np).to(self.device)
                    logits, _ = self.model(inputs, cv, ck)
                else:
                    logits, _ = self.model(inputs)
                preds = logits.argmax(dim=-1).cpu().numpy()

                valid = targets_np != -100
                ids = targets_np[valid]
                ok = preds[valid] == ids
                for tid, hit in zip(ids.tolist(), ok.tolist(), strict=True):
                    per_id_total[tid] = per_id_total.get(tid, 0) + 1
                    per_id_correct[tid] = per_id_correct.get(tid, 0) + int(hit)
                total_correct += int(ok.sum())
                total_seen += int(ok.size)

        per_id = {
            tid: {
                "correct": per_id_correct.get(tid, 0),
                "total": per_id_total[tid],
                "accuracy": per_id_correct.get(tid, 0) / per_id_total[tid],
            }
            for tid in per_id_total
        }
        return {
            "mlm_accuracy": total_correct / max(1, total_seen),
            "n_predictions": total_seen,
            "per_token_id": per_id,
        }

    @staticmethod
    def load_encoder(
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ):
        """Load just the encoder weights from a saved MLM checkpoint.

        Returns a :class:`ModernTransformerEncoder` for the legacy
        token-only pretraining, or a
        :class:`agents.networks.hybrid_encoder.HybridTokenEncoder`
        for hybrid pretraining (the type is auto-detected from the
        state-dict keys: hybrid checkpoints carry sub-modules named
        ``encoder.ratio_ple.*``, ``encoder.time2vec.*``, etc.).
        """
        from agents.networks.hybrid_encoder import HybridTokenEncoder

        ckpt = torch.load(path, map_location=device, weights_only=False)
        vocab_size = int(ckpt["vocab_size"])
        kw = ckpt["encoder_kwargs"]
        state = ckpt["model"]

        is_hybrid = any(
            k.startswith("encoder.ratio_ple.")
            or k.startswith("encoder.count_ple.")
            or k.startswith("encoder.time2vec.")
            or k.startswith("encoder.fourier.")
            for k in state.keys()
        )

        if is_hybrid:
            extras = ckpt.get("extras") or {}
            sim_time_scale = float(extras.get("sim_time_scale", 200_000.0))
            encoder = HybridTokenEncoder(
                vocab_size=vocab_size,
                d_model=int(kw["d_model"]),
                n_layers=int(kw["n_layers"]),
                n_heads=int(kw.get("n_heads", 6)),
                max_seq_len=int(kw["max_seq_len"]),
                pad_token_id=int(kw["pad_token_id"]),
                sim_time_scale=sim_time_scale,
            )
        else:
            encoder = ModernTransformerEncoder(
                vocab_size=vocab_size,
                d_model=int(kw["d_model"]),
                n_layers=int(kw["n_layers"]),
                n_heads=int(kw.get("n_heads", 6)),
                max_seq_len=int(kw["max_seq_len"]),
                pad_token_id=int(kw["pad_token_id"]),
            )
        enc_state = {
            k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")
        }
        encoder.load_state_dict(enc_state)
        encoder.to(device)
        return encoder

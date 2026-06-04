"""Dynamic vocabulary tokenizer for Transformer-based observation encoding.

The ``StateTokenizer`` converts string-valued observation tokens into integer
IDs suitable for embedding layers.  Observations use a key-value format where
field names (``SIM_TIME``, ``MACHINE_BROKEN``, …) and bucketed value tokens
(``T_0_50``, ``TRUE``, ``R_HIGH``, …) are separate entries, keeping the
vocabulary bounded.  New tokens are added on the fly until the vocabulary is
frozen.

Special tokens:
    0 = <PAD>   – used to pad sequences to fixed length
    1 = <UNK>   – reserved (also reused as the MASK token during MLM)
    2 = <BOS>   – beginning-of-sequence
    3 = <EOS>   – end-of-sequence
    4 = <NUM>   – placeholder at continuous-value positions in the
                  ``hybrid`` observation mode (the actual scalar travels
                  on a parallel float channel; the placeholder keeps
                  attention masks / position counts honest).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Special token IDs (always present at the start of the vocabulary)
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
NUM_ID = 4

_SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<NUM>"]


class StateTokenizer:
    """Dynamic-vocabulary tokenizer that maps observation strings to integers.

    Parameters
    ----------
    seq_length:
        Fixed output length.  Sequences shorter than this are padded with
        ``PAD_ID``; sequences longer are truncated.
    freeze:
        When ``True``, unseen tokens map to ``UNK_ID`` instead of being added
        to the vocabulary.  Call :meth:`freeze` after warmup to lock the vocab.

    """

    def __init__(self, seq_length: int = 64, *, freeze: bool = False) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: list[str] = []
        self.seq_length = seq_length
        self._frozen = freeze

        # Register special tokens
        for tok in _SPECIAL_TOKENS:
            self._add_token(tok)

    # ------------------------------------------------------------------
    # Vocabulary management
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size including special tokens."""
        return len(self._id_to_token)

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        """Freeze the vocabulary — unseen tokens will map to UNK."""
        self._frozen = True

    def unfreeze(self) -> None:
        """Unfreeze the vocabulary so new tokens can be added."""
        self._frozen = False

    def _add_token(self, token: str) -> int:
        """Add *token* to the vocabulary and return its ID."""
        if token in self._token_to_id:
            return self._token_to_id[token]
        idx = len(self._id_to_token)
        self._token_to_id[token] = idx
        self._id_to_token.append(token)
        return idx

    def token_to_id(self, token: str) -> int:
        """Return the integer ID for *token*, adding it if the vocab is open."""
        if token in self._token_to_id:
            return self._token_to_id[token]
        if self._frozen:
            return UNK_ID
        return self._add_token(token)

    def id_to_token(self, token_id: int) -> str:
        """Return the string token for *token_id*."""
        if 0 <= token_id < len(self._id_to_token):
            return self._id_to_token[token_id]
        return "<UNK>"

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, tokens: list[str] | tuple[str, ...]) -> NDArray[np.int64]:
        """Convert a sequence of string tokens to a fixed-length integer array.

        Adds ``<BOS>`` at the start and ``<EOS>`` at the end, then pads or
        truncates to ``self.seq_length``.
        """
        ids = [BOS_ID]
        for tok in tokens:
            ids.append(self.token_to_id(tok))
        ids.append(EOS_ID)

        # Truncate or pad to seq_length
        if len(ids) > self.seq_length:
            ids = ids[: self.seq_length]
        else:
            ids.extend([PAD_ID] * (self.seq_length - len(ids)))

        return np.array(ids, dtype=np.int64)

    def encode_batch(
        self, batch: list[list[str] | tuple[str, ...]]
    ) -> NDArray[np.int64]:
        """Encode multiple sequences, returning a 2-D array."""
        return np.stack([self.encode(seq) for seq in batch])

    def decode(self, ids: NDArray[np.int64] | list[int]) -> list[str]:
        """Convert integer IDs back to string tokens (strips special tokens)."""
        return [
            self.id_to_token(int(i))
            for i in ids
            if int(i) not in (PAD_ID, BOS_ID, EOS_ID)
        ]

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def get_vocab(self) -> dict[str, int]:
        """Return a copy of the current vocabulary mapping."""
        return dict(self._token_to_id)

    def load_vocab(self, vocab: dict[str, int]) -> None:
        """Load a previously saved vocabulary, replacing the current one."""
        self._id_to_token = [""] * len(vocab)
        self._token_to_id = {}
        for tok, idx in vocab.items():
            if idx >= len(self._id_to_token):
                self._id_to_token.extend([""] * (idx - len(self._id_to_token) + 1))
            self._id_to_token[idx] = tok
            self._token_to_id[tok] = idx

    # ------------------------------------------------------------------
    # Full-vocabulary construction
    # ------------------------------------------------------------------

    @classmethod
    def build_vocab(
        cls,
        machine_types: list[str],
        n_technicians: int,
        *,
        seq_length: int = 64,
        component_types: list[str] | None = None,
        next_ticket_lookahead: int = 4,
    ) -> StateTokenizer:
        """Create a tokenizer pre-populated with every possible token.

        This avoids the need for warm-up episodes: the vocabulary is
        deterministically constructed from the environment's configuration.

        Parameters
        ----------
        machine_types:
            Unique machine type strings used in the factory (e.g.
            ``["CNC", "Assembly"]``).
        n_technicians:
            Number of technicians in the environment.
        seq_length:
            Fixed output sequence length.
        component_types:
            Unique component type strings used in the factory (e.g.
            ``["motor", "spindle", "pump"]``).  Only required for the
            ``tech_aware`` observation mode; unknown component types
            still work via the tokenizer's UNK fallback but will all
            collide on a single ID.

        Returns
        -------
        StateTokenizer
            A frozen tokenizer whose vocab covers all observation tokens.

        """
        tok = cls(seq_length=seq_length)

        # -- observation mode values --
        for mode in ("ticket_only", "broken_machine", "factory_level", "tech_aware"):
            tok._add_token(mode)

        # -- key tokens --
        keys = [
            "OBS_MODE",
            "SIM_TIME",
            "HAS_TICKET",
            "TICKET_AGE",
            "TICKET_MACHINE_TYPE",
            "TICKET_COMPONENT_TYPE",
            "MACHINE_TYPE",
            "MACHINE_BROKEN",
            "MACHINE_PROCESSING",
            "MACHINE_TOTAL_PROCESSED",
            "MACHINE_INPUT_BUF",
            "MACHINE_OUTPUT_BUF",
            "FACTORY_MACHINES",
            "FACTORY_BROKEN",
            "FACTORY_PROCESSING",
            "FACTORY_PRODUCED",
            "FACTORY_QUEUE",
            "BUSY",
            "FATIGUE",
            "KNOWLEDGE",
            "MATCH",
            "ETA",
            "LAST_AGE",
            "ASSIGN_COUNT",
        ]
        for k in keys:
            tok._add_token(k)
        for slot in range(1, max(1, int(next_ticket_lookahead)) + 1):
            for suffix in ("MACHINE_TYPE", "COMPONENT_TYPE", "AGE"):
                tok._add_token(f"NEXT{slot}_{suffix}")

        # -- per-technician prefix tokens --
        for i in range(n_technicians):
            tok._add_token(f"TECH_{i}")

        # -- bucket values (single source of truth: src/kata/env.py) --
        from kata.env import _COUNT_BUCKETS, _RATIO_BUCKETS, _TIME_BUCKETS

        for v in _TIME_BUCKETS:
            tok._add_token(v)
        for v in _RATIO_BUCKETS:
            tok._add_token(v)
        for v in _COUNT_BUCKETS:
            tok._add_token(v)

        # -- boolean values --
        tok._add_token("TRUE")
        tok._add_token("FALSE")

        # -- machine type values (including sentinel) --
        tok._add_token("NONE")
        for mt in machine_types:
            tok._add_token(mt)
            # Per-machine-type broken-count key used by the opt-in
            # ``include_broken_by_type_tokens`` token block.
            tok._add_token(f"BROKEN_{mt}")

        # -- component type values (for tech_aware mode) --
        if component_types:
            for ct in component_types:
                tok._add_token(ct)
                # Per-component-type queue-composition key used by the
                # opt-in ``include_queue_composition_tokens`` block.
                tok._add_token(f"QC_{ct}")

        # ``BROKEN_NONE`` / ``QC_NONE`` can show up when a machine or
        # request advertises no concrete type yet — register them so
        # the tokenizer never falls back to UNK.
        tok._add_token("BROKEN_NONE")
        tok._add_token("QC_NONE")

        tok.freeze()
        return tok

    @classmethod
    def build_set_vocab(
        cls,
        machine_types: list[str],
        component_types: list[str],
        technician_templates: list[str],
        *,
        seq_length: int = 64,
    ) -> StateTokenizer:
        """Create a *frozen* tokenizer for the ``set`` observation mode.

        Equivalent to :meth:`build_vocab` for the flat-stream modes:
        deterministically enumerates every token string the
        :class:`kata.env._SetEmitter` can produce.  Token IDs are
        assigned in a fixed order driven by the input lists, so
        training and eval calls with the same ``machine_types`` /
        ``component_types`` / ``technician_templates`` lists produce
        identical id mappings — that's what makes the agent's
        embedding table portable across runs.

        Parameters
        ----------
        machine_types:
            Every machine type string the scenario sampler can emit
            (typically ``sampler.all_machine_types()``).
        component_types:
            Every component type string (typically
            ``sampler.all_component_types()``).
        technician_templates:
            Every technician template name in the scenario pool
            (typically ``rcfg.technician_templates``).
        seq_length:
            Carried forward to the returned tokenizer; the set mode
            does not consume it for sequence length but the field is
            kept consistent with the flat-stream tokenizer's API.
        """
        tok = cls(seq_length=seq_length)

        # -- fixed continuous-feature tokens (one per emitter call,
        #    regardless of the scalar value) --------------------------
        fixed_continuous = [
            # tech slot
            "<RATIO:FATIGUE>",
            "<COUNT:ASSIGNS>",
            "<COUNT:KNOW_VOL>",
            "<COUNT:KNOW_MAX>",
            "<RATIO:KNOW_SPEC>",
            "<COUNT:KNOW_ENT>",
            "<RATIO:MATCH>",
            "<RATIO:MATCH_N1>",
            "<RATIO:MATCH_N2>",
            "<TIME:ETA>",
            "<TIME:LAST_AGE>",
            # machine slot
            "<COUNT:PROC_TOT>",
            "<COUNT:IN_BUF>",
            "<COUNT:OUT_BUF>",
            "<COUNT:BD_COUNT>",
            "<TIME:DOWNTIME>",
            "<TIME:MEAN_TBF>",
            # env slot
            "<FOUR:SIM_T>",
            "<TIME:T_AGE>",
            "<COUNT:Q_SIZE>",
            "<COUNT:BROKEN_N>",
            "<COUNT:PROC_N>",
            "<TIME:N1_AGE>",
            "<TIME:N2_AGE>",
        ]
        for t in fixed_continuous:
            tok._add_token(t)

        # -- boolean cat tokens --------------------------------------
        for key in ("BUSY", "DISRUPT", "BROKEN", "PROC", "IS_CURRENT", "HAS_T"):
            tok._add_token(f"{key}=T")
            tok._add_token(f"{key}=F")

        # -- per-template tokens -------------------------------------
        for t in technician_templates:
            tok._add_token(f"TEMPLATE={t}")

        # -- per-machine-type tokens (M_TYPE on machine slot;
        #    T_M_TYPE / N1_M_TYPE / N2_M_TYPE on env slot) -----------
        machine_keys = ("M_TYPE", "T_M_TYPE", "N1_M_TYPE", "N2_M_TYPE")
        for k in machine_keys:
            for mt in machine_types:
                tok._add_token(f"{k}={mt}")
        # NONE sentinel for env-side keys when no ticket / no lookahead
        for k in ("T_M_TYPE", "N1_M_TYPE", "N2_M_TYPE"):
            tok._add_token(f"{k}=NONE")

        # -- per-component-type tokens (CUR_COMP on machine slot;
        #    T_C_TYPE / N1_C_TYPE / N2_C_TYPE on env slot) -----------
        comp_keys = ("CUR_COMP", "T_C_TYPE", "N1_C_TYPE", "N2_C_TYPE")
        for k in comp_keys:
            for ct in component_types:
                tok._add_token(f"{k}={ct}")
        for k in comp_keys:
            tok._add_token(f"{k}=NONE")

        tok.freeze()
        return tok

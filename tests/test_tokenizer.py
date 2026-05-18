"""Tests for StateTokenizer."""

import numpy as np

from kata.core.tokenizer import BOS_ID, EOS_ID, PAD_ID, UNK_ID, StateTokenizer


class TestVocabulary:
    def test_special_tokens_present(self):
        tok = StateTokenizer()
        assert tok.vocab_size >= 4
        assert tok.token_to_id("<PAD>") == PAD_ID
        assert tok.token_to_id("<UNK>") == UNK_ID
        assert tok.token_to_id("<BOS>") == BOS_ID
        assert tok.token_to_id("<EOS>") == EOS_ID

    def test_new_token_added(self):
        tok = StateTokenizer()
        initial = tok.vocab_size
        tok.token_to_id("HELLO")
        assert tok.vocab_size == initial + 1

    def test_same_token_not_duplicated(self):
        tok = StateTokenizer()
        id1 = tok.token_to_id("FOO")
        id2 = tok.token_to_id("FOO")
        assert id1 == id2

    def test_frozen_returns_unk(self):
        tok = StateTokenizer()
        tok.token_to_id("KNOWN")
        tok.freeze()
        assert tok.token_to_id("UNKNOWN_VALUE") == UNK_ID
        assert tok.frozen

    def test_unfreeze_allows_growth(self):
        tok = StateTokenizer()
        tok.freeze()
        tok.unfreeze()
        idx = tok.token_to_id("NEW")
        assert idx != UNK_ID

    def test_id_to_token_roundtrip(self):
        tok = StateTokenizer()
        idx = tok.token_to_id("ROUNDTRIP")
        assert tok.id_to_token(idx) == "ROUNDTRIP"

    def test_id_to_token_out_of_range(self):
        tok = StateTokenizer()
        assert tok.id_to_token(99999) == "<UNK>"


class TestEncoding:
    def test_output_shape(self):
        tok = StateTokenizer(seq_length=32)
        result = tok.encode(["A", "B", "C"])
        assert result.shape == (32,)
        assert result.dtype == np.int64

    def test_bos_eos_present(self):
        tok = StateTokenizer(seq_length=16)
        result = tok.encode(["X"])
        assert result[0] == BOS_ID
        # EOS should be at position 2 (BOS, X, EOS, PAD...)
        assert result[2] == EOS_ID

    def test_padding(self):
        tok = StateTokenizer(seq_length=10)
        result = tok.encode(["A"])
        # BOS, A, EOS, then 7 PADs
        assert (result[3:] == PAD_ID).all()

    def test_truncation(self):
        tok = StateTokenizer(seq_length=5)
        result = tok.encode(["A", "B", "C", "D", "E", "F", "G"])
        assert len(result) == 5

    def test_encode_batch(self):
        tok = StateTokenizer(seq_length=8)
        batch = [["A", "B"], ["C", "D", "E"]]
        result = tok.encode_batch(batch)
        assert result.shape == (2, 8)


class TestDecode:
    def test_decode_strips_special(self):
        tok = StateTokenizer(seq_length=10)
        ids = tok.encode(["HELLO", "WORLD"])
        decoded = tok.decode(ids)
        assert "HELLO" in decoded
        assert "WORLD" in decoded
        assert "<PAD>" not in decoded
        assert "<BOS>" not in decoded
        assert "<EOS>" not in decoded


class TestBuildVocabCoversAllEmittedTokens:
    """End-to-end audit: ``build_vocab`` must cover every token any
    observation mode of the env can actually produce.

    Runs short rollouts over all four observation modes with both
    fatigue and knowledge tokens enabled, and asserts that *every*
    token emitted lands on a real vocab ID (i.e. ``token_to_id`` never
    returns ``UNK_ID``).
    """

    def test_no_silent_unk_across_all_modes(self):
        import os
        import random

        os.environ["KATA_CONF_PATH"] = "/dev/null/__no_file__"

        from kata.core.config import KATAConfig
        from kata.env import KataEnv
        from kata.scenario import ScenarioBuilder

        # Use the richest run config we ship so the audit exercises
        # all machine types, all component types, and the largest
        # technician fleet.
        with open("run_configs/factory_v2.json") as f:
            import json as _json
            data = _json.load(f)
        base_cfg = KATAConfig(**data)

        machine_types = sorted({m.machine_type for m in base_cfg.machines.values()})
        component_types = sorted(
            {
                c.component_type
                for m in base_cfg.machines.values()
                for c in m.components.values()
            }
        )
        tok = StateTokenizer.build_vocab(
            machine_types=machine_types,
            n_technicians=len(base_cfg.technicians),
            seq_length=base_cfg.gym.tokenizer_seq_length or 128,
            component_types=component_types,
        )
        assert tok.frozen, "build_vocab should freeze the tokenizer"

        for mode in (
            "ticket_only",
            "broken_machine",
            "factory_level",
            "tech_aware",
        ):
            cfg = KATAConfig(**data)
            cfg.gym.observation_representation = "tokens"
            cfg.gym.observation_mode = mode
            cfg.gym.include_technician_fatigue_tokens = True
            cfg.gym.include_technician_knowledge_tokens = True
            cfg.gym.token_observation_length = 128

            env = KataEnv(
                scenario_factory=lambda c=cfg: ScenarioBuilder(c).build(),
                config=cfg.gym,
            )
            env.reset(seed=0)

            unknowns: list[tuple[str, int]] = []  # (token, step_idx)
            for tok_str in env._token_obs()["tokens"]:
                if tok_str == "<PAD>":
                    continue
                if tok.token_to_id(tok_str) == UNK_ID:
                    unknowns.append((tok_str, -1))

            random.seed(0)
            done = False
            step_i = 0
            while not done and step_i < 80:
                a = random.randrange(env.action_space.n)
                _, _, done, _, _ = env.step(a)
                for tok_str in env._token_obs()["tokens"]:
                    if tok_str == "<PAD>":
                        continue
                    if tok.token_to_id(tok_str) == UNK_ID:
                        unknowns.append((tok_str, step_i))
                step_i += 1

            assert not unknowns, (
                f"observation_mode={mode!r} emitted tokens that are not in the "
                f"frozen vocab and would silently map to <UNK>: "
                f"{sorted(set(t for t, _ in unknowns))}"
            )

    def test_build_vocab_contains_every_documented_key(self):
        """Make sure build_vocab keys haven't drifted away from the env's
        token emission code (regression catch for token-name typos)."""
        tok = StateTokenizer.build_vocab(
            machine_types=["CNC"],
            n_technicians=2,
            seq_length=64,
        )
        # Every key the env's _token_obs may emit
        required_keys = {
            "OBS_MODE", "SIM_TIME", "HAS_TICKET", "TICKET_AGE",
            "TICKET_MACHINE_TYPE", "TICKET_COMPONENT_TYPE",
            "MACHINE_TYPE", "MACHINE_BROKEN", "MACHINE_PROCESSING",
            "MACHINE_TOTAL_PROCESSED", "MACHINE_INPUT_BUF", "MACHINE_OUTPUT_BUF",
            "FACTORY_MACHINES", "FACTORY_BROKEN", "FACTORY_PROCESSING",
            "FACTORY_PRODUCED", "FACTORY_QUEUE",
            "BUSY", "FATIGUE", "KNOWLEDGE",
            "MATCH", "ETA", "LAST_AGE",
            "NEXT1_MACHINE_TYPE", "NEXT1_COMPONENT_TYPE", "NEXT1_AGE",
            "NEXT2_MACHINE_TYPE", "NEXT2_COMPONENT_TYPE", "NEXT2_AGE",
        }
        for key in required_keys:
            assert tok.token_to_id(key) != UNK_ID, f"missing vocab entry: {key}"
        for mode in ("ticket_only", "broken_machine", "factory_level", "tech_aware"):
            assert tok.token_to_id(mode) != UNK_ID, f"missing OBS_MODE value: {mode}"
        # Bucket values (finer-resolution buckets land here as well)
        for t in ("T_NONE", "T_0_50", "T_1K_5K", "T_10K_50K", "T_100K+"):
            assert tok.token_to_id(t) != UNK_ID
        for r in ("R_0", "R_0_10", "R_40_50", "R_90_100"):
            assert tok.token_to_id(r) != UNK_ID
        for c in ("C_0", "C_11_20", "C_21_50", "C_100+"):
            assert tok.token_to_id(c) != UNK_ID
        for b in ("TRUE", "FALSE"):
            assert tok.token_to_id(b) != UNK_ID
        # NONE sentinel + the one machine type we supplied
        assert tok.token_to_id("NONE") != UNK_ID
        assert tok.token_to_id("CNC") != UNK_ID
        # Per-tech prefixes
        assert tok.token_to_id("TECH_0") != UNK_ID
        assert tok.token_to_id("TECH_1") != UNK_ID


class TestSerialisation:
    def test_get_and_load_vocab(self):
        tok1 = StateTokenizer(seq_length=8)
        tok1.encode(["A", "B", "C"])
        vocab = tok1.get_vocab()

        tok2 = StateTokenizer(seq_length=8)
        tok2.load_vocab(vocab)
        assert tok2.vocab_size == tok1.vocab_size
        assert tok2.token_to_id("A") == tok1.token_to_id("A")

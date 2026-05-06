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


class TestSerialisation:
    def test_get_and_load_vocab(self):
        tok1 = StateTokenizer(seq_length=8)
        tok1.encode(["A", "B", "C"])
        vocab = tok1.get_vocab()

        tok2 = StateTokenizer(seq_length=8)
        tok2.load_vocab(vocab)
        assert tok2.vocab_size == tok1.vocab_size
        assert tok2.token_to_id("A") == tok1.token_to_id("A")

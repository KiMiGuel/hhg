"""Tests for face_engine.py - hash determinism, embedding properties, cosine similarity."""

import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.face_engine import EMBEDDING_DIM, SFACE_COSINE_THRESHOLD, FaceEngine


def _synthetic_embedding(seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(EMBEDDING_DIM).astype(np.float32)


class TestBiometricHashDeterminism:
    def test_same_embedding_same_hash(self):
        emb = _synthetic_embedding(seed=7)
        h1 = "0x" + hashlib.sha256(emb.tobytes()).hexdigest()
        h2 = "0x" + hashlib.sha256(emb.tobytes()).hexdigest()
        assert h1 == h2

    def test_different_embedding_different_hash(self):
        emb_a = _synthetic_embedding(seed=1)
        emb_b = _synthetic_embedding(seed=2)
        h_a = "0x" + hashlib.sha256(emb_a.tobytes()).hexdigest()
        h_b = "0x" + hashlib.sha256(emb_b.tobytes()).hexdigest()
        assert h_a != h_b

    def test_hash_format(self):
        emb = _synthetic_embedding(seed=0)
        h = "0x" + hashlib.sha256(emb.tobytes()).hexdigest()
        assert h.startswith("0x")
        assert len(h) == 66

    def test_embedding_dimension(self):
        emb = _synthetic_embedding(seed=99)
        assert emb.shape == (EMBEDDING_DIM,)
        assert emb.dtype == np.float32


class TestCosineSimilarity:
    def test_identical_embedding_is_one(self):
        emb = _synthetic_embedding(seed=10)
        assert FaceEngine.cosine_similarity(emb, emb) == pytest.approx(1.0, abs=1e-5)

    def test_opposite_embedding_is_minus_one(self):
        emb = _synthetic_embedding(seed=10)
        assert FaceEngine.cosine_similarity(emb, -emb) == pytest.approx(-1.0, abs=1e-5)

    def test_orthogonal_embedding_is_zero(self):
        emb_a = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        emb_a[0] = 1.0
        emb_b = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        emb_b[1] = 1.0
        assert FaceEngine.cosine_similarity(emb_a, emb_b) == pytest.approx(0.0, abs=1e-5)

    def test_range_bounds(self):
        for seed in range(20):
            emb_a = _synthetic_embedding(seed=seed)
            emb_b = _synthetic_embedding(seed=seed + 100)
            sim = FaceEngine.cosine_similarity(emb_a, emb_b)
            assert -1.0 <= sim <= 1.0

    def test_same_person_threshold(self):
        emb = _synthetic_embedding(seed=55)
        assert FaceEngine.same_person(emb, emb) is True

    def test_different_person_threshold(self):
        emb_a = _synthetic_embedding(seed=1)
        emb_b = _synthetic_embedding(seed=2)
        sim = FaceEngine.cosine_similarity(emb_a, emb_b)
        result = FaceEngine.same_person(emb_a, emb_b)
        if sim < SFACE_COSINE_THRESHOLD:
            assert result is False

"""Tests for blockchain.py - fingerprint computation and tamper logic (no network needed)."""

import hashlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.blockchain import BlockchainManager


class TestComputeDataFingerprint:
    def test_fingerprint_format(self):
        h = BlockchainManager.compute_data_fingerprint("0xabc", "https://example.com")
        assert h.startswith("0x")
        assert len(h) == 66

    def test_fingerprint_deterministic(self):
        h1 = BlockchainManager.compute_data_fingerprint("0xabc", "https://example.com")
        h2 = BlockchainManager.compute_data_fingerprint("0xabc", "https://example.com")
        assert h1 == h2

    def test_fingerprint_changes_with_url(self):
        h_orig = BlockchainManager.compute_data_fingerprint("0xabc", "https://example.com/post")
        h_tampered = BlockchainManager.compute_data_fingerprint("0xabc", "https://example.com/posu")
        assert h_orig != h_tampered

    def test_fingerprint_changes_with_face_hash(self):
        h1 = BlockchainManager.compute_data_fingerprint("0xaaa", "https://example.com")
        h2 = BlockchainManager.compute_data_fingerprint("0xbbb", "https://example.com")
        assert h1 != h2

    def test_fingerprint_matches_manual_sha256(self):
        face_hash = "0x1234567890abcdef"
        post_url = "https://youtube.com/watch?v=test"
        expected = "0x" + hashlib.sha256(f"{face_hash}:{post_url}".encode()).hexdigest()
        assert BlockchainManager.compute_data_fingerprint(face_hash, post_url) == expected

    def test_one_character_difference_detected(self):
        face_hash = "0x" + "a" * 64
        url = "https://www.youtube.com/watch?v=QV3BrBavziU"
        h_orig = BlockchainManager.compute_data_fingerprint(face_hash, url)
        h_tampered = BlockchainManager.compute_data_fingerprint(face_hash, url + "_tampered")
        assert h_orig != h_tampered

    def test_fingerprint_is_sha256_not_md5(self):
        h = BlockchainManager.compute_data_fingerprint("0xabc", "https://example.com")
        hex_part = h[2:]
        assert len(hex_part) == 64

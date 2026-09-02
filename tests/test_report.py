"""Tests for report.py - JSON + Markdown report generation."""
import json
import os
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.report import write_report


class TestWriteReport:
    def _sample_data(self):
        return {
            "generated_at": "2026-09-03T12:00:00+00:00",
            "network": {
                "rpc_url": "http://127.0.0.1:8545",
                "chain_id": 31337,
                "contract_address": "0x5FbDB2315678afecb367f032d93F642f64180aa3",
            },
            "stage1": {
                "image": "data/sample_face.jpg",
                "bbox": [634, 555, 812, 1060],
                "embedding_dim": 128,
                "face_hash": "0x361968f0dbfdf6e6d92556f355ca3215cd2c1b1e5b3c872649c339aae28",
            },
            "stage2": {
                "image_host_url": "https://files.catbox.moe/abc.jpg",
                "title": "Satya Nadella - CEO of Microsoft",
                "platform": "youtube.com",
                "post_url": "https://www.youtube.com/watch?v=QV3BrBavziU",
            },
            "stage3": {
                "fingerprint": "0x539684dc23c3548",
                "already_anchored": False,
                "tx_hash": "0xe1c4762a",
                "block": 5,
                "gas_used": 183743,
            },
            "stage4": {
                "verification": "PASSED",
                "on_chain_fingerprint": "0x539684dc23c3548",
                "tamper_detected": True,
                "tampered_url": "https://www.youtube.com/watch?v=QV3BrBavziU_tampered",
            },
        }

    def test_creates_both_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = self._sample_data()
            json_path = write_report(data, out_dir=tmpdir)
            assert os.path.exists(json_path)
            md_path = json_path.replace(".json", ".md")
            assert os.path.exists(md_path)

    def test_json_is_valid_and_parseable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = self._sample_data()
            json_path = write_report(data, out_dir=tmpdir)
            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded["stage1"]["face_hash"] == data["stage1"]["face_hash"]
            assert loaded["stage4"]["verification"] == "PASSED"

    def test_markdown_contains_key_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = self._sample_data()
            json_path = write_report(data, out_dir=tmpdir)
            md_path = json_path.replace(".json", ".md")
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            assert "Pipeline Verification Report" in md_content
            assert data["stage1"]["face_hash"] in md_content
            assert data["stage2"]["post_url"] in md_content
            assert "PASSED" in md_content

    def test_report_timestamps_differ(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = self._sample_data()
            path1 = write_report(data, out_dir=tmpdir)
            time.sleep(1.1)
            path2 = write_report(data, out_dir=tmpdir)
            assert path1 != path2

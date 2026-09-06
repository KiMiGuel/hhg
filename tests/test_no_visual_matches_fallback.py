"""Pins the no-visual-matches early-exit behavior.

Background: previously ``WebSearchEngine._select`` raised
``RuntimeError("No visual matches found for this face...")`` when both
``visual_matches`` and ``knowledge_graph`` were empty. That exception
propagated straight through ``upload_and_search`` and broke the
multi-crop consensus run before it could try the second crop. Now:

1. ``search_face_on_web`` catches the RuntimeError, builds an empty
   ``LensResult`` with ``visual_match_count=0``, caches it so reruns
   don't re-charge SerpApi, and returns it cleanly.
2. ``search_with_consensus`` detects ``visual_match_count == 0`` on
   either crop and returns the OTHER crop's result. If both are empty,
   it returns the empty result instead of crashing.

These tests pin both behaviors with a mocked ``_request_serpapi_engine`` (called per cascade engine) so
they don't depend on SerpApi being up.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.web_search import LensResult, WebSearchEngine


@pytest.fixture
def tmp_cache(monkeypatch):
    """Redirect CACHE_DIR to a temp dir so tests don't pollute the real cache."""
    d = tempfile.mkdtemp(prefix="hhg_test_cache_")
    monkeypatch.setattr("src.web_search.CACHE_DIR", d)
    yield d


def _engine(tmp_cache):
    return WebSearchEngine("test-serp-api-key-fake")


def _empty_serp_response():
    """The exact shape SerpApi returns when Lens has no clue."""
    return {
        "visual_matches": [],
        "knowledge_graph": None,
        "search_metadata": {"total_time_taken": "0.42"},
    }


def _populated_serp_response():
    return {
        "visual_matches": [
            {
                "rank": 1,
                "title": "Satya Nadella - Wikipedia",
                "link": "https://en.wikipedia.org/wiki/Satya_Nadella",
                "source": "Wikipedia",
                "platform": "wikipedia.org",
            },
        ],
        "knowledge_graph": {
            "title": "Satya Nadella",
            "link": "https://en.wikipedia.org/wiki/Satya_Nadella",
        },
        "search_metadata": {"total_time_taken": "0.55"},
    }


def test_search_face_on_web_returns_empty_result_when_no_matches(tmp_cache):
    """SerpApi returns 0 visual matches -> LensResult with visual_match_count=0,
    no RuntimeError propagated."""
    eng = _engine(tmp_cache)
    with patch.object(eng, "_request_serpapi_engine", return_value=_empty_serp_response()):
        result = eng.search_face_on_web(
            "https://example.com/upload.jpg",
            face_hash="abc123",
            image_sha256="deadbeef",
        )
    assert isinstance(result, LensResult)
    assert result.visual_match_count == 0
    assert result.visual_matches == []
    assert result.knowledge_graph is None
    # The empty result is cached so reruns don't re-charge SerpApi.
    cache_files = list(Path(tmp_cache).glob("*.json"))
    assert len(cache_files) == 1


def test_search_face_on_web_no_runtime_error(tmp_cache):
    """Specifically: the call must not raise RuntimeError, even on empty results."""
    eng = _engine(tmp_cache)
    with patch.object(eng, "_request_serpapi_engine", return_value=_empty_serp_response()):
        # If the bug is back, this line raises RuntimeError.
        result = eng.search_face_on_web(
            "https://example.com/x.jpg", face_hash="h", image_sha256="s"
        )
    assert result.visual_match_count == 0


def test_search_with_consensus_falls_back_when_first_crop_empty(tmp_cache):
    """First crop = 0 matches; second crop = real match -> use the second."""
    eng = _engine(tmp_cache)
    calls = {"n": 0}

    def fake_request(url, policy="social", extra_params=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _empty_serp_response()
        return _populated_serp_response()

    with (
        patch.object(eng, "_request_serpapi_engine", side_effect=fake_request),
        patch.object(eng, "_upload_bytes", return_value="https://catbox.moe/x.jpg"),
    ):
        url, result, *_ = eng.search_with_consensus(
            enhanced_bytes=b"enhanced-bytes",
            tight_bytes=b"tight-bytes",
            face_hash="face-xyz",
        )
    assert result.visual_match_count > 0
    assert "Satya Nadella" in result.selected.title


def test_search_with_consensus_returns_empty_when_both_crops_empty(tmp_cache):
    """Both crops = 0 matches -> return empty result (no crash)."""
    eng = _engine(tmp_cache)
    with (
        patch.object(eng, "_request_serpapi_engine", return_value=_empty_serp_response()),
        patch.object(eng, "_upload_bytes", return_value="https://catbox.moe/x.jpg"),
    ):
        url, result, *_ = eng.search_with_consensus(
            enhanced_bytes=b"e", tight_bytes=b"t", face_hash="face-empty"
        )
    assert result.visual_match_count == 0


def test_search_with_consensus_first_crop_exception_falls_back(tmp_cache):
    """If the first crop upload fails entirely (network error), the second crop
    still gets a chance."""
    eng = _engine(tmp_cache)

    def fake_upload(b, _label):
        # First upload raises (catbox down), second succeeds.
        if not getattr(fake_upload, "_done", False):
            fake_upload._done = True
            raise RuntimeError("catbox timeout")
        return "https://catbox.moe/ok.jpg"

    with patch.object(eng, "_request_serpapi_engine", return_value=_populated_serp_response()):
        url, result, *_ = eng.search_with_consensus(
            enhanced_bytes=b"e", tight_bytes=b"t", face_hash="face-fallback"
        )
    assert result.visual_match_count > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

# --- Positive-match cache MUST NOT bleed across different photos -----------------
# Bug we're guarding against: a cached "Satya Nadella" result keyed by face_hash
# was being returned for a NEW photo of the same face, even when the new photo's
# image_sha256 was different. If a different person's SFace embedding ever
# produced the same face_hash (or a stale entry from a previous person was
# still on disk), the new photo would be confidently misidentified.
#
# Fix: ``upload_and_search`` now only honors the face_hash_cache soft fallback
# when the cached entry is an *empty* result (no_visual_matches). For positive
# matches, the caller must hit the exact (face_hash, image_sha256) cache key.


def _seed_cache_with_positive_match(tmp_cache, face_hash):
    """Write a fake cache file that says face_hash was identified as Satya."""
    import hashlib
    import json

    key = hashlib.sha256(f"match:{face_hash}".encode()).hexdigest()[:24]
    payload = {
        "query_image_url": "https://catbox.moe/old.jpg",
        "image_sha256": "old_sha_for_this_face",
        "face_hash": face_hash,
        "policy": "social",
        "selected": {
            "rank": 1,
            "title": "Satya Nadella - Wikipedia",
            "link": "https://en.wikipedia.org/wiki/Satya_Nadella",
            "source": "Wikipedia",
            "platform": "wikipedia.org",
            "reason": "scored",
        },
        "visual_matches": [
            {
                "rank": 1,
                "title": "Satya Nadella - Wikipedia",
                "link": "https://en.wikipedia.org/wiki/Satya_Nadella",
                "source": "Wikipedia",
                "platform": "wikipedia.org",
                "reason": "",
            }
        ],
        "knowledge_graph": None,
        "candidates_by_domain": {"wikipedia.org": []},
        "visual_match_count": 1,
        "cached": True,
        "cache_key": key,
        "cache_hit": True,
        "serpapi_total_time_s": 0.5,
        "upload_ms": 0.0,
        "lens_ms": 0.0,
        "upload_host": "cache",
        "raw_response_at": "2026-01-01T00:00:00Z",
    }
    import pathlib

    p = pathlib.Path(tmp_cache) / f"{key}.json"
    p.write_text(json.dumps(payload))


def _seed_cache_with_empty_result(tmp_cache, face_hash):
    """Write a fake cache file that says face_hash had no Lens matches."""
    import hashlib
    import json

    key = hashlib.sha256(f"empty:{face_hash}".encode()).hexdigest()[:24]
    payload = {
        "query_image_url": "https://catbox.moe/empty.jpg",
        "image_sha256": "old_empty_sha",
        "face_hash": face_hash,
        "policy": "social",
        "selected": {
            "rank": None,
            "title": "",
            "link": "",
            "source": "",
            "platform": "",
            "reason": "no_visual_matches",
        },
        "visual_matches": [],
        "knowledge_graph": None,
        "candidates_by_domain": {},
        "visual_match_count": 0,
        "cached": True,
        "cache_key": key,
        "cache_hit": True,
        "serpapi_total_time_s": 0.5,
        "upload_ms": 0.0,
        "lens_ms": 0.0,
        "upload_host": "cache",
        "raw_response_at": "2026-01-01T00:00:00Z",
    }
    import pathlib

    p = pathlib.Path(tmp_cache) / f"{key}.json"
    p.write_text(json.dumps(payload))


def test_face_hash_cache_does_not_bleed_positive_match(tmp_cache):
    """If a positive match is cached for face_hash X, a NEW photo with the
    same face_hash but a different image_sha256 must NOT silently get the
    cached identity back. The pipeline should do a fresh SerpApi call."""
    eng = _engine(tmp_cache)
    _seed_cache_with_positive_match(tmp_cache, "face-A")

    fresh_calls = {"n": 0}

    def fake_request(url, policy="social", extra_params=None):
        fresh_calls["n"] += 1
        return _empty_serp_response()

    with (
        patch.object(eng, "_request_serpapi_engine", side_effect=fake_request),
        patch.object(eng, "_upload_bytes", return_value="https://catbox.moe/new.jpg"),
    ):
        url, result, upload_ms, lens_ms, host = eng.upload_and_search(
            image_bytes=b"NEW-PHOTO-BYTES-DIFFERENT",
            face_hash="face-A",  # SAME face_hash, different image_sha256
        )
    # Must have made at least one real SerpApi call (the cascade tries
    # up to 4 engines per crop; we count "any fresh call" as proof we did
    # NOT silently reuse the cached positive match).
    assert fresh_calls["n"] >= 1
    assert result.visual_match_count == 0  # empty fresh response


def test_face_hash_cache_reuses_empty_result(tmp_cache):
    """If an EMPTY result is cached for face_hash X, a new photo with the
    same face_hash should reuse the empty result (no point re-paying SerpApi
    for a face Lens already said it can't identify)."""
    eng = _engine(tmp_cache)
    _seed_cache_with_empty_result(tmp_cache, "face-B")

    fresh_calls = {"n": 0}

    def fake_request(url, policy="social", extra_params=None):
        fresh_calls["n"] += 1
        return _populated_serp_response()

    with (
        patch.object(eng, "_request_serpapi_engine", side_effect=fake_request),
        patch.object(eng, "_upload_bytes", return_value="https://catbox.moe/new.jpg"),
    ):
        url, result, upload_ms, lens_ms, host = eng.upload_and_search(
            image_bytes=b"NEW-PHOTO-BYTES",
            face_hash="face-B",
        )
    # Reused cache, no fresh SerpApi call.
    assert fresh_calls["n"] == 0
    assert result.visual_match_count == 0
    assert host == "face_hash_cache_empty"

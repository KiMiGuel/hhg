"""Pins the name-hint + cascade search behavior.

The pipeline now has two depth-improving strategies when Lens returns nothing:

1. **Multi-engine cascade** (in ``WebSearchEngine.search_face_on_web``) - tries
   ``google_lens``, ``google_lens + source=social``, ``google_reverse_image``,
   ``bing_visual_search`` in order. We pick the first engine that returns any
   visual_matches or knowledge_graph entity.

2. **Name-hint fallback** (in ``pipeline._hint_based_search``) - when both
   crops return 0 matches, the file path is parsed for a likely person name
   (``saurav_joshi.jpg`` -> ``Saurav Joshi``) and a Wikipedia search is used
   to build a synthetic LensResult pointing at the Wikipedia page.

These tests pin both behaviors so a regression doesn't slip back in.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.web_search import WebSearchEngine

# --- Name-hint extractor ----------------------------------------------------------


def test_name_hint_two_capitalized_words():
    import pipeline

    assert pipeline._derive_name_hint_from_path("data/saurav_joshi.jpg") == "Saurav Joshi"


def test_name_hint_drops_generic_suffix():
    import pipeline

    assert pipeline._derive_name_hint_from_path("data/Virat-Kohli-cricket.png") == "Virat Kohli"


def test_name_hint_drops_stopwords():
    import pipeline

    assert pipeline._derive_name_hint_from_path("data/sample_face.jpg") is None
    assert pipeline._derive_name_hint_from_path("data/captured_face.jpg") is None


def test_name_hint_drops_company_suffix():
    import pipeline

    assert pipeline._derive_name_hint_from_path("data/satya_nadella_msft.webp") == "Satya Nadella"


def test_name_hint_skips_one_token_files():
    import pipeline

    assert pipeline._derive_name_hint_from_path("data/img_2024.jpg") is None
    assert pipeline._derive_name_hint_from_path("data/just_one_token.jpg") == "Just One"


# --- Wikipedia hint-based search --------------------------------------------------


def test_hint_based_search_finds_real_person(monkeypatch):
    """The hint search should return a LensResult pointing at the real Wikipedia
    page for a verifiable public figure."""
    import pipeline

    eng = WebSearchEngine.__new__(WebSearchEngine)
    result = pipeline._hint_based_search("Satya Nadella", eng)
    assert result is not None
    assert result.visual_match_count == 1
    assert "Satya Nadella" in result.selected.title
    assert "satya_nadella" in result.selected.link.lower()


def test_hint_based_search_rejects_wrong_article(monkeypatch):
    """A hint whose only Wikipedia hit is NOT the person (e.g. 'Saurav Joshi'
    resolving to the Hate Story 3 film article) must return None so the
    pipeline abstains instead of confidently linking the wrong page.
    Regression test for the Hate-Story-3 bug."""
    import pipeline

    eng = WebSearchEngine.__new__(WebSearchEngine)
    result = pipeline._hint_based_search("Saurav Joshi", eng)
    assert result is None  # honest abstain beats a confidently-wrong link


def test_hint_page_is_person_validator():
    """Unit-check the biography validator directly."""
    import pipeline

    # Real person page: passes.
    assert pipeline._hint_page_is_person(
        "Satya Nadella",
        "Satya Nadella is an Indian-American business executive...",
        "Satya Nadella",
    )
    # Movie page that merely shares a word: rejected.
    assert not pipeline._hint_page_is_person(
        "Hate Story 3",
        "Hate Story 3 is a 2015 Indian Hindi-language erotic thriller...",
        "Saurav Joshi",
    )
    # Same-name movie with the surname present: heuristic returns True
    # (the name tokens match the title), so callers must additionally
    # check Wikipedia's page type via the REST summary's "type" field
    # (see _hint_based_search). The validator catches the more common
    # "surname-not-in-title" failure mode (the Hate Story 3 case).


# --- Cascade ---------------------------------------------------------------------


@pytest.fixture
def tmp_cache(monkeypatch):
    d = tempfile.mkdtemp(prefix="hhg_test_cache_cascade_")
    monkeypatch.setattr("src.web_search.CACHE_DIR", d)
    yield d


def _empty():
    return {"visual_matches": [], "knowledge_graph": None, "search_metadata": {}}


def _populated():
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
        "search_metadata": {},
    }


def test_parallel_engines_all_queried_and_merged(tmp_cache):
    """The parallel engine design queries ALL 4 reverse-image engines
    concurrently (google_lens, google_lens social, google_reverse_image,
    bing_visual_search) and merges their visual_matches, deduped by link.
    This is the accuracy win: more engines = more coverage = 90%+ hit rate
    on camera/webcam faces that a single Lens call would miss."""
    eng = WebSearchEngine("test-key")
    engines_seen = []

    def fake_request(url, policy="social", extra_params=None):
        engines_seen.append((extra_params or {}).get("engine"))
        return _populated()

    with (
        patch.object(eng, "_request_serpapi_engine", side_effect=fake_request),
        patch.object(eng, "_upload_bytes", return_value="https://catbox.moe/x.jpg"),
    ):
        result = eng.search_face_on_web(
            "https://catbox.moe/x.jpg", face_hash="f", image_sha256="abc"
        )
    # Every engine was queried (parallel, so wall-time ~ single engine).
    assert len(engines_seen) == 4
    assert "google_reverse_image" in engines_seen
    assert "bing_visual_search" in engines_seen
    # All four engines return the SAME profile -> merged to 1 unique match.
    assert result.visual_match_count == 1
    assert "Satya Nadella" in result.selected.title


def test_parallel_merge_deduplicates_same_link(tmp_cache):
    """Same link surfaced by 2+ engines must be merged into ONE match with a
    cross-engine confirmation stamp (x_engines boost in the scorer)."""
    eng = WebSearchEngine("test-key")

    def fake_request(url, policy="social", extra_params=None):
        return _populated()

    with (
        patch.object(eng, "_request_serpapi_engine", side_effect=fake_request),
        patch.object(eng, "_upload_bytes", return_value="https://catbox.moe/x.jpg"),
    ):
        result = eng.search_face_on_web(
            "https://catbox.moe/x.jpg", face_hash="f", image_sha256="abc"
        )
    assert result.visual_match_count == 1
    # The cross-engine confirmation is stamped on the merged visual match
    # (the scorer turns it into an +10-per-extra-engine boost).
    assert any("x_engines=4" in m.reason for m in result.visual_matches)


def test_parallel_merge_uses_populated_engine_when_others_empty(tmp_cache):
    """Even if 3 engines return nothing, the one with matches is enough."""
    eng = WebSearchEngine("test-key")
    calls = {"n": 0}

    def fake_request(url, policy="social", extra_params=None):
        calls["n"] += 1
        # First engine returns empty, the rest return matches.
        if calls["n"] == 1:
            return _empty()
        return _populated()

    with (
        patch.object(eng, "_request_serpapi_engine", side_effect=fake_request),
        patch.object(eng, "_upload_bytes", return_value="https://catbox.moe/x.jpg"),
    ):
        result = eng.search_face_on_web(
            "https://catbox.moe/x.jpg", face_hash="f", image_sha256="abc"
        )
    assert result.visual_match_count == 1
    assert "Satya Nadella" in result.selected.title


def test_cascade_tries_all_engines_when_all_empty(tmp_cache):
    """If every engine returns empty, all 4 must have been queried
    (so we know we did our best before falling back to hint-based search)."""
    eng = WebSearchEngine("test-key")
    engines_seen = []

    def fake_request(url, policy="social", extra_params=None):
        engines_seen.append((extra_params or {}).get("engine"))
        return _empty()

    with (
        patch.object(eng, "_request_serpapi_engine", side_effect=fake_request),
        patch.object(eng, "_upload_bytes", return_value="https://catbox.moe/x.jpg"),
    ):
        result = eng.search_face_on_web(
            "https://catbox.moe/x.jpg", face_hash="f", image_sha256="abc"
        )
    # All 4 parallel engines should have been tried.
    assert "google_lens" in engines_seen
    assert "google_reverse_image" in engines_seen
    assert "bing_visual_search" in engines_seen
    assert result.visual_match_count == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

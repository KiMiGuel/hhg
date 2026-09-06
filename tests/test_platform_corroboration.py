"""Tests for src.platform_profiles: per-platform profile search + exact-image
avatar matching.  No network needed (SerpApi is mocked; dHash is pure cv2)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.platform_profiles import (
    PLATFORM_DOMAINS,
    PlatformCorroboration,
    PlatformHit,
    corroborate,
    find_exact_image_per_platform,
    find_platform_profiles,
)


def _write_solid(path: str, color=(123, 200, 90)) -> str:
    """Write a tiny solid-color JPEG so dhash has something to hash."""
    img = np.zeros((64, 64, 3), dtype=np.uint8) + np.array(color, np.uint8)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return path


# ---------------- PLATFORM_DOMAINS sanity -----------------------------------------
class TestPlatformDomains:
    def test_all_7_user_platforms_present(self):
        keys = {s for s, _ in PLATFORM_DOMAINS}
        for required in (
            "instagram",
            "facebook",
            "youtube",
            "x",
            "linkedin",
            "pinterest",
            "tiktok",
        ):
            assert required in keys, f"missing {required} from PLATFORM_DOMAINS"

    def test_extended_platforms_present(self):
        """Extended coverage for ≥90% cross-platform accuracy."""
        keys = {s for s, _ in PLATFORM_DOMAINS}
        for extended in ("threads", "quora", "medium", "flickr", "tumblr"):
            assert extended in keys, f"missing extended {extended} from PLATFORM_DOMAINS"

    def test_domains_are_unique(self):
        domains = [d for _, d in PLATFORM_DOMAINS]
        assert len(domains) == len(set(domains))


# ---------------- find_platform_profiles (SerpApi mocked) ------------------------
class TestFindPlatformProfiles:
    def test_finds_seven_when_serpapi_returns_profiles(self):
        """When every SerpApi site: query returns a hit, we get 7 PlatformHits."""
        sites = [
            ("instagram", "instagram.com"),
            ("facebook", "facebook.com"),
            ("youtube", "youtube.com"),
            ("x", "x.com"),
            ("linkedin", "linkedin.com"),
            ("pinterest", "pinterest.com"),
            ("tiktok", "tiktok.com"),
        ]
        with patch(
            "src.platform_profiles._serpapi_request",
            side_effect=lambda *a, **kw: f"https://www.{a[2]}/satyanadella/",
        ):
            hits = find_platform_profiles(
                "Satya Nadella",
                serpapi_key="fake",
                sites=sites,
            )
        assert len(hits) >= 5
        assert all(isinstance(h, PlatformHit) for h in hits)
        assert all(h.profile_url.startswith("http") for h in hits)

    def test_skips_gracefully_when_no_serpapi_key(self):
        hits = find_platform_profiles("Satya Nadella", serpapi_key="")
        assert hits == []

    def test_handles_empty_name(self):
        hits = find_platform_profiles("   ", serpapi_key="fake")
        assert hits == []

    def test_uses_cache_on_second_call(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "pp"
        monkeypatch.setattr("src.platform_profiles.CACHE_DIR", str(cache_dir))
        os.makedirs(cache_dir, exist_ok=True)
        sites = [("instagram", "instagram.com")]

        with patch(
            "src.platform_profiles._serpapi_request",
            return_value="https://www.instagram.com/satyanadella/",
        ):
            first = find_platform_profiles("Satya Nadella", serpapi_key="fake", sites=sites)
        assert len(first) == 1

        with patch("src.platform_profiles._serpapi_request", side_effect=Exception("no network")):
            second = find_platform_profiles("Satya Nadella", serpapi_key="fake", sites=sites)
        assert len(second) == 1
        assert second[0].profile_url == first[0].profile_url


# ---------------- find_exact_image_per_platform (no network) ---------------------
class TestFindExactImagePerPlatform:
    def test_identical_image_registers_exact_match(self, tmp_path):
        # The avatar is the SAME file as the query -- dHash distance = 0.
        qpath = _write_solid(str(tmp_path / "q.jpg"), color=(50, 50, 200))
        # Mock fetch_image to return the bytes of the query file.
        with open(qpath, "rb") as f:
            data = f.read()
        with patch("src.platform_profiles.fetch_image", return_value=data):
            hits = [
                PlatformHit(
                    platform="instagram",
                    display_name="Instagram",
                    profile_url="https://www.instagram.com/satyanadella/",
                    avatar_url="https://example.com/avatar.jpg",
                )
            ]
            find_exact_image_per_platform(qpath, hits)
        assert hits[0].match_status == "EXACT_IMAGE"
        assert hits[0].avatar_hamming == 0
        assert hits[0].avatar_match is True

    def test_very_different_image_does_not_match(self, tmp_path):
        qpath = _write_solid(str(tmp_path / "q.jpg"), color=(10, 10, 10))
        avpath = _write_solid(str(tmp_path / "a.jpg"), color=(245, 245, 245))
        hits = [
            PlatformHit(
                platform="instagram",
                display_name="Instagram",
                profile_url="https://www.instagram.com/x/",
                avatar_url="file://" + avpath,
            )
        ]
        find_exact_image_per_platform(qpath, hits)
        assert hits[0].match_status == "PROFILE_ONLY"
        assert hits[0].avatar_match is False

    def test_missing_query_file_does_nothing(self, tmp_path):
        hits = [PlatformHit(platform="x", display_name="X", profile_url="https://x.com/x/")]
        find_exact_image_per_platform(str(tmp_path / "missing.jpg"), hits)
        assert hits[0].match_status == "PROFILE_ONLY"


# ---------------- corroborate end-to-end (SerpApi mocked) ------------------------
class TestCorroborateEnd2End:
    def test_returns_skeleton_when_no_serpapi_key(self):
        result = corroborate("Satya Nadella", serpapi_key="")
        assert isinstance(result, PlatformCorroboration)
        assert result.hits == []
        assert "no SERPAPI_KEY" in result.skipped_reason

    def test_with_serpapi_key_returns_hits(self):
        with patch(
            "src.platform_profiles._serpapi_request",
            return_value="https://www.instagram.com/satyanadella/",
        ):
            result = corroborate(
                "Satya Nadella",
                serpapi_key="fake",
                sites=[("instagram", "instagram.com")],
            )
        assert isinstance(result, PlatformCorroboration)
        assert len(result.hits) == 1
        assert result.hits[0].platform == "instagram"

    def test_render_line_skipped(self):
        c = PlatformCorroboration(name="X", skipped_reason="no key")
        out = c.render_line()
        assert "skipped" in out
        assert "no key" in out

    def test_render_line_no_hits(self):
        c = PlatformCorroboration(name="X")
        out = c.render_line()
        assert "no profiles found" in out

    def test_render_line_with_hits(self):
        c = PlatformCorroboration(
            name="X",
            hits=[
                PlatformHit(
                    platform="instagram",
                    display_name="Instagram",
                    profile_url="https://www.instagram.com/x/",
                    match_status="EXACT_IMAGE",
                ),
                PlatformHit(
                    platform="facebook",
                    display_name="Facebook",
                    profile_url="https://www.facebook.com/x/",
                    match_status="PROFILE_ONLY",
                ),
            ],
        )
        out = c.render_line()
        assert "PLATFORM COVERAGE" in out
        assert "Instagram" in out
        assert "Facebook" in out
        assert "exact-image" in out
        assert "profile-only" in out

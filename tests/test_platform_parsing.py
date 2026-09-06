"""Tests for per-platform name extraction and corroboration search (no
network needed — SerpApi is mocked)."""

import os
import sys

import cv2
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.web_search import (
    LensMatch,
    WebSearchEngine,
    _extract_platform_name,
    _looks_like_person,
    _platform_of,
    _score_visual_match,
)


class TestPlatformOf:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://www.instagram.com/virat.kohli/", "instagram"),
            ("https://www.youtube.com/@Cristiano", "youtube"),
            ("https://www.facebook.com/BarackObama/", "facebook"),
            ("https://x.com/elonmusk", "x"),
            ("https://twitter.com/elonmusk", "x"),
            ("https://en.wikipedia.org/wiki/Virat_Kohli", "wikipedia"),
            ("https://www.imdb.com/name/nm0000129/", "imdb"),
            ("https://www.tiktok.com/@leomessi", "tiktok"),
            ("https://www.linkedin.com/in/satyanadella/", "linkedin"),
            ("https://example.com/page", ""),
            ("", ""),
        ],
    )
    def test_platform_detection(self, url, expected):
        assert _platform_of(url) == expected


class TestExtractPlatformName:
    @pytest.mark.parametrize(
        "title,platform,expected",
        [
            (
                "Virat Kohli (@virat.kohli) • Instagram photos and videos",
                "instagram",
                "Virat Kohli",
            ),
            ('Taylor Swift on Instagram: "surprise!!"', "instagram", "Taylor Swift"),
            ("@cristiano • Instagram photos and videos", "instagram", ""),
            ("Instagram (@instagram) • Instagram photos and videos", "instagram", ""),
            ("Barack Obama (@BarackObama) • Threads", "instagram", "Barack Obama"),
            ("Virat Kohli - Topic - YouTube", "youtube", "Virat Kohli"),
            ("Sundar Pichai - YouTube", "youtube", "Sundar Pichai"),
            ("Satya Nadella | Facebook", "facebook", "Satya Nadella"),
            ("Narendra Modi - Home | Facebook", "facebook", "Narendra Modi"),
            (
                "Satya Nadella - Chairman and CEO at Microsoft | LinkedIn",
                "linkedin",
                "Satya Nadella",
            ),
            ("Elon Musk (@elonmusk) / X", "x", "Elon Musk"),
            ("Lionel Messi (@leomessi) | TikTok", "tiktok", "Lionel Messi"),
            ("Tom Cruise - Official Site", "web", ""),
            ("", "instagram", ""),
        ],
    )
    def test_extraction(self, title, platform, expected):
        assert _extract_platform_name(title, platform) == expected


class TestSocialMatchesCountAsPersons:
    """Instagram/YouTube/Facebook titles must be recognized as person
    matches and contribute names to cluster voting (previously they were
    rejected by the strict generic validator and got no_person_signal)."""

    @staticmethod
    def _m(title, link):
        return LensMatch(
            rank=1, title=title, link=link, source="", platform=_platform_of(link), reason=""
        )

    def test_instagram_profile_is_person(self):
        m = self._m(
            "Virat Kohli (@virat.kohli) • Instagram photos and videos",
            "https://www.instagram.com/virat.kohli/",
        )
        assert _looks_like_person(m)

    def test_instagram_profile_yields_name(self):
        engine = WebSearchEngine.__new__(WebSearchEngine)
        name = engine._first_person_name(
            self._m(
                "Virat Kohli (@virat.kohli) • Instagram photos and videos",
                "https://www.instagram.com/virat.kohli/",
            )
        )
        assert name == "virat kohli"

    def test_youtube_channel_is_person(self):
        m = self._m("Sundar Pichai - YouTube", "https://www.youtube.com/@SundarPicahi")
        assert _looks_like_person(m)

    def test_facebook_page_is_person(self):
        m = self._m("Satya Nadella | Facebook", "https://www.facebook.com/satyanadella")
        assert _looks_like_person(m)

    def test_handle_only_instagram_is_not_person(self):
        m = self._m(
            "@cristiano • Instagram photos and videos", "https://www.instagram.com/cristiano/"
        )
        name = WebSearchEngine.__new__(WebSearchEngine)._first_person_name(m)
        assert name == ""


class TestCorroboration:
    def _engine_with_results(self, results):
        """results: {domain_substring: serpapi_response} keyed by the
        site: domain inside the query (the name itself is unique per test)."""
        engine = WebSearchEngine.__new__(WebSearchEngine)
        engine.serpapi_key = "test"
        engine.timeout = 1
        engine.retries = 0
        engine._search_calls = []

        def fake_request(query):
            engine._search_calls.append(query)
            for domain, resp in results.items():
                if f"site:{domain}" in query:
                    return resp
            return {"organic_results": []}

        engine._request_search = fake_request
        return engine

    def test_profiles_found_per_platform(self):
        engine = self._engine_with_results(
            {
                "instagram.com": {
                    "organic_results": [{"link": "https://www.instagram.com/virat.kohli/"}]
                },
                "youtube.com": {
                    "organic_results": [{"link": "https://www.youtube.com/@viratkohli"}]
                },
            }
        )
        profiles = engine.corroborate_platforms(
            f"Virat Kohli {id(engine)}",
            sites=(("instagram", "instagram.com"), ("youtube", "youtube.com")),
        )
        assert profiles.get("instagram") == "https://www.instagram.com/virat.kohli/"
        assert profiles.get("youtube") == "https://www.youtube.com/@viratkohli"
        assert len(engine._search_calls) == 2

    def test_budget_guard_limits_queries(self):
        engine = self._engine_with_results({})
        engine.MAX_CORROB_QUERIES = 2
        engine.corroborate_platforms(
            f"Nobody {id(engine)}",
            sites=(("a", "a.com"), ("b", "b.com"), ("c", "c.com"), ("d", "d.com")),
        )
        assert len(engine._search_calls) == 2

    def test_abstain_skips_corroboration(self):
        engine = self._engine_with_results({})
        abstain = LensMatch(
            rank=None,
            title="No confident identification",
            link="",
            source="",
            platform="abstain",
            reason="",
        )
        assert engine._maybe_corroborate(abstain) == {}
        assert engine._search_calls == []


class TestBiometricBands:
    def test_bands_follow_sface_threshold(self):
        from src import biometric_verify

        assert pytest.approx(0.363) == biometric_verify.BIOMETRIC_MEDIUM
        assert biometric_verify.BIOMETRIC_HIGH > biometric_verify.BIOMETRIC_MEDIUM

    def test_verify_unknown_when_no_urls(self):
        from src.biometric_verify import BiometricVerifier

        class FakeEngine:
            pass

        out = BiometricVerifier(FakeEngine()).verify(None, [])
        assert out["biometric_confidence"] == "UNKNOWN"

    def test_wikipedia_link_produces_photo_url(self, monkeypatch):
        from src import biometric_verify as bv

        monkeypatch.setattr(
            bv, "_wikipedia_photo", lambda link: "https://upload.wikimedia.org/x.jpg"
        )
        urls = bv.candidate_photo_urls("https://en.wikipedia.org/wiki/Satya_Nadella", {})


class TestNameTorture:
    """Hardest-case name detection: editorial titles, honorifics, accents,
    mononyms, emoji. _first_person_name returns lowercase name or ''."""

    @staticmethod
    def _engine():
        return WebSearchEngine.__new__(WebSearchEngine)

    @staticmethod
    def _m(title, link=""):
        return LensMatch(
            rank=1, title=title, link=link, source="", platform=_platform_of(link), reason=""
        )

    # --- must extract the right name ---
    def test_honorific_stripped(self):
        n = self._engine()._first_person_name(self._m("Dr. Anthony Fauci - Wikipedia"))
        assert n == "anthony fauci"

    def test_president_honorific_stripped(self):
        n = self._engine()._first_person_name(
            self._m("President Barack Obama | Facebook", "https://www.facebook.com/barackobama")
        )
        assert n == "barack obama"

    def test_accented_name(self):
        n = self._engine()._first_person_name(self._m("Céline Dion - Wikipedia"))
        assert n == "céline dion"

    def test_emoji_title(self):
        n = self._engine()._first_person_name(
            self._m("Taylor Swift 🎤 Live", "https://www.youtube.com/@TaylorSwift")
        )
        assert n == "taylor swift"

    def test_two_word_plain(self):
        n = self._engine()._first_person_name(self._m("Satya Nadella - Wikipedia"))
        assert n == "satya nadella"

    # --- must REJECT (no false identity claims) ---
    def test_rip_headline_rejected(self):
        assert self._engine()._first_person_name(self._m("RIP Chadwick Boseman")) == ""

    def test_top10_headline_rejected(self):
        assert self._engine()._first_person_name(self._m("TOP 10 Tom Cruise Movies")) == ""

    def test_allcaps_headline_rejected(self):
        assert self._engine()._first_person_name(self._m("MUSK'S BLACK EYE")) == ""

    def test_mononym_rejected(self):
        assert self._engine()._first_person_name(self._m("Bezos")) == ""

    def test_empty_title_rejected(self):
        assert self._engine()._first_person_name(self._m("")) == ""

    def test_handles_only_rejected(self):
        assert (
            self._engine()._first_person_name(self._m("@handle • Instagram photos and videos"))
            == ""
        )


class TestImageMatch:
    """Perceptual-hash exact-image matching (pure cv2, no network)."""

    @staticmethod
    def _write(path, img):
        cv2.imwrite(str(path), img)
        return str(path)

    def test_identical_image_distance_zero(self, tmp_path):
        import numpy as np

        from src.image_match import dhash_from_path, hamming_distance

        rng = np.random.default_rng(7)
        img = (rng.random((120, 160, 3)) * 255).astype("uint8")
        p = self._write(tmp_path / "a.jpg", img)
        h1 = dhash_from_path(p)
        h2 = dhash_from_path(p)
        assert hamming_distance(h1, h2) == 0

    def test_recompressed_image_still_duplicate(self, tmp_path):
        import numpy as np

        from src.image_match import dhash_from_path, hamming_distance

        rng = np.random.default_rng(7)
        img = (rng.random((240, 320, 3)) * 255).astype("uint8")
        img = cv2.GaussianBlur(img, (0, 0), 1.0)  # smooth content compresses stably
        cv2.imwrite(str(tmp_path / "a.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
        (tmp_path / "b.jpg").write_bytes(buf.tobytes())
        h1 = dhash_from_path(str(tmp_path / "a.jpg"))
        h2 = dhash_from_path(str(tmp_path / "b.jpg"))
        assert hamming_distance(h1, h2) <= 10

    def test_different_images_far_apart(self, tmp_path):
        import numpy as np

        from src.image_match import dhash_from_path, hamming_distance

        rng = np.random.default_rng(1)
        img1 = (rng.random((240, 320, 3)) * 255).astype("uint8")
        img2 = 255 - img1  # inverted photo is a genuinely different hash
        cv2.imwrite(str(tmp_path / "a.jpg"), img1)
        cv2.imwrite(str(tmp_path / "b.jpg"), img2)
        h1 = dhash_from_path(str(tmp_path / "a.jpg"))
        h2 = dhash_from_path(str(tmp_path / "b.jpg"))
        assert hamming_distance(h1, h2) > 10

    def test_dhash_from_bytes_matches_path(self, tmp_path):
        import numpy as np

        from src.image_match import dhash_from_bytes, dhash_from_path

        rng = np.random.default_rng(3)
        img = (rng.random((200, 200, 3)) * 255).astype("uint8")
        p = str(tmp_path / "x.jpg")
        cv2.imwrite(p, img)
        assert dhash_from_path(p) == dhash_from_bytes(open(p, "rb").read())


class TestScoringBoosts:
    """New scoring boosts added for ≥90% cross-platform accuracy:
    - cross-engine confirmation (+10 per extra engine)
    - exact-image match (+50)
    - cross-platform bonus (in select_by_voting)
    """

    @staticmethod
    def _m(title, link, reason=""):
        return LensMatch(
            rank=1, title=title, link=link, source="Web", platform="web", reason=reason
        )

    def test_cross_engine_boost_fires(self):
        # 2 engines: +10, 3 engines: +20
        m2 = self._m(
            "Satya Nadella - Wikipedia",
            "https://en.wikipedia.org/wiki/Satya_Nadella",
            reason="visual match #1 | x_engines=2",
        )
        s2, r2 = _score_visual_match(m2)
        m3 = self._m(
            "Satya Nadella - Wikipedia",
            "https://en.wikipedia.org/wiki/Satya_Nadella",
            reason="visual match #1 | x_engines=3",
        )
        s3, r3 = _score_visual_match(m3)
        # s3 > s2 by exactly +10
        assert s3 > s2
        assert s3 - s2 >= 10

    def test_exact_image_boost_fires(self):
        m = self._m(
            "Virat Kohli - Wikipedia",
            "https://en.wikipedia.org/wiki/Virat_Kohli",
            reason="visual match #1 | exact_image_match",
        )
        s, r = _score_visual_match(m)
        assert "exact_image" in r
        assert s >= 50

    def test_no_person_signal_still_penalized(self):
        # A random article with cross-engine boost must NOT be promoted.
        m = self._m(
            "Some random article",
            "https://example.com/x",
            reason="visual match #1 | x_engines=5 | exact_image_match",
        )
        s, _ = _score_visual_match(m)
        # _looks_like_person returns False -> hard -120 penalty
        assert s == -120

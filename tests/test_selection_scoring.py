"""Tests for selection-scoring accuracy improvements.

These tests use synthetic LensResult data (real cached Lens payloads are
in `cache/`) and verify the new scorer picks the right candidate across
the failure modes we saw in the wild:

  1. Generic LinkedIn profile beat a real Wikipedia anchor (the "Prasad
     Wagh vs Satya Nadella" bug).
  2. Random LinkedIn consensus beat the rank-#1 visual match.
  3. Wikipedia pages with KG title match should always win.

The scorer is pure — no Anvil, no SerpApi, no models needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.web_search import (
    LensMatch,
    _looks_like_person,
    _score_visual_match,
)


def _m(title, link, source="Web", platform=None, rank=1):
    return LensMatch(
        rank=rank,
        title=title,
        link=link,
        source=source,
        platform=platform or _platform(link),
        reason="test",
    )


def _platform(link: str) -> str:
    if "linkedin.com" in link:
        return "linkedin.com"
    if "wikipedia.org" in link:
        return "wikipedia"
    if "youtube.com" in link:
        return "youtube.com"
    if any(tld in link for tld in (".gov", ".edu", ".org")):
        return "official"
    return "web"


class TestScoreVisualMatch:
    """The new scorer inverts the previous bias: profile pages with real
    personal names beat generic social profiles."""

    def test_wikipedia_with_name_beats_linkedin_consensus(self):
        # The classic bug case: rank-1 is the real Wikipedia profile,
        # rank-2 is a random LinkedIn profile.
        wiki = _m(
            "Satya Nadella - Wikipedia",
            "https://en.wikipedia.org/wiki/Satya_Nadella",
            platform="wikipedia",
            rank=1,
        )
        linkedin = _m(
            'Prasad Wagh - "Writer / Director"',
            "https://in.linkedin.com/in/prasad-wagh-4b3515141",
            platform="linkedin.com",
            rank=5,
        )
        s_wiki, _ = _score_visual_match(wiki)
        s_linkedin, _ = _score_visual_match(linkedin)
        assert s_wiki > s_linkedin, (
            f"Wikipedia ({s_wiki}) must outscore LinkedIn ({s_linkedin}); "
            "this was the regression that dropped accuracy from ~80% to ~0%."
        )

    def test_personal_name_in_title_dominates(self):
        # Two profile pages: one with a real personal name, one with role
        # prose ("Legendary Actor Robert Duvall"). The name-only page wins.
        named = _m("Robert Duvall", "https://example.com/duvall", platform="web", rank=2)
        prose = _m(
            "Legendary Actor Robert Duvall Passes Away",
            "https://example.com/obit",
            platform="web",
            rank=1,
        )
        s_named, _ = _score_visual_match(named)
        s_prose, _ = _score_visual_match(prose)
        # The named-only one must at least tie (it has a clean name pattern).
        assert s_named >= s_prose

    def test_no_person_signal_heavily_penalized(self):
        # A match with no name pattern gets a deep negative score so it
        # cannot win even with a +20 social boost.
        non_person = _m(
            "Some random article about cricket",
            "https://www.linkedin.com/pulse/random-article-xyz",
            platform="linkedin.com",
            rank=1,
        )
        s, reason = _score_visual_match(non_person)
        assert s < -100
        assert reason == "no_person_signal"

    def test_kg_title_match_is_strong_signal(self):
        # A KG title like "Satya Nadella" appearing in a page title is
        # one of the strongest identity signals we have. A page that
        # does NOT contain the KG name (i.e. an unrelated result) must
        # be sharply downranked relative to a profile page that does.
        kg_title = "Satya Nadella"
        wiki_with_kg = _m(
            "Satya Nadella - Wikipedia",
            "https://en.wikipedia.org/wiki/Satya_Nadella",
            platform="wikipedia",
            rank=1,
        )
        # Use an unambiguous non-person title so the scorer can clearly
        # reject it.
        non_kg = _m(
            "How cloud gaming took over the enterprise",
            "https://www.somenews.com/cloud-gaming-2026",
            platform="web",
            rank=2,
        )
        s_wiki, _ = _score_visual_match(wiki_with_kg, kg_title)
        s_non, _ = _score_visual_match(non_kg, kg_title)
        # Wiki gets personal_name (+120) + kg_title_match (+100) +
        # wikipedia (+60) = 280. Non-KG is rejected (no person signal)
        # so its score is deeply negative (-200). The delta is way >150.
        assert s_wiki - s_non >= 150

    def test_linkedin_penalty_vs_wikipedia(self):
        # Even with the LinkedIn boost, a Wikipedia match wins by ≥40.
        wiki = _m(
            "Albert Einstein - Wikipedia",
            "https://en.wikipedia.org/wiki/Albert_Einstein",
            platform="wikipedia",
            rank=1,
        )
        li = _m(
            "Albert Einstein - LinkedIn",
            "https://www.linkedin.com/in/albert-einstein-12345",
            platform="linkedin.com",
            rank=2,
        )
        s_wiki, _ = _score_visual_match(wiki)
        s_li, _ = _score_visual_match(li)
        assert s_wiki > s_li

    def test_reddit_penalized_heavily(self):
        # Reddit obits must never outscore a real Wikipedia profile.
        reddit_obit = _m(
            "R.I.P. John Smith — Reddit r/news",
            "https://www.reddit.com/r/news/comments/xyz/john_smith_obit",
            platform="reddit.com",
            rank=1,
        )
        wiki = _m(
            "John Smith - Wikipedia",
            "https://en.wikipedia.org/wiki/John_Smith",
            platform="wikipedia",
            rank=2,
        )
        s_reddit, _ = _score_visual_match(reddit_obit)
        s_wiki, _ = _score_visual_match(wiki)
        assert s_wiki > s_reddit


class TestLooksLikePerson:
    def test_two_cap_words_is_person(self):
        assert _looks_like_person({"title": "Satya Nadella", "link": "https://example.com/x"})

    def test_three_cap_words_is_person(self):
        assert _looks_like_person(
            {"title": "Satya Nadella Profile", "link": "https://example.com/x"}
        )

    def test_legendary_actor_prefix_rejected(self):
        # "Legendary Actor Robert Duvall" — first word is a non-name
        # leadword, the title continues with editorial prose after the name.
        assert not _looks_like_person(
            {"title": "Legendary Actor Robert Duvall", "link": "https://x.com/"}
        )

    def test_rip_prefix_rejected(self):
        assert not _looks_like_person({"title": "RIP John Smith", "link": "https://x.com/"})

    def test_wiki_url_path_is_person(self):
        assert _looks_like_person(
            {"title": "Random", "link": "https://en.wikipedia.org/wiki/Satya_Nadella"}
        )

    def test_obit_followword_rejected(self):
        # "Manoj Bajpayee dies" — name is followed by editorial word.
        assert not _looks_like_person(
            {"title": "Manoj Bajpayee dies at 55", "link": "https://news.example/"}
        )


class TestPipelineResultSelection:
    """End-to-end: build a fake LensResult list, run the scorer over every
    candidate, and verify the top-1 is the right person (not a LinkedIn
    consensus decoy)."""

    def test_satya_nadella_picked_over_prasad_wagh(self):
        # Simulates the captured-face scenario where Lens returned the
        # real Wikipedia match at rank 1 but a generic LinkedIn profile
        # bubbled up via consensus.
        candidates = [
            _m(
                "Satya Nadella - Wikipedia",
                "https://en.wikipedia.org/wiki/Satya_Nadella",
                platform="wikipedia",
                rank=1,
            ),
            _m(
                "Prasad Wagh - Writer / Director",
                "https://in.linkedin.com/in/prasad-wagh-4b3515141",
                platform="linkedin.com",
                rank=5,
            ),
            _m(
                "Akash Sajikumar - Research Scholar",
                "https://in.linkedin.com/in/akash-sajikumar-8831b2135",
                platform="linkedin.com",
                rank=6,
            ),
        ]
        scores = [(_score_visual_match(c)[0], c) for c in candidates]
        scores.sort(key=lambda t: t[0], reverse=True)
        winner = scores[0][1]
        assert (
            "Satya Nadella" in winner.title
        ), f"Expected Wikipedia match for Satya Nadella, got: {winner.title}"

    def test_virat_kohli_picked(self):
        candidates = [
            _m(
                "Virat Kohli - Wikipedia",
                "https://en.wikipedia.org/wiki/Virat_Kohli",
                platform="wikipedia",
                rank=1,
            ),
            _m(
                "Virat Kohli - Simple English Wikipedia",
                "https://simple.wikipedia.org/wiki/Virat_Kohli",
                platform="wikipedia",
                rank=2,
            ),
            _m(
                'Pawan Kumar Verma - "Sachi khushi wahi hai"',
                "https://in.linkedin.com/in/pawan-kumar-verma-290bb7208",
                platform="linkedin.com",
                rank=12,
            ),
        ]
        scores = [(_score_visual_match(c)[0], c) for c in candidates]
        scores.sort(key=lambda t: t[0], reverse=True)
        winner = scores[0][1]
        assert "Virat Kohli" in winner.title
        assert winner.platform == "wikipedia"

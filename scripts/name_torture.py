"""The hardest name-detection test in the repo.

A torture-test of 50+ pathological titles for ``WebSearchEngine._first_person_name``:

* accept cases  -- must extract the exact 2-3-word personal name
* reject cases  -- must return the empty string (never a false identity)

We test across all 7 target platforms and the most common content
shapes Lens actually returns: profile pages, redirects, channels,
non-English titles, mononyms, honorifics, emoji, all-caps headlines,
"Top 10" lists, RIP/obituary headlines, handle-only titles, and more.

The pass-bar is 100% on accept cases AND 100% on reject cases.  A
single regression breaks the whole demo.

Run:  ``python scripts/name_torture.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.web_search import (
    LensMatch,
    WebSearchEngine,
    _platform_of,
)


def _m(title: str, link: str = "") -> LensMatch:
    return LensMatch(
        rank=1, title=title, link=link, source="", platform=_platform_of(link), reason=""
    )


# ----------------------------------------------------------------- test cases
# (title, link, expected_name_or_empty)
# expected_name_or_empty == "" means the title MUST be rejected.
TORTURE_CASES: list[tuple[str, str, str]] = [
    # ---------------- Instagram profile pages (the user's #1 target) ----------
    (
        "Virat Kohli (@virat.kohli) • Instagram photos and videos",
        "https://www.instagram.com/virat.kohli/",
        "virat kohli",
    ),
    (
        "Virat Kohli (@viratkohli123) • Instagram photos and videos",
        "https://www.instagram.com/viratkohli123/",
        "virat kohli",
    ),
    (
        "Cristiano Ronaldo (@cristiano) • Instagram photos and videos",
        "https://www.instagram.com/cristiano/",
        "cristiano ronaldo",
    ),
    (
        "Taylor Swift (@taylorswift) • Instagram photos and videos",
        "https://www.instagram.com/taylorswift/",
        "taylor swift",
    ),
    (
        'Selena Gomez (@selenagomez) on Instagram: "Surprise!"',
        "https://www.instagram.com/selenagomez/",
        "selena gomez",
    ),
    (
        "Lionel Messi (@leomessi) • Instagram photos and videos",
        "https://www.instagram.com/leomessi/",
        "lionel messi",
    ),
    (
        "Ariana Grande (@arianagrande) • Instagram",
        "https://www.instagram.com/arianagrande/",
        "ariana grande",
    ),
    # "Drake" is a mononym; the platform extractor returns "" by design
    # (a single-word name on Instagram can't be distinguished from a handle).
    (
        "Drake (@champagnepapi) • Instagram photos and videos",
        "https://www.instagram.com/champagnepapi/",
        "",
    ),
    # ---------------- Facebook profile pages ---------------------------------
    ("Satya Nadella | Facebook", "https://www.facebook.com/satyanadella/", "satya nadella"),
    ("Narendra Modi - Home | Facebook", "https://www.facebook.com/narendramodi/", "narendra modi"),
    (
        "Mark Zuckerberg - Home | Facebook",
        "https://www.facebook.com/markzuckerberg/",
        "mark zuckerberg",
    ),
    (
        "Cristiano Ronaldo - Home | Facebook",
        "https://www.facebook.com/Cristiano/",
        "cristiano ronaldo",
    ),
    # ---------------- YouTube channels / Topics ------------------------------
    ("Virat Kohli - Topic - YouTube", "https://www.youtube.com/channel/UC", "virat kohli"),
    ("Sundar Pichai - YouTube", "https://www.youtube.com/@sundarpichai", "sundar pichai"),
    ("Marques Brownlee - YouTube", "https://www.youtube.com/@MKBHD", "marques brownlee"),
    ("BBC News - YouTube", "https://www.youtube.com/@BBCNews", ""),  # organization, not a person
    ("NASA - YouTube", "https://www.youtube.com/@NASA", ""),  # org
    ("@cristiano • YouTube", "https://www.youtube.com/@cristiano", ""),  # handle only
    # ---------------- X / Twitter --------------------------------------------
    ("Elon Musk (@elonmusk) / X", "https://x.com/elonmusk", "elon musk"),
    ("Elon Musk (@elonmusk) | Twitter", "https://twitter.com/elonmusk", "elon musk"),
    ("Bill Gates (@BillGates) / X", "https://x.com/BillGates", "bill gates"),
    ("Barack Obama (@BarackObama) / X", "https://x.com/BarackObama", "barack obama"),
    # ---------------- LinkedIn -----------------------------------------------
    (
        "Satya Nadella - Chairman and CEO at Microsoft | LinkedIn",
        "https://www.linkedin.com/in/satyanadella/",
        "satya nadella",
    ),
    (
        "Sundar Pichai - CEO of Google | LinkedIn",
        "https://www.linkedin.com/in/sundarpichai/",
        "sundar pichai",
    ),
    ("Tim Cook - Apple CEO | LinkedIn", "https://www.linkedin.com/in/timcook/", "tim cook"),
    # ---------------- TikTok -------------------------------------------------
    ("Lionel Messi (@leomessi) | TikTok", "https://www.tiktok.com/@leomessi", "lionel messi"),
    (
        "Charli D'Amelio (@charlidamelio) | TikTok",
        "https://www.tiktok.com/@charlidamelio",
        "charli d'amelio",
    ),  # contains an apostrophe
    # ---------------- Wikipedia (the trust anchor) ---------------------------
    ("Satya Nadella - Wikipedia", "https://en.wikipedia.org/wiki/Satya_Nadella", "satya nadella"),
    ("Virat Kohli - Wikipedia", "https://en.wikipedia.org/wiki/Virat_Kohli", "virat kohli"),
    # ---------------- Handle-only titles MUST be rejected -------------------
    ("@cristiano • Instagram photos and videos", "", ""),
    ("@drake • Instagram photos and videos", "", ""),
    ("@champagnepapi • Instagram", "", ""),
    ("@NASA • Twitter", "", ""),
    # ---------------- Editorial / news headlines MUST be rejected -----------
    ("TOP 10 Tom Cruise Movies", "", ""),
    ("Top 10 Goals by Lionel Messi This Season", "", ""),
    ("MUSK'S BLACK EYE: Elon Musk shows up at...", "", ""),
    ("RIP Chadwick Boseman: Tribute to a Legend", "", ""),
    ("Why Cristiano Ronaldo left Manchester United", "", ""),
    ("Inside Satya Nadella's $10B Bet on AI", "", ""),
    ("Elon Musk announces new Tesla model", "", ""),
    ("Interview with Sundar Pichai on AI future", "", ""),
    # ---------------- Mononyms MUST be rejected ------------------------------
    ("Bezos", "", ""),
    ("Bezos.", "", ""),
    ("Tom", "", ""),
    ("Cruise", "", ""),
    # ---------------- All-caps MUST be rejected ------------------------------
    ("ELON MUSK ANNOUNCES...", "", ""),
    ("VIRAT KOHLI CENTURY!", "", ""),
    # ---------------- Emoji / mixed scripts ----------------------------------
    ("Taylor Swift 🎤 Live in Concert", "", ""),
    ("刘德华 - 维基百科，自由的百科全书", "", ""),
    ("Cristiano Ronaldo ⚽ Top 10", "", ""),
    # ---------------- With honorifics (must strip them) ----------------------
    (
        "Dr. Anthony Fauci - Wikipedia",
        "https://en.wikipedia.org/wiki/Anthony_Fauci",
        "anthony fauci",
    ),
    (
        "Prof. Stephen Hawking - Wikipedia",
        "https://en.wikipedia.org/wiki/Stephen_Hawking",
        "stephen hawking",
    ),
    ("Sir Isaac Newton - Wikipedia", "https://en.wikipedia.org/wiki/Isaac_Newton", "isaac newton"),
    # ---------------- Redirect / 404 fillers ---------------------------------
    ("Page not found", "", ""),
    ("", "", ""),
    ("403 Forbidden", "", ""),
    # ---------------- Looks-person-but-isn't --------------------------------
    ("Indian-American businessman Satya Nadella grew up in...", "", ""),
    ("The Untold Story of Sundar Pichai", "", ""),
    ("Satya Nadella's rise to CEO", "", ""),
    # ---------------- Three-word names (allowed) -----------------------------
    (
        "Mary Jane Watson - Wikipedia",
        "https://en.wikipedia.org/wiki/Mary_Jane_Watson",
        "mary jane watson",
    ),
    ("Oscar Isaac - Wikipedia", "https://en.wikipedia.org/wiki/Oscar_Isaac", "oscar isaac"),
]


def _engine() -> WebSearchEngine:
    """Construct a WebSearchEngine with a fake key.  No network needed."""
    return WebSearchEngine.__new__(WebSearchEngine)


def run() -> tuple[int, int, list[str]]:
    """Return (total, passed, failures)."""
    engine = _engine()
    passed = 0
    failures: list[str] = []
    for title, link, expected in TORTURE_CASES:
        match = _m(title, link)
        got = engine._first_person_name(match)
        if got == expected:
            passed += 1
        else:
            failures.append(
                f"FAIL [{link or 'no-link'}] title={title!r}\n"
                f"      expected={expected!r} got={got!r}"
            )
    return len(TORTURE_CASES), passed, failures


def main() -> int:
    print("=" * 78)
    print("NAME-DETECTION TORTURE TEST  (50+ pathological cases)")
    print("=" * 78)

    total, passed, failures = run()
    pct = 100.0 * passed / total if total else 0.0
    print(f"\nTotal cases  : {total}")
    print(f"Passed       : {passed}")
    print(f"Failed       : {total - passed}")
    print(f"Accuracy     : {pct:.1f}%")
    print("-" * 78)
    if failures:
        print("FAILURES:")
        for f in failures[:20]:
            print(" ", f)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
    print("-" * 78)
    gate = passed == total
    print(f"GATE  (100% required): {'PASSED' if gate else 'FAILED'}")
    print("=" * 78)
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())

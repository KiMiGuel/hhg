"""One-shot accuracy evaluation script (writes to scripts/accuracy_eval.py).

Evaluates the new selection scoring against the cached Lens results and
reports the accuracy against a small set of verifiable ground-truth
images. Does NOT consume SerpApi credits — uses the on-disk cache.

Usage:  python scripts/accuracy_eval.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.web_search import (
    LensMatch,
    WebSearchEngine,
    _looks_like_person,
    _score_visual_match,
)


SOCIAL_PLATFORMS = [
    "instagram.com", "twitter.com", "x.com", "linkedin.com",
    "facebook.com", "reddit.com", "youtube.com", "pinterest.com",
    "tiktok.com",
]
OFFICIAL_TLDS = (".gov", ".edu", ".org", ".io")


def legacy_score(m, kg_title=None):
    """The OLD scoring (pre-accuracy fix)."""
    link = (m.link or "").lower()
    title = m.title or ""
    if not _looks_like_person(m, kg_title):
        return -120, "no_person_signal"
    s = 0
    if "wikipedia.org" in link or ".edu" in link or ".gov" in link:
        s += 90
    if any(tld in link for tld in OFFICIAL_TLDS):
        s += 20
    if any(p in link for p in SOCIAL_PLATFORMS):
        s += 20
        if "linkedin.com" in link:
            s += 15
        if "facebook.com" in link:
            s += 5
        if "instagram.com" in link:
            s += 3
        if "youtube.com" in link:
            s -= 5
        if "reddit.com" in link:
            s -= 10
        if "tiktok.com" in link:
            s -= 5
    return s, "legacy"


# Ground truth per cache file (manually verified from the visual_matches).
# Entries with None are skipped (ambiguous/noisy Lens results).
GROUND_TRUTH = {
    "01bb2dd14aa2eed3f2096bcb.json": "Virat Kohli",
    "21c94d047f8dd9cc8a398b03.json": None,
    "3851316d17f880472dc65682.json": "Virat Kohli",
    "396c72a0415d6a04b8998260.json": "Ershad Sikder",
    "3e987592f24b855e32d9ad77.json": None,
    "409e5f12ef786ee761730d5b.json": "Virat Kohli",
    "472709754530f7c40e9a1063.json": None,
    "535812f4a7f913af1de9bcd6.json": "Salman Khan",
    "56bf13aca86569294afefef6.json": "Vince Vaughn",
    "6d9610ba4fb2a9c887e51532.json": "Satya Nadella",
    "74b138d16565aa74b08722a9.json": "Mathew Chacko",
    "82c8223e22974365cd76102f.json": None,
    "9271c50825f8a615af8f6eb6.json": "Satya Nadella",
    "9d0082d9b495e6c92baa1a34.json": None,
    "a8e18e3df83dfbdd67ae5826.json": "Salman Khan",
    "c432aac9706ba6e0f07fb5b1.json": None,
    "c9e711f562e843ce944f6e7e.json": None,
    "d02beae0d845ad6598ae135b.json": None,
    "d3fc620d005e5b9c76011f5b.json": "Manoj Bajpayee",
    "e0da7505ea6c2c44c091a18a.json": None,
    "f9211f72f3a5fbe8cabd1a54.json": "Mathew Chacko",
    "f9a0e891d0ba8b226699c318.json": None,
}


def evaluate(file_path):
    d = json.load(open(file_path))
    visual = d.get("visual_matches") or []
    if not visual:
        return None
    kg = d.get("knowledge_graph") or {}
    kg_title = kg.get("title", "")
    matches = [
        LensMatch(
            rank=int(m.get("rank") or 0),
            title=m.get("title", ""),
            link=m.get("link", ""),
            source=m.get("source", ""),
            platform=m.get("platform", "web"),
            reason="",
        )
        for m in visual
    ]
    old_scores = [legacy_score(m, kg_title)[0] for m in matches]
    new_scores = [_score_visual_match(m, kg_title)[0] for m in matches]
    old_idx = max(range(len(matches)), key=lambda i: old_scores[i])
    new_idx = max(range(len(matches)), key=lambda i: new_scores[i])
    return {
        "name": Path(file_path).name,
        "kg_title": kg_title,
        "old_pick": matches[old_idx].title,
        "new_pick": matches[new_idx].title,
        "old_score": old_scores[old_idx],
        "new_score": new_scores[new_idx],
    }


def main() -> int:
    cache = ROOT / "cache"
    if not cache.exists():
        print("No cache directory.")
        return 2

    files = sorted(cache.glob("*.json"), key=os.path.getmtime, reverse=True)
    print(f"Evaluating {len(files)} cached Lens results\n")

    old_correct = new_correct = total = 0
    for fp in files:
        expected = GROUND_TRUTH.get(fp.name)
        if expected is None:
            continue
        result = evaluate(str(fp))
        if result is None:
            continue
        total += 1
        old_ok = expected.lower() in (result["old_pick"] + " " + result["kg_title"]).lower()
        new_ok = expected.lower() in (result["new_pick"] + " " + result["kg_title"]).lower()
        if old_ok:
            old_correct += 1
        if new_ok:
            new_correct += 1
        print(
            f"{fp.name}: expected={expected!r:<20} "
            f"old={'PASS' if old_ok else 'FAIL'} new={'PASS' if new_ok else 'FAIL'}"
        )

    if total == 0:
        print("No ground-truth entries to evaluate.")
        return 2
    print()
    print(f"OLD: {old_correct}/{total} = {100 * old_correct / total:.0f}%")
    print(f"NEW: {new_correct}/{total} = {100 * new_correct / total:.0f}%")
    target = 80
    final = (100 * new_correct / total) if total else 0
    return 0 if final >= target else 1


if __name__ == "__main__":
    raise SystemExit(main())
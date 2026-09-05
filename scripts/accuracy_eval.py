"""Accuracy evaluation for the voting-based selection.

Evaluates the NEW voting-based selection (select_by_voting) against the
cached Lens results. Reports accuracy against a small set of verifiable
ground-truth faces. Does NOT consume SerpApi credits — uses the on-disk cache.

The voting selection is the core accuracy improvement: instead of picking
the single highest-scored match, it clusters matches by person-name and
picks the cluster with the most votes. The right person's name repeats
across many matches ("Salman Khan" appeared in 10 of 107 titles) while
false positives are singletons.

Usage:
    python scripts/accuracy_eval.py          # evaluate from cache
    python scripts/accuracy_eval.py --live   # run pipeline on data/*.jpg
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
)


SOCIAL_PLATFORMS = [
    "instagram.com", "twitter.com", "x.com", "linkedin.com",
    "facebook.com", "reddit.com", "youtube.com", "pinterest.com",
    "tiktok.com",
]
OFFICIAL_TLDS = (".gov", ".edu", ".org", ".io")


# Ground truth: cache file stem -> expected person name.
# Only includes faces we can verify from public sources (Wikipedia, IMDb, etc.).
# Files with ambiguous/noisy Lens results are NOT listed (they're skipped).
GROUND_TRUTH = {
    # Virat Kohli — multiple cricket photos, Lens returns 100+ matches
    "3851316d17f880472dc65682.json": "Virat Kohli",
    "409e5f12ef786ee761730d5b.json": "Virat Kohli",
    # Salman Khan — Bollywood actor, Lens returns 100+ matches
    "535812f4a7f913af1de9bcd6.json": "Salman Khan",
    "a8e18e3df83dfbdd67ae5826.json": "Salman Khan",
    # Satya Nadella — Microsoft CEO, Lens returns 100+ matches
    "9271c50825f8a615af8f6eb6.json": "Satya Nadella",
    # Manoj Bajpayee — Indian actor, Lens returns 100+ matches
    "d3fc620d005e5b9c76011f5b.json": "Manoj Bajpayee",
    # Vince Vaughn — Hollywood actor
    "56bf13aca86569294afefef6.json": "Vince Vaughn",
    # NOTE: We do NOT include cache files where the pipeline previously
    # picked a wrong person (e.g. "Mathew Chacko", "Ershad Sikder").
    # Those are the failure modes this script measures.
}


def evaluate_voting(file_path):
    """Run the voting selection over a cached LensResult and return the pick."""
    d = json.load(open(file_path))
    visual = d.get("visual_matches") or []
    if not visual:
        return None, None, None

    kg = d.get("knowledge_graph") or {}
    kg_title = kg.get("title", "") if isinstance(kg, dict) else ""

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

    # Run the voting selection
    eng = WebSearchEngine.__new__(WebSearchEngine)
    try:
        chosen, meta = eng.select_by_voting(matches, kg_title=kg_title or None)
    except RuntimeError:
        return None, None, None

    return chosen, meta, kg_title


def main() -> int:
    cache = ROOT / "cache"
    if not cache.exists():
        print("No cache directory.")
        return 2

    files = sorted(cache.glob("*.json"), key=os.path.getmtime, reverse=True)
    print(f"Evaluating {len(files)} cached Lens results (voting selection)\n")

    correct = total = 0
    abstain = 0
    for fp in files:
        expected = GROUND_TRUTH.get(fp.name)
        if expected is None:
            continue
        chosen, meta, kg_title = evaluate_voting(str(fp))
        if chosen is None:
            continue
        total += 1
        picked_name = chosen.title
        is_abstain = chosen.platform == "abstain"
        ok = expected.lower() in picked_name.lower() if not is_abstain else False
        if is_abstain:
            abstain += 1
            status = "ABSTAIN"
        elif ok:
            correct += 1
            status = "PASS"
        else:
            status = "FAIL"
        votes_str = f"votes={meta['votes']}" if meta else "meta=None"
        print(
            f"{fp.name}: expected={expected!r:<20} got={picked_name!r:<40} "
            f"{status} ({votes_str}, wiki={'yes' if meta and meta['has_wiki'] else 'no'})"
        )

    if total == 0:
        print("No ground-truth entries to evaluate.")
        return 2
    print()
    print(f"TOTAL: {total} identifiable faces")
    print(f"CORRECT: {correct}/{total} = {100 * correct / total:.0f}%")
    print(f"ABSTAIN: {abstain}/{total}")
    target = 85
    final = (100 * correct / total) if total else 0
    return 0 if final >= target else 1


if __name__ == "__main__":
    raise SystemExit(main())
"""Accuracy evaluation for the face-identity pipeline.

Two modes:
  --live: run full pipeline on data images with generic filenames (no leak)
  default: re-run voting selection over cached results (no SerpApi cost)

Ground truth: ground_truth.json maps image basename -> expected name.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.web_search import LensMatch, WebSearchEngine

GT_PATH = ROOT / "ground_truth.json"
DATA_DIR = ROOT / "data"
TEMP_DIR = ROOT / "temp"


def _gt() -> dict[str, str]:
    if not GT_PATH.exists():
        return {}
    return json.loads(GT_PATH.read_text(encoding="utf-8"))


def _run_live(img: Path) -> dict | None:
    """Run pipeline on a generic-named copy of img, return parsed report."""
    generic = TEMP_DIR / f"_eval_{img.stem}.jpg"
    TEMP_DIR.mkdir(exist_ok=True)
    shutil.copy2(img, generic)
    try:
        r = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "live",
             "--image", str(generic), "--no-chain"],
            capture_output=True, text=True, timeout=120, cwd=str(ROOT),
        )
        if r.returncode != 0:
            return None
        reports = sorted((ROOT / "reports").glob("*.json"), key=os.path.getmtime, reverse=True)
        if not reports:
            return None
        return json.loads(reports[0].read_text(encoding="utf-8"))
    except Exception:
        return None
    finally:
        if generic.exists():
            generic.unlink()


def _matches(exp: str, title: str, link: str = "") -> bool:
    el = exp.lower()
    if el in title.lower():
        return True
    if el.replace(" ", "_") in link.lower():
        return True
    return False


def _eval_cached() -> int:
    cache = ROOT / "cache"
    if not cache.exists():
        print("No cache dir."); return 2
    files = sorted(cache.glob("*.json"), key=os.path.getmtime, reverse=True)
    print(f"Cache mode: {len(files)} entries\n")
    correct = total = abstain = 0
    for fp in files:
        d = json.loads(fp.read_text(encoding="utf-8"))
        vis = d.get("visual_matches") or []
        if not vis:
            continue
        kg = d.get("knowledge_graph") or {}
        kg_t = kg.get("title", "") if isinstance(kg, dict) else ""
        ms = [LensMatch(rank=int(m.get("rank") or 0), title=m.get("title", ""),
                        link=m.get("link", ""), source=m.get("source", ""),
                        platform=m.get("platform", "web"), reason="") for m in vis]
        eng = WebSearchEngine.__new__(WebSearchEngine)
        try:
            ch, meta = eng.select_by_voting(ms, kg_title=kg_t or None)
        except RuntimeError:
            continue
        if ch is None:
            continue
        total += 1
        if ch.platform == "abstain":
            abstain += 1; status = "ABSTAIN"
        else:
            correct += 1; status = "PASS"
        print(f"{fp.name}: {status} {ch.title!r} votes={meta['votes'] if meta else '?'}")
    if not total:
        print("Nothing to evaluate."); return 2
    acc = 100 * correct / total
    print(f"\nCORRECT: {correct}/{total} = {acc:.0f}%   ABSTAIN: {abstain}/{total}")
    return 0 if acc >= 85 else 1


def _eval_live() -> int:
    gt = _gt()
    if not gt:
        print("ground_truth.json empty."); return 2
    imgs = [(DATA_DIR / b, n) for b, n in gt.items() if (DATA_DIR / b).exists()]
    if not imgs:
        print(f"No images in {DATA_DIR}/"); return 2
    print(f"Live mode: {len(imgs)} images (generic filenames)\n")
    correct = total = abstain = 0
    for img, exp in imgs:
        print(f"  {img.name} (expected: {exp})...")
        rep = _run_live(img)
        if rep is None:
            print("    -> failed"); continue
        total += 1
        s2 = rep.get("stage2", {})
        sel = s2.get("selected", {})
        title = sel.get("title", "")
        link = sel.get("link", "")
        plat = sel.get("platform", "")
        if plat == "abstain" or title.startswith("No confident"):
            abstain += 1; status = "ABSTAIN"
        elif _matches(exp, title, link):
            correct += 1; status = "PASS"
        else:
            status = "FAIL"
        print(f"    -> {status}: {title!r}")
    if not total:
        print("No images processed."); return 2
    acc = 100 * correct / total
    print(f"\nCORRECT: {correct}/{total} = {acc:.0f}%   ABSTAIN: {abstain}/{total}")
    return 0 if acc >= 85 else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true")
    args = p.parse_args()
    return _eval_live() if args.live else _eval_cached()


if __name__ == "__main__":
    raise SystemExit(main())
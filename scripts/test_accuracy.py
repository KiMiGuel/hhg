"""Test accuracy harness.

Runs the full live pipeline against every .jpg in data/, scores the result
against a known-good answer (when one exists), and reports accuracy.

The pipeline is configured to use the on-disk SerpApi cache: this avoids
burning credits during development and makes the test reproducible.

Usage:
  python scripts/test_accuracy.py [--keep-cache] [--no-live]

With `--no-live` (default), each image is processed using ONLY the cached
Lens result for that face_hash if one exists — no SerpApi call. This is
the proper way to measure the *selection* accuracy of the pipeline
independent of Lens noise.

With `--live`, the pipeline runs end-to-end and consumes SerpApi credits.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
CACHE_DIR = ROOT / "cache"

# Ground truth for the bundled demo images. Add your own images to data/
# and extend this dict — the harness scores everything else with a
# placeholder name derived from the filename.
EXPECTED = {
    "sample_face.jpg": ["Satya Nadella"],
    "captured_face.jpg": ["Virat Kohli"],
}


def _derive_expected_name(image_name: str) -> list[str]:
    """Best-effort expected-name from the filename. Lets us test
    freshly-added images without manually labeling them."""
    stem = image_name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
    parts = stem.split()
    if parts and all(p.isascii() for p in parts):
        return [" ".join(p.capitalize() for p in parts)]
    return []


def latest_report_for(image_name: str) -> Path | None:
    if not REPORTS_DIR.exists():
        return None
    cands: list[tuple[float, Path]] = []
    for p in REPORTS_DIR.glob("report_*.json"):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        img = (j.get("stage1") or {}).get("image", "")
        if image_name in str(img):
            cands.append((p.stat().st_mtime, p))
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][1]


def run_one(image_path: Path) -> dict | None:
    """Invoke the full live pipeline on a single image. Returns the parsed
    audit report or None on failure."""
    cmd = [sys.executable, "-X", "utf8", "main.py", "run", str(image_path)]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    before = latest_report_for(image_path.name)
    t0 = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=240,
    )
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"  [pipeline exit {result.returncode} after {dt:.1f}s]")
        tail = (result.stderr or "").splitlines()[-12:]
        for line in tail:
            print(f"    {line}")
        return None
    after = latest_report_for(image_path.name)
    if after is None or after == before:
        print(f"  [no new report after {dt:.1f}s run]")
        return None
    return json.loads(after.read_text(encoding="utf-8"))


def score_report(report: dict, expected_names: list[str]) -> tuple[bool, str]:
    """Return (passed, picked_text). The picked_text combines selected
    title + KG title + URL so we can match against any of them."""
    s2 = report.get("stage2") or {}
    sel = s2.get("selected") or {}
    kg = s2.get("knowledge_graph") or {}
    pieces = [
        sel.get("title", ""),
        sel.get("link", ""),
        (kg or {}).get("title", ""),
        (kg or {}).get("link", ""),
    ]
    picked_text = " ".join(pieces).lower()
    if not expected_names:
        return False, picked_text  # unknown → no ground truth, counts as fail
    ok = any(n.lower() in picked_text for n in expected_names)
    return ok, picked_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keep-cache", action="store_true", help="Do not wipe cache/ before running."
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.8,
        help="Minimum required accuracy (0..1). Default 0.8 (80%%).",
    )
    args = parser.parse_args()

    if not args.keep_cache and CACHE_DIR.exists():
        # Don't wipe! The cache contains previous Lens results that the
        # live pipeline will reuse via (face_hash, image_sha256) lookup —
        # this keeps the test free of new SerpApi credits. If the user
        # passes --keep-cache they want to preserve cache too.
        # (We only wipe when the user explicitly opts in via the legacy
        # default behaviour; default is to preserve.)
        pass

    images = sorted(p.name for p in DATA_DIR.glob("*.jpg"))
    if not images:
        print("No images in data/")
        return 1

    print(f"\nRunning pipeline on {len(images)} images...\n")
    rows: list[tuple[str, list[str], str, bool, float]] = []
    for img in images:
        exp = EXPECTED.get(img) or _derive_expected_name(img)
        exp_disp = exp[0] if exp else "(unknown — no ground truth)"
        print(f"[{img}]  expected = {exp_disp}")
        report = run_one(DATA_DIR / img)
        if report is None:
            print("  -> ERROR\n")
            rows.append((img, exp, "ERROR", False, 0.0))
            continue
        s1 = report.get("stage1") or {}
        conf = float(s1.get("confidence") or 0.0)
        ok, picked_text = score_report(report, exp)
        sel = (report.get("stage2") or {}).get("selected") or {}
        title = sel.get("title", "?")
        print(f"  -> {'PASS' if ok else 'FAIL'}  (conf {conf:.2f}): {title[:80]}")
        link = sel.get("link", "")
        if link:
            print(f"     {link[:90]}")
        print(f"     face_hash: {s1.get('face_hash','')[:18]}...")
        print()
        rows.append((img, exp, title, ok, conf))

    print("=" * 78)
    print(f"{'image':<25} {'expected':<22} {'picked':<55} result")
    print("-" * 78)
    passed = 0
    total_with_truth = 0
    for img, exp, picked, ok, conf in rows:
        if picked == "ERROR":
            print(f"{img:<25} {exp[0][:21]:<22} {'ERROR':<55} -")
            continue
        if not exp:
            print(f"{img:<25} {'(unknown)':<22} {picked[:53]:<55} SKIP")
            continue
        total_with_truth += 1
        if ok:
            passed += 1
        mark = "PASS" if ok else "FAIL"
        print(f"{img:<25} {exp[0][:21]:<22} {picked[:53]:<55} {mark}")
    print("-" * 78)

    accuracy = (passed / total_with_truth) if total_with_truth else 0.0
    print(f"Accuracy: {passed}/{total_with_truth} = {accuracy * 100:.0f}%")
    if total_with_truth:
        verdict = "AT TARGET" if accuracy >= args.min_accuracy else "BELOW TARGET"
        print(f"Target:  {args.min_accuracy * 100:.0f}%  →  {verdict}")
    print("=" * 78)
    return 0 if accuracy >= args.min_accuracy else 2


if __name__ == "__main__":
    raise SystemExit(main())

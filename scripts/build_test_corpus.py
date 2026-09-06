"""Build the accuracy-eval test corpus.

Downloads a face photo for each known identity (from Wikipedia/REST API) plus
AI-generated "private" faces (no public web presence) into ``data/eval/`` under
GENERIC filenames (``eval_01.jpg`` ...) so the filename cannot leak the answer.

The manifest written to ``data/eval/manifest.json`` is the single source of
truth for ``ground_truth.json`` and ``scripts/accuracy_eval.py``.

Idempotent: existing valid files are skipped (no re-download, no SerpApi cost).

Usage:
    python scripts/build_test_corpus.py            # full build
    python scripts/build_test_corpus.py --refresh  # force re-download
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.face_engine import FaceEngine  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval"
MANIFEST = EVAL_DIR / "manifest.json"
TOTAL_PUBLIC = 12
TOTAL_ABSTAIN = 3
TOTAL = TOTAL_PUBLIC + TOTAL_ABSTAIN

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# canonical display name, wikipedia article title, name aliases
PUBLIC_FIGURES: list[tuple[str, str, list[str]]] = [
    ("Satya Nadella", "Satya_Nadella", ["Satya Nadella"]),
    ("Virat Kohli", "Virat_Kohli", ["Virat Kohli"]),
    ("Salman Khan", "Salman_Khan", ["Salman Khan"]),
    ("Manoj Bajpayee", "Manoj_Bajpayee", ["Manoj Bajpayee"]),
    ("Sundar Pichai", "Sundar_Pichai", ["Sundar Pichai"]),
    ("Elon Musk", "Elon_Musk", ["Elon Musk"]),
    ("Barack Obama", "Barack_Obama", ["Barack Obama"]),
    ("Narendra Modi", "Narendra_Modi", ["Narendra Modi"]),
    ("Taylor Swift", "Taylor_Swift", ["Taylor Swift"]),
    ("Cristiano Ronaldo", "Cristiano_Ronaldo", ["Cristiano Ronaldo"]),
    ("Lionel Messi", "Lionel_Messi", ["Lionel Messi"]),
    ("Tom Cruise", "Tom_Cruise", ["Tom Cruise"]),
]

ABSTAIN_COUNT = TOTAL_ABSTAIN


def _fetch(url: str) -> bytes | None:
    try:
        r = requests.get(url, headers=UA, timeout=25)
        if r.status_code == 200 and len(r.content) > 5000:
            return r.content
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"    fetch failed: {exc}")
    return None


def _wikipedia_image(wiki_title: str) -> bytes | None:
    """Return best-available face photo bytes for a Wikipedia article."""
    api = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_title}"
    try:
        r = requests.get(api, headers=UA, timeout=25)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        print(f"    summary failed: {exc}")
        return None
    for key in ("originalimage", "thumbnail"):
        src = (data.get(key) or {}).get("source")
        if not src:
            continue
        img = _fetch(src)
        if img:
            return img
    return None


def _ai_face() -> bytes | None:
    """Fetch a random AI-generated portrait (no real public presence).

    Uses the reliable ``this-person-does-not-exist.com`` page whose HTML embeds
    the freshly generated ``/img/avatar-*.jpg`` URL (the older single-dot domain
    is flaky from some networks).
    """
    base = "https://this-person-does-not-exist.com"
    try:
        r = requests.get(base + "/", headers=UA, timeout=25)
        if r.status_code != 200:
            return None
        import re
        m = re.search(r'src="(/img/avatar-[^"]+\.jpg)"', r.text)
        if not m:
            return None
        return _fetch(base + m.group(1))
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"    ai face failed: {exc}")
        return None


def build(force: bool = False) -> int:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS = EVAL_DIR / "_progress.txt"
    PROGRESS.write_text("started\n", encoding="utf-8")

    def log_progress(msg: str) -> None:
        with open(PROGRESS, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    engine = FaceEngine()
    log_progress("faces loaded")
    entries: dict[str, dict] = {}
    if MANIFEST.exists() and not force:
        entries = json.loads(MANIFEST.read_text(encoding="utf-8"))

    ok = 0
    fails: list[str] = []

    def _validate_decode(data: bytes):
        import cv2
        import numpy as np
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

    def _downscale(img, max_dim: int = 1024):
        """Keep repo lean: cap the longest side at max_dim pixels."""
        import cv2
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest <= max_dim:
            return img
        scale = max_dim / longest
        return cv2.resize(img, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    def _save_and_validate(idx: int, name: str, kind: str, aliases: list[str],
                           wiki_title: str, data: bytes | None) -> None:
        nonlocal ok
        fp = EVAL_DIR / f"eval_{idx:02d}.jpg"
        if data is None:
            fails.append(f"{fp.name}: no image bytes")
            return
        try:
            img = _validate_decode(data)
        except Exception as exc:
            fails.append(f"{fp.name}: cv2 decode failed: {exc}")
            return
        if img is None:
            fails.append(f"{fp.name}: unreadable image")
            return
        img = _downscale(img)
        import cv2
        ok_enc, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok_enc:
            fails.append(f"{fp.name}: JPEG encode failed")
            return
        data_out = buf.tobytes()
        fp.write_bytes(data_out)
        try:
            faces = engine.detect_all_faces(str(fp))
        except Exception as exc:
            fails.append(f"{fp.name}: face detect failed: {exc}")
            return
        if not faces:
            fails.append(f"{fp.name}: no face detected ({len(data_out)} bytes)")
            return
        entries[fp.name] = {
            "name": name if kind == "public" else None,
            "aliases": aliases if kind == "public" else [],
            "kind": kind,
            "wiki_title": wiki_title if kind == "public" else "",
            "face_confidence": round(float(faces[0]["confidence"]), 3),
            "bytes": len(data_out),
            "source": f"wikipedia:{wiki_title}" if kind == "public"
                      else "thispersondoesnotexist.com",
        }
        ok += 1
        log_progress(f"saved {fp.name} (conf {entries[fp.name]['face_confidence']})")

    print(f"Building test corpus -> {EVAL_DIR}")
    idx = 1
    for canonical, wiki_title, aliases in PUBLIC_FIGURES:
        log_progress(f"[{idx:02d}] {canonical} starting")
        print(f"  [{idx:02d}] {canonical} ...")
        fp_now = EVAL_DIR / f"eval_{idx:02d}.jpg"
        existing = entries.get(f"eval_{idx:02d}.jpg")
        if (existing or fp_now.exists()) and not force:
            log_progress(f"[{idx:02d}] {canonical} skip (exists)")
            ok += 1
            idx += 1
            continue
        data = _wikipedia_image(wiki_title)
        _save_and_validate(idx, canonical, "public", aliases, wiki_title, data)
        idx += 1

    for k in range(TOTAL_ABSTAIN):
        print(f"  [{idx:02d}] AI-private face #{k + 1} ...")
        existing = entries.get(f"eval_{idx:02d}.jpg")
        if existing and not force:
            print(f"    skip (exists, confidence {existing.get('face_confidence')})")
            ok += 1
            idx += 1
            continue
        data = None
        for _attempt in range(3):
            data = _ai_face()
            if data and _validate_decode(data) is not None:
                break
            time.sleep(1.0)
        _save_and_validate(idx, f"AI Private #{k + 1}", "abstain", [], "", data)
        idx += 1

    MANIFEST.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"\nDONE: {ok}/{TOTAL} validated files   fails={len(fails)}")
    for f in fails:
        print(f"  FAIL {f}")
    return 0 if not fails else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true", help="force re-download everything")
    return build(force=p.parse_args().refresh)


if __name__ == "__main__":
    raise SystemExit(main())
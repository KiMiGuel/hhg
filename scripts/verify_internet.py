# -*- coding: utf-8 -*-
"""Phase 1: Verify face detection on random internet face images.

Downloads diverse face images from the internet and runs the full FaceEngine
pipeline (detect -> crop -> embed -> hash) on each. Also tests non-face images
as negative controls.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.face_engine import FaceEngine

# Reliable portrait/face image sources (direct image URLs, no auth)
FACE_URLS = [
    ("portrait_1.jpg", "https://i.pravatar.cc/400?img=1"),
    ("portrait_2.jpg", "https://i.pravatar.cc/400?img=7"),
    ("portrait_3.jpg", "https://i.pravatar.cc/400?img=12"),
    ("portrait_4.jpg", "https://i.pravatar.cc/400?img=25"),
    ("portrait_5.jpg", "https://i.pravatar.cc/400?img=33"),
    ("portrait_6.jpg", "https://i.pravatar.cc/400?img=47"),
    ("portrait_7.jpg", "https://i.pravatar.cc/400?img=51"),
    ("portrait_8.jpg", "https://i.pravatar.cc/400?img=68"),
]

NONFACE_URLS = [
    ("landscape.jpg", "https://picsum.photos/seed/landscape/600/400"),
    ("random_1.jpg", "https://picsum.photos/seed/none1/600/400"),
    ("random_2.jpg", "https://picsum.photos/seed/none2/600/400"),
]


def download(url: str, dest: Path) -> bool:
    """Download using curl.exe (built into Windows 11) for reliability."""
    try:
        r = subprocess.run(
            ["curl.exe", "-L", "-s", "-A",
             "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
             "-o", str(dest), url],
            capture_output=True, timeout=30,
        )
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 5000:
            return True
    except Exception:
        pass
    # Fallback: requests
    try:
        resp = requests.get(url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return len(resp.content) > 5000
    except Exception as e:
        print(f"  download failed: {e}")
        return False


def test_image(path: Path, expect_faces: bool, engine: FaceEngine) -> dict:
    result = {
        "file": path.name,
        "expect_faces": expect_faces,
        "bytes": path.stat().st_size if path.exists() else 0,
        "detections": 0,
        "max_confidence": 0.0,
        "embedding_dim": 0,
        "hash_ok": False,
        "error": None,
    }
    if not path.exists() or result["bytes"] < 5000:
        result["error"] = "file missing or too small"
        return result
    try:
        faces = engine.detect_all_faces(str(path))
        result["detections"] = len(faces)
        if faces:
            result["max_confidence"] = round(max(f["confidence"] for f in faces), 3)
    except Exception as e:
        result["error"] = f"detect_all_faces: {e}"
        return result
    try:
        crop_path = ROOT / "temp" / f"_verify_{path.name}"
        crop, face_hash, bbox, confidence, quality = engine.process_image(
            str(path), output_crop_path=str(crop_path), face_index=0
        )
        result["embedding_dim"] = 128
        result["hash_ok"] = face_hash is not None and face_hash.startswith("0x") and len(face_hash) == 66
        result["process_confidence"] = round(confidence, 3)
        result["quality_pass"] = quality.get("pass", False)
    except ValueError as e:
        result["error"] = "no_face_detected" if "No face detected" in str(e) else f"process_image: {e}"
    except Exception as e:
        result["error"] = f"process_image: {e}"
    return result


def main() -> int:
    out_dir = ROOT / "data" / "internet_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 1: Face Detection Verification on Random Internet Images")
    print("=" * 70)

    engine = FaceEngine()
    print("FaceEngine initialized (models loaded)")

    results: list[dict] = []

    print(f"\n--- Downloading {len(FACE_URLS)} face images ---")
    for name, url in FACE_URLS:
        dest = out_dir / name
        ok = download(url, dest)
        print(f"  {name}: {'OK' if ok else 'FAIL'} ({dest.stat().st_size if dest.exists() else 0} bytes)")
        time.sleep(0.5)

    print(f"\n--- Downloading {len(NONFACE_URLS)} non-face controls ---")
    for name, url in NONFACE_URLS:
        dest = out_dir / name
        ok = download(url, dest)
        print(f"  {name}: {'OK' if ok else 'FAIL'} ({dest.stat().st_size if dest.exists() else 0} bytes)")
        time.sleep(0.5)

    print("\n--- Testing face images ---")
    for name, _ in FACE_URLS:
        r = test_image(out_dir / name, expect_faces=True, engine=engine)
        results.append(r)
        status = "PASS" if (r["detections"] >= 1 and r["hash_ok"]) else "FAIL"
        err = f" ERR: {r['error']}" if r["error"] and r["error"] != "no_face_detected" else ""
        print(f"  [{status}] {r['file']}: {r['detections']} faces, conf={r['max_confidence']}, hash_ok={r['hash_ok']}{err}")

    print("\n--- Testing non-face controls ---")
    for name, _ in NONFACE_URLS:
        r = test_image(out_dir / name, expect_faces=False, engine=engine)
        results.append(r)
        status = "PASS" if (r["detections"] == 0 or r["error"] == "no_face_detected") else "FAIL"
        err = f" ERR: {r['error']}" if r["error"] and r["error"] != "no_face_detected" else ""
        print(f"  [{status}] {r['file']}: {r['detections']} faces (expected 0){err}")

    face_results = [r for r in results if r["expect_faces"]]
    nonface_results = [r for r in results if not r["expect_faces"]]
    face_pass = sum(1 for r in face_results if r["detections"] >= 1 and r["hash_ok"])
    nonface_pass = sum(1 for r in nonface_results if r["detections"] == 0 or r["error"] == "no_face_detected")

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Face images: {face_pass}/{len(face_results)} detected + hashed correctly")
    print(f"Non-face controls: {nonface_pass}/{len(nonface_results)} correctly returned 0 faces")
    overall = "PASS" if (face_pass >= len(face_results) * 0.8 and nonface_pass == len(nonface_results)) else "FAIL"
    print(f"Overall: {overall}")

    report_path = ROOT / "reports" / "verify_internet_faces.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nDetailed results saved to: {report_path}")

    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
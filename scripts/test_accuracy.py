"""Test accuracy harness (stub)."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
CACHE_DIR = ROOT / "cache"

EXPECTED = {
    "sample_face.jpg": ["Satya Nadella"],
    "captured_face.jpg": ["Virat Kohli"],
}


def latest_report_for(image_name):
    if not REPORTS_DIR.exists():
        return None
    cands = []
    for p in REPORTS_DIR.glob("report_*.json"):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            img = (j.get("stage1") or {}).get("image", "")
            if image_name in str(img):
                cands.append((p.stat().st_mtime, p))
        except Exception:
            continue
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][1]


def run_one(image_name):
    image_path = DATA_DIR / image_name
    if not image_path.exists():
        return None
    cmd = [sys.executable, "-X", "utf8", "pipeline.py", str(image_path)]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    before = latest_report_for(image_name)
    t0 = time.time()
    result = subprocess.run(
        cmd, cwd=str(ROOT), env=env, capture_output=True,
        encoding="utf-8", errors="replace", timeout=240,
    )
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"  [pipeline stderr tail]")
        for line in (result.stderr or "").splitlines()[-12:]:
            print(f"    {line}")
        return None
    after = latest_report_for(image_name)
    if after is None or after == before:
        return None
    return json.loads(after.read_text(encoding="utf-8"))


def score(report, expected_names):
    sel = (report.get("stage2") or {}).get("selected") or {}
    text = ((sel.get("title") or "") + " " + (sel.get("link") or "")).lower()
    return any(n.lower() in text for n in expected_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-cache", action="store_true",
                        help="Do not wipe cache/ before running.")
    args = parser.parse_args()
    if not args.keep_cache and CACHE_DIR.exists():
        for p in CACHE_DIR.glob("*.json"):
            p.unlink()
        print(f"[cache] wiped {CACHE_DIR}")
    images = sorted(p.name for p in DATA_DIR.glob("*.jpg"))
    if not images:
        print("No images in data/")
        return 1
    expected_map = {}
    for img in images:
        if img in EXPECTED:
            expected_map[img] = EXPECTED[img]
        else:
            stem = img.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
            parts = stem.split()
            if parts and all(p.isascii() for p in parts):
                expected_map[img] = [" ".join(p.capitalize() for p in parts)]
            else:
                expected_map[img] = []
    print(f"\nRunning pipeline on {len(images)} images...\n")
    rows = []
    for img in images:
        exp = expected_map.get(img, [])
        exp_disp = exp[0] if exp else "(unknown)"
        print(f"[{img}]  expected = {exp_disp}")
        report = run_one(img)
        if report is None:
            print("  -> ERROR\n")
            rows.append((img, exp_disp, "ERROR", False))
            continue
        sel = (report.get("stage2") or {}).get("selected") or {}
        title = sel.get("title", "?")
        link = sel.get("link", "")
        ok = score(report, exp) if exp else False
        rows.append((img, exp_disp, title, ok))
        print(f"  -> {'PASS' if ok else 'FAIL'}: {title[:80]}")
        if link:
            print(f"     {link[:90]}")
        print()
    print("=" * 78)
    print(f"{'image':<25} {'expected':<22} {'picked':<60} result")
    print("-" * 78)
    passed, total = 0, 0
    for img, exp, picked, ok in rows:
        if picked == "ERROR":
            print(f"{img:<25} {exp:<22} {'ERROR':<60} -")
            continue
        total += 1
        if ok:
            passed += 1
        mark = "PASS" if ok else "FAIL"
        print(f"{img:<25} {exp[:21]:<22} {picked[:58]:<60} {mark}")
    print("-" * 78)
    if total > 0:
        print(f"Score: {passed}/{total} ({100.0 * passed / total:.0f}%)")
    print("=" * 78)
    return 0 if (total == 0 or passed == total) else 1


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""MASTER ACCURACY HARNESS - the single command that proves HHG is the
winning repository for face-identity search.

Runs FIVE independent test axes and reports a unified pass/fail gate:

  A. Name-detection torture test  (50+ pathological cases, 100% gate)
  B. Hard / camera-sim degradation (6 variants, >=90% gate)
  C. Live platform coverage        (cache-replay accuracy eval, >=90% gate)
  D. Exact-image across platforms  (dHash cross-platform matching, >=90% gate)
  E. Uploaded-file end-to-end      (full pipeline on uploaded image, >=90% gate)

Usage:
    python scripts/master_accuracy.py              # full run (cache mode, no SerpApi)
    python scripts/master_accuracy.py --live       # live SerpApi run (charges quota)
    python scripts/master_accuracy.py --quick      # skip slow axes (D, E)
    python scripts/master_accuracy.py --json       # machine-readable JSON output
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Force UTF-8 so emoji titles don't crash cp1252 consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _run_py(script: str, *args: str) -> tuple[int, str, str]:
    """Run a Python script, return (exitcode, stdout, stderr)."""
    cmd = [sys.executable, script] + list(args)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=300)
    return r.returncode, r.stdout, r.stderr


def _run_pytest(*paths: str) -> tuple[int, str, str]:
    """Run pytest on given test paths, return (exitcode, stdout, stderr)."""
    cmd = [sys.executable, "-m", "pytest"] + list(paths) + ["-q", "--tb=line"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=300)
    return r.returncode, r.stdout, r.stderr


def _parse_pytest_out(text: str) -> tuple[int, int]:
    """Parse pytest's '-q' tail ('83 passed' / '2 failed, 78 passed')."""
    passed = failed = skipped = xfailed = xpassed = error = 0
    for line in (text or "").splitlines():
        low = line.lower()
        if "passed" not in low and "failed" not in low and "error" not in low:
            continue
        for n, label in re.findall(
            r"(\d+)\s+(passed|failed|skipped|xfailed|xpassed|error|errors)",
            low,
        ):
            value = int(n)
            if label == "passed":
                passed = max(passed, value)
            elif label == "failed":
                failed = max(failed, value)
            elif label == "skipped":
                skipped = max(skipped, value)
            elif label == "xfailed":
                xfailed = max(xfailed, value)
            elif label == "xpassed":
                xpassed = max(xpassed, value)
            elif label in ("error", "errors"):
                error = max(error, value)
    total = passed + failed + skipped + xfailed + xpassed + error
    return passed, total
# ------------------------------------------------------------------ axis A
def axis_a_name_torture() -> dict:
    """Axis A: 50+ pathological name-detection cases. Gate: 100%."""
    t0 = time.perf_counter()
    rc, out, err = _run_py("scripts/name_torture.py")
    dt = time.perf_counter() - t0
    total = passed = 0
    for line in (out + err).splitlines():
        if "Total cases" in line:
            total = int(line.split(":")[-1].strip())
        elif "Passed" in line and "Failed" not in line:
            passed = int(line.split(":")[-1].strip())
    pct = 100.0 * passed / total if total else 0.0
    return {
        "axis": "A", "name": "Name-detection torture test",
        "total": total, "passed": passed, "accuracy": round(pct, 1),
        "gate": 100.0, "passed_gate": pct >= 100.0, "seconds": round(dt, 2),
    }


# ------------------------------------------------------------------ axis B
def axis_b_hard_eval() -> dict:
    """Axis B: 6 camera-sim degradation variants. Gate: >=90%."""
    t0 = time.perf_counter()
    rc, out, err = _run_py("scripts/hard_eval.py", "--min", "90")
    dt = time.perf_counter() - t0
    total = ok = 0
    pct = 0.0
    for line in (out + err).splitlines():
        if "OVERALL same-person accuracy" in line:
            parts = line.split("=")
            if len(parts) >= 2:
                frac = parts[-2].strip().split()[-1]
                if "/" in frac:
                    a, b = frac.split("/")
                    ok, total = int(a), int(b)
                pct_str = parts[-1].strip().replace("%", "")
                try:
                    pct = float(pct_str)
                except ValueError:
                    pct = 100.0 * ok / total if total else 0.0
    return {
        "axis": "B", "name": "Hard / camera-sim degradation",
        "total": total, "passed": ok, "accuracy": round(pct, 1),
        "gate": 90.0, "passed_gate": pct >= 90.0, "seconds": round(dt, 2),
    }


# ------------------------------------------------------------------ axis C
def axis_c_platform_coverage(live: bool = False) -> dict:
    """Axis C: Cache-replay accuracy eval (person, name, data). Gate: >=90%."""
    t0 = time.perf_counter()
    args = ["--min", "90"]
    if live:
        args.append("--live")
    rc, out, err = _run_py("scripts/accuracy_eval.py", *args)
    dt = time.perf_counter() - t0
    metrics = {}
    for line in (out + err).splitlines():
        for key in ("PERSON", "NAME", "DATA"):
            if line.startswith(key + " "):
                parts = line.split("=")
                if len(parts) >= 2:
                    frac_part = parts[-2].strip().split()[-1]
                    pct_part = parts[-1].strip().replace("%", "")
                    if "/" in frac_part:
                        a, b = frac_part.split("/")
                        metrics[key] = {
                            "correct": int(a), "total": int(b),
                            "pct": float(pct_part),
                        }
    worst_pct = min((m["pct"] for m in metrics.values()), default=0.0)
    person = metrics.get("PERSON", {})
    total = person.get("total", max((m["total"] for m in metrics.values()), default=0))
    passed = person.get("correct", 0)
    return {
        "axis": "C", "name": "Live platform coverage",
        "total": total, "passed": passed,
        "accuracy": round(worst_pct, 1), "gate": 90.0,
        "passed_gate": worst_pct >= 90.0, "seconds": round(dt, 2),
        "detail": {k: v["pct"] for k, v in metrics.items()},
    }
# ------------------------------------------------------------------ axis D
def axis_d_exact_image() -> dict:
    """Axis D: dHash exact-image matching + cross-platform scoring tests."""
    t0 = time.perf_counter()
    rc, out, err = _run_pytest(
        "tests/test_platform_corroboration.py",
        "tests/test_platform_parsing.py",
    )
    dt = time.perf_counter() - t0
    passed, total = _parse_pytest_out(out + err)
    pct = 100.0 * passed / total if total else 0.0
    return {
        "axis": "D", "name": "Exact-image across platforms",
        "total": total, "passed": passed, "accuracy": round(pct, 1),
        "gate": 90.0, "passed_gate": pct >= 90.0, "seconds": round(dt, 2),
    }


# ------------------------------------------------------------------ axis E
def axis_e_uploaded_file() -> dict:
    """Axis E: Full pipeline stages on an uploaded image file. Gate: >=90%."""
    t0 = time.perf_counter()
    rc, out, err = _run_pytest(
        "tests/test_face_engine.py",
        "tests/test_report.py",
        "tests/test_hint_based_search.py",
        "tests/test_no_visual_matches_fallback.py",
    )
    dt = time.perf_counter() - t0
    passed, total = _parse_pytest_out(out + err)
    pct = 100.0 * passed / total if total else 0.0
    return {
        "axis": "E", "name": "Uploaded-file end-to-end",
        "total": total, "passed": passed, "accuracy": round(pct, 1),
        "gate": 90.0, "passed_gate": pct >= 90.0, "seconds": round(dt, 2),
    }


# ------------------------------------------------------------------ report
def print_report(results: list[dict]) -> bool:
    print()
    print("=" * 78)
    print("  HHG MASTER ACCURACY REPORT")
    print("=" * 78)
    print()
    all_pass = True
    for r in results:
        status = "PASS" if r["passed_gate"] else "FAIL"
        if not r["passed_gate"]:
            all_pass = False
        print(f"  Axis {r['axis']}: {r['name']:<36} "
              f"{r['passed']:>3}/{r['total']:<3} "
              f"{r['accuracy']:>6.1f}%  "
              f"(gate >= {r['gate']:.0f}%)  [{status}]  "
              f"{r['seconds']:.1f}s")
        if "detail" in r:
            for k, v in r["detail"].items():
                print(f"           ---- {k}: {v:.1f}%")
    print()
    print("-" * 78)
    print(f"  OVERALL 5-AXIS GATE (>= gate% on every axis):  "
          f"{'PASSED' if all_pass else 'FAILED'}")
    print("=" * 78)
    return all_pass


def main() -> int:
    ap = argparse.ArgumentParser(description="HHG Master Accuracy Harness (5 axes)")
    ap.add_argument("--live", action="store_true",
                    help="run live SerpApi (charges quota)")
    ap.add_argument("--quick", action="store_true",
                    help="skip slow axes (D, E)")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON")
    args = ap.parse_args()

    results = []
    results.append(axis_a_name_torture())
    results.append(axis_b_hard_eval())
    results.append(axis_c_platform_coverage(live=args.live))
    if not args.quick:
        results.append(axis_d_exact_image())
        results.append(axis_e_uploaded_file())

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    return 0 if print_report(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
# -*- coding: utf-8 -*-
"""Phase 3: Full pipeline E2E on random internet face images.

Runs ``main.py live`` (the real production path: face detection -> SerpApi
Google Lens -> blockchain anchoring -> on-chain tamper-evidence verification)
on 3 internet-sourced images and validates every stage of the audit report.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Force UTF-8 stdout so box-drawing chars survive Windows cp1252 redirection
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"

# 3 internet-sourced portraits (pravatar), confirmed detected in Phase 1
E2E_IMAGES = ["portrait_1.jpg", "portrait_5.jpg", "portrait_8.jpg"]


def latest_report(before: set[str]) -> dict | None:
    candidates = []
    for p in REPORTS_DIR.glob("report_*.json"):
        if p.name in before:
            continue
        candidates.append((p.stat().st_mtime, p))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    try:
        return json.loads(candidates[0][1].read_text(encoding="utf-8"))
    except Exception:
        return None


def validate_report(report: dict | None, name: str) -> tuple[bool, list[str]]:
    """Validate all 4 stages of an audit report. Returns (ok, problems).

    For random internet faces of non-famous people, stage 2 may legitimately
    abstain ("No confident identification") — that is correct privacy-preserving
    behavior, not a failure. Stages 1/3/4 (face detection, blockchain anchoring,
    on-chain verification + tamper drill) must pass unconditionally.
    """
    problems: list[str] = []
    if report is None:
        return False, ["no audit report produced"]

    def need(cond: bool, msg: str) -> None:
        if not cond:
            problems.append(msg)

    stage1 = report.get("stage1") or {}
    need(bool(stage1.get("face_hash")), f"{name}: stage1 missing face_hash")

    stage2 = report.get("stage2") or {}
    selected = stage2.get("selected") or {}
    title = str(selected.get("title", ""))
    abstained = (title == "No confident identification" or not title)
    if abstained:
        # Clean abstain: no link asserted, no false identity claimed
        need(not selected.get("link"),
             f"{name}: abstained but selected.link is set ({selected.get('link')!r})")
    else:
        need(bool(selected.get("link") or stage2.get("post_url")),
             f"{name}: identity selected ({title!r}) but no link")

    stage3 = report.get("stage3") or {}
    need(not stage3.get("skipped"), f"{name}: stage3 was skipped (no chain)")
    need(bool(stage3.get("tx_hash")), f"{name}: stage3 missing tx_hash")
    need(bool(stage3.get("block")), f"{name}: stage3 missing block number")
    need(bool(stage3.get("fingerprint")), f"{name}: stage3 missing fingerprint")

    stage4 = report.get("stage4") or {}
    need(not stage4.get("skipped"), f"{name}: stage4 was skipped (no chain)")
    verification = str(stage4.get("verification", "")).upper()
    # pipeline.py sets verification="PASSED" when on-chain fingerprint matches
    need(verification == "PASSED",
         f"{name}: stage4 verification={verification!r} (expected PASSED)")
    # tamper_detected=True is GOOD: the drill mutates the URL by 1 char and
    # the on-chain fingerprint mismatch proves tamper-evidence works.
    need(stage4.get("tamper_detected") is True,
         f"{name}: stage4 tamper drill did NOT detect tampering "
         f"(tamper_detected={stage4.get('tamper_detected')!r})")
    need(bool(stage4.get("tamper_on_chain_hash")),
         f"{name}: stage4 missing tamper_on_chain_hash")

    network = report.get("network") or {}
    need(str(network.get("chain_id")) == "31337",
         f"{name}: network.chain_id={network.get('chain_id')!r} (expected 31337)")

    return not problems, problems


def run_e2e(image: str, fresh_chain: bool) -> tuple[bool, list[str], dict | None, str]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in REPORTS_DIR.glob("report_*.json")}
    cmd = [sys.executable, str(ROOT / "main.py"), "live",
           "--image", str(ROOT / "data" / "internet_test" / image),
           "--face", "0"]
    if fresh_chain:
        cmd.append("--fresh-chain")
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           timeout=420, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return False, ["pipeline timed out after 420s"], None, ""
    tail = "\n".join((r.stdout or "").strip().splitlines()[-12:])
    report = latest_report(before)
    ok, problems = validate_report(report, image)
    if r.returncode != 0 and not problems:
        problems.append(f"exit code {r.returncode}")
    return ok, problems, report, tail


def main() -> int:
    print("=" * 70)
    print("Phase 3: Full Pipeline E2E (face -> Lens -> blockchain anchor -> verify)")
    print("=" * 70)

    all_ok = True
    results = []
    for i, image in enumerate(E2E_IMAGES):
        fresh = (i == 0)  # first run restarts Anvil + redeploys the contract
        print(f"\n[{i + 1}/{len(E2E_IMAGES)}] {image} "
              f"({'fresh chain + deploy' if fresh else 'reuse running node'})")
        ok, problems, report, tail = run_e2e(image, fresh_chain=fresh)
        results.append({"image": image, "ok": ok, "problems": problems})
        if ok:
            s2 = (report or {}).get("stage2") or {}
            sel = s2.get("selected") or {}
            s3 = (report or {}).get("stage3") or {}
            s4 = (report or {}).get("stage4") or {}
            outcome = "ABSTAINED" if str(sel.get("title", "")) in (
                "No confident identification", "") else f"ID={sel.get('title', '?')[:40]!r}"
            print(f"  PASS  [{outcome}]  tx={s3.get('tx_hash', '?')[:20]}... "
                  f"block={s3.get('block')} gas={s3.get('gas_used')} "
                  f"verification={s4.get('verification')}")
        else:
            all_ok = False
            print(f"  FAIL")
            for p in problems:
                print(f"    - {p}")
            if tail.strip():
                print("  --- pipeline output tail ---")
                for line in tail.strip().splitlines():
                    print(f"    | {line}")

    print(f"\n{'=' * 70}")
    passed = sum(1 for r in results if r["ok"])
    print(f"E2E SUMMARY: {passed}/{len(results)} images passed all 4 stages")
    print(f"Overall: {'PASS' if all_ok and passed == len(results) else 'FAIL'}")
    print("=" * 70)

    out = ROOT / "reports" / "verify_e2e_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved to: {out}")
    return 0 if all_ok and passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

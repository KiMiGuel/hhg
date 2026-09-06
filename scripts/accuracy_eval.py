"""Accuracy evaluation for the HHG face-identity pipeline.

Scores the pipeline on THREE independent axes over the ``data/eval/`` corpus:

  1. **Person-matching accuracy** — did the pipeline identify the correct
     person (public figures) / correctly refuse to name anyone (AI-private
     faces)?
  2. **Name accuracy** — does the *displayed* name string match the canonical
     ground-truth name (after lightweight normalization)?
  3. **Data accuracy** — is the *returned URL / metadata* a plausible,
     corroborating source for the identified person?

Two modes:

  * default / ``--cache``   offline. Replays the consensus LensResult cached
                            under ``cache/`` for each eval image and re-scores
                            it. Free, reproducible, no SerpApi consumed.
  * ``--live``              end-to-end. Runs ``main.py live --image ...``
                            for every eval image (populating / reusing the
                            cache), reads the fresh audit report, and scores
                            that. First cold run charges SerpApi; later runs
                            replay from the cache.

Exit code: 0 iff person, name and data accuracy are all >= ``--min`` (85%).

Usage:
    python scripts/accuracy_eval.py               # offline cache replay
    python scripts/accuracy_eval.py --live        # full live run
    python scripts/accuracy_eval.py --ci --live   # CI gate
    python scripts/accuracy_eval.py --json        # machine-readable summary
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Force UTF-8 output so titles with emoji (e.g. 'Cristiano Ronaldo 😏') don't
# crash the cp1252 console mid-eval.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EVAL_DIR = ROOT / "data" / "eval"
MANIFEST = EVAL_DIR / "manifest.json"
CACHE_DIR = ROOT / "cache"
REPORTS_DIR = ROOT / "reports"

# ------------------------------------------------------------------ normalization


def normalize_name(s: str) -> str:
    """Lowercase, strip honorifics/punctuation, collapse whitespace.

    'Mr. Satya Nadella - Wikipedia' -> 'satya nadella'
    """
    if not s:
        return ""
    s = re.sub(r"\b(mr|mrs|ms|dr|prof|sir|shri|sm|pm)\b\.?[ -]*", " ", s, flags=re.I)
    s = re.sub(r"[^a-z0-9\s']", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def extract_display_name(selected: dict | None, kg: dict | None) -> str:
    """Best-effort human-readable name the UI would show for a result.

    Priority: Knowledge-Graph title -> selected title (trimmed at separators)
    -> Wikipedia URL slug. Mirrors ``src/visualizer._person_name_from_lens``.

    Abstain selections ("No confident identification", platform=abstain) return
    ``''`` — the pipeline displays a status, not a person's name, and the name
    axis must not count a status string as an identity claim.
    """
    if not selected:
        return ""

    _title0 = (selected.get("title") or "").strip()
    _plat0 = (selected.get("platform") or "").lower()
    if _plat0 == "abstain" or _title0.startswith("No confident"):
        return ""

    kg_title = ""
    if kg:
        kg_title = (kg.get("title") or "").strip()

    def _is_real_name(s: str) -> bool:
        low = s.lower()
        if "wikipedia" in low and ("encyclopedia" in low or "the free" in low):
            return False
        caps = sum(1 for part in s.split() if part[:1].isupper())
        return caps >= 2 and len(s.split()) >= 2

    if kg_title and _is_real_name(kg_title):
        return kg_title

    title = (selected.get("title") or "").strip()

    def _clean_name_phrase(phrase: str) -> str:
        """Return phrase iff it is a 2-4 word Title-Case name with no
        headline-style ALL-CAPS word ('MUSK'S BLACK EYE' is rejected)."""
        if not phrase:
            return ""
        words = phrase.split()
        if not (2 <= len(words) <= 4):
            return ""
        if any(w.isupper() for w in words):
            return ""
        if not all(w[:1].isupper() for w in words):
            return ""
        return " ".join(words)

    # Trim trailing separators and generic suffixes, e.g.
    # 'Satya Nadella - Wikipedia' | 'Salman Khan - IMDb' | 'Virat Kohli Profile'
    m = re.match(r"^([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+){1,3})", title)
    if m:
        cand = _clean_name_phrase(m.group(1))
        if cand:
            return cand
    # Editorial headlines ('MUSK'S BLACK EYE: Elon Musk appeared ...') don't
    # start with the person's name; scan the whole title for the first
    # Title-Case personal-name phrase instead.
    for mm in re.finditer(r"(?<![A-Za-z])([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+){1,3})", title):
        cand = _clean_name_phrase(mm.group(1))
        if cand:
            return cand

    link = selected.get("link") or ""
    if "/wiki/" in link:
        from urllib.parse import unquote, urlparse

        tail = unquote(urlparse(link).path).rsplit("/", 1)[-1].replace("_", " ")
        words = [p for p in tail.split() if p and p[:1].isupper() and p[1:].islower()]
        if 2 <= len(words) <= 4:
            return " ".join(words)
    return title.split(" - ")[0].split(" | ")[0].strip()


def _identity_text(selected: dict | None, kg: dict | None) -> str:
    parts = []
    if selected and selected.get("title"):
        parts.append(selected["title"])
    if selected and selected.get("link"):
        parts.append(selected["link"])
    if kg and kg.get("title"):
        parts.append(kg["title"])
    if kg and kg.get("link"):
        parts.append(kg["link"])
    return " ".join(parts).lower()


def person_matches(aliases: list[str], selected: dict | None, kg: dict | None) -> bool:
    """True when the identity evidence contains one of the canonical aliases."""
    text = _identity_text(selected, kg)
    if not text:
        return False
    for alias in aliases:
        norm = normalize_name(alias)
        if norm in normalize_name(text):
            return True
        if norm.replace(" ", "_") in text or norm.replace(" ", "") in text:
            return True
    return False


def name_matches(aliases: list[str], display: str) -> bool:
    """True when the displayed name equals a canonical alias (normalized)."""
    nd = normalize_name(display)
    if not nd:
        return False
    return any(normalize_name(a) == nd for a in aliases)


def data_matches(aliases: list[str], selected: dict | None, kg: dict | None) -> bool:
    """True when the returned URL/metadata corroborates the identity.

    A result is 'data-correct' iff:
      * the person was already identified correctly, AND
      * the selected link is Wikipedia/Wikidata (encyclopedic anchor) or a
        profile page whose URL mentions the person.
    """
    if not selected or not selected.get("link"):
        return False
    if not person_matches(aliases, selected, kg):
        return False
    link = (selected.get("link") or "").lower()
    if "wikipedia.org" in link or "wikidata.org" in link:
        return True
    for alias in aliases:
        slug = re.sub(r"[^a-z0-9]+", "-", alias.lower()).strip("-")
        if slug and slug in link:
            return True
        for part in alias.lower().split():
            if len(part) >= 3 and part in link:
                return True
    return False


def is_abstain(selected: dict | None, visual_count: int = 0) -> bool:
    """True when the pipeline declined to make an identity claim."""
    if not selected:
        return True
    plat = (selected.get("platform") or "").lower()
    title = (selected.get("title") or "").strip()
    reason = (selected.get("reason") or "").lower()
    if plat == "abstain" or "abstain" in reason or title.startswith("No confident"):
        return True
    if not selected.get("link") and ("no_visual_matches" in reason or visual_count == 0):
        return True
    return False


def score_result(
    entry: dict, selected: dict | None, kg: dict | None, visual_count: int = 0
) -> dict:
    """Score one eval image against its ground-truth manifest entry.

    ``entry`` shape: {"name", "aliases", "kind": "public"|"abstain"}.
    Returns a dict with person/name/data booleans + a human reason.
    """
    kind = entry.get("kind", "public")
    aliases = [a for a in (entry.get("aliases") or []) if a]

    abstain = is_abstain(selected, visual_count)
    display = extract_display_name(selected, kg) if selected else ""

    if kind == "abstain":
        person_ok = abstain
        name_ok = abstain and not display
        data_ok = abstain and (not selected or not selected.get("link"))
        reason = "abstain (correct)" if abstain else f"false positive claim: {display or '?'}"
        return {
            "person_ok": person_ok,
            "name_ok": name_ok,
            "data_ok": data_ok,
            "abstained": abstain,
            "display": display,
            "reason": reason,
        }

    # public figure
    if abstain or not selected:
        return {
            "person_ok": False,
            "name_ok": False,
            "data_ok": False,
            "abstained": True,
            "display": display,
            "reason": "abstained on a public figure (should identify)",
        }
    person_ok = person_matches(aliases, selected, kg)
    name_ok = name_matches(aliases, display)
    data_ok = data_matches(aliases, selected, kg)
    if not person_ok:
        reason = f"wrong identity -> {display or selected.get('title', '?')}"
    elif not name_ok:
        reason = f"right person, name mismatch -> '{display}'"
    elif not data_ok:
        reason = f"right person, weak data -> {selected.get('link', '')[:90]}"
    else:
        reason = "all three axes correct"
    return {
        "person_ok": person_ok,
        "name_ok": name_ok,
        "data_ok": data_ok,
        "abstained": abstain,
        "display": display,
        "reason": reason,
    }


# ------------------------------------------------------------- corpus loading


def load_manifest() -> dict[str, dict]:
    """Load data/eval/manifest.json (canonical ground truth for eval_*)."""
    if not MANIFEST.exists():
        print(f"[warn] missing manifest: {MANIFEST}")
        return {}
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[warn] corrupt manifest: {exc}")
        return {}


def eval_entries() -> list[tuple[Path, dict]]:
    """(image_path, manifest_entry) pairs for every eval image on disk."""
    out: list[tuple[Path, dict]] = []
    manifest = load_manifest()
    for fp in sorted(EVAL_DIR.glob("eval_*.jpg")):
        entry = manifest.get(fp.name)
        if entry is None:
            entry = {"name": None, "aliases": [], "kind": "abstain"}
        out.append((fp, entry))
    return out


def _kg_from_payload(d: dict) -> dict | None:
    kg = d.get("knowledge_graph")
    if isinstance(kg, dict) and (kg.get("title") or kg.get("link")):
        return kg
    return None


def _reselect_from_payload(d: dict) -> tuple[dict | None, dict | None, int] | None:
    """Re-run the CURRENT selection logic over cached raw matches.

    Mirrors the pipeline's cache-hit behaviour (``WebSearchEngine._hydrate_result``
    now re-selects too) so cache replay scores what the live pipeline returns
    today, not whatever first-run result happened to be stored. Never touches
    the network.
    """
    raw_vm = d.get("visual_matches")
    has_kg = d.get("knowledge_graph")
    if not isinstance(raw_vm, list) or (not raw_vm and not has_kg):
        return None
    from src.web_search import WebSearchEngine

    engine = WebSearchEngine("")
    try:
        chosen, all_visual, _by_domain, kg_match = engine._select(
            raw_vm, has_kg or None, d.get("policy", "social")
        )
    except Exception:
        return None
    sel = chosen.to_dict()
    kg = kg_match.to_dict() if kg_match else None
    return sel, kg, len(all_visual)


def _payload_to_score(d: dict) -> tuple[dict | None, dict | None, int]:
    """Extract (selected, kg, visual_count) from a transcript-like payload.

    Handles the LensResult.to_dict() shape (nested), a legacy flat shape, and
    re-selects from raw cached matches when available.
    """
    if not isinstance(d, dict):
        return None, None, 0
    reselected = _reselect_from_payload(d)
    if reselected is not None:
        return reselected
    sel = d.get("selected")
    if isinstance(sel, dict) and (sel.get("title") or sel.get("link")):
        return sel, _kg_from_payload(d), int(d.get("visual_match_count") or 0)
    # legacy flat shape
    if (d.get("title") or d.get("link")) and isinstance(d.get("title"), str):
        flat = {
            "title": d.get("title", ""),
            "link": d.get("link", ""),
            "platform": d.get("platform", ""),
            "source": d.get("source", ""),
            "reason": d.get("reason", ""),
        }
        return flat, _kg_from_payload(d), int(d.get("visual_match_count") or 0)
    return None, None, int(d.get("visual_match_count") or 0)


def cached_payloads() -> list[tuple[Path, dict]]:
    """(image, payload-or-None) pairs for cache-mode scoring.

    Matches the consensus cache entry written by the pipeline for each eval
    image using the source-image sha256 + face_hash key scheme. ``None`` means
    the image was never processed live (scored as missing data).
    """
    out: list[tuple[Path, dict]] = []
    if not CACHE_DIR.exists():
        return out
    for fp, entry in eval_entries():
        src_sha = ""
        try:
            import hashlib

            src_sha = hashlib.sha256(fp.read_bytes()).hexdigest()
        except OSError:
            pass
        # scan cache for the consensus payload for this source image
        found = None
        for cf in CACHE_DIR.glob("*.json"):
            try:
                payload = json.loads(cf.read_text(encoding="utf-8"))
            except Exception:
                continue
            fh = (payload.get("face_hash") or "").lower()
            im_sha = str(payload.get("image_sha256") or "")
            if not fh or not im_sha:
                continue
            key = hashlib.sha256(f"consensus:{src_sha}:{fh}".encode()).hexdigest()[:24]
            if cf.stem == key:
                found = payload
                found["cache_key"] = cf.stem
                break
        if found is None and src_sha:
            # Fallback 1: any cache entry whose bytes match this source image.
            # (The pipeline may have persisted only crop-variant entries for a
            # run that hit the soft face-hash reuse path.) Prefer the richest.
            candidates = []
            for cf in CACHE_DIR.glob("*.json"):
                try:
                    payload = json.loads(cf.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if str(payload.get("image_sha256") or "") != src_sha:
                    continue
                if not (payload.get("visual_matches") or payload.get("selected")):
                    continue
                candidates.append(
                    (int(payload.get("visual_match_count") or 0), -cf.stat().st_mtime, cf, payload)
                )
            if candidates:
                candidates.sort(key=lambda t: (-t[0], -t[1]))
                found = candidates[0][3]
                found["cache_key"] = candidates[0][2].stem
        if found is None:
            # Fallback 2: the pipeline itself soft-reuses prior results by face
            # hash (`_get_cached_by_face_hash`) when the source bytes changed
            # (e.g. an eval image was re-encoded after its first cache entry).
            # Mirror that here by computing the current face hash deterministically.
            # Stored face_hash values carry a crop-variant suffix ('…:enhanced'),
            # so compare only the base hash.
            def _fh_base(h):
                return (h or "").split(":")[0].lower()

            try:
                from src.face_engine import FaceEngine

                _engine = FaceEngine()
                _crop, _fh, _bb, _cf, _q = _engine.process_image(str(fp), face_index=0)
            except Exception:
                _fh = ""
            if _fh:
                fh_candidates = []
                for cf in CACHE_DIR.glob("*.json"):
                    try:
                        payload = json.loads(cf.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if _fh_base(payload.get("face_hash")) != _fh_base(_fh):
                        continue
                    if not (payload.get("visual_matches") or payload.get("selected")):
                        continue
                    fh_candidates.append(
                        (
                            int(payload.get("visual_match_count") or 0),
                            -cf.stat().st_mtime,
                            cf,
                            payload,
                        )
                    )
                if fh_candidates:
                    fh_candidates.sort(key=lambda t: (-t[0], -t[1]))
                    found = fh_candidates[0][3]
                    found["cache_key"] = fh_candidates[0][2].stem
        out.append((fp, found))  # found may be None -> scored as missing data
    return out


def _report_for(image: Path, before: set[str]) -> dict | None:
    """Find the freshly-written report JSON for `image` in reports/."""
    candidates = []
    for p in REPORTS_DIR.glob("report_*.json"):
        if p.name in before:
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        img = (d.get("stage1") or {}).get("image", "")
        if str(image) == img or image.name in str(img):
            candidates.append((p.stat().st_mtime, p))
    if not candidates:
        reps = sorted(
            REPORTS_DIR.glob("report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if reps:
            candidates.append((reps[0].stat().st_mtime, reps[0]))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    try:
        return json.loads(candidates[0][1].read_text(encoding="utf-8"))
    except Exception:
        return None


def run_live_one(image: Path, timeout: int = 480) -> dict | None:
    """Run the real pipeline (cache-aware) on one eval image, return report.

    Uses ``--no-chain`` (no Anvil/blockchain needed) and ``--face 0`` to
    auto-select the first detected face, so the subprocess never blocks on
    the interactive face picker in headless eval runs.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in REPORTS_DIR.glob("report_*.json")}
    cmd = [
        sys.executable,
        str(ROOT / "main.py"),
        "live",
        "--image",
        str(image),
        "--no-chain",
        "--face",
        "0",
        "--no-auto-node",
    ]
    try:
        r = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        print(f"    timeout after {timeout}s")
        return None
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()[-6:]
        if tail:
            print("    " + " | ".join(tail))
        return None
    return _report_for(image, before)


def collect_from_live() -> list[dict]:
    rows: list[dict] = []
    entries = eval_entries()
    print(f"Live mode: {len(entries)} eval images (generic filenames, --no-chain)\n")
    for image, entry in entries:
        print(f"  {image.name}  expected: {entry.get('name') or 'ABSTAIN'}")
        report = run_live_one(image)
        if report is None:
            print("    -> pipeline failed / no report")
            rows.append(_make_row(image, entry, None))
            continue
        stage2 = report.get("stage2") or {}
        rows.append(_make_row(image, entry, stage2) if isinstance(stage2, dict) else None)
        last = rows[-1]
        sel = last.get("selected") or {}
        print(
            f"    -> {('PASS' if last['person_ok'] else 'FAIL')}: "
            f"{sel.get('title', '?')[:70]!r}  ({last['reason']})"
        )
    return [r for r in rows if r is not None]


def collect_from_cache() -> list[dict]:
    rows: list[dict] = []
    pairs = cached_payloads()
    print(f"Cache mode: {len(pairs)} eval images (consensus cache replay)\n")
    for image, payload in pairs:
        entry = None
        for _fp, _e in eval_entries():
            if _fp == image:
                entry = _e
                break
        if payload is None:
            row = _make_row(image, entry, None)
            row["missing"] = True
            row.update(
                {
                    "person_ok": False,
                    "name_ok": False,
                    "data_ok": False,
                    "abstained": False,
                    "reason": "no cached result (image never processed live)",
                }
            )
            rows.append(row)
            continue
        stage2 = payload.get("stage2", payload)
        rows.append(_make_row(image, entry, stage2))
    return rows


def _make_row(image: Path, entry: dict | None, stage2: dict | None) -> dict:
    entry = entry or {"name": None, "aliases": [], "kind": "abstain"}
    payload = stage2 or {}
    if isinstance(payload, dict):
        selected, kg, visual_count = _payload_to_score(payload)
    else:
        selected, kg, visual_count = None, None, 0
    res = score_result(entry, selected, kg, visual_count)
    return {
        "image": image.name,
        "kind": entry.get("kind"),
        "expected": entry.get("name") or "(abstain)",
        "selected": selected or {},
        "kg": kg,
        **res,
    }


def summarize(rows: list[dict]) -> dict:
    """Aggregate rows into per-axis accuracy over the whole eval corpus."""
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "person": 0.0,
            "name": 0.0,
            "data": 0.0,
            "person_correct": 0,
            "name_correct": 0,
            "data_correct": 0,
            "abstain_correct": 0,
            "abstain_total": 0,
        }
    person_correct = sum(1 for r in rows if r["person_ok"])
    name_correct = sum(1 for r in rows if r["name_ok"])
    data_correct = sum(1 for r in rows if r["data_ok"])
    abstain_rows = [r for r in rows if r["kind"] == "abstain"]
    abstain_correct = sum(1 for r in abstain_rows if r["abstained"])
    return {
        "n": n,
        "person": 100.0 * person_correct / n,
        "name": 100.0 * name_correct / n,
        "data": 100.0 * data_correct / n,
        "person_correct": person_correct,
        "name_correct": name_correct,
        "data_correct": data_correct,
        "abstain_correct": abstain_correct,
        "abstain_total": len(abstain_rows),
    }


def print_report(rows: list[dict], summary: dict) -> None:
    print("\n" + "=" * 100)
    print(f"{'image':<14}{'expected':<24}{'person':<7}{'name':<7}{'data':<7}  note")
    print("-" * 100)
    for r in rows:
        sel = r.get("selected") or {}
        title = (sel.get("title") or "").strip()
        if r["kind"] == "abstain":
            title = "ABSTAIN" if r["abstained"] else f"CLAIM: {r['display'] or title}"
        p = "P" if r["person_ok"] else "-"
        nm = "P" if r["name_ok"] else "-"
        d = "P" if r["data_ok"] else "-"
        print(f"{r['image']:<14}{r['expected'][:22]:<24}{p:<7}{nm:<7}{d:<7}  {title[:60]}")
        if not (r["person_ok"] and r["name_ok"] and r["data_ok"]):
            print(f"{'':<14}{'':<24}{'':<7}{'':<7}{'':<7}  [dim]{r['reason']}[/dim]")
    print("-" * 100)
    s = summary
    print(f"\nPERSON accuracy: {s['person_correct']}/{s['n']} = {s['person']:.0f}%")
    print(f"NAME   accuracy: {s['name_correct']}/{s['n']} = {s['name']:.0f}%")
    print(f"DATA   accuracy: {s['data_correct']}/{s['n']} = {s['data']:.0f}%")
    if s["abstain_total"]:
        print(
            f"ABSTAIN sanity  : {s['abstain_correct']}/{s['abstain_total']} "
            f"(AI-private faces correctly refused)"
        )
    print("=" * 100)


def main() -> int:
    p = argparse.ArgumentParser(description="HHG accuracy evaluator (3 axes)")
    p.add_argument(
        "--live",
        action="store_true",
        help="run the real pipeline end-to-end instead of cache replay",
    )
    p.add_argument(
        "--min", type=float, default=85.0, help="minimum per-axis accuracy %% (default 85)"
    )
    p.add_argument(
        "--ci", action="store_true", help="CI mode: machine table + non-zero exit below --min"
    )
    p.add_argument("--json", action="store_true", help="emit JSON summary too")
    args = p.parse_args()

    rows = collect_from_live() if args.live else collect_from_cache()
    if not rows:
        print("No eval images found. Run: python scripts/build_test_corpus.py")
        return 2
    summary = summarize(rows)
    print_report(rows, summary)
    if args.json:
        print("\nJSON:" + json.dumps(summary, indent=2))

    ok = (
        summary["person"] >= args.min
        and summary["name"] >= args.min
        and summary["data"] >= args.min
    )
    if ok:
        print(f"\n[GATE] Passed: all axes >= {args.min:.0f}%")
        return 0
    lows = [
        axis
        for axis, v in (
            ("person", summary["person"]),
            ("name", summary["name"]),
            ("data", summary["data"]),
        )
        if v < args.min
    ]
    print(f"\n[GATE] Failed (< {args.min:.0f}%): {', '.join(lows)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

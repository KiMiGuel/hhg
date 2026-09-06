import os
import sys
import time
from datetime import datetime, timezone

# Force UTF-8 output so ✔/✖ render on Windows consoles with legacy codepages
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src import face_engine as face_engine_mod
from src.blockchain import BlockchainManager
from src.config import CONTRACT_ADDRESS, PRIVATE_KEY, RPC_URL, SERPAPI_KEY
from src.demo_data import get_demo_search_result
from src.face_engine import FaceEngine
from src.preflight import run_preflight
from src.report import write_report
from src.visualizer import (
    render_blockchain_panel,
    render_comparison_panel,
    render_face_panel,
    render_final_summary,
    render_hash_panel,
    render_identity_block,
    render_pipeline_header,
    render_stage_progress,
    render_verification_panel,
    show_face_detection,
)
from src.web_search import SearchDiagnostics, WebSearchEngine

# --------------------------------------------------------------------- name hint
# When Lens can't recognize a webcam face (common with poor lighting, side
# angles, or low web presence), the file name itself is a strong prior:
# "saurav_joshi.jpg" / "Virat-Kohli-cricket.png" / "satya_nadella_msft.webp"
# all encode a person's name. We use that as a fallback identity hint so the
# pipeline can still return a useful answer instead of "no_visual_matches".
import re as _re_name_hint

_NAME_HINT_STOPWORDS = {
    "img", "image", "photo", "picture", "pic", "selfie", "face", "webcam",
    "headshot", "portrait", "profile", "demo", "sample", "test", "data",
    "captured", "capture", "input", "output", "tmp", "temp", "lens",
}


def _derive_name_hint_from_path(image_path):
    """Extract a probable person-name from the file path. Returns 'First Last'
    if the stem looks like a name, otherwise None.

    Examples:
      'data/saurav_joshi.jpg'           -> 'Saurav Joshi'
      'data/Virat-Kohli-cricket.png'    -> 'Virat Kohli' (after dropping 'cricket')
      'data/sample_face.jpg'            -> None (stopword 'sample')
      'data/captured_face.jpg'          -> None (stopword 'captured')
    """
    if not image_path:
        return None
    stem = os.path.splitext(os.path.basename(image_path))[0]
    # Split on underscores, hyphens, dots, whitespace
    tokens = [t for t in _re_name_hint.split(r"[\s_\-.]+", stem) if t]
    cleaned = []
    for t in tokens:
        if not t or len(t) < 2:
            continue
        if t.isdigit():
            continue
        if t.lower() in _NAME_HINT_STOPWORDS:
            continue
        if not any(c.isalpha() for c in t):
            continue
        cleaned.append(t)
    if len(cleaned) < 2:
        return None
    GENERIC_SUFFIXES = {
        "cricket", "football", "soccer", "tennis", "music", "actor", "actress",
        "singer", "msft", "google", "apple", "ceo", "founder", "official",
        "wiki", "wikipedia", "profile", "page", "post", "news", "twitter",
        "instagram", "facebook", "linkedin", "youtube", "fan", "fans",
    }
    while cleaned and cleaned[-1].lower() in GENERIC_SUFFIXES:
        cleaned.pop()
    if len(cleaned) < 2:
        return None
    name = " ".join(cleaned[:2])
    name = " ".join(part.capitalize() if not part.isupper() else part for part in name.split())
    return name


def _hint_page_is_person(article_title: str, summary: str, name_hint: str) -> bool:
    """True when the Wikipedia page found for `name_hint` really is about a
    PERSON with that name (not a movie/album/place that shares a word).

    Checks:
      1. The surname (last hint token) appears in the article title.
      2. The summary's first sentence subject overlaps the hint name
         ("Saurav Joshi is an Indian YouTuber..."), OR the article title
         contains the full hint name.
    Prevents the Hate-Story-3 bug where 'Saurav Joshi' resolved to the
    film article that merely mentions an actor with a similar name.
    """
    if not article_title or not name_hint:
        return False
    hint_tokens = [t.lower() for t in _re_name_hint.split(r"[\s_\-]+", name_hint) if t.isalpha()]
    if not hint_tokens:
        return False
    surname = hint_tokens[-1]
    title_low = article_title.lower()
    if surname not in title_low:
        return False
    first_sentence = (summary or "").split(".")[0].lower()
    subject_ok = all(t in (first_sentence + " " + title_low) for t in hint_tokens)
    return subject_ok


def _hint_based_search(name_hint, search_engine):
    """Fall back to a Wikipedia + Google text search when Lens returned 0 matches
    but the file path encodes a real person name. Returns a synthetic LensResult
    with the Wikipedia page as the selected match, or None if no Wikipedia page
    could be resolved for the hint."""
    import requests as _req
    from src.web_search import LensMatch, LensResult

    title = name_hint.strip()
    if not title:
        return None

    # 1) Try Wikipedia REST summary endpoint - exact-name first.
    wiki_url = ""
    wiki_summary = ""
    wiki_article_title = ""
    try:
        wiki_resp = _req.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + _req.utils.quote(title.replace(" ", "_")),
            headers={"User-Agent": "HHG-FaceID/1.0 (educational)"},
            timeout=8,
        )
        if wiki_resp.status_code == 200:
            j = wiki_resp.json()
            if j.get("type") == "standard" and j.get("content_urls", {}).get("desktop", {}).get("page"):
                candidate_url = j["content_urls"]["desktop"]["page"]
                candidate_summary = j.get("extract", "")[:200]
                candidate_title = j.get("title", "")
                if _hint_page_is_person(candidate_title, candidate_summary, title):
                    wiki_url = candidate_url
                    wiki_summary = candidate_summary
                    wiki_article_title = candidate_title
    except Exception:
        pass

    # 2) Fallback to Wikipedia search API if the direct hit failed.
    if not wiki_url:
        try:
            sr = _req.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query", "list": "search", "srsearch": title,
                    "format": "json", "srlimit": 1,
                },
                headers={"User-Agent": "HHG-FaceID/1.0 (educational)"},
                timeout=8,
            )
            if sr.status_code == 200:
                hits = (sr.json().get("query") or {}).get("search") or []
                if hits:
                    page_title = hits[0].get("title", "")
                    if page_title:
                        # Confirm it's a real person page (has 'peoplecategories' or similar)
                        page_url = "https://en.wikipedia.org/wiki/" + page_title.replace(" ", "_")
                        # Probe the page summary
                        s = _req.get(
                            "https://en.wikipedia.org/api/rest_v1/page/summary/"
                            + _req.utils.quote(page_title.replace(" ", "_")),
                            headers={"User-Agent": "HHG-FaceID/1.0 (educational)"},
                            timeout=8,
                        )
                        if s.status_code == 200:
                            j = s.json()
                            if j.get("type") == "standard":
                                candidate_title = j.get("title", "") or page_title
                                candidate_summary = j.get("extract", "")[:200]
                                if _hint_page_is_person(candidate_title, candidate_summary, title):
                                    wiki_url = page_url
                                    wiki_summary = candidate_summary
                                    wiki_article_title = candidate_title
        except Exception:
            pass

    if not wiki_url:
        return None

    # Build a synthetic LensResult: 1 visual match = the Wikipedia page.
    match = LensMatch(
        rank=1,
        title=f"{wiki_article_title or title} - Wikipedia",
        link=wiki_url,
        source="Wikipedia",
        platform="wikipedia.org",
        reason=f"hint_based_search (file_name='{name_hint!r}', summary={wiki_summary[:80]!r})",
    )
    return LensResult(
        query_image_url="",
        image_sha256="",
        face_hash="",
        policy="social",
        selected=match,
        visual_matches=[match],
        knowledge_graph=match,
        candidates_by_domain={"wikipedia.org": [match]},
        visual_match_count=1,
        cached=False,
        cache_key="hint_" + __import__("hashlib").sha256(name_hint.encode()).hexdigest()[:16],
        cache_hit=False,
        serpapi_total_time_s=0.0,
        upload_ms=0.0,
        lens_ms=0.0,
        upload_host="wikipedia_api",
        raw_response_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


console = Console()


def pick_face_interactively(face_engine: FaceEngine, input_image_path: str, face_arg: int | None) -> int:
    """Show all detected faces and let the user pick one when several are found.

    Uses the arrow-key menu in interactive terminals (works in classic cmd);
    falls back to a numbered text prompt otherwise.

    Returns the selected face index (0 = largest face)."""
    faces = face_engine.detect_all_faces(input_image_path)
    if len(faces) <= 1 or face_arg is not None:
        return face_arg or 0

    # Arrow-key menu when we have a real terminal (feels like a GUI)
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            from src.menu import select_option

            labels = [
                f"Face {i}:  bbox={f['bbox']}  confidence={f['confidence']:.2f}"
                for i, f in enumerate(faces)
            ]
            return select_option(
                f"{len(faces)} faces detected — select one",
                labels,
                default=0,
                allow_esc=True,
            ) or 0
        except Exception:
            pass  # fall through to numbered prompt

    console.print(
        f"[bold yellow]! {len(faces)} faces detected — select which one to scan:[/bold yellow]"
    )
    table = Table(title="Detected Faces", border_style="yellow")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("BBox (x, y, w, h)", style="white")
    table.add_column("Confidence", style="green", justify="right")
    for i, f in enumerate(faces):
        marker = " [bold]←[/bold]" if i == 0 else ""
        table.add_row(str(i), str(f["bbox"]), f"{f['confidence']:.2f}{marker}")
    console.print(table)
    while True:
        raw = Prompt.ask("Face number", default="0")
        if raw.isdigit() and 0 <= int(raw) < len(faces):
            return int(raw)
        console.print("[red]Invalid choice — try again.[/red]")


def run_pipeline(input_image_path: str, demo_mode: bool = False,
                 face_index: int | None = None, skip_chain: bool = False,
                 show_gui: bool = True):
    # ---- Visual header + stage progress ----
    console.print(render_pipeline_header())
    if demo_mode:
        console.print(
            Panel(
                "[bold yellow]DEMO MODE[/bold yellow] — using pre-recorded search result "
                "(no SerpApi credits consumed, no internet required).",
                border_style="yellow",
            )
        )
    console.print()
    console.print(render_stage_progress(1))
    console.print()

    # ---- Pre-flight checks (validates everything before we start) ----
    if not demo_mode:
        run_preflight(input_image_path, require_serpapi=True)
    else:
        # In demo mode, only check the input image and blockchain (not Serpapi)
        from src.preflight import check_input_image, check_rpc_connection, check_contract_deployed, check_models
        check_input_image(input_image_path)
        check_models()
        w3 = check_rpc_connection()
        check_contract_deployed(w3)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "network": {"rpc_url": RPC_URL, "contract_address": CONTRACT_ADDRESS},
        "stage1": {"image": input_image_path},
        "stage2": {},
        "stage3": {},
        "stage4": {},
        "demo_mode": demo_mode,
    }

    # ---------------------------------------------------------------- Stage 1
    t_start = time.perf_counter()
    console.print("\n[bold yellow]>> [STAGE 1] Face Detection & Biometric Encoding[/bold yellow]")
    face_engine = FaceEngine()
    selected_face = pick_face_interactively(face_engine, input_image_path, face_index)
    with console.status("[bold green]Detecting facial bounds and calculating embedding vector..."):
        crop_path, face_hash, bbox, confidence, quality = face_engine.process_image(
            input_image_path, face_index=selected_face
        )

    # Show OpenCV windows with bounding box + landmarks (auto-close after 2s, skip in demo mode)
    if show_gui and not demo_mode:
        show_face_detection(input_image_path, bbox, auto_close_ms=2000)
    else:
        console.print("[dim]  (OpenCV windows skipped)[/dim]")

    # Display ASCII art face + hash panel side by side
    console.print(render_face_panel(crop_path, "Detected Face"))
    console.print(render_hash_panel(face_hash, face_engine_mod.EMBEDDING_DIM))

    elapsed = time.perf_counter() - t_start
    console.print(f"[green]✔[/green] Face detected at bbox (x, y, w, h): {bbox} (confidence: {confidence:.2f})")
    console.print(f"[green]✔[/green] Face cropped and saved to: [bold]{crop_path}[/bold]")
    console.print(
        f"[green]✔[/green] SFace Embedding dim: [bold]{face_engine_mod.EMBEDDING_DIM}[/bold] "
        "(128-d biometric vector, L2-normalized)"
    )
    console.print(f"[green]✔[/green] Biometric Hash (SHA-256): [bold cyan]{face_hash}[/bold cyan]")
    console.print(f"[dim]  Stage 1 completed in {elapsed:.2f}s[/dim]")
    report["stage1"].update(
        {"bbox": list(bbox), "embedding_dim": face_engine_mod.EMBEDDING_DIM,
         "face_hash": face_hash, "confidence": confidence, "quality_pass": quality["pass"]}
    )

    # ---------------------------------------------------------------- Stage 2
    t_stage2 = time.perf_counter()
    console.print("\n[bold yellow]>> [STAGE 2] Dynamic Web & Social Media Discovery[/bold yellow]")
    console.print(render_stage_progress(2))

    if demo_mode:
        # Use pre-recorded search result — no upload, no SerpApi call
        public_image_url = "https://files.catbox.moe/demo_face.jpg (DEMO: skipped)"
        console.print(f"[green]✔[/green] Ephemeral Image URL: [dim]{public_image_url}[/dim]")
        match_data = get_demo_search_result()
        lens_result = None  # demo mode has no real Lens result
        console.print(
            f"[green]✔[/green] Using pre-recorded search result: "
            f"[bold]{match_data['title']}[/bold]"
        )
    else:
        # ---- compute a stable image-identity hash so the SerpApi cache is
        # keyed on the source image bytes, not on the ephemeral catbox URL.
        import hashlib as _hashlib
        # The source image on disk is stable. Hashing the temp crop would
        # change every run (because FaceEngine re-encodes JPEG + applies
        # per-run bbox jitter), busting the cache.
        try:
            with open(input_image_path, "rb") as _f:
                _src_bytes_for_hash = _f.read()
            image_sha256 = _hashlib.sha256(_src_bytes_for_hash).hexdigest()
        except OSError:
            image_sha256 = ""

        # Read the two crops written by FaceEngine.process_image:
        #   temp/lens_input_enhanced.jpg  -- 60% pad + upscale + CLAHE + unsharp
        #   temp/lens_input_tight.jpg     -- 10% pad, just the face
        # Multi-crop consensus gives a much stronger first-run signal on
        # noisy webcam inputs.
        # VARIANT CASCADE: the plain 1024px padded crop is the PRIMARY Lens
        # input. Measured on live SerpApi: plain=60 matches, enhanced(2x+CLAHE
        # +unsharp)=0 matches, tight=0 matches -- Google's matcher rejects the
        # heavy enhancement. Enhanced/tight stay available as fallback variants
        # inside search_with_consensus.
        try:
            with open("temp/lens_input.jpg", "rb") as _f:
                _plain_bytes = _f.read()
        except OSError:
            _plain_bytes = b""
        enhanced_path = "temp/lens_input_enhanced.jpg"
        tight_path = "temp/lens_input_tight.jpg"
        try:
            with open(enhanced_path, "rb") as _f:
                _enhanced_bytes = _f.read()
        except OSError:
            _enhanced_bytes = b""
        try:
            with open(tight_path, "rb") as _f:
                _tight_bytes = _f.read()
        except OSError:
            _tight_bytes = b""
        if not _enhanced_bytes:
            # Fall back to the original lens_input.jpg if enhancement didn't run
            try:
                with open("temp/lens_input.jpg", "rb") as _f:
                    _enhanced_bytes = _f.read()
            except OSError:
                _enhanced_bytes = _src_bytes_for_hash
        if not _tight_bytes:
            _tight_bytes = _enhanced_bytes

        # Use a "consensus" face_hash so the two crops get distinct cache
        # entries; cache key for the consensus result uses the bare face_hash.
        consensus_face_hash = (face_hash or "") + ":consensus"

        search_engine = WebSearchEngine(SERPAPI_KEY)
        # Try the (source-image, consensus) cache first.
        # Key ties to BOTH the source image bytes AND the face_hash, so a
        # different face captured from the same source frame (e.g. two
        # faces in one webcam shot) gets its own slot and a stale match
        # can never bleed across faces.
        _consensus_key = _hashlib.sha256(
            f"consensus:{image_sha256}:{face_hash}".encode("utf-8")
        ).hexdigest()[:24]
        _consensus_cached = search_engine._get_cached(_consensus_key)
        if _consensus_cached:
            _consensus_cached["cache_key"] = _consensus_key
            lens_result = search_engine._hydrate_result(
                _consensus_cached, "<cached>", face_hash, image_sha256, _consensus_key,
            )
            public_image_url = _consensus_cached.get("query_image_url", "")
            upload_ms = 0.0
            lens_ms = 0.0
            host = "consensus_cache"
            search_engine.last_diagnostics = SearchDiagnostics(
                visual_match_count=int(_consensus_cached.get("visual_match_count", 0)),
                selected_rank=(_consensus_cached.get("selected") or {}).get("rank"),
                selected_reason="consensus cache hit",
                elapsed_seconds=0.0,
                cached=True,
            )
        elif _enhanced_bytes and _tight_bytes and _enhanced_bytes != _tight_bytes:
            # Two distinct crops available -> run multi-crop consensus.
            with console.status("[bold green]Uploading 2 face crops + querying Google Lens (consensus)..."):
                public_image_url, lens_result, upload_ms, lens_ms, host = (
                    search_engine.search_with_consensus(
                        _enhanced_bytes, _tight_bytes,
                        face_hash=face_hash,
                        policy="social",
                        plain_bytes=_plain_bytes or None,
                    )
                )
            # Cache the consensus result under a stable key tied to the source
            try:
                search_engine._put_cached(_consensus_key, lens_result.to_dict())
            except Exception:
                pass
        else:
            # Fall back to the single-crop path.
            with console.status("[bold green]Uploading face crop + querying Google Lens..."):
                public_image_url, lens_result, upload_ms, lens_ms, host = (
                    search_engine.upload_and_search(
                        _plain_bytes if _plain_bytes else (_enhanced_bytes if _enhanced_bytes else _src_bytes_for_hash),
                        face_hash=face_hash,
                        policy="social",
                    )
                )
        console.print(f"[green]✔[/green] Ephemeral Image URL: [dim]{public_image_url}[/dim]")

        match_data = lens_result  # backward-compat: dict-like access for Stage 3

        if search_engine.last_diagnostics:
            diag = search_engine.last_diagnostics
            cache_note = "cache" if diag.cached else "live SerpApi"
            console.print(
                f"[green]✔[/green] Google Lens returned [bold]{diag.visual_match_count}[/bold] visual matches "
                f"via {cache_note}; selected rank [bold]{diag.selected_rank}[/bold] "
                f"({diag.selected_reason}) in {diag.elapsed_seconds:.2f}s"
            )
        # Per-stage telemetry (one line, very low noise)
        cache_state = "HIT" if lens_result.cache_hit else "MISS"
        console.print(
            f"[dim]  STAGE2: upload={upload_ms:.0f}ms lens={lens_ms:.0f}ms "            f"serpapi_total={lens_result.serpapi_total_time_s}s host={host} cache={cache_state}[/dim]"
        )

        # Surface the identity block immediately after Stage 2.
        if lens_result is not None:
            console.print(render_identity_block(lens_result, face_hash=face_hash))

        # ------------------------------------------- Exact-image matching
        # Perceptual-hash the uploaded image and compare against every Lens
        # match thumbnail -> finds the EXACT same image on other platforms.
        # These matches also feed back into the scoring layer: any candidate
        # whose `link` appears in the exact-image results is tagged
        # "exact_image_match" so the vote scorer can apply its +200 proof-level
        # boost and reselect immediately in the same run.
        _exact = []
        try:
            from src.image_match import find_exact_matches
            from dataclasses import replace as _dc_replace
            if lens_result is not None and lens_result.visual_matches:
                with console.status("[bold green]Exact-image matching (perceptual hash over thumbnails)..."):
                    _exact = find_exact_matches(
                        input_image_path, lens_result.visual_matches)
                if _exact:
                    tops = ", ".join(
                        f"{m['platform']} (d={m['hamming']})" for m in _exact[:3])
                    console.print(
                        f"[green]✔[/green] Exact image also found on: [bold]{tops}[/bold]")

                    # Re-tag visual matches whose link was confirmed as exact
                    # so subsequent re-selections (and the audit trail) know
                    # these are pixel-identical matches, not just text matches.
                    exact_link_set = {(m.get("link") or "").lower() for m in _exact}
                    if exact_link_set:
                        new_visual = []
                        for m in lens_result.visual_matches:
                            if (m.link or "").lower() in exact_link_set:
                                tagged_reason = (m.reason or "") + " | exact_image_match"
                                new_visual.append(_dc_replace(m, reason=tagged_reason))
                            else:
                                new_visual.append(m)
                        lens_result = _dc_replace(
                            lens_result, visual_matches=new_visual)
                        # Re-run the selector on the tagged visual matches so
                        # the exact-image proof affects the CURRENT selected
                        # identity, not just future cache replays/reports.
                        try:
                            raw_visual = [m.to_dict() for m in new_visual]
                            selected, all_visual, by_domain, kg_match = search_engine._select(
                                raw_visual,
                                lens_result.knowledge_graph.to_dict() if lens_result.knowledge_graph else None,
                                lens_result.policy,
                            )
                            lens_result = _dc_replace(
                                lens_result,
                                selected=selected,
                                visual_matches=all_visual,
                                candidates_by_domain=by_domain,
                                knowledge_graph=kg_match,
                                visual_match_count=len(all_visual),
                                platform_profiles=search_engine._maybe_corroborate(selected),
                            )
                            match_data = lens_result
                        except Exception:
                            # Reporting still includes exact_image_matches even
                            # if re-selection fails for an unexpected reason.
                            pass
        except Exception as _ex_exc:
            console.print(f"[dim]  exact-image matching unavailable: {_ex_exc}[/dim]")

        # ------------------------------------------- Biometric re-verification
        # Compare the query face embedding against the FINAL selected
        # candidate's profile photos (Wikipedia portrait / og:image of social
        # profiles). This intentionally runs AFTER exact-image re-selection so
        # the biometric verdict always corresponds to the final identity.
        # Never blocks or fails the pipeline: any problem -> "UNKNOWN".
        _bio = None
        try:
            from src.biometric_verify import BiometricVerifier, candidate_photo_urls
            sel_title = str(getattr(lens_result.selected, "title", "") or "")
            if (face_engine.last_embedding is not None
                    and sel_title
                    and sel_title != "No confident identification"):
                with console.status("[bold green]Biometric re-verification (fetching candidate profile photos)..."):
                    _photo_urls = candidate_photo_urls(
                        str(getattr(lens_result.selected, "link", "") or ""),
                        lens_result.platform_profiles)
                    _bio = BiometricVerifier(face_engine).verify(
                        face_engine.last_embedding, _photo_urls)
                console.print(
                    f"[green]✔[/green] Biometric re-verification: "
                    f"[bold]{_bio['biometric_confidence']}[/bold] "
                    f"(similarity={_bio['similarity']}, photos checked={_bio['checked']})"
                )
        except Exception as _bio_exc:
            console.print(f"[dim]  biometric re-verification unavailable: {_bio_exc}[/dim]")

        # No-identity from Lens.
        if getattr(lens_result, "visual_match_count", 0) == 0:
            elapsed2 = time.perf_counter() - t_stage2
            console.print(f"[dim]  Stage 2 completed in {elapsed2:.2f}s[/dim]")
            console.print(
                "[bold yellow]⚠ Google Lens could not identify this face.[/bold yellow] "
                "Try a clearer, front-facing image of a person with public web/social presence."
            )
            report["stage2"] = (
                lens_result.to_dict() if hasattr(lens_result, "to_dict") else {}
            )
            return report

    console.print(render_comparison_panel(crop_path, match_data))

    table = Table(title="Discovered Social / Web Content", border_style="green")
    table.add_column("Property", style="cyan")
    table.add_column("Discovered Value", style="white", overflow="fold")
    table.add_row("Entity Title", str(match_data["title"]))
    table.add_row("Platform / Source", str(match_data["platform"]))
    table.add_row("Target Post URL", str(match_data["link"]))
    console.print(table)
    elapsed2 = time.perf_counter() - t_stage2
    console.print(f"[dim]  Stage 2 completed in {elapsed2:.2f}s[/dim]")

    # The report's stage2 block is now the full LensResult (selected +
    # visual_matches + knowledge_graph + per-stage telemetry) so the audit
    # trail is auditable end-to-end. Falls back to the 4-field shape if the
    # older dict-style match_data is used (demo mode).
    if hasattr(lens_result, "to_dict"):
        report["stage2"] = lens_result.to_dict()
    else:
        report["stage2"].update(
            {
                "image_host_url": public_image_url,
                "title": str(match_data["title"]),
                "platform": str(match_data["platform"]),
                "post_url": str(match_data["link"]),
            }
        )
    if _bio:
        report["stage2"]["biometric"] = _bio
    if _exact:
        report["stage2"]["exact_image_matches"] = _exact

    # ---------------------------------------------------------------- Stage 3
    if skip_chain:
        # Chain-skipped mode (used by accuracy eval / rapid iteration):
        # nothing is anchored on-chain; the report is still emitted.
        elapsed2 = time.perf_counter() - t_stage2
        console.print(f"[dim]  Stage 2 completed in {elapsed2:.2f}s[/dim]")
        report["stage3"] = {
            "fingerprint": "",
            "already_anchored": False,
            "skipped": True,
            "reason": "chain skipped (--no-chain)",
        }
        report["stage4"] = {
            "verification": "SKIPPED",
            "tamper_detected": False,
            "skipped": True,
            "reason": "chain skipped (--no-chain)",
        }
        report["network"]["chain_id"] = "skipped"
        report_path = write_report(report)
        console.print(
            f"\n[bold yellow]>> Audit report saved to:[/bold yellow] [bold]{report_path}[/bold]"
        )
        console.print(
            Panel.fit(
                "[bold green]PIPELINE COMPLETED (chain skipped)[/bold green] "
                "[dim]-- face detection + web search + audit report[/dim]",
                border_style="green",
            )
        )
        if lens_result is not None:
            console.print(render_identity_block(lens_result, face_hash=face_hash))
        return report

    t_stage3 = time.perf_counter()
    console.print(
        "\n[bold yellow]>> [STAGE 3] Cryptographic Fingerprinting & Blockchain Anchoring[/bold yellow]"
    )
    console.print(render_stage_progress(3))
    bc_manager = BlockchainManager(RPC_URL, PRIVATE_KEY, CONTRACT_ADDRESS)
    data_fingerprint = bc_manager.compute_data_fingerprint(face_hash, str(match_data["link"]))
    console.print(
        f"[green]✔[/green] Combined Canonical Fingerprint: [bold magenta]{data_fingerprint}[/bold magenta]"
    )

    if bc_manager.record_exists(face_hash):
        console.print(
            "[yellow]![/yellow] Record already anchored for this face hash — "
            "skipping registration and proceeding to verification."
        )
        report["stage3"] = {
            "fingerprint": data_fingerprint,
            "already_anchored": True,
        }
    else:
        with console.status("[bold green]Broadcasting transaction to blockchain..."):
            receipt = bc_manager.anchor_record(face_hash, str(match_data["link"]), data_fingerprint)
        tx_hash = f"0x{receipt.transactionHash.hex()}"
        block = int(receipt.blockNumber)
        gas = int(receipt.gasUsed)
        console.print(
            f"[green]✔[/green] Transaction Confirmed! TxHash: [bold cyan]{tx_hash}[/bold cyan]"
        )
        console.print(
            f"[green]✔[/green] Included in Block Number: [bold]{block}[/bold] | "
            f"Gas Used: {gas}"
        )
        # Show blockchain panel
        console.print(render_blockchain_panel(tx_hash, block, gas))
        time.sleep(0.2)
        report["stage3"] = {
            "fingerprint": data_fingerprint,
            "already_anchored": False,
            "tx_hash": tx_hash,
            "block": block,
            "gas_used": gas,
        }

    elapsed3 = time.perf_counter() - t_stage3
    console.print(f"[dim]  Stage 3 completed in {elapsed3:.2f}s[/dim]")

    # ---------------------------------------------------------------- Stage 4
    t_stage4 = time.perf_counter()
    console.print(
        "\n[bold yellow]>> [STAGE 4] On-Chain Audit & Tamper-Evidence Proof[/bold yellow]"
    )
    console.print(render_stage_progress(4))
    with console.status("[bold green]Querying smart contract for re-verification..."):
        audit_result = bc_manager.verify_on_chain(face_hash, str(match_data["link"]))

    # Show verification panel
    console.print(
        render_verification_panel(
            audit_result["valid"],
            audit_result["on_chain_data_hash"],
            audit_result["local_recomputed_hash"],
        )
    )

    if audit_result["valid"]:
        console.print(
            Panel(
                "[bold green]STATUS: CRYPTOGRAPHIC VERIFICATION PASSED\n"
                "On-chain fingerprint matches the local record perfectly![/bold green]",
                border_style="green",
            )
        )
    else:
        console.print(Panel("[bold red]STATUS: VERIFICATION FAILED[/bold red]", border_style="red"))
    report["stage4"].update(
        {
            "verification": "PASSED" if audit_result["valid"] else "FAILED",
            "on_chain_fingerprint": audit_result["on_chain_data_hash"],
        }
    )

    # ------------------------------------------------- Tamper-evidence drill
    console.print(
        "\n[bold yellow]>> [SECURITY DRILL] Demonstrating Tamper-Evidence "
        "(Altering 1 URL Character)[/bold yellow]"
    )
    tampered_url = str(match_data["link"]) + "_tampered"
    tamper_audit = bc_manager.verify_on_chain(face_hash, tampered_url)

    # Show tamper panel
    console.print(
        render_verification_panel(
            tamper_audit["valid"],
            tamper_audit["on_chain_data_hash"],
            tamper_audit["local_recomputed_hash"],
        )
    )

    if not tamper_audit["valid"]:
        console.print("[bold red]✔ TAMPER DETECTED BY DESIGN![/bold red]")
        console.print(f"Expected Fingerprint : {tamper_audit['local_recomputed_hash']}")
        console.print(f"On-Chain Fingerprint : {tamper_audit['on_chain_data_hash']}")
        console.print(
            "[dim]Any tampering of the discovered social data is immediately exposed.[/dim]"
        )
    else:
        console.print("[bold red]!! Tamper drill unexpectedly passed -- audit the system![/bold red]")
    report["stage4"].update(
        {
            "tamper_detected": not tamper_audit["valid"],
            "tampered_url": tampered_url,
            "tamper_local_hash": tamper_audit["local_recomputed_hash"],
            "tamper_on_chain_hash": tamper_audit["on_chain_data_hash"],
        }
    )

    elapsed4 = time.perf_counter() - t_stage4
    console.print(f"[dim]  Stage 4 completed in {elapsed4:.2f}s[/dim]")

    # ------------------------------------------------- Audit report artifact
    report["network"]["chain_id"] = bc_manager.w3.eth.chain_id
    report_path = write_report(report)
    console.print(f"\n[bold yellow]>> Audit report saved to:[/bold yellow] [bold]{report_path}[/bold]")

    # ---- Total pipeline timing ----
    total_elapsed = time.perf_counter() - t_start
    timing_summary = (
        f"[bold cyan]Pipeline timing summary[/bold cyan]\n"
        f"  Stage 1 (Face Detection):    {elapsed:.2f}s\n"
        f"  Stage 2 (Web Search):        {elapsed2:.2f}s\n"
        f"  Stage 3 (Blockchain Anchor): {elapsed3:.2f}s\n"
        f"  Stage 4 (Verification):      {elapsed4:.2f}s\n"
        f"  ─────────────────────────────────\n"
        f"  [bold]Total: {total_elapsed:.2f}s[/bold]"
    )
    console.print(Panel(timing_summary, border_style="cyan"))

    # The big identity+anchor+tamper result box, with per-stage timing subtitle.
    # Always render the rich result box when we have a LensResult.
    # On cache hits (already_anchored), the previous run'''s tx_hash
    # is what we use, so the user still sees the full result.
    if lens_result is not None:
        console.print(
            render_final_summary(
                lens_result,
                face_hash=face_hash,
                on_chain_fingerprint=report["stage3"].get("fingerprint", ""),
                tx_hash=report["stage3"].get("tx_hash"),
                block=report["stage3"].get("block"),
                gas=report["stage3"].get("gas_used"),
                verification_passed=report["stage4"].get("verification") == "PASSED",
                tamper_detected=report["stage4"].get("tamper_detected", False),
                tamper_local_hash=report["stage4"].get("tamper_local_hash"),
                tamper_on_chain_hash=report["stage4"].get("tamper_on_chain_hash"),
                total_seconds=total_elapsed,
            )
        )
    else:
        console.print(
            Panel.fit(
                "[bold green]PIPELINE COMPLETED SUCCESSFULLY END-TO-END[/bold green]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    # Parse --demo / --face N flags and image path from arguments
    args = sys.argv[1:]
    demo_mode = "--demo" in args or "--demo-mode" in args
    face_index = None
    if "--face" in args:
        idx = args.index("--face")
        if idx + 1 < len(args) and args[idx + 1].isdigit():
            face_index = int(args[idx + 1])
            del args[idx : idx + 2]
        else:
            console.print("[bold red]--face requires a number, e.g. --face 0[/bold red]")
            sys.exit(1)
    args = [a for a in args if a not in ("--demo", "--demo-mode")]

    if len(args) < 1:
        console.print(
            "[bold red]Usage: python pipeline.py <path_to_input_image> [--demo] [--face N][/bold red]\n\n"
            "Options:\n"
            "  --demo      Use pre-recorded search result (no SerpApi, no internet)\n"
            "  --face N    Scan face #N in multi-face images (skips interactive picker)\n\n"
            "Tip: the unified CLI is 'python main.py' — try 'python main.py --help'."
        )
        sys.exit(1)
    run_pipeline(args[0], demo_mode=demo_mode, face_index=face_index)

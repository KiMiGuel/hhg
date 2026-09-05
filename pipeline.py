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
from src.web_search import WebSearchEngine

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


def run_pipeline(input_image_path: str, demo_mode: bool = False, face_index: int | None = None):
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
    if not demo_mode:
        show_face_detection(input_image_path, bbox, auto_close_ms=2000)
    else:
        console.print("[dim]  (OpenCV windows skipped in demo mode)[/dim]")

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
        # Prefer the lens-friendly context image (60% padded, 1024 long edge, q=95)
        # written by FaceEngine._write_lens_input. Fall back to the tight 15%
        # crop if the lens-friendly file is missing (e.g. demo mode or older run).
        lens_input_path = "temp/lens_input.jpg"
        src_path = lens_input_path if os.path.exists(lens_input_path) else crop_path
        try:
            with open(src_path, "rb") as _f:
                _image_bytes = _f.read()
            image_sha256 = _hashlib.sha256(_image_bytes).hexdigest()
        except OSError:
            image_sha256 = ""

        search_engine = WebSearchEngine(SERPAPI_KEY)
        # upload_and_search is cache-aware: if (face_hash, image_sha256) is in
        # the on-disk JSON cache, the upload and the SerpApi call are skipped
        # entirely and we return the cached LensResult in ~5 ms.
        with console.status("[bold green]Uploading face crop + querying Google Lens..."):
            public_image_url, lens_result, upload_ms, lens_ms, host = (
                search_engine.upload_and_search(
                    _image_bytes if _image_bytes else b"",
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

    # ---------------------------------------------------------------- Stage 3
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

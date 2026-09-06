"""Unified CLI for the Face Identification & Blockchain Verification pipeline.

Launch with no arguments for the full-app interactive dashboard (arrow-key
menus, live system status), or use the direct subcommands:
    python main.py live [--image IMG|--camera]        one-click REAL live run
    python main.py run <image> [--demo] [--face N]   full 4-stage pipeline
    python main.py verify <hash> <url>               audit by face hash
    python main.py verify-image <image> <url>        re-derive hash, then audit
    python main.py records                           list all on-chain records
    python main.py export [--format csv|json]        export registry for auditors
    python main.py setup                             interactive setup wizard
    python main.py smoke                             end-to-end smoke test
    python main.py ui                                force-launch the dashboard
"""

import argparse
import sys

# Force UTF-8 output so ✔/✖ render on Windows consoles with legacy codepages
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel

console = Console()

BANNER = (
    "[bold cyan]Face Identification & Blockchain Verification[/bold cyan]  "
    "[dim](HH Goa 2026 — Task 3)[/dim]"
)


def _download_image_url(url: str, dest: str) -> str:
    """Download an image from any public URL (Instagram/YouTube/Facebook CDN,
    news articles, Wikipedia, any direct image link).

    Raises ValueError when the URL does not serve an image.
    """
    import os
    from datetime import datetime

    import requests

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        )
    }
    resp = requests.get(url, timeout=30, headers=headers)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    is_img = "image" in ctype or resp.content[:3] in (b"\xff\xd8\xff", b"\x89PNG")
    if not is_img:
        raise ValueError(f"URL does not point to an image (content-type={ctype or 'unknown'})")
    if not dest:
        dest = os.path.join(
            "data", "captured_from_url", f"url_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "wb") as f:
        f.write(resp.content)
    return dest


def cmd_run(args):
    from pipeline import run_pipeline

    image = getattr(args, "image", None)
    if getattr(args, "url", None):
        with console.status("Downloading image from URL..."):
            image = _download_image_url(args.url, "")
        console.print(f"[green]✔[/green] Downloaded image -> [bold]{image}[/bold]")
    if not image:
        console.print("[red]✖ Provide an image path or --url <image-url>[/red]")
        sys.exit(2)
    run_pipeline(image, demo_mode=args.demo, face_index=args.face)


def cmd_live(args):
    from scripts.live_run import main as live_main

    argv_backup = sys.argv[:]
    try:
        live_args = ["live_run.py"]
        if args.image:
            live_args.extend(["--image", args.image])
        if getattr(args, "image_url", None):
            with console.status("Downloading image from URL..."):
                downloaded = _download_image_url(args.image_url, "")
            console.print(f"[green]✔[/green] Downloaded image -> [bold]{downloaded}[/bold]")
            live_args.extend(["--image", downloaded])
        if args.camera:
            live_args.append("--camera")
        if args.camera_index is not None:
            live_args.extend(["--camera-index", str(args.camera_index)])
        if args.output:
            live_args.extend(["--output", args.output])
        if args.face is not None:
            live_args.extend(["--face", str(args.face)])
        if args.fresh_chain:
            live_args.append("--fresh-chain")
        if args.visible_node:
            live_args.append("--visible-node")
        if args.no_auto_node:
            live_args.append("--no-auto-node")
        if args.force_deploy:
            live_args.append("--force-deploy")
        if args.no_chain:
            live_args.append("--no-chain")
        sys.argv = live_args
        code = live_main()
        if code:
            sys.exit(code)
    finally:
        sys.argv = argv_backup


def cmd_verify(args):
    from verify import verify_entry

    verify_entry(args.face_hash, args.url)


def cmd_verify_image(args):
    """Re-derive the biometric hash from an image, then audit against the URL."""
    from src.blockchain import BlockchainManager
    from src.config import CONTRACT_ADDRESS, PRIVATE_KEY, RPC_URL
    from src.face_engine import FaceEngine
    from web3.exceptions import ContractLogicError

    console.print("\n[bold]Deriving biometric hash from image...[/bold]")
    engine = FaceEngine()
    _, face_hash, bbox, confidence, _ = engine.process_image(args.image)
    console.print(f"  Face at {bbox} (confidence {confidence:.2f})")
    console.print(f"  [cyan]Face hash: {face_hash}[/cyan]")

    bc = BlockchainManager(RPC_URL, PRIVATE_KEY, CONTRACT_ADDRESS)
    try:
        result = bc.verify_on_chain(face_hash, args.url)
    except ContractLogicError:
        console.print(
            Panel(
                "[bold yellow]✖ No on-chain record for this image's face hash.\n"
                "Run the pipeline first: python main.py run <image>[/bold yellow]",
                border_style="yellow",
            )
        )
        sys.exit(2)

    if result["valid"]:
        console.print(
            Panel(
                f"[bold green]✔ IMAGE IDENTITY VERIFIED ON-CHAIN\n"
                f"On-chain URL : {result['on_chain_url']}\n"
                f"Fingerprint  : {result['on_chain_data_hash']}\n"
                f"Block Time   : {result['block_timestamp']}[/bold green]",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                "[bold red]✖ MISMATCH — this image's hash does not match the anchored "
                "fingerprint for that URL.[/bold red]",
                border_style="red",
            )
        )
        sys.exit(1)


def cmd_records(_args):
    from src.blockchain import BlockchainManager
    from src.config import CONTRACT_ADDRESS, PRIVATE_KEY, RPC_URL
    from src.registry import fetch_all_records, render_records_table

    bc = BlockchainManager(RPC_URL, PRIVATE_KEY, CONTRACT_ADDRESS)
    with console.status("[dim]Querying RecordRegistered events..."):
        records = fetch_all_records(bc)
    render_records_table(records)


def cmd_export(args):
    from src.blockchain import BlockchainManager
    from src.config import CONTRACT_ADDRESS, PRIVATE_KEY, RPC_URL
    from src.registry import export_records, fetch_all_records

    bc = BlockchainManager(RPC_URL, PRIVATE_KEY, CONTRACT_ADDRESS)
    with console.status("[dim]Fetching on-chain records..."):
        records = fetch_all_records(bc)
    if not records:
        console.print("[yellow]Nothing to export — no records anchored yet.[/yellow]")
        return
    path = export_records(records, fmt=args.format)
    console.print(f"[green]✔[/green] Exported {len(records)} record(s) to [bold]{path}[/bold]")


def cmd_setup(_args):
    from src.setup_wizard import run_setup

    run_setup()


def cmd_smoke(_args):
    import subprocess

    result = subprocess.run([sys.executable, "scripts/smoke_test.py"])
    sys.exit(result.returncode)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Face Identification & Blockchain Verification pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py live --image data/sample_face.jpg\n"
            "  python main.py live --camera\n"
            "  python main.py run data/sample_face.jpg --demo\n"
            "  python main.py run data/sample_face.jpg --face 1\n"
            "  python main.py records\n"
            "  python main.py verify-image data/sample_face.jpg https://youtube.com/watch?v=x\n"
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    live_p = sub.add_parser(
        "live", help="one-click REAL live run (camera/image + SerpApi + blockchain)"
    )
    live_p.add_argument("--image", help="path to input face image")
    live_p.add_argument(
        "--image-url",
        dest="image_url",
        help="download the face image from a public URL "
        "(Instagram/YouTube/Facebook/news images) and run on it",
    )
    live_p.add_argument("--camera", action="store_true", help="capture from webcam before running")
    live_p.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    live_p.add_argument(
        "--output", default="data/captured_face.jpg", help="camera capture output path"
    )
    live_p.add_argument("--face", type=int, default=None, help="face index for multi-face images")
    live_p.add_argument(
        "--fresh-chain", action="store_true", help="restart local Anvil and redeploy before running"
    )
    live_p.add_argument(
        "--visible-node", action="store_true", help="start Anvil visibly if auto-starting"
    )
    live_p.add_argument("--no-auto-node", action="store_true", help="do not auto-start local Anvil")
    live_p.add_argument(
        "--force-deploy",
        action="store_true",
        help="redeploy FaceRegistry even if current address is valid",
    )
    live_p.add_argument(
        "--no-chain",
        action="store_true",
        help="skip Anvil/blockchain entirely (face + web search only)",
    )
    live_p.set_defaults(func=cmd_live)

    run_p = sub.add_parser("run", help="run the full 4-stage pipeline on an image")
    run_p.add_argument("image", nargs="?", help="path to the input face image")
    run_p.add_argument(
        "--url",
        dest="url",
        default=None,
        help="download the face image from a public URL instead of a local path",
    )
    run_p.add_argument(
        "--demo", action="store_true", help="use pre-recorded search result (no SerpApi/internet)"
    )
    run_p.add_argument(
        "--face", type=int, default=None, help="face index for multi-face images (skips picker)"
    )
    run_p.set_defaults(func=cmd_run)

    ver_p = sub.add_parser("verify", help="audit a record by face hash + URL")
    ver_p.add_argument("face_hash", help="0x… biometric hash from a pipeline run")
    ver_p.add_argument("url", help="the social post URL that was anchored")
    ver_p.set_defaults(func=cmd_verify)

    vi_p = sub.add_parser("verify-image", help="re-derive the hash from an image, then audit")
    vi_p.add_argument("image", help="path to the face image")
    vi_p.add_argument("url", help="the social post URL to verify against")
    vi_p.set_defaults(func=cmd_verify_image)

    rec_p = sub.add_parser("records", help="list every record anchored on-chain")
    rec_p.set_defaults(func=cmd_records)

    exp_p = sub.add_parser("export", help="export the on-chain registry")
    exp_p.add_argument("--format", choices=["csv", "json"], default="csv")
    exp_p.set_defaults(func=cmd_export)

    setup_p = sub.add_parser("setup", help="interactive setup wizard")
    setup_p.set_defaults(func=cmd_setup)

    smoke_p = sub.add_parser("smoke", help="run the end-to-end smoke test")
    smoke_p.set_defaults(func=cmd_smoke)

    ui_p = sub.add_parser(
        "ui", help="launch the interactive dashboard (also the default with no arguments)"
    )
    ui_p.set_defaults(func=cmd_ui)

    return p


def cmd_ui(_args):
    from src.app import run_dashboard

    run_dashboard()


def main():
    # No arguments: interactive users get the dashboard, scripts get help.
    if len(sys.argv) == 1:
        if sys.stdin.isatty() and sys.stdout.isatty():
            cmd_ui(None)
            return
        build_parser().print_help()
        sys.exit(1)

    parser = build_parser()
    args = parser.parse_args()
    console.print(BANNER)
    console.print()
    try:
        args.func(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except ConnectionError as e:
        console.print(
            Panel(
                f"[bold red]✖ Blockchain node not reachable.[/bold red]\n\n{e}\n\n"
                "[dim]Start it with: anvil   (or re-run: python main.py setup)[/dim]",
                border_style="red",
            )
        )
        sys.exit(1)
    except FileNotFoundError as e:
        console.print(Panel(f"[bold red]✖ {e}[/bold red]", border_style="red"))
        sys.exit(1)
    except RuntimeError as e:
        console.print(Panel(f"[bold red]✖ {e}[/bold red]", border_style="red"))
        sys.exit(1)


if __name__ == "__main__":
    main()

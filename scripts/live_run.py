"""One-click REAL pipeline runner.

This is the production path, not demo mode:

    image/camera -> preflight -> live SerpApi Google Lens -> blockchain anchor -> verify

It is built for a hackathon demo: by default it will ensure a local Anvil node
is reachable and deploy FaceRegistry automatically if the configured contract
address is missing or stale.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from web3 import Web3

console = Console()


def _env_path() -> Path:
    return ROOT / ".env"


def _read_env() -> dict[str, str]:
    load_dotenv(_env_path(), override=True)
    return {
        "SERPAPI_KEY": os.getenv("SERPAPI_KEY", "").strip().strip('"').strip("'"),
        "RPC_URL": os.getenv("RPC_URL", "http://127.0.0.1:8545").strip().strip('"').strip("'"),
        "PRIVATE_KEY": os.getenv(
            "PRIVATE_KEY",
            "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        ).strip().strip('"').strip("'"),
        "CONTRACT_ADDRESS": os.getenv("CONTRACT_ADDRESS", "").strip().strip('"').strip("'"),
    }


def _write_env_value(key: str, value: str) -> None:
    path = _env_path()
    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    found = False
    for line in lines:
        if line.startswith(f"{key}="):
            out.append(f'{key}="{value}"')
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f'{key}="{value}"')
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    os.environ[key] = value


def _anvil_exe() -> str:
    candidates = [
        ROOT / "anvil.exe",
        Path.home() / ".foundry" / "bin" / "anvil.exe",
        Path.home() / ".foundry" / "bin" / "anvil",
        Path("anvil"),
    ]
    for candidate in candidates:
        if str(candidate) == "anvil" or candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "Anvil executable not found. Install Foundry or run scripts/run_local_node.ps1 manually."
    )


def _connect(rpc_url: str) -> Web3:
    return Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 5}))


def _wait_for_rpc(rpc_url: str, seconds: int = 15) -> Web3:
    deadline = time.time() + seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            w3 = _connect(rpc_url)
            if w3.is_connected():
                return w3
        except Exception as exc:  # pragma: no cover - depends on local node timing
            last_error = exc
        time.sleep(0.5)
    raise ConnectionError(f"RPC not reachable at {rpc_url}: {last_error or 'timeout'}")


def ensure_anvil(rpc_url: str, auto_start: bool = True, fresh: bool = False, visible: bool = False) -> Web3:
    """Ensure local Anvil is reachable. Public/testnet RPCs are never auto-started."""
    is_local = "127.0.0.1" in rpc_url or "localhost" in rpc_url
    if not is_local:
        w3 = _wait_for_rpc(rpc_url, seconds=8)
        console.print(f"[green]OK[/green] RPC reachable: {rpc_url} (chain {w3.eth.chain_id})")
        return w3

    if fresh:
        subprocess.run(["taskkill", "/IM", "anvil.exe", "/F"], capture_output=True, text=True)
        time.sleep(1)

    try:
        w3 = _wait_for_rpc(rpc_url, seconds=2)
        console.print(f"[green]OK[/green] Local Anvil already running (block {w3.eth.block_number})")
        return w3
    except Exception:
        if not auto_start:
            raise

    exe = _anvil_exe()
    console.print(f"[cyan]Starting Anvil automatically:[/cyan] {exe}")
    stdout = None if visible else subprocess.DEVNULL
    stderr = None if visible else subprocess.DEVNULL
    subprocess.Popen([exe, "--host", "127.0.0.1", "--port", "8545"], stdout=stdout, stderr=stderr)
    w3 = _wait_for_rpc(rpc_url, seconds=15)
    console.print(f"[green]OK[/green] Local Anvil ready (chain {w3.eth.chain_id})")
    return w3


def _contract_has_code(w3: Web3, address: str) -> bool:
    if not address:
        return False
    try:
        return len(w3.eth.get_code(Web3.to_checksum_address(address))) > 0
    except Exception:
        return False


def ensure_contract(w3: Web3, force_deploy: bool = False) -> str:
    """Deploy FaceRegistry if .env address is empty/stale, and update .env."""
    env = _read_env()
    current = env["CONTRACT_ADDRESS"]
    if current and not force_deploy and _contract_has_code(w3, current):
        console.print(f"[green]OK[/green] Contract live at {current}")
        return current

    console.print("[cyan]Deploying FaceRegistry automatically...[/cyan]")
    from scripts.deploy import deploy

    address = deploy()
    _write_env_value("CONTRACT_ADDRESS", address)
    console.print(f"[green]OK[/green] .env updated: CONTRACT_ADDRESS={address}")
    return address


def resolve_image(args: argparse.Namespace) -> str:
    """Return the image path to use, optionally capturing from webcam."""
    if args.camera:
        from src.camera import capture_from_camera

        console.print(Panel("[bold cyan]Camera capture mode[/bold cyan]\nSPACE = capture, ESC = cancel"))
        captured = capture_from_camera(output_path=args.output, camera_index=args.camera_index)
        if not captured:
            raise RuntimeError("Camera capture cancelled or failed.")
        return captured

    if args.image:
        return args.image

    default = ROOT / "data" / "sample_face.jpg"
    if default.exists():
        return str(default)
    raise FileNotFoundError("No image supplied. Use --image <path> or --camera.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ONE-CLICK LIVE: camera/image -> SerpApi Google Lens -> blockchain proof",
    )
    parser.add_argument("--image", help="path to input face image")
    parser.add_argument("--camera", action="store_true", help="capture a photo from webcam before running")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--output", default=str(ROOT / "data" / "captured_face.jpg"), help="camera capture output path")
    parser.add_argument("--face", type=int, default=None, help="face index for multi-face images")
    parser.add_argument("--fresh-chain", action="store_true", help="restart local Anvil and redeploy for a clean demo")
    parser.add_argument("--visible-node", action="store_true", help="start Anvil with a visible console window if possible")
    parser.add_argument("--no-auto-node", action="store_true", help="do not auto-start local Anvil")
    parser.add_argument("--force-deploy", action="store_true", help="redeploy FaceRegistry even if current address has code")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        env = _read_env()
        if not env["SERPAPI_KEY"]:
            raise RuntimeError("SERPAPI_KEY is missing in .env; live Google Lens search cannot run.")

        console.print(
            Panel.fit(
                "[bold green]ONE-CLICK LIVE PIPELINE[/bold green]\n"
                "Real camera/image + live SerpApi Google Lens + blockchain verification",
                border_style="green",
            )
        )

        w3 = ensure_anvil(
            env["RPC_URL"],
            auto_start=not args.no_auto_node,
            fresh=args.fresh_chain,
            visible=args.visible_node,
        )
        ensure_contract(w3, force_deploy=args.force_deploy or args.fresh_chain)

        # Reload dotenv-aware config after possible CONTRACT_ADDRESS update.
        load_dotenv(_env_path(), override=True)
        image = resolve_image(args)
        console.print(Panel.fit(f"[bold cyan]LIVE RUN STARTING[/bold cyan]\nImage: {image}"))

        from pipeline import run_pipeline

        run_pipeline(image, demo_mode=False, face_index=args.face)
        return 0
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        return 130
    except Exception as exc:
        # Print the full traceback to stderr (and capture it for the panel)
        # so the next time something breaks we can see WHERE in the codebase
        # the error happened, not just the exception message.
        import traceback as _tb
        tb_text = _tb.format_exc()
        try:
            print(tb_text, file=sys.stderr, flush=True)
        except Exception:
            pass
        console.print(
            Panel(
                f"[bold red]LIVE RUN FAILED[/bold red]\n\n{exc}\n\n"
                f"[dim]{tb_text.splitlines()[-3] if tb_text else ''}[/dim]\n\n"
                "[dim]Try: python main.py live --image data/sample_face.jpg --fresh-chain[/dim]",
                border_style="red",
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

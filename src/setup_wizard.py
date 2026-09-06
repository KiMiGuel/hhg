"""Interactive setup wizard: checks deps, models, node, contract."""

import os
import subprocess
import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

console = Console()


def _check_python_deps() -> bool:
    """Verify all required Python packages are importable."""
    required = ["cv2", "numpy", "requests", "web3", "solcx", "dotenv", "rich"]
    missing = []
    for mod in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        console.print(f"[red]✖ Missing packages: {', '.join(missing)}[/red]")
        console.print("[dim]  Fix: pip install -r requirements.txt[/dim]")
        return False
    console.print("[green]✔[/green] All Python packages installed")
    return True


def _check_models() -> bool:
    """Verify YuNet + SFace ONNX models exist."""
    from src.face_engine import SFACE_MODEL_PATH, YUNET_MODEL_PATH

    ok = True
    for name, path in [("YuNet", YUNET_MODEL_PATH), ("SFace", SFACE_MODEL_PATH)]:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            console.print(f"[green]✔[/green] {name} model: {path}")
        else:
            console.print(f"[red]✖[/red] {name} model missing: {path}")
            ok = False
    return ok


def _check_node() -> bool:
    """Verify the blockchain node is reachable."""
    from web3 import Web3

    from src.config import RPC_URL

    try:
        w3 = Web3(Web3.HTTPProvider(RPC_URL))
        if w3.is_connected():
            console.print(
                f"[green]✔[/green] Node reachable: {RPC_URL} (block {w3.eth.block_number})"
            )
            return True
        else:
            console.print(f"[red]✖[/red] Node not responding: {RPC_URL}")
            console.print("[dim]  Fix: .\\scripts\\run_local_node.ps1[/dim]")
            return False
    except Exception as e:
        console.print(f"[red]✖[/red] Node error: {e}[/dim]")
        return False


def _check_contract() -> bool:
    """Verify the contract is deployed."""
    from src.config import CONTRACT_ADDRESS

    if CONTRACT_ADDRESS:
        console.print(f"[green]✔[/green] Contract: {CONTRACT_ADDRESS[:12]}…")
        return True
    console.print("[yellow]–[/yellow] Contract not deployed")
    console.print("[dim]  Fix: python scripts/deploy.py[/dim]")
    return False


def _check_serpapi() -> bool:
    """Verify SerpApi key is set."""
    from src.config import SERPAPI_KEY

    if SERPAPI_KEY:
        console.print("[green]✔[/green] SerpApi key set (live search ready)")
        return True
    console.print("[yellow]–[/yellow] SerpApi key not set (demo mode only)")
    return False


def run_setup():
    """Run the full interactive setup wizard."""
    console.print(Panel.fit("[bold cyan]Setup Wizard[/bold cyan]", border_style="cyan"))

    console.print("\n[bold]1. Python dependencies[/bold]")
    deps_ok = _check_python_deps()

    console.print("\n[bold]2. ML models[/bold]")
    models_ok = _check_models()

    console.print("\n[bold]3. Blockchain node[/bold]")
    node_ok = _check_node()

    console.print("\n[bold]4. Smart contract[/bold]")
    contract_ok = _check_contract()

    console.print("\n[bold]5. SerpApi key[/bold]")
    serpapi_ok = _check_serpapi()

    console.print("\n[bold]6. Sample image[/bold]")
    sample = os.environ.get("HHG_SAMPLE_IMAGE", "data/sample_face.jpg")
    if os.path.exists(sample):
        console.print(f"[green]✔[/green] Sample image: {sample}")
    else:
        console.print(f"[yellow]–[/yellow] Sample image missing: {sample}")

    # Summary
    console.print()
    all_ok = deps_ok and models_ok and node_ok and contract_ok
    if all_ok:
        console.print(
            Panel(
                "[bold green]✔ System ready — all checks passed![/bold green]", border_style="green"
            )
        )
    else:
        console.print(
            Panel(
                "[bold yellow]⚠ Some checks failed — see above[/bold yellow]", border_style="yellow"
            )
        )

    # Offer to fix
    if not all_ok and Confirm.ask("\nAttempt automatic fixes?", default=False):
        if not node_ok:
            console.print("[dim]Starting local node...[/dim]")
            try:
                subprocess.Popen(
                    [
                        os.path.join(os.path.expanduser("~"), ".foundry", "bin", "anvil.exe"),
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "8545",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                console.print("[green]✔[/green] Anvil started in background")
            except Exception as e:
                console.print(f"[red]✖ Could not start Anvil: {e}[/red]")

        if not contract_ok and node_ok:
            console.print("[dim]Deploying contract...[/dim]")
            try:
                result = subprocess.run(
                    [sys.executable, "scripts/deploy.py"], capture_output=True, text=True
                )
                if result.returncode == 0:
                    console.print("[green]✔[/green] Contract deployed")
                else:
                    console.print(f"[red]✖ Deploy failed: {result.stderr[:200]}[/red]")
            except Exception as e:
                console.print(f"[red]✖ Deploy error: {e}[/red]")

    return all_ok

"""One-click demo: runs the full pipeline in demo mode (no SerpApi, no internet).

Usage: python scripts/demo.py
"""

import os
import sys

# Force UTF-8 output for Windows console compatibility
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.panel import Panel

console = Console()

BANNER = "[bold cyan]DEMO MODE - Full Pipeline Walkthrough[/bold cyan]"


def main():
    console.print(BANNER)
    console.print()
    console.print(
        Panel(
            "[dim]This demo runs the complete 4-stage pipeline using a\n"
            "pre-recorded search result — no SerpApi credits consumed,\n"
            "no internet required. Perfect for the hackathon recording.[/dim]",
            border_style="dim",
        )
    )

    # Check prerequisites
    if not os.path.exists("data/sample_face.jpg"):
        console.print("[red]✖ Sample image missing: data/sample_face.jpg[/red]")
        console.print("[dim]  Download a face image or capture one from the dashboard.[/dim]")
        sys.exit(1)

    # Check node
    from web3 import Web3
    from src.config import RPC_URL

    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        console.print(f"[red]✖ Node not reachable: {RPC_URL}[/red]")
        console.print("[dim]  Start it: .\\scripts\\run_local_node.ps1[/dim]")
        sys.exit(1)

    # Check contract
    from src.config import CONTRACT_ADDRESS

    if not CONTRACT_ADDRESS:
        console.print("[yellow]⚠ Contract not deployed — deploying now...[/yellow]")
        from scripts.deploy import deploy
        deploy()

    console.print("\n[bold green]✔ All prerequisites met — starting pipeline...[/bold green]\n")

    # Run the pipeline in demo mode
    from pipeline import run_pipeline

    run_pipeline("data/sample_face.jpg", demo_mode=True)


if __name__ == "__main__":
    main()

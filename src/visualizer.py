"""ASCII-art + Rich visual helpers for the pipeline output."""

import os

from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def render_pipeline_header() -> Panel:
    """Top-level pipeline header with ASCII branding."""
    banner = Text(
        "  ╔══════════════════════════════════════════════════╗\n"
        "  ║   FACE IDENTIFICATION & BLOCKCHAIN VERIFICATION  ║\n"
        "  ╚══════════════════════════════════════════════════╝",
        style="bold cyan",
    )
    return Panel(banner, border_style="cyan", padding=(1, 2))


def render_stage_progress(stage: int) -> Text:
    """Show a 4-stage progress indicator."""
    stages = ["Face Detection", "Web Search", "Blockchain", "Verification"]
    parts = []
    for i, name in enumerate(stages, 1):
        if i < stage:
            parts.append(f"[green]✔ {name}[/green]")
        elif i == stage:
            parts.append(f"[bold reverse cyan] → {name} [/bold reverse cyan]")
        else:
            parts.append(f"[dim]  {name}[/dim]")
    return Text(" → ".join(parts))


def render_face_panel(crop_path: str, title: str = "Detected Face") -> Panel:
    """Show face crop info as a panel."""
    if os.path.exists(crop_path):
        size = os.path.getsize(crop_path)
        body = f"[cyan]{crop_path}[/cyan]\n[dim]size: {size:,} bytes[/dim]"
    else:
        body = f"[dim]{crop_path}[/dim]"
    return Panel(body, title=f"[bold]{title}[/bold]", border_style="cyan", padding=(1, 2))


def render_hash_panel(face_hash: str, embedding_dim: int) -> Panel:
    """Show the biometric hash prominently."""
    body = (
        f"[bold cyan]Biometric Hash (SHA-256)[/bold cyan]\n"
        f"[white]{face_hash}[/white]\n"
        f"[dim]from {embedding_dim}-d SFace embedding vector[/dim]"
    )
    return Panel(body, title="[bold]Stage 1 Output[/bold]", border_style="green", padding=(1, 2))


def render_comparison_panel(crop_path: str, match_data: dict) -> Panel:
    """Show side-by-side: face crop info vs discovered post."""
    left = Text(f"[Detected Face]\n[cyan]{crop_path}[/cyan]")
    right = Text(
        f"[Discovered Post]\n"
        f"[bold]{match_data.get('title', 'N/A')}[/bold]\n"
        f"[green]{match_data.get('platform', 'N/A')}[/green]\n"
        f"[dim]{match_data.get('link', '')[:50]}[/dim]"
    )
    cols = Columns([left, right], expand=True, equal=True)
    return Panel(cols, title="[bold]Stage 2: Face → Web Match[/bold]", border_style="green", padding=(1, 2))


def render_blockchain_panel(tx_hash: str, block: int, gas: int) -> Panel:
    """Show the anchoring transaction details."""
    body = (
        f"[bold]Transaction Hash:[/bold] [cyan]{tx_hash}[/cyan]\n"
        f"[bold]Block Number:[/bold]     {block}\n"
        f"[bold]Gas Used:[/bold]         {gas:,}\n"
        f"[bold]Status:[/bold]           [green]✔ Confirmed[/green]"
    )
    return Panel(body, title="[bold]Stage 3: Blockchain Anchor[/bold]", border_style="magenta", padding=(1, 2))


def render_verification_panel(valid: bool, on_chain_hash: str, local_hash: str) -> Panel:
    """Show on-chain verification result with hash comparison."""
    if valid:
        status = "[bold green]✔ VERIFIED[/bold green]"
        border = "green"
    else:
        status = "[bold red]✖ MISMATCH[/bold red]"
        border = "red"

    body = (
        f"{status}\n\n"
        f"[dim]On-Chain Fingerprint:[/dim]\n[cyan]{on_chain_hash}[/cyan]\n\n"
        f"[dim]Local Recomputed:[/dim]\n[cyan]{local_hash}[/cyan]"
    )
    return Panel(body, title="[bold]Stage 4: Verification[/bold]", border_style=border, padding=(1, 2))


def show_face_detection(input_image_path: str, bbox: tuple, auto_close_ms: int = 2000):
    """Show an OpenCV window with the detected face bounding box."""
    try:
        import cv2

        img = cv2.imread(input_image_path)
        if img is None:
            return
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Face Detection", img)
        cv2.waitKey(auto_close_ms)
        cv2.destroyAllWindows()
    except Exception as e:
        console.print(f"[dim]Could not display OpenCV window: {e}[/dim]")

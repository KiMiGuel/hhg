"""Visual layer for the HH Goa 2026 Task 3 pipeline.

Turns the four pipeline stages into a visually compelling screen recording:
- OpenCV windows show the actual face detection (bounding box + landmarks)
- The cropped face is rendered as ASCII art directly in the terminal
- A rich dashboard updates in real time as data flows through
- Blockchain mining gets an ASCII block animation
- Final side-by-side comparison of input face vs discovered post

No web frontend, no hosting -- purely local CLI visuals.
"""

import os
import time

import cv2
import numpy as np
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

# ASCII art rendering: maps pixel brightness to characters
ASCII_CHARS = " .:-=+*#%@"
ASCII_WIDTH = 60  # terminal columns for the face ASCII art

console = Console()


def show_face_detection(image_path: str, bbox: tuple, landmarks: np.ndarray = None,
                       auto_close_ms: int = 0):
    """Display OpenCV windows showing the detected face with bounding box
    and facial landmarks.

    Args:
        image_path: Path to the input image.
        bbox: (x, y, w, h) bounding box.
        landmarks: Optional 10-element array of 5 (x, y) landmark coordinates.
        auto_close_ms: If > 0, windows auto-close after this many milliseconds.
                       If 0, blocks until user closes windows (default).
    """
    image = cv2.imread(image_path)
    if image is None:
        return
    x, y, w, h = bbox

    # Draw bounding box
    annotated = image.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Draw landmarks if available
    if landmarks is not None:
        for i in range(5):
            lx, ly = int(landmarks[i]), int(landmarks[i + 5])
            cv2.circle(annotated, (lx, ly), 5, (0, 0, 255), -1)

    # Crop the face region
    pad_x = int(w * 0.15)
    pad_y = int(h * 0.15)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + w + pad_x)
    y2 = min(image.shape[0], y + h + pad_y)
    cropped = image[y1:y2, x1:x2]

    # Show windows
    cv2.imshow("Stage 1: Face Detection", annotated)
    cv2.imshow("Stage 1: Cropped Face", cropped)
    cv2.waitKey(auto_close_ms)
    cv2.destroyAllWindows()


def face_to_ascii(face_image_path: str, width: int = ASCII_WIDTH) -> str:
    """Convert a face image to ASCII art string using brightness mapping."""
    from PIL import Image

    img = Image.open(face_image_path).convert("L")  # grayscale

    # Calculate height maintaining aspect ratio (terminal chars are ~2x tall)
    aspect_ratio = img.height / img.width
    height = int(width * aspect_ratio * 0.5)
    height = max(height, 10)

    img = img.resize((width, height))
    pixels = np.array(img)

    # Map brightness to ASCII characters
    chars = []
    for row in pixels:
        for pixel in row:
            idx = int(pixel / 255 * (len(ASCII_CHARS) - 1))
            chars.append(ASCII_CHARS[idx])
        chars.append("\n")

    return "".join(chars)

def render_face_panel(face_image_path: str, title: str = "Detected Face") -> Panel:
    """Render the face as ASCII art inside a rich Panel."""
    ascii_art = face_to_ascii(face_image_path)
    return Panel(
        ascii_art,
        title=f"[bold cyan]{title}[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    )


def render_hash_panel(face_hash: str, embedding_dim: int) -> Panel:
    """Render the biometric hash info panel."""
    content = (
        f"[dim]SFace Embedding:[/dim] [bold]{embedding_dim}-d[/bold] biometric vector\n"
        f"[dim]SHA-256 Hash:[/dim] [bold cyan]{face_hash[:20]}...[/bold cyan]\n"
        f"[dim]Hash Length:[/dim] [bold]256 bits[/bold] (EVM bytes32 compatible)"
    )
    return Panel(
        content,
        title="[bold green]Biometric Fingerprint[/bold green]",
        border_style="green",
    )


def render_discovery_panel(match_data: dict) -> Panel:
    """Render the discovered social media post panel."""
    platform_colors = {
        "youtube.com": "red",
        "instagram.com": "magenta",
        "twitter.com": "blue",
        "x.com": "blue",
        "linkedin.com": "blue",
        "facebook.com": "blue",
        "reddit.com": "orange",
        "pinterest.com": "red",
    }
    platform = match_data.get("platform", "web")
    color = platform_colors.get(platform, "white")

    content = (
        f"[dim]Platform:[/dim] [bold {color}]{platform.upper()}[/bold {color}]\n"
        f"[dim]Title:[/dim] [bold]{match_data.get('title', 'N/A')}[/bold]\n"
        f"[dim]URL:[/dim] [link]{match_data.get('link', '')}[/link]"
    )
    return Panel(
        content,
        title="[bold yellow]Discovered Social Post[/bold yellow]",
        border_style="yellow",
    )


def render_blockchain_panel(tx_hash: str, block: int, gas: int) -> Panel:
    """Render the blockchain transaction panel with ASCII block."""
    block_ascii = (
        "  +------------------+\n"
        "  |  BLOCK #" + str(block) + " " * max(0, 8 - len(str(block))) + "|\n"
        "  +------------------+"
    )
    content = (
        f"[dim]{block_ascii}[/dim]\n\n"
        f"[dim]Tx Hash:[/dim] [bold cyan]{tx_hash[:20]}...[/bold cyan]\n"
        f"[dim]Block:[/dim] [bold]{block}[/bold]\n"
        f"[dim]Gas Used:[/dim] [bold]{gas:,}[/bold]"
    )
    return Panel(
        content,
        title="[bold magenta]Blockchain Anchored[/bold magenta]",
        border_style="magenta",
    )


def render_verification_panel(valid: bool, on_chain_hash: str, local_hash: str) -> Panel:
    """Render the verification result panel."""
    if valid:
        content = (
            "[bold green]  +--------------------------------+\n"
            "  |   CRYPTOGRAPHIC VERIFICATION   |\n"
            "  |          PASSED                |\n"
            "  +--------------------------------+[/bold green]\n\n"
            f"[dim]On-Chain Hash:[/dim] [bold cyan]{on_chain_hash[:20]}...[/bold cyan]\n"
            f"[dim]Local Hash:[/dim] [bold cyan]{local_hash[:20]}...[/bold cyan]\n"
            f"[dim]Match:[/dim] [bold green]IDENTICAL[/bold green]"
        )
        return Panel(
            content,
            title="[bold green]Verification PASSED[/bold green]",
            border_style="green",
        )
    else:
        content = (
            "[bold red]  +--------------------------------+\n"
            "  |     TAMPER DETECTED            |\n"
            "  +--------------------------------+[/bold red]\n\n"
            f"[dim]On-Chain Hash:[/dim] [bold cyan]{on_chain_hash[:20]}...[/bold cyan]\n"
            f"[dim]Local Hash:[/dim] [bold red]{local_hash[:20]}...[/bold red]\n"
            f"[dim]Match:[/dim] [bold red]MISMATCH[/bold red]"
        )
        return Panel(
            content,
            title="[bold red]Tamper Detected[/bold red]",
            border_style="red",
        )

def render_comparison_panel(face_image_path: str, match_data: dict) -> Columns:
    """Render side-by-side face ASCII art and discovered post."""
    face_panel = render_face_panel(face_image_path, "Input Face")
    post_panel = render_discovery_panel(match_data)
    return Columns([face_panel, post_panel], equal=True)


def render_pipeline_header() -> Panel:
    """Render the pipeline title header."""
    return Panel.fit(
        "[bold cyan]HH GOA 2026 - TASK 3[/bold cyan]\n"
        "[dim]Face Identification & Blockchain Verification[/dim]",
        border_style="cyan",
    )


def render_stage_progress(stage: int, total: int = 4) -> str:
    """Render a visual stage progress indicator."""
    stages = []
    for i in range(1, total + 1):
        if i < stage:
            stages.append(f"[bold green]>> Stage {i}[/bold green]")
        elif i == stage:
            stages.append(f"[bold yellow]-> Stage {i}[/bold yellow]")
        else:
            stages.append(f"[dim]  Stage {i}[/dim]")
    return "  ---  ".join(stages)


def animate_mining(console: Console, tx_hash: str):
    """Show a mining animation while waiting for transaction."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Mining transaction..."),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    task = progress.add_task("mining", total=100)

    with Live(console=console, refresh_per_second=10) as live:
        for i in range(100):
            progress.update(task, advance=1)
            live.update(progress)
            time.sleep(0.02)

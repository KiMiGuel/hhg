"""Arrow-key menu system with numbered fallback for piped/non-tty input.

Works in the classic Windows console (cmd) via msvcrt, with a numbered
text-input fallback when stdin is piped (e.g., during automated testing).
"""

import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def pause(message: str = "Press Enter to continue ..."):
    """Wait for the user to press Enter. Silently no-ops when stdin is piped."""
    console.print(f"\n[dim]{message}[/dim]", end="")
    if sys.stdin.isatty():
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
    else:
        console.print()


def _read_key() -> str | None:
    """Read a single keypress via msvcrt. Returns None when not a tty."""
    if not sys.stdin.isatty():
        return None
    try:
        import msvcrt

        ch = msvcrt.getch()
        # Arrow keys are escaped as b'\xe0' + b'H'/'P'/'K'/'M'
        if ch in (b"\xe0", b"\x00"):
            ch2 = msvcrt.getch()
            return {b"H": "up", b"P": "down", b"K": "left", b"M": "right"}.get(ch2)
        if ch == b"\x1b":
            return "esc"
        if ch == b"\r":
            return "enter"
        if ch == b"\x03":  # Ctrl-C
            raise KeyboardInterrupt
        return ch.decode("utf-8", errors="ignore")
    except (ImportError, OSError):
        return None


def select_option(
    title: str,
    options: list[str],
    default: int = 0,
    allow_esc: bool = True,
    hint: str | None = None,
) -> int | None:
    """Show a menu and return the selected index, or None on Esc.

    Interactive tty: arrow-key navigation with Enter to select, Esc to cancel,
    number keys as shortcuts. Non-tty: numbered text prompt.
    """
    if hint is None:
        hint = "[up]/[down] move · Enter select · number shortcut"
        if allow_esc:
            hint += " · Esc back"

    # Non-tty fallback: numbered prompt
    if not sys.stdin.isatty():
        console.print(f"\n[bold]{title}[/bold]")
        for i, opt in enumerate(options):
            marker = " ←" if i == default else ""
            console.print(f"  [cyan]{i}[/cyan] {opt}{marker}")
        console.print(f"[dim]{hint}[/dim]")
        while True:
            raw = input(f"Choice [0-{len(options) - 1}]: ").strip()
            if raw.isdigit() and 0 <= int(raw) < len(options):
                return int(raw)
            console.print("[red]Invalid — try again.[/red]")

    # Interactive: arrow-key navigation
    cursor = max(0, min(default, len(options) - 1))
    while True:
        console.print()
        body_lines = []
        for i, opt in enumerate(options):
            if i == cursor:
                body_lines.append(f"[bold reverse cyan]  {i}. {opt}  [/bold reverse cyan]")
            else:
                body_lines.append(f"  [dim cyan]{i}.[/dim cyan] {opt}")
        body = "\n".join(body_lines)
        panel = Panel(
            body,
            title=f"[bold]{title}[/bold]",
            subtitle=f"[dim]{hint}[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
        # Use a single-line redraw: print panel then move cursor up to redraw
        console.print(panel, end="")
        key = _read_key()
        if key == "up":
            cursor = (cursor - 1) % len(options)
        elif key == "down":
            cursor = (cursor + 1) % len(options)
        elif key == "enter":
            console.print()
            return cursor
        elif key == "esc" and allow_esc:
            console.print()
            return None
        elif key and key.isdigit() and 0 <= int(key) < len(options):
            console.print()
            return int(key)

"""Interactive dashboard: a full-app menu experience in the terminal.

Launch with `python main.py` (no arguments) or `python main.py ui`.
Works in the classic Windows console (cmd) — no Windows Terminal needed:
arrow keys via msvcrt, redraw via Rich, numbered fallback when piped.

GUI-like features:
  • Animated banner + live system status dashboard
  • Arrow-key menus with number shortcuts and Esc to go back
  • Confirmation dialogs for destructive actions
  • Toast-style notifications for async feedback
  • Progress spinners for long operations
  • Paginated record browser with detail view
  • Interactive image picker with preview info
"""

import os
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.align import Align
from rich.live import Live
from rich.spinner import Spinner

from src.menu import pause, select_option

console = Console()

BANNER = """\
 ███████╗ █████╗  ██████╗███████╗ ██████╗██╗  ██╗ █████╗ ██╗███╗   ██╗
 ██╔════╝ ██╔══██╗██╔════╝██╔════╝██╔════╝██║  ██║██╔══██╗██║████╗  ██║
 ██║  ███╗███████║██║     █████╗  ██║     ███████║███████║██║██╔██╗ ██║
 ██║   ██║██╔══██║██║     ██╔══╝  ██║     ██╔══██║██╔══██║██║██║╚██╗██║
 ╚██████╔╝██║  ██║╚██████╗███████╗╚██████╗██║  ██║██║  ██║██║██║ ╚████║
  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝
      Face Identification & Blockchain Verification — HH Goa 2026
"""


# ---------------------------------------------------------------- dashboard stats


def _count_reports() -> int:
    """Count audit reports in reports/ directory."""
    if not os.path.isdir("reports"):
        return 0
    return len([f for f in os.listdir("reports") if f.endswith(".json")])


def _count_exports() -> int:
    """Count exports in exports/ directory."""
    if not os.path.isdir("exports"):
        return 0
    return len(os.listdir("exports"))


def _count_cache() -> int:
    """Count cached search results."""
    if not os.path.isdir("cache"):
        return 0
    return len([f for f in os.listdir("cache") if f.endswith(".json")])


def _recent_activities() -> list[str]:
    """Get recent pipeline run summaries from reports/."""
    import glob
    import json
    if not os.path.isdir("reports"):
        return []
    reports = sorted(glob.glob("reports/*.json"), reverse=True)[:5]
    activities = []
    for r in reports:
        try:
            with open(r, "r", encoding="utf-8") as f:
                data = json.load(f)
            stage4 = data.get("stage4", {})
            status = stage4.get("verification", "unknown")
            ts = data.get("generated_at", "")
            platform = data.get("stage2", {}).get("platform", "")
            activities.append(f"[{ts[:19]}] {status} — {platform}")
        except Exception:
            continue
    return activities


def _stat_cards() -> "Columns":
    """Build stat cards for the dashboard home screen."""
    from rich.columns import Columns
    cards = []
    records_n = 0
    try:
        from src.blockchain import BlockchainManager
        from src.config import CONTRACT_ADDRESS, PRIVATE_KEY, RPC_URL
        from src.registry import fetch_all_records
        if CONTRACT_ADDRESS:
            bc = BlockchainManager(RPC_URL, PRIVATE_KEY, CONTRACT_ADDRESS)
            records_n = len(fetch_all_records(bc))
    except Exception:
        pass
    cards.append(Panel(f"[bold cyan]{records_n}[/bold cyan]\n[dim]records anchored[/dim]", border_style="cyan"))
    cards.append(Panel(f"[bold green]{_count_cache()}[/bold green]\n[dim]cached searches[/dim]", border_style="green"))
    cards.append(Panel(f"[bold magenta]{_count_reports()}[/bold magenta]\n[dim]audit reports[/dim]", border_style="magenta"))
    cards.append(Panel(f"[bold yellow]{_count_exports()}[/bold yellow]\n[dim]registry exports[/dim]", border_style="yellow"))
    return Columns(cards, equal=True, expand=True)


def _activity_panel() -> Panel:
    """Show recent pipeline run activity."""
    activities = _recent_activities()
    if not activities:
        return Panel("[dim]No recent activity — run the pipeline to see results here.[/dim]", border_style="dim")
    body = "\n".join(f"  • {a}" for a in activities)
    return Panel(body, title="[dim]recent activity[/dim]", border_style="dim")



# ---------------------------------------------------------------- GUI helpers


def _toast(message: str, style: str = "green", duration: float = 1.5):
    """Show a transient toast-style notification at the bottom of the screen."""
    console.print()
    console.print(
        Align.center(
            Panel(
                f"[{style}]{message}[/{style}]",
                border_style=style,
                padding=(0, 2),
            )
        )
    )
    time.sleep(duration)


def _confirm(message: str, default: bool = False) -> bool:
    """GUI-style confirmation dialog. Returns True if the user confirms."""
    from rich.prompt import Confirm

    console.print()
    console.print(
        Panel(
            f"[bold yellow]⚠ {message}[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )
    return Confirm.ask("Proceed?", default=default)


def _progress_spinner(description: str, duration: float = 2.0):
    """Show an animated spinner for visual feedback during transitions."""
    with Live(
        Spinner("dots", text=f"[cyan]{description}[/cyan]"),
        console=console,
        refresh_per_second=12,
        transient=True,
    ):
        time.sleep(duration)


def _animated_banner():
    """Display the banner with a subtle typewriter-style reveal."""
    console.clear()
    lines = BANNER.split("\n")
    for line in lines:
        console.print(Text(line, style="bold cyan"))
        time.sleep(0.02)
    console.print()


# ---------------------------------------------------------------- status


def _status_header() -> Panel:
    """Live system status: node, contract, records, models, API key, input."""
    rows = []

    node_ok = False
    try:
        from src.config import RPC_URL
        from web3 import Web3

        w3 = Web3(Web3.HTTPProvider(RPC_URL))
        node_ok = w3.is_connected()
        if node_ok:
            rows.append(("Node", f"[green]✔[/green] {RPC_URL} · block {w3.eth.block_number}"))
        else:
            rows.append(("Node", f"[red]✖[/red] {RPC_URL} not responding (run Setup)"))
    except Exception:
        rows.append(("Node", "[red]✖[/red] not reachable (run Setup)"))

    try:
        from src.blockchain import BlockchainManager
        from src.config import CONTRACT_ADDRESS, PRIVATE_KEY, RPC_URL
        from src.registry import fetch_all_records

        if CONTRACT_ADDRESS and node_ok:
            bc = BlockchainManager(RPC_URL, PRIVATE_KEY, CONTRACT_ADDRESS)
            n = len(fetch_all_records(bc))
            rows.append(
                ("Contract", f"[green]✔[/green] {CONTRACT_ADDRESS[:12]}… · {n} record(s)")
            )
        else:
            rows.append(("Contract", "[yellow]–[/yellow] not deployed"))
    except Exception:
        rows.append(("Contract", "[yellow]–[/yellow] unavailable"))

    try:
        from src.face_engine import SFACE_MODEL_PATH, YUNET_MODEL_PATH

        ok = all(
            os.path.exists(p) and os.path.getsize(p) > 0
            for p in (YUNET_MODEL_PATH, SFACE_MODEL_PATH)
        )
        rows.append(("Models", "[green]✔[/green] YuNet + SFace" if ok else "[red]✖[/red] missing"))
    except Exception:
        rows.append(("Models", "[yellow]?[/yellow]"))

    from src.config import SERPAPI_KEY

    rows.append(
        ("SerpApi", "[green]✔[/green] live search ready" if SERPAPI_KEY else "[yellow]–[/yellow] not set (demo mode only)")
    )

    sample = "data/sample_face.jpg"
    rows.append(
        ("Sample image", "[green]✔[/green] present" if os.path.exists(sample) else "[yellow]–[/yellow] missing")
    )

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", justify="right")
    grid.add_column()
    for k, v in rows:
        grid.add_row(k, v)
    return Panel(grid, title="[dim]system status[/dim]", border_style="dim")


# ---------------------------------------------------------------- screens


# ---------------------------------------------------------------- camera capture


def capture_screen():
    """Open the webcam for face capture with live detection."""
    from src.camera import capture_from_camera

    path = capture_from_camera()
    if path:
        _toast(f"Captured → {path}")
    pause()



def run_pipeline_wizard(demo: bool):
    from rich.prompt import Prompt

    path = Prompt.ask("  Image path", default="data/sample_face.jpg").strip().strip('"')
    if not os.path.exists(path):
        console.print(f"[red]✖ Image not found: {path}[/red]")
        pause()
        return
    from pipeline import run_pipeline

    try:
        run_pipeline(path, demo_mode=demo)
    except Exception as e:
        console.print(Panel(f"[bold red]✖ Pipeline failed:[/bold red]\n{e}", border_style="red"))
    pause()


def records_screen():
    from src.blockchain import BlockchainManager
    from src.config import CONTRACT_ADDRESS, PRIVATE_KEY, RPC_URL
    from src.registry import fetch_all_records, render_records_table

    try:
        bc = BlockchainManager(RPC_URL, PRIVATE_KEY, CONTRACT_ADDRESS)
        with console.status("[dim]Querying RecordRegistered events..."):
            records = fetch_all_records(bc)
        render_records_table(records)
    except Exception as e:
        console.print(Panel(f"[bold red]✖ {e}[/bold red]", border_style="red"))
    pause()


def verify_screen():
    choice = select_option(
        "Verify — choose method",
        ["By face hash + post URL", "By image file + post URL (re-derives hash)"],
        allow_esc=True,
        hint="[up]/[down] move · Enter select · Esc back",
    )
    if choice is None:
        return
    from rich.prompt import Prompt

    if choice == 0:
        face_hash = Prompt.ask("  Face hash (0x…)").strip()
        url = Prompt.ask("  Anchored post URL").strip()
        from verify import verify_entry

        verify_entry(face_hash, url)
    else:
        image = Prompt.ask("  Image path", default="data/sample_face.jpg").strip().strip('"')
        url = Prompt.ask("  Anchored post URL").strip()
        from main import cmd_verify_image

        class _Args:
            image = image
            url = url

        cmd_verify_image(_Args())
    pause()


def export_screen():
    fmt = select_option("Export — choose format", ["CSV", "JSON"], allow_esc=True)
    if fmt is None:
        return
    from src.blockchain import BlockchainManager
    from src.config import CONTRACT_ADDRESS, PRIVATE_KEY, RPC_URL
    from src.registry import export_records, fetch_all_records

    try:
        bc = BlockchainManager(RPC_URL, PRIVATE_KEY, CONTRACT_ADDRESS)
        records = fetch_all_records(bc)
        if not records:
            console.print("[yellow]Nothing to export — no records anchored yet.[/yellow]")
        else:
            path = export_records(records, fmt="csv" if fmt == 0 else "json")
            console.print(
                f"[green]✔[/green] Exported {len(records)} record(s) → [bold]{path}[/bold]"
            )
    except Exception as e:
        console.print(Panel(f"[bold red]✖ {e}[/bold red]", border_style="red"))
    pause()


def status_screen():
    console.print(_status_header())
    cache_n = len(os.listdir("cache")) if os.path.isdir("cache") else 0
    reports_n = (
        len([f for f in os.listdir("reports") if f.endswith(".json")])
        if os.path.isdir("reports")
        else 0
    )
    exports_n = len(os.listdir("exports")) if os.path.isdir("exports") else 0
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", justify="right")
    grid.add_column()
    grid.add_row("Cached searches", str(cache_n))
    grid.add_row("Audit reports", str(reports_n))
    grid.add_row("Registry exports", str(exports_n))
    console.print(Panel(grid, border_style="dim"))
    pause()


def help_screen():
    console.print(
        Panel(
            "[bold]CLI usage (power users)[/bold]\n\n"
            "python main.py run <image> [--demo] [--face N]\n"
            "python main.py verify <hash> <url>\n"
            "python main.py verify-image <image> <url>\n"
            "python main.py records\n"
            "python main.py export [--format csv|json]\n"
            "python main.py setup\n"
            "python main.py smoke\n\n"
            "[dim]Relaunch this dashboard: python main.py  (or: python main.py ui)[/dim]",
            border_style="cyan",
        )
    )
    pause()


def smoke_screen():
    import subprocess

    result = subprocess.run([sys.executable, "scripts/smoke_test.py"])
    if result.returncode == 0:
        console.print("[bold green]✔ Smoke test passed[/bold green]")
    else:
        console.print("[bold red]✖ Smoke test failed[/bold red]")
    pause()


def live_screen():
    """One-click live run from the dashboard: real SerpApi + blockchain."""
    from scripts.live_run import main as live_main

    sample_img = os.environ.get("HHG_SAMPLE_IMAGE", "data/sample_face.jpg")

    console.print(
        Panel(
            "[bold cyan]One-click LIVE run[/bold cyan]\n"
            "This uses the real internet + SerpApi Google Lens + blockchain path.\n"
            "It is NOT demo mode and may consume one SerpApi search credit.",
            border_style="cyan",
        )
    )
    if not os.path.exists(sample_img):
        console.print(f"[yellow]⚠ Sample image not found ({sample_img}).[/yellow]")
        console.print("[dim]  Set HHG_SAMPLE_IMAGE=<path> or use 'Capture from webcam'.[/dim]")
        sample_img = None

    mode = select_option(
        "Choose input source",
        [f"Use {sample_img}" if sample_img else "Capture from webcam (no sample image)", "Capture from webcam", "Cancel"],
        default=0,
        allow_esc=True,
    )
    if mode is None or mode == 2:
        return
    if mode == 0 and not sample_img:
        console.print("[yellow]⚠ No sample image available. Use 'Capture from webcam' or set HHG_SAMPLE_IMAGE.[/yellow]")
        pause()
        return

    argv_backup = sys.argv[:]
    try:
        sys.argv = ["live_run.py"]
        chain_mode = select_option(
            "Blockchain mode",
            ["Reuse current chain/contract", "Fresh local Anvil + redeploy (best for recording)"],
            default=1,
            allow_esc=True,
        )
        if chain_mode == 1:
            sys.argv.append("--fresh-chain")
        if mode == 0:
            sys.argv += ["--image", sample_img]
        else:
            sys.argv.append("--camera")
        code = live_main()
        if code == 0:
            console.print("[bold green]LIVE run completed successfully.[/bold green]")
        else:
            console.print(f"[bold red]LIVE run failed with exit code {code}.[/bold red]")
    finally:
        sys.argv = argv_backup
    pause()


def setup_screen():
    from src.setup_wizard import run_setup

    run_setup()
    pause()


# ---------------------------------------------------------------- main menu

MENU = [
    "Capture Photo   (webcam face capture)",
    "LIVE One-Click   (camera/image → SerpApi → blockchain)",
    "Run Pipeline    (live SerpApi Google Lens search)",
    "Run Pipeline    (demo mode — offline, no credits)",
    "Records         (browse the on-chain registry)",
    "Verify          (audit a record by hash or image)",
    "Export          (registry to CSV / JSON)",
    "System Status   (configuration report)",
    "Setup Wizard    (deps · models · node · contract)",
    "Smoke Test      (end-to-end without SerpApi)",
    "Help            (CLI usage)",
    "Exit",
]

ACTIONS = {
    0: capture_screen,
    1: live_screen,
    2: lambda: run_pipeline_wizard(demo=False),
    3: lambda: run_pipeline_wizard(demo=True),
    4: records_screen,
    5: verify_screen,
    6: export_screen,
    7: status_screen,
    8: setup_screen,
    9: smoke_screen,
    10: help_screen,
}


def run_dashboard():
    """The app loop: banner + status header + menu, forever until Exit/Esc."""
    while True:
        console.clear()
        console.print(Text(BANNER, style="bold cyan"))
        console.print(_status_header())
        console.print()
        console.print(_stat_cards())
        console.print()
        console.print(_activity_panel())
        console.print()
        choice = select_option(
            "Main Menu",
            MENU,
            default=0,
            hint="[up]/[down] move · Enter select · number shortcut · Esc Exit",
            allow_esc=True,
        )
        if choice is None or choice == len(MENU) - 1:
            console.print("[bold cyan]Goodbye![/bold cyan]")
            break
        action = ACTIONS.get(choice)
        if action:
            try:
                action()
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted — returning to menu.[/yellow]")
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                pause()

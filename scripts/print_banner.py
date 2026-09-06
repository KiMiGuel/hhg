"""Pre-flight banner + post-flight result printer for the one-click .bat."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make src/ importable when this script is run as scripts/print_banner.py
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
console = Console()
LOGO = [
    r"  _    _  _    _  _      ",
    r" | |  | || |  | || |     ",
    r" | |__| || |__| || |  _  ",
    r" |  __  ||  __  || | | | ",
    r" | |  | || |  | || |_| | ",
    r" |_|  |_||_|  |_| |____/ ",
]


def _logo():
    t = Text()
    palette = ["bold cyan", "cyan", "deep_sky_blue1", "blue1", "magenta", "bright_magenta"]
    for i, line in enumerate(LOGO):
        t.append(line + chr(10), style=palette[i])
    t.append(chr(10))
    t.append("  H H G  " + chr(0x00B7) + "  ", style="bold white")
    t.append("FACE IDENTITY", style="bold cyan")
    t.append("  +  ", style="dim")
    t.append("BLOCKCHAIN PROOF", style="bold magenta")
    t.append(
        "  " + chr(0x00B7) + "  HH Goa 2026  " + chr(0x00B7) + "  Task 3" + chr(10),
        style="dim white",
    )
    return t


def banner(mode_pick, chain_pick):
    console.clear()
    console.print(
        Panel(
            _logo(),
            border_style="bright_cyan",
            padding=(1, 4),
            title="[bold white on cyan] HHG :: FACE IDENTITY + BLOCKCHAIN PROOF [/bold white on cyan]",
            subtitle="[dim]Double-click run_live_pipeline.bat  real SerpApi + local Anvil[/dim]",
        )
    )
    t1 = Table.grid(padding=(0, 2))
    t1.add_column(style="bold yellow", justify="right", width=4)
    t1.add_column(style="white")
    t1.add_column(style="dim")
    for idx, label, hint in [
        ("1", "Use bundled sample", r"data\sample_face.jpg"),
        ("2", "Capture from webcam", "press SPACE to capture"),
        ("3", "Pick another image file", "you type the path"),
        ("4", "Cancel", ""),
    ]:
        tag = "  [bold green][default][/bold green]" if idx == "1" else ""
        t1.add_row("[" + idx + "]", label + tag, hint)
    console.print(
        Panel(
            t1,
            title="[bold cyan] " + chr(0x25C6) + " INPUT SOURCE [/bold cyan]",
            border_style="cyan",
            padding=(0, 2),
        )
    )
    t2 = Table.grid(padding=(0, 2))
    t2.add_column(style="bold yellow", justify="right", width=4)
    t2.add_column(style="white")
    for idx, label, hint in [
        ("F", "Fresh chain (auto-restart Anvil + redeploy)", "[default]"),
        ("R", "Reuse existing chain and contract", ""),
    ]:
        tag = "  [bold green]" + hint + "[/bold green]" if hint else ""
        t2.add_row("[" + idx + "]", label + tag)
    console.print(
        Panel(
            t2,
            title="[bold magenta] " + chr(0x25C6) + " BLOCKCHAIN MODE [/bold magenta]",
            border_style="magenta",
            padding=(0, 2),
        )
    )
    console.print()


def confirm(input_pick, chain_pick):
    """One-line confirmation after the user picks input + chain mode.

    Replaces the previous behaviour of re-printing the entire banner
    (which flickered the screen and printed the same 30-line panel twice).
    """
    mode_map = {
        "1": "bundled sample (data\\sample_face.jpg)",
        "2": "webcam capture (data\\captured_face.jpg)",
        "3": "user image",
        "4": "CANCEL",
    }
    chain_map = {
        "F": "FRESH chain (auto-restart Anvil + redeploy)",
        "R": "REUSE existing chain and contract",
    }
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold yellow", justify="right", width=14)
    grid.add_column(style="bold white")
    grid.add_row("Input picked:", mode_map.get(input_pick, input_pick))
    grid.add_row("Chain picked:", chain_map.get(chain_pick, chain_pick))
    console.print(Panel(grid, border_style="green", padding=(0, 2)))
    console.print()


def start_block(input_pick, chain_pick):
    console.clear()
    bar = chr(0x2501) * 78
    console.print("[bold cyan]" + bar + "[/bold cyan]")
    console.print(
        "  [bold bright_white]" + chr(0x25B6) + " STARTING LIVE PIPELINE[/bold bright_white]"
    )
    console.print("    [dim]input[/dim]  : [bold]" + str(input_pick) + "[/bold]")
    console.print("    [dim]chain[/dim]  : [bold]" + str(chain_pick) + "[/bold]")
    console.print("[bold cyan]" + bar + "[/bold cyan]")
    console.print()


def result(report_path):
    from src.visualizer import render_final_summary
    from src.web_search import LensMatch, LensResult, _person_name_from_lens

    p = Path(report_path)
    if not p.exists():
        console.print("[red]No report at " + str(report_path) + "[/red]")
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        console.print("[red]Could not parse report: " + str(e) + "[/red]")
        return
    stage2 = data.get("stage2", {}) or {}
    stage3 = data.get("stage3", {}) or {}
    stage4 = data.get("stage4", {}) or {}
    sel_raw = stage2.get("selected") or {}
    sel = LensMatch(
        rank=sel_raw.get("rank"),
        title=sel_raw.get("title", "Unknown"),
        link=sel_raw.get("link", ""),
        source=sel_raw.get("source", ""),
        platform=sel_raw.get("platform", "web"),
        reason=sel_raw.get("reason", ""),
    )
    kg_raw = stage2.get("knowledge_graph")
    kg = None
    if kg_raw:
        kg = LensMatch(
            rank=kg_raw.get("rank"),
            title=kg_raw.get("title", ""),
            link=kg_raw.get("link", ""),
            source=kg_raw.get("source", ""),
            platform=kg_raw.get("platform", "web"),
            reason=kg_raw.get("reason", ""),
        )
    visual = [LensMatch(**m) for m in (stage2.get("visual_matches") or []) if m.get("link")][:59]
    by_domain = {}
    for k, v in (stage2.get("candidates_by_domain") or {}).items():
        by_domain[k] = [LensMatch(**m) for m in v if m.get("link")]
    lens = LensResult(
        query_image_url=stage2.get("image_host_url") or stage2.get("query_image_url") or "",
        image_sha256=stage2.get("image_sha256", ""),
        face_hash=stage2.get("face_hash"),
        policy=stage2.get("policy", "social"),
        selected=sel,
        visual_matches=visual,
        knowledge_graph=kg,
        candidates_by_domain=by_domain,
        visual_match_count=stage2.get("visual_match_count", 0),
        cached=stage2.get("cached", False),
        cache_hit=stage2.get("cache_hit", False),
        cache_key=stage2.get("cache_key", ""),
        serpapi_total_time_s=stage2.get("serpapi_total_time_s"),
        lens_ms=stage2.get("lens_ms", 0.0),
        raw_response_at=stage2.get("raw_response_at", ""),
    )
    person = _person_name_from_lens(lens)
    title = sel.title or "Unknown"
    # Compute a simple confidence badge so the user knows how much to trust
    # the displayed identity. HIGH = Lens gave us a real anchor (KG or
    # consensus). LOW = the name is our best guess from unrelated matches.
    # ABSTAIN = the pipeline declined to name anyone.
    is_abstain = (sel.platform == "abstain") or title.startswith("No confident")
    has_kg = (stage2.get("knowledge_graph") or {}).get("title")
    has_consensus = "consensus=yes" in (sel.reason or "")
    if is_abstain:
        confidence = "ABSTAIN"
        conf_color = "yellow"
        conf_note = "no high-confidence match found in Lens results"
    elif has_kg or has_consensus:
        confidence = "HIGH"
        conf_color = "green"
        conf_note = "Lens Knowledge Graph" if has_kg else "multi-crop consensus hit"
    else:
        confidence = "LOW"
        conf_color = "red"
        conf_note = "best visual-match guess; Lens returned no actual matches for this face"
    console.print()
    console.rule(
        "[bold bright_cyan]"
        + chr(0x2605)
        + "  PIPELINE RESULT  "
        + chr(0x2605)
        + "[/bold bright_cyan]",
        style="bright_cyan",
    )
    console.print(
        "  [bold white]Confidence:[/bold white] [bold "
        + conf_color
        + "]"
        + confidence
        + "[/bold "
        + conf_color
        + "] [dim]("
        + conf_note
        + ")[/dim]"
    )
    if person:
        console.print(
            "  [bold white]Identity (Google Knowledge Graph):[/bold white] [bold bright_magenta]"
            + person
            + "[/bold bright_magenta]"
        )
    if is_abstain:
        console.print(
            "  [bold white]Best guess:[/bold white] [dim]"
            + (title or "")[:80]
            + "[/dim] [yellow](unverified)[/yellow]"
        )
    else:
        console.print(
            "  [bold white]Match title:[/bold white] [cyan]" + (title or "")[:80] + "[/cyan]"
        )
    if sel.link and not is_abstain:
        console.print(
            "  [bold white]Source URL:[/bold white] [link="
            + sel.link
            + "][blue underline]"
            + sel.link[:90]
            + "[/blue underline][/link]"
        )
    console.print(
        "  [bold white]Platform:[/bold white] [yellow]" + (sel.platform or "web") + "[/yellow]"
    )
    console.print()
    console.print(
        render_final_summary(
            lens,
            face_hash=data.get("stage1", {}).get("face_hash", ""),
            on_chain_fingerprint=stage3.get("fingerprint", ""),
            tx_hash=stage3.get("tx_hash"),
            block=stage3.get("block"),
            gas=stage3.get("gas_used"),
            verification_passed=stage4.get("verification") == "PASSED",
            tamper_detected=stage4.get("tamper_detected", False),
            tamper_local_hash=stage4.get("tamper_local_hash"),
            tamper_on_chain_hash=stage4.get("tamper_on_chain_hash"),
            total_seconds=data.get("total_seconds", 0.0),
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["banner", "confirm", "start", "result"], required=True)
    p.add_argument("--input-pick", default="1")
    p.add_argument("--chain-pick", default="F")
    p.add_argument("--report", default="")
    a = p.parse_args()
    if a.mode == "banner":
        banner(a.input_pick, a.chain_pick)
    if a.mode == "confirm":
        confirm(a.input_pick, a.chain_pick)
    if a.mode == "start":
        start_block(a.input_pick, a.chain_pick)
    if a.mode == "result":
        rp = a.report
        if not rp:
            rd = ROOT / "reports"
            cands = sorted(rd.glob("report_*.json"), key=lambda p: p.stat().st_mtime)
            if not cands:
                console.print("[red]No reports/report_*.json found[/red]")
                return 1
            rp = str(cands[-1])
        result(rp)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""On-chain registry helpers: fetch records, render table, export."""

import csv
import json
import os
from datetime import timezone, datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def _parse_record(log_entry) -> dict:
    """Parse a RecordRegistered event log entry into a dict."""
    args = log_entry["args"]
    return {
        "face_hash": args["faceHash"].hex() if isinstance(args["faceHash"], bytes) else args["faceHash"],
        "post_url": args.get("postUrl", ""),
        "data_hash": args["dataHash"].hex() if isinstance(args["dataHash"], bytes) else args["dataHash"],
        "timestamp": args.get("timestamp", 0),
        "block_number": int(log_entry.get("blockNumber", 0)),
    }


def fetch_all_records(bc) -> list[dict]:
    """Query the blockchain manager for all RecordRegistered events."""
    try:
        # Get events from the contract
        events = bc.contract.events.RecordRegistered.get_logs(fromBlock=0)
        records = []
        for evt in events:
            try:
                records.append(_parse_record(evt))
            except Exception:
                continue
        # Sort by block number, newest first
        records.sort(key=lambda r: r["block_number"], reverse=True)
        return records
    except Exception as e:
        console.print(f"[red]Error fetching records: {e}[/red]")
        return []


def render_records_table(records: list[dict]):
    """Display records in a formatted Rich table."""
    if not records:
        console.print(Panel("[yellow]No records anchored yet.[/yellow]", border_style="yellow"))
        return

    table = Table(title=f"On-Chain Registry ({len(records)} record(s))", border_style="cyan")
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Face Hash", style="cyan", min_width=20, max_length=18)
    table.add_column("Platform / URL", style="white", min_width=30)
    table.add_column("Block", justify="right", style="green")
    table.add_column("Timestamp", style="dim")

    for i, rec in enumerate(records):
        ts = rec.get("timestamp", 0)
        ts_str = ""
        if ts > 0:
            try:
                ts_str = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            except (ValueError, OSError):
                ts_str = str(ts)

        # Extract platform from URL
        url = rec.get("post_url", "")
        platform = ""
        for domain in ["youtube.com", "twitter.com", "x.com", "instagram.com", "linkedin.com", "facebook.com", "reddit.com", "tiktok.com"]:
            if domain in url:
                platform = domain
                break
        if not platform:
            platform = url[:30] if url else "—"

        hash_display = rec.get("face_hash", "")
        if len(hash_display) > 18:
            hash_display = hash_display[:10] + "…" + hash_display[-6:]

        table.add_row(str(i), hash_display, f"{platform}\n[dim]{url[:40]}[/dim]", str(rec.get("block_number", "")), ts_str)

    console.print(table)


def export_records(records: list[dict], fmt: str = "csv") -> str:
    """Export records to CSV or JSON. Returns the output file path."""
    os.makedirs("exports", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if fmt == "json":
        path = f"exports/registry_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str)
    else:
        path = f"exports/registry_{ts}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["face_hash", "post_url", "data_hash", "timestamp", "block_number"])
            writer.writeheader()
            for rec in records:
                writer.writerow(rec)

    return path

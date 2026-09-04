"""On-chain registry explorer: queries RecordRegistered events from FaceRegistry.

Lets users discover what is anchored on-chain without knowing face hashes,
and export the full registry for auditing.
"""

import csv
import json
import os
import time
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table

from src.config import CONTRACT_ADDRESS, PRIVATE_KEY, RPC_URL

console = Console()

EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)


def fetch_all_records(bc_manager) -> list:
    """Query every RecordRegistered event ever emitted by the contract.

    Returns a list of dicts sorted by block number:
        {face_hash, post_url, data_hash, timestamp, registrant, block, tx_hash}
    """
    events = bc_manager.contract.events.RecordRegistered.get_logs(from_block=0)
    records = []
    for ev in events:
        args = ev["args"]
        block = bc_manager.w3.eth.get_block(ev["blockNumber"])
        records.append(
            {
                "face_hash": "0x" + args["faceHash"].hex(),
                "post_url": args["postUrl"],
                "data_hash": "0x" + args["dataHash"].hex(),
                "timestamp": int(args["timestamp"]),
                "registrant": args["registeredBy"],
                "block": ev["blockNumber"],
                "tx_hash": "0x" + ev["transactionHash"].hex(),
                "block_time": datetime.fromtimestamp(
                    int(block["timestamp"]), tz=None
                ).isoformat(sep=" ", timespec="seconds"),
            }
        )
    return records


def render_records_table(records: list):
    """Render all anchored records as a rich table."""
    if not records:
        console.print(
            "[yellow]No records anchored on this contract yet. "
            "Run 'python main.py run <image>' first.[/yellow]"
        )
        return
    table = Table(title=f"FaceRegistry — {len(records)} anchored record(s)", border_style="cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Face Hash", style="cyan", overflow="fold")
    table.add_column("Post URL", style="white", overflow="fold")
    table.add_column("Platform Fingerprint", style="magenta", overflow="fold")
    table.add_column("Block", style="green", justify="right")
    table.add_column("Tx Hash", style="dim", overflow="fold")
    for i, rec in enumerate(records, 1):
        table.add_row(
            str(i),
            rec["face_hash"],
            rec["post_url"],
            rec["data_hash"],
            str(rec["block"]),
            rec["tx_hash"],
        )
    console.print(table)


def export_records(records: list, fmt: str = "csv") -> str:
    """Export the registry to CSV or JSON. Returns the output file path."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if fmt == "json":
        path = os.path.join(EXPORT_DIR, f"registry_{stamp}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    else:
        path = os.path.join(EXPORT_DIR, f"registry_{stamp}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "face_hash",
                    "post_url",
                    "data_hash",
                    "timestamp",
                    "block_time",
                    "block",
                    "tx_hash",
                    "registrant",
                ],
            )
            writer.writeheader()
            writer.writerows(records)
    return path

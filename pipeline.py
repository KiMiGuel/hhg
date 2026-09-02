import sys
import time
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src import face_engine as face_engine_mod
from src.blockchain import BlockchainManager
from src.config import CONTRACT_ADDRESS, PRIVATE_KEY, RPC_URL, SERPAPI_KEY
from src.face_engine import FaceEngine
from src.report import write_report
from src.web_search import WebSearchEngine

console = Console()


def run_pipeline(input_image_path: str):
    console.print(
        Panel.fit(
            "[bold cyan]HH GOA 2026 - TASK 3: FACE IDENTIFICATION & BLOCKCHAIN VERIFICATION[/bold cyan]",
            border_style="cyan",
        )
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "network": {"rpc_url": RPC_URL, "contract_address": CONTRACT_ADDRESS},
        "stage1": {"image": input_image_path},
        "stage2": {},
        "stage3": {},
        "stage4": {},
    }

    # ---------------------------------------------------------------- Stage 1
    console.print("\n[bold yellow]>> [STAGE 1] Face Detection & Biometric Encoding[/bold yellow]")
    face_engine = FaceEngine()
    with console.status("[bold green]Detecting facial bounds and calculating embedding vector..."):
        crop_path, face_hash, bbox = face_engine.process_image(input_image_path)
    console.print(f"[green]+[/green] Face detected at bbox (x, y, w, h): {bbox}")
    console.print(f"[green]+[/green] Face cropped and saved to: [bold]{crop_path}[/bold]")
    console.print(
        f"[green]+[/green] SFace Embedding dim: [bold]{face_engine_mod.EMBEDDING_DIM}[/bold] "
        "(128-d biometric vector)"
    )
    console.print(f"[green]+[/green] Biometric Hash (SHA-256): [bold cyan]{face_hash}[/bold cyan]")
    report["stage1"].update(
        {"bbox": list(bbox), "embedding_dim": face_engine_mod.EMBEDDING_DIM, "face_hash": face_hash}
    )

    # ---------------------------------------------------------------- Stage 2
    console.print("\n[bold yellow]>> [STAGE 2] Dynamic Web & Social Media Discovery[/bold yellow]")
    search_engine = WebSearchEngine(SERPAPI_KEY)

    with console.status("[bold green]Uploading face crop to ephemeral image host..."):
        public_image_url = search_engine.upload_image_to_temp_host(crop_path)
    console.print(f"[green]+[/green] Ephemeral Image URL: [dim]{public_image_url}[/dim]")

    with console.status("[bold green]Executing genuine Google Lens reverse search via SerpApi..."):
        match_data = search_engine.search_face_on_web(public_image_url)

    table = Table(title="Discovered Social / Web Content", border_style="green")
    table.add_column("Property", style="cyan")
    table.add_column("Discovered Value", style="white", overflow="fold")
    table.add_row("Entity Title", str(match_data["title"]))
    table.add_row("Platform / Source", str(match_data["platform"]))
    table.add_row("Target Post URL", str(match_data["link"]))
    console.print(table)
    report["stage2"].update(
        {
            "image_host_url": public_image_url,
            "title": str(match_data["title"]),
            "platform": str(match_data["platform"]),
            "post_url": str(match_data["link"]),
        }
    )

    # ---------------------------------------------------------------- Stage 3
    console.print(
        "\n[bold yellow]>> [STAGE 3] Cryptographic Fingerprinting & Blockchain Anchoring[/bold yellow]"
    )
    bc_manager = BlockchainManager(RPC_URL, PRIVATE_KEY, CONTRACT_ADDRESS)
    data_fingerprint = bc_manager.compute_data_fingerprint(face_hash, str(match_data["link"]))
    console.print(
        f"[green]+[/green] Combined Canonical Fingerprint: [bold magenta]{data_fingerprint}[/bold magenta]"
    )

    if bc_manager.record_exists(face_hash):
        console.print(
            "[yellow]![/yellow] Record already anchored for this face hash — "
            "skipping registration and proceeding to verification."
        )
        report["stage3"] = {
            "fingerprint": data_fingerprint,
            "already_anchored": True,
        }
    else:
        with console.status("[bold green]Broadcasting transaction to blockchain..."):
            receipt = bc_manager.anchor_record(face_hash, str(match_data["link"]), data_fingerprint)
        tx_hash = f"0x{receipt.transactionHash.hex()}"
        console.print(
            f"[green]+[/green] Transaction Confirmed! TxHash: [bold cyan]{tx_hash}[/bold cyan]"
        )
        console.print(
            f"[green]+[/green] Included in Block Number: [bold]{receipt.blockNumber}[/bold] | "
            f"Gas Used: {receipt.gasUsed}"
        )
        time.sleep(0.2)
        report["stage3"] = {
            "fingerprint": data_fingerprint,
            "already_anchored": False,
            "tx_hash": tx_hash,
            "block": int(receipt.blockNumber),
            "gas_used": int(receipt.gasUsed),
        }

    # ---------------------------------------------------------------- Stage 4
    console.print(
        "\n[bold yellow]>> [STAGE 4] On-Chain Audit & Tamper-Evidence Proof[/bold yellow]"
    )
    with console.status("[bold green]Querying smart contract for re-verification..."):
        audit_result = bc_manager.verify_on_chain(face_hash, str(match_data["link"]))

    if audit_result["valid"]:
        console.print(
            Panel(
                "[bold green]STATUS: CRYPTOGRAPHIC VERIFICATION PASSED\n"
                "On-chain fingerprint matches the local record perfectly![/bold green]",
                border_style="green",
            )
        )
    else:
        console.print(Panel("[bold red]STATUS: VERIFICATION FAILED[/bold red]", border_style="red"))
    report["stage4"].update(
        {
            "verification": "PASSED" if audit_result["valid"] else "FAILED",
            "on_chain_fingerprint": audit_result["on_chain_data_hash"],
        }
    )

    # ------------------------------------------------- Tamper-evidence drill
    console.print(
        "\n[bold yellow]>> [SECURITY DRILL] Demonstrating Tamper-Evidence "
        "(Altering 1 URL Character)[/bold yellow]"
    )
    tampered_url = str(match_data["link"]) + "_tampered"
    tamper_audit = bc_manager.verify_on_chain(face_hash, tampered_url)

    if not tamper_audit["valid"]:
        console.print("[bold red]+ TAMPER DETECTED BY DESIGN![/bold red]")
        console.print(f"Expected Fingerprint : {tamper_audit['local_recomputed_hash']}")
        console.print(f"On-Chain Fingerprint : {tamper_audit['on_chain_data_hash']}")
        console.print(
            "[dim]Any tampering of the discovered social data is immediately exposed.[/dim]"
        )
    else:
        console.print("[bold red]!! Tamper drill unexpectedly passed -- audit the system![/bold red]")
    report["stage4"].update(
        {
            "tamper_detected": not tamper_audit["valid"],
            "tampered_url": tampered_url,
            "tamper_local_hash": tamper_audit["local_recomputed_hash"],
            "tamper_on_chain_hash": tamper_audit["on_chain_data_hash"],
        }
    )

    # ------------------------------------------------- Audit report artifact
    report["network"]["chain_id"] = bc_manager.w3.eth.chain_id
    report_path = write_report(report)
    console.print(f"\n[bold yellow]>> Audit report saved to:[/bold yellow] [bold]{report_path}[/bold]")

    console.print(
        Panel.fit(
            "[bold green]PIPELINE COMPLETED SUCCESSFULLY END-TO-END[/bold green]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print(
            "[bold red]Usage: python pipeline.py <path_to_input_image>[/bold red]"
        )
        sys.exit(1)
    run_pipeline(sys.argv[1])

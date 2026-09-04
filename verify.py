import sys

from rich.console import Console
from rich.panel import Panel

from src.blockchain import BlockchainManager
from src.config import CONTRACT_ADDRESS, PRIVATE_KEY, RPC_URL

console = Console()


def verify_entry(face_hash: str, expected_url: str):
    console.print(
        Panel.fit("[bold cyan]STANDALONE ON-CHAIN AUDIT TOOL[/bold cyan]", border_style="cyan")
    )
    from web3.exceptions import ContractLogicError

    bc = BlockchainManager(RPC_URL, PRIVATE_KEY, CONTRACT_ADDRESS)
    try:
        result = bc.verify_on_chain(face_hash, expected_url)
    except ContractLogicError:
        console.print(
            Panel(
                "[bold yellow]✖ NO RECORD FOUND for this face hash on the configured contract.\n"
                f"Contract : {CONTRACT_ADDRESS}\n"
                "Run the pipeline first, or check CONTRACT_ADDRESS in .env.\n"
                "Tip: 'python main.py records' lists every anchored record.[/bold yellow]",
                border_style="yellow",
            )
        )
        sys.exit(2)

    if result["valid"]:
        console.print(
            Panel(
                f"[bold green]✔ DATA RECORD AUTHENTIC & VERIFIED\n"
                f"Face Hash      : {result['on_chain_face_hash']}\n"
                f"Post URL       : {result['on_chain_url']}\n"
                f"Fingerprint    : {result['on_chain_data_hash']}\n"
                f"Block Timestamp: {result['block_timestamp']}\n"
                f"Registrant     : {result['registrant']}[/bold green]",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold red]✖ TAMPER DETECTED / AUDIT FAILED\n"
                f"On-Chain Hash : {result['on_chain_data_hash']}\n"
                f"Computed Hash : {result['local_recomputed_hash']}[/bold red]",
                border_style="red",
            )
        )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        console.print(
            "[bold red]Usage: python verify.py <face_hash> <social_post_url>[/bold red]"
        )
        sys.exit(1)
    verify_entry(sys.argv[1], sys.argv[2])

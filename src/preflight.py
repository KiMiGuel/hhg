"""Pre-flight checks: validates all prerequisites before running the pipeline.

Fails fast with clear, actionable error messages instead of crashing mid-pipeline.
"""

import os
import sys

import requests
from rich.console import Console
from rich.panel import Panel

from src.config import CONTRACT_ADDRESS, PRIVATE_KEY, RPC_URL, SERPAPI_KEY

console = Console()


class PreflightError(Exception):
    """Raised when a pre-flight check fails."""


def check_input_image(image_path: str):
    """Validate the input image exists and is readable."""
    if not image_path:
        raise PreflightError("No input image specified.")
    if not os.path.exists(image_path):
        raise PreflightError(
            f"Input image not found: {image_path}\n"
            "  Fix: place your face photo at the specified path."
        )
    if os.path.getsize(image_path) == 0:
        raise PreflightError(f"Input image is empty: {image_path}")


def check_rpc_connection(rpc_url: str = RPC_URL):
    """Validate the blockchain RPC is reachable and mining."""
    try:
        from web3 import Web3

        w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not w3.is_connected():
            raise PreflightError(
                f"Cannot connect to blockchain RPC at {rpc_url}\n"
                "  Fix: start Anvil in a separate terminal:\n"
                "       .\\scripts\\run_local_node.ps1"
            )
        try:
            block = w3.eth.block_number
        except Exception:
            raise PreflightError(
                f"RPC at {rpc_url} is connected but not responding to eth_blockNumber.\n"
                "  Fix: restart Anvil."
            )
        return w3
    except ImportError:
        raise PreflightError(
            "web3.py is not installed.\n"
            "  Fix: pip install web3"
        )


def check_contract_deployed(w3, contract_address: str = CONTRACT_ADDRESS):
    """Validate the FaceRegistry contract is deployed at the given address."""
    if not contract_address:
        raise PreflightError(
            "CONTRACT_ADDRESS is not set in .env\n"
            "  Fix: run `python scripts/deploy.py` and copy the address into .env"
        )
    try:
        checksum = w3.to_checksum_address(contract_address)
    except Exception:
        raise PreflightError(
            f"Invalid CONTRACT_ADDRESS: {contract_address}\n"
            "  Fix: redeploy with `python scripts/deploy.py`"
        )
    code = w3.eth.get_code(checksum)
    if len(code) == 0:
        raise PreflightError(
            f"No contract deployed at {checksum}\n"
            "  Fix: run `python scripts/deploy.py`"
        )


def check_serpapi_key(serpapi_key: str = SERPAPI_KEY):
    """Validate the SerpApi key is set (does not validate with the API to avoid
    consuming a search credit)."""
    if not serpapi_key:
        raise PreflightError(
            "SERPAPI_KEY is not set in .env\n"
            "  Fix: get a free key (no credit card) at https://serpapi.com\n"
            "       then add it to .env as SERPAPI_KEY=your_key_here"
        )


def check_models():
    """Validate the YuNet and SFace ONNX models are present."""
    from src.face_engine import SFACE_MODEL_PATH, YUNET_MODEL_PATH

    missing = []
    for path in [YUNET_MODEL_PATH, SFACE_MODEL_PATH]:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            missing.append(path)
    if missing:
        raise PreflightError(
            f"Missing model files: {', '.join(missing)}\n"
            "  Fix: models will auto-download on first run if internet is available.\n"
            "       If offline, download manually from:\n"
            "       https://github.com/opencv/opencv_zoo/tree/main/models"
        )


def run_preflight(image_path: str, require_serpapi: bool = True):
    """Run all pre-flight checks. Returns the connected Web3 instance if
    the blockchain checks pass. Raises PreflightError on any failure."""
    console.print(Panel.fit("[bold]Running pre-flight checks...[/bold]", border_style="dim"))

    checks = [
        ("Input image", lambda: check_input_image(image_path)),
        ("Model files", check_models),
        ("Blockchain RPC", check_rpc_connection),
        ("Contract deployment", lambda: check_contract_deployed(check_rpc_connection())),
    ]
    if require_serpapi:
        checks.append(("SerpApi key", check_serpapi_key))

    for name, check_fn in checks:
        try:
            with console.status(f"[dim]Checking {name}..."):
                result = check_fn()
            console.print(f"  [green]✔[/green] {name}")
        except PreflightError:
            raise
        except Exception as e:
            raise PreflightError(f"Unexpected error during {name} check: {e}")

    console.print()
    return check_rpc_connection()

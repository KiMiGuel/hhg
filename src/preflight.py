"""Pre-flight checks before running the pipeline."""

import os
import sys
import requests

from rich.console import Console

console = Console()


def check_input_image(path: str):
    """Verify the input image exists and is readable."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input image not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"Input image is empty: {path}")
    console.print(f"[green]✔[/green] Input image: {path} ({os.path.getsize(path):,} bytes)")


def check_models():
    """Verify YuNet + SFace ONNX models are present."""
    from src.face_engine import SFACE_MODEL_PATH, YUNET_MODEL_PATH

    for name, path in [("YuNet", YUNET_MODEL_PATH), ("SFace", SFACE_MODEL_PATH)]:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise FileNotFoundError(f"{name} model missing: {path} — run setup wizard to download")
        console.print(f"[green]✔[/green] {name} model: {path}")


def check_rpc_connection():
    """Verify the blockchain node is reachable."""
    from src.config import RPC_URL
    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        raise ConnectionError(f"Node not reachable at {RPC_URL} — start with: .\\scripts\\run_local_node.ps1")
    console.print(f"[green]✔[/green] Node reachable: {RPC_URL} (block {w3.eth.block_number})")
    return w3


def check_contract_deployed(w3):
    """Verify the contract address is set and has code on-chain."""
    from src.config import CONTRACT_ADDRESS

    if not CONTRACT_ADDRESS:
        raise RuntimeError("CONTRACT_ADDRESS not set — run: python scripts/deploy.py")
    code = w3.eth.get_code(CONTRACT_ADDRESS)
    if len(code) == 0:
        raise RuntimeError(f"No contract code at {CONTRACT_ADDRESS} — run: python scripts/deploy.py")
    console.print(f"[green]✔[/green] Contract deployed: {CONTRACT_ADDRESS[:12]}…")


def check_serpapi():
    """Verify SerpApi key is set."""
    from src.config import SERPAPI_KEY

    if not SERPAPI_KEY:
        raise RuntimeError("SERPAPI_KEY not set — get a free key at https://serpapi.com")
    console.print("[green]✔[/green] SerpApi key set")


def check_internet():
    """Lightweight network check that does not spend SerpApi credits."""
    try:
        resp = requests.get("https://serpapi.com/", timeout=8)
        if resp.status_code >= 500:
            raise RuntimeError(f"serpapi.com HTTP {resp.status_code}")
        console.print("[green]✔[/green] Internet reachable")
    except Exception as exc:
        raise RuntimeError(f"Internet/SerpApi host not reachable: {exc}") from exc


def check_image_hosts():
    """Verify at least one configured image host endpoint is reachable."""
    hosts = ["https://catbox.moe/", "https://tmpfiles.org/"]
    for url in hosts:
        try:
            resp = requests.get(url, timeout=8)
            if resp.status_code < 500:
                console.print(f"[green]✔[/green] Image host reachable: {url}")
                return
        except Exception:
            continue
    raise RuntimeError("No image host reachable (catbox.moe/tmpfiles.org). Check internet/firewall.")


def run_preflight(input_image_path: str, require_serpapi: bool = True):
    """Run all pre-flight checks."""
    console.print("[bold cyan]Pre-flight checks[/bold cyan]")
    check_input_image(input_image_path)
    check_models()
    w3 = check_rpc_connection()
    check_contract_deployed(w3)
    if require_serpapi:
        check_serpapi()
        check_internet()
        check_image_hosts()
    console.print("[green]✔ All checks passed[/green]\n")

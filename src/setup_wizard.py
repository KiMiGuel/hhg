"""Interactive setup wizard: one command to go from clean checkout to ready.

Checks dependencies, downloads models, verifies/starts Anvil, deploys the
contract, and writes .env — replacing six manual setup steps.
"""

import os
import shutil
import subprocess
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

console = Console()

ANVIL_KEY_DEFAULT = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
RPC_DEFAULT = "http://127.0.0.1:8545"

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)


def _ok(msg: str):
    console.print(f"  [green]✔[/green] {msg}")


def _skip(msg: str):
    console.print(f"  [dim]– {msg} (skipped)[/dim]")


def _fail(msg: str, fix: str) -> bool:
    console.print(f"  [red]✖ {msg}[/red]")
    console.print(f"      [dim]Fix: {fix}[/dim]")
    return False


def check_python() -> bool:
    v = sys.version_info
    if v >= (3, 10):
        _ok(f"Python {v.major}.{v.minor}.{v.micro}")
        return True
    return _fail(
        f"Python {v.major}.{v.minor} found; 3.10+ required",
        "install Python 3.10+ from python.org",
    )


def check_deps() -> bool:
    import importlib.util as u

    mods = {
        "cv2": "opencv-python",
        "web3": "web3",
        "solcx": "py-solc-x",
        "dotenv": "python-dotenv",
        "rich": "rich",
        "serpapi": "google-search-results",
        "requests": "requests",
        "numpy": "numpy",
        "PIL": "pillow",
        "pytest": "pytest",
    }
    missing = [pkg for mod, pkg in mods.items() if u.find_spec(mod) is None]
    if not missing:
        _ok("All Python dependencies installed")
        return True
    return _fail(
        f"Missing packages: {', '.join(missing)}",
        "run: pip install -r requirements.txt",
    )


def _download_model(model_path: str, url: str):
    if os.path.exists(model_path) and os.path.getsize(model_path) == 0:
        os.remove(model_path)
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        return
    import requests

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        f.write(resp.content)


def check_models() -> bool:
    from src.face_engine import SFACE_MODEL_PATH, YUNET_MODEL_PATH

    try:
        with console.status("[dim]Ensuring model files (downloads if missing)..."):
            _download_model(YUNET_MODEL_PATH, YUNET_URL)
            _download_model(SFACE_MODEL_PATH, SFACE_URL)
        _ok("YuNet + SFace models present")
        return True
    except Exception as e:
        return _fail(f"Model files unavailable: {e}", "check internet connection and re-run")


def check_anvil() -> bool:
    """Check whether the Anvil node is reachable. Offers to start it if not."""
    from web3 import Web3

    from src.config import RPC_URL

    rpc = RPC_URL if os.path.exists(".env") else RPC_DEFAULT
    w3 = Web3(Web3.HTTPProvider(rpc))
    if w3.is_connected():
        _ok(f"Blockchain node reachable at {rpc} (block {w3.eth.block_number})")
        return True
    console.print(f"  [yellow]! No node at {rpc}[/yellow]")
    if shutil.which("anvil") and Confirm.ask(
        "      Start Anvil now in a new window?", default=True
    ):
        subprocess.Popen(
            ["cmd", "/c", "start", "anvil", "--host", "127.0.0.1", "--port", "8545"],
            shell=True,
        )
        with console.status("[dim]Waiting for Anvil to boot..."):
            for _ in range(30):
                time.sleep(1)
                w3 = Web3(Web3.HTTPProvider(rpc))
                if w3.is_connected():
                    break
        if w3.is_connected():
            _ok(f"Anvil started and reachable at {rpc}")
            return True
    return _fail(
        f"No blockchain node at {rpc}",
        "install Foundry (getfoundry.sh) and run: anvil  (or re-run setup)",
    )


def deploy_contract() -> str | None:
    """Deploy FaceRegistry if CONTRACT_ADDRESS is not already configured."""
    from src.config import CONTRACT_ADDRESS, RPC_URL

    if CONTRACT_ADDRESS:
        try:
            from web3 import Web3

            w3 = Web3(Web3.HTTPProvider(RPC_URL))
            code = w3.eth.get_code(w3.to_checksum_address(CONTRACT_ADDRESS))
            if len(code) > 0:
                _ok(f"Contract already deployed at {CONTRACT_ADDRESS}")
                return CONTRACT_ADDRESS
        except Exception:
            pass

    if not Confirm.ask("      Deploy FaceRegistry contract now?", default=True):
        _skip("contract deployment")
        return None
    result = subprocess.run(
        [os.path.join("venv", "Scripts", "python.exe"), "scripts/deploy.py"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    address = None
    for line in output.splitlines():
        if "Contract Address:" in line:
            address = line.split(":", 1)[1].strip()
            break
    if address:
        _ok(f"Contract deployed at {address}")
        return address
    console.print(f"      [dim]{output[-500:]}[/dim]")
    _fail("Deployment failed", "run 'python scripts/deploy.py' manually and check output")
    return None


def write_env(contract_address: str | None):
    serpapi = Prompt.ask(
        "      SerpApi key (free at serpapi.com — Enter to skip; demo mode works without it)",
        default="",
        show_default=False,
    ).strip()
    lines = [
        f"SERPAPI_KEY={serpapi}",
        f"RPC_URL={RPC_DEFAULT}",
        f"PRIVATE_KEY={ANVIL_KEY_DEFAULT}",
        f"CONTRACT_ADDRESS={contract_address or ''}",
    ]
    with open(".env", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    _ok(".env written" + (" (add SERPAPI_KEY later for live search)" if not serpapi else ""))


def run_setup():
    console.print(Panel.fit("[bold cyan]Interactive Setup Wizard[/bold cyan]", border_style="cyan"))
    console.print()
    ok = True
    console.print("[bold]1/5  Python version[/bold]")
    ok &= check_python()
    console.print("[bold]2/5  Dependencies[/bold]")
    ok &= check_deps()
    console.print("[bold]3/5  Face models[/bold]")
    ok &= check_models()
    console.print("[bold]4/5  Blockchain node[/bold]")
    ok &= check_anvil()
    console.print("[bold]5/5  Contract & configuration[/bold]")
    address = None
    if ok:
        address = deploy_contract()
        if address:
            write_env(address)
    console.print()
    if ok:
        console.print(
            Panel.fit(
                "[bold green]✔ Setup complete![/bold green]\n\n"
                "Next:  python main.py run data/sample_face.jpg --demo\n"
                "Or:    python main.py run data/sample_face.jpg   (live SerpApi search)",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel.fit(
                "[bold yellow]Setup finished with issues — resolve the ✖ items above "
                "and re-run `python main.py setup`.[/bold yellow]",
                border_style="yellow",
            )
        )

"""One-shot setup: install missing deps, then verify the runtime is healthy.

This script is meant to be run once after a fresh checkout. It installs
the packages listed in requirements.txt (idempotent), prints a summary,
and exits non-zero on failure so the caller can tell whether the runtime
is ready.
"""

import importlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "requirements.txt"

REQUIRED = [
    "cv2",
    "numpy",
    "requests",
    "rich",
    "web3",
    "dotenv",
    "pytest",
]


def is_installed(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def main() -> int:
    missing = [m for m in REQUIRED if not is_installed(m)]
    if not missing:
        print("[ok] all required packages already installed")
        return 0

    print(f"[setup] missing packages: {missing}")
    print(f"[setup] installing from {REQ}")
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(REQ)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(f"[setup] pip exit {r.returncode} in {time.time()-t0:.1f}s")
    if r.returncode != 0:
        print("--- pip stdout ---")
        print(r.stdout[-2000:])
        print("--- pip stderr ---")
        print(r.stderr[-2000:])
        return 1

    still_missing = [m for m in REQUIRED if not is_installed(m)]
    if still_missing:
        print(f"[setup] STILL MISSING after install: {still_missing}")
        return 2
    print("[ok] all packages now installed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

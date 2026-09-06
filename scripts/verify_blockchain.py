# -*- coding: utf-8 -*-
"""Phase 2: Verify blockchain anchoring + verification with a real Anvil node.

Starts a local Anvil, deploys FaceRegistry.sol, anchors a record, verifies
it on-chain, and runs the tamper-evidence drill.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["RPC_URL"] = "http://127.0.0.1:8545"
os.environ["PRIVATE_KEY"] = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

ANVIL_EXE = str(Path.home() / ".foundry" / "bin" / "anvil.exe")


def wait_rpc(timeout: int = 30) -> bool:
    from web3 import Web3
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545", request_kwargs={"timeout": 2}))
            if w3.is_connected() and w3.eth.chain_id == 31337:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def deploy_contract() -> str:
    from solcx import compile_standard, install_solc
    from web3 import Web3
    install_solc("0.8.20")
    w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
    account = w3.eth.account.from_key(os.environ["PRIVATE_KEY"])
    src = (ROOT / "contracts" / "FaceRegistry.sol").read_text(encoding="utf-8")
    compiled = compile_standard(
        {"language": "Solidity",
         "sources": {"FaceRegistry.sol": {"content": src}},
         "settings": {"outputSelection": {"*": {"*": ["abi", "evm.bytecode"]}}}},
        solc_version="0.8.20")
    data = compiled["contracts"]["FaceRegistry.sol"]["FaceRegistry"]
    abi, bytecode = data["abi"], data["evm"]["bytecode"]["object"]
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    nonce = w3.eth.get_transaction_count(account.address)
    tx = Contract.constructor().build_transaction(
        {"from": account.address, "nonce": nonce, "gas": 3_000_000, "gasPrice": w3.eth.gas_price})
    signed = w3.eth.account.sign_transaction(tx, private_key=account.key)
    raw = getattr(signed, "rawTransaction", None) or signed.raw_transaction
    tx_hash = w3.eth.send_raw_transaction(raw)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
    if receipt.status != 1:
        raise RuntimeError(f"Deploy failed: {receipt}")
    return receipt.contractAddress


def main() -> int:
    print("=" * 70)
    print("Phase 2: Blockchain Verification with Real Anvil Node")
    print("=" * 70)
    print("\n[1] Starting Anvil...")
    anvil_proc = subprocess.Popen(
        [ANVIL_EXE, "--chain-id", "31337", "--port", "8545"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_rpc(timeout=30):
            print("  FAIL: Anvil RPC not reachable after 30s")
            return 1
        print("  OK: Anvil RPC reachable (chain_id=31337)")
        print("\n[2] Deploying FaceRegistry.sol...")
        contract_address = deploy_contract()
        print(f"  OK: Deployed at {contract_address}")
        from src.blockchain import BlockchainManager
        bm = BlockchainManager(rpc_url="http://127.0.0.1:8545",
            private_key=os.environ["PRIVATE_KEY"], contract_address=contract_address)
        face_hash = "0x" + "a" * 64
        post_url = "https://en.wikipedia.org/wiki/Satya_Nadella"
        data_hash = bm.compute_data_fingerprint(face_hash, post_url)
        print(f"\n[3] Anchoring record...")
        receipt = bm.anchor_record(face_hash, post_url, data_hash)
        print(f"    OK: tx={receipt.transactionHash.hex()} block={receipt.blockNumber} gas={receipt.gasUsed} status={receipt.status}")
        print(f"\n[4] Verifying on-chain...")
        exists = bm.record_exists(face_hash)
        print(f"    record_exists: {exists}")
        if not exists:
            print("    FAIL: record not found after anchoring")
            return 1
        result = bm.verify_on_chain(face_hash, post_url)
        print(f"    valid: {result['valid']}")
        if not result["valid"]:
            print("    FAIL: verification returned invalid")
            return 1
        print("    OK: verification PASSED")
        print(f"\n[5] Tamper-evidence drill...")
        tampered_url = post_url.replace("Satya", "Setya")
        tamper_result = bm.verify_on_chain(face_hash, tampered_url)
        print(f"    valid: {tamper_result['valid']} (expected False)")
        if tamper_result["valid"]:
            print("    FAIL: tampered data incorrectly validated")
            return 1
        print("    OK: tamper detected correctly")
        print(f"\n[6] Negative control (random hash)...")
        random_exists = bm.record_exists("0x" + "f" * 64)
        print(f"    record_exists(random): {random_exists} (expected False)")
        if random_exists:
            print("    FAIL: random hash incorrectly reported as existing")
            return 1
        print("    OK: random hash correctly reported as missing")
        print(f"\n{'=' * 70}")
        print("Phase 2: ALL CHECKS PASSED")
        print(f"{'=' * 70}")
        return 0
    finally:
        anvil_proc.terminate()
        try:
            anvil_proc.wait(timeout=5)
        except Exception:
            anvil_proc.kill()
        print("\n[cleanup] Anvil stopped.")


if __name__ == "__main__":
    raise SystemExit(main())
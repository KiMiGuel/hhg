# -*- coding: utf-8 -*-
"""Blockchain integration tests against a real local Anvil node.

Starts Anvil on port 8546 (avoids clashing with a dev node on 8545), deploys
FaceRegistry.sol, then exercises anchoring, on-chain verification, tamper
evidence, and negative controls through BlockchainManager.

Skipped automatically when Anvil (foundry) is not installed.
"""
import os
import shutil
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ANVIL = shutil.which("anvil") or os.path.expanduser(
    os.path.join("~", ".foundry", "bin", "anvil.exe"))
RPC = "http://127.0.0.1:8546"
TEST_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

pytestmark = pytest.mark.skipif(
    not (ANVIL and os.path.exists(ANVIL)),
    reason="Anvil not installed (expected at ~/.foundry/bin/anvil.exe)",
)


def _rpc_up(timeout: float = 30.0) -> bool:
    from web3 import Web3
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            w3 = Web3(Web3.HTTPProvider(RPC, request_kwargs={"timeout": 2}))
            if w3.is_connected() and w3.eth.chain_id == 31337:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


@pytest.fixture(scope="module")
def anvil_node():
    proc = subprocess.Popen(
        [ANVIL, "--chain-id", "31337", "--port", "8546"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert _rpc_up(), "Anvil did not come up on port 8546"
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


@pytest.fixture(scope="module")
def contract_address(anvil_node):
    """Compile + deploy FaceRegistry.sol, return the deployed address."""
    from solcx import compile_standard, install_solc
    from web3 import Web3
    install_solc("0.8.20")
    w3 = Web3(Web3.HTTPProvider(RPC))
    account = w3.eth.account.from_key(TEST_KEY)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src = open(os.path.join(root, "contracts", "FaceRegistry.sol"),
               encoding="utf-8").read()
    compiled = compile_standard(
        {"language": "Solidity",
         "sources": {"FaceRegistry.sol": {"content": src}},
         "settings": {"outputSelection": {"*": {"*": ["abi", "evm.bytecode"]}}}},
        solc_version="0.8.20")
    data = compiled["contracts"]["FaceRegistry.sol"]["FaceRegistry"]
    contract = w3.eth.contract(abi=data["abi"], bytecode=data["evm"]["bytecode"]["object"])
    tx = contract.constructor().build_transaction({
        "from": account.address, "nonce": w3.eth.get_transaction_count(account.address),
        "gas": 3_000_000, "gasPrice": w3.eth.gas_price})
    signed = w3.eth.account.sign_transaction(tx, private_key=account.key)
    raw = getattr(signed, "rawTransaction", None) or signed.raw_transaction
    tx_hash = w3.eth.send_raw_transaction(raw)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
    assert receipt.status == 1, "FaceRegistry deployment reverted"
    return receipt.contractAddress


@pytest.fixture(scope="module")
def bm(contract_address):
    from src.blockchain import BlockchainManager
    return BlockchainManager(RPC, TEST_KEY, contract_address)


class TestBlockchainIntegration:
    def test_chain_id(self, bm):
        assert bm.w3.eth.chain_id == 31337

    def test_anchor_and_verify(self, bm):
        face_hash = "0x" + "ab" * 32
        url = "https://en.wikipedia.org/wiki/Test_Person"
        fingerprint = bm.compute_data_fingerprint(face_hash, url)
        assert not bm.record_exists(face_hash), "hash should not exist before anchoring"
        receipt = bm.anchor_record(face_hash, url, fingerprint)
        assert receipt.status == 1
        assert bm.record_exists(face_hash), "record missing after anchoring"
        result = bm.verify_on_chain(face_hash, url)
        assert result["valid"] is True, "on-chain verification failed for anchored record"

    def test_tamper_evidence(self, bm):
        face_hash = "0x" + "cd" * 32
        url = "https://en.wikipedia.org/wiki/Tamper_Target"
        fingerprint = bm.compute_data_fingerprint(face_hash, url)
        bm.anchor_record(face_hash, url, fingerprint)
        tampered = bm.verify_on_chain(face_hash, url + "_tampered")
        assert tampered["valid"] is False, "tampered URL incorrectly validated"

    def test_unknown_hash_absent(self, bm):
        assert not bm.record_exists("0x" + "00" * 32), "random hash reported as existing"

    def test_fingerprint_is_sha256_of_hash_and_url(self, bm):
        import hashlib
        face_hash, url = "0x" + "11" * 32, "https://example.com/x"
        expected = "0x" + hashlib.sha256(f"{face_hash}:{url}".encode()).hexdigest()
        assert bm.compute_data_fingerprint(face_hash, url) == expected

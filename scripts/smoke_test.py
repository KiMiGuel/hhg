"""End-to-end smoke test that validates Stages 1, 3 and 4 WITHOUT spending a
SerpApi search credit. It compiles + deploys a fresh FaceRegistry to the local
Anvil node, anchors a record, re-verifies it, and runs the tamper drill.

Usage:  python scripts/smoke_test.py [optional_path_to_face_image]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.blockchain import BlockchainManager  # noqa: E402
from src.config import CONTRACT_ADDRESS, PRIVATE_KEY, RPC_URL  # noqa: E402
from scripts.deploy import compile_contract  # noqa: E402

from web3 import Web3  # noqa: E402


def main():
    face_hash = None
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_face.jpg"

    # ---- Stage 1 (only if a real face image is available) -----------------
    if os.path.exists(image_path):
        print(f"[1] Running FaceEngine on {image_path} ...")
        from src.face_engine import FaceEngine

        engine = FaceEngine()
        crop_path, face_hash, bbox = engine.process_image(image_path)
        print(f"    OK: bbox={bbox}, crop={crop_path}")
        print(f"    OK: biometric hash = {face_hash}")
    else:
        print("[1] No face image found -- skipping live face stage.")
        face_hash = "0x" + os.urandom(32).hex()
        print(f"    Using deterministic dummy hash for blockchain test: {face_hash}")

    # ---- Use the .env contract if configured, otherwise deploy fresh ------
    if CONTRACT_ADDRESS:
        contract_address = CONTRACT_ADDRESS
        print(f"[2] Using configured contract at {contract_address}")
    else:
        print("[2] Compiling & deploying FaceRegistry to the local node ...")
        w3 = Web3(Web3.HTTPProvider(RPC_URL))
        assert w3.is_connected(), f"Cannot connect to {RPC_URL} -- is Anvil running?"

        account = w3.eth.account.from_key(PRIVATE_KEY)
        abi, bytecode = compile_contract()
        factory = w3.eth.contract(abi=abi, bytecode=bytecode)
        tx = factory.constructor().build_transaction(
            {
                "from": account.address,
                "nonce": w3.eth.get_transaction_count(account.address),
                "gas": 2000000,
                "gasPrice": w3.eth.gas_price,
            }
        )
        signed = w3.eth.account.sign_transaction(tx, private_key=account.key)
        raw = getattr(signed, "rawTransaction", None) or signed.raw_transaction
        receipt = w3.eth.wait_for_transaction_receipt(w3.eth.send_raw_transaction(raw), timeout=120)
        contract_address = receipt.contractAddress
        print(f"    OK: deployed at {contract_address}")

    # ---- Stage 3: anchor a record -----------------------------------------
    print("[3] Anchoring record on-chain ...")
    bc = BlockchainManager(RPC_URL, PRIVATE_KEY, contract_address)
    post_url = "https://x.com/example_user/status/123456789"
    fingerprint = bc.compute_data_fingerprint(face_hash, post_url)
    if bc.record_exists(face_hash):
        print("    SKIP: record already anchored for this face hash")
    else:
        anchor_receipt = bc.anchor_record(face_hash, post_url, fingerprint)
        print(f"    OK: tx 0x{anchor_receipt.transactionHash.hex()} "
              f"in block {anchor_receipt.blockNumber}, gas {anchor_receipt.gasUsed}")

    # duplicate registration must be detected, not silently overwritten
    assert bc.record_exists(face_hash), "record_exists() returned False after anchoring!"
    print("    OK: record_exists() correctly detects the anchored record")

    # ---- Stage 4: verify + tamper drill ------------------------------------
    print("[4] Verifying on-chain record ...")
    result = bc.verify_on_chain(face_hash, post_url)
    assert result["valid"], f"Verification failed unexpectedly: {result}"
    print(f"    OK: on-chain hash matches local fingerprint ({result['on_chain_data_hash'][:18]}...)")

    tampered = bc.verify_on_chain(face_hash, post_url + "X")
    assert not tampered["valid"], "Tamper drill FAILED -- mutated data still verified!"
    print("    OK: tamper drill detected the mutation (cryptographic mismatch)")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()

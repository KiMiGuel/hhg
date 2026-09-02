"""One-click compiler & deployer for FaceRegistry.sol (local Anvil or Sepolia)."""
import os

from dotenv import load_dotenv
from solcx import compile_standard, install_solc
from web3 import Web3

load_dotenv()

RPC_URL = os.getenv("RPC_URL", "http://127.0.0.1:8545")
PRIVATE_KEY = os.getenv(
    "PRIVATE_KEY",
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
)
SOLC_VERSION = "0.8.20"


def compile_contract(source_path: str = "contracts/FaceRegistry.sol"):
    """Compile the Solidity source and return (abi, bytecode)."""
    install_solc(SOLC_VERSION)

    with open(source_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    compiled = compile_standard(
        {
            "language": "Solidity",
            "sources": {"FaceRegistry.sol": {"content": source_code}},
            "settings": {
                "outputSelection": {"*": {"*": ["abi", "evm.bytecode"]}},
            },
        },
        solc_version=SOLC_VERSION,
    )
    contract_data = compiled["contracts"]["FaceRegistry.sol"]["FaceRegistry"]
    return contract_data["abi"], contract_data["evm"]["bytecode"]["object"]


def deploy():
    print(f"Connecting to RPC: {RPC_URL}...")
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        raise ConnectionError(
            f"Could not connect to {RPC_URL}. Ensure Anvil is running "
            "(scripts/run_local_node.ps1) in another terminal."
        )

    account = w3.eth.account.from_key(PRIVATE_KEY)
    print(f"Deployer account: {account.address}")

    print(f"Ensuring Solidity compiler {SOLC_VERSION} is installed...")
    abi, bytecode = compile_contract()
    print("Contract compiled successfully.")

    print("Deploying FaceRegistry...")
    FaceRegistry = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx = FaceRegistry.constructor().build_transaction(
        {
            "from": account.address,
            "nonce": w3.eth.get_transaction_count(account.address),
            "gas": 2000000,
            "gasPrice": w3.eth.gas_price,
        }
    )
    signed = w3.eth.account.sign_transaction(tx, private_key=account.key)
    raw = getattr(signed, "rawTransaction", None) or signed.raw_transaction
    tx_hash = w3.eth.send_raw_transaction(raw)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    print("\n" + "=" * 60)
    print(" CONTRACT DEPLOYED SUCCESSFULLY!")
    print(f" Contract Address: {receipt.contractAddress}")
    print(f" Transaction Hash: 0x{receipt.transactionHash.hex()}")
    print(f" Gas Used        : {receipt.gasUsed}")
    print("=" * 60 + "\n")
    print(f'Update your .env file with:\nCONTRACT_ADDRESS="{receipt.contractAddress}"\n')
    return receipt.contractAddress


if __name__ == "__main__":
    deploy()

import hashlib

from web3 import Web3

# ABI for FaceRegistry.sol (registerRecord + getRecord)
CONTRACT_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "_faceHash", "type": "bytes32"},
            {"internalType": "string", "name": "_postUrl", "type": "string"},
            {"internalType": "bytes32", "name": "_dataHash", "type": "bytes32"},
        ],
        "name": "registerRecord",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "_faceHash", "type": "bytes32"}],
        "name": "getRecord",
        "outputs": [
            {"internalType": "bytes32", "name": "faceHash", "type": "bytes32"},
            {"internalType": "string", "name": "postUrl", "type": "string"},
            {"internalType": "bytes32", "name": "dataHash", "type": "bytes32"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"internalType": "address", "name": "registeredBy", "type": "address"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "faceHash", "type": "bytes32"},
            {"indexed": False, "internalType": "string", "name": "postUrl", "type": "string"},
            {"indexed": False, "internalType": "bytes32", "name": "dataHash", "type": "bytes32"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"indexed": True, "internalType": "address", "name": "registeredBy", "type": "address"},
        ],
        "name": "RecordRegistered",
        "type": "event",
    },
]


class BlockchainManager:
    """Stage 3/4: Web3 client for anchoring and auditing verification records."""

    def __init__(self, rpc_url: str, private_key: str, contract_address: str):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise ConnectionError(
                f"Failed to connect to blockchain RPC at {rpc_url}. "
                "Ensure the local node (Anvil) is running."
            )
        self.account = self.w3.eth.account.from_key(private_key)
        self.contract_address = Web3.to_checksum_address(contract_address)
        self.contract = self.w3.eth.contract(address=self.contract_address, abi=CONTRACT_ABI)

    @staticmethod
    def compute_data_fingerprint(face_hash: str, post_url: str) -> str:
        """Canonical SHA-256 fingerprint over the biometric hash and discovered post URL."""
        payload = f"{face_hash}:{post_url}".encode("utf-8")
        return "0x" + hashlib.sha256(payload).hexdigest()

    def record_exists(self, face_hash_hex: str) -> bool:
        """True if a record is already anchored for this face hash (getRecord reverts otherwise)."""
        try:
            self.contract.functions.getRecord(bytes.fromhex(face_hash_hex[2:])).call()
            return True
        except Exception:
            return False

    def anchor_record(self, face_hash_hex: str, post_url: str, data_hash_hex: str):
        """Broadcast a transaction registering the verified record on-chain."""
        face_hash_bytes = bytes.fromhex(face_hash_hex[2:])
        data_hash_bytes = bytes.fromhex(data_hash_hex[2:])

        nonce = self.w3.eth.get_transaction_count(self.account.address)
        tx = self.contract.functions.registerRecord(
            face_hash_bytes,
            post_url,
            data_hash_bytes,
        ).build_transaction({
            "from": self.account.address,
            "nonce": nonce,
            "gas": 300000,
            "gasPrice": self.w3.eth.gas_price,
        })

        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=self.account.key)
        # web3.py v6 exposes .rawTransaction (v7 renamed it to .raw_transaction)
        raw = getattr(signed_tx, "rawTransaction", None) or signed_tx.raw_transaction
        tx_hash = self.w3.eth.send_raw_transaction(raw)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        return receipt

    def verify_on_chain(self, face_hash_hex: str, expected_url: str) -> dict:
        """Fetch the on-chain record and recompute the fingerprint locally for comparison."""
        face_hash_bytes = bytes.fromhex(face_hash_hex[2:])
        rec = self.contract.functions.getRecord(face_hash_bytes).call()

        on_chain_face_hash = "0x" + rec[0].hex()
        on_chain_post_url = rec[1]
        on_chain_data_hash = "0x" + rec[2].hex()
        timestamp = rec[3]
        registered_by = rec[4]

        expected_fingerprint = self.compute_data_fingerprint(face_hash_hex, expected_url)
        is_valid = (
            on_chain_data_hash.lower() == expected_fingerprint.lower()
            and on_chain_post_url == expected_url
        )

        return {
            "valid": is_valid,
            "on_chain_face_hash": on_chain_face_hash,
            "on_chain_url": on_chain_post_url,
            "on_chain_data_hash": on_chain_data_hash,
            "local_recomputed_hash": expected_fingerprint,
            "block_timestamp": timestamp,
            "registrant": registered_by,
        }

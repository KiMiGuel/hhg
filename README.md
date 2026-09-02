# Face Identification & Blockchain Verification Pipeline (HH Goa 2026 — Task 3)

An end-to-end automated pipeline that accepts a facial image input, discovers
corresponding social media content through **dynamic** visual search (no
hardcoded results), and anchors the discovered metadata into an EVM blockchain
for **tamper-evident** verification.

## Key Features

- **Biometric face encoding** — OpenCV YuNet DNN face detection, a 15%-padded
  crop for search context, and a deterministic SHA-256 hash of the normalized
  128x128 grayscale face matrix (native EVM `bytes32` compatible).
- **Genuine dynamic web discovery** — upload to a free zero-auth public host
  (`catbox.moe`, with `tmpfiles.org` fallback) + Google Lens reverse search via
  SerpApi, dynamically filtered against a social platform whitelist
  (instagram.com, x.com/twitter.com, linkedin.com, facebook.com, reddit.com,
  youtube.com, pinterest.com).
- **Smart contract anchoring** — `contracts/FaceRegistry.sol` (Solidity ^0.8.20)
  stores the face hash, discovered post URL, combined SHA-256 fingerprint,
  block timestamp, and registrant address. Records are immutable once written.
- **On-chain auditing & tamper-evidence** — Stage 4 re-queries the contract,
  recomputes the fingerprint locally, asserts parity, and then runs an
  automated tamper drill that mutates one character of the URL and proves the
  cryptographic mismatch is detected.
- **Zero cost architecture** — local Anvil node (zero gas), free SerpApi tier
  (100 searches/month, no credit card), free ephemeral image hosting.

## Pipeline Architecture

```
+------------------+     +--------------------+     +--------------------------+
|   Input Image    | --> | Stage 1: Face      | --> | Stage 2: Dynamic Web     |
|  (local .jpg)    |     | Detection & Crop   |     | Search (Google Lens)     |
+------------------+     +--------------------+     +--------------------------+
                                                          |
                                                          v
+------------------+     +--------------------+     +--------------------------+
| Stage 4: Proof   | <-- | Stage 3: Hashing   | <-- | Discovered Social Post   |
| & Verification   |     | & On-Chain Proof   |     | (URL + metadata)         |
+------------------+     +--------------------+     +--------------------------+
```

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Face detection | OpenCV YuNet DNN (`FaceDetectorYN`, OpenCV 4.5.4+ / 5.x compatible) |
| Reverse search | SerpApi Google Lens engine |
| Image hosting | `catbox.moe` primary + `tmpfiles.org` fallback (free, no account) |
| Blockchain | Local EVM via Foundry **Anvil** (Sepolia optional) |
| Web3 client | `web3.py` v7 |
| Smart contract | Solidity ^0.8.20 via `py-solc-x` |

## Repository Structure

```
hhg/
├── contracts/FaceRegistry.sol   # Anchoring smart contract
├── data/                        # Put your input face image here
├── models/                      # YuNet face-detection ONNX model (auto-downloaded)
├── scripts/
│   ├── deploy.py                # Compile + deploy FaceRegistry.sol
│   ├── run_local_node.ps1       # Start the local Anvil node (Windows)
│   ├── reset_demo.ps1           # Fresh chain + redeploy for a clean recording
│   └── smoke_test.py            # Stage 1/3/4 test without SerpApi credits
├── src/
│   ├── config.py                # .env configuration loader
│   ├── face_engine.py           # YuNet detection, cropping, biometric hashing
│   ├── web_search.py            # Image hosting + Google Lens search
│   └── blockchain.py            # Web3 anchoring, auditing, tamper check
├── temp/                        # Ephemeral face crops (gitignored)
├── pipeline.py                  # Master end-to-end CLI
├── verify.py                    # Standalone on-chain audit CLI
├── requirements.txt
└── .env.example
```

## Setup & Execution Runbook

### 1. Install Foundry (Anvil)

Download from <https://getfoundry.sh> (or via `foundryup`). Verify:

```powershell
anvil --version
```

### 2. Python environment

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```powershell
Copy-Item .env.example .env
# Edit .env:
#   SERPAPI_KEY      -> free key from https://serpapi.com (no credit card)
#   RPC_URL          -> http://127.0.0.1:8545 (local Anvil)
#   PRIVATE_KEY      -> Anvil default pre-funded key (test only)
#   CONTRACT_ADDRESS -> filled after deployment
```

### 4. Start the local blockchain

In a separate terminal (keep it visible for the demo recording):

```powershell
.\scripts\run_local_node.ps1
```

### 5. Compile & deploy the contract

```powershell
python scripts/deploy.py
# Copy the printed contract address into CONTRACT_ADDRESS in .env
```

### 6. Add an input image

Place a face photo at `data/sample_face.jpg`. **Use an image you have the
right to use** — your own photo or a consenting person's public photo —
because Stage 2 performs a genuine reverse search of the face.

### 7. Run the pipeline

```powershell
python pipeline.py data/sample_face.jpg
```

### 8. Standalone audit

```powershell
python verify.py <face_hash_from_output> <discovered_url_from_output>
```

### Quick validation without SerpApi credits

```powershell
python scripts/smoke_test.py
```

## Blockchain Network Used

Primary: **local EVM via Anvil** (`http://127.0.0.1:8545`) — zero gas cost,
instant confirmations, deterministic demo. To use **Sepolia** instead, set
`RPC_URL` to a free public endpoint and fund a test account via a faucet.

## Limitations

- Face matching via Google Lens is heuristic — a face with no public web
  presence will not return social matches (test with a well-known face).
- The biometric hash is deterministic per pixel crop; slightly different
  input images produce different hashes (not a re-identification embedding).
- `tmpfiles.org` URLs are ephemeral; the anchor stores the URL as discovered
  at run time.
- Records in `FaceRegistry` are intentionally immutable — re-running the
  pipeline with the same face hash skips re-registration. Run
  `scripts/reset_demo.ps1` (or restart Anvil + `python scripts/deploy.py`)
  for a fresh chain before recording your demo video.
- Detection uses the YuNet DNN because OpenCV 5 removed the legacy
  `CascadeClassifier` API; the ONNX model is committed under `models/` and is
  also auto-downloaded if missing.
- Image hosting uses `catbox.moe` because `tmpfiles.org` links are not
  reliably crawlable by Google Lens; uploads are public by design, so do not
  upload sensitive photos.
- Sample image: `data/sample_face.jpg` is a freely licensed press photo of
  Satya Nadella from Wikimedia Commons ("MS-Exec-Nadella-Satya-2017"), used
  only as a demo input.
- Privacy note: reverse face search identifies people from images. Only use
  images of yourself or of people who have consented.

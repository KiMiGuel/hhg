# Face Identification & Blockchain Verification Pipeline (HH Goa 2026 — Task 3)

An end-to-end automated pipeline that accepts a facial image input, discovers
corresponding social media content through **dynamic** visual search (no
hardcoded results), and anchors the discovered metadata into an EVM blockchain
for **tamper-evident** verification.

> **One click runs the entire pipeline.** On Windows, double-click
> `run_live_pipeline.bat`. From any shell: `python main.py live --image
> data\sample_face.jpg --fresh-chain`. The runner auto-starts Anvil, deploys
> the smart contract, runs SerpApi Google Lens, anchors the result on-chain,
> and runs the tamper-evidence drill. See [One-Click Run](#one-click-run).

## Key Features

- **Biometric face encoding** — OpenCV YuNet DNN face detection, a 15%-padded
  crop for search context, and a deterministic SHA-256 hash of the **128-d SFace
  embedding vector** (a true biometric fingerprint, native EVM `bytes32` compatible).
- **Face re-identification** — cosine similarity between SFace embeddings lets the
  pipeline compare two faces and decide if they are the same person.
- **Genuine dynamic web discovery** — upload to a free zero-auth public host
  (`catbox.moe`, with `tmpfiles.org` fallback) + Google Lens reverse search via
  SerpApi, dynamically filtered against a social platform whitelist
  (instagram.com, x.com/twitter.com, linkedin.com, facebook.com, reddit.com,
  youtube.com, pinterest.com).
- **Identity-aware SerpApi integration** — the engine receives the biometric
  `face_hash` and a stable `image_sha256` from Stage 1 and returns a rich
  `LensResult` (selected match + full visual_matches list + knowledge-graph
  fallback + per-stage telemetry). The on-disk JSON cache is keyed on the
  source image bytes, so reruns of the same face are served in milliseconds.
- **Smart contract anchoring** — `contracts/FaceRegistry.sol` (Solidity ^0.8.20)
  stores the face hash, discovered post URL, combined SHA-256 fingerprint,
  block timestamp, and registrant address. Records are immutable once written.
- **On-chain auditing & tamper-evidence** — Stage 4 re-queries the contract,
  recomputes the fingerprint locally, asserts parity, and then runs an
  automated tamper drill that mutates one character of the URL and proves the
  cryptographic mismatch is detected.
- **Auto-generated audit artifacts** — every run emits a JSON + Markdown report
  under `reports/` with all hashes, tx data, every Lens candidate, and the
  verification result.
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
| Face encoding | OpenCV SFace (`FaceRecognizerSF`, 128-d biometric embeddings) |
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
├── models/                      # YuNet + SFace ONNX models (auto-downloaded)
├── reports/                     # Auto-generated audit artifacts (gitignored)
├── scripts/
│   ├── deploy.py                # Compile + deploy FaceRegistry.sol
│   ├── live_run.py              # Auto-Anvil + auto-deploy one-click runner (called by .bat)
│   ├── run_local_node.ps1       # Start the local Anvil node (Windows)
│   ├── reset_demo.ps1           # Fresh chain + redeploy for a clean recording
│   └── smoke_test.py            # Stage 1/3/4 test without SerpApi credits
├── src/
│   ├── config.py                # .env configuration loader
│   ├── face_engine.py           # YuNet detection, SFace encoding, cosine similarity
│   ├── web_search.py            # LensResult + LensMatch + identity-aware cache
│   ├── camera.py                # Real-time webcam face capture (SPACE to save)
│   ├── blockchain.py            # Web3 anchoring, auditing, tamper check
│   ├── accuracy.py              # Face quality checks, confidence, normalization
│   ├── preflight.py             # Pre-flight validation (incl. image-host reachability)
│   ├── registry.py              # On-chain record explorer + CSV/JSON export
│   ├── setup_wizard.py          # Interactive one-command setup
│   ├── menu.py                  # Arrow-key menu component (classic-cmd compatible)
│   ├── app.py                   # Full-app terminal dashboard (banner/status/menu)
│   ├── demo_data.py             # Pre-recorded result for --demo mode
│   └── report.py                # JSON + Markdown audit report generation
├── tests/
│   ├── test_face_engine.py      # Hash determinism, embedding, cosine similarity
│   ├── test_blockchain.py       # Fingerprint computation, tamper logic
│   └── test_report.py           # Report artifact generation
├── temp/                        # Ephemeral face crops (gitignored)
├── cache/                       # SerpApi result cache, keyed by (face_hash, image_sha256)
├── main.py                      # Unified CLI (run/live/verify/records/export/setup/smoke)
├── pipeline.py                  # Master end-to-end CLI (legacy entry point)
├── verify.py                    # Standalone on-chain audit CLI (legacy entry point)
├── run_live_pipeline.bat        # Windows one-click: auto-Anvil + auto-deploy + live pipeline
├── requirements.txt
└── .env.example
```

## One-Click Run

The fastest way to run the full pipeline is the `run_live_pipeline.bat` at the
repository root. It handles everything: it sets the working directory, verifies
the venv and Foundry are present, shows a pre-run banner, and runs the
non-demo pipeline end-to-end.

**Windows (double-click):**
1. Double-click `run_live_pipeline.bat` in the project root.
2. Read the pre-run banner. Press any key to start.
3. Wait ~5-15 s for the pipeline to finish; read the green SUCCESS banner.

**Windows (cmd / PowerShell):**
```powershell
.\run_live_pipeline.bat
```

**macOS / Linux / any shell:**
```bash
python main.py live --image data/sample_face.jpg --fresh-chain
```

**Optional environment variables** (works in both `.bat` and shell):

| Variable     | Effect                                                                              |
|--------------|-------------------------------------------------------------------------------------|
| `HHG_IMAGE`  | Override the input image (default: `data\sample_face.jpg`).                         |
| `HHG_REUSE=1`| Reuse the existing Anvil chain and contract (skip the `--fresh-chain` reset).       |

**What happens during the run** (in order, with auto-recovery at every step):

1. **(Re)start Anvil** at `http://127.0.0.1:8545` if not already running.
2. **Compile + deploy `FaceRegistry.sol`** if the contract is missing or stale,
   and write the new address back to `.env`.
3. **Stage 1** — detect the face, crop with 15% padding, derive the SHA-256
   biometric hash over the L2-normalized 128-d SFace embedding.
4. **Stage 2** — upload the crop to `catbox.moe` (fallback: `tmpfiles.org`)
   and run **SerpApi Google Lens** with the `source=social` filter. A
   per-stage telemetry line is printed:
   `STAGE2: upload=…ms lens=…ms serpapi_total=…s host=catbox.moe cache=…`.
   The first run consumes one SerpApi credit; reruns hit the on-disk cache
   (keyed by `face_hash` + `image_sha256`) and skip both the upload and the
   Lens call.
5. **Stage 3** — anchor the canonical SHA-256 fingerprint
   `sha256(faceHash + ":" + postUrl)` on-chain via `registerRecord`.
6. **Stage 4** — re-query the contract, recompute the fingerprint locally,
   and prove parity. Then mutate one character of the URL and prove the
   cryptographic mismatch is detected.
7. **Audit report** — `reports\report_<UTC>.json` (full `LensResult` + all
   candidates by domain) and `reports\report_<UTC>.md` (human-readable).

**Timing on the bundled `data\sample_face.jpg`** (local Anvil, single SerpApi credit):

| Run | Stage 1 | Stage 2 | Stage 3 | Stage 4 | **Total** |
|---|---|---|---|---|---|
| Fresh (live SerpApi)        | ~2.6 s | ~11 s  | ~0.4 s | ~0.1 s | **~14.7 s** |
| Warm rerun (cache HIT)      | ~2.6 s | ~0.06 s| ~0.05 s| ~0.1 s | **~2.7 s**  |

**Interactive dashboard (no command line needed):** `python main.py ui` or
just `python main.py` — choose **`LIVE One-Click`** from the menu. Same
auto-Anvil + auto-deploy behavior as the `.bat`, with arrow-key navigation.

## Setup & Execution Runbook

> **If you just want to run the pipeline, use the [One-Click Run](#one-click-run)
> section above. The setup wizard and manual steps below are only needed the
> first time you use this machine.**

### 1. Quick path: interactive setup wizard (recommended)

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python main.py setup
```

The wizard checks Python/deps, downloads the face models, verifies (or starts)
Anvil, deploys the contract, and writes `.env` — one command instead of six
manual steps.

### Manual setup (alternative)

<details>
<summary>Show manual steps</summary>

1. **Install Foundry (Anvil)** — <https://getfoundry.sh>, verify with `anvil --version`.
2. **Configure environment** — copy `.env.example` to `.env` (SerpApi key free at
   serpapi.com; no credit card).
3. **Start the local blockchain** — `.\scripts\run_local_node.ps1`.
4. **Deploy the contract** — `python scripts/deploy.py`, then copy the printed
   address into `CONTRACT_ADDRESS` in `.env`.

</details>

### 2. Add an input image

Place a face photo at `data/sample_face.jpg`. **Use an image you have the
right to use** — your own photo or a consenting person's public photo —
because Stage 2 performs a genuine reverse search of the face.

### 3. Run the pipeline

The recommended one-click path is described in [One-Click Run](#one-click-run).
From the shell, the equivalent commands are:

```powershell
# Windows: double-click run_live_pipeline.bat, OR from cmd:
.\run_live_pipeline.bat

# Any shell (macOS / Linux / Windows PowerShell):
python main.py live --image data\sample_face.jpg --fresh-chain

# Lower-level direct commands
python main.py run data/sample_face.jpg           # direct: live SerpApi Google Lens search
python main.py run data/sample_face.jpg --demo    # direct: offline, pre-recorded result
python main.py run group_photo.jpg --face 1       # direct: pick face #1 in multi-face images
```

**The dashboard** (default when you run `python main.py` with no arguments) is a
full-app terminal experience that works in the classic Windows console:
an ASCII banner, a live system-status header (node, contract, record count,
models, API key), and an arrow-key main menu (↑/↓ + Enter, number shortcuts,
Esc to exit) covering every operation: run pipeline, browse on-chain records,
verify, export, status, setup, smoke test, and help. The **`LIVE One-Click`**
menu item runs the same end-to-end live pipeline as the `.bat`, with
arrow-key navigation instead of a typed command.

### 4. Audit & explore the on-chain registry

```powershell
python main.py records                                   # list every anchored record
python main.py verify <face_hash> <post_url>             # audit by hash
python main.py verify-image data/sample_face.jpg <url>   # re-derive hash from image, then audit
python main.py export --format json                      # export registry for auditors
```

### Quick validation without SerpApi credits

```powershell
python main.py smoke
```

## Testing & CI

Tests cover hash determinism, embedding properties, cosine similarity,
fingerprint computation, tamper logic, and report generation. No network or
API keys needed.

```powershell
pip install pytest
python -m pytest tests/ -v
```

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs the test suite
on Python 3.10 / 3.11 / 3.12 on every push and PR.

## Blockchain Network Used

Primary: **local EVM via Anvil** (`http://127.0.0.1:8545`) — zero gas cost,
instant confirmations, deterministic demo. To use **Sepolia** instead, set
`RPC_URL` to a free public endpoint and fund a test account via a faucet.

## Security & Privacy Considerations

- **Only hashes go on-chain.** The pipeline anchors a SHA-256 fingerprint of
  the face embedding and the discovered URL — never the raw image or any
  personally identifiable pixel data. The on-chain record cannot be reversed
  to reconstruct the face.
- **Consent required.** Reverse face search identifies people from images.
  Only use images of yourself or of people who have consented to a public
  search. The sample image is a freely licensed press photo used purely as
  a demo input.
- **Ephemeral image hosting.** The cropped face is uploaded to a free public
  host (catbox.moe / tmpfiles.org) solely to generate a temporary URL for
  Google Lens. These hosts are public by design — do not upload sensitive
  photos.
- **Test credentials only.** The default private key is Anvil's well-known
  pre-funded test key. For Sepolia, use a dedicated test account with a
  faucet amount — never a mainnet key.
- **No PII in reports.** Auto-generated reports contain only hashes and
  public URLs, never raw images or personal data.

## Limitations

- Face matching via Google Lens is heuristic — a face with no public web
  presence will not return social matches (test with a well-known face).
- The biometric hash is deterministic per embedding; different input images
  of the same person produce different hashes unless they yield similar SFace
  embeddings (cosine similarity can be used to compare them).
- `catbox.moe` / `tmpfiles.org` URLs are ephemeral; the anchor stores the URL
  as discovered at run time.
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

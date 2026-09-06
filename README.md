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
│   ├── smoke_test.py            # Stage 1/3/4 test without SerpApi credits
│   ├── master_accuracy.py       # 5-axis accuracy gate (single command)
│   ├── name_torture.py          # 60+ pathological name-detection cases (100% gate)
│   ├── hard_eval.py             # Camera-sim degradation eval (>=90% gate)
│   └── accuracy_eval.py         # Person/name/data accuracy on the eval corpus
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

## Dataset setup

**No images are shipped with this repository.** All eval corpora, demo seeds, and
test fixtures are generated on demand by the scripts below. This keeps the repo
tiny and avoids redistributing third-party portraits.

```bash
# Accuracy corpus (12 public figures + 3 AI-generated abstain faces)
python scripts/build_test_corpus.py

# Hardness corpus (blur / low-JPEG / low-light / noise / rotate / small)
python scripts/hard_eval.py
```

After running `build_test_corpus.py`, the accuracy benchmark will work locally:

```bash
python scripts/master_accuracy.py --quick      # needs data/eval/
```

The face-detection tests (`tests/test_face_detection_real.py`) require a
separate `data/internet_test/` corpus of real-world images. They are
**skipped automatically** when the corpus is absent — no action required
unless you want those tests to run.

For demo/smoke-test runs, place any face image at `data/sample_face.jpg` or set
the `HHG_SAMPLE_IMAGE` environment variable (see `.env.example`).

---

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

## Accuracy Improvements (v2)

This release hardens the selection logic so the pipeline picks the
*correct* identity, not just any plausible-looking LinkedIn profile.

| Change | Where | Effect |
|---|---|---|
| Multi-scale face detection (TTA): 1× and 2× YuNet, IoU ≥ 0.5 dedupe | `src/face_engine.py:_detect_at_scale` | Sample face confidence 0.529 → **0.921** |
| YuNet threshold 0.5 → 0.3 | `src/face_engine.py:SCORE_THRESHOLD` | Real faces no longer silently dropped at the model boundary |
| Ensemble SFace embedding (orig + flip + ±5°) | `src/face_engine.py:process_image` | More stable 128-d biometric hash across pose/lighting |
| Inverted selection scoring (personal-name +120, KG +100, wiki +60, social +15, no-person -200) | `src/web_search.py:_score_visual_match` | Wikipedia / profile pages always beat random LinkedIn matches |
| Strict consensus gating (unique lead-name only) | `src/web_search.py:search_with_consensus` | "Prasad Wagh" no longer outranks "Satya Nadella" via spurious consensus |
| Cache hydration re-validation | `src/web_search.py:_hydrate_result` | Stale `consensus_hit` / `consensus=yes` tags stripped on read |
| Quality gates (Laplacian / brightness / contrast) | `src/accuracy.py:assess_face_quality` | Surfaces bad inputs in the audit report |

Measured on the 12 cached Lens results with verifiable ground truth
(see `scripts/test_accuracy.py` for the live harness):

| Scorer | Correct | Total | Accuracy |
|---|---|---|---|
| Old (v1) | 12 | 12 | 100% |
| New (v2) | 12 | 12 | 100% |

On inputs where Lens returned only generic LinkedIn profiles (i.e. the
photo had no public presence match), the new scorer abstains honestly
(no name guessed) once the cache-hydration fix strips the stale
`consensus_hit` flag.

## Testing & CI

Tests cover hash determinism, embedding properties, cosine similarity,
fingerprint computation, tamper logic, and report generation. No network or
API keys needed.

```powershell
pip install pytest
python -m pytest tests/ -v
```

### Accuracy verification (5-axis gate)

One command proves the accuracy claims across every axis:

```powershell
python scripts/master_accuracy.py          # full 5-axis run (cache mode)
python scripts/master_accuracy.py --quick  # skip the two slowest axes
python scripts/master_accuracy.py --json   # machine-readable summary
```

| Axis | What it proves | Gate |
|---|---|---|
| A — Name-detection torture | 60+ pathological titles extract the exact person name (or honestly reject) | 100% |
| B — Hard / camera-sim | Same-person match under blur, noise, low-res, JPEG q15, rotation, low light | ≥ 90% per variant |
| C — Platform coverage | Person / name / data accuracy on the eval corpus (cache replay) | ≥ 90% |
| D — Exact-image matching | dHash cross-platform exact-image tests + scoring boosts | ≥ 90% |
| E — Uploaded-file e2e | Detection, report, hint-fallback, no-match handling | ≥ 90% |

Current measured results: **A 100% · B 95.6% · C 93% · D/E 100%** (overall gate
PASSED). Axis B uses best-of-3 face matching — the standard 1:N identification
protocol — because under heavy degradation the detector may rank a different
face largest, exactly the situation the live pipeline's biometric-verification
stage handles.

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

## What is HHG?

HHG takes a face image and:

1. Detects + embeds the face (YuNet detector + SFace recognizer).
2. Uploads the face to a temporary, public, zero-auth image host.
3. Queries **four reverse-image search engines in parallel**:
   `google_lens`, `google_lens (source=social)`, `google_reverse_image`,
   and `bing_visual_search`.  Cross-engine duplicates get a scoring boost.
4. Runs **name-cluster voting** with cross-platform diversity bonus
   and an exact-image (`dHash`) proof signal.
5. **Corroborates** the discovered identity across 15 platforms
   (Instagram, Facebook, YouTube, X, LinkedIn, TikTok, Pinterest, Reddit,
   Threads, Quora, Medium, Flickr, Tumblr, Snapchat, Substack).
6. **Anchors** the evidence (face hash + post URL + fingerprint) on an
   EVM chain (Anvil by default) for tamper-evident audit.
7. Writes a JSON + Markdown report under `reports/`.

It is a **research / educational** project, not a forensic ID tool.  It
only uses public web results and explicitly **refuses to claim** an
identity when confidence is too low.

---

## Features

- **Multi-engine parallel search** with merge + dedup by URL.
- **Exact-image matching** across platform thumbnails via perceptual hash
  (`dHash`), with proof-level `+200` cluster boost.
- **Name-cluster voting** with cross-platform diversity, KG-title bonus,
  cross-crop consensus, and abstain thresholds that prevent
  single-Wikipedia-hit false positives.
- **Per-platform corroboration** via `site:<domain> "<name>"` lookups.
- **EVM anchoring** of biometric fingerprint (`face_hash + URL`) for
  tamper-evident audit; built-in tamper-detection drill.
- **Camera-sim degradation robustness** via multiscale TTA, blur-aware
  preprocessing, and best-of-3 1:N matching.
- **Identity-aware cache** keyed on `(face_hash, image_sha256)` so reruns
  cost zero SerpApi credits.
- **Master 5-axis accuracy harness** with hard reproducibility gates
  (`scripts/master_accuracy.py`).
- **Cold fresh live benchmark** for the strongest possible real-world
  accuracy measurement (`scripts/dev/run_cold_online_benchmark.py`).

---

## Demo

A typical `python main.py live` run produces (truncated):

```text
  IDENTITY MATCH
  Person / Entity   : Satya Nadella
  Matched title     : Satya Nadella - Wikipedia
  Platform          : wikipedia.org
  Source URL        : https://en.wikipedia.org/wiki/Satya_Nadella
  Selection reason  : vote_winner (score=320, votes=4, cluster=415, ...)
  Lens candidates   : 87
  SerpApi total     : 4.32s
  Upload + Lens     : 280ms + 1.95s
  Cache             : MISS
  Face hash         : 0x361968f0db...
  Verification      : CONFIRMED
  Mismatch detected : MISMATCH detected (expected)
```

See [`docs/benchmark.md`](docs/benchmark.md) for the full numerical
results of the 6-axis gate.

---

## Architecture

```text
+------------------+     +---------------------+     +---------------------------+
|   Input Image    | --> |  Stage 1: Face      | --> |  Stage 2: Multi-Engine   |
|  (local .jpg)    |     |  Detection + SFace  |     |  Reverse-Image Search    |
+------------------+     +---------------------+     +---------------------------+
                                                                  |
                                                                  v
+------------------+     +---------------------+     +---------------------------+
|  Stage 4: Proof  | <-- |  Stage 3: Hashing    | <-- |  + Exact-Image Match     |
|  & Verification  |     |  + On-Chain Anchor   |     |  + Cluster Voting        |
+------------------+     +---------------------+     +---------------------------+
```

Per-stage details in [`docs/architecture.md`](docs/architecture.md).

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/<you>/hhg.git
cd hhg

# 2. Install
python -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt

# 3. Configure
cp .env.example .env
# Edit .env and set SERPAPI_KEY (free at https://serpapi.com).
# Do NOT commit .env - it is gitignored.

# 4. Run the demo
python main.py live --image data/sample_face.jpg
```

The first run downloads the YuNet and SFace ONNX models into `models/`
(~5 MB total) and primes the SerpApi cache under `cache/`.

---

## Usage

### CLI

```bash
# Full live pipeline (camera/image -> SerpApi -> blockchain anchor)
python main.py live --image data/sample_face.jpg

# From a URL (downloads first, then runs the live pipeline)
python main.py run <image> [--url <image-url>]

# Use the demo (cached) result, no SerpApi call
python main.py run data/sample_face.jpg --demo

# Audit an on-chain record by face hash + URL
python main.py verify <face_hash> <url>

# List / export all on-chain records
python main.py records
python main.py export --format json
```

### One-click Windows launcher

`run_live_pipeline.bat` walks through banner &rarr; image pick
(&rarr; optional Anvil restart &rarr; deploy) &rarr; real run.

### Tests &amp; accuracy

```bash
# Unit tests (fast, no SerpApi, no Anvil)
pytest tests/ -v

# 5-axis local accuracy harness (no SerpApi, no network)
python scripts/master_accuracy.py

# Just the quick subset (A: name torture, B: hard-cam, C: platform coverage)
python scripts/master_accuracy.py --quick

# Live SerpApi accuracy (charges quota)
python scripts/accuracy_eval.py --live --min 90

# Cold fresh live benchmark (archives cache, runs live, restores cache)
python scripts/dev/run_cold_online_benchmark.py
```

---

## Accuracy &amp; benchmarks

The pipeline is gated by a 6-axis harness.  The current measured results
from this build are:

| Axis | What it measures | Gate | **Measured** |
|------|------------------|------|--------------|
| **A** | Name-detection torture (61 pathological titles) | 100% | **100%** (61/61) |
| **B** | Hard / camera-sim degradation (6 variants Ã— 15 images) | â‰¥90% | **95.6%** (86/90) |
| **C** | Platform/person coverage on cached eval | â‰¥90% | **100%** (15/15) |
| **D** | Exact-image + platform parsing + corroboration tests | â‰¥90% | **100%** (69/69) |
| **E** | Uploaded-file end-to-end tests | â‰¥90% | **100%** (33/33) |
| **F** | **Cold fresh live online** (real SerpApi) | â‰¥90% | **93.3%** (14/15) |

The cold fresh online benchmark (Axis F) is the closest measurement to
"real accuracy on the open internet".  It archives the user's `cache/`,
runs `scripts/accuracy_eval.py --live` on an empty cache, and restores
the original cache when finished.

The single Axis F failure is `eval_04.jpg` (Manoj Bajpayee) -- Lens
returned an unrelated "Civil War captain's uniform" result, and the
abstain policy let it through.  This is a known weak spot: when the only
match is a noisy, low-quality visual match with no corroboration, the
singleton-abstain rule sometimes still picks it.  See
[`docs/troubleshooting.md`](docs/troubleshooting.md) for workarounds.

See [`docs/benchmark.md`](docs/benchmark.md) for full outputs and
[`reports/`](reports/) for the JSON / Markdown audit artifacts.

---

## Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `SERPAPI_KEY` | (empty) | Required for `--live` / Axis F.  Free at https://serpapi.com. |
| `RPC_URL` | `http://127.0.0.1:8545` | EVM RPC endpoint.  Local Anvil by default. |
| `PRIVATE_KEY` | Anvil's well-known pre-funded test key | Test key only -- replace for any non-local deployment. |
| `CONTRACT_ADDRESS` | (empty) | Filled in automatically by `scripts/deploy.py`. |

See [`.env.example`](.env.example) for the full template.  `SERPAPI_KEY`
**must never be committed** -- the repo's `.gitignore` excludes `.env`.

---

## Security &amp; privacy

- See [`SECURITY.md`](SECURITY.md) for how to report vulnerabilities and
  what the project handles safely.
- HHG uses **only public, HTTPS** reverse-image search results.  It does
  not bypass private accounts, login gates, or captchas.
- No raw face crops, biometric hashes, or PII are committed to the repo.
- For real deployments, **rotate** `SERPAPI_KEY` and use a dedicated
  testnet wallet, never the Anvil default.

**Ethical reminder:** Reverse-image searching faces of private
individuals without consent is a privacy violation.  Only use HHG on
public figures with public-facing images, or on people who have
explicitly consented.

---

## Project layout

```text
hhg/
â”œâ”€â”€ main.py                       # Unified CLI (live / run / verify / records / ...)
â”œâ”€â”€ pipeline.py                   # 4-stage pipeline orchestration
â”œâ”€â”€ requirements.txt              # Runtime deps
â”œâ”€â”€ requirements-dev.txt          # Dev / test / lint deps
â”œâ”€â”€ pyproject.toml                # Ruff + black + pytest config
â”œâ”€â”€ Makefile                      # Convenience targets (make test / bench / lint)
â”œâ”€â”€ LICENSE
â”œâ”€â”€ CHANGELOG.md
â”œâ”€â”€ CONTRIBUTING.md
â”œâ”€â”€ CODE_OF_CONDUCT.md
â”œâ”€â”€ SECURITY.md
â”œâ”€â”€ README.md                     # (this file)
â”œâ”€â”€ data/                         # Generated on demand (see Dataset setup); not shipped in repo
â”œâ”€â”€ src/                          # face_engine, web_search, platform_profiles, ...
â”œâ”€â”€ scripts/                      # master_accuracy, accuracy_eval, live_run, hard_eval, ...
â”‚   â””â”€â”€ dev/                      # dev / reproducibility helpers
â”œâ”€â”€ tests/                        # pytest suite (no SerpApi, no Anvil)
â”œâ”€â”€ contracts/                    # FaceRegistry.sol
â”œâ”€â”€ .github/
â”‚   â”œâ”€â”€ workflows/ci.yml
â”‚   â”œâ”€â”€ ISSUE_TEMPLATE/
â”‚   â”œâ”€â”€ PULL_REQUEST_TEMPLATE.md
â”‚   â””â”€â”€ CODEOWNERS
â””â”€â”€ docs/                         # architecture, benchmark, troubleshooting
```

---

## Contributing

Contributions are very welcome -- bug reports, docs, tests, new platform
parsers, better scoring rules.  See [`CONTRIBUTING.md`](CONTRIBUTING.md)
and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).  All code contributions
should pass:

```bash
pytest tests/ -q
ruff check src tests scripts
python scripts/master_accuracy.py --quick
```

---

## License

MIT -- see [`LICENSE`](LICENSE).  Bundled ONNX models retain their
upstream MIT/Apache 2.0 licenses.

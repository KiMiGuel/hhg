# Troubleshooting

Common issues and how to fix them.

## "No visual matches found" on a real public figure

Most common causes:

1. **First run, models not downloaded yet** — check `models/` and look
   for `face_detection_yunet_2023mar.onnx` and
   `face_recognition_sface_2021dec.onnx`.  If missing, the pipeline
   will download them on first Stage 1 call.
2. **`SERPAPI_KEY` not set** — `cat .env` should show a non-empty value.
   `python main.py smoke` checks this.
3. **You exceeded the free SerpApi tier quota** (100 searches/month).
   Wait for the next billing cycle or upgrade.
4. **The image is heavily degraded** — run `scripts/hard_eval.py` to
   confirm the face is detectable at all.

## Axis F failure: Manoj Bajpayee is mislabeled as a Civil War captain

Known weak spot: when Lens returns only one visual match with a noisy,
low-quality thumbnail and no corroboration, the singleton-abstain rule
sometimes still picks it.  Workarounds:

- **Better query image** — use a clearer, higher-resolution, front-facing
  photo.  Even a different Wikipedia press photo of the same person
  usually works.
- **Manual confirmation** — for a few false-positive cases, the
  pipeline will surface the Wikipedia article as the strongest
  corroborating signal.  Cross-check the selected URL against
  expectations.
- **File-name hint** — name the file `manoj_bajpayee.jpg` so the
  filename-based hint search (`pipeline._hint_based_search`) can build
  a synthetic result pointing at the Wikipedia page.

This is a known weakness in the scoring model and is on the roadmap.

## "Failed to connect to Anvil"

- `python main.py live --no-chain` skips the on-chain anchoring
  entirely (good for offline / cache replay).
- `scripts/live_run.py` auto-starts Anvil if it is installed and
  `--no-auto-node` is not passed.
- Verify Foundry is installed and `anvil.exe` is on your PATH or at
  `~/.foundry/bin/anvil.exe`.

## "ModuleNotFoundError: No module named 'cv2'"

Install the runtime requirements:

```bash
pip install -r requirements.txt
```

If you already did and still see this, your virtual environment may
not be activated.  Check with `which python`.

## Pip install fails on Windows with "Microsoft Visual C++ 14.0 is required"

Some Python packages (`dlib`, etc.) need MSVC.  HHG deliberately
avoids those — `opencv-python` ships its own compiled wheels for
Windows.  If you see this error, you are probably trying to install a
package HHG does not require; remove it and re-run `pip install -r
requirements.txt`.

## Cache directory is huge

`cache/` is gitignored.  To clear it:

```bash
rm -rf cache/
```

The pipeline will rebuild it lazily on the next `--live` run.

## "Permission denied" on the blockchain anchoring step

- Use `--no-chain` to skip the on-chain step entirely.
- Or set `PRIVATE_KEY` to a testnet key with real test ETH (never the
  Anvil default key in production).
- Or set `RPC_URL` to a public Sepolia RPC and fund the deployer wallet
  via a public faucet.

## Tests pass locally but fail on CI

- The CI matrix runs on **Ubuntu** (not Windows).  If a test was
  inadvertently Windows-only, file an issue.
- The CI job `unit` ignores the Anvil-dependent and corpus-dependent
  tests (`test_blockchain_integration.py`, `test_face_detection_real.py`)
  because they require local Anvil / a downloaded corpus.
- The CI job `benchmark` runs the 5-axis harness in `--quick` mode.

## "YuNet model not found"

The first run downloads ~1.5 MB of ONNX weights.  If you are offline,
copy `face_detection_yunet_2023mar.onnx` and
`face_recognition_sface_2021dec.onnx` into `models/` from any machine
with internet.

## How do I run only the camera path?

```bash
python main.py live --camera
```

SPACE = capture, ESC = cancel.  The captured face is saved to
`temp/captured_face.jpg` and fed into the same 4-stage pipeline.

## Where do I report a security issue?

**Do not** open a public issue.  See [`SECURITY.md`](../SECURITY.md) for
the private reporting channel.

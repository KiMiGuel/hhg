# Architecture

This document gives a deeper view of the 4-stage pipeline, the scoring
formula, and the biometric anchoring model.  See [`README.md`](../README.md)
for the 30-second overview.

## 1. Pipeline stages

```
+-------------+      +-------------------+      +------------------------+
|  Input      | ---> |  Stage 1: Face     | ---> |  Stage 2: Multi-Engine  |
|  (image)    |      |  Detection + SFace |      |  Reverse-Image Search   |
+-------------+      +-------------------+      +------------------------+
                                                          |
                                                          v
                                                    +----------------+
                                                    |  Side path:    |
                                                    |  Exact-Image   |
                                                    |  Match (dHash) |
                                                    +----------------+
                                                          |
                                                          v
+-------------+      +-------------------+      +------------------------+
|  Stage 4     | <--- |  Stage 3: Hashing  | <--- |  Cluster Voting        |
|  Audit +     |      |  + On-Chain Anchor |      |  + Abstain policy      |
|  Verify      |      |                   |      |                        |
+-------------+      +-------------------+      +------------------------+
```

### 1.1 Stage 1 — Face detection + embedding

- Detector: **YuNet** (OpenCV Zoo, 2023Mar, MIT).
- Recognizer: **SFace** (OpenCV Zoo, 2021Dec, MIT), 128-d unit embedding.
- Multi-scale TTA: detect at native size and at 2x upscale; aligncrops
  both; compute embeddings; average.
- Degraded-image preprocessing: blur-aware unsharp mask + bilateral
  cleanup, then a 2x re-alignment pass on the full image.
- Best-of-N face selection: 1:N identification; compare up to 3 faces
  per image and take the best cosine match against the reference.

### 1.2 Stage 2 — Multi-engine reverse-image search

Four engines are queried in parallel via `ThreadPoolExecutor`:

| Engine | Source | What it adds |
|--------|--------|--------------|
| `google_lens` | SerpApi | canonical Lens reverse-image search |
| `google_lens` + `source=social` | SerpApi | Lens social profile slice |
| `google_reverse_image` | SerpApi | Google Images reverse match |
| `bing_visual_search` | SerpApi | Microsoft Bing visual match |

Results are merged and deduped by `link` (with thumbnail as tie-breaker).
Each merged match records the count of engines that surfaced it.

### 1.3 Scoring formula

For each visual match, `_score_visual_match` computes:

```
if not _looks_like_person(match): -120   # hard person-signal filter

+90  if wikipedia.org / .edu / .gov
+20  if OFFICIAL_TLDS_HINT (.gov .edu .org .io)
+20  if any SOCIAL_PLATFORMS in link
+15  if linkedin.com in link
+5   if facebook.com
+3   if instagram.com
-5   if youtube.com
-10  if reddit.com
-5   if tiktok.com
+50  if KG title is contained in match title
+10  if match title is "First Last" + role keyword (CEO, founder, ...)

+10 * (engine_hits - 1)   # cross-engine confirmation bonus
+200 if exact_image_match   # dHash proof-level boost
```

Name-cluster voting then aggregates per-name:

```
cluster_score = best_member_score
              + VOTE_WEIGHT * (votes - 1)              # = 8
              + (WIKI_MEMBER_WEIGHT if any wiki member) # = 15
              + (CONSENSUS_WEIGHT if cross-crop match) # = 25
              + (KG_TITLE_WEIGHT if KG match)          # = 50
              + (EXACT_IMAGE_BOOST if dHash match)     # = 200
              + CROSS_PLATFORM_BONUS * (n_platforms - 1) # = 15
              + domain_bonus (10 if 2+ domains)
```

The cluster with the highest score wins.  If `votes < 2` and there is no
cross-crop consensus, the cluster must also clear the stricter
`MIN_CLUSTER_SCORE_SINGLETON = 150` to be accepted (avoids
"single-Wikipedia-hit" false positives on AI/private faces).

### 1.4 Side path: exact-image matching

After Stage 2, `find_exact_matches` computes a `dHash` (perceptual hash)
for the uploaded image and every Lens thumbnail, then returns matches
with Hamming distance <= 12.  The pipeline re-tags the corresponding
`visual_matches` with `exact_image_match` and re-runs selection so the
exact-image proof affects the current selected identity (not just future
cache replays).

### 1.5 Stage 3 â€” On-chain anchoring

The pipeline computes:

```
fingerprint = SHA-256(face_hash || post_url)
```

and calls `FaceRegistry.anchorRecord(face_hash, post_url, fingerprint)` on
the EVM chain (Anvil by default, or any EVM RPC).  Subsequent audits:

- `recordExists(faceHash)` - quick "have we seen this face?" check
- `verifyOnChain(faceHash, postUrl)` - re-derives the fingerprint and
  compares it to the on-chain one (rejects tampered URLs)

### 1.6 Stage 4 â€” Tamper drill

After a successful anchor, the audit mutates one character of the URL
and re-runs `verifyOnChain`.  Expected: `valid = False`.  This is the
live tamper-evidence test in `pipeline.py`.

## 2. Caching

The Stage 2 cache is keyed on `(face_hash, image_sha256)`.  This means:

- A rerun of the same image on the same face is served from cache in
  ~5 ms (no SerpApi call).
- A new image of the same face, with a different SHA-256, gets a
  separate cache slot, so the soft "face-hash match" fallback only
  kicks in for *empty* previous results, never for positive matches
  on different photo bytes.
- A different person (different `face_hash`) never matches even with
  a hash collision (extremely unlikely with SFace embeddings).

## 3. Filesystem layout

```
src/face_engine.py          - YuNet detection, SFace embedding, multiscale TTA
src/web_search.py          - multi-engine SerpApi + cluster voting + scoring
src/image_match.py         - perceptual-hash cross-platform matching
src/platform_profiles.py   - 14-platform corroboration + dHash avatar match
src/biometric_verify.py    - SFace re-verification against candidate photos
src/accuracy.py            - face quality gates (blur / brightness / contrast)
src/blockchain.py          - EVM anchoring + tamper drill
src/visualizer.py          - Rich terminal panels
src/app.py                 - interactive dashboard
src/setup_wizard.py        - one-time setup helper
src/config.py              - .env loader
src/demo_data.py           - demo (no-SerpApi) result

pipeline.py                - orchestrates the 4 stages
main.py                    - unified CLI (live / run / verify / records / ...)
```

## 4. Performance budget

| Stage | Latency (typical) | Latency ceiling |
|-------|-------------------:|----------------:|
| 1. Detection + embedding | 200-500 ms | 2 s |
| 1. TTA + 2x pass | +100-200 ms | +1 s |
| 2. Catbox / tmpfiles upload | 200 ms - 1 s | 5 s |
| 2. Image host retries | (up to 2) | +4 s |
| 2. Multi-engine SerpApi (parallel) | 3-8 s | 20 s (each) |
| 3. EVM anchor | 1-2 s | 5 s |
| 4. Verify + tamper drill | 0.5-1 s | 3 s |
| **Total live** | **5-15 s** | **30 s** |

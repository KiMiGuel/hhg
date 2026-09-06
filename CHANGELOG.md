# Changelog

All notable changes to HHG are documented in this file.  The format is
loosely based on [Keep a Changelog](https://keepachangelog.com), and this
project adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Planned
* Better "unknown face" / private-face redaction rules.
* Optional web UI for non-CLI users.

## [0.3.0] - 2026-09-06

### Added
* **Multi-engine reverse-image search** – `google_lens`, `google_lens
  (source=social)`, `google_reverse_image`, and `bing_visual_search` are
  queried in parallel; results are merged and deduped by URL.
  Cross-engine duplicates get a `+10/engine` scoring boost.
* **Exact-image cross-platform matching** – dHash comparison between the
  uploaded image and every Lens thumbnail.  Matched candidates are tagged
  with `exact_image_match` and re-ranked inside the same run.
* **Expanded platform coverage** in `src/platform_profiles.py`:
  Instagram, Facebook, YouTube, X / Twitter, LinkedIn, TikTok, Pinterest,
  Reddit, Threads, Quora, Medium, Flickr, Tumblr, Snapchat, Substack.
* **Biometric parallel verification** – `BiometricVerifier` now fetches
  up to 8 candidate profile photos concurrently (`biometric_verify.py`).
* **New master accuracy harness** – `scripts/master_accuracy.py` runs five
  gates (name-torture, hard-degradation, platform-coverage,
  exact-image/platform, uploaded e2e) and reports a single PASS/FAIL.
* **New cold-fresh online benchmark** – `scripts/dev/run_cold_online_benchmark.py`
  archives the user's `cache/`, runs `accuracy_eval.py --live` on a clean
  cache, and restores the original cache when finished.

### Changed
* **Singleton abstain bar** – clusters whose name appears in exactly
  **one** title now require the stricter `MIN_CLUSTER_SCORE_SINGLETON`
  (150), preventing single-Wikipedia-hit false positives on AI/private
  faces.
* **Cross-platform bonus** in `select_by_voting` rewards names that
  appear on multiple distinct platforms (`+15/platform` beyond the first).
* **Better degraded-image preprocessing** – universal light bilateral
  cleanup, blur-aware unsharp mask, plus 2× multiscale re-alignment for
  improved matching on hard-camera inputs.
* **Refactored cascade** – engines are queried in parallel and merged
  with weighted dedup; the previous "stop at first engine with results"
  is removed.
* **Live run orchestration** (`scripts/live_run.py`) hardened for cold
  runs against real SerpApi responses.

### Fixed
* Exact-image hits now affect the current selected identity, not just
  later cache replays.
* Singleton Wikipedia hits on AI/private faces no longer leak as
  confident identities.
* Hard-eval uses the best-of-3 1:N identification protocol so a degraded
  image ranking a different face largest no longer scores 0.
* Cached visual-match reasons preserve `exact_image_match` and
  `x_engines=N` across cache replay.

### Measured (local, no SerpApi)
| Axis | Gate | Measured |
|------|------|----------|
| A — name torture | 100% | 100% (61/61) |
| B — hard degradation | ≥90% | 95.6% (86/90) |
| C — platform coverage (cache) | ≥90% | 100% (15/15) |
| D — exact/platform tests | ≥90% | 100% (69/69) |
| E — uploaded-file e2e | ≥90% | 100% (33/33) |

### Measured (cold fresh live SerpApi)
| Axis | Gate | Measured |
|------|------|----------|
| F — live online | ≥90% | 93.3% (14/15) |

## [0.2.0] - 2025-XX-XX (internal)

Earlier milestones covered initial YNet + SFace integration, the first
cascading SerpApi search, and the consensus two-crop selection logic.

## [0.1.0] - 2025-XX-XX (initial)

First working end-to-end pipeline: face detection → upload → reverse image
search → cluster voting → blockchain anchor.

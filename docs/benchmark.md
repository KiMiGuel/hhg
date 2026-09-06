# Benchmark results

This document is the running record of HHG's accuracy measurements
across all five benchmark axes.  Numbers below are from the v0.3.0
release build.

## Summary

| Axis | Source | Gate | **Measured** |
|------|--------|------|--------------|
| **A** | Local | 100% | **100%** (61/61) |
| **B** | Local | >=90% | **95.6%** (86/90) |
| **C** | Local (cache) | >=90% | **100%** (15/15) |
| **D** | Local tests | >=90% | **100%** (69/69) |
| **E** | Local tests | >=90% | **100%** (33/33) |
| **F** | Live SerpApi (cold) | >=90% | **93.3%** (14/15) |

All 6 axes pass the gate.  Cold-fresh live is 93.3% with one known
weak spot on `eval_04.jpg` (Manoj Bajpayee).

## How to reproduce

```bash
# A, B, C, D, E -- no SerpApi, no network
python scripts/master_accuracy.py --quick     # A + B + C
python scripts/master_accuracy.py            # A + B + C + D + E

# F -- real SerpApi call.  Charges quota.
python scripts/accuracy_eval.py --live --min 90 --json

# F "cold fresh" -- archives your cache/, runs live, restores cache
python scripts/dev/run_cold_online_benchmark.py
```

Results are written to `reports/real_accuracy_benchmark_*.json`.

## Axis A — Name torture (100%)

`scripts/name_torture.py` runs 61 pathological title cases across
Instagram, Facebook, YouTube, X, LinkedIn, TikTok, IMDb, etc. and
checks the strict name extractor.  Expected `100%`.

## Axis B — Hard / camera-sim (95.6%)

`scripts/hard_eval.py --min 90` runs 6 camera-sim degradations
(blur, noise, small, lowjpeg, rotate, lowlight) on 15 eval images.
Per-variant:

| Variant | Detection | Same-person | Gate |
|---|---:|---:|---|
| blur    | 15/15 = 100% | 14/15 = 93.3% | PASS |
| noise   | 15/15 = 100% | 14/15 = 93.3% | PASS |
| small   | 15/15 = 100% | 15/15 = 100%  | PASS |
| lowjpeg | 15/15 = 100% | 14/15 = 93.3% | PASS |
| rotate  | 15/15 = 100% | 15/15 = 100%  | PASS |
| lowlight| 15/15 = 100% | 14/15 = 93.3% | PASS |
| **OVERALL** | 90/90 = 100% | **86/90 = 95.6%** | **PASS** |

Hard eval uses best-of-3 face matching so a degraded image that ranks a
non-target face largest still scores well if the real face is in the
top 3.

## Axis C — Platform/person coverage (100%)

`scripts/accuracy_eval.py` (cache mode) re-evaluates the 15-image
eval corpus against cached Lens results.  Public figures: 12/12
correct.  AI/private faces: 3/3 correctly abstained.

```
PERSON accuracy: 15/15 = 100%
NAME   accuracy: 15/15 = 100%
DATA   accuracy: 15/15 = 100%
ABSTAIN sanity  : 3/3 (AI-private faces correctly refused)
```

## Axis D — Exact-image + platform tests (100%)

`pytest tests/test_platform_corroboration.py tests/test_platform_parsing.py`
runs 69 tests covering platform classification, name extraction,
corroboration flow, scoring boosts, and exact-image hashing.  Expected
`100%`.

## Axis E — Uploaded e2e tests (100%)

`pytest tests/test_face_engine.py tests/test_report.py tests/test_hint_based_search.py tests/test_no_visual_matches_fallback.py`
runs 33 tests covering Stage 1 hashing, report generation, hint-based
fallback, and the no-visual-matches early-exit path.  Expected `100%`.

## Axis F — Cold fresh live (93.3%)

`scripts/dev/run_cold_online_benchmark.py` archives the user's `cache/`,
runs `scripts/accuracy_eval.py --live` on an empty cache, and restores
the original cache when finished.  This is the closest measurement
to "real accuracy on the open internet".

Per-image (cold fresh SerpApi, 11 minutes total):

| # | Image | Expected | Selected | OK? |
|---|-------|----------|----------|----|
| 1 | eval_01.jpg | Satya Nadella     | Satya Nadella - Wikipedia            | yes |
| 2 | eval_02.jpg | Virat Kohli       | Virat Kohli - Simple English Wikipedia | yes |
| 3 | eval_03.jpg | Salman Khan       | Salman Khan - Wikipedia              | yes |
| 4 | eval_04.jpg | Manoj Bajpayee    | Civil War captain's uniform...       | **NO** |
| 5 | eval_05.jpg | Sundar Pichai     | Sundar Pichai - Wikipedia            | yes |
| 6 | eval_06.jpg | Elon Musk         | Elon Musk - Wikipedia                | yes |
| 7 | eval_07.jpg | Barack Obama      | Barack Obama - Wikipedia             | yes |
| 8 | eval_08.jpg | Narendra Modi     | Narendra Modi - Wikipedia            | yes |
| 9 | eval_09.jpg | Taylor Swift      | Taylor Swift - Wikipedia             | yes |
| 10 | eval_10.jpg | Cristiano Ronaldo | Cristiano Ronaldo (Football Star)   | yes |
| 11 | eval_11.jpg | Lionel Messi      | Lionel Messi - Wikipedia             | yes |
| 12 | eval_12.jpg | Tom Cruise        | Tom Cruise - Wikipedia               | yes |
| 13 | eval_13.jpg | (abstain)         | No confident identification         | yes |
| 14 | eval_14.jpg | (abstain)         | No confident identification         | yes |
| 15 | eval_15.jpg | (abstain)         | No confident identification         | yes |

**14/15 = 93.3%.**  See [troubleshooting.md](troubleshooting.md) for
the known Manoj Bajpayee failure and workarounds.

## Reproducibility note

Numbers above come from the v0.3.0 build at commit time.  SerpApi
result quality can drift slightly as Google tunes the Lens backend, so
expect ±2% variation on Axis F across runs.

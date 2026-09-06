# Data directory

This directory is **not shipped** in the repository. All files here are generated
on demand by the scripts documented in the main
[README](../README.md#dataset-setup).

## Corpus generation

| Corpus | Script | Source |
|--------|--------|--------|
| `eval/` | `scripts/build_test_corpus.py` | Wikipedia portraits (CC-BY-SA / public domain) + thispersondoesnotexist.com (AI-generated) |
| `hard_eval/` | `scripts/hard_eval.py` | Generated from `eval/` with synthetic distortions |
| `internet_test/` | Manual download | thispersondoesnotexist.com + Unsplash |

## Attribution

- **Wikipedia portraits** — Used under Creative Commons Attribution-ShareAlike
  (CC-BY-SA) or public domain licenses. Individual image attribution is
  recorded in `eval/manifest.json` after running `build_test_corpus.py`.
- **AI-generated faces** — From [thispersondoesnotexist.com](https://thispersondoesnotexist.com),
  which uses StyleGAN3-generated faces. No real person is depicted.
- **Unsplash images** — Used under the [Unsplash License](https://unsplash.com/license)
  (free for commercial and non-commercial use).

## `.gitignore`

All files in this directory are excluded from version control. See the parent
[`.gitignore`](../.gitignore) for the full exclusion rules.

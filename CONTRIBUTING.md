# Contributing to HHG (Face Identification & Blockchain Verification)

Thanks for your interest in contributing!  HHG is a small, hackathon-derived
research/educational codebase.  Issues, PRs, and benchmarks are all welcome.

## Quick start

```bash
# 1. Clone & install
git clone https://github.com/<you>/hhg.git
cd hhg
python -m venv .venv && source .venv/bin/activate      # (or .venv\Scripts\activate on Windows)
pip install -r requirements.txt -r requirements-dev.txt

# 2. Configure
cp .env.example .env
# Edit .env: set SERPAPI_KEY (free at https://serpapi.com). NEVER commit your key.

# 3. Run the tests + benchmarks
pytest tests/ -v
python scripts/master_accuracy.py --quick      # offline axes A/B/C/D
python scripts/master_accuracy.py            # full A/B/C/D/E (no SerpApi)
python scripts/accuracy_eval.py --live       # Axis F: real SerpApi call
```

## How to contribute

* **Bug reports** – Open an issue using the *Bug report* template. Include:
  Python version, OS, exact command + error output, and (if relevant) the
  `data/eval/` image that triggered it. **Never paste your `SERPAPI_KEY`.**
* **Feature requests** – Open an issue using the *Feature request* template.
* **Documentation** – PRs that fix typos, add diagrams, or improve the
  troubleshooting guide are very welcome.
* **Code** – PRs should:
  1. Have a clear scope (one feature / one fix per PR).
  2. Include or update tests (`tests/`).
  3. Pass `pytest tests/ -q` and `python scripts/master_accuracy.py --quick`
     locally before you open the PR.
  4. Not introduce new dependencies without discussion.

## Code style

* PEP 8, 4-space indent, type hints on public APIs.
* `ruff` and `black` are recommended:

```bash
pip install ruff black
ruff check src tests scripts
black --check src tests scripts
```

* Keep the dependency surface small.  If you need a new package, mention why
  in the PR description.

## Privacy & ethics reminder

HHG performs **reverse image search** on uploaded faces.  Before adding
examples or test data:

* Only use images you have the legal right to process.
* Never upload images of private individuals without consent.
* Never commit biometric hashes or raw face crops to the repo.
* Respect the `data/eval/manifest.json` ground truth – do not edit it
  unless you are also adding new labeled images.

## Reporting security issues

See [`SECURITY.md`](SECURITY.md).  Please **do not** open a public issue
for security-sensitive bugs; email the maintainer instead.

.PHONY: help install install-dev test lint format bench bench-quick bench-live clean clean-cache

PYTHON ?= python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev: install  ## Install dev/test/lint dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt

test:  ## Run the pytest suite (no SerpApi, no Anvil)
	$(PYTHON) -m pytest tests/ -v --tb=short \
		--ignore=tests/test_blockchain_integration.py \
		--ignore=tests/test_face_detection_real.py

lint:  ## Run ruff + black --check
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m black --check src tests scripts

format:  ## Auto-format with black + ruff --fix
	$(PYTHON) -m black src tests scripts
	$(PYTHON) -m ruff check --fix src tests scripts

bench:  ## Run the 5-axis local accuracy harness
	$(PYTHON) scripts/master_accuracy.py

bench-quick:  ## Run the quick subset of axes A/B/C
	$(PYTHON) scripts/master_accuracy.py --quick

bench-live:  ## Run the cold fresh live online benchmark (charges SerpApi quota)
	$(PYTHON) scripts/dev/run_cold_online_benchmark.py

clean:  ## Remove temp/runtime artifacts (does not touch cache)
	rm -rf temp/ reports/ __pycache__/ */__pycache__/ */*/__pycache__/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-cache:  ## Remove the SerpApi search cache (forces fresh live results)
	rm -rf cache/

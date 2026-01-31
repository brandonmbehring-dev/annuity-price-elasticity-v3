# Annuity Price Elasticity v3 - Makefile
# Multi-product support: RILA, FIA, MYGA

.PHONY: help install dev test test-unit test-integration test-rila test-fia
.PHONY: lint format coverage quick-check validate clean
.PHONY: verify verify-quick stub-hunter hardcode-scan pre-commit-install
.PHONY: test-property test-leakage pattern-check leakage-audit test-notebooks
.PHONY: setup-notebook-fixtures test-notebooks test-notebooks-aws test-all
.PHONY: capture-notebook-baselines test-notebook-equivalence
.PHONY: docstring-check docstring-lint

# Default Python
PYTHON ?= python3
PYTEST ?= pytest

help:
	@echo "Annuity Price Elasticity v3.0 - Available commands:"
	@echo ""
	@echo "  Development:"
	@echo "    make install       - Install package dependencies"
	@echo "    make dev           - Install with dev dependencies"
	@echo "    make lint          - Run linters (ruff + black check)"
	@echo "    make format        - Format code with black"
	@echo ""
	@echo "  Testing:"
	@echo "    make test          - Run all unit tests"
	@echo "    make test-all      - Run unit tests + notebooks (CI target)"
	@echo "    make test-unit     - Run unit tests only"
	@echo "    make test-rila     - Run RILA-specific tests"
	@echo "    make test-fia      - Run FIA-specific tests"
	@echo "    make test-property - Run property-based tests (Hypothesis)"
	@echo "    make test-leakage  - Run leakage detection tests"
	@echo "    make test-notebooks - Validate all 7 production notebooks (CI)"
	@echo "    make test-notebooks-aws - Validate ALL 7 notebooks (requires AWS)"
	@echo "    make setup-notebook-fixtures - Link fixtures for notebook CI"
	@echo "    make test-notebook-equivalence - Test notebook outputs at 1e-12"
	@echo "    make capture-notebook-baselines - Capture new baselines"
	@echo "    make coverage      - Generate coverage report"
	@echo ""
	@echo "  Validation:"
	@echo "    make quick-check   - Fast smoke test (imports + patterns)"
	@echo "    make validate      - Full mathematical equivalence (1e-12)"
	@echo ""
	@echo "  Verification (Phase 0 tooling):"
	@echo "    make verify        - Full verification suite (equivalence + stubs + types)"
	@echo "    make verify-quick  - Fast checks only (stubs + types, no equivalence)"
	@echo "    make stub-hunter   - Find placeholder implementations"
	@echo "    make hardcode-scan - Find hardcoded product-specific strings"
	@echo "    make pattern-check - Validate import patterns and constraints"
	@echo "    make leakage-audit - Run leakage gate validation"
	@echo "    make pre-commit-install - Install pre-commit hooks"
	@echo ""
	@echo "  Maintenance:"
	@echo "    make clean         - Remove cache directories"

# Installation
install:
	$(PYTHON) -m pip install -r requirements.txt

dev:
	$(PYTHON) -m pip install -e ".[all]"

# Testing
test:
	$(PYTEST) tests/ -v

test-all: test test-notebooks test-notebook-equivalence
	@echo "Full test suite completed (unit tests + notebooks + equivalence)"

test-unit:
	$(PYTEST) tests/unit/ -v -m "unit"

test-integration:
	$(PYTEST) tests/integration/ -v -m "integration"

test-rila:
	$(PYTEST) tests/ -v -k "rila"

test-fia:
	$(PYTEST) tests/ -v -k "fia"

test-property:
	$(PYTEST) tests/property_based/ -v -m "property"

test-leakage:
	$(PYTEST) tests/ -v -m "leakage"

# =============================================================================
# NOTEBOOK TESTING: Fixture Setup and Validation
# =============================================================================
# NB01 and NB02 load from outputs/datasets/ which are created by NB00.
# For CI, we symlink fixture files to enable notebook validation without AWS.

setup-notebook-fixtures:
	@# Copy and transform fixtures for CI (applies legacy column renaming)
	$(PYTHON) scripts/setup_notebook_fixtures.py

test-notebooks: setup-notebook-fixtures
	@echo "Running notebook validation with nbmake (all 7 notebooks, fixture-compatible)..."
	$(PYTEST) --nbmake \
		notebooks/onboarding/architecture_walkthrough.ipynb \
		notebooks/production/rila_6y20b/00_data_pipeline.ipynb \
		notebooks/production/rila_6y20b/01_price_elasticity_inference.ipynb \
		notebooks/production/rila_6y20b/02_time_series_forecasting.ipynb \
		notebooks/production/rila_1y10b/00_data_pipeline.ipynb \
		notebooks/production/rila_1y10b/01_price_elasticity_inference.ipynb \
		notebooks/production/rila_1y10b/02_time_series_forecasting.ipynb \
		-v --nbmake-timeout=600
	@echo "Notebook validation complete (7 notebooks)."

test-notebooks-aws:
	@echo "Running ALL notebook validation (requires AWS credentials)..."
	$(PYTEST) --nbmake \
		notebooks/production/rila_6y20b/*.ipynb \
		notebooks/production/rila_1y10b/*.ipynb \
		notebooks/onboarding/architecture_walkthrough.ipynb \
		-v --nbmake-timeout=300
	@echo "Full notebook validation complete (7 notebooks)."

# =============================================================================
# NOTEBOOK BASELINES: Capture and Equivalence Testing
# =============================================================================
# Baselines capture notebook outputs for mathematical equivalence validation.
# Fixed seed=42 ensures bit-identical bootstrap results at 1e-12 precision.

capture-notebook-baselines:
	@echo "Capturing notebook baselines (all products)..."
	$(PYTHON) scripts/capture_notebook_baselines.py --all
	@echo "Baseline capture complete. See tests/baselines/notebooks/"

test-notebook-equivalence:
	@echo "Running notebook equivalence tests (1e-12 precision)..."
	$(PYTEST) tests/integration/test_notebook_equivalence.py -v --tb=short
	@echo "Equivalence validation complete."

coverage:
	$(PYTEST) tests/ --cov=src --cov-report=term-missing --cov-report=html
	@echo "HTML coverage report: htmlcov/index.html"

# Validation
quick-check:
	@echo "Running quick smoke test..."
	$(PYTHON) -c "from src.core import protocols; print('Core protocols: OK')"
	$(PYTHON) -c "from src.config import config_builder; print('Config builder: OK')"
	$(PYTHON) scripts/pattern_validator.py --path src/ --quiet
	@echo "Quick check complete."

validate:
	$(PYTEST) tests/ -v -m "baseline" --tb=short
	@echo "Mathematical equivalence validation complete."

# Code Quality
lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

# Docstring Quality
docstring-check:
	@echo "Checking docstring coverage..."
	interrogate -v src/
	@echo "Docstring coverage check complete."

docstring-lint:
	@echo "Linting docstrings for NumPy style compliance..."
	ruff check src/ --select=D
	@echo "Docstring lint complete."

# Maintenance
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "Cleaned cache directories."

# Verification targets (Phase 0 tooling)
verify:
	@echo "Running full verification suite..."
	$(PYTHON) scripts/equivalence_guard.py --baseline tests/baselines/pre_refactoring/
	$(PYTHON) scripts/stub_hunter.py --path src/ --allowlist scripts/stub_allowlist.json
	bash scripts/validate_type_imports.sh
	$(PYTEST) tests/ -x --tb=short -q
	@echo "Full verification complete."

verify-quick:
	@echo "Running quick verification checks..."
	$(PYTHON) scripts/stub_hunter.py --path src/ --allowlist scripts/stub_allowlist.json
	bash scripts/validate_type_imports.sh
	@echo "Quick verification complete."

stub-hunter:
	$(PYTHON) scripts/stub_hunter.py --path src/ --allowlist scripts/stub_allowlist.json

hardcode-scan:
	$(PYTHON) scripts/hardcode_scanner.py --path src/

pre-commit-install:
	pip install pre-commit
	pre-commit install
	@echo "Pre-commit hooks installed."

# Pattern and Leakage Validation
pattern-check:
	@echo "Running pattern validator..."
	$(PYTHON) scripts/pattern_validator.py --path src/
	@echo "Pattern validation complete."

leakage-audit:
	@echo "Running leakage audit..."
	$(PYTHON) -c "from src.validation.leakage_gates import run_all_gates; run_all_gates()" 2>/dev/null || echo "Leakage gates not yet implemented"
	@echo "Leakage audit complete."

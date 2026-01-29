# RILA Price Elasticity - Refactoring Package

This package contains everything you need to refactor the RILA Price Elasticity system
in a non-AWS environment while maintaining mathematical equivalence at 1e-12 precision.

## Quick Start (5 minutes)

1. **Extract this package**:
   ```bash
   unzip rila-refactoring-package.zip
   cd rila-refactoring-package
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Validate package integrity**:
   ```bash
   python validate_package.py
   ```
   ✓ Should see "Package validation: PASSED"

5. **Run test suite to establish baseline**:
   ```bash
   pytest -v
   ```
   ✓ Should see ~2,500 tests pass

6. **Start refactoring!**
   - Make your changes to `src/`
   - Run tests frequently: `pytest`
   - Validate equivalence: `python validate_equivalence.py`

## What's Included

### Source Code (`src/`)
All production code for the RILA Price Elasticity system:
- **core/**: Pipeline orchestration and configuration
- **data/**: Data loading adapters (AWS + fixture modes)
- **features/**: Feature engineering and selection
- **models/**: Ridge regression and bootstrap models
- **products/**: Product-specific logic (FIA, RILA, MYGA, etc.)
- **notebooks/**: Notebook execution support
- **config/**: Configuration management

### Tests (`tests/`)
Complete test suite with 2,500+ tests:
- **unit/**: Fast unit tests (~1,200 tests)
- **integration/**: Module integration tests (~800 tests)
- **e2e/**: End-to-end pipeline tests (~200 tests)
- **performance/**: Performance baseline tests
- **property_based/**: Property-based tests with Hypothesis
- **fixtures/**: 74 MB of captured AWS data
- **baselines/**: 144 MB of validation baselines

### Documentation (`docs/`)
Complete project documentation:
- **onboarding/**: Getting started guides
  - `OFFLINE_DEVELOPMENT.md` ← **Key reference for offline workflow**
  - `GETTING_STARTED.md`
- **development/**: Development guides
  - `TESTING_GUIDE.md` ← **Comprehensive testing strategy**
  - `CODING_STANDARDS.md`
- **business/**: Business context and requirements
- **methodology/**: Statistical methodology
- **operations/**: Operational procedures

### Notebooks (`notebooks/`)
Jupyter notebooks for analysis:
- `00_data_pipeline.ipynb`: Data pipeline execution
- `01_price_elasticity_inference.ipynb`: Inference generation
- `02_time_series_forecasting.ipynb`: Forecasting

## No AWS Required

This package is **100% self-contained** for offline development:

✓ **All data captured in fixtures** (74 MB)
- Raw sales data from S3
- WINK competitive rates
- Market share weights
- Economic indicators
- All 10 pipeline stage outputs

✓ **All baselines pre-computed** (144 MB)
- Expected outputs for all pipeline stages
- Executed notebook results
- Golden reference data

✓ **Mock layer for AWS operations**
- S3 operations use fixtures instead
- No AWS credentials needed
- No network calls required

✓ **Complete offline test suite**
- 2,500+ tests run without AWS
- Mathematical equivalence validation
- Performance benchmarks

## Mathematical Equivalence Guaranteed

Your refactored code **must maintain mathematical equivalence** with the original:

### Precision Requirements
- **1e-12 precision**: Critical calculations (AIC, BIC, coefficients, predictions)
- **1e-6 precision**: General calculations (feature engineering, data transformations)

### Validation Levels

**Stage-by-Stage Validation**:
- Each of 10 pipeline stages validated independently
- Output must match baseline at 1e-12 precision
- Test: `pytest tests/integration/test_pipeline_stage_equivalence.py`

**Bootstrap Statistical Validation**:
- Coefficient distributions stable across runs
- Coefficient of variation < 5%
- Test: `pytest tests/integration/test_bootstrap_statistical_equivalence.py`

**End-to-End Pipeline Validation**:
- Complete pipeline from raw data to final predictions
- Final output matches baseline at 1e-12 precision
- Test: `pytest tests/e2e/test_full_pipeline_offline.py`

**Property-Based Validation**:
- Economic constraints satisfied (e.g., demand curves slope down)
- Edge cases handled correctly
- Test: `pytest tests/property_based/`

### Run All Equivalence Tests

```bash
python validate_equivalence.py
```

This runs all equivalence tests and generates a validation report.

## Development Workflow

### Recommended Workflow

1. **Before making changes**:
   ```bash
   pytest -v  # Establish baseline (all tests should pass)
   ```

2. **During refactoring** (after each significant change):
   ```bash
   # Run unit tests for the module you're working on
   pytest tests/unit/path/to/module/ -v

   # Run related integration tests
   pytest tests/integration/test_related.py -v
   ```

3. **After major changes**:
   ```bash
   # Validate mathematical equivalence
   python validate_equivalence.py
   ```

4. **Before committing**:
   ```bash
   # Run full test suite
   pytest -v

   # Update CHANGELOG_REFACTORING.md with your changes
   ```

### Test Execution Modes

**Fast feedback** (unit tests only, ~2 min):
```bash
pytest tests/unit/ -v
```

**Medium feedback** (unit + integration, ~7 min):
```bash
pytest tests/unit/ tests/integration/ -m "not aws" -v
```

**Full validation** (all tests, ~10 min):
```bash
pytest -v
```

**Equivalence only** (critical validation, ~10 min):
```bash
python validate_equivalence.py
```

## Project Structure

```
.
├── src/                          # Source code
│   ├── core/                     # Core pipeline
│   ├── data/                     # Data loading
│   ├── features/                 # Feature engineering
│   ├── models/                   # Models
│   └── products/                 # Product logic
│
├── tests/                        # Test suite
│   ├── fixtures/                 # Test data (74 MB)
│   │   └── rila/
│   │       ├── raw_sales_data.parquet
│   │       ├── raw_wink_data.parquet
│   │       └── [20 more files]
│   │
│   ├── baselines/                # Validation baselines (144 MB)
│   │   ├── rila/reference/
│   │   ├── golden/
│   │   └── [237 files]
│   │
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   ├── e2e/                      # E2E tests
│   ├── performance/              # Performance tests
│   └── property_based/           # Property tests
│
├── docs/                         # Documentation
│   ├── onboarding/
│   ├── development/
│   ├── business/
│   └── methodology/
│
├── notebooks/                    # Analysis notebooks
│   └── rila/
│
├── validate_package.py           # Validate package integrity
├── validate_equivalence.py       # Validate mathematical equivalence
├── prepare_reintegration.py      # Prepare for reintegration
│
├── README_REFACTORING.md         # This file (start here)
├── VALIDATION_GUIDE.md           # How to validate changes
├── REINTEGRATION_GUIDE.md        # How to bring changes back
├── CHANGELOG_REFACTORING.md      # Log your changes here
│
├── requirements.txt              # Python dependencies
├── requirements-dev.txt          # Dev dependencies
├── pyproject.toml               # Project config
└── pytest.ini                   # Pytest config
```

## Common Tasks

### Running Specific Tests

**Test a specific module**:
```bash
pytest tests/unit/features/test_feature_engineering.py -v
```

**Test a specific function**:
```bash
pytest tests/unit/features/test_feature_engineering.py::test_create_lag_features -v
```

**Test a specific pipeline stage**:
```bash
pytest tests/integration/test_pipeline_stage_equivalence.py::test_stage_05_sales_cleanup -v
```

**Run tests matching a pattern**:
```bash
pytest -k "test_bootstrap" -v
```

### Debugging Test Failures

**Show full error output**:
```bash
pytest tests/unit/path/to/test.py -vsx
```

**Drop into debugger on failure**:
```bash
pytest tests/unit/path/to/test.py --pdb
```

**Show print statements**:
```bash
pytest tests/unit/path/to/test.py -v -s
```

### Checking Code Coverage

```bash
pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

### Running Notebooks

```bash
jupyter notebook notebooks/rila/00_data_pipeline.ipynb
```

Or execute programmatically:
```bash
jupyter nbconvert --to notebook --execute notebooks/rila/00_data_pipeline.ipynb
```

## Reintegration

When you're ready to bring your changes back to the AWS environment:

1. **Run final validation**:
   ```bash
   pytest -v
   python validate_equivalence.py
   ```

2. **Update changelog**:
   - Edit `CHANGELOG_REFACTORING.md`
   - Document all changes, new dependencies, breaking changes

3. **Generate reintegration report**:
   ```bash
   python prepare_reintegration.py
   ```

4. **Create reintegration package**:
   ```bash
   ./create_reintegration_package.sh
   ```

5. **Transfer to AWS environment**:
   - Transfer the generated `rila-refactored-YYYYMMDD-HHMMSS.zip`
   - Follow instructions in `REINTEGRATION_GUIDE.md`

## Key Documents

- **VALIDATION_GUIDE.md**: Comprehensive guide to testing your changes
- **REINTEGRATION_GUIDE.md**: Step-by-step guide to bringing changes back
- **docs/onboarding/OFFLINE_DEVELOPMENT.md**: Complete offline development workflow
- **docs/development/TESTING_GUIDE.md**: Testing strategy and best practices
- **CHANGELOG_REFACTORING.md**: Document your changes here

## Troubleshooting

### "Fixture not found" error

**Cause**: Package extraction incomplete

**Fix**:
```bash
python validate_package.py  # Check package integrity
```

### Tests fail immediately after extraction

**Cause**: Dependencies not installed

**Fix**:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### "Import error" when running tests

**Cause**: Source code not in Python path

**Fix**:
```bash
# Ensure you're in the package root directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Or install in development mode
pip install -e .
```

### Mathematical equivalence test fails

**Cause**: You broke equivalence (or found a real bug)

**Fix**:
1. Identify which stage failed:
   ```bash
   pytest tests/integration/test_pipeline_stage_equivalence.py -v
   ```

2. Debug the specific stage:
   ```bash
   pytest tests/integration/test_pipeline_stage_equivalence.py::test_stage_XX -vsx
   ```

3. Compare your output with baseline in `tests/baselines/`

4. Fix the issue or relax tolerance if appropriate

## Success Criteria

Your refactoring is ready for reintegration when:

✓ All unit tests pass (100%)
✓ All integration tests pass (100%)
✓ All E2E tests pass (100%)
✓ Mathematical equivalence maintained (1e-12 precision)
✓ Performance baselines not regressed
✓ No new external dependencies (or approved if necessary)
✓ Documentation updated
✓ CHANGELOG_REFACTORING.md complete

Run `python prepare_reintegration.py` to verify all criteria.

## Support & Questions

### In this package:
- **VALIDATION_GUIDE.md**: How to test your changes
- **REINTEGRATION_GUIDE.md**: How to bring changes back
- **docs/onboarding/OFFLINE_DEVELOPMENT.md**: Offline development workflow
- **docs/development/TESTING_GUIDE.md**: Testing best practices

### Validation scripts:
- `python validate_package.py`: Check package integrity
- `python validate_equivalence.py`: Check mathematical equivalence
- `python prepare_reintegration.py`: Check reintegration readiness

### Common commands:
- `pytest -v`: Run all tests
- `pytest --lf`: Re-run last failed tests
- `pytest -k "pattern"`: Run tests matching pattern
- `pytest --cov=src`: Check code coverage

## Package Information

**Package created**: 2026-01-29 21:30:00

**Source git commit**: 2b3f7549 (branch: feature/refactor-eda-notebooks)

**Python version**: 3.12.9

**Package size**: ~220 MB
- Source code: ~5 MB
- Tests (excluding data): ~1 MB
- Fixtures: ~74 MB
- Baselines: ~144 MB
- Documentation: ~2 MB

---

**Ready to start refactoring? Run `python validate_package.py` to verify package integrity!**

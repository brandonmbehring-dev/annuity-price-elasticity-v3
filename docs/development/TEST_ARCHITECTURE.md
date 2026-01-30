# Test Architecture

**Last Updated:** 2026-01-30
**Status:** Production documentation for 95 test files, 2,467 tests
**Pattern Reference:** [docs/practices/testing.md](../practices/testing.md) (6-layer validation)

---

## Overview

The test suite implements the **6-layer validation architecture** from lever_of_archimedes patterns, providing defense-in-depth against model risk, data leakage, and regression bugs.

**Key Metrics:**
- 95 test files across 6 directories
- 2,467 total tests
- 44% code coverage (target: >60% for core modules)
- 1e-12 precision mathematical equivalence

---

## Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    6-LAYER VALIDATION                        │
├─────────────────────────────────────────────────────────────┤
│ Layer 6: Property-Based Tests                               │
│   tests/property_based/                                     │
│   Invariants that ALWAYS hold, random input generation      │
├─────────────────────────────────────────────────────────────┤
│ Layer 5: End-to-End Tests                                   │
│   tests/e2e/                                                │
│   Full pipeline from raw data to inference output           │
├─────────────────────────────────────────────────────────────┤
│ Layer 4: Integration Tests                                  │
│   tests/integration/                                        │
│   Multi-module workflows, adapter interactions              │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: Unit Tests                                         │
│   tests/unit/                                               │
│   Single function/class isolation testing                   │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: Input Validation                                   │
│   src/validation/, fail-fast at function entry             │
│   (Not separate tests - validated via unit tests)          │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: Type Safety                                        │
│   Type hints + mypy checks                                 │
│   (Enforced at lint time, not test time)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
tests/
├── conftest.py              # Shared fixtures and pytest configuration
├── unit/                    # Layer 3: Unit tests
│   ├── core/                # Protocols, types, exceptions
│   ├── data/                # Data extraction, preprocessing
│   ├── features/            # Feature engineering, selection
│   ├── models/              # Inference models
│   ├── config/              # Configuration builders
│   ├── products/            # Product methodologies
│   ├── validation/          # Validators
│   └── validation_support/  # Mock layers
├── integration/             # Layer 4: Integration tests
│   └── test_*.py            # Multi-module workflows
├── e2e/                     # Layer 5: End-to-end tests
│   └── test_*.py            # Full pipeline tests
├── property_based/          # Layer 6: Property-based tests
│   └── test_*.py            # Hypothesis-generated tests
├── performance/             # Performance baselines
│   └── test_*.py            # Timing and memory tests
├── baselines/               # Reference outputs
│   ├── rila/                # RILA baseline coefficients
│   ├── fia/                 # FIA baseline coefficients
│   └── golden/              # Golden master outputs
└── fixtures/                # Test data
    ├── rila/                # RILA fixture data (73MB)
    └── fia/                 # FIA fixture data (14MB)
```

---

## Layer Details

### Layer 3: Unit Tests (tests/unit/)

**Purpose:** Test individual functions and classes in isolation.

**Coverage Target:** 80%+ for modules, 90%+ for core systems.

| Directory | Tests | Focus |
|-----------|-------|-------|
| `unit/core/` | ~50 | Protocols, types, exceptions, registry |
| `unit/data/` | ~200 | Extraction, preprocessing, pipelines |
| `unit/features/` | ~300 | Engineering, aggregation, selection |
| `unit/models/` | ~100 | Inference, bootstrap, ridge |
| `unit/config/` | ~150 | Builders, product configs |
| `unit/products/` | ~100 | RILA/FIA methodologies |
| `unit/validation/` | ~200 | Schema, constraint validators |

**Example:**
```python
# tests/unit/features/test_engineering_timeseries.py
class TestCreateLagFeatures:
    """Unit tests for lag feature creation."""

    def test_creates_expected_lags(self, sample_weekly_data):
        """Verify lag features created for all specified lags."""
        result = create_lag_features(sample_weekly_data, lags=[1, 2, 3])
        assert 'feature_t1' in result.columns
        assert 'feature_t2' in result.columns
        assert 'feature_t3' in result.columns

    def test_excludes_lag_0_competitors(self, sample_weekly_data):
        """Verify no lag-0 competitor features created."""
        result = create_lag_features(sample_weekly_data, lags=[0, 1, 2])
        competitor_cols = [c for c in result.columns if 'competitor' in c.lower()]
        assert not any('_t0' in c for c in competitor_cols)
```

### Layer 4: Integration Tests (tests/integration/)

**Purpose:** Test multi-module workflows and component interactions.

**Focus Areas:**
- Data adapter chains (S3 → preprocessing → features)
- Interface wiring (UnifiedNotebookInterface → adapters)
- Pipeline stages (10-stage data pipeline)
- AWS/Fixture equivalence

**Example:**
```python
# tests/integration/test_data_pipeline_integration.py
class TestDataPipelineIntegration:
    """Test complete data pipeline from extraction to features."""

    def test_fixture_pipeline_produces_valid_features(self, rila_fixture_adapter):
        """Full pipeline run produces valid feature set."""
        df = rila_fixture_adapter.load_sales_data()
        rates = rila_fixture_adapter.load_competitive_rates()

        # Run through pipeline
        features = run_pipeline(df, rates)

        # Validate output
        assert features.shape[1] == 598, "Expected 598 features"
        assert not features.isnull().any().any(), "No nulls allowed"
```

### Layer 5: End-to-End Tests (tests/e2e/)

**Purpose:** Test complete user workflows from input to output.

**Focus Areas:**
- Full inference workflow (load → preprocess → model → predict)
- Export functionality (Excel, CSV, plots)
- Multi-product support
- Error recovery

**Example:**
```python
# tests/e2e/test_inference_workflow.py
class TestInferenceWorkflow:
    """End-to-end inference workflow tests."""

    def test_complete_inference_workflow(self):
        """Run complete inference from fixture to results."""
        interface = create_interface("6Y20B", environment="fixture")
        df = interface.load_data()
        results = interface.run_inference(df)

        # Validate full output
        assert 'coefficients' in results
        assert 'metrics' in results
        assert results['metrics']['r_squared'] > 0.5
```

### Layer 6: Property-Based Tests (tests/property_based/)

**Purpose:** Test invariants that should ALWAYS hold using random inputs.

**Framework:** Hypothesis

**Key Properties:**
- Coefficient signs (own rate +, competitor rate -)
- No future leakage (features only use past data)
- Bounded predictions (0 ≤ sales ≤ max_historical)
- Mathematical equivalence (refactored = original at 1e-12)

**Example:**
```python
# tests/property_based/test_economic_constraints.py
from hypothesis import given, strategies as st

class TestEconomicConstraints:
    """Property-based tests for economic invariants."""

    @given(rate_change=st.floats(min_value=-0.05, max_value=0.05))
    def test_own_rate_positive_elasticity(self, fitted_model, rate_change):
        """Higher own rate should never decrease predicted sales."""
        baseline = fitted_model.predict(baseline_features)
        higher_rate = fitted_model.predict(
            baseline_features.assign(prudential_rate_t0=baseline_rate + rate_change)
        )

        if rate_change > 0:
            assert higher_rate >= baseline, "Higher rate should not decrease sales"
```

---

## Test Markers

Use pytest markers for selective test execution:

| Marker | Description | Command |
|--------|-------------|---------|
| `@pytest.mark.slow` | Tests taking >30 seconds | `pytest -m "not slow"` |
| `@pytest.mark.aws` | Requires AWS credentials | `pytest -m "not aws"` |
| `@pytest.mark.e2e` | End-to-end tests | `pytest -m e2e` |
| `@pytest.mark.integration` | Integration tests | `pytest -m integration` |
| `@pytest.mark.property` | Property-based tests | `pytest -m property` |
| `@pytest.mark.leakage` | Leakage detection tests | `pytest -m leakage` |

---

## Running Tests

### Quick Commands

```bash
# Fast smoke test (~30 seconds)
make quick-check

# Full test suite (~5 minutes)
make test

# By layer
make test-unit          # Unit tests only
make test-integration   # Integration tests
make test-e2e           # End-to-end tests
make test-property      # Property-based tests

# By product
make test-rila          # RILA-specific tests
make test-fia           # FIA-specific tests

# With coverage
make coverage           # Generate HTML report
pytest --cov=src --cov-report=html
```

### Parallel Execution

```bash
# Auto-detect CPU cores
pytest -n auto

# Specific parallelism
pytest -n 8
```

---

## Fixture System

### Three-Tier Hierarchy

| Tier | Size | Load Time | Use Case |
|------|------|-----------|----------|
| SMALL | 20-100 rows | <0.1s | Unit tests, TDD iteration |
| MEDIUM | 100-1,000 rows | 0.1-1s | Integration tests |
| LARGE | Full dataset | 1-5s | E2E validation, baselines |

### Fixture Locations

- `tests/fixtures/rila/` - RILA fixture data (73MB)
- `tests/fixtures/fia/` - FIA fixture data (14MB)
- `tests/baselines/` - Reference outputs for equivalence testing

### Mathematical Equivalence

Fixtures maintain **1e-12 precision equivalence** with AWS data:

```python
def test_fixture_aws_equivalence():
    """Fixture produces same results as AWS at 1e-12 precision."""
    fixture_result = run_with_fixture()
    aws_result = load_baseline("aws_mode/coefficients.parquet")

    max_diff = np.abs(fixture_result - aws_result).max()
    assert max_diff < 1e-12
```

---

## Coverage Targets

| Category | Target | Current | Notes |
|----------|--------|---------|-------|
| Overall | >60% | 44% | Focus on core modules |
| `src/core/` | 90% | ~85% | Protocols, exceptions |
| `src/data/` | 80% | ~70% | Extraction, pipelines |
| `src/features/` | 80% | ~65% | Engineering, selection |
| `src/models/` | 80% | ~50% | Inference models |
| `src/validation/` | 80% | ~75% | Validators |
| Scripts | 60% | ~30% | Harder to test |

---

## Anti-Pattern Detection

### Automated Leakage Gates

Run via `make leakage-audit`:

1. **Shuffled Target Test**: Model should fail on random targets
2. **Temporal Boundary Check**: No future data in training
3. **Competitor Lag Check**: Minimum 2-week lag enforced
4. **Suspicious Results Check**: Performance not "too good"
5. **Coefficient Sign Check**: Economic theory validation

### Pattern Validator

Run via `make pattern-check`:

- Import hygiene validation
- Lag-0 competitor detection
- Constraint enforcement
- Competing implementation detection

---

## Adding New Tests

### Checklist for New Functionality

- [ ] Unit tests for each new function (Layer 3)
- [ ] Integration test if crosses module boundaries (Layer 4)
- [ ] E2E test if user-facing workflow (Layer 5)
- [ ] Property test if invariant should hold (Layer 6)
- [ ] Leakage test if touches competitor data

### Test File Naming

```
tests/unit/[module]/test_[module_name].py
tests/integration/test_[workflow_name].py
tests/e2e/test_[scenario_name].py
tests/property_based/test_[property_name].py
```

---

## Related Documentation

- [../practices/testing.md](../practices/testing.md) - 6-layer validation pattern
- [../practices/LEAKAGE_CHECKLIST.md](../practices/LEAKAGE_CHECKLIST.md) - Pre-deployment validation
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Comprehensive testing guide
- [TEST_COVERAGE_REPORT.md](TEST_COVERAGE_REPORT.md) - Current coverage tracking

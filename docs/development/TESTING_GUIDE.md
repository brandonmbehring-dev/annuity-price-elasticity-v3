# Testing Guide

Comprehensive guide to testing practices in the RILA Price Elasticity project.

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Test Structure](#test-structure)
3. [Running Tests](#running-tests)
4. [Writing Unit Tests](#writing-unit-tests)
5. [Fixture Patterns](#fixture-patterns)
6. [Hierarchical Fixture Strategy](#hierarchical-fixture-strategy)
7. [Performance Baseline Testing](#performance-baseline-testing)
8. [Property-Based Testing](#property-based-testing)
9. [Common Test Patterns](#common-test-patterns)
10. [AWS Integration Testing](#aws-integration-testing)
11. [Coverage Goals](#coverage-goals)
12. [Troubleshooting](#troubleshooting)

---

## Testing Philosophy

### Core Principles

1. **Tests Document Behavior**: Tests are the most reliable documentation of how code actually works
2. **Fast Feedback**: Unit tests run in milliseconds, enabling rapid iteration
3. **Mathematical Equivalence**: Critical validation ensures refactoring preserves results
4. **Test What Matters**: Focus on business logic, not trivial getters/setters

### What We Test

[DONE] **DO Test**:
- Business logic and calculations
- Edge cases (NaN, empty DataFrames, boundary conditions)
- Error handling and validation
- Mathematical correctness
- API contracts and interfaces

[ERROR] **DON'T Test**:
- External libraries (pandas, numpy already tested)
- Trivial property getters
- Private implementation details that may change
- Exact formatting of error messages (test that error occurs, not exact text)

---

## Test Structure

### Directory Layout

```
tests/
├── conftest.py                      # Shared fixtures (55 fixtures)
├── fixtures/
│   └── rila/                        # Test data files
│       ├── sales_data.parquet
│       ├── competitor_rates.parquet
│       └── market_weights.parquet
├── unit/
│   ├── features/
│   │   ├── test_competitive_features.py      # 36 tests, 100% coverage
│   │   ├── test_engineering_timeseries.py    # 28 tests, 100% coverage
│   │   ├── test_engineering_integration.py   # 45 tests, 99% coverage
│   │   ├── test_engineering_temporal.py      # 37 tests, 98% coverage
│   │   └── test_aggregation_strategies.py    # 39 tests, 95% coverage
│   ├── models/
│   │   ├── test_inference.py
│   │   └── test_forecasting.py
│   ├── validation_support/
│   │   └── test_mathematical_equivalence.py  # 27 tests, 57% coverage
│   └── config/
│       └── test_builders.py
└── integration/
    └── test_end_to_end.py
```

### Naming Conventions

- **Test files**: `test_<module_name>.py`
- **Test functions**: `test_<function>_<scenario>`
- **Fixtures**: descriptive names with context

**Examples**:
```python
# Good
def test_weighted_aggregation_basic()
def test_weighted_aggregation_zero_weights_fallback()
def test_lag_features_handles_nan()

# Bad
def test_1()  # Not descriptive
def test_weighted_agg()  # Too abbreviated
def test_all_cases()  # Too broad
```

---

## Running Tests

### Quick Commands

```bash
# Run all tests
make test

# Run specific test file
pytest tests/unit/features/test_competitive_features.py -v

# Run specific test function
pytest tests/unit/features/test_competitive_features.py::test_median_ranking_basic -v

# Run tests matching pattern
pytest -k "weighted" -v  # All tests with "weighted" in name

# Run with coverage
make coverage

# Run tests in parallel (faster)
pytest -n auto

# Run only failed tests from last run
pytest --lf

# Run tests and stop on first failure
pytest -x
```

### Verbosity Levels

```bash
pytest               # Minimal output
pytest -v            # Verbose (show each test)
pytest -vv           # Very verbose (show detailed diffs)
pytest -s            # Show print statements
pytest --tb=short    # Short traceback
pytest --tb=line     # One-line traceback
```

### Coverage Commands

```bash
# Coverage for specific module
pytest tests/unit/features/test_competitive_features.py \
    --cov=src.features.competitive_features \
    --cov-report=term-missing

# Coverage with branch coverage
pytest --cov=src --cov-branch --cov-report=html

# View HTML coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

---

## Writing Unit Tests

### Basic Test Structure

```python
"""
Unit Tests for <Module Name>
============================

Tests for src/<path>/<module>.py covering:
- <functionality 1>
- <functionality 2>
- <edge cases>

Target: 85% coverage

Author: <Your Name>
Date: YYYY-MM-DD
"""

import numpy as np
import pandas as pd
import pytest

from src.<path>.<module> import function_to_test


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def simple_test_data():
    """Descriptive docstring explaining fixture purpose."""
    return pd.DataFrame({
        'column_a': [1, 2, 3],
        'column_b': [4, 5, 6]
    })


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================

def test_function_basic(simple_test_data):
    """Test basic functionality with simple data."""
    result = function_to_test(simple_test_data)

    # Assert return type
    assert isinstance(result, pd.DataFrame)

    # Assert shape
    assert len(result) == len(simple_test_data)

    # Assert values
    assert result['output'].iloc[0] == expected_value


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

def test_function_handles_nan():
    """Test that function handles NaN values correctly."""
    df = pd.DataFrame({'value': [1.0, np.nan, 3.0]})
    result = function_to_test(df)

    # Verify NaN handling
    assert not result.isna().any().any()


def test_function_empty_dataframe():
    """Test function with empty DataFrame."""
    empty = pd.DataFrame()
    result = function_to_test(empty)

    assert len(result) == 0


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

def test_function_raises_on_missing_column():
    """Test that function raises ValueError on missing column."""
    df = pd.DataFrame({'wrong_column': [1, 2, 3]})

    with pytest.raises(ValueError, match="column not found"):
        function_to_test(df)


# =============================================================================
# IMMUTABILITY TESTS
# =============================================================================

def test_function_immutable(simple_test_data):
    """Test that function doesn't modify original DataFrame."""
    original_cols = set(simple_test_data.columns)
    _ = function_to_test(simple_test_data)

    assert set(simple_test_data.columns) == original_cols
```

### Test Organization Principles

1. **Group by functionality**: Use section comments to organize tests
2. **One assertion per concept**: Don't test multiple unrelated things
3. **Clear test names**: Test name should describe what it verifies
4. **Minimal setup**: Use fixtures to reduce setup code
5. **Test isolation**: Tests should not depend on each other

---

## Fixture Patterns

### Using Built-in Fixtures

The project provides 55 centralized fixtures in `tests/conftest.py`:

```python
def test_with_fixture_data(raw_sales_data, aws_config):
    """Use pre-loaded fixtures from conftest.py."""
    assert len(raw_sales_data) > 1000
    assert aws_config['xid'] == "x259830"
```

### Creating Custom Fixtures

#### Simple Data Fixture

```python
@pytest.fixture
def simple_rates_data():
    """Simple DataFrame for testing rate calculations."""
    return pd.DataFrame({
        'company_a': [4.5, 4.6, 4.7],
        'company_b': [4.2, 4.3, 4.4],
        'company_c': [4.0, 4.1, 4.2]
    })
```

#### Parameterized Fixture

```python
@pytest.fixture(params=[3, 5, 7])
def n_companies(request):
    """Test with different numbers of companies."""
    return request.param

def test_function_various_company_counts(n_companies):
    # Will run 3 times: n_companies=3, 5, 7
    companies = [f"company_{i}" for i in range(n_companies)]
    # ... test logic
```

#### Fixture with Teardown

```python
@pytest.fixture
def temp_file():
    """Create temporary file, cleanup after test."""
    import tempfile
    import os

    fd, path = tempfile.mkstemp()
    yield path  # Test runs here

    # Teardown
    os.close(fd)
    os.unlink(path)
```

#### Session-Scoped Fixture (Expensive Setup)

```python
@pytest.fixture(scope="session")
def large_dataset():
    """Load once per test session (expensive operation)."""
    import time
    print("\\nLoading large dataset...")
    time.sleep(2)  # Simulate slow load
    return pd.read_parquet("tests/fixtures/large_file.parquet")
```

### Fixture Scopes

- `function` (default): New instance per test function
- `module`: One instance per test file
- `session`: One instance for entire test run
- `class`: One instance per test class

**Use session scope for**:
- Large data files (>10MB)
- Expensive computations
- Database connections

**Use function scope for**:
- Mutable data (DataFrames being modified)
- Quick fixtures (<10ms)

---

## Hierarchical Fixture Strategy

### Overview

The project uses a **three-tier hierarchical fixture system** optimized for test performance and developer productivity:

- **TIER 1 - SMALL**: Fast unit tests (20-100 rows, <0.1s load time)
- **TIER 2 - MEDIUM**: Integration tests (100-1000 rows, <1s load time)
- **TIER 3 - LARGE**: End-to-end tests (full production data, 1-5s load time)

This strategy enables **complete offline development** with fixtures while maintaining mathematical equivalence at 1e-12 precision.

### Three-Tier Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ TIER 1: SMALL (UNIT TESTS)                                  │
├─────────────────────────────────────────────────────────────┤
│ Size:         20-100 rows, 5-20 features                    │
│ Bootstrap:    10-100 samples                                │
│ Load time:    < 0.1 seconds                                 │
│ Scope:        function (fresh per test)                     │
│ Use case:     Fast unit tests, TDD iteration                │
│ Examples:     - test_calculation_accuracy()                 │
│               - test_validation_rules()                     │
│               - test_edge_cases()                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ TIER 2: MEDIUM (INTEGRATION TESTS)                          │
├─────────────────────────────────────────────────────────────┤
│ Size:         100-1000 rows, 20-100 features               │
│ Bootstrap:    100-1000 samples                              │
│ Load time:    0.1-1 seconds                                 │
│ Scope:        module (shared within file)                   │
│ Use case:     Module integration, pipeline stages           │
│ Examples:     - test_feature_engineering_pipeline()         │
│               - test_module_interactions()                  │
│               - test_aggregation_strategies()               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ TIER 3: LARGE (END-TO-END TESTS)                           │
├─────────────────────────────────────────────────────────────┤
│ Size:         Full dataset (167 weeks, 598 features)       │
│ Bootstrap:    10,000 samples (production)                   │
│ Load time:    1-5 seconds                                   │
│ Scope:        session (load once per test run)             │
│ Use case:     E2E validation, mathematical equivalence      │
│ Examples:     - test_full_pipeline_equivalence()            │
│               - test_production_baseline_comparison()       │
│               - test_notebook_execution()                   │
│ Marker:       @pytest.mark.slow (skip in fast CI)          │
└─────────────────────────────────────────────────────────────┘
```

### Fixture Selection Decision Tree

**Start here:**

```
Q1: Are you testing pure calculation logic or validation rules?
    ├─ YES → Use TIER 1 (SMALL)
    │         - tiny_dataset (20 rows × 5 features)
    │         - small_bootstrap_config (10 samples)
    │         - small_inference_dataset (20 rows × 10 features)
    │
    └─ NO → Continue to Q2

Q2: Are you testing module integration or pipeline stages?
    ├─ YES → Use TIER 2 (MEDIUM)
    │         - medium_dataset (100 rows × 50 features)
    │         - medium_bootstrap_config (100 samples)
    │         - medium_inference_dataset (100 rows × 50 features)
    │
    └─ NO → Continue to Q3

Q3: Are you testing full E2E pipeline or mathematical equivalence?
    ├─ YES → Use TIER 3 (LARGE)
    │         - full_production_dataset (167 rows × 598 features)
    │         - production_bootstrap_config (10,000 samples)
    │         - large_inference_dataset (167 rows × 598 features)
    │         - Mark test with @pytest.mark.slow
    │
    └─ NO → Use parametrize with multiple tiers

Q4: Are you testing behavior scales correctly across data sizes?
    └─ YES → Use @pytest.mark.parametrize with all three tiers
              to verify calculation works regardless of data size
```

### Usage Examples

#### TIER 1: SMALL Fixtures (Unit Tests)

**When to use:**
- Testing pure calculation logic
- Testing validation functions
- Testing edge cases (NaN, zeros, negatives)
- Rapid TDD iteration

**Example:**
```python
def test_aic_calculation_accuracy(tiny_dataset, small_bootstrap_config):
    """Test AIC calculation with minimal data (< 0.1s)."""
    model = BootstrapInference(small_bootstrap_config)
    result = model.fit_predict(tiny_dataset)

    # AIC should be positive
    assert result['metrics']['AIC'] > 0

    # Should complete very fast
    assert len(tiny_dataset) == 20  # Only 20 rows

def test_validation_logic_with_tiny_dataset(tiny_dataset):
    """Test validation rules on small data."""
    positive_sales = (tiny_dataset['sales'] > 0).all()
    assert positive_sales == True
    assert tiny_dataset.isnull().sum().sum() == 0
```

#### TIER 2: MEDIUM Fixtures (Integration Tests)

**When to use:**
- Testing module integration
- Testing pipeline stage output
- Testing feature engineering transformations
- Validating intermediate results

**Example:**
```python
def test_feature_engineering_pipeline(medium_dataset):
    """Test feature engineering with realistic data (< 1s)."""
    # Medium dataset has 50 features including lags
    assert len(medium_dataset) == 100
    assert medium_dataset.shape[1] == 50

    # Test feature transformation
    engineered = create_polynomial_features(medium_dataset)

    assert engineered.shape[1] > medium_dataset.shape[1]
    assert engineered.isnull().sum().sum() == 0

def test_module_integration(medium_dataset, medium_bootstrap_config):
    """Test module interactions with moderate data."""
    n_bootstrap = medium_bootstrap_config['n_bootstrap']

    # Integration test with 100 bootstrap samples
    bootstrap_results = []
    for i in range(min(5, n_bootstrap)):
        sample = medium_dataset.sample(n=len(medium_dataset), replace=True)
        result = calculate_statistics(sample)
        bootstrap_results.append(result)

    assert len(bootstrap_results) == 5
```

#### TIER 3: LARGE Fixtures (E2E Tests)

**When to use:**
- Testing full end-to-end pipeline
- Validating mathematical equivalence with baseline
- Production simulation
- Performance regression testing

**Example:**
```python
@pytest.mark.slow
def test_full_pipeline_mathematical_equivalence(full_production_dataset):
    """Test full pipeline with production data (1-5s).

    Validates mathematical equivalence at 1e-12 precision.
    Marked as slow - skip in fast CI runs.
    """
    # Full production dimensions
    assert 150 < len(full_production_dataset) < 250
    assert full_production_dataset.shape[1] > 590

    # Run full pipeline
    pipeline = DataPipeline()
    result = pipeline.run_full_pipeline(full_production_dataset)

    # Compare to baseline at 1e-12 precision
    baseline = load_baseline_results()
    validate_equivalence(result, baseline, tolerance=1e-12)

@pytest.mark.slow
def test_production_bootstrap_simulation(
    full_production_dataset, production_bootstrap_config
):
    """Test with full production bootstrap configuration.

    Uses 10,000 bootstrap samples - takes 30-60s.
    """
    assert production_bootstrap_config['n_bootstrap'] == 10000

    model = BootstrapInference(production_bootstrap_config)
    results = model.fit_predict(full_production_dataset)

    # Validate production performance
    assert results['metrics']['R²'] > 0.75
    assert results['metrics']['MAPE'] < 0.15
```

### Performance Optimization Guidelines

#### [DONE] GOOD: Use smallest fixture that validates behavior

```python
def test_aic_formula(tiny_dataset, small_bootstrap_config):
    """Fast test of AIC calculation (< 0.1s)."""
    result = calculate_aic(tiny_dataset, small_bootstrap_config)
    assert result > 0
```

#### [ERROR] BAD: Using full production dataset for unit test

```python
def test_aic_formula(full_production_dataset, production_bootstrap_config):
    """Slow test of AIC calculation (> 30s) - ANTIPATTERN!"""
    result = calculate_aic(full_production_dataset, production_bootstrap_config)
    assert result > 0
```

#### [DONE] GOOD: Use MEDIUM for integration tests

```python
def test_competitive_features_integration(medium_dataset):
    """Sweet spot for integration testing (< 1s)."""
    lag_features = create_lag_features(medium_dataset)
    assert len(lag_features.columns) == 20
```

#### [ERROR] BAD: Using SMALL for integration test

```python
def test_competitive_features_integration(tiny_dataset):
    """Only 20 rows - insufficient for realistic integration testing."""
    lag_features = create_lag_features(tiny_dataset)
    # Lag features may not be meaningful with only 20 rows
```

### Available Fixtures Reference

#### TIER 1 - SMALL Fixtures

```python
@pytest.fixture
def tiny_dataset():
    """Minimal dataset: 20 rows × 5 features, function scope."""
    # Features: own_cap_rate, competitor_avg_rate, vix, dgs5, sales
    # Use for: Unit tests, calculation validation

@pytest.fixture
def small_bootstrap_config():
    """Fast bootstrap: 10 samples, function scope."""
    # Use for: Testing bootstrap logic without waiting

@pytest.fixture
def small_inference_dataset():
    """Small inference data: 20 rows × 10 features, function scope."""
    # Use for: Rapid TDD iteration on inference logic
```

#### TIER 2 - MEDIUM Fixtures

```python
@pytest.fixture(scope="module")
def medium_dataset():
    """Medium dataset: 100 rows × 50 features, module scope."""
    # Features: 10 base + 20 lag features + 10 polynomials + 10 macros
    # Use for: Integration tests, pipeline stages

@pytest.fixture(scope="module")
def medium_bootstrap_config():
    """Moderate bootstrap: 100 samples, module scope."""
    # Use for: Integration testing with realistic sampling

@pytest.fixture(scope="module")
def medium_inference_dataset():
    """Medium inference data: 100 rows × 50 features, module scope."""
    # Use for: Testing pipeline stage outputs
```

#### TIER 3 - LARGE Fixtures

```python
@pytest.fixture(scope="session")
def full_production_dataset():
    """Full production: 167 weeks × 598 features, session scope."""
    # Loaded from: tests/fixtures/rila/final_weekly_dataset.parquet
    # Use for: E2E tests, mathematical equivalence

@pytest.fixture(scope="session")
def production_bootstrap_config():
    """Production bootstrap: 10,000 samples, session scope."""
    # Use for: Production simulation, baseline comparison

@pytest.fixture(scope="session")
def large_inference_dataset():
    """Full inference data: 167 rows × 598 features, session scope."""
    # Alias for full_production_dataset
    # Use for: Mathematical equivalence testing
```

### Parametrized Tests Across Fixture Sizes

Test behavior scales correctly by parametrizing across all three tiers:

```python
@pytest.mark.parametrize("fixture_name,expected_min_size", [
    ("tiny_dataset", 20),
    ("medium_dataset", 100),
])
def test_calculation_scales_across_sizes(fixture_name, expected_min_size, request):
    """Verify calculation works regardless of data size."""
    dataset = request.getfixturevalue(fixture_name)

    # Calculation should work on any size
    result = dataset['sales'].mean()

    assert result > 0
    assert len(dataset) >= expected_min_size
```

### Anti-Patterns to Avoid

#### [ERROR] ANTIPATTERN 1: Using LARGE fixture for simple unit test

```python
@pytest.mark.skip(reason="Anti-pattern demonstration")
def test_antipattern_large_for_unit(full_production_dataset):
    """[ERROR] BAD: Takes 1-3 seconds just to test simple calculation.

    Why bad:
    - Loads 167 weeks × 598 features (1-3s)
    - Only tests simple mean calculation
    - Should use tiny_dataset instead
    - Slows down test suite unnecessarily
    """
    result = full_production_dataset['sales'].mean()
    assert result > 0
```

**Fix:** Use `tiny_dataset` instead (< 0.1s)

#### [ERROR] ANTIPATTERN 2: Using SMALL fixture for integration test

```python
@pytest.mark.skip(reason="Anti-pattern demonstration")
def test_antipattern_small_for_integration(tiny_dataset):
    """[ERROR] BAD: Only 20 rows insufficient for integration testing.

    Why bad:
    - Only 20 rows may not expose integration issues
    - Insufficient lag features for realistic testing
    - Should use medium_dataset instead
    """
    lag_features = create_lag_features(tiny_dataset)
    # Lag features won't be meaningful with only 20 rows
```

**Fix:** Use `medium_dataset` instead (100 rows)

### Performance Impact

Real-world timing measurements:

| Fixture Tier | Load Time | Test Execution | Total Time | Use Case |
|-------------|-----------|----------------|------------|----------|
| SMALL | < 0.01s | 0.05-0.1s | **< 0.1s** | Unit tests |
| MEDIUM | 0.1-0.5s | 0.2-1s | **< 1s** | Integration |
| LARGE | 1-3s | 2-30s | **3-60s** | E2E, baselines |

**Example test suite impact:**
- 100 unit tests with SMALL fixtures: ~10 seconds
- 100 unit tests with LARGE fixtures: ~5 minutes (30x slower!)
- 20 integration tests with MEDIUM fixtures: ~20 seconds
- 5 E2E tests with LARGE fixtures: ~3 minutes (marked as slow)

### Best Practices Summary

1. **Use SMALLEST fixture that validates your behavior**
   - Unit test simple calculation? → tiny_dataset
   - Integration test pipeline stage? → medium_dataset
   - E2E test full pipeline? → full_production_dataset

2. **Use MEDIUM for most integration tests** (sweet spot)
   - Fast enough for rapid iteration (< 1s)
   - Realistic enough to catch integration issues
   - Module-scoped (shared within test file)

3. **Reserve LARGE for E2E and baseline comparisons**
   - Mark with `@pytest.mark.slow`
   - Skip in fast CI runs with `pytest -m "not slow"`
   - Use for mathematical equivalence validation
   - Session-scoped (load once per test run)

4. **Scope fixtures appropriately**
   - SMALL: function scope (fresh per test, mutable)
   - MEDIUM: module scope (shared within file, faster)
   - LARGE: session scope (load once, immutable)

5. **Run targeted test sets during development**
   ```bash
   # Fast unit tests only (< 1 minute)
   pytest tests/unit/ -m "not slow"

   # Integration tests (< 5 minutes)
   pytest tests/integration/

   # Full suite including E2E (10-15 minutes)
   pytest

   # Only slow E2E tests
   pytest -m slow
   ```

See `tests/test_hierarchical_fixtures.py` for comprehensive examples demonstrating proper usage of all three tiers.

---

## Performance Baseline Testing

### Overview

Performance baseline tests establish timing and memory consumption thresholds for critical operations. These tests detect performance regressions early by measuring actual execution time and memory usage against established baselines.

**Purpose:**
- Detect performance regressions before production
- Establish clear performance expectations
- Identify memory leaks and excessive memory consumption
- Provide performance documentation for operations

**Location:** `tests/performance/`

### Running Performance Tests

```bash
# Run all performance tests
pytest tests/performance/ -v

# Run only performance baseline tests (timing)
pytest tests/performance/test_performance_baselines.py -v

# Run only memory baseline tests
pytest tests/performance/test_memory_baselines.py -v

# Exclude slow tests (> 30s)
pytest tests/performance/ -m "not slow" -v

# Run only slow tests (bootstrap 10K, full pipeline)
pytest tests/performance/ -m slow -v
```

### Performance Baselines

Current timing thresholds for operations:

| Operation | Max Time | Dataset Size | Notes |
|-----------|----------|--------------|-------|
| Feature engineering | 2s | Medium (100×50) | Data transformation |
| Feature selection | 10s | Medium (100×50) | Correlation analysis |
| Bootstrap 100 | 5s | Medium | Fast iteration |
| Bootstrap 1000 | 30s | Medium | Integration testing |
| Bootstrap 10000 | 300s (5 min) | Full production | Production config |
| Full pipeline | 360s (6 min) | Full production | All 10 stages |
| Fixture loading | 1s | Full production | I/O performance |

### Memory Baselines

Current memory consumption thresholds:

| Operation | Max Memory | Dataset Size | Notes |
|-----------|-----------|--------------|-------|
| Medium operations | 500 MB | Medium (100×50) | General operations |
| Full operations | 2 GB | Full (167×598) | Production data |
| Feature engineering | 1 GB | Medium | Feature creation |
| Feature selection | 2 GB | Medium | Correlation matrices |
| Bootstrap 10000 | 8 GB | Full production | Production inference |
| Full pipeline | 4 GB | Full production | All stages |

### Writing Performance Tests

#### Timing Baseline Test

```python
import time

def test_operation_performance(medium_dataset, performance_baselines):
    """Test that operation completes within baseline threshold."""

    max_time = performance_baselines['operation_name']['max_seconds']

    # Measure execution time
    start = time.time()
    result = perform_operation(medium_dataset)
    elapsed = time.time() - start

    # Validate result
    assert result is not None

    # Check performance baseline
    assert elapsed < max_time, (
        f"Operation took {elapsed:.2f}s (max: {max_time}s). "
        f"Performance regression detected."
    )
```

#### Memory Baseline Test

```python
import os
import gc
import psutil

def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_operation_memory(medium_dataset, memory_baselines):
    """Test that operation uses acceptable memory."""

    max_mb = memory_baselines['operation_name']['max_mb']

    # Clean up before test
    gc.collect()

    # Measure memory before
    mem_before = get_memory_usage_mb()

    # Run operation
    result = perform_operation(medium_dataset)

    # Measure memory after
    mem_after = get_memory_usage_mb()
    mem_delta = mem_after - mem_before

    # Validate memory usage
    assert mem_delta < max_mb, (
        f"Operation used {mem_delta:.1f} MB (max: {max_mb} MB). "
        f"Memory regression detected."
    )
```

### Memory Leak Detection

Tests that run operations multiple times to detect memory leaks:

```python
def test_repeated_operations_no_leak(tiny_dataset):
    """Verify repeated operations don't leak memory."""

    mem_start = get_memory_usage_mb()

    # Run operation 100 times
    for i in range(100):
        result = tiny_dataset.copy()
        result['new_col'] = result.iloc[:, 0] * 2

    # Force garbage collection
    gc.collect()

    mem_end = get_memory_usage_mb()
    mem_growth = mem_end - mem_start

    # Memory growth should be bounded (< 100 MB)
    max_growth = 100
    assert mem_growth < max_growth, (
        f"Repeated operations grew memory by {mem_growth:.1f} MB. "
        f"Potential memory leak detected."
    )
```

### Interpreting Performance Test Failures

#### Timing Baseline Failure

```
AssertionError: Feature engineering took 3.45s (max: 2.0s).
Performance regression detected.
```

**Possible causes:**
- New code added in hot path
- Inefficient algorithm introduced
- External system slowdown (disk I/O, network)
- Increased data size

**Actions:**
1. Profile the code to find bottleneck: `python -m cProfile script.py`
2. Check if data size changed
3. Review recent code changes
4. If intentional (e.g., new feature), update baseline in `tests/performance/baselines.json`

#### Memory Baseline Failure

```
AssertionError: Bootstrap (10000 samples) used 9500.0 MB (max: 8000 MB).
Memory regression detected.
```

**Possible causes:**
- Memory leak (objects not garbage collected)
- Unnecessary data duplication
- Inefficient data structures
- Missing cleanup code

**Actions:**
1. Use memory profiler: `pip install memory-profiler`
2. Check for unnecessary `.copy()` operations
3. Verify garbage collection happens
4. Look for growing lists/dicts in loops
5. Use memory leak detection tests

### Updating Baselines

When code improvements are intentional and validated:

1. **Verify improvement is real** - Run tests multiple times to confirm
2. **Document reason** - Add comment in baselines.json explaining change
3. **Update baseline** - Modify threshold in `tests/performance/baselines.json`

```json
{
  "performance_baselines": {
    "feature_engineering": {
      "max_seconds": 1.5,  // Reduced from 2.0 due to vectorization optimization
      "description": "Feature engineering on medium dataset (100 rows × 50 features)",
      "updated": "2026-01-29",
      "reason": "Vectorized operations replace loops (PR #123)"
    }
  }
}
```

### Dependencies

Performance tests require:
- `psutil` for memory measurements: `pip install psutil`
- pytest markers for slow tests

### Best Practices

[DONE] **DO:**
- Use appropriate fixture size (SMALL for unit, MEDIUM for integration)
- Mark slow tests (> 30s) with `@pytest.mark.slow`
- Clean up memory before tests (`gc.collect()`)
- Provide clear failure messages with actual vs threshold
- Update baselines when code improvements are validated

[ERROR] **DON'T:**
- Use full production dataset for simple performance tests
- Update baselines to hide regressions
- Skip performance tests in CI (except slow ones)
- Ignore memory leak warnings
- Test external system performance (AWS, network)

### Example: Complete Performance Test

```python
import time
import pytest


class TestFeatureEngineeringPerformance:
    """Performance tests for feature engineering module."""

    def test_feature_engineering_timing(self, medium_dataset):
        """Feature engineering should complete in < 2 seconds."""

        start = time.time()
        result = engineer_features(medium_dataset)
        elapsed = time.time() - start

        assert result is not None
        assert elapsed < 2.0, (
            f"Feature engineering took {elapsed:.2f}s (max: 2.0s)"
        )

        print(f"[PASS] Feature engineering: {elapsed:.2f}s")

    def test_feature_engineering_memory(self, medium_dataset):
        """Feature engineering should use < 1 GB memory."""

        mem_before = get_memory_usage_mb()
        result = engineer_features(medium_dataset)
        mem_after = get_memory_usage_mb()

        mem_delta = mem_after - mem_before

        assert mem_delta < 1000, (
            f"Feature engineering used {mem_delta:.1f} MB (max: 1000 MB)"
        )

        print(f"[PASS] Feature engineering memory: {mem_delta:.1f} MB")
```

### Performance Testing Checklist

Before merging code:
- [ ] All performance tests pass
- [ ] No unexpected slowdowns (check test output)
- [ ] Memory usage within baselines
- [ ] No memory leaks detected
- [ ] Slow tests marked with `@pytest.mark.slow`
- [ ] If baselines updated, reason documented

---

## Property-Based Testing

### Introduction

Property-based testing generates random test cases to find edge cases you didn't think of.

### Using Hypothesis

```python
from hypothesis import given, strategies as st
import numpy as np

@given(
    rates=st.lists(st.floats(min_value=0, max_value=10), min_size=3, max_size=20),
    weights=st.lists(st.floats(min_value=0, max_value=1), min_size=3, max_size=20)
)
def test_weighted_mean_properties(rates, weights):
    """Property test: weighted mean should be between min and max rate."""
    # Ensure same length
    n = min(len(rates), len(weights))
    rates = rates[:n]
    weights = weights[:n]

    # Calculate weighted mean
    if sum(weights) > 0:
        weighted_mean = sum(r*w for r, w in zip(rates, weights)) / sum(weights)

        # Property: result should be between min and max
        assert min(rates) <= weighted_mean <= max(rates)
```

### Common Properties to Test

1. **Idempotence**: `f(f(x)) == f(x)`
   ```python
   def test_normalize_idempotent(data):
       normalized_once = normalize(data)
       normalized_twice = normalize(normalized_once)
       assert np.allclose(normalized_once, normalized_twice)
   ```

2. **Commutativity**: `f(x, y) == f(y, x)`
   ```python
   def test_weighted_mean_commutative(a, b, w1, w2):
       result1 = weighted_mean([a, b], [w1, w2])
       result2 = weighted_mean([b, a], [w2, w1])
       assert np.isclose(result1, result2)
   ```

3. **Bounds**: Output within expected range
   ```python
   def test_probability_in_range(data):
       probs = calculate_probabilities(data)
       assert (probs >= 0).all()
       assert (probs <= 1).all()
       assert np.isclose(probs.sum(), 1.0)
   ```

---

## Common Test Patterns

### Testing Calculations

```python
def test_weighted_mean_calculation():
    """Test weighted mean with known values."""
    rates = [4.0, 6.0]
    weights = [0.25, 0.75]

    result = calculate_weighted_mean(rates, weights)

    # 4*0.25 + 6*0.75 = 1 + 4.5 = 5.5
    expected = 5.5
    assert np.isclose(result, expected, atol=0.01)
```

### Testing NaN Handling

```python
def test_function_handles_nan():
    """Test NaN values are handled correctly."""
    df = pd.DataFrame({
        'value': [1.0, np.nan, 3.0],
        'other': [4.0, 5.0, 6.0]
    })

    result = process_data(df)

    # Option 1: NaN filled with specific value
    assert not result['value'].isna().any()
    assert result['value'].iloc[1] == 0.0

    # Option 2: NaN propagates
    assert result['value'].isna().iloc[1]

    # Option 3: Row removed
    assert len(result) == 2
```

### Testing Error Conditions

```python
def test_function_validates_input():
    """Test that invalid input raises appropriate error."""
    invalid_df = pd.DataFrame()  # Empty

    with pytest.raises(ValueError) as exc_info:
        process_data(invalid_df)

    # Check error message contains useful info
    assert "empty" in str(exc_info.value).lower()


def test_function_raises_specific_exception():
    """Test that specific exception type is raised."""
    from src.core.exceptions import DataValidationError

    with pytest.raises(DataValidationError):
        validate_data(invalid_data)
```

### Testing Immutability

```python
def test_function_preserves_original():
    """Test that function doesn't modify input."""
    original = pd.DataFrame({'value': [1, 2, 3]})
    original_copy = original.copy()

    _ = transform_data(original)

    # Original unchanged
    pd.testing.assert_frame_equal(original, original_copy)


def test_function_returns_new_dataframe():
    """Test that function returns a new DataFrame, not a view."""
    original = pd.DataFrame({'value': [1, 2, 3]})
    result = transform_data(original)

    # Modify result
    result['value'].iloc[0] = 999

    # Original should be unaffected
    assert original['value'].iloc[0] == 1
```

### Testing DataFrames

```python
def test_dataframe_structure():
    """Test output DataFrame has expected structure."""
    result = create_features(input_df)

    # Check columns exist
    assert 'feature_1' in result.columns
    assert 'feature_2' in result.columns

    # Check types
    assert result['feature_1'].dtype == np.float64
    assert result['feature_2'].dtype == np.int64

    # Check shape
    assert result.shape == (100, 10)

    # Check no NaN (if expected)
    assert not result.isna().any().any()


def test_dataframe_values_approx():
    """Test DataFrame values are approximately equal."""
    expected = pd.DataFrame({'value': [1.0, 2.0, 3.0]})
    result = calculate_something()

    # Use pandas testing utilities
    pd.testing.assert_frame_equal(result, expected, atol=0.01)

    # Or manual check
    assert np.allclose(result['value'].values, expected['value'].values)
```

### Testing with Tolerances

```python
from src.validation_support.mathematical_equivalence import TOLERANCE

def test_calculation_within_tolerance():
    """Test result matches expected within TOLERANCE (1e-12)."""
    result = precise_calculation(input_data)
    expected = 4.123456789012

    # Use project's TOLERANCE constant
    assert abs(result - expected) < TOLERANCE


def test_floating_point_comparison():
    """Test floating point values with appropriate tolerance."""
    result = [0.1 + 0.2]  # Floating point quirk: not exactly 0.3
    expected = [0.3]

    # Use numpy's isclose
    assert np.isclose(result, expected, atol=1e-10).all()

    # Or pytest's approx
    assert result == pytest.approx(expected, abs=1e-10)
```

---

## AWS Integration Testing

### Overview

AWS integration tests validate that fixture-based development produces results
identical to AWS execution. These tests ensure offline development maintains
mathematical equivalence (1e-12 precision) with production AWS data.

**Purpose:**
- Validate fixture-AWS equivalence
- Detect fixture staleness
- Verify production data access
- Ensure offline development accuracy

**Location:** `tests/integration/test_aws_fixture_equivalence.py`

### When to Run AWS Tests

AWS tests require AWS credentials and are **skipped by default** in offline mode:

```bash
# Default: Skip AWS tests (offline development)
pytest tests/integration/ -v

# Explicit skip
pytest tests/integration/ -m "not aws" -v
```

**Run AWS tests when:**
- Before production deployment
- After fixture refresh (quarterly)
- Validating fixture-AWS equivalence
- Testing new AWS data sources
- Debugging data loading issues

### Running AWS Tests

#### Prerequisites

Set required environment variables:

```bash
export STS_ENDPOINT_URL="https://sts.us-east-1.amazonaws.com"
export ROLE_ARN="arn:aws:iam::123456789:role/MyRole"
export XID="user123"
export BUCKET_NAME="my-bucket"
```

#### Run AWS Equivalence Tests

```bash
# Run all AWS tests (requires credentials)
pytest tests/integration/test_aws_fixture_equivalence.py -m aws -v

# Run only connection tests (fast sanity check)
pytest tests/integration/test_aws_fixture_equivalence.py::TestAWSConnection -m aws -v

# Run full equivalence validation (slow)
pytest tests/integration/test_aws_fixture_equivalence.py::TestAWSFixtureEquivalence -m aws -v
```

### AWS Test Categories

#### Connection Tests (Fast)

Verify AWS credentials and S3 access:

```python
@pytest.mark.aws
def test_aws_s3_connection(aws_config):
    """Verify AWS S3 connection can be established."""
    adapter = S3Adapter(aws_config)
    adapter._ensure_connection()
```

#### Data Loading Tests

Validate raw data equivalence:

```python
@pytest.mark.aws
def test_data_loading_sales_equivalence(aws_adapter, fixture_adapter):
    """Verify sales data from AWS matches fixture data."""
    aws_sales = aws_adapter.load_sales_data(product_filter="FlexGuard 6Y20B")
    fixture_sales = fixture_adapter.load_sales_data(product_filter="FlexGuard 6Y20B")

    validate_dataframe_equivalence(
        actual=fixture_sales,
        expected=aws_sales,
        tolerance=1e-12,
        stage="sales_data_loading"
    )
```

#### Pipeline Tests (Slow)

Validate pipeline transformations:

```python
@pytest.mark.aws
@pytest.mark.slow
def test_pipeline_full_equivalence(aws_adapter, fixture_adapter):
    """Verify full pipeline produces identical results."""
    aws_pipeline = DataPipeline(adapter=aws_adapter)
    aws_result = aws_pipeline.run_full_pipeline()

    fixture_pipeline = DataPipeline(adapter=fixture_adapter)
    fixture_result = fixture_pipeline.run_full_pipeline()

    validate_dataframe_equivalence(
        actual=fixture_result,
        expected=aws_result,
        tolerance=1e-12,
        stage="pipeline_full_execution"
    )
```

#### Inference Tests (Slow)

Ultimate validation - inference equivalence:

```python
@pytest.mark.aws
@pytest.mark.slow
def test_inference_equivalence(aws_adapter, fixture_adapter):
    """Verify inference results match between AWS and fixture."""
    # Run with AWS data
    aws_interface = create_interface("6Y20B", environment="aws",
                                     adapter_kwargs={'adapter': aws_adapter})
    aws_inference = aws_interface.run_inference(aws_interface.load_data())

    # Run with fixture data
    fixture_interface = create_interface("6Y20B", environment="fixture")
    fixture_inference = fixture_interface.run_inference(fixture_interface.load_data())

    # Validate coefficients at 1e-12 precision
    np.testing.assert_allclose(
        fixture_inference['coefficients'],
        aws_inference['coefficients'],
        rtol=1e-12,
        atol=1e-12
    )
```

### Fixture Refresh Workflow

#### 1. Run Fixture Refresh Script

```bash
# Set AWS credentials
export STS_ENDPOINT_URL="..."
export ROLE_ARN="..."
export XID="..."
export BUCKET_NAME="..."

# Run refresh
python tests/fixtures/refresh_fixtures.py
```

#### 2. Validate Refreshed Fixtures

```bash
# Check fixture validity
pytest tests/fixtures/test_fixture_validity.py -v

# Should see:
# [PASS] Fixtures are X days old (< 90 days)
# [PASS] All required fixture files exist
# [PASS] Sales data quality: (2817439, 11)
```

#### 3. Validate AWS Equivalence

```bash
# Run AWS equivalence tests
pytest tests/integration/test_aws_fixture_equivalence.py -m aws -v

# Should see all tests pass at 1e-12 precision
```

#### 4. Commit Refreshed Fixtures

```bash
git add tests/fixtures/rila/
git commit -m "Refresh fixtures from AWS (2026-01-29)"
```

### Fixture Validation Tests

Location: `tests/fixtures/test_fixture_validity.py`

These tests run in CI to detect stale or corrupted fixtures:

```bash
# Run all fixture validation tests
pytest tests/fixtures/test_fixture_validity.py -v
```

**What's validated:**
- **Freshness**: Fixtures < 90 days old
- **Completeness**: All required files exist
- **Data Quality**: Schema, size, missing values
- **Metadata**: refresh_metadata.json valid and matches data

**Example output:**
```
[PASS] Fixtures are 3 days old (< 90 days)
[PASS] All 8 required fixture files exist
[PASS] Sales data quality: (2817439, 11), 0.000% missing premiums
[PASS] Final weekly dataset quality: (203, 598), 0.234% NaN
```

### Troubleshooting AWS Tests

#### Issue: Missing AWS Credentials

```
ValueError: Missing required AWS environment variables: ['ROLE_ARN']
```

**Solution:** Set all required environment variables:
```bash
export ROLE_ARN="arn:aws:iam::123456789:role/MyRole"
```

#### Issue: S3 Access Denied

```
boto3.exceptions.S3AccessDenied: Access Denied
```

**Solution:**
- Verify IAM role has S3 read permissions
- Check bucket name is correct
- Verify STS endpoint URL is correct

#### Issue: Equivalence Test Fails

```
AssertionError: Stage pipeline_full_execution: Shape mismatch.
Fixture: (167, 598), AWS: (203, 598)
```

**Solution:**
- Fixtures are stale - refresh with: `python tests/fixtures/refresh_fixtures.py`
- AWS data has been updated since last fixture refresh
- Check if fixture refresh completed successfully

#### Issue: Tests Take Too Long

**Solution:** Run only fast connection tests first:
```bash
pytest tests/integration/test_aws_fixture_equivalence.py::TestAWSConnection -m aws -v
```

If connection tests pass, run full equivalence tests:
```bash
pytest tests/integration/test_aws_fixture_equivalence.py::TestAWSFixtureEquivalence -m aws -v
```

### AWS Integration Best Practices

[DONE] **DO:**
- Run AWS tests before production deployment
- Refresh fixtures quarterly (every 90 days)
- Validate fixture equivalence after refresh
- Keep refresh_metadata.json up to date
- Skip AWS tests in offline development (`-m "not aws"`)

[ERROR] **DON'T:**
- Run AWS tests in fast CI loops (expensive)
- Commit AWS credentials to repository
- Update fixtures without running equivalence tests
- Ignore fixture freshness warnings (> 60 days)
- Force push fixtures without validation

### Fixture Refresh Schedule

**Recommended Schedule:**
- Quarterly (every 90 days) for routine updates
- After major data source changes
- Before production deployment validation
- When fixture validation tests fail (staleness)

**Fixture Age Thresholds:**
- < 60 days: [DONE] Fresh
- 60-90 days: [WARN] Consider refreshing
- > 90 days: [ERROR] Stale - must refresh

### CI Integration

**Fast CI (default):**
```yaml
# Skip AWS tests in fast CI
pytest tests/ -m "not aws" -v
```

**Pre-deployment CI:**
```yaml
# Run AWS tests before deployment
- name: Validate AWS Equivalence
  if: github.ref == 'refs/heads/main'
  env:
    STS_ENDPOINT_URL: ${{ secrets.STS_ENDPOINT_URL }}
    ROLE_ARN: ${{ secrets.ROLE_ARN }}
    XID: ${{ secrets.XID }}
    BUCKET_NAME: ${{ secrets.BUCKET_NAME }}
  run: pytest tests/integration/test_aws_fixture_equivalence.py -m aws -v
```

---

## Coverage Goals

### Target Coverage by Module

| Module | Target | Current | Status |
|--------|--------|---------|--------|
| features/competitive_features.py | 85% | 100% | [DONE] |
| features/engineering_timeseries.py | 85% | 100% | [DONE] |
| features/engineering_integration.py | 85% | 99% | [DONE] |
| features/engineering_temporal.py | 85% | 98% | [DONE] |
| features/aggregation/ | 85% | 95% | [DONE] |
| validation_support/ | 90% | 57% | [WARN] In progress |
| models/inference.py | 80% | 44% |  Planned |
| products/rila_methodology.py | 80% | 25% |  Planned |
| **Overall** | **80%** | **~50%** |  In progress |

### What to Prioritize

1. **Critical business logic**: Feature engineering, model inference
2. **High-risk areas**: Validation, data loading
3. **Complex algorithms**: Bootstrap, lag features
4. **Public API**: Functions users directly call

### What to Skip

- Trivial getters/setters
- Pass-through functions
- External library wrappers
- Deprecated code paths

---

## Troubleshooting

### Common Issues

#### Issue: Tests pass locally but fail in CI

**Cause**: Environment differences, random seeds, file paths

**Solution**:
```python
# Use absolute paths from project root
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
fixture_path = project_root / "tests/fixtures/data.parquet"

# Set random seeds
np.random.seed(42)
random.seed(42)

# Use temp directories, not hardcoded paths
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    output_path = Path(tmpdir) / "output.csv"
```

#### Issue: Floating point comparison failures

**Problem**:
```python
assert result == 0.3  # Fails! result = 0.30000000000000004
```

**Solution**:
```python
assert np.isclose(result, 0.3, atol=1e-10)
# or
assert result == pytest.approx(0.3, abs=1e-10)
```

#### Issue: Tests are slow

**Causes**:
- Loading large fixtures repeatedly
- Not using session-scoped fixtures
- Testing integration instead of units

**Solutions**:
```python
# Use session scope for expensive fixtures
@pytest.fixture(scope="session")
def large_dataset():
    return pd.read_parquet("large_file.parquet")

# Mock expensive operations
from unittest.mock import patch

@patch('src.data.loader.load_from_s3')
def test_with_mock(mock_load):
    mock_load.return_value = small_test_df
    # Test runs fast without S3 call

# Run tests in parallel
pytest -n auto  # Uses all CPU cores
```

#### Issue: Fixture not found

**Error**: `fixture 'my_fixture' not found`

**Causes**:
- Fixture in wrong conftest.py
- Typo in fixture name
- Missing import

**Solution**:
```python
# Fixtures must be in conftest.py or same file
# Check fixture name matches exactly
def test_example(my_fixture):  # Must match @pytest.fixture name
    pass

# List available fixtures
pytest --fixtures
```

#### Issue: Tests modify global state

**Problem**: Test A passes alone, fails when run with Test B

**Cause**: Shared mutable state

**Solution**:
```python
# Bad: Module-level mutable default
CACHE = {}

def get_data(key):
    if key not in CACHE:
        CACHE[key] = expensive_load(key)
    return CACHE[key]

# Good: Fixture provides fresh state
@pytest.fixture
def cache():
    return {}

def test_with_cache(cache):
    result = get_data_cached(key, cache)
```

---

## Best Practices

### Do's

[DONE] **Write tests first for new code** (TDD)
```python
# 1. Write failing test
def test_new_feature():
    result = new_feature(input_data)
    assert result == expected

# 2. Implement feature until test passes
# 3. Refactor
```

[DONE] **Test one thing per test**
```python
# Good
def test_weighted_mean_basic():
    result = weighted_mean([4, 6], [0.5, 0.5])
    assert result == 5.0

def test_weighted_mean_handles_zero_weights():
    result = weighted_mean([4, 6], [0, 0])
    assert result == 5.0  # Falls back to equal weights

# Bad
def test_weighted_mean():  # Tests too many things
    result1 = weighted_mean([4, 6], [0.5, 0.5])
    assert result1 == 5.0
    result2 = weighted_mean([4, 6], [0, 0])
    assert result2 == 5.0
    result3 = weighted_mean([np.nan, 6], [0.5, 0.5])
    assert not np.isnan(result3)
```

[DONE] **Use descriptive test names**
```python
# Good
def test_median_ranking_returns_middle_value()
def test_lag_features_handles_insufficient_history()

# Bad
def test_1()
def test_func()
```

[DONE] **Test edge cases explicitly**
```python
def test_division_by_zero_handled():
    result = safe_divide(1, 0)
    assert result == 0  # or np.inf, depending on spec

def test_empty_list_handled():
    result = calculate_mean([])
    assert np.isnan(result)
```

### Don'ts

[ERROR] **Don't test implementation details**
```python
# Bad: Testing private method
def test_internal_helper():
    obj = MyClass()
    result = obj._internal_method()  # May change

# Good: Test public API
def test_public_method():
    obj = MyClass()
    result = obj.public_method()  # Stable interface
```

[ERROR] **Don't use random data without seeds**
```python
# Bad
def test_with_random():
    data = np.random.rand(100)  # Non-reproducible
    result = process(data)
    assert result > 0

# Good
def test_with_random():
    np.random.seed(42)
    data = np.random.rand(100)  # Reproducible
    result = process(data)
    assert np.isclose(result, 0.512, atol=0.01)
```

[ERROR] **Don't test external libraries**
```python
# Bad: Testing pandas
def test_pandas_merge():
    df1 = pd.DataFrame({'a': [1]})
    df2 = pd.DataFrame({'a': [1], 'b': [2]})
    result = df1.merge(df2)
    assert len(result) == 1  # Pandas already tested this

# Good: Test your logic
def test_merge_with_validation():
    result = safe_merge_with_checks(df1, df2)
    assert 'validation_flag' in result.columns
```

---

## Quick Reference

### Assertions

```python
# Equality
assert x == y
assert result.equals(expected)  # Pandas
pd.testing.assert_frame_equal(df1, df2)
np.testing.assert_array_equal(arr1, arr2)

# Approximate equality
assert np.isclose(x, y, atol=1e-10)
assert x == pytest.approx(y, abs=1e-10)

# Type checks
assert isinstance(result, pd.DataFrame)
assert result.dtype == np.float64

# Membership
assert 'column' in df.columns
assert value in list_of_values

# Boolean
assert condition is True
assert not condition

# Exceptions
with pytest.raises(ValueError):
    function_that_should_fail()
```

### Useful pytest Flags

```bash
-v              # Verbose
-s              # Show print statements
-x              # Stop on first failure
--lf            # Run last failed tests
--tb=short      # Short traceback
-k "pattern"    # Run tests matching pattern
-n auto         # Parallel execution
--cov           # Coverage report
--pdb           # Drop into debugger on failure
```

---

## Examples from the Codebase

### Example 1: Testing Calculations

From `tests/unit/features/test_aggregation_strategies.py`:

```python
def test_weighted_aggregation_basic(simple_rates_data, company_columns, weights_long_format):
    """Test basic weighted aggregation calculation."""
    strategy = WeightedAggregation()
    result = strategy.aggregate(simple_rates_data, company_columns, weights_long_format)

    # First row: [4.5, 4.2, 4.0, 3.8, 3.5] with weights [0.3, 0.25, 0.2, 0.15, 0.1]
    # Weighted mean = 4.5*0.3 + 4.2*0.25 + 4.0*0.2 + 3.8*0.15 + 3.5*0.1 = 4.12
    expected = 4.12
    assert np.isclose(result.iloc[0], expected, atol=0.01)
```

### Example 2: Testing Edge Cases

From `tests/unit/features/test_engineering_temporal.py`:

```python
def test_create_lag_features_missing_column_skipped(lag_features_data):
    """Test that missing source columns are gracefully skipped."""
    lag_configs = [
        {'source_col': 'nonexistent', 'prefix': 'ne', 'lag_direction': 'backward'}
    ]

    result = create_lag_features_for_columns(lag_features_data, lag_configs, max_lag_periods=2)

    # Should not create any lag columns for missing source
    assert 'ne_current' not in result.columns
    assert 'ne_t1' not in result.columns
```

### Example 3: Testing Validation

From `tests/unit/validation_support/test_mathematical_equivalence.py`:

```python
def test_tolerance_constant_value():
    """Test TOLERANCE constant has correct value."""
    assert TOLERANCE == 1e-12, "CRITICAL: TOLERANCE must be exactly 1e-12"

def test_tolerance_enforcement_precision():
    """Test tolerance is enforced at exactly 1e-12 precision."""
    df1 = pd.DataFrame({'A': [1.0]})
    df2_pass = pd.DataFrame({'A': [1.0 + 1e-13]})  # Within tolerance
    df2_fail = pd.DataFrame({'A': [1.0 + 1e-11]})  # Exceeds tolerance

    validator = DataFrameEquivalenceValidator()

    # Should pass: difference < 1e-12
    result_pass = validator.validate_equivalence(df1, df2_pass)
    assert result_pass.is_equivalent

    # Should fail: difference > 1e-12
    result_fail = validator.validate_equivalence(df1, df2_fail)
    assert not result_fail.is_equivalent
```

---

**Last Updated**: 2026-01-29
**Version**: 1.0
**Maintainer**: Data Science Team

# Offline Development Guide

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Fixture System](#fixture-system)
- [Mathematical Equivalence](#mathematical-equivalence)
- [Testing Offline](#testing-offline)
- [AWS Integration](#aws-integration)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)
- [References](#references)

---

## Overview

### What is Offline Development?

Offline development allows you to build, test, and validate the entire RILA Price Elasticity pipeline **without AWS credentials or network connectivity**. All data sources, pipeline stages, and validation baselines are available as local fixtures.

### Why Use Fixtures vs AWS?

**Fixtures Provide**:
- ⚡ **Speed**: 10-100x faster data loading (no S3 latency)
- 🔒 **No Credentials**: Work without AWS access or VPN
- 🔄 **Reproducibility**: Identical results across machines
- 🌐 **Offline Work**: Develop on planes, trains, anywhere
- 💰 **Cost Savings**: No S3 read costs during development

**AWS Required For**:
- Pre-production deployment validation
- Quarterly fixture refresh
- New data source integration
- Production data verification

### Benefits for Team Development

1. **New Team Members**: Start coding day one without AWS onboarding
2. **Fast Iteration**: Test changes in seconds, not minutes
3. **Deterministic Tests**: No flaky tests from network issues
4. **Parallel Development**: Multiple developers without S3 contention
5. **CI/CD Efficiency**: Fast test suite without AWS dependencies

### When You Still Need AWS

- **Pre-Deployment**: Validate against live data before production release
- **Quarterly Refresh**: Update fixtures with latest production data
- **Data Source Changes**: Validate new data sources or schema changes
- **Production Debugging**: Investigate issues in production environment

---

## Quick Start

### Installation

Dependencies are already included in the project:

```bash
# Clone repository
git clone <repository-url>
cd RILA_6Y20B_refactored

# Install dependencies (includes pytest, pandas, hypothesis, etc.)
pip install -r requirements.txt

# Verify fixtures exist
ls -lh tests/fixtures/rila/
# Should show ~73 MB of parquet files
```

No AWS credentials needed for offline development!

### Basic Usage

#### Using Fixtures in Notebooks

```python
from src.notebooks import create_interface

# Offline mode (default) - uses fixtures
interface = create_interface(
    product_code="6Y20B",
    environment="fixture"  # or omit - fixture is default
)

# Load data (from fixtures)
df = interface.load_data()
print(f"Loaded {len(df)} weeks of data")

# Run inference (offline)
results = interface.run_inference(df)
print(f"R² Score: {results['metrics']['R²']:.4f}")
```

#### Using Fixtures in Scripts

```python
from src.adapters.fixture_adapter import FixtureAdapter
from src.pipeline.data_pipeline import DataPipeline

# Initialize fixture adapter
adapter = FixtureAdapter()

# Load raw data
sales = adapter.load_sales_data(product_filter="FlexGuard 6Y20B")
rates = adapter.load_competitive_rates(start_date="2020-01-01")

# Run pipeline
pipeline = DataPipeline(adapter=adapter)
modeling_dataset = pipeline.run_full_pipeline()

print(f"Final dataset: {modeling_dataset.shape}")
# Output: Final dataset: (203, 598)
```

### Running Tests

```bash
# All tests (offline by default)
pytest
# Runs 2,500+ tests in ~10 minutes

# Specific test types
pytest tests/unit/                    # Unit tests (~1,200 tests, < 2 min)
pytest tests/integration/             # Integration tests (~800 tests, < 5 min)
pytest tests/e2e/                     # End-to-end tests (~200 tests, < 3 min)
pytest tests/performance/             # Performance tests (~50 tests, < 5 min)

# Skip slow tests (> 30s)
pytest -m "not slow"
# Runs fast test suite in ~5 minutes

# Skip AWS tests (default for offline)
pytest -m "not aws"
# Explicitly skip AWS tests (same as default)

# With coverage report
pytest --cov=src --cov-report=html
# Opens htmlcov/index.html with coverage report
```

### Your First Offline Development Session

```python
# 1. Load fixture data
from src.notebooks import create_interface
interface = create_interface("6Y20B", environment="fixture")
df = interface.load_data()

# 2. Make changes to code
# Edit src/models/inference.py or other files

# 3. Test changes immediately
pytest tests/unit/models/test_inference_module.py -v

# 4. Run integration test
pytest tests/integration/test_pipeline_integration.py -v

# 5. Validate E2E
pytest tests/e2e/test_full_pipeline_offline.py -v

# 6. All tests pass - commit!
git add .
git commit -m "Add feature X with offline testing"
```

---

## Fixture System

### Available Fixtures

The project includes **73 MB of captured production data** in `tests/fixtures/rila/`:

#### Raw Data Fixtures

1. **Sales Data** (`raw_sales_data.parquet`)
   - Size: 11 MB
   - Rows: 2.8M transactions
   - Columns: 11 (application_signed_date, product_name, premium, etc.)
   - Date Range: 2015-01-01 to 2024-12-31
   - Products: All RILA products (6Y20B, 6Y10B, 10Y20B, etc.)

2. **WINK Competitive Rates** (`raw_wink_data.parquet`)
   - Size: 12 MB
   - Rows: 1.1M rate observations
   - Columns: 19 (date, company, product, cap_rate, etc.)
   - Date Range: 2015-01-01 to 2024-12-31
   - Companies: 50+ competitors

3. **Market Share Weights** (`market_share_weights.parquet`)
   - Size: 8 KB
   - Rows: 19 companies
   - Columns: 11 (company_name, market_share, weight_2023, etc.)
   - Updated: Quarterly

4. **Economic Indicators** (`economic_indicators/`)
   - CPI (`cpi.parquet`): 120 months, 3 columns
   - DGS5 (`dgs5.parquet`): 2,500 days, 2 columns
   - VIX (`vix.parquet`): 6,000 days, 2 columns
   - Treasury Rates: Multiple maturities (1Y, 2Y, 5Y, 10Y, 30Y)

#### Processed Data Fixtures

5. **Pipeline Stage Outputs** (`stage_01_output.parquet` through `stage_10_output.parquet`)
   - 10 intermediate stage outputs
   - Captured from full pipeline run
   - Used for stage-by-stage validation

6. **Final Modeling Dataset** (`final_weekly_dataset.parquet`)
   - Size: 1.1 MB
   - Rows: 203 weeks
   - Columns: 598 features
   - Ready for model inference

#### Baseline Fixtures

7. **Inference Baselines** (`baselines/rila/reference/`)
   - 110 parquet files
   - Coefficients, predictions, metrics
   - Used for mathematical equivalence testing

### Fixture Hierarchy

The fixture system uses a **three-tier hierarchy** optimized for test performance:

#### SMALL Fixtures (Unit Tests)

**Purpose**: Fast unit tests, TDD iteration

**Characteristics**:
- Size: 20-100 rows, 5-20 features
- Bootstrap: 10-100 samples
- Load time: < 0.1 seconds
- Scope: `function` (fresh per test)

**Usage**:
```python
def test_calculation_accuracy(tiny_dataset, small_bootstrap_config):
    """Fast unit test with minimal data."""
    result = calculate_aic(tiny_dataset, small_bootstrap_config)
    assert result > 0
```

**Available Fixtures**:
- `tiny_dataset`: 20 rows × 5 features
- `small_bootstrap_config`: n_bootstrap=10
- `small_inference_dataset`: 20 weeks × 10 features

#### MEDIUM Fixtures (Integration Tests)

**Purpose**: Module integration, pipeline stage testing

**Characteristics**:
- Size: 100-1,000 rows, 20-100 features
- Bootstrap: 100-1,000 samples
- Load time: 0.1-1 seconds
- Scope: `module` (cached per test file)

**Usage**:
```python
def test_feature_engineering_pipeline(medium_dataset, medium_bootstrap_config):
    """Integration test with realistic data."""
    features = engineer_features(medium_dataset)
    assert len(features) == len(medium_dataset)
    assert features.shape[1] > medium_dataset.shape[1]  # Features added
```

**Available Fixtures**:
- `medium_dataset`: 100 rows × 50 features
- `medium_bootstrap_config`: n_bootstrap=100
- `medium_inference_dataset`: 100 weeks × 50 features

#### LARGE Fixtures (E2E Tests)

**Purpose**: End-to-end validation, mathematical equivalence

**Characteristics**:
- Size: Full production dataset (200+ rows, 500+ features)
- Bootstrap: 10,000 samples (production setting)
- Load time: 1-5 seconds
- Scope: `session` (cached once per test run)

**Usage**:
```python
def test_full_pipeline_equivalence(full_production_dataset, production_bootstrap_config):
    """E2E test with full production data."""
    pipeline = DataPipeline()
    result = pipeline.run_full_pipeline(full_production_dataset)

    # Validate against baseline
    baseline = load_baseline_results()
    validate_equivalence(result, baseline, tolerance=1e-12)
```

**Available Fixtures**:
- `full_production_dataset`: 203 weeks × 598 features
- `production_bootstrap_config`: n_bootstrap=10000
- `final_weekly_dataset`: Complete production dataset

### Using Fixtures in Tests

#### Automatic Fixture Loading

Fixtures are defined in `tests/conftest.py` and automatically available:

```python
# No imports needed - fixtures auto-discovered
def test_example(final_weekly_dataset):
    """Fixture automatically loaded from conftest.py."""
    assert len(final_weekly_dataset) == 203
    assert final_weekly_dataset.shape[1] == 598
```

#### Choosing the Right Fixture Size

**Use SMALL fixtures when**:
- Testing pure calculation logic
- Testing validation functions (NaN, zeros, negatives)
- Testing edge cases
- Rapid TDD iteration

```python
def test_aic_calculation(tiny_dataset, small_bootstrap_config):
    """Fast unit test (< 0.1s)."""
    aic = calculate_aic(tiny_dataset, small_bootstrap_config)
    assert aic > 0
```

**Use MEDIUM fixtures when**:
- Testing module integration
- Testing pipeline stage output
- Testing feature engineering transformations
- Validating intermediate results

```python
def test_competitive_features(medium_dataset):
    """Integration test (< 1s)."""
    features = create_competitive_features(medium_dataset)
    assert 'competitor_avg_rate' in features.columns
    assert len(features) == len(medium_dataset)
```

**Use LARGE fixtures when**:
- Testing full end-to-end pipeline
- Validating mathematical equivalence with baseline
- Performance regression testing
- Production simulation

```python
def test_full_pipeline(full_production_dataset, production_bootstrap_config):
    """E2E test (< 60s)."""
    result = run_full_pipeline(full_production_dataset, production_bootstrap_config)
    assert result['metrics']['R²'] > 0.75
```

### Fixture Refresh Schedule

Fixtures are refreshed **quarterly** from AWS production data:

- **Q1 Refresh**: January (after year-end data)
- **Q2 Refresh**: April
- **Q3 Refresh**: July
- **Q4 Refresh**: October

Fixtures older than **90 days** trigger a warning in `test_fixture_validity.py`.

---

## Mathematical Equivalence

### How 1e-12 Precision is Maintained

The fixture system maintains **numerical precision equivalent to AWS execution**:

1. **Deterministic Capture**: Fixtures captured from AWS at specific timestamp
2. **Exact Reproduction**: All transformations use deterministic operations
3. **Stage Validation**: Each pipeline stage validated independently
4. **Baseline Comparison**: 110 baseline files ensure equivalence

### Precision Strategy

Different operations require different precision levels:

#### 1e-12 Precision (Strict)

Used for **critical mathematical operations**:
- Baseline comparisons
- AIC/BIC calculations
- Bootstrap coefficient distributions
- Feature selection scores

```python
import numpy as np

def test_baseline_equivalence(result, baseline):
    """Critical operations need 1e-12 precision."""
    np.testing.assert_allclose(
        result['coefficients'],
        baseline['coefficients'],
        rtol=1e-12,
        atol=1e-12,
        err_msg="Coefficient mismatch exceeds 1e-12 precision"
    )
```

#### 1e-6 Precision (Pragmatic)

Used for **general operations**:
- Pipeline transformations
- Intermediate calculations
- Feature engineering
- Data aggregations

```python
def test_transformation(result, expected):
    """General operations use 1e-6 precision."""
    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-6,
        err_msg="Transformation differs beyond 1e-6"
    )
```

#### pytest.approx (Flexible)

Used for **unit tests**:
- Simple calculations
- Non-critical validations
- Quick sanity checks

```python
def test_calculation(tiny_dataset):
    """Unit tests use pytest.approx with appropriate tolerance."""
    result = calculate_average(tiny_dataset)
    assert result == pytest.approx(expected, rel=1e-6)
```

### Validation Approach

#### 1. Stage-by-Stage Validation

Each of the 10 pipeline stages is validated independently:

```python
# tests/integration/test_pipeline_stage_equivalence.py

def test_stage_01_product_filtering(raw_sales_data, baseline_stage_01):
    """Stage 1 must match baseline at 1e-12 precision."""
    filtered = filter_product(raw_sales_data, product_name="FlexGuard 6Y20B")

    validate_dataframe_equivalence(
        actual=filtered,
        expected=baseline_stage_01,
        tolerance=1e-12,
        stage="01_product_filtering"
    )
```

If a test fails, you know **exactly which stage** introduced the divergence.

#### 2. Statistical Validation

Bootstrap models are stochastic, so we validate **statistical properties**:

```python
# tests/integration/test_bootstrap_statistical_equivalence.py

def test_bootstrap_coefficient_distribution(medium_dataset):
    """Bootstrap distributions should be stable across runs."""

    # Run 10 times with different seeds
    coefficients = []
    for seed in range(10):
        model = BootstrapInference({'random_state': seed})
        result = model.fit_predict(medium_dataset)
        coefficients.append(result['coefficients'])

    # Coefficient of variation should be < 5%
    coef_array = np.array(coefficients)
    cv = coef_array.std(axis=0) / np.abs(coef_array.mean(axis=0))
    assert np.all(cv < 0.05), f"High coefficient variation: {cv}"
```

#### 3. End-to-End Validation

Full pipeline results validated against AWS baseline:

```python
# tests/e2e/test_full_pipeline_offline.py

def test_offline_aws_equivalence(fixture_adapter, baseline_aws_results):
    """Fixture-based run must match AWS-captured baseline."""

    # Run with fixtures
    pipeline = DataPipeline(adapter=fixture_adapter)
    offline_result = pipeline.run_full_pipeline()

    # Compare to AWS baseline (1e-12 precision)
    validate_dataframe_equivalence(
        actual=offline_result,
        expected=baseline_aws_results,
        tolerance=1e-12,
        stage="full_pipeline_offline_vs_aws"
    )
```

### When Equivalence Matters

**High Importance** (1e-12 required):
- Pre-deployment validation
- Production model comparison
- Regulatory compliance
- Audit trails

**Medium Importance** (1e-6 acceptable):
- Development iteration
- Feature experimentation
- Integration testing
- Performance optimization

**Low Importance** (pytest.approx):
- Unit tests
- Edge case handling
- Error condition testing
- Documentation examples

---

## Testing Offline

### Test Organization

The test suite is organized by **test type and scope**:

```
tests/
├── unit/                      # Fast, isolated tests (~1,200 tests)
│   ├── validation_support/    # Leakage, constraints, quality
│   ├── models/                # Inference, bootstrap, forecasting
│   ├── products/              # Product methodologies
│   └── pipeline/              # Data transformations
│
├── integration/               # Module integration (~800 tests)
│   ├── test_pipeline_integration.py
│   ├── test_pipeline_stage_equivalence.py
│   ├── test_bootstrap_statistical_equivalence.py
│   └── test_aws_fixture_equivalence.py  # Marked @pytest.mark.aws
│
├── e2e/                       # Full pipeline tests (~200 tests)
│   ├── test_full_pipeline_offline.py
│   ├── test_multiproduct_pipeline.py
│   └── test_production_notebooks.py
│
├── performance/               # Timing & memory (~50 tests)
│   ├── test_performance_baselines.py
│   └── test_memory_baselines.py
│
├── property_based/            # Hypothesis tests (~300 tests)
│   ├── test_pipeline_properties.py
│   ├── test_numerical_stability.py
│   └── test_economic_constraints.py
│
├── fixtures/                  # Fixture validation
│   ├── test_fixture_validity.py
│   └── refresh_fixtures.py    # Fixture refresh script
│
└── conftest.py               # Global fixtures and configuration
```

### Running Tests by Type

#### Unit Tests

Fast, isolated tests for individual functions and classes:

```bash
# All unit tests
pytest tests/unit/ -v
# ~1,200 tests in < 2 minutes

# Specific module
pytest tests/unit/models/test_inference_module.py -v

# Specific test
pytest tests/unit/models/test_inference_module.py::test_inference_basic_run -v
```

#### Integration Tests

Module integration and pipeline stage tests:

```bash
# All integration tests (offline)
pytest tests/integration/ -m "not aws" -v
# ~800 tests in < 5 minutes

# Pipeline stage equivalence
pytest tests/integration/test_pipeline_stage_equivalence.py -v

# Bootstrap statistical validation
pytest tests/integration/test_bootstrap_statistical_equivalence.py -v
```

#### End-to-End Tests

Full pipeline and notebook execution tests:

```bash
# All E2E tests
pytest tests/e2e/ -v
# ~200 tests in < 3 minutes

# Full pipeline offline
pytest tests/e2e/test_full_pipeline_offline.py -v

# Multi-product pipeline
pytest tests/e2e/test_multiproduct_pipeline.py -v

# Notebook execution
pytest tests/e2e/test_production_notebooks.py -v
```

#### Performance Tests

Timing and memory baseline tests:

```bash
# All performance tests (skip slow)
pytest tests/performance/ -m "not slow" -v
# Fast tests in < 2 minutes

# Include slow tests (bootstrap 10K, full pipeline)
pytest tests/performance/ -v
# All tests in < 10 minutes

# Timing baselines only
pytest tests/performance/test_performance_baselines.py -v

# Memory baselines only
pytest tests/performance/test_memory_baselines.py -v
```

#### Property-Based Tests

Hypothesis-driven property tests:

```bash
# All property tests
pytest tests/property_based/ -v
# ~300 tests in < 3 minutes

# Pipeline properties
pytest tests/property_based/test_pipeline_properties.py -v

# Numerical stability
pytest tests/property_based/test_numerical_stability.py -v

# Economic constraints
pytest tests/property_based/test_economic_constraints.py -v
```

### Test Markers

Tests are marked for selective execution:

```bash
# Skip AWS tests (default for offline)
pytest -m "not aws"

# Skip slow tests (> 30s)
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Run only E2E tests
pytest -m e2e

# Run only performance tests
pytest -m performance

# Combine markers
pytest -m "integration and not aws and not slow"
```

**Available Markers**:
- `@pytest.mark.aws`: Requires AWS credentials (skipped by default)
- `@pytest.mark.slow`: Takes > 30 seconds
- `@pytest.mark.e2e`: End-to-end tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.performance`: Performance baseline tests

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html
# Opens htmlcov/index.html

# Generate terminal coverage report
pytest --cov=src --cov-report=term-missing

# Coverage for specific module
pytest tests/unit/models/ --cov=src.models --cov-report=term

# Fail if coverage < 80%
pytest --cov=src --cov-fail-under=80
```

### Parallel Test Execution

Speed up test suite with parallel execution:

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel (auto-detect cores)
pytest -n auto

# Run with 4 workers
pytest -n 4

# Parallel with coverage
pytest -n auto --cov=src --cov-report=html
```

### Debugging Test Failures

```bash
# Verbose output
pytest -v

# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb

# Run specific test with full output
pytest tests/unit/models/test_inference_module.py::test_inference_basic_run -vsx
```

---

## AWS Integration

### When to Test with AWS

While offline development covers 95% of workflows, **AWS testing is required for**:

1. **Pre-Deployment Validation**
   - Before releasing to production
   - Validate against latest production data
   - Ensure no AWS-specific issues (permissions, data formats)

2. **Quarterly Fixture Refresh**
   - Update fixtures with latest data
   - Validate fixtures match production
   - Ensure mathematical equivalence maintained

3. **Data Source Changes**
   - New data sources added
   - Schema changes in existing sources
   - New economic indicators

4. **Production Debugging**
   - Investigate production issues
   - Compare production vs fixture results
   - Validate bug fixes against live data

### Setting Up AWS Credentials

AWS tests require environment variables:

```bash
# Required environment variables
export STS_ENDPOINT_URL="https://sts.us-east-1.amazonaws.com"
export ROLE_ARN="arn:aws:iam::123456789012:role/YourRoleName"
export XID="your-external-id"
export BUCKET_NAME="your-s3-bucket"

# Optional: AWS region
export AWS_DEFAULT_REGION="us-east-1"

# Verify credentials
aws sts get-caller-identity
```

**Security Notes**:
- Never commit credentials to git
- Use environment variables or AWS profiles
- Credentials expire - refresh before AWS test runs
- Use least-privilege IAM roles

### Running AWS Tests

AWS tests are marked with `@pytest.mark.aws` and **skipped by default**:

```bash
# Run AWS tests explicitly
pytest -m aws -v
# Runs ~100 AWS tests in < 15 minutes

# AWS vs fixture equivalence
pytest tests/integration/test_aws_fixture_equivalence.py -m aws -v

# Specific AWS test
pytest tests/integration/test_aws_fixture_equivalence.py::test_data_loading_equivalence -m aws -v

# AWS tests with verbose output
pytest -m aws -vsx
```

### Fixture Refresh Workflow

Refresh fixtures from AWS production data (5 steps):

#### Step 1: Set AWS Credentials

```bash
export STS_ENDPOINT_URL="https://sts.us-east-1.amazonaws.com"
export ROLE_ARN="arn:aws:iam::123456789012:role/YourRole"
export XID="your-external-id"
export BUCKET_NAME="your-bucket"
```

#### Step 2: Run Fixture Refresh Script

```bash
python tests/fixtures/refresh_fixtures.py
```

**What it does**:
- Loads all data sources from S3 (sales, WINK, weights, macro)
- Runs full pipeline and captures all 10 stage outputs
- Runs inference and captures baseline results
- Saves metadata (refresh_date, data shapes, AWS config)

**Output**:
```
Refreshing fixtures from AWS...
Loading sales data...          ✓ (2.8M rows, 11 columns)
Loading competitive rates...   ✓ (1.1M rows, 19 columns)
Loading market weights...      ✓ (19 rows, 11 columns)
Loading macro data...          ✓ (5 indicators)
Running full pipeline...       ✓ (10 stages captured)
Running inference...           ✓ (baseline saved)
Fixtures refreshed successfully! Output: tests/fixtures/rila/
```

#### Step 3: Validate Fixtures

```bash
pytest tests/fixtures/test_fixture_validity.py -v
```

**Validates**:
- Fixture freshness (< 90 days old)
- File completeness (all required files exist)
- Data quality (schema, size, missing values)
- Metadata validity

#### Step 4: Test AWS Equivalence

```bash
pytest tests/integration/test_aws_fixture_equivalence.py -m aws -v
```

**Validates**:
- Data loading equivalence (AWS vs fixture)
- Pipeline equivalence (all 10 stages)
- Inference equivalence (coefficients, predictions)

#### Step 5: Commit Refreshed Fixtures

```bash
# Review changes
git status
git diff tests/fixtures/rila/refresh_metadata.json

# Commit refreshed fixtures
git add tests/fixtures/rila/
git commit -m "Refresh fixtures from AWS production data (Q1 2024)"
git push
```

### AWS Test Types

#### Data Loading Equivalence

Validates that fixture data matches AWS S3 data:

```python
@pytest.mark.aws
def test_data_loading_equivalence(aws_adapter, fixture_adapter):
    """Data loaded from AWS should match fixture data."""

    # Load from AWS S3
    aws_sales = aws_adapter.load_sales_data(product_filter="FlexGuard 6Y20B")

    # Load from fixture
    fixture_sales = fixture_adapter.load_sales_data(product_filter="FlexGuard 6Y20B")

    # Should be identical (fixtures captured from AWS)
    pd.testing.assert_frame_equal(aws_sales, fixture_sales, check_exact=False, rtol=1e-12)
```

#### Pipeline Equivalence

Validates that pipeline results match between AWS and fixtures:

```python
@pytest.mark.aws
def test_pipeline_equivalence(aws_adapter, fixture_adapter):
    """Pipeline results should match between AWS and fixture."""

    # Run with AWS data
    aws_pipeline = DataPipeline(adapter=aws_adapter)
    aws_result = aws_pipeline.run_full_pipeline()

    # Run with fixture data
    fixture_pipeline = DataPipeline(adapter=fixture_adapter)
    fixture_result = fixture_pipeline.run_full_pipeline()

    # Results should be mathematically equivalent (1e-12 precision)
    validate_dataframe_equivalence(
        actual=fixture_result,
        expected=aws_result,
        tolerance=1e-12,
        stage="aws_vs_fixture_full_pipeline"
    )
```

#### Inference Equivalence

Validates that inference results match between AWS and fixtures:

```python
@pytest.mark.aws
def test_inference_equivalence(aws_adapter, fixture_adapter):
    """Inference results should match between AWS and fixture."""

    # Run inference on AWS data
    aws_interface = create_interface("6Y20B", environment="aws", adapter_kwargs={'adapter': aws_adapter})
    aws_inference = aws_interface.run_inference()

    # Run inference on fixture data
    fixture_interface = create_interface("6Y20B", environment="fixture", adapter_kwargs={'adapter': fixture_adapter})
    fixture_inference = fixture_interface.run_inference()

    # Coefficients should match at 1e-12 precision
    validate_inference_equivalence(
        actual=fixture_inference,
        expected=aws_inference,
        tolerance=1e-12
    )
```

---

## Troubleshooting

### Fixture Loading Errors

#### Issue: `FileNotFoundError: final_weekly_dataset.parquet`

**Cause**: Required fixture file is missing

**Solution**:
```bash
# Check fixture directory
ls -lh tests/fixtures/rila/

# If missing, refresh from AWS (requires credentials)
python tests/fixtures/refresh_fixtures.py

# Or restore from git
git checkout tests/fixtures/rila/final_weekly_dataset.parquet
```

#### Issue: `KeyError: 'own_cap_rate'`

**Cause**: Fixture schema mismatch (column missing)

**Solution**:
```bash
# Validate fixture schema
pytest tests/fixtures/test_fixture_validity.py::test_fixture_schema -v

# If schema invalid, refresh fixtures
python tests/fixtures/refresh_fixtures.py
```

### Mathematical Equivalence Failures

#### Issue: Baseline comparison fails at 1e-12 precision

**Example Error**:
```
AssertionError: Coefficient mismatch exceeds 1e-12 precision
Expected: 0.123456789012
Actual:   0.123456789999
```

**Causes**:
1. Fixtures are stale (> 90 days old)
2. Code changes broke determinism
3. Random seed not set correctly

**Solutions**:

**1. Check Fixture Age**:
```bash
pytest tests/fixtures/test_fixture_validity.py::test_fixture_freshness -v
```

**2. Refresh Fixtures** (if stale):
```bash
python tests/fixtures/refresh_fixtures.py
```

**3. Check Random Seeds**:
```python
# Ensure random_state is set in all stochastic operations
model = BootstrapInference({'random_state': 42})  # ✓ Deterministic
model = BootstrapInference()                       # ✗ Non-deterministic
```

**4. Isolate Failing Stage**:
```bash
# Run stage-by-stage equivalence tests
pytest tests/integration/test_pipeline_stage_equivalence.py -v

# Identify which stage diverged
# Example output:
# PASSED test_stage_01_product_filtering
# PASSED test_stage_02_sales_cleanup
# FAILED test_stage_03_feature_engineering  ← Problem here
```

### Missing Fixture Files

#### Issue: Required fixture file doesn't exist

**Solution**:

**1. Validate Fixture Completeness**:
```bash
pytest tests/fixtures/test_fixture_validity.py::test_fixture_completeness -v
```

**2. Check Required Files**:
```bash
# List required fixtures
cat tests/fixtures/test_fixture_validity.py | grep required_files

# Verify files exist
ls tests/fixtures/rila/ | grep -E "(raw_sales|raw_wink|market_share|final_weekly|inference_baseline)"
```

**3. Refresh Missing Fixtures**:
```bash
python tests/fixtures/refresh_fixtures.py
```

### Stale Fixtures

#### Issue: Fixtures are > 90 days old

**Warning**:
```
WARNING: Fixtures are 120 days old (max: 90 days)
Consider refreshing: python tests/fixtures/refresh_fixtures.py
```

**Solution**:

**1. Check Fixture Age**:
```bash
pytest tests/fixtures/test_fixture_validity.py::test_fixture_freshness -v
```

**2. Review Metadata**:
```bash
cat tests/fixtures/rila/refresh_metadata.json
# Shows: refresh_date, data_shape, aws_config
```

**3. Refresh Fixtures** (requires AWS credentials):
```bash
python tests/fixtures/refresh_fixtures.py
```

**4. Quarterly Schedule**:
- Q1: January (after year-end data)
- Q2: April
- Q3: July
- Q4: October

### Performance Test Failures

#### Issue: Performance test exceeds baseline

**Example Error**:
```
AssertionError: Feature engineering took 3.2s (max: 2.0s)
```

**Causes**:
1. Code regression (new inefficiency)
2. System load (CI/CD contention)
3. Baseline too aggressive

**Solutions**:

**1. Isolate Performance Issue**:
```bash
# Run performance test with profiling
pytest tests/performance/test_performance_baselines.py::test_feature_engineering_performance -v -s
```

**2. Profile Code**:
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run operation
result = engineer_features(medium_dataset)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 slowest functions
```

**3. Update Baseline** (if appropriate):
```bash
# Update tests/performance/baselines.json
# Only if code change is intentional and justified
```

### Test Flakiness

#### Issue: Tests pass sometimes, fail others

**Causes**:
1. Non-deterministic operations (missing random_state)
2. Floating-point precision issues
3. Test order dependencies
4. Race conditions (parallel tests)

**Solutions**:

**1. Check Determinism**:
```bash
# Run test 10 times
for i in {1..10}; do
    pytest tests/unit/models/test_inference_module.py::test_inference_basic_run -v
done
```

**2. Fix Random Seeds**:
```python
# ✓ Deterministic
np.random.seed(42)
model = BootstrapInference({'random_state': 42})

# ✗ Non-deterministic
model = BootstrapInference()  # Missing random_state
```

**3. Disable Parallel Execution**:
```bash
# Run serially (if parallel causes issues)
pytest tests/ -v  # No -n flag
```

**4. Use Hypothesis Stateful Testing**:
```python
from hypothesis.stateful import RuleBasedStateMachine

# Hypothesis will find flaky tests automatically
```

---

## Best Practices

### DO ✓

1. **Use Fixtures for All Development**
   ```python
   # ✓ GOOD: Offline development
   interface = create_interface("6Y20B", environment="fixture")

   # ✗ BAD: Unnecessary AWS dependency
   interface = create_interface("6Y20B", environment="aws")
   ```

2. **Run AWS Tests Before Deployment**
   ```bash
   # ✓ GOOD: Validate against production before release
   pytest -m aws -v

   # ✗ BAD: Deploy without AWS validation
   git push origin main  # Without AWS tests
   ```

3. **Refresh Fixtures Quarterly**
   ```bash
   # ✓ GOOD: Regular refresh schedule
   python tests/fixtures/refresh_fixtures.py  # Q1, Q2, Q3, Q4

   # ✗ BAD: Let fixtures become stale (> 90 days)
   # Stale fixtures → divergence from production
   ```

4. **Use Smallest Fixture for Each Test**
   ```python
   # ✓ GOOD: Fast unit test with tiny fixture
   def test_aic_formula(tiny_dataset, small_bootstrap_config):
       aic = calculate_aic(tiny_dataset, small_bootstrap_config)
       assert aic > 0

   # ✗ BAD: Slow unit test with full production data
   def test_aic_formula(full_production_dataset, production_bootstrap_config):
       aic = calculate_aic(full_production_dataset, production_bootstrap_config)
       assert aic > 0  # Same assertion, but 100x slower
   ```

5. **Skip Slow Tests in Fast CI**
   ```bash
   # ✓ GOOD: Fast CI loop (< 5 minutes)
   pytest -m "not slow and not aws"

   # ✗ BAD: Slow CI loop (> 15 minutes)
   pytest  # Runs all tests including slow ones
   ```

6. **Set Random Seeds for Determinism**
   ```python
   # ✓ GOOD: Reproducible results
   model = BootstrapInference({'random_state': 42})

   # ✗ BAD: Non-reproducible results
   model = BootstrapInference()  # Results vary across runs
   ```

7. **Use Hierarchical Fixtures**
   ```python
   # ✓ GOOD: Appropriate fixture size
   def test_unit(tiny_dataset):           # Unit test
   def test_integration(medium_dataset):  # Integration test
   def test_e2e(full_production_dataset): # E2E test

   # ✗ BAD: Always using largest fixture
   def test_unit(full_production_dataset):  # Unnecessary
   ```

8. **Validate Mathematical Equivalence**
   ```python
   # ✓ GOOD: Validate against baseline
   np.testing.assert_allclose(result, baseline, rtol=1e-12)

   # ✗ BAD: No validation
   # Changes might break equivalence unnoticed
   ```

### DON'T ✗

1. **Don't Commit AWS Credentials**
   ```bash
   # ✗ BAD: Credentials in code
   aws_config = {
       'role_arn': 'arn:aws:iam::123456789012:role/MyRole',
       'xid': 'my-secret-id'  # ← Security risk!
   }

   # ✓ GOOD: Use environment variables
   aws_config = {
       'role_arn': os.environ['ROLE_ARN'],
       'xid': os.environ['XID']
   }
   ```

2. **Don't Edit Fixtures Manually**
   ```bash
   # ✗ BAD: Manually edit fixture
   vim tests/fixtures/rila/final_weekly_dataset.parquet

   # ✓ GOOD: Refresh from AWS
   python tests/fixtures/refresh_fixtures.py
   ```

3. **Don't Use Stale Fixtures**
   ```bash
   # ✗ BAD: Ignore fixture freshness warnings
   # WARNING: Fixtures are 120 days old

   # ✓ GOOD: Refresh when warned
   python tests/fixtures/refresh_fixtures.py
   ```

4. **Don't Run AWS Tests in Fast CI Loops**
   ```yaml
   # ✗ BAD: CI runs AWS tests on every commit
   script:
     - pytest  # Includes AWS tests → slow + expensive

   # ✓ GOOD: CI skips AWS tests (pre-deploy only)
   script:
     - pytest -m "not aws and not slow"  # Fast CI
   ```

5. **Don't Use Large Fixtures for Unit Tests**
   ```python
   # ✗ BAD: Unit test with full production data
   def test_simple_calculation(full_production_dataset):
       result = calculate_average(full_production_dataset)
       assert result > 0  # Takes 5s instead of 0.01s

   # ✓ GOOD: Unit test with tiny fixture
   def test_simple_calculation(tiny_dataset):
       result = calculate_average(tiny_dataset)
       assert result > 0  # Fast!
   ```

6. **Don't Ignore Performance Regressions**
   ```bash
   # ✗ BAD: Ignore performance test failure
   # FAILED: Feature engineering took 5.2s (max: 2.0s)
   # "It's probably fine..." ← No!

   # ✓ GOOD: Investigate and fix
   # Profile code, identify bottleneck, optimize
   ```

7. **Don't Break Determinism**
   ```python
   # ✗ BAD: Non-deterministic operation
   shuffled = df.sample(frac=1.0)  # Different order each run

   # ✓ GOOD: Deterministic operation
   shuffled = df.sample(frac=1.0, random_state=42)  # Same order
   ```

8. **Don't Skip Documentation Updates**
   ```bash
   # ✗ BAD: Code changes without doc updates
   git commit -m "Add new feature"  # No README update

   # ✓ GOOD: Update docs with code
   git add README.md
   git commit -m "Add new feature (with docs)"
   ```

### Code Review Checklist

When reviewing PRs, check:

- [ ] All tests pass offline (`pytest -m "not aws"`)
- [ ] New tests use appropriate fixture size (small/medium/large)
- [ ] Random seeds set for stochastic operations
- [ ] Mathematical equivalence validated (1e-12 or 1e-6)
- [ ] Performance baselines not regressed
- [ ] No AWS credentials in code
- [ ] Documentation updated
- [ ] Fixtures not manually edited
- [ ] Test markers added (`@pytest.mark.aws`, `@pytest.mark.slow`)

---

## References

### Documentation

- **Testing Guide**: `docs/development/TESTING_GUIDE.md`
  - Comprehensive testing strategy
  - All test types (unit, integration, E2E, property-based)
  - Mathematical equivalence testing
  - Performance baseline testing
  - AWS integration testing

- **Fixture Management**: `tests/fixtures/README.md`
  - Fixture refresh procedures
  - Fixture validation tests
  - Metadata tracking

- **Performance Testing**: `tests/performance/README.md`
  - Performance baseline configuration
  - Memory baseline configuration
  - Regression detection

### Key Files

- **Fixture Adapter**: `src/adapters/fixture_adapter.py`
  - Loads data from local fixtures
  - Drop-in replacement for S3Adapter

- **Global Fixtures**: `tests/conftest.py`
  - Hierarchical fixture definitions (small/medium/large)
  - Hypothesis strategies
  - Pytest configuration

- **Fixture Refresh**: `tests/fixtures/refresh_fixtures.py`
  - Automated fixture refresh from AWS
  - Metadata generation

- **Baseline Validation**: `tests/integration/test_pipeline_stage_equivalence.py`
  - Stage-by-stage equivalence testing
  - 1e-12 precision validation

### External Resources

- **pytest Documentation**: https://docs.pytest.org/
- **Hypothesis Documentation**: https://hypothesis.readthedocs.io/
- **pandas Testing**: https://pandas.pydata.org/docs/reference/general_utility_functions.html#testing-functions
- **numpy Testing**: https://numpy.org/doc/stable/reference/routines.testing.html

### Getting Help

- **Issues**: Create an issue in the project repository
- **Questions**: Ask in team Slack channel
- **AWS Access**: Contact DevOps team for credentials
- **Fixture Refresh**: Automated quarterly, or run manually with AWS credentials

---

**Last Updated**: 2026-01-29

**Version**: 1.0.0

**Status**: Complete - All 8 Phases Implemented

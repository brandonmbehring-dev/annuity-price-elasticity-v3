# Testing Strategy

**Version**: 1.0.0 | **Last Updated**: 2026-01-31

This document defines the testing approach for the annuity price elasticity v3 repository, a multi-product system for estimating price sensitivity in insurance products (RILA, FIA, MYGA).

---

## Core Testing Principles

### 1. Real Tests Only

**NO stubs, TODOs, or placeholder tests**. Every test must:
- Actually execute the code being tested
- Assert meaningful correctness properties
- Use real or realistic fixture data (not just "doesn't crash")

```python
# BAD - Stub test
def test_calculate_elasticity():
    pass  # TODO: implement

# GOOD - Real test
def test_calculate_elasticity_positive_for_own_rate():
    """Own rate elasticity should be positive (higher rates attract customers)."""
    result = calculate_elasticity(own_rate=0.085, competitor_mean=0.075)

    assert result.own_rate_coefficient > 0, (
        f"Expected positive own-rate elasticity, got {result.own_rate_coefficient}"
    )
    assert result.r_squared > 0.10, "R-squared too low for meaningful model"
```

### 2. Coverage Targets

| Code Type | Target Coverage | Rationale |
|-----------|----------------|-----------|
| **Core modules** (`src/core/`, `src/models/`) | 80%+ | Critical business logic |
| **Infrastructure** (`src/data/`, `src/config/`) | 60%+ | More I/O, lower bar |
| **Leakage gates** (`src/validation/leakage_gates.py`) | 90%+ | Safety-critical path |
| **Feature engineering** (`src/features/`) | 70%+ | Complex transformations |
| **Visualization** (`src/visualization/`) | 50%+ | Display logic, harder to test |

### 3. Test Categories

annuity-price-elasticity-v3 uses a **6-layer validation architecture**:

```
Layer 1: Unit tests (pure functions, isolated logic)
Layer 2: Integration tests (adapters + pipelines, components working together)
Layer 3: Anti-pattern tests (detect leakage bugs, verify gates catch violations)
Layer 4: Property tests (statistical invariants hold across random inputs)
Layer 5: Benchmarks (notebook equivalence, baseline comparisons)
Layer 6: End-to-end validation (full workflow from data to inference)
```

---

## Testing Layers

### Layer 1: Unit Tests

**Purpose**: Test individual functions in isolation.

**Location**: `tests/unit/` (mirroring `src/` structure)

**Coverage**:
- All public functions in `src.*`
- Edge cases (empty DataFrames, single row, missing columns)
- Parameter validation (raises on invalid input)
- Economic constraints (coefficient signs, bounds)

**Example**:
```python
class TestCalculateDaysBetweenDates:
    """Unit tests for date calculation utility."""

    def test_same_date_returns_zero(self):
        result = calculate_days_between_dates("2024-01-15", "2024-01-15")
        assert result == 0

    def test_one_week_apart(self):
        result = calculate_days_between_dates("2024-01-08", "2024-01-15")
        assert result == 7

    def test_raises_on_invalid_date_format(self):
        with pytest.raises(ValueError, match="Invalid date format"):
            calculate_days_between_dates("not-a-date", "2024-01-15")
```

### Layer 2: Integration Tests

**Purpose**: Test interactions between components.

**Location**: `tests/integration/`

**Coverage**:
- Data adapters + pipeline orchestration
- UnifiedNotebookInterface + methodology classes
- Aggregation strategies + data sources
- Configuration builders + validators

**Example**:
```python
class TestPipelineIntegration:
    """Test data pipeline with fixture adapter."""

    def test_fixture_to_inference_pipeline(self, fixture_adapter):
        """End-to-end: Load fixtures -> Transform -> Validate -> Output."""
        # Load via DI pattern
        df = fixture_adapter.load_final_weekly_dataset()

        # Run preprocessing
        processed = preprocess_for_inference(df)

        # Validate output
        assert "own_rate_t0" in processed.columns
        assert "competitor_weighted_t2" in processed.columns
        assert processed["date"].is_monotonic_increasing
```

### Layer 3: Anti-Pattern Tests

**Purpose**: Verify that leakage bugs are **caught** by validation gates.

**Location**: `tests/anti_patterns/`

**Coverage**:
- 10 bug categories from domain-specific leakage patterns
- Each anti-pattern must trigger HALT status
- Tests that deliberately encode bugs to verify gates catch them

**Example**:
```python
class TestLag0LeakageDetection:
    """Anti-pattern: Lag-0 competitor features violate causal identification."""

    def test_gate_catches_lag0_competitor(self):
        """Gate should HALT when lag-0 competitor features are present."""
        features = ["own_rate_t0", "C_lag0", "competitor_mean_t2"]  # C_lag0 is leakage

        result = detect_lag0_features(features)

        assert result.status == GateStatus.HALT, (
            "Gate failed to catch lag-0 leakage"
        )
        assert "C_lag0" in result.message

    def test_gate_passes_clean_features(self):
        """Gate should PASS when all competitor features are lagged."""
        features = ["own_rate_t0", "competitor_mean_t2", "competitor_weighted_t3"]

        result = detect_lag0_features(features)

        assert result.status == GateStatus.PASS
```

**Bug categories tested** (from `knowledge/practices/LEAKAGE_AUDIT_TRAIL.md`):
1. Lag leakage (competitor rates at t=0)
2. Temporal boundary violations (future data in training)
3. Threshold leakage (feature selection on full data)
4. Market weight leakage (weights from full dataset)
5. Scaling leakage (standardization on all data)
6. Product mix confounding (6Y20B vs 6Y10B populations)
7. Own-rate endogeneity (circular reasoning)
8. Holiday effect leakage (holiday flags computed globally)
9. Cross-validation shuffling (temporal order violated)
10. Macro feature lookahead (economic indicators from future)

### Layer 4: Property Tests

**Purpose**: Verify statistical properties hold over many random inputs.

**Location**: `tests/property_based/`

**Coverage**:
- Economic constraints hold across random valid inputs
- Statistical properties (R-squared bounds, coefficient ranges)
- DataFrame invariants (row counts preserved, no NaN introduction)
- Pipeline idempotency (same input -> same output)

**Example**:
```python
from hypothesis import given, strategies as st

class TestElasticityProperties:
    """Property-based tests for elasticity calculations."""

    @given(
        own_rate=st.floats(min_value=0.01, max_value=0.20),
        competitor_mean=st.floats(min_value=0.01, max_value=0.20),
    )
    def test_r_squared_bounded(self, own_rate, competitor_mean):
        """R-squared must always be in [0, 1]."""
        result = fit_elasticity_model(own_rate, competitor_mean)

        assert 0 <= result.r_squared <= 1, (
            f"R-squared {result.r_squared} out of bounds"
        )

    @pytest.mark.parametrize("n_rows", [10, 100, 1000])
    def test_preprocessing_preserves_row_count(self, n_rows):
        """Preprocessing should not add or remove rows (for valid data)."""
        df = create_valid_dataframe(n_rows=n_rows)

        result = preprocess_for_inference(df)

        assert len(result) == n_rows
```

### Layer 5: Benchmarks

**Purpose**: Validate accuracy and consistency against baselines.

**Location**: `tests/baselines/notebooks/`

**Coverage**:
- Notebook execution produces expected outputs
- Key metrics match captured baselines (within tolerance)
- Mathematical equivalence during refactoring (1e-12 precision)

**Metrics**:
- R-squared, RMSE, MAPE
- Coefficient values and signs
- Runtime performance

**Example**:
```python
class TestNotebookEquivalence:
    """Verify notebooks produce consistent outputs."""

    def test_rila_6y20b_inference_matches_baseline(self):
        """NB02 should produce results matching captured baseline."""
        # Run notebook with fixture data
        result = execute_notebook("notebooks/production/rila_6y20b/02_inference.ipynb")

        # Load baseline
        baseline = pd.read_parquet("tests/baselines/rila_6y20b/inference_results.parquet")

        # Compare key metrics
        np.testing.assert_allclose(
            result["own_rate_coefficient"],
            baseline["own_rate_coefficient"],
            rtol=1e-6,
            err_msg="Own rate coefficient changed"
        )
```

### Layer 6: End-to-End Validation

**Purpose**: Test complete user workflows from data loading to deployment.

**Location**: `tests/integration/test_e2e_*.py`

**Coverage**:
- UnifiedNotebookInterface → Data → Transform → Model → Results
- Notebook execution (`nbconvert --execute` or `papermill`)
- Example scripts run without errors
- Output validation against expected schemas

**Example**:
```python
class TestFullWorkflow:
    """End-to-end: Complete inference workflow."""

    def test_rila_inference_workflow(self):
        """Full workflow: Load → Preprocess → Model → Validate → Output."""
        # 1. Create interface with fixture adapter
        interface = create_interface(
            product_code="6Y20B",
            environment="fixture"
        )

        # 2. Load data
        df = interface.load_data()
        assert len(df) > 1000, "Insufficient data loaded"

        # 3. Run inference
        results = interface.run_inference(df)

        # 4. Validate economic constraints
        assert results.own_rate_coefficient > 0, (
            "Own rate should have positive coefficient"
        )

        # 5. Run leakage gates
        report = run_all_gates(
            feature_names=list(df.columns),
            r_squared=results.r_squared
        )
        assert report.passed, f"Leakage gates failed: {report}"
```

---

## Test Organization

```
tests/
├── conftest.py                          # Shared fixtures
├── fixtures/                            # Test data fixtures
│   ├── rila/                            # RILA product fixtures
│   │   ├── final_weekly_dataset.parquet # 73 MB production-like data
│   │   └── economic_indicators/         # Supporting data
│   └── fia/                             # FIA product fixtures
├── unit/                                # Layer 1: Unit tests
│   ├── core/                            # Core module tests
│   ├── data/                            # Data layer tests
│   │   └── adapters/                    # Adapter tests
│   ├── features/                        # Feature engineering tests
│   │   └── selection/                   # Feature selection tests
│   ├── models/                          # Model tests
│   ├── products/                        # Product methodology tests
│   ├── validation/                      # Validation tests
│   └── visualization/                   # Visualization tests
├── integration/                         # Layer 2 & 6: Integration + E2E
│   ├── test_pipeline_fixture_equivalence.py
│   └── test_notebook_equivalence.py
├── anti_patterns/                       # Layer 3: Leakage detection
│   ├── test_lag0_competitor_detection.py
│   ├── test_coefficient_sign_validation.py
│   ├── test_future_leakage.py
│   └── test_economic_plausibility.py
├── property_based/                      # Layer 4: Property tests
│   ├── test_economic_invariants.py
│   ├── test_statistical_invariants.py
│   ├── test_dataframe_invariants.py
│   └── test_pipeline_properties.py
├── baselines/                           # Layer 5: Benchmarks
│   └── notebooks/
│       ├── rila_6y20b/
│       └── rila_1y10b/
└── benchmark/                           # Performance benchmarks
```

---

## Running Tests

### Quick Validation (Development)

```bash
# Quick smoke test (~30 seconds)
make quick-check

# Run all unit tests with coverage
pytest tests/unit/ -v --cov=src --cov-report=term-missing

# Fail if coverage < 60%
pytest tests/unit/ --cov=src --cov-fail-under=60
```

### Selective Testing

```bash
# Unit tests only (fastest, ~2 minutes)
make test

# Integration tests
pytest tests/integration/ -v

# Anti-pattern tests (leakage detection)
pytest tests/anti_patterns/ -v -m leakage

# Property tests (may be slower due to Hypothesis)
pytest tests/property_based/ -v --hypothesis-show-statistics

# Notebook validation (fixture-compatible)
make test-notebooks

# All notebooks including AWS-dependent
make test-notebooks-aws
```

### Full CI Suite

```bash
# Full test suite (unit + notebooks)
make test-all

# With coverage report
make coverage
```

---

## Continuous Integration (CI)

### GitHub Actions Workflow

**Triggers**:
- Every PR
- Push to `main`
- Nightly (for full benchmarks)

**Jobs**:
1. **Unit tests**: Layer 1-2 (fast, <5 min)
2. **Anti-pattern tests**: Layer 3 (critical path, blocks deployment)
3. **Integration tests**: Layer 6 (full workflow)
4. **Notebook validation**: Execute 5 fixture-compatible notebooks
5. **Type checking**: `mypy src/`
6. **Linting**: `ruff check src/ tests/`

**Matrix**:
- **Python**: 3.10, 3.11
- **Dependencies**: Minimum versions, latest versions

---

## Test Data Strategy

### Fixture Data (Default)

Most tests use captured production-like fixtures for reproducibility:

```python
@pytest.fixture
def rila_fixture_data():
    """Load RILA fixture data for testing."""
    return pd.read_parquet("tests/fixtures/rila/final_weekly_dataset.parquet")
```

**Pros**:
- Deterministic (reproducible)
- Fast (no I/O to AWS)
- Production-realistic (1.1M rows)
- Integrity validated (checksums)

**Cons**:
- May drift from production data over time
- Requires periodic refresh

### Synthetic Data (Unit Tests)

For isolated unit tests, use synthetic data:

```python
def test_with_synthetic_data():
    np.random.seed(42)  # Fixed seed for reproducibility
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=100, freq="W"),
        "own_rate": np.random.uniform(0.05, 0.15, 100),
        "competitor_mean": np.random.uniform(0.05, 0.15, 100),
    })
    # ... test logic
```

### Production Data (AWS Tests)

Some tests require AWS access for production data:

```bash
# These tests require AWS credentials
make test-notebooks-aws
```

---

## Assertion Style Guide

### Preferred Patterns

```python
# Use numpy.testing for float comparisons
np.testing.assert_almost_equal(result, expected, decimal=6)
np.testing.assert_allclose(result, expected, rtol=1e-5)

# Use pytest.raises for error checking
with pytest.raises(ValueError, match="coefficient must be positive"):
    validate_elasticity(coefficient=-0.5)

# Use descriptive assertion messages
assert result.r_squared > 0.10, (
    f"Expected R-squared > 0.10 for meaningful model, got {result.r_squared}"
)

# Include business context in messages
assert own_rate_coef > 0, (
    f"Own rate coefficient should be positive (higher rates attract customers). "
    f"Got {own_rate_coef}. Check for data issues or model specification."
)
```

### Anti-Patterns

```python
# BAD: Direct float comparison (floating point errors)
assert result == 0.333333  # May fail due to precision

# BAD: Vague assertions
assert result  # What property are we testing?

# BAD: Low information assertions
assert len(df) > 0  # What should the length be?

# BAD: Multiple unrelated assertions
def test_everything():
    assert model.r_squared > 0.1
    assert gate.status == "PASS"
    assert len(features) == 15
    # Too many concerns -> split into separate tests
```

---

## Common Testing Patterns

### Testing Economic Constraints

```python
class TestEconomicConstraints:
    """Tests for economic plausibility of model outputs."""

    def test_own_rate_positive_effect(self, inference_results):
        """Higher own rates should attract customers (positive coefficient)."""
        assert inference_results.own_rate_coefficient > 0, (
            "Economic theory: higher rates should increase sales"
        )

    def test_competitor_rate_negative_effect(self, inference_results):
        """Higher competitor rates should reduce our relative attractiveness."""
        assert inference_results.competitor_coefficient < 0, (
            "Economic theory: higher competitor rates hurt our sales"
        )
```

### Testing Validation Gates

```python
class TestLeakageGates:
    """Tests for leakage detection gates."""

    def test_gate_pass_on_valid_config(self):
        """Gate passes on valid configuration."""
        result = check_temporal_boundary(
            train_dates=pd.date_range("2024-01-01", periods=80, freq="W"),
            test_dates=pd.date_range("2025-07-01", periods=20, freq="W"),
        )
        assert result.status == GateStatus.PASS

    def test_gate_halt_on_overlap(self):
        """Gate halts when train/test dates overlap."""
        result = check_temporal_boundary(
            train_dates=pd.date_range("2024-01-01", periods=100, freq="W"),
            test_dates=pd.date_range("2024-06-01", periods=50, freq="W"),  # Overlaps!
        )
        assert result.status == GateStatus.HALT
        assert "overlap" in result.message.lower()
```

### Testing Adapters

```python
class TestFixtureAdapter:
    """Tests for fixture data adapter."""

    def test_load_returns_expected_schema(self, fixture_adapter):
        """Loaded data should have expected columns."""
        df = fixture_adapter.load_final_weekly_dataset()

        required_columns = {"date", "product_name", "premium_amount", "own_rate"}
        assert required_columns.issubset(set(df.columns))

    def test_load_returns_expected_row_count(self, fixture_adapter):
        """Loaded data should have expected number of rows."""
        df = fixture_adapter.load_final_weekly_dataset()

        assert len(df) == 50873, (
            f"Expected 50873 rows from fixture, got {len(df)}"
        )
```

---

## Test-Driven Development (TDD)

For new features, follow the **Red-Green-Refactor** cycle:

1. **Red**: Write failing test
2. **Green**: Implement minimal code to pass
3. **Refactor**: Clean up while keeping tests passing

**Example**: Adding a new aggregation strategy

```python
# Step 1: RED - Write test first
def test_top_n_aggregation_selects_largest():
    """TopN strategy should select N largest competitors by premium."""
    df = create_competitor_df(companies=["A", "B", "C", "D"], premiums=[100, 200, 300, 50])

    strategy = TopNAggregationStrategy(n=2)
    result = strategy.aggregate(df)

    assert set(result["company"]) == {"B", "C"}  # Top 2 by premium

# Step 2: GREEN - Implement
class TopNAggregationStrategy(AggregationStrategy):
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.nlargest(self.n, "premium")

# Step 3: REFACTOR - Add edge cases, improve implementation
```

---

## Randomness and Reproducibility

### Seeds in Tests

**ALL tests must be deterministic**. Use fixed random seeds:

```python
def test_with_random_data():
    np.random.seed(42)  # Fixed seed for reproducibility
    df = create_random_features(n=100)
    # ... test logic
```

### Seeds in Production Code

Functions that use randomness accept `random_state` parameter:

```python
def run_shuffled_target_test(
    model, X, y,
    n_shuffles: int = 5,
    random_state: Optional[int] = None
) -> GateResult:
    """
    Parameters
    ----------
    random_state : int or None
        Random seed for reproducibility. If None, use system randomness.
    """
    rng = np.random.default_rng(random_state)
    # ... use rng for all random operations
```

---

## Anti-Pattern Gallery

### Anti-Pattern #1: Lag-0 Competitor Features

**Why it's bad**: Violates causal identification. Customers can't react to rates they haven't seen.

```python
# DON'T
features = ["own_rate_t0", "competitor_rate_t0"]  # Lag-0 leakage!

# DO
features = ["own_rate_t0", "competitor_rate_t2"]  # Proper 2-week lag
```

**Gate detection**: `detect_lag0_features()` returns HALT

### Anti-Pattern #2: Random Train/Test Split

**Why it's bad**: Temporal data requires temporal splits. Random splits create leakage.

```python
# DON'T
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# DO
cutoff = int(len(X) * 0.8)
X_train, X_test = X[:cutoff], X[cutoff:]  # Temporal split
```

**Gate detection**: `check_temporal_boundary()` returns HALT

### Anti-Pattern #3: Fitting Scaler on Full Data

**Why it's bad**: Test data statistics leak into training.

```python
# DON'T
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit on ALL data
X_train, X_test = X_scaled[:split], X_scaled[split:]

# DO
scaler = StandardScaler()
X_train = scaler.fit_transform(X[:split])  # Fit on train only
X_test = scaler.transform(X[split:])       # Transform test
```

### Anti-Pattern #4: Centered Rolling Window

**Why it's bad**: `center=True` uses future values.

```python
# DON'T
df["rolling_mean"] = df["rate"].rolling(4, center=True).mean()

# DO
df["rolling_mean"] = df["rate"].rolling(4).mean()  # center=False default
```

### Anti-Pattern #5: Backfilled NaN Values

**Why it's bad**: Backfilling implicitly uses future information.

```python
# DON'T
df["rolling_mean"] = df["rate"].rolling(4).mean().bfill()  # Uses future!

# DO
df["rolling_mean"] = df["rate"].rolling(4).mean()  # Keep NaN at start
```

---

## Fixture Management

### 3-Tier Fixture Architecture

```
Tier 1: Minimal Fixtures (Unit Tests)
├── 100-1000 rows
├── Synthetic but realistic
└── Fast to create

Tier 2: Product Fixtures (Integration Tests)
├── 10,000-50,000 rows per product
├── Captured from production
└── Validated integrity

Tier 3: Full Production Fixtures (E2E Tests)
├── 1M+ rows
├── Complete production snapshot
└── Monthly refresh
```

### Fixture Validation

```python
class TestFixtureIntegrity:
    """Ensure fixture data is valid and consistent."""

    def test_fixture_has_expected_date_range(self):
        df = load_fixture("rila/final_weekly_dataset.parquet")

        assert df["date"].min() >= pd.Timestamp("2020-01-01")
        assert df["date"].max() <= pd.Timestamp("2025-12-31")

    def test_fixture_has_no_duplicate_keys(self):
        df = load_fixture("rila/final_weekly_dataset.parquet")

        duplicates = df.duplicated(subset=["date", "product_code"])
        assert not duplicates.any(), f"Found {duplicates.sum()} duplicate rows"
```

---

## Performance Testing

### Runtime Benchmarks

```python
@pytest.mark.benchmark
def test_inference_completes_in_reasonable_time(benchmark, fixture_data):
    """Full inference should complete in under 5 seconds."""
    result = benchmark(run_inference, fixture_data)

    assert result.stats.mean < 5.0, (
        f"Inference took {result.stats.mean:.2f}s, expected < 5s"
    )
```

### Memory Profiling

```python
@pytest.mark.memory
def test_inference_memory_usage(fixture_data):
    """Memory usage should stay under 2GB for standard fixture."""
    import tracemalloc

    tracemalloc.start()
    result = run_inference(fixture_data)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 2 * 1024 * 1024 * 1024, (
        f"Peak memory {peak / 1024**3:.2f}GB exceeds 2GB limit"
    )
```

---

## References

1. **6-Layer Validation Architecture**: `~/Claude/lever_of_archimedes/patterns/testing.md`
2. **Data Leakage Prevention**: `~/Claude/lever_of_archimedes/patterns/data_leakage_prevention.md`
3. **Anti-Pattern Tests**: `tests/anti_patterns/`
4. **pytest Documentation**: https://docs.pytest.org
5. **numpy.testing Guide**: https://numpy.org/doc/stable/reference/routines.testing.html
6. **Hypothesis Property Testing**: https://hypothesis.readthedocs.io

---

## Appendix: Test Commands Quick Reference

| Command | Scope | Duration |
|---------|-------|----------|
| `make quick-check` | Smoke test | ~30s |
| `make test` | Unit tests | ~2min |
| `make test-notebooks` | 5 fixture notebooks | ~5min |
| `make test-all` | Unit + notebooks | ~7min |
| `make coverage` | Full + HTML report | ~8min |
| `pytest -k "leakage"` | Leakage tests only | ~1min |
| `pytest --hypothesis-show-statistics` | Property tests with stats | ~3min |

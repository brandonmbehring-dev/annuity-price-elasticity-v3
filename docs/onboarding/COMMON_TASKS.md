# Common Tasks for Data Scientists

**Copy-paste recipes for everyday workflows.**

Each task includes the minimal code needed. For deeper understanding, see the linked documentation.

---

## Quick Reference

| Task | Time | Section |
|------|------|---------|
| Run RILA inference | 2 min | [Task 1](#task-1-run-rila-inference) |
| Run inference with AWS | 5 min | [Task 2](#task-2-run-inference-with-aws-data) |
| Add a new product variant | 15 min | [Task 3](#task-3-add-a-new-product-variant) |
| Validate coefficient signs | 1 min | [Task 4](#task-4-validate-economic-constraints) |
| Run feature selection | 3 min | [Task 5](#task-5-run-feature-selection) |
| Export results | 1 min | [Task 6](#task-6-export-results) |
| Run tests | 2 min | [Task 7](#task-7-run-tests) |
| Check for leakage | 5 min | [Task 8](#task-8-check-for-data-leakage) |
| Create a baseline | 10 min | [Task 9](#task-9-create-a-baseline-for-testing) |
| Debug a constraint violation | 10 min | [Task 10](#task-10-debug-constraint-violations) |

---

## Task 1: Run RILA Inference

**Use fixture data (no AWS credentials needed).**

```python
from src.notebooks import create_interface

# Create interface for 6Y20B product
interface = create_interface("6Y20B", environment="fixture")

# Load fixture data
df = interface.load_data()

# Run inference
results = interface.run_inference(df)

# View results
print("Coefficients:")
for feature, coef in results["coefficients"].items():
    print(f"  {feature}: {coef:.4f}")

print(f"\nElasticity point estimate: {results['elasticity_point']:.4f}")
print(f"R-squared: {results['model_fit']['r_squared']:.4f}")
```

**Supported products**: `6Y20B`, `6Y10B`, `10Y20B`

---

## Task 2: Run Inference with AWS Data

**Requires AWS credentials.**

```python
from src.notebooks import create_interface

# AWS configuration
aws_config = {
    "sts_endpoint_url": "https://sts.us-east-1.amazonaws.com",
    "role_arn": "arn:aws:iam::123456789012:role/YourRole",
    "xid": "your_user_id",
    "bucket_name": "your-data-bucket",
}

# Create interface with AWS
interface = create_interface(
    "6Y20B",
    environment="aws",
    adapter_kwargs={"config": aws_config}
)

# Rest is identical to fixture mode
df = interface.load_data()
results = interface.run_inference(df)
```

See `docs/onboarding/AWS_SETUP_GUIDE.md` for credential setup.

---

## Task 3: Add a New Product Variant

**Example: Add 6Y15B (6-year, 15% buffer)**

### Step 1: Add to product registry

```python
# In src/config/product_registry.py

# Find the RILA_PRODUCTS section and add:
ProductConfig(
    product_code="6Y15B",
    product_type="rila",
    name="FlexGuard 6Y 15% Buffer",
    buffer_level=0.15,
    term_years=6,
    index="S&P 500",
    own_rate_prefix="prudential",
)
```

### Step 2: Create fixture data

```bash
# Create directory
mkdir -p tests/fixtures/rila/6Y15B

# Copy and modify existing fixtures (or generate from AWS)
cp tests/fixtures/rila/final_weekly_dataset.parquet \
   tests/fixtures/rila/6Y15B/final_weekly_dataset.parquet
```

### Step 3: Verify with tests

```bash
# Run product registry tests
pytest tests/unit/config/test_product_registry.py -v

# Run multiproduct equivalence
pytest tests/integration/test_multiproduct.py -v
```

### Step 4: Test the new product

```python
from src.notebooks import create_interface

interface = create_interface("6Y15B", environment="fixture")
df = interface.load_data()
results = interface.run_inference(df)
print(results["coefficients"])
```

---

## Task 4: Validate Economic Constraints

**Check coefficient signs match economic theory.**

```python
from src.notebooks import create_interface

interface = create_interface("6Y20B", environment="fixture")
df = interface.load_data()
results = interface.run_inference(df)

# Validate constraints
validation = interface.validate_coefficients(results["coefficients"])

# Report
print("PASSED constraints:")
for item in validation["passed"]:
    print(f"  {item['feature']}: {item['coefficient']:.4f} ({item['expected']})")

print("\nVIOLATED constraints:")
for item in validation["violated"]:
    print(f"  {item['feature']}: {item['coefficient']:.4f}")
    print(f"    Expected {item['expected']}, got {item['actual']}")
```

**Expected signs:**
- Own rate (prudential_*): **Positive**
- Competitor rates: **Negative**

---

## Task 5: Run Feature Selection

**Auto-select best features for the model.**

```python
from src.notebooks import create_interface

interface = create_interface("6Y20B", environment="fixture")
df = interface.load_data()

# Run feature selection
selection_results = interface.run_feature_selection(
    data=df,
    config={
        "target_column": "sales_target_current",
        "max_features": 5,
    }
)

# View selected features
print("Selected features:")
for feature in selection_results.selected_features:
    print(f"  {feature}")
```

---

## Task 6: Export Results

**Save results to file.**

```python
from src.notebooks import create_interface

interface = create_interface("6Y20B", environment="fixture")
df = interface.load_data()
results = interface.run_inference(df)

# Export to Excel
path = interface.export_results(results, format="excel")
print(f"Saved to: {path}")

# Or CSV
path_csv = interface.export_results(results, format="csv", name="my_results")
print(f"Saved to: {path_csv}")
```

**Formats**: `excel`, `csv`, `parquet`

---

## Task 7: Run Tests

**Validate your changes before committing.**

```bash
# Quick smoke test (30 seconds)
make quick-check

# Full test suite (2-3 minutes)
make test

# With coverage report
make coverage

# Run specific test file
pytest tests/unit/notebooks/test_interface.py -v

# Run tests matching pattern
pytest -k "test_validation" -v

# Run tests with print output visible
pytest -s tests/unit/notebooks/test_interface.py
```

---

## Task 8: Check for Data Leakage

**Verify no lag-0 competitors in features.**

### Manual check

```python
# Check if features contain forbidden patterns
from src.notebooks import create_interface

interface = create_interface("6Y20B", environment="fixture")

features_to_check = [
    "prudential_rate_current",
    "competitor_mid_t2",
    "competitor_mid_current",  # This should be flagged!
]

for f in features_to_check:
    is_forbidden = interface._is_competitor_lag_zero(f)
    status = "FORBIDDEN" if is_forbidden else "OK"
    print(f"{f}: {status}")
```

### Automated check via Makefile

```bash
# Run leakage audit
make pattern-check

# Or the full leakage gate
pytest tests/test_leakage_gates.py -v
```

### Pre-deployment checklist

See `knowledge/practices/LEAKAGE_CHECKLIST.md` for the full gate.

---

## Task 9: Create a Baseline for Testing

**Capture current outputs for regression testing.**

```python
import pandas as pd
from pathlib import Path
from src.notebooks import create_interface

# Run analysis
interface = create_interface("6Y20B", environment="fixture")
df = interface.load_data()
results = interface.run_inference(df)

# Save baseline
baseline_dir = Path("tests/baselines/rila")
baseline_dir.mkdir(parents=True, exist_ok=True)

# Save coefficients
coef_df = pd.DataFrame([results["coefficients"]])
coef_df.to_parquet(baseline_dir / "coefficients_baseline.parquet")

# Save model fit
fit_df = pd.DataFrame([results["model_fit"]])
fit_df.to_parquet(baseline_dir / "model_fit_baseline.parquet")

print(f"Baselines saved to {baseline_dir}")
```

### Compare against baseline

```python
# In tests
import pandas as pd

baseline = pd.read_parquet("tests/baselines/rila/coefficients_baseline.parquet")
current = pd.DataFrame([results["coefficients"]])

# Check equivalence at 1e-12 precision
pd.testing.assert_frame_equal(
    baseline, current, rtol=1e-12, atol=1e-12
)
```

---

## Task 10: Debug Constraint Violations

**When coefficient signs are wrong.**

### Step 1: Check the violation

```python
validation = interface.validate_coefficients(results["coefficients"])

for item in validation["violated"]:
    print(f"Feature: {item['feature']}")
    print(f"Coefficient: {item['coefficient']:.6f}")
    print(f"Expected: {item['expected']}, Got: {item['actual']}")
```

### Step 2: Common causes

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Positive competitor coefficient | Multicollinearity | Remove correlated features |
| Negative own-rate coefficient | Wrong feature or data issue | Check feature selection |
| Very large coefficients | Scale issues | Normalize features |
| Near-zero coefficients | Feature not predictive | Use different lags |

### Step 3: Check data quality

```python
# Check for outliers
print(df["prudential_rate_current"].describe())
print(df["competitor_mid_t2"].describe())

# Check for missing data
print(df.isnull().sum())

# Check correlation matrix
print(df[["prudential_rate_current", "competitor_mid_t2", "sales_target_current"]].corr())
```

### Step 4: Try different features

```python
# Try different lag
features_alt = [
    "prudential_rate_current",
    "competitor_mid_t3",  # Try lag 3 instead of lag 2
    "econ_treasury_5y_t1",
]

results_alt = interface.run_inference(df, features=features_alt)
validation_alt = interface.validate_coefficients(results_alt["coefficients"])
```

---

## Task 11: Run Forecasting

**Time series forecasting with benchmark comparison.**

```python
from src.notebooks import create_interface

interface = create_interface("6Y20B", environment="fixture")
df = interface.load_data()

# Run forecasting
forecasting_results = interface.run_forecasting(
    data=df,
    config={
        "bootstrap_samples": 500,
        "ridge_alpha": 1.0,
    }
)

# View results
print(f"Model MAPE: {forecasting_results.model_mape:.2%}")
print(f"Benchmark MAPE: {forecasting_results.benchmark_mape:.2%}")
print(f"Improvement: {forecasting_results.mape_improvement:.1%}")
```

---

## Task 12: Load Raw Data Separately

**When you need individual data sources.**

```python
from src.data.adapters import get_adapter
from pathlib import Path

# Create adapter directly
adapter = get_adapter("fixture", fixtures_dir=Path("tests/fixtures/rila"))

# Load individual datasets
sales_df = adapter.load_sales_data(product_filter="FlexGuard")
rates_df = adapter.load_competitive_rates(start_date="2022-01-01")
weights_df = adapter.load_market_weights()

print(f"Sales: {len(sales_df)} rows")
print(f"Rates: {len(rates_df)} rows")
print(f"Weights: {len(weights_df)} rows")
```

---

## Quick Troubleshooting

| Problem | Quick Fix |
|---------|-----------|
| "No methodology registered" | Check product code spelling |
| "Lag-0 competitor features detected" | Use `_t1`, `_t2` instead of `_current` |
| Import error | Run `pip install -e .` from repo root |
| Tests failing | Run `make quick-check` first |
| AWS credential error | See `AWS_SETUP_GUIDE.md` |

For more detailed debugging, see `TROUBLESHOOTING.md`.

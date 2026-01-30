# FIA Test Fixtures - Comprehensive Testing Package

**Created**: 2026-01-09
**Total Size**: 1.9MB (21 files)
**Purpose**: Enable TDD refactoring in environments without AWS access

---

## Fixture Inventory

### Core Data Fixtures (Phase 1.1-1.4)

| File | Size | Rows | Description |
|------|------|------|-------------|
| `test_tde_raw.parquet` | 216KB | 599 | Stratified sample of PruSecure sales contracts (2022-2025) |
| `test_wink_filtered.parquet` | 359KB | 5,898 | WINK competitor rates (FIA 5yr S&P 500, Annual PTP, PR=1.00) |
| `test_wink_pivot.parquet` | 61KB | 4,023 | Daily pivoted time series (2015-2026, 15 competitors) |
| `test_dgs5.parquet` | 260KB | 15,870 | 5-Year Treasury Constant Maturity Rate |
| `test_cpi.parquet` | 175KB | 6,293 | Consumer Price Index (scaled) |

### Notebook Context (Phase 1.5b)

| File | Size | Rows | Description |
|------|------|------|-------------|
| `test_df_FIA.parquet` | 27KB | 2,396 | Daily sales time series from notebook execution |
| `test_df_rates.parquet` | 67KB | 4,027 | Pivoted WINK rates with competitor summaries |

### Feature Engineering (Phase 1.5)

| File | Size | Shape | Description |
|------|------|-------|-------------|
| `test_features_full.parquet` | 205KB | 1,136 × 102 | Complete feature dataset with lags (0-14 days) |

**Features include:**
- Sales metrics (premium, count, smoothed)
- WINK competitor rates (15 companies)
- Macro indicators (DGS5, CPI adjustments)
- Lag features for all metrics
- Date range: 2022-11-10 to 2025-12-19

### Model Artifacts (Phase 1.6)

| File | Size | Description |
|------|------|-------------|
| `test_model_baseline.pkl` | 221KB | Trained RandomForest model (50 trees) |
| `baseline_train_predictions.parquet` | 14KB | Training set predictions (908 samples) |
| `baseline_test_predictions.parquet` | 5.2KB | Test set predictions (228 samples) |
| `baseline_feature_importance.parquet` | 4.1KB | Feature importance rankings |

**Model Performance:**
- Train R²: 0.7118, MSE: 2.37B
- Test R²: 0.4360, MSE: 6.52B
- Top feature: count_lag_0 (importance: 0.6610)

### Enhanced Baselines (Phase 5)

| File | Size | Rows | Purpose |
|------|------|------|---------|
| `baseline_tde_cleaned.parquet` | 31KB | 565 | Cleaned contracts (post sales_cleanup_v3) |
| `baseline_sales_daily.parquet` | 16KB | 1,150 | Daily aggregated sales |
| `baseline_sales_smoothed.parquet` | 24KB | 1,150 | 7-day rolling average |
| `baseline_merged_wink.parquet` | 38KB | 1,150 | Sales + WINK rates |
| `baseline_merged_macro.parquet` | 58KB | 1,150 | Full merged dataset |
| `baseline_stats_summary.parquet` | 6.3KB | 1 | Statistical summaries |
| `baseline_edge_high_premium.parquet` | 8.2KB | 10 | Top 10 highest premium contracts |
| `baseline_edge_low_premium.parquet` | 8.2KB | 10 | Bottom 10 lowest premium contracts |
| `baseline_correlation_matrix.parquet` | 25KB | 30×30 | Feature correlation matrix |

---

## Testing Use Cases

### 1. Unit Testing
Test individual functions with realistic data:
```python
import pandas as pd

# Test TDE cleanup
df_raw = pd.read_parquet('test_fixtures/test_tde_raw.parquet')
df_clean = sales_cleanup_v3(df_raw)
assert len(df_clean) == 565  # Expected after cleanup

# Test WINK pivoting
df_wink = pd.read_parquet('test_fixtures/test_wink_filtered.parquet')
df_pivot = time_series_pivot_wink(df_wink, product_ids, ...)
assert df_pivot.shape == (4027, 24)  # Expected shape
```

### 2. Integration Testing
Test full pipeline with intermediate checkpoints:
```python
# Load baseline at each stage
baseline_cleaned = pd.read_parquet('test_fixtures/baseline_tde_cleaned.parquet')
baseline_daily = pd.read_parquet('test_fixtures/baseline_sales_daily.parquet')
baseline_merged = pd.read_parquet('test_fixtures/baseline_merged_wink.parquet')

# Run refactored pipeline
result = run_full_pipeline(test_tde_raw)

# Compare at each stage
assert_frame_equal(result['cleaned'], baseline_cleaned)
assert_frame_equal(result['daily'], baseline_daily)
assert_frame_equal(result['merged'], baseline_merged)
```

### 3. Regression Testing
Ensure numerical stability (±0.01% tolerance):
```python
# Load golden predictions
baseline_pred = pd.read_parquet('test_fixtures/baseline_test_predictions.parquet')

# Run refactored model
model = pickle.load(open('test_fixtures/test_model_baseline.pkl', 'rb'))
new_pred = model['model'].predict(X_test)

# Check drift
drift = np.abs((new_pred - baseline_pred['predicted']) / baseline_pred['predicted'])
assert (drift < 0.0001).all()  # Max 0.01% drift
```

### 4. Edge Case Testing
Validate boundary conditions:
```python
# Load edge cases
high_premium = pd.read_parquet('test_fixtures/baseline_edge_high_premium.parquet')
low_premium = pd.read_parquet('test_fixtures/baseline_edge_low_premium.parquet')

# Test handling
for edge_case in [high_premium, low_premium]:
    result = process_contracts(edge_case)
    assert not result.isnull().any()  # No NaN introduced
    assert (result['premium'] >= 0).all()  # Valid premiums
```

### 5. Statistical Property Testing
Verify distributions remain stable:
```python
stats = pd.read_parquet('test_fixtures/baseline_stats_summary.parquet')

new_stats = calculate_stats(refactored_output)

assert abs(new_stats['premium_mean'] - stats['premium_mean']) < 100
assert abs(new_stats['premium_std'] - stats['premium_std']) < 50
assert new_stats['contract_count'] == stats['contract_count']
```

---

## Data Characteristics

### TDE Sales Sample
- **Stratification**: Top 5 firms (200 each) + 200 random
- **Date Range**: 2022-10-27 to 2025-12-19
- **Total Premium**: $50.3M across 565 contracts
- **Mean Premium**: $88,992 per contract
- **Firms**: 36 different firms represented
- **Products**: FIA7YR, FIA5YR, FIACA5YR, FIACA7YR

### WINK Rates
- **Filter Criteria**:
  - Product Type: Fixed Indexed
  - Index: S&P 500
  - Surrender Charge Duration: 5 years
  - Crediting Frequency: Annual
  - Indexing Method: Annual PTP
  - Participation Rate: 1.00
  - Premium Band: Excludes 0 and 250
- **Competitors**: 15 companies (Prudential + 14 competitors)
- **Date Range**: 2006-01-01 to 2026-01-05

### Feature Dataset
- **Temporal Coverage**: 1,136 days (2022-11-10 to 2025-12-19)
- **Feature Count**: 102 columns
  - 4 base metrics (premium, count, DGS5, competitor rates)
  - 15 lags per metric (0-14 days)
  - Smoothed versions
  - CPI adjustments
- **Missing Data**: 14 rows dropped (1.2%) due to lagging at start

---

## Important Notes

### Data Quality Flags
1. **Premium Reconciliation**: Only 44.63% match between summed and reported premiums
   - This is a known data quality issue in source data
   - Threshold trigger: <85% match rate
   - Tests should account for this discrepancy

2. **Outlier Removal**: Top 1% removed (6 contracts, $6.8M)
   - Threshold: >$515,998 premium
   - Reason: Extreme values affecting model stability

3. **Missing Competitors**: Eagle (2697) and Nationwide (1829) productIDs not found in filtered WINK data
   - Expected in notebook but excluded from fixtures
   - Tests should handle 15 competitors instead of 17

### AWS Dependencies Removed
These fixtures are **completely portable** - no AWS access required for testing. Original AWS paths for reference:
- TDE Sales: `s3://pruvpcaws031-east-isg-ie-lake/access/ierpt/tde_sales_by_product_by_fund/`
- WINK Rates: `s3://pruvpcaws031-east-isg-ie-lake/access/ierpt/wink_ann_product_rates/`
- Macro Data: `s3://cdo-annuity-364524684987-bucket/ANN_Price_Elasticity_Data_Science/MACRO_ECONOMIC_DATA/`

---

## Fixture Validation Checklist

Before using fixtures, verify:

```python
import pandas as pd
import pickle

# 1. All files present
expected_files = [
    'test_tde_raw.parquet', 'test_wink_filtered.parquet', 'test_wink_pivot.parquet',
    'test_dgs5.parquet', 'test_cpi.parquet', 'test_df_FIA.parquet', 'test_df_rates.parquet',
    'test_features_full.parquet', 'test_model_baseline.pkl',
    'baseline_tde_cleaned.parquet', 'baseline_sales_daily.parquet',
    # ... (all 21 files)
]
for f in expected_files:
    assert os.path.exists(f'test_fixtures/{f}'), f"Missing: {f}"

# 2. Data integrity
df = pd.read_parquet('test_fixtures/test_tde_raw.parquet')
assert len(df) == 599, "TDE raw should have 599 records"

df = pd.read_parquet('test_fixtures/test_features_full.parquet')
assert df.shape == (1136, 102), "Features should be 1136×102"

# 3. Model loadable
with open('test_fixtures/test_model_baseline.pkl', 'rb') as f:
    model_data = pickle.load(f)
    assert 'model' in model_data
    assert 'feature_cols' in model_data
    assert len(model_data['feature_cols']) == 78
```

---

## Next Steps

1. **Transfer**: Package fixtures for transfer to non-AWS environment
2. **Setup Tests**: Create pytest suite using these fixtures
3. **Refactor**: Modify code with confidence, running tests after each change
4. **Validate**: Return refactored code to AWS environment for final validation

**Estimated test runtime**: <10 seconds for full suite (no AWS calls)

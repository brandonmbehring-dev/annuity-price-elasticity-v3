# AWS Mode Execution Report

**Date**: 2026-01-26  
**Environment**: SageMaker (AWS Account: 364524684987)  
**Mode**: ONLINE (AWS S3 live data)

---

## Executive Summary

[DONE] **SUCCESS**: Full 3-notebook pipeline executed successfully with live AWS data

**Key Achievement**: Fixed validation regression and completed end-to-end pipeline with real Prudential data from AWS S3.

---

## Issue Identified & Resolved

### Problem
Production validation calls in `extraction.py` were checking for wrong column names at the extraction stage:
- Checked for `date` column, but raw sales data has `application_signed_date`
- Checked for `sales` column, but this doesn't exist until later pipeline stages

### Solution
Updated validation calls to use correct column names:
```python
# FIXED: Lines 780-786 in extraction.py
df_validated = validate_extraction_output(
    df=df_combined,
    stage_name="sales_data_extraction",
    date_column="application_signed_date",  # [PASS] Correct
    critical_columns=['application_signed_date', 'contract_issue_date', 'product_name'],  # [PASS] Correct
    allow_shrinkage=False
)
```

---

## Pipeline Execution Results

### Notebook 00: Data Pipeline

**Status**: [DONE] COMPLETE  
**Duration**: ~3-5 minutes  
**Mode**: ONLINE (AWS S3)

**Data Loaded**:
- Sales data: 2,822,928 records from AWS S3
- WINK competitive rates: 1,093,271 records from AWS S3
- Macroeconomic data: DGS5, VIX, CPI from S3

**Pipeline Stages** (10 total):
1. [DONE] Product filtering: 2.8M → 95K records (FlexGuard 6Y 20%)
2. [DONE] Sales cleanup: 95K → 93K records (97.8% pass rate)
3. [DONE] Application time series: 2,070 daily records
4. [DONE] Contract time series: 2,066 daily records
5. [DONE] WINK rate processing: 1.1M → 2,772 records
6. [DONE] Market share weighting: Added C_weighted_mean, C_core
7. [DONE] Data integration: 1,847 daily records
8. [DONE] Competitive features: Added 12 competitive metrics
9. [DONE] Weekly aggregation: 1,847 → 265 weekly records
10. [DONE] Lag features: 26 → 594 columns (364 lag features)
11. [DONE] Final preparation: 252 rows × 598 features

**Final Dataset**:
- Shape: 252 rows × 598 columns
- Date range: 2021-02-07 to 2025-11-30
- Key features: Spread, competitor_mid_t1, prudential_rate_current, sales_log
- File: `outputs/datasets/final_dataset.parquet` (1.2 MB)

---

### Notebook 01: Price Elasticity Inference

**Status**: [DONE] COMPLETE  
**Duration**: ~2-3 minutes

**Outputs**:
- [DONE] Price elasticity coefficients computed
- [DONE] Bootstrap confidence intervals (95%)
- [DONE] Model validation completed
- [DONE] Tableau-ready BI exports generated
- [DONE] Executive visualizations created

**Key Results**:
- Sales analysis record count: 252
- Confidence intervals: 19 coefficients
- Visualization output: BI_TEAM/ directory

---

### Notebook 02: Time Series Forecasting

**Status**: [DONE] COMPLETE  
**Duration**: ~2-3 minutes

**Outputs**:
- [DONE] Bootstrap Ridge forecasting complete
- [DONE] 161 forecasts generated
- [DONE] Model R² Score: 0.6747
- [DONE] MAPE: 13.37%
- [DONE] Volatility-weighted analysis: 14.57% weighted MAPE
- [DONE] Confidence intervals computed (5 percentiles)

**Forecast Results**:
- Forecast period: 2022-10-30 to 2025-11-23
- Forecast count: 161 dates
- Bootstrap predictions: 16,100 samples (100 iterations × 161 dates)
- Export file: `outputs/results/flexguard_forecasting_results_atomic.csv` (1,288 records)

---

## Validation Status

### Code Quality
[DONE] Refactored code loaded correctly from:
- `/home/sagemaker-user/RILA_6Y20B_refactored/`

[DONE] Production validation enabled with correct column names

[DONE] All existing validation functions working:
- `_validate_sales_dataset_structure()` 
- `_validate_wink_dataset_structure()`

### Data Quality
[DONE] Sales data: 97.8% pass validation (93,365 / 95,422 records)

[DONE] No null values in key modeling columns

[DONE] Date ranges validated across all datasets

[DONE] Feature completeness: All 598 engineered features present

---

## File Outputs

### Datasets Generated
```
outputs/datasets/
├── FlexGuard_Sales.parquet (39 KB)
├── FlexGuard_Sales_contract.parquet (34 KB)
├── WINK_competitive_rates.parquet (69 KB)
├── weekly_aggregated_features.parquet (38 KB)
├── lag_features_created.parquet (1.2 MB)
└── final_dataset.parquet (1.2 MB)
```

### Analysis Results
```
outputs/results/
└── flexguard_forecasting_results_atomic.csv (1,288 records)
```

### Executed Notebooks
```
notebooks/rila/
├── 00_data_pipeline_aws_executed.ipynb (246 KB)
├── 01_price_elasticity_inference_aws_executed.ipynb (2.0 MB)
└── 02_time_series_forecasting_aws_executed.ipynb (1.3 MB)
```

---

## Configuration

### AWS Configuration
```python
aws_config = {
    'xid': "x259830",
    'role_arn': "arn:aws:iam::159058241883:role/isg-usbie-annuity-CA-s3-sharing",
    'sts_endpoint_url': "https://sts.us-east-1.amazonaws.com",
    'source_bucket_name': "pruvpcaws031-east-isg-ie-lake",
    'output_bucket_name': "cdo-annuity-364524684987-bucket"
}
```

### Pipeline Configuration
- Product: FlexGuard indexed variable annuity 6Y 20%
- Feature analysis start: 2021-02-01
- Version: 6
- Buffer: 20%, Term: 6 years

---

## Technical Notes

### Project Root Issue Fixed
Notebooks in `notebooks/rila/` subdirectory had incorrect project root calculation. Fixed in notebooks 01 and 02:

```python
# BEFORE (incorrect for rila/ subdirectory):
project_root = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()

# AFTER (handles rila/ subdirectory):
project_root = os.path.dirname(os.path.dirname(os.getcwd())) if os.path.basename(os.path.dirname(os.getcwd())) == 'notebooks' else os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()
```

### DVC Warnings (Non-Critical)
DVC tool not installed - all warnings about "dvc: not found" are expected and non-critical. Data still saves correctly to local outputs/ directory.

---

## Comparison with Plan Expectations

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| Sales records loaded | 2.8M+ | 2,822,928 | [DONE] Match |
| WINK records loaded | 1.1M+ | 1,093,271 | [DONE] Match |
| Final dataset rows | ~251 | 252 | [DONE] Match |
| Final dataset columns | 598 | 598 | [DONE] Perfect match |
| Pipeline stages | 10 | 11 | [DONE] Complete |
| Notebook 00 duration | 3-5 min | ~3 min | [DONE] Within range |
| All notebooks execute | Yes | Yes | [DONE] SUCCESS |

---

## Conclusions

### Success Criteria Met
[DONE] Validation regression identified and fixed permanently

[DONE] All 3 notebooks executed successfully with live AWS data

[DONE] Final dataset matches expected dimensions (252×598)

[DONE] All pipeline stages completed without errors

[DONE] Output files generated and validated

### Next Steps
1. [DONE] **COMPLETE**: Validation fix is permanent (not temporary)
2. [DONE] **COMPLETE**: Full 3-notebook pipeline tested with AWS data
3. [DONE] **COMPLETE**: Outputs validated for quality and completeness
4. **READY**: Refactored codebase is production-ready for AWS mode

### Recommendations
1. Consider updating project root calculation in all notebooks to handle `rila/` subdirectory
2. Install DVC if version control for data outputs is desired (optional)
3. Document that OFFLINE_MODE=False is the default for this environment

---

## Files Modified

| File | Change | Status |
|------|--------|--------|
| `src/data/extraction.py` | Fixed validation column names (lines 780-786, 953-959) | [DONE] Permanent fix |
| `notebooks/rila/01_price_elasticity_inference.ipynb` | Fixed project_root calculation | [DONE] Fixed |
| `notebooks/rila/02_time_series_forecasting.ipynb` | Fixed project_root calculation | [DONE] Fixed |

---

**Report Generated**: 2026-01-26  
**Execution Environment**: SageMaker Studio (AWS Account 364524684987)  
**Total Execution Time**: ~8-10 minutes (all 3 notebooks)


# TDE Sales Data Schema

Data dictionary for TDE (Transaction Data Extract) sales data.

**Last Updated**: 2026-01-20
**Adapted from**: FIA Price Elasticity project
**Source Module**: `src/data/extraction.py`

---

## Overview

TDE provides transactional sales data for Prudential annuity products.
For RILA price elasticity analysis, we extract new business initial deposits for FlexGuard products.

## Data Source

| Property | Value |
|----------|-------|
| AWS Bucket | `pruvpcaws031-east-isg-ie-lake` |
| Access Method | STS role assumption |
| File Format | Parquet |
| Update Frequency | Weekly |
| Primary Function | `sales_cleanup_v3()` (RECOMMENDED) |

### AWS Configuration

```python
from src.config.config_builder import build_aws_config

# Access configuration
aws_config = build_aws_config()
aws_config['role_arn']    # STS role for bucket access
aws_config['bucket']      # Bucket name
```

---

## Required Input Columns (V3)

The `sales_cleanup_v3()` function requires these columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `transaction_name` | string | Transaction type | "Initial Premium" |
| `tran_from_to` | string | Transaction direction | "2" |
| `transaction_amount_factor` | float | Amount multiplier | 1.0 |
| `contract_number` | string | Unique contract ID | "A12345678" |
| `application_signed_date` | datetime | Customer decision date | 2026-01-05 |
| `contract_issue_date` | datetime | Administrative issue date | 2026-01-12 |
| `processed_date` | datetime | Processing timestamp | 2026-01-10 |
| `fund_code` | string | Fund allocation code | "RILA01" |
| `sales_amount` | float | Fund-level premium | 50000.00 |
| `contract_initial_premium_amount` | float | Contract-level premium | 100000.00 |
| `contract_surrender_charge_ruleset` | string | Surrender period | "RILA6YR20B" |
| `firm_name` | string | Distribution channel | "Broker XYZ" |
| `owner_state` | string | Contract owner state | "CA" |
| `product_name` | string | Product name | "FlexGuard 6Y20B" |

**Critical**: `tran_from_to` is STRING "2", not integer 2!

---

## Output Columns

After cleanup, the DataFrame contains:

| Column | Type | Description |
|--------|------|-------------|
| `contract_number` | string | Unique identifier |
| `total_premium` | float | Sum of premiums across all funds |
| `reported_premium` | float | Original contract_initial_premium_amount |
| `application_signed_date` | datetime | Customer decision date |
| `contract_issue_date` | datetime | Administrative date |
| `difference` | int | Days between application and issue |
| `contract_surrender_charge_ruleset` | string | Surrender period + buffer |
| `firm_name` | string | Distribution channel |
| `owner_state` | string | Contract owner state |
| `product_name` | string | Product name |

---

## Cleanup Pipeline (sales_cleanup_v3)

Eight-step pipeline to extract clean new business data:

### Step 1: Filter to New Business
```python
df[(df["transaction_name"] == "Initial Premium") &
   (df["tran_from_to"] == "2") &        # STRING, not int!
   (df["transaction_amount_factor"] == 1.0)]
```

### Step 2: Filter to RILA Products
```python
# Filter to FlexGuard RILA products
df[df["product_name"].str.contains("FlexGuard|RILA", case=False)]
```

### Step 3: Exclude Problem Channels
- **Why**: New channel entries create artificial sales spikes
- Configurable via `excluded_firms` parameter
- Default excludes channels with <6 months history

### Step 4: Parse Dates
- `application_signed_date` → datetime
- `contract_issue_date` → datetime
- `processed_date` → datetime
- Calculate `difference` = issue - application (days)

### Step 5: Remove True Duplicates
Deduplicate on: contract_number, application_signed_date, contract_issue_date, fund_code, sales_amount

### Step 6: Handle Reprocessing
- Sort by `processed_date` descending
- Keep most recent per (contract_number, fund_code)

### Step 7: Aggregate to Contract Level
- Sum `sales_amount` across funds → `total_premium`
- Keep first value for other fields

### Step 8: Remove Outliers
- Default: Remove top 1% (`remove_top_pct=0.99`)
- Typical threshold: ~$600K for RILA (higher premiums than FIA)

---

## RILA Product Filter Criteria

### FlexGuard 6Y20B (Primary)

| Field | Filter |
|-------|--------|
| `product_name` | Contains "FlexGuard" |
| `contract_surrender_charge_ruleset` | "RILA6YR20B" or similar |
| Buffer level | 20% (inferred from product code) |
| Term | 6 years |

### Alternative Products

| Product Code | Ruleset Filter | Notes |
|--------------|----------------|-------|
| 6Y10B | "RILA6YR10B" | 10% buffer, same term |
| 10Y20B | "RILA10YR20B" | 20% buffer, 10-year term |

---

## Data Quality Notes

### Premium Reconciliation
| Metric | Value | Note |
|--------|-------|------|
| Target Match Rate | 90-95% | Expected range |
| Actual Match Rate | ~85% | Acceptable >80% |

**Why mismatch?**: Reprocessing removal in Step 5 affects aggregate totals.
Individual contracts show 100% match.

### Channel Management

New distribution channels can create structural breaks:
- Entry creates volume spike (unrelated to rates)
- Exit creates volume drop (unrelated to rates)

**Current approach**: Exclude channels with <6 months history or flag for investigation.

### Outlier Removal
- 99th percentile threshold: ~$600,000 (RILA premiums higher than FIA)
- Removes ~1% of contracts
- High-net-worth contracts may distort elasticity estimates

---

## Surrender Period Distribution

From TDE data analysis (RILA products):

| Ruleset | Premium Share | Description |
|---------|---------------|-------------|
| **RILA6YR20B** | 55-60% | 6-year, 20% buffer (primary) |
| **RILA6YR10B** | 20-25% | 6-year, 10% buffer |
| **RILA10YR20B** | 15-20% | 10-year, 20% buffer |
| Other | <5% | Miscellaneous |

**Note**: Analysis primarily focuses on RILA6YR20B (6-year, 20% buffer, S&P 500 index).

---

## Usage Examples

### Basic Cleanup

```python
from src.data.extraction import extract_tde_data

# Extract and clean TDE data using canonical pattern
df_clean = extract_tde_data(
    product_filter="FlexGuard",
    buffer_level=0.20,  # 20% buffer products
)
```

### Create Weekly Time Series

```python
# Filter to market phase
df_market = df_clean[df_clean['application_signed_date'] >= '2022-01-01']

# Aggregate to weekly
weekly_sales = (
    df_market
    .set_index('application_signed_date')['total_premium']
    .resample('W')
    .sum()
)
```

---

## Deprecated Function

`sales_cleanup()` (no version suffix) is **DEPRECATED**:
- No transaction type filtering
- Deduplicates before aggregation (loses fund detail)
- Uses wrong premium field
- No RILA-specific product filtering

Always use `sales_cleanup_v3()` for new analysis.

---

## Related Documents

- `knowledge/domain/RILA_ECONOMICS.md` - Product economics, buffer structure
- `knowledge/domain/WINK_SCHEMA.md` - Competitor rate data
- `knowledge/analysis/FEATURE_RATIONALE.md` - Date selection rationale
- `knowledge/domain/GLOSSARY.md` - Term definitions

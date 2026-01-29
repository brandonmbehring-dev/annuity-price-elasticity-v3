# WINK Rate Data Schema

Data dictionary for competitor cap rate data from WINK.

**Last Updated**: 2026-01-20
**Adapted from**: FIA Price Elasticity project
**Source Module**: `src/data/extraction.py`, `src/data/preprocessing.py`

---

## Overview

WINK provides competitive intelligence data on annuity rates across the industry.
For RILA price elasticity analysis, we extract cap rates for S&P 500 indexed products with buffer protection.

## Data Source

| Property | Value |
|----------|-------|
| AWS Bucket | `pruvpcaws095-east-...` |
| Access Method | STS role assumption |
| File Format | Parquet |
| Update Frequency | Weekly |
| Primary Function | `download_from_095()` |

### AWS Configuration

```python
from src.config.config_builder import build_aws_config

# Access configuration
aws_config = build_aws_config()
aws_config['role_arn']    # STS role for bucket access
aws_config['bucket']      # Bucket name
```

Environment variable overrides:
- `WINK_ROLE_ARN`: Override role ARN
- `WINK_BUCKET`: Override bucket name
- `STS_ENDPOINT`: Override STS endpoint

---

## Raw Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `effectiveDate` | string | Original date field | "2026-01-05" |
| `date` | datetime | Parsed datetime (added by loader) | 2026-01-05 |
| `productID` | string | Unique product identifier | "4521" |
| `companyName` | string | Carrier name | "Prudential" |
| `productName` | string | Product name | "FlexGuard" |
| `capRate` | float | Maximum credited rate (decimal) | 0.155 (15.5%) |
| `participationRate` | float | Index participation rate (decimal) | 1.00 (100%) |
| `bufferLevel` | float | Buffer protection level (decimal) | 0.20 (20%) |
| `bufferModifier` | string | Buffer type indicator | "standard", "enhanced" |

---

## RILA 6Y20B Filter Criteria

For RILA price elasticity, data is filtered to FlexGuard 6Y20B products:

| Criterion | Value | Rationale |
|-----------|-------|-----------|
| Product Type | "RILA" / "Registered Index-Linked" | RILA only, not FIA or MYGA |
| Index | "S&P 500" | Most competitive, most liquid |
| Surrender Duration | "6" years | Primary product focus (6Y20B) |
| Buffer Level | 20% | 20% buffer products |
| Crediting Frequency | "Annual" | Standard comparison basis |
| Indexing Method | "Annual PTP" | Point-to-point crediting |
| Participation Rate | ≥100% | Competitive products only |
| Premium Band | Excludes 0, 250 | Standard premium tiers |

### Alternative Product Configurations

| Product Code | Term | Buffer | Filter Change |
|--------------|------|--------|---------------|
| **6Y20B** (primary) | 6 years | 20% | As above |
| **6Y10B** | 6 years | 10% | Buffer = 10% |
| **10Y20B** | 10 years | 20% | Term = 10 years |

**Important**: Different buffer levels attract different buyer segments (risk tolerance varies).

---

## Derived Fields

Created by `time_series_pivot_wink()` and `add_summary_and_rank()`:

### Rate Columns (Per Product)

| Field Pattern | Type | Description |
|---------------|------|-------------|
| `[ProductName]` | float | Cap rate (%) for specific product |
| `Pru` | float | Prudential (FlexGuard) cap rate (primary product) |

### Summary Statistics

| Field | Calculation | Purpose |
|-------|-------------|---------|
| `raw_mean` | Mean of all rate columns | Market average |
| `raw_median` | Median of all rate columns | Market median |

### Competitor Rankings

| Field | Definition | Purpose |
|-------|------------|---------|
| `first_highest_benefit` | Top competitor rate | Best competitive option |
| `second_highest_benefit` | 2nd highest rate | Competitive context |
| `third_highest_benefit` | 3rd highest rate | Competitive context |

### Tier Averages

| Field | Definition | Note |
|-------|------------|------|
| `top_5` | Mean of top 5 competitor rates | Context measure |
| `top_7` | Mean of top 7 competitor rates | Broader market view |
| `top_10` | Mean of top 10 competitor rates | Full market view |
| `C_weighted_mean` | Market-share weighted competitor mean | **Primary for RILA** |

**Critical Difference from FIA**: RILA uses market-share weighted means for competitor aggregation, while FIA uses simple arithmetic means. The "Weighted Mean" label is accurate for RILA.

---

## Market-Share Weighting (RILA-Specific)

Unlike FIA which uses simple top-N means, RILA competitor rates are weighted by market share:

```python
C_weighted_mean = sum(rate_i * market_share_i) / sum(market_share_i)
```

**Rationale**: RILA market is more concentrated; weighting by share better reflects competitive pressure from large players (Allianz, Lincoln, Athene).

### Excluded Carriers

Certain carriers are excluded from competitor calculations:
- **Allianz**: Dominates market share; would overwhelm weighted mean
- **Trans**: Data quality issues

See `COMPETITIVE_ANALYSIS.md` for detailed exclusion rationale.

---

## Data Quality Notes

### Forward-Fill Behavior
- Missing dates are forward-filled from last known rate
- Rates converted from decimal to percentage (× 100)

### Rolling Average
- Optional smoothing via `rolling` parameter (default=1, no smoothing)
- Production typically uses 7-day rolling mean

### Date Range
- Daily granularity from start_date to end_date
- Index: pd.date_range with freq="d"

### Known Issues
- Some carriers report rates inconsistently
- Late rate updates may cause data lag
- Holiday periods may have stale rates (forward-filled)
- Buffer level may not be consistently reported across all carriers

### Data Validation Checks

```python
# Validate WINK data quality
assert df['capRate'].max() <= 0.30, "Cap rate outlier (>30%)"
assert df['bufferLevel'].isin([0.10, 0.15, 0.20, 0.25]).all(), "Unexpected buffer level"
assert df['participationRate'].max() <= 3.0, "Participation rate outlier (>300%)"
```

---

## Usage Examples

### Download Raw Data

```python
from src.data.extraction import extract_wink_data

df_wink = extract_wink_data()
# Returns: DataFrame with effectiveDate, date, productID, companyName, productName, capRate, bufferLevel
```

### Create Time Series

```python
from src.data.preprocessing import process_wink_time_series

# Process WINK data into time series format
df_ts = process_wink_time_series(
    df_wink=df_wink,
    start_date="2022-01-01",
    end_date="2024-12-31",
    rolling=7,  # 7-day rolling average
    use_market_share_weights=True,  # RILA uses weighted means
)
```

---

## Related Documents

- `knowledge/domain/RILA_ECONOMICS.md` - Product economics
- `knowledge/domain/CREDITING_METHODS.md` - Cap rate mechanics
- `knowledge/analysis/FEATURE_RATIONALE.md` - Why features are constructed this way
- `knowledge/domain/COMPETITIVE_ANALYSIS.md` - Competitor selection and weighting

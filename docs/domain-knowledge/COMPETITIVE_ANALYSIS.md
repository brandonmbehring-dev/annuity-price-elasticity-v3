# Competitive Analysis: RILA Market Positioning

**Tier**: [T2] Empirical (WINK data)
**Last Updated**: 2026-01-20
**Source**: WINK Data, Internal Analysis, annuity-pricing project

---

## Overview

Competitive analysis determines where a product's rate falls within the market distribution. For RILA, this includes:
- Rate positioning (percentiles, quartiles)
- Company rankings
- Spread-over-Treasury calculations
- **Market-share weighted aggregations** (RILA-specific)

---

## Rate Positioning [T2]

### Percentile Calculation

```python
percentile = (count_below / total_count) * 100
```

Where:
- `count_below`: Number of comparable products with rates below the target rate
- `total_count`: Total number of comparable products

### Quartile Classification

| Quartile | Percentile Range | Label |
|----------|-----------------|-------|
| 1 | 75-100 | Top Quartile |
| 2 | 50-75 | Above Median |
| 3 | 25-50 | Below Median |
| 4 | 0-25 | Bottom Quartile |

**Note**: Higher percentile = more competitive rate.

---

## Market-Share Weighted Aggregation (RILA-Specific)

### Why RILA Uses Weighted Means

Unlike FIA (which uses simple top-N means), RILA uses market-share weighted competitor aggregations.

**Rationale**: The RILA market is more concentrated than FIA:
- Top 3 RILA carriers: ~60% market share
- Top 3 FIA carriers: ~35% market share

Market-share weighting better reflects competitive pressure from dominant players.

### Calculation

```python
def market_share_weighted_mean(
    rates: pd.Series,
    market_shares: pd.Series
) -> float:
    """
    Calculate market-share weighted competitor mean.

    Args:
        rates: Cap rates by company
        market_shares: Market share by company (sum to 1.0)

    Returns:
        Weighted mean rate
    """
    return (rates * market_shares).sum() / market_shares.sum()
```

### Example

| Company | Cap Rate | Market Share | Weighted Contribution |
|---------|----------|--------------|----------------------|
| Lincoln | 15.0% | 25% | 3.75% |
| Athene | 14.5% | 20% | 2.90% |
| Allianz | 14.0% | 35% | (excluded) |
| Others | 15.5% | 20% | 3.10% |
| **Weighted Mean** | | | **14.84%** |

---

## Excluded Carriers

Certain carriers are excluded from competitor calculations:

### Allianz Exclusion

**Reason**: Market dominance would overwhelm weighted mean.

| Metric | Allianz | Rest of Market |
|--------|---------|----------------|
| RILA Market Share | ~35% | ~65% |
| Rate Behavior | Often outlier | More consistent |

**Impact if included**: Allianz would drive ~35% of weighted mean, masking competitive dynamics among other carriers.

**Alternative approach**: Include Allianz as separate indicator variable rather than in competitor mean.

### Trans Exclusion

**Reason**: Data quality issues.

| Issue | Description |
|-------|-------------|
| Stale rates | Rates not updated regularly |
| Product coverage | Incomplete product lineup in WINK |

---

## Spread Over Treasury [T1]

### Definition

```
Spread = Product_Rate - Treasury_Rate
```

For RILA, spreads are typically calculated against the matching Treasury duration:

| Product Term | Treasury Benchmark |
|-------------|-------------------|
| 6-year | 5-year or 7-year (interpolate) |
| 10-year | 10-year Treasury |

### Interpretation [T2]

| Spread Range | Interpretation |
|-------------|----------------|
| > 800 bps | Aggressive pricing (RILA caps are high) |
| 600-800 bps | Competitive |
| 400-600 bps | Conservative |
| < 400 bps | Uncompetitive |

**Note**: RILA spreads are much higher than MYGA spreads because caps reflect option value, not just interest margin.

---

## Company Rankings [T2]

### Ranking Metrics

1. **Rate Rank**: Direct comparison of offered caps
2. **Spread Rank**: Comparison of caps over Treasury
3. **Buffer-Adjusted Rank**: Caps normalized for buffer level

### Buffer-Adjusted Comparison

Because caps vary with buffer level, normalize before comparing:

```python
def buffer_adjusted_cap(cap: float, buffer: float) -> float:
    """
    Normalize cap for buffer level.

    Higher buffer = lower caps expected.
    Normalize to 20% buffer equivalent.
    """
    buffer_adjustment = {
        0.10: +0.03,  # 10% buffer: add 3% to normalize
        0.15: +0.015,  # 15% buffer: add 1.5%
        0.20: 0.0,     # 20% buffer: baseline
        0.25: -0.02,   # 25% buffer: subtract 2%
    }
    return cap + buffer_adjustment.get(buffer, 0.0)
```

---

## RILA Competitive Landscape

### Major Competitors (2024-2025)

| Company | Approx. Market Share | Rate Strategy |
|---------|---------------------|---------------|
| **Allianz** | ~35% | Consistent, high-volume |
| **Lincoln** | ~15% | Competitive, advisor-focused |
| **Athene** | ~12% | Aggressive pricing |
| **AIG** | ~8% | Moderate, stable |
| **Nationwide** | ~7% | Conservative |
| **Prudential** | ~6% | Our product |
| Others | ~17% | Varied |

### FlexGuard Positioning

| Metric | FlexGuard 6Y20B | Market Average |
|--------|-----------------|----------------|
| Cap Rate | ~15% | ~14.5% |
| Percentile | ~65th | 50th |
| Buffer | 20% | 20% (mode) |

---

## WINK Data Integration [T2]

### Key Columns for RILA Competitive Analysis

| Column | Purpose |
|--------|---------|
| `cap_rate` | Cap for crediting |
| `participation_rate` | Participation multiplier |
| `buffer_rate` | Buffer level |
| `buffer_modifier` | Buffer type (standard/enhanced) |
| `index_used` | Index (S&P 500, etc.) |
| `surrender_years` | Product term |

### Data Quality Checks

Before analysis, validate:
1. Cap rates typically 10-25% for RILA
2. Buffer rates typically 10-25%
3. Participation rates typically 80-200%
4. No negative values (except floor rates)

---

## Filtering Dimensions

| Dimension | Values | WINK Column |
|-----------|--------|-------------|
| Product Type | RILA | `product_type` |
| Term | 6, 10 years | `surrender_years` |
| Buffer | 10%, 15%, 20%, 25% | `buffer_rate` |
| Index | S&P 500, Russell, etc. | `index_used` |
| Premium Band | $10K, $100K, $250K | `premium_band` |

---

## Implementation Notes

### Competitor List Construction

```python
# RILA competitor list (excluding Allianz, Trans)
RILA_COMPETITORS = [
    "Lincoln",
    "Athene",
    "AIG",
    "Nationwide",
    "Pacific",
    "Brighthouse",
    "Equitable",
    "Principal",
]

# Exclusions
EXCLUDED_FROM_WEIGHTED_MEAN = ["Allianz", "Trans"]
```

### Market Share Data Source

Market shares from LIMRA quarterly reports or internal sales intelligence.

```python
MARKET_SHARES = {
    "Lincoln": 0.20,
    "Athene": 0.18,
    "AIG": 0.12,
    "Nationwide": 0.10,
    "Pacific": 0.10,
    "Brighthouse": 0.10,
    "Equitable": 0.10,
    "Principal": 0.10,
}
# Note: Sums to 1.0 after Allianz exclusion
```

---

## Usage Example

```python
# Calculate competitive positioning using data module
from src.data.preprocessing import calculate_competitive_position

# Calculate where 15.0% cap falls in RILA 6Y20B market
result = calculate_competitive_position(
    rate=0.15,
    wink_data=df,
    use_weighted=True,  # RILA uses weighted means
    exclude_carriers=["Allianz", "Trans"],
)

print(f"Percentile: {result['percentile']}th")
print(f"Weighted Competitor Mean: {result['weighted_mean']:.2%}")
# Output: Percentile: 65th, Weighted Mean: 14.50%
```

---

## Related Documents

- `knowledge/domain/WINK_SCHEMA.md` - WINK data dictionary
- `knowledge/domain/RILA_MECHANICS_DEEP.md` - Product mechanics
- `knowledge/analysis/FEATURE_RATIONALE.md` - Why market-share weighting
- `knowledge/integration/CROSS_PRODUCT_COMPARISON.md` - FIA vs RILA differences

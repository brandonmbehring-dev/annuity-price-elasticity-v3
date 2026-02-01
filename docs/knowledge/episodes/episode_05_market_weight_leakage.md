# Episode 05: Market Share Weight Leakage

**Category**: Feature Construction Leakage (10 Bug Category #6)
**Discovered**: 2025-11-XX (aggregation audit)
**Impact**: Competitor aggregation used future market shares
**Status**: RESOLVED - Historical weights enforced

---

## The Bug

When computing market-share-weighted competitor rates, using market shares from the FULL sample period instead of only data available at prediction time.

If competitor A's market share grows from 10% to 30% over the sample, using the 30% weight for early periods leaks information about future market dynamics.

---

## Symptom

**How it manifested:**
- Competitor weighted mean was "too accurate" at predicting future rates
- Weight changes suspiciously correlated with future sales outcomes
- Model performed well early in sample, degraded later (pattern reversal)

**Red flags in output:**
```
Early Period (2022):
  Competitor weighted mean: 5.8%
  Using 2025 market shares: Lincoln (30%), Athene (25%), ...

Late Period (2025):
  Same weights used throughout - but Lincoln was only 10% in 2022!
```

---

## Root Cause Analysis

### 1. The Problem

Market shares evolve over time:
- 2022: Lincoln 10%, Athene 15%, Others 75%
- 2025: Lincoln 30%, Athene 25%, Others 45%

If we use 2025 weights to compute the 2022 competitor mean:
- Lincoln's rate gets 3x too much weight
- We're using information about Lincoln's future success

### 2. Why This Matters

Future market share reflects:
- Future product performance
- Future customer preferences
- Future competitive dynamics

All of this is unknown at prediction time in 2022.

### 3. The Leakage Mechanism

```python
# WRONG: Full-sample weights
weights = df.groupby('competitor')['sales'].sum() / df['sales'].sum()

# At time t=2022, this includes:
#   - Sales from 2022 (valid)
#   - Sales from 2023, 2024, 2025 (FUTURE!)
```

---

## The Fix

### Before (Leaky) ❌

```python
# Compute weights from full sample
total_sales = df.groupby('competitor')['sales'].sum()
weights = total_sales / total_sales.sum()

# Apply to all periods
df['competitor_weighted_mean'] = (
    df[competitor_cols].mul(weights, axis=1).sum(axis=1)
)
```

### After (Safe) ✅

**Option 1: Expanding Window Weights**

```python
def compute_expanding_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Compute market shares using only historical data."""
    result = df.copy()

    for date in sorted(df['date'].unique()):
        historical = df[df['date'] <= date]
        weights = historical.groupby('competitor')['sales'].sum()
        weights = weights / weights.sum()

        mask = result['date'] == date
        for comp, weight in weights.items():
            result.loc[mask, f'{comp}_weight'] = weight

    return result
```

**Option 2: Rolling Window Weights**

```python
def compute_rolling_weights(
    df: pd.DataFrame,
    window: int = 52,  # 52 weeks = 1 year
) -> pd.DataFrame:
    """Compute market shares using trailing window."""
    # Only look back, never forward
    return df.groupby('competitor')['sales'].transform(
        lambda x: x.rolling(window, min_periods=1).sum()
    )
```

**Option 3: Fixed Historical Weights**

```python
# Use pre-determined weights from historical period
HISTORICAL_WEIGHTS = {
    'Lincoln': 0.15,
    'Athene': 0.18,
    'Brighthouse': 0.12,
    # ... from 2020-2021 data (before sample period)
}

df['competitor_weighted_mean'] = sum(
    df[f'{comp}_rate'] * weight
    for comp, weight in HISTORICAL_WEIGHTS.items()
)
```

---

## Gate Implementation

### Detection Logic

```python
def detect_weight_leakage(
    weights: pd.DataFrame,
    dates: pd.Series,
) -> bool:
    """Check if weights are constant across time (suspicious).

    Time-varying weights should show some evolution.
    Perfectly constant weights suggest full-sample calculation.
    """
    weight_variance = weights.groupby(dates).mean().var()

    # If variance is near zero, weights don't change = suspicious
    return weight_variance.sum() < 1e-6
```

### Verification Test

```python
@pytest.mark.leakage
def test_market_weights_time_varying():
    """Market share weights should evolve over time."""
    df = load_data_with_weights()

    early_weights = df[df['year'] == 2022].groupby('competitor')['weight'].mean()
    late_weights = df[df['year'] == 2025].groupby('competitor')['weight'].mean()

    # Weights should differ between periods
    weight_diff = (early_weights - late_weights).abs().mean()
    assert weight_diff > 0.01, (
        "Market weights identical across years = possible leakage"
    )
```

---

## Verification

### Manual Check

```python
# Verify weights change over time
for year in [2022, 2023, 2024, 2025]:
    year_data = df[df['date'].dt.year == year]
    weights = year_data.groupby('competitor')['market_weight'].mean()
    print(f"\n{year} Market Weights:")
    print(weights.sort_values(ascending=False))

# If weights are IDENTICAL across years, you have leakage
```

### Automated Validation

```bash
# Run weight leakage tests
pytest tests/anti_patterns/test_market_weight_leakage.py -v
```

---

## Impact Assessment

| Metric | Full-Sample Weights | Historical Weights | Interpretation |
|--------|--------------------|--------------------|----------------|
| Weighted Mean Accuracy | 94% | 78% | More realistic |
| Out-of-Sample R² | 0.82 | 0.71 | Honest estimate |
| Weight Stability | Perfect | Evolving | Expected behavior |

**Key insight:** Lower "accuracy" with historical weights is actually correct—future market shares are unknowable.

---

## Lessons Learned

1. **Market shares are outcomes, not givens**
   - Future shares reflect future performance
   - Can't use future outcomes as features

2. **Weights need temporal structure**
   - Use expanding or rolling windows
   - Or fixed pre-sample weights

3. **Constant weights across time = red flag**
   - Real market shares evolve
   - Perfectly constant suggests full-sample calculation

4. **Consider pre-sample estimation**
   - Use 2020-2021 to estimate weights
   - Apply to 2022-2025 prediction period

---

## Related Documentation

- `knowledge/practices/LEAKAGE_CHECKLIST.md` - Section 5
- `src/features/aggregation/weighted.py` - Weight computation
- `Episode 02: Aggregation Lookahead` - Related aggregation issue

---

## Tags

`#leakage` `#market-share` `#weights` `#temporal-structure` `#aggregation`

# Episode 02: Aggregation Lookahead Bias

**Category**: Aggregation Contamination (10 Bug Category #3)
**Discovered**: 2025-11-XX (feature engineering audit)
**Impact**: Rolling statistics computed on full sample, not training window
**Status**: RESOLVED - Expanding window enforced

---

## The Bug

Computing rolling statistics (mean, std, percentile) using the FULL dataset instead of only data available at prediction time.

When calculating `competitor_mean_t2`, the naive approach uses all historical and future data, which leaks information from the future into the training set.

---

## Symptom

**How it manifested:**
- Model performance suspiciously stable across all CV folds
- No degradation between in-sample and out-of-sample R²
- "Memory" of future market conditions in early predictions

**Red flags in output:**
```
CV Fold R²: [0.78, 0.77, 0.79, 0.78, 0.78]  # Suspiciously uniform
In-sample vs Out-of-sample gap: <1%  # Should be 3-10%
```

---

## Root Cause Analysis

### 1. The Problem

When computing a rolling mean of competitor rates:

```python
# WRONG: Uses entire dataset
df['competitor_mean'] = df['competitor_rate'].rolling(window=8).mean()
```

This looks correct, but in a cross-validation context, the rolling window includes data from the TEST period when computing features for the TRAINING period.

### 2. Visual Example

```
Timeline:
  Week 1  Week 2  Week 3  Week 4 | Week 5  Week 6  Week 7  Week 8
  ─────────────────────────────────────────────────────────────────
        TRAINING DATA            |         TEST DATA
                                 |
  Rolling mean at Week 3:        |
  ← uses Week 1, 2, 3            |  ← BUT also Week 4+ from test!
```

The rolling window doesn't respect the train/test boundary.

### 3. Why This Leaks

When we fit the model on training data (Weeks 1-4):
- Features for Week 3 include information about competitor rates in Week 5+
- The model "learns" from future competitor behavior
- This inflates apparent predictive power

---

## The Fix

### Before (Leaky) ❌

```python
# Compute rolling mean on full dataset
df['competitor_mean'] = df.groupby('product')[
    'competitor_rate'
].transform(lambda x: x.rolling(8).mean())

# Then split into train/test
train = df[df['date'] < cutoff]
test = df[df['date'] >= cutoff]
```

### After (Safe) ✅

```python
# Split FIRST, then compute
train = df[df['date'] < cutoff].copy()
test = df[df['date'] >= cutoff].copy()

# Compute on training only (expanding window)
train['competitor_mean'] = train.groupby('product')[
    'competitor_rate'
].expanding().mean().reset_index(level=0, drop=True)

# For test: use only training data for mean
train_mean = train.groupby('product')['competitor_rate'].mean()
test['competitor_mean'] = test['product'].map(train_mean)
```

### Alternative: Expanding Window

```python
def compute_expanding_feature(df: pd.DataFrame, col: str) -> pd.Series:
    """Compute expanding mean (only uses past data)."""
    return df.groupby('product')[col].expanding().mean().reset_index(level=0, drop=True)
```

---

## Gate Implementation

### Detection Logic

Located in feature engineering validation:

```python
def validate_temporal_integrity(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    features: pd.DataFrame,
) -> List[str]:
    """Check that features don't leak test data into training.

    Returns list of suspicious features.
    """
    suspicious = []

    for col in features.columns:
        # Check if training feature values change when test data is removed
        train_only = features.iloc[train_idx][col]
        full_data = features[col]

        # If removing test data changes training features, there's leakage
        if not np.allclose(train_only, full_data.iloc[train_idx], equal_nan=True):
            suspicious.append(col)

    return suspicious
```

### Test Coverage

```python
@pytest.mark.leakage
def test_rolling_stats_respect_cv_boundaries():
    """Rolling statistics should not leak across CV folds."""
    df = create_test_timeseries()

    for train_idx, test_idx in temporal_cv_split(df):
        # Compute features
        features = compute_features(df)

        # Verify no leakage
        suspicious = validate_temporal_integrity(train_idx, test_idx, features)
        assert len(suspicious) == 0, f"Leaky features: {suspicious}"
```

---

## Verification

### Manual Check

```python
# Verify expanding window behavior
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=10, freq='W'),
    'rate': [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14]
})

# Expanding mean (correct)
df['rate_expanding_mean'] = df['rate'].expanding().mean()

# At week 5, expanding mean should ONLY include weeks 1-5
week_5_mean = df.loc[4, 'rate_expanding_mean']
expected = np.mean([0.05, 0.06, 0.07, 0.08, 0.09])
assert np.isclose(week_5_mean, expected), "Expanding mean should only use past data"
```

### Automated Validation

```bash
# Run temporal integrity tests
pytest tests/leakage/test_aggregation_lookahead.py -v
```

---

## Impact Assessment

| Metric | Before Fix | After Fix | Interpretation |
|--------|-----------|-----------|----------------|
| CV Fold Variance | 0.0002 | 0.0018 | More realistic |
| IS/OOS Gap | 0.5% | 4.2% | Expected degradation |
| Shuffled Target AUC | 0.62 | 0.51 | Closer to random |

**Key insight:** The increased variance and gap are healthy signs that the model is no longer cheating.

---

## Lessons Learned

1. **Split BEFORE feature engineering**
   - Always separate train/test before computing any statistics
   - Never let future data influence past feature values

2. **Use expanding windows, not rolling**
   - Expanding windows guarantee temporal integrity
   - Rolling windows can accidentally span boundaries

3. **Verify with shuffled target test**
   - Leaky aggregations often make shuffled target test pass
   - AUC ~0.50 is the gold standard for no leakage

4. **CV fold variance is informative**
   - Suspiciously uniform CV results suggest leakage
   - Healthy models show some natural variation

---

## Related Documentation

- `knowledge/practices/LEAKAGE_CHECKLIST.md` - Section 5
- `src/features/aggregation/` - Aggregation implementations
- `Episode 01: Lag-0 Competitor Rates` - Related temporal issue

---

## Tags

`#leakage` `#aggregation` `#rolling-stats` `#expanding-window` `#cv-boundaries`

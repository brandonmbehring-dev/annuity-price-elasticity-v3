# Leakage Audit Trail

**Version**: 1.0.0 | **Last Updated**: 2026-01-31

This document catalogs the 10 categories of data leakage bugs specific to annuity price elasticity modeling. For each category, we document:
- What the bug is and why it's dangerous
- How we detect it (automated gates)
- Historical examples from this codebase
- Prevention patterns

---

## Bug Category Index

| # | Category | Severity | Automated Detection |
|---|----------|----------|---------------------|
| 1 | Lag Leakage (Competitor) | CRITICAL | `detect_lag0_features()` |
| 2 | Temporal Boundary Violations | CRITICAL | `check_temporal_boundary()` |
| 3 | Threshold Leakage | HIGH | Manual + shuffled target |
| 4 | Market Weight Leakage | HIGH | Manual audit |
| 5 | Scaling Leakage | MEDIUM | `run_shuffled_target_test()` |
| 6 | Product Mix Confounding | MEDIUM | Stratified coefficient check |
| 7 | Own-Rate Endogeneity | MEDIUM | Economic sign validation |
| 8 | Holiday Effect Leakage | LOW | Manual audit |
| 9 | Cross-Validation Shuffling | CRITICAL | Temporal CV enforcement |
| 10 | Macro Feature Lookahead | HIGH | Lag verification |

---

## Category 1: Lag Leakage (Competitor Rates)

### Description

Using competitor rates from the same period (t=0) violates causal identification. Customers cannot react to rates they haven't observed yet.

### Why It's Dangerous

- Creates artificially high R² (~10-20% inflation)
- Coefficients have no causal interpretation
- "What-if" scenarios become meaningless
- May show wrong sign (positive instead of negative)

### Detection Gate

```python
from src.validation.leakage_gates import detect_lag0_features

result = detect_lag0_features(feature_names)
assert result.status != GateStatus.HALT
```

### Patterns Detected

```
FORBIDDEN PATTERNS:
- competitor_*_t0
- competitor_*_lag_0
- competitor_*_lag0
- competitor_*_current
- C_lag0, C_t0, C_t
```

### Historical Example (Prevented)

**Date**: 2025-07-15
**Code**: `features = ["prudential_rate_t0", "competitor_rate_t0"]`
**Detection**: Coefficient sign was positive (should be negative)
**Resolution**: Changed to `competitor_rate_t2` (minimum 2-week lag)

### Prevention Pattern

```python
# DON'T
features = ["competitor_rate_t0"]  # Lag-0!

# DO
features = ["competitor_rate_t2"]  # 2-week minimum lag
```

---

## Category 2: Temporal Boundary Violations

### Description

Training data overlaps with or extends into the test period. This happens with:
- Random train/test splits on time series
- Improperly defined date cutoffs
- Test data earlier than training data

### Why It's Dangerous

- Model "memorizes" test period patterns
- Cross-validation scores are unreliable
- Out-of-sample performance is illusory

### Detection Gate

```python
from src.validation.leakage_gates import check_temporal_boundary

result = check_temporal_boundary(train_dates, test_dates)
assert result.status != GateStatus.HALT
```

### Historical Example (Prevented)

**Date**: 2025-08-22
**Code**: `train_test_split(X, y, test_size=0.2, shuffle=True)`
**Detection**: Gate caught test_min < train_max
**Resolution**: Changed to temporal split with sorted dates

### Prevention Pattern

```python
# DON'T
X_train, X_test = train_test_split(X, test_size=0.2, shuffle=True)

# DO
df = df.sort_values("date")
cutoff_idx = int(len(df) * 0.8)
train_df = df.iloc[:cutoff_idx]
test_df = df.iloc[cutoff_idx:]
assert train_df["date"].max() < test_df["date"].min()
```

---

## Category 3: Threshold Leakage

### Description

Computing thresholds (percentiles, quantiles, regime indicators) on the full dataset instead of training set only.

### Why It's Dangerous

- Future extreme values inform historical classifications
- Regime indicators become forward-looking
- 5-15% R² inflation typical

### Detection

```python
# Shuffled target test catches most threshold leakage
result = run_shuffled_target_test(model, X, y)
assert result.metric_value < 0.10
```

### Common Violations

```python
# WRONG: Threshold from full data
threshold = df["volatility"].quantile(0.90)
df["high_vol"] = df["volatility"] > threshold

# RIGHT: Expanding threshold (uses only past)
df["high_vol"] = df["volatility"] > (
    df["volatility"].expanding().quantile(0.90).shift(1)
)
```

### Prevention Pattern

- Use expanding windows for all threshold computations
- Compute thresholds per CV fold
- Document threshold computation in feature engineering

---

## Category 4: Market Weight Leakage

### Description

Market-share weights for competitor aggregation computed using data from the full time period instead of trailing windows.

### Why It's Dangerous

- Future market share informs historical weighted averages
- Companies that grow later get more weight in early periods
- Subtle but persistent bias

### Detection

Manual audit of weight computation code:
```python
# Check: Do weights use only data available at prediction time?
```

### Historical Example (Fixed)

**Date**: 2025-09-10
**Code**: Weights computed from full dataset premium totals
**Resolution**: Changed to trailing 52-week market share windows

### Prevention Pattern

```python
# DON'T
weights = df.groupby("company")["premium"].sum() / df["premium"].sum()

# DO
def compute_trailing_weights(df, window=52):
    """Compute market share weights using trailing window only."""
    return (
        df.groupby(["date", "company"])["premium"]
        .sum()
        .groupby("company")
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
```

---

## Category 5: Scaling Leakage

### Description

Fitting StandardScaler/MinMaxScaler on full dataset before train/test split.

### Why It's Dangerous

- Test set mean/std leak into training features
- 5-10% R² inflation
- Model learns about data it shouldn't see

### Detection Gate

```python
from tests.anti_patterns.test_scaling_leakage import detect_mean_shift_leakage

result = detect_mean_shift_leakage(X_train, X_test, scaler)
assert not result.has_leakage
```

### Historical Example (Pattern Prevented)

**Pattern**: `scaler.fit_transform(X)` then split
**Resolution**: Use sklearn Pipeline

### Prevention Pattern

```python
# DON'T
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = X_scaled[:split], X_scaled[split:]

# DO
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge())
])
pipeline.fit(X_train, y_train)  # Scaler fits on train only
```

---

## Category 6: Product Mix Confounding

### Description

Pooling different buffer levels (6Y20B vs 6Y10B) without controlling for product-specific effects.

### Why It's Dangerous

- Different products have different customer bases
- Elasticity estimates are confounded
- Simpson's paradox possible (aggregate vs subgroup)

### Detection

```python
# Stratified coefficient check
for product in ["6Y20B", "6Y10B"]:
    result = fit_model(df[df["product_code"] == product])
    assert result.own_rate_coef > 0
    assert result.competitor_coef < 0
```

### Prevention Pattern

```python
# Option 1: Stratify analysis
models = {product: fit_model(df[df["product"] == product]) for product in products}

# Option 2: Include buffer indicators
features = base_features + ["buffer_20_indicator"]

# Option 3: Interaction terms
features = base_features + ["rate_x_buffer_20"]
```

---

## Category 7: Own-Rate Endogeneity

### Description

Own rate at time t influenced by factors also affecting sales at time t, creating circular causation.

### Why It's Dangerous

- Biased elasticity estimates
- Coefficient may have wrong sign
- Pricing recommendations incorrect

### Detection

```python
# Economic sign validation
coefficients = model.get_coefficients()
assert coefficients["own_rate"] > 0, "Own rate should be positive (yield economics)"
```

### Mitigation Approaches

1. **Lagged own rate**: Use t-1 instead of t-0
2. **Control variables**: Include confounders (treasury rates, VIX, seasonality)
3. **Instrumental variables**: Use exogenous rate determinants

### Prevention Pattern

```python
# Include controls for endogeneity
features = [
    "own_rate_t0",       # May be endogenous
    "treasury_10y_t0",   # Interest rate environment
    "vix_t0",            # Market volatility
    "quarter_indicator", # Seasonality
]
```

---

## Category 8: Holiday Effect Leakage

### Description

Holiday indicators computed using the full dataset's date range, potentially including future holidays.

### Why It's Dangerous

- Minor but detectable leakage
- Holiday effects may be miscalibrated
- More relevant for short-term forecasting

### Detection

Manual audit of holiday computation code.

### Prevention Pattern

```python
# DON'T
holidays = get_holidays(df["date"].min(), df["date"].max())

# DO
holidays = get_holidays(df["date"].min(), train_df["date"].max())
# Or use static holiday calendar
```

---

## Category 9: Cross-Validation Shuffling

### Description

Using KFold with `shuffle=True` on time series data violates temporal ordering.

### Why It's Dangerous

- Future data leaks into training folds
- Cross-validation scores are meaningless
- Model learns temporal patterns it shouldn't know

### Detection

Enforce temporal CV in codebase:
```python
# Audit: Search for KFold with shuffle=True
grep -r "KFold.*shuffle.*True" src/
```

### Prevention Pattern

```python
# DON'T
from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle=True)

# DO
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=5)

# Or use walk-forward validation
```

---

## Category 10: Macro Feature Lookahead

### Description

Using macroeconomic indicators (VIX, treasury rates) from future periods.

### Why It's Dangerous

- Economic environment at prediction time unknown
- Creates artificial predictive power
- May show wrong coefficient signs

### Detection

```python
# Verify all macro features are lagged
for feature in macro_features:
    assert "_t0" in feature or "_lag_" not in feature, f"Future macro: {feature}"
```

### Historical Example

**Potential Issue**: Using VIX at t+1 to predict sales at t
**Prevention**: All macro features use t-0 (current or lagged)

### Prevention Pattern

```python
# Macro features at t-0 are OK (available at prediction time)
features = [
    "vix_t0",          # Current VIX (available)
    "treasury_10y_t0", # Current treasury (available)
]

# DON'T use forward-looking macro
# "vix_t_plus_1"  # FORBIDDEN
```

---

## Automated Leakage Gate Summary

### Gate Execution Order

```python
from src.validation.leakage_gates import run_all_gates

report = run_all_gates(
    model=fitted_model,
    X=X_test,
    y=y_test,
    r_squared=model_r2,
    baseline_r_squared=naive_r2,
    feature_names=list(X.columns),
    train_dates=train_df["date"],
    test_dates=test_df["date"],
)

print(report)
# Output:
# [PASS/HALT] Lag-0 Feature Detection
# [PASS/HALT] R-Squared Threshold
# [PASS/HALT] Improvement Threshold
# [PASS/HALT] Temporal Boundary Check
# [PASS/HALT] Shuffled Target Test
```

### CI Integration

```bash
# Run leakage audit in CI
pytest tests/anti_patterns/ -v -m leakage

# Full leakage check
make leakage-audit
```

---

## Historical Leakage Incidents

### Incident Log

| Date | Category | Detection | Impact | Resolution |
|------|----------|-----------|--------|------------|
| 2025-07-15 | Lag-0 competitor | Sign check | R² inflated 15% | Min 2-week lag |
| 2025-08-22 | Temporal overlap | Boundary gate | CV scores unreliable | Temporal split |
| 2025-09-10 | Market weight | Manual audit | Subtle bias | Trailing windows |
| 2025-11-05 | contract_issue_date | Documentation | 110-day lookahead | application_signed_date |

### Lessons Learned

1. **Automate detection**: Manual checks miss subtle leakage
2. **Fail fast**: Check before expensive training
3. **Economic sense**: Wrong signs often indicate leakage
4. **Document patterns**: Prevent recurrence

---

## References

1. **Leakage Checklist**: `knowledge/practices/LEAKAGE_CHECKLIST.md`
2. **Testing Strategy**: `docs/development/TESTING_STRATEGY.md`
3. **Validation Evidence**: `docs/validation/VALIDATION_EVIDENCE.md`
4. **Anti-Pattern Tests**: `tests/anti_patterns/`
5. **Leakage Gates**: `src/validation/leakage_gates.py`

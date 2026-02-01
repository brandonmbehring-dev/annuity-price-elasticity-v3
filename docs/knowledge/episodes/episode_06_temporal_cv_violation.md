# Episode 06: Temporal Cross-Validation Violation

**Category**: Train/Test Contamination (10 Bug Category #7)
**Discovered**: 2025-11-XX (validation audit)
**Impact**: Random CV allowed future data in training folds
**Status**: RESOLVED - TimeSeriesSplit enforced

---

## The Bug

Using random K-Fold cross-validation instead of TimeSeriesSplit for time series data, allowing future observations to leak into training folds.

---

## Symptom

- Model performed identically on random CV and shuffled data
- No autocorrelation structure in residuals (should exist)
- Predictions had no temporal pattern

---

## Root Cause

Random CV shuffles data, so fold 1 might contain:
- Week 5 (train), Week 10 (train), Week 2 (test)

When predicting Week 2, you've already trained on Weeks 5 and 10!

---

## The Fix

```python
# WRONG: Random CV
from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle=True)  # DON'T SHUFFLE!

# CORRECT: Time series CV
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=5, gap=2)  # 2-week gap between train/test
```

---

## Gate Implementation

```python
def validate_cv_temporal_order(cv, X, dates):
    """Ensure CV respects temporal ordering."""
    for train_idx, test_idx in cv.split(X):
        max_train_date = dates.iloc[train_idx].max()
        min_test_date = dates.iloc[test_idx].min()
        assert max_train_date < min_test_date, "Train data extends into test period"
```

---

## Lessons Learned

1. **Never shuffle time series data for CV**
2. **TimeSeriesSplit maintains temporal order**
3. **Add gap between train and test to prevent leakage**
4. **Autocorrelation in residuals is expected for time series**

---

## Tags

`#cv` `#time-series` `#temporal-order` `#train-test-split`

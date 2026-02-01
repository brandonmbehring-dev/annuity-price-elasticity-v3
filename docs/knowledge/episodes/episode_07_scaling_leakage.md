# Episode 07: Scaling/Normalization Leakage

**Category**: Preprocessing Leakage (10 Bug Category #8)
**Discovered**: 2025-12-XX (feature audit)
**Impact**: Scaler fit on full data, not training only
**Status**: RESOLVED - Pipeline-based scaling

---

## The Bug

Fitting StandardScaler on the ENTIRE dataset before train/test split, allowing test set statistics (mean, std) to influence training features.

---

## Symptom

- Features had suspiciously similar distributions across train/test
- Out-of-sample predictions perfectly centered at zero
- Residuals showed no drift between train and test periods

---

## Root Cause

```python
# WRONG: Scale on full data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses test data statistics!
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
```

The scaler's mean and std include test period values.

---

## The Fix

```python
# CORRECT: Scale on training only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train ONLY
X_test_scaled = scaler.transform(X_test)  # Transform (don't fit!) test

# BEST: Use Pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge()),
])
# Pipeline handles scaling correctly inside CV
```

---

## Gate Implementation

```python
def validate_scaler_fit_on_train_only(scaler, train_data, full_data):
    """Verify scaler was fit on training data only."""
    expected_mean = train_data.mean()
    actual_mean = scaler.mean_

    assert np.allclose(expected_mean, actual_mean, rtol=1e-5), (
        "Scaler mean doesn't match training data - possible leakage"
    )
```

---

## Lessons Learned

1. **fit() on training, transform() on test**
2. **Use sklearn Pipelines to prevent mistakes**
3. **Different scaling for different CV folds is normal**
4. **Test data should look "different" from training after scaling**

---

## Tags

`#scaling` `#normalization` `#preprocessing` `#pipeline`

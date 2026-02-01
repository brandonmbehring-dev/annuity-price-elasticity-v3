# Episode 03: Feature Selection Bias

**Category**: Selection/Training Leakage (10 Bug Category #4)
**Discovered**: 2025-11-XX (model audit)
**Impact**: Overly optimistic feature importance and model performance
**Status**: RESOLVED - Nested CV implemented

---

## The Bug

Performing feature selection on the ENTIRE dataset before cross-validation, rather than performing selection within each training fold.

When you select "best" features using all data, then evaluate with CV, the test folds have already influenced feature selection—leaking information.

---

## Symptom

**How it manifested:**
- Feature importance rankings suspiciously stable across CV folds
- Selected features performed identically in-sample and out-of-sample
- Model generalized poorly to truly unseen data (production)

**Red flags in output:**
```
Best features: [feature_A, feature_B, feature_C]
CV Performance: R² = 0.82, MAPE = 10.1%
Production Performance: R² = 0.58, MAPE = 18.4%  # Much worse!
```

---

## Root Cause Analysis

### 1. The Problem

```python
# WRONG: Select features using ALL data
X, y = load_data()
best_features = select_features(X, y)  # Uses full dataset!

# Then cross-validate
for train_idx, test_idx in cv.split(X):
    X_train = X[train_idx][best_features]
    X_test = X[test_idx][best_features]
    model.fit(X_train, y[train_idx])
    score = model.score(X_test, y[test_idx])
```

The `select_features()` call sees the test data, biasing feature selection toward features that happen to work well on test folds.

### 2. Why This Leaks

Feature selection is a form of model training. When you:
1. Look at all data to pick features
2. Then "validate" on subsets of that same data

You've already used the test data during training (feature selection phase).

### 3. The Magnitude

This bias can be substantial:
- Overestimates R² by 10-20 percentage points
- Selected features may be spuriously correlated in-sample
- Features that worked historically may not generalize

---

## The Fix

### Before (Leaky) ❌

```python
# Feature selection on full data
X, y = load_data()
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)  # Sees ALL data

# Cross-validation
cv_scores = cross_val_score(model, X_selected, y, cv=5)
```

### After (Safe) ✅

```python
# Nested cross-validation: selection INSIDE each fold
X, y = load_data()

outer_cv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for train_idx, test_idx in outer_cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Feature selection on TRAINING only
    selector = SelectKBest(k=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)  # Only transform!

    model.fit(X_train_selected, y_train)
    score = model.score(X_test_selected, y_test)
    cv_scores.append(score)
```

### Pipeline Approach

```python
from sklearn.pipeline import Pipeline

# Create pipeline that includes feature selection
pipeline = Pipeline([
    ('selector', SelectKBest(k=10)),
    ('model', Ridge()),
])

# Cross-validate the ENTIRE pipeline
cv_scores = cross_val_score(pipeline, X, y, cv=TimeSeriesSplit(5))
```

---

## Gate Implementation

### Detection Logic

```python
def check_selection_leakage(
    selected_features: List[str],
    cv_fold_features: List[List[str]],
) -> bool:
    """Check if feature selection leaked across CV folds.

    If selected features are IDENTICAL across all folds,
    selection was likely done on full data.
    """
    if len(cv_fold_features) < 2:
        return False

    # All folds have identical features = suspicious
    first_set = set(cv_fold_features[0])
    for fold_features in cv_fold_features[1:]:
        if set(fold_features) != first_set:
            return False  # Different features = likely correct

    return True  # All identical = suspicious
```

### Verification Test

```python
@pytest.mark.leakage
def test_feature_selection_varies_across_folds():
    """Features selected should vary somewhat across CV folds."""
    cv = TimeSeriesSplit(n_splits=5)
    fold_features = []

    for train_idx, test_idx in cv.split(X):
        selector = fit_selector(X[train_idx], y[train_idx])
        fold_features.append(selector.get_selected_features())

    # Should have SOME variation
    n_unique_sets = len(set(tuple(f) for f in fold_features))
    assert n_unique_sets > 1, "Features identical across folds = selection leakage"
```

---

## Verification

### Check Your Pipeline

```python
# Verify feature selection is inside CV loop
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('selector', SelectKBest(k=10)),
    ('model', Ridge()),
])

# This is CORRECT - selection happens inside each fold
scores = cross_val_score(pipeline, X, y, cv=TimeSeriesSplit(5))

# Verify by checking if different folds have different features
# (Would require custom scorer that logs selected features)
```

### Compare Nested vs Non-Nested

```python
# WRONG: Non-nested (selection before CV)
selector = SelectKBest(k=10).fit(X, y)
X_sel = selector.transform(X)
non_nested_score = cross_val_score(model, X_sel, y, cv=5).mean()

# CORRECT: Nested (selection inside CV)
pipeline = Pipeline([('sel', SelectKBest(k=10)), ('mod', Ridge())])
nested_score = cross_val_score(pipeline, X, y, cv=5).mean()

# Nested should be LOWER (more honest)
assert nested_score < non_nested_score, "Nested CV should have lower score"
```

---

## Impact Assessment

| Metric | Non-Nested CV | Nested CV | Interpretation |
|--------|--------------|-----------|----------------|
| CV R² | 0.82 | 0.71 | More realistic |
| Production R² | 0.58 | 0.68 | Better generalization |
| Feature Stability | 100% | 75% | Expected variation |

**Key insight:** The lower nested CV score actually predicts production performance better.

---

## Lessons Learned

1. **Feature selection is model training**
   - Treat it like any other learning step
   - Must be inside the CV loop

2. **Use sklearn Pipelines**
   - Pipelines automatically handle selection inside CV
   - Less error-prone than manual implementation

3. **Expect feature variation across folds**
   - Identical features across all folds = red flag
   - Some variation is healthy and expected

4. **Production is the true test**
   - Non-nested CV overestimates production performance
   - Nested CV gives more honest estimates

---

## Related Documentation

- `knowledge/practices/LEAKAGE_CHECKLIST.md` - Section 5
- `src/features/selection/` - Feature selection implementations
- `Episode 02: Aggregation Lookahead` - Related data leakage

---

## Tags

`#leakage` `#feature-selection` `#nested-cv` `#generalization` `#pipeline`

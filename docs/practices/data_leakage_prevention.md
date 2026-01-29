# Data Leakage Prevention Pattern

**Source**: Lessons from annuity_forecasting and myga-forecasting-v2 (10+ bugs discovered)
**Last Updated**: 2026-01-20
**Copied from**: ~/Claude/lever_of_archimedes/patterns/data_leakage_prevention.md

---

## Executive Summary

Data leakage is the #1 cause of false positive research results. This pattern documents prevention strategies learned from painful experience in time series forecasting projects.

**Core truth**: "Too good" results (>20% improvement) almost always indicate leakage, not discovery.

---

## The 10 Bug Categories

From myga-forecasting-v2's post-mortem:

| # | Bug Category | What Went Wrong | Prevention |
|---|--------------|-----------------|------------|
| 1 | Target Alignment | Train/test date boundaries violated | Assert date ranges don't overlap |
| 2 | Future Data in Lags | Lag features used information from future | Build lags WITHIN each training split |
| 3 | Persistence Implementation | Predicted levels instead of changes | Test against known-answer calculations |
| 4 | Feature Selection Target | BIC used LEVEL target, not CHANGE | Verify target construction matches model |
| 5 | Regime Computation | Used full series instead of training-only | Compute regime indicators per-split |
| 6 | Weights Computation | Look-ahead in weighting scheme | Audit all weight/scaling calculations |
| 7 | Walk-Forward Splits | Gap miscalculation between train/test | Unit test split boundaries explicitly |
| 8 | Multiple Sources of Truth | Conflicting documentation | Single authoritative document |
| 9 | Internal-Only Validation | No external verification | Always run synthetic + shuffled tests FIRST |
| 10 | Architecture Mismatch | MIDAS incompatible with weekly aggregates | Research fundamentals before implementing |

---

## Mandatory Verification Protocol

### BEFORE Internal Validation

Run these tests FIRST, before evaluating on real data:

```python
# 1. Shuffled Target Test
# If model performs well on shuffled target, it's leaking
def test_shuffled_target_fails():
    shuffled_y = np.random.permutation(y)
    model.fit(X, shuffled_y)
    # Model should perform WORSE than random
    assert model.score(X_test, shuffled_y_test) < 0.0

# 2. Synthetic AR(1) Test
# Model should achieve theoretical bounds, not exceed them
def test_synthetic_ar1():
    y = generate_ar1(phi=0.9, n=500)
    mae = evaluate_model(model, y)
    # Cannot beat theoretical lower bound by more than 10%
    assert mae >= 0.9 * theoretical_mae
```

### Suspicious Results Protocol

| Observation | Action |
|-------------|--------|
| >20% improvement over baseline | HALT - run shuffled target test |
| MAE < theoretical minimum | HALT - check for leakage |
| R² > 0.3 on near-random-walk | HALT - verify feature construction |
| Same features selected all horizons | HALT - check selection timing |

---

## Walk-Forward Validation Checklist

```
□ TEMPORAL BOUNDARIES
  □ Training data ends BEFORE test data starts
  □ Gap of h periods between train end and test start
  □ No future dates in training set
  □ Assert: max(train_dates) < min(test_dates) - horizon

□ FEATURE CONSTRUCTION
  □ Lag features computed WITHIN training split only
  □ Rolling statistics use expanding window in training only
  □ No lag-0 (contemporaneous) features for competitors
  □ All features differenced (if target is differenced)

□ TARGET CONSTRUCTION
  □ Target computed consistently (levels vs changes)
  □ Target alignment matches feature alignment
  □ Horizon semantics documented (h=2 weekly = 14 calendar days)

□ FEATURE SELECTION
  □ Run ONCE upfront on first 80% of data, OR
  □ Run in FIRST training split only, apply to all
  □ NEVER run per-split (causes massive leakage)

□ REGIME/WEIGHTS
  □ Computed from TRAINING data only
  □ No look-ahead in regime classification
  □ Volatility windows stay within training bounds

□ EXTERNAL VERIFICATION
  □ Synthetic AR(1) test passes
  □ Shuffled target test fails (as expected)
  □ Known-answer calculations match
```

---

## Anti-Patterns to Recognize

### 1. "Feature Selection Per-Split"
```python
# WRONG: Leakage - sees future data in CV
for train_idx, test_idx in cv.split(X):
    X_train = X[train_idx]
    selected = select_features(X_train, y_train)  # LEAKS!

# CORRECT: Select once upfront
selected = select_features(X[:first_80_pct], y[:first_80_pct])
for train_idx, test_idx in cv.split(X):
    X_train = X[train_idx][selected]  # Apply pre-selected
```

### 2. "Regime from Full Series"
```python
# WRONG: Uses future to classify past
regime = classify_regime(full_series)  # LEAKS!

# CORRECT: Classify within training only
regime = classify_regime(train_series)
```

### 3. "Contemporaneous Features"
```python
# WRONG: y[t] predicted using X[t] (not available yet)
features = [x[t], x[t-1], x[t-2]]  # x[t] is LEAKAGE!

# CORRECT: Only lagged features
features = [x[t-1], x[t-2], x[t-3]]  # All available at forecast time
```

### 4. "Lag-0 Competitor Rates" (RILA-Specific)
```python
# WRONG: Competitor rates at t (simultaneity bias)
features = ['P_lag_0', 'C_weighted_mean_lag_0']  # C_lag_0 is problematic!

# CORRECT: Lagged competitor rates
features = ['P_lag_0', 'C_weighted_mean_lag_2', 'C_weighted_mean_lag_3']
```

---

## Red Flags Checklist

**If any of these are true, STOP and investigate:**

- [ ] Model improvement >20% over persistence/naive baseline
- [ ] MAE suspiciously low (near theoretical minimum)
- [ ] R² > 0.3 on near-random-walk series (ACF > 0.95)
- [ ] Feature selection chooses same features for all horizons
- [ ] Results "too good to be true" on first implementation
- [ ] Multiple validation documents with conflicting results

---

## Documentation Requirements

### Every Research Project Must Have:

1. **Single source of truth** - One authoritative methodology document
2. **Horizon semantics** - Clear definition of what h=1, h=2 means
3. **Leakage audit trail** - Document all leakage bugs found and fixed
4. **External verification results** - Synthetic + shuffled test outcomes
5. **Suspicious results log** - What triggered investigation, what was found

---

## RILA-Specific Considerations

### Buffer Level as Confounder

Buffer level selection may be correlated with unobserved buyer characteristics:
- Don't use buffer level to predict sales without controlling for rate
- Consider buffer-rate interaction effects

### Market-Share Weights

If market shares are updated using recent data:
- Ensure weight updates don't leak future information
- Use lagged market shares if computing weights dynamically

---

## References

- `knowledge/practices/testing.md` - 6-layer validation architecture
- `knowledge/practices/LEAKAGE_CHECKLIST.md` - RILA-specific checklist
- `knowledge/analysis/CAUSAL_FRAMEWORK.md` - Identification strategy
- `knowledge/integration/LESSONS_LEARNED.md` - Critical traps

---

**Last updated**: 2026-01-20
**Learned from**: 7+ data leakage bugs across 2 major projects

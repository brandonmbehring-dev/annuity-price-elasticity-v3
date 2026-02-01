# Episode 01: Lag-0 Competitor Rate Leakage

**Category**: Simultaneous Causality (10 Bug Category #2)
**Discovered**: 2025-10-XX (initial development)
**Impact**: Elasticity overestimated by ~40%
**Status**: RESOLVED - Gate implemented

---

## The Bug

Using `competitor_rate[t]` instead of `competitor_rate[t-1]` when modeling `own_rate[t]` at time `t`.

At time `t`, both Prudential and competitors observe the same market conditions and make simultaneous pricing decisions. Including competitor's current-period rate as a feature creates spurious correlation that inflates apparent model skill.

---

## Symptom

**How it manifested:**
- Model R² suspiciously high (>85%) for weekly sales forecasting
- Shuffled target test PASSED when it should have failed (AUC ~0.50 expected)
- Competitor coefficient had WRONG SIGN (positive instead of negative)

**Red flags in output:**
```
R² = 0.91 (expected: 0.50-0.80 for valid model)
competitor_rate_current coefficient: +0.045 (expected: NEGATIVE)
```

---

## Root Cause Analysis

### 1. The Identification Problem

At time `t`, both you and your competitor observe:
- Market conditions (VIX, Treasury rates)
- Customer demand signals
- Regulatory environment

Both firms adjust rates based on these **common shocks**. This creates correlation that looks like competitive response but is actually **confounding**.

### 2. The Causal Graph

```
     Market Conditions (t)
          /           \
         ↓             ↓
  Own Rate (t)    Competitor Rate (t)  ← CORRELATED but not CAUSAL
         \             /
          ↓           ↓
         Sales (t)
```

**The problem:** When we regress sales on both rates at time `t`, we're capturing the correlation from common shocks, not the causal effect of competitor pricing.

### 3. The Spurious Correlation

If both firms raise rates when VIX is high, we'll see:
- High own rate correlates with high competitor rate
- Model "learns" that high competitor rate = high sales
- This is backwards! (Should be: high competitor rate = LOW sales)

---

## The Fix

### Before (Leaky) ❌

```python
# WRONG: Lag-0 competitor feature
features = pd.DataFrame({
    'own_rate': df['prudential_rate_current'],
    'competitor_rate': df['competitor_mid_current'],  # Lag-0!
})
```

### After (Safe) ✅

```python
# CORRECT: Lagged competitor feature (t-2 minimum)
features = pd.DataFrame({
    'own_rate': df['prudential_rate_current'],
    'competitor_rate': df['competitor_mid_t2'],  # 2-week lag
})
```

### Why t-2?

- **t-1** might still have some simultaneous decision-making
- **t-2** ensures competitor rates are truly "predetermined" before our decision
- Matches the business reality: rate decisions take ~1 week to implement

---

## Gate Implementation

### Detection Code

Located in `src/validation/leakage_gates.py`:

```python
def detect_lag0_features(feature_names: List[str]) -> List[str]:
    """Detect lag-0 competitor features.

    Returns list of forbidden features found.
    """
    lag0_patterns = [
        r"competitor.*_t0$",
        r"competitor.*_current$",
        r"competitor.*_lag_0$",
        r"^C_.*_t0$",
    ]

    forbidden = []
    for feature in feature_names:
        for pattern in lag0_patterns:
            if re.search(pattern, feature, re.IGNORECASE):
                forbidden.append(feature)
                break

    return forbidden
```

### Test Coverage

Located in `tests/anti_patterns/test_lag0_competitor_detection.py`:

```python
@pytest.mark.parametrize("feature,should_fail", [
    ("competitor_mid_t0", True),
    ("competitor_current", True),
    ("competitor_mid_t2", False),
    ("competitor_top5_t3", False),
])
def test_lag0_detection(feature, should_fail):
    forbidden = detect_lag0_features([feature])
    assert bool(forbidden) == should_fail
```

---

## Verification

### Run Detection

```bash
# Check for lag-0 features in current model
python -c "
from src.validation.leakage_gates import detect_lag0_features
features = ['prudential_rate_current', 'competitor_mid_t2', 'competitor_top5_t3']
forbidden = detect_lag0_features(features)
print(f'Forbidden features: {forbidden}')
assert len(forbidden) == 0, 'LEAKAGE DETECTED!'
"
```

### Test Suite

```bash
# Run anti-pattern tests
pytest tests/anti_patterns/test_lag0_competitor_detection.py -v
```

---

## Impact Assessment

| Metric | Before Fix | After Fix | Improvement |
|--------|-----------|-----------|-------------|
| Model R² | 0.91 | 0.78 | Realistic |
| Competitor Coefficient | +0.045 | -0.031 | Correct Sign |
| Out-of-Sample MAPE | 8% | 13% | Realistic |

**Key insight:** The "worse" performance after the fix is actually BETTER because it reflects reality. The leaky model was overfitting to spurious correlations.

---

## Lessons Learned

1. **Suspiciously good performance is a red flag**
   - R² > 85% for weekly sales forecasting is almost certainly leakage
   - Always run shuffled target test to validate

2. **Sign constraints catch leakage**
   - Economic theory says competitor rate should be NEGATIVE
   - Positive coefficient = something is wrong

3. **Lag structure matters for causal inference**
   - Using lagged values ensures temporal ordering
   - Minimum 2-week lag for competitor features

4. **Gates prevent recurrence**
   - Automated detection catches future mistakes
   - Test coverage ensures gates work

---

## Related Documentation

- `knowledge/practices/LEAKAGE_CHECKLIST.md` - Section 3
- `src/validation/leakage_gates.py` - Detection implementation
- `tests/anti_patterns/test_lag0_competitor_detection.py` - Test coverage
- `knowledge/analysis/CAUSAL_FRAMEWORK.md` - Identification strategy

---

## Tags

`#leakage` `#causality` `#competitor-rates` `#lag-structure` `#gate`

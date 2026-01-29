# Leakage Gate Verification

**Date**: YYYY-MM-DD
**Product**: [6Y20B | 6Y10B | 10Y20B]
**Auditor**: [Name]
**Model Version**: [version or commit SHA]

---

## Summary

| Gate | Status | Notes |
|------|--------|-------|
| Lag-0 Detection | [ ] PASS / [ ] HALT | |
| R-squared Threshold | [ ] PASS / [ ] WARN / [ ] HALT | |
| Improvement Threshold | [ ] PASS / [ ] WARN / [ ] HALT | |
| Temporal Boundary | [ ] PASS / [ ] HALT | |
| Shuffled Target Test | [ ] PASS / [ ] HALT | |

**Overall Status**: [ ] PASSED / [ ] FAILED

---

## Gate Details

### 1. Lag-0 Competitor Detection

**Purpose**: Verify no concurrent competitor features (violates causal identification).

**Check**:
```bash
make pattern-check
```

**Result**:
- [ ] No lag-0 features detected
- [ ] Lag-0 features found: [list features]

**Action if HALT**: Remove lag-0 features, use t-1 or earlier lags.

---

### 2. R-squared Threshold

**Purpose**: Detect suspiciously high R-squared (typical range: 0.05-0.25).

**Thresholds**:
- WARN: R2 > 0.20
- HALT: R2 > 0.30

**Observed R2**: _____

**Result**:
- [ ] Within expected range
- [ ] Higher than typical (investigate)
- [ ] Suspiciously high (HALT)

**Action if HALT**: Full leakage audit, check feature construction, verify train/test split.

---

### 3. Improvement Threshold

**Purpose**: Detect suspiciously large improvements over baseline.

**Thresholds**:
- WARN: Improvement > 10%
- HALT: Improvement > 20%

**Baseline R2**: _____
**New R2**: _____
**Improvement**: _____%

**Result**:
- [ ] Reasonable improvement
- [ ] Large improvement (investigate)
- [ ] Suspiciously large (HALT)

**Action if HALT**: Audit all changes since baseline, check for accidental leakage.

---

### 4. Temporal Boundary Check

**Purpose**: Verify no future data in training features.

**Check**:
- Training period: _____ to _____
- Test period: _____ to _____
- Gap between: _____ days

**Walk-forward splits validated**: [ ] Yes / [ ] No

**Result**:
- [ ] Proper temporal split
- [ ] Temporal leakage detected

**Action if HALT**: Fix feature engineering to respect temporal boundaries.

---

### 5. Shuffled Target Test

**Purpose**: Model should FAIL on randomly shuffled target.

**Check**:
```python
from src.validation.leakage_gates import run_shuffled_target_test
result = run_shuffled_target_test(model, X, y)
print(result)
```

**Shuffled R2**: _____ (should be < 0.10)

**Result**:
- [ ] Model appropriately fails on shuffled target
- [ ] Model performs on shuffled target (HALT - leakage!)

**Action if HALT**: Critical leakage. Full audit of feature construction.

---

## Additional Checks (Optional)

### Feature Importance Audit

**Top 5 Features by Importance**:
1. _____ (importance: _____)
2. _____ (importance: _____)
3. _____ (importance: _____)
4. _____ (importance: _____)
5. _____ (importance: _____)

**Sanity Check**: Do these align with economic intuition?
- [ ] Yes, expected features dominate
- [ ] No, suspicious features present

---

### Coefficient Sign Verification

| Feature | Coefficient | Expected Sign | Match? |
|---------|-------------|---------------|--------|
| prudential_cap | | Positive | [ ] |
| C_lag_1 | | Negative | [ ] |
| C_lag_2 | | Negative | [ ] |
| DGS5 | | Negative | [ ] |

---

## Conclusions

**Leakage Status**: [ ] No leakage detected / [ ] Potential leakage / [ ] Confirmed leakage

**Recommended Actions**:
1.
2.
3.

**Sign-off**: _____________________ Date: _____

---

## References

- `knowledge/practices/data_leakage_prevention.md` - Leakage prevention patterns
- `knowledge/practices/LEAKAGE_CHECKLIST.md` - Pre-deployment checklist
- `knowledge/integration/LESSONS_LEARNED.md` - Critical traps

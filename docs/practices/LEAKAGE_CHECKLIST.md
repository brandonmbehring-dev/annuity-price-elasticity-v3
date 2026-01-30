# Multi-Product Pre-Deployment Leakage Checklist

**Purpose**: MANDATORY validation before deploying any elasticity model (RILA, FIA, MYGA).
**Last Updated**: 2026-01-24
**Status**: REQUIRED - Do not skip any checks
**Products Covered**: RILA, FIA (MYGA pending)

---

## Pre-Deployment Gate

**Run ALL checks before deployment. Any failure = BLOCK deployment.**

---

## 1. Shuffled Target Test (MANDATORY)

```python
def test_shuffled_target_fails():
    """Model should NOT work on shuffled targets."""
    X_train, y_train, X_test, y_test = prepare_data()

    # Shuffle target
    y_train_shuffled = np.random.permutation(y_train)
    y_test_shuffled = np.random.permutation(y_test)

    # Fit on shuffled
    model.fit(X_train, y_train_shuffled)
    score = model.score(X_test, y_test_shuffled)

    # Model should perform POORLY
    assert score < 0.05, f"Model works on shuffled data (R²={score:.3f}). LEAKAGE!"
```

**If this test PASSES (model works on shuffled data)**: STOP. You have data leakage.

---

## 2. Temporal Boundary Check (MANDATORY)

```python
def test_temporal_boundaries():
    """No future data in training."""
    for fold in cv_folds:
        train_dates = data.loc[fold['train_idx'], 'date']
        test_dates = data.loc[fold['test_idx'], 'date']

        max_train = train_dates.max()
        min_test = test_dates.min()

        # Gap should exist
        gap_days = (min_test - max_train).days
        assert gap_days >= 7, f"Train/test gap only {gap_days} days"

        # No overlap
        assert max_train < min_test, "Train data extends into test period"
```

---

## 3. Competitor Lag Check (MANDATORY)

```python
def test_no_lag0_competitors():
    """Competitor features must be lagged (no lag-0).

    Handles both RILA and FIA naming conventions:
    - RILA: competitor_mid_t0, competitor_current
    - FIA: comp_mean_lag_0, competitor_lag_0
    """
    features = model.feature_names_

    competitor_features = [
        f for f in features
        if 'competitor' in f.lower()
        or f.lower().startswith('comp_')
        or f.lower().startswith('c_') and 'rate' in f.lower()
    ]

    for feat in competitor_features:
        # FIA pattern
        assert '_lag_0' not in feat, f"Lag-0 competitor (FIA): {feat}"
        # RILA patterns
        assert '_t0' not in feat, f"Lag-0 competitor (RILA): {feat}"
        assert '_current' not in feat, f"Current competitor: {feat}"
```

**Product-Specific Patterns**:
| Product | Lag-0 Pattern | Safe Pattern |
|---------|---------------|--------------|
| RILA | `competitor_mid_t0`, `competitor_current` | `competitor_mid_t2` |
| FIA | `comp_mean_lag_0` | `comp_mean_lag_2` |

---

## 4. Suspicious Results Check (MANDATORY)

```python
def test_results_not_suspicious():
    """Results should not be 'too good'."""
    baseline_mae = evaluate_baseline(data)
    model_mae = evaluate_model(model, data)

    improvement = (baseline_mae - model_mae) / baseline_mae

    assert improvement < 0.20, (
        f"Improvement of {improvement:.0%} exceeds 20% threshold. "
        "Investigate for leakage before deployment."
    )

    # R² should not be suspiciously high
    r2 = model.score(X_test, y_test)
    assert r2 < 0.30, f"R² of {r2:.3f} is suspiciously high for elasticity model."
```

---

## 5. Feature Construction Audit

**Manual check required:**

- [ ] Lag features computed WITHIN each training split
- [ ] Rolling statistics use expanding window only
- [ ] No full-sample statistics (mean, max) in features
- [ ] Market-share weights don't use future data
- [ ] Holiday mask applied consistently

---

## 6. Coefficient Sign Check (MANDATORY)

```python
def test_coefficient_signs():
    """Coefficients have expected economic signs."""
    coefs = model.get_coefficients()

    # Own rate: POSITIVE (yield economics)
    p_coef = coefs.get('prudential_rate_current', coefs.get('P_lag_0'))
    assert p_coef > 0, f"Own rate coefficient is {p_coef}, expected > 0"

    # Competitor rate: NEGATIVE (substitution)
    c_coef = [v for k, v in coefs.items() if 'competitor' in k.lower() or 'C_' in k]
    for i, c in enumerate(c_coef):
        assert c < 0, f"Competitor coefficient {i} is {c}, expected < 0"
```

---

## 7. Buffer Level Control Check (RILA-Specific)

```python
def test_buffer_controlled():
    """Buffer level is controlled in model."""
    features = model.feature_names_

    # Either buffer indicators or stratified analysis
    has_buffer_control = (
        any('buffer' in f.lower() for f in features) or
        model.metadata.get('stratified_by_buffer', False)
    )

    assert has_buffer_control, "Buffer level not controlled in model"
```

---

## 8. Market-Share Weighting Check (RILA-Specific)

```python
def test_market_share_weighting():
    """Competitor aggregation uses market-share weighting."""
    # Check feature engineering code
    competitor_mean_col = [c for c in data.columns if 'C_weighted' in c]

    assert len(competitor_mean_col) > 0, (
        "No market-share weighted competitor mean found. "
        "RILA should use weighted means, not simple top-N."
    )
```

---

## 9. Top-N Aggregation Check (FIA-Specific)

```python
def test_top_n_aggregation():
    """FIA uses Top-N aggregation (not weighted)."""
    from src.notebooks.interface import create_interface

    interface = create_interface("FIA5YR", environment="fixture")

    # FIA should use top_n aggregation
    assert interface.aggregation.strategy_name == "top_n", (
        "FIA should use top_n aggregation strategy"
    )
    assert interface.aggregation.requires_weights is False, (
        "FIA top_n should not require weights"
    )
```

---

## 10. FIA Feature Naming Validation (FIA-Specific)

```python
def test_fia_feature_patterns():
    """FIA features use correct naming conventions."""
    features = model.feature_names_

    # Check for expected FIA patterns
    pru_features = [f for f in features if 'pru_rate' in f.lower()]
    comp_features = [f for f in features if 'comp_mean' in f.lower()]

    # Should have lagged features (not raw)
    for feat in pru_features + comp_features:
        assert '_lag_' in feat, f"Feature {feat} should use _lag_N suffix"

    # No lag-0 or lag-1 competitors
    for feat in comp_features:
        assert '_lag_0' not in feat, f"Forbidden lag-0: {feat}"
        assert '_lag_1' not in feat, f"Decision lag too short: {feat}"
```

**Reference**: See `knowledge/domain/FIA_FEATURE_MAPPING.md` for complete feature inventory.

---

## 11. Out-of-Sample Validation (MANDATORY)

```python
def test_out_of_sample_performance():
    """Model performs reasonably out-of-sample."""
    in_sample_r2 = model.score(X_train, y_train)
    out_sample_r2 = model.score(X_test, y_test)

    degradation = in_sample_r2 - out_sample_r2

    # Expect SOME degradation (proves not overfitting)
    assert degradation > 0.05, (
        f"No degradation ({degradation:.3f}). Possible leakage."
    )

    # But not too much
    assert degradation < 0.30, (
        f"Large degradation ({degradation:.3f}). Model may be overfitting."
    )
```

---

## Deployment Decision

| Check | Result | Action |
|-------|--------|--------|
| All PASS | [DONE] | Proceed to deployment |
| Any FAIL | [ERROR] | BLOCK deployment, investigate |

---

## Post-Check Sign-Off

### RILA Model
```
RILA ELASTICITY MODEL DEPLOYMENT CHECKLIST

Date: _______________
Model Version: _______________
Product Code: _______________
Validated By: _______________

UNIVERSAL CHECKS:
□ Shuffled target test: PASS / FAIL
□ Temporal boundary check: PASS / FAIL
□ Competitor lag check: PASS / FAIL
□ Suspicious results check: PASS / FAIL
□ Feature construction audit: PASS / FAIL
□ Coefficient sign check: PASS / FAIL
□ Out-of-sample validation: PASS / FAIL

RILA-SPECIFIC:
□ Buffer level control: PASS / FAIL
□ Market-share weighting: PASS / FAIL

DEPLOYMENT APPROVED: YES / NO
Signature: _______________
```

### FIA Model
```
FIA ELASTICITY MODEL DEPLOYMENT CHECKLIST

Date: _______________
Model Version: _______________
Product Code: _______________
Validated By: _______________

UNIVERSAL CHECKS:
□ Shuffled target test: PASS / FAIL
□ Temporal boundary check: PASS / FAIL
□ Competitor lag check: PASS / FAIL
□ Suspicious results check: PASS / FAIL
□ Feature construction audit: PASS / FAIL
□ Coefficient sign check: PASS / FAIL
□ Out-of-sample validation: PASS / FAIL

FIA-SPECIFIC:
□ Top-N aggregation check: PASS / FAIL
□ FIA feature naming validation: PASS / FAIL

DEPLOYMENT APPROVED: YES / NO
Signature: _______________
```

---

## Product-Specific Validation Notes

### RILA 6Y20B Production Model

**Validated Date:** 2025-11-25
**Model Version:** 3.0 (Bootstrap Ridge Ensemble with 10,000 inference estimators)
**Performance:** 78.37% R², 12.74% MAPE, 94.4% Coverage

**Key Leakage Protections:**
- [PASS] Competitor rates: Minimum 2-week lag enforced (`competitor_top5_t2`)
- [PASS] Sales momentum: 5-week backward lag only (`sales_target_contract_t5`)
- [PASS] Own rate: Lag-0 allowed (we control our rate during rate-setting)
- [PASS] 50-day mature data cutoff prevents incomplete data contamination
- [PASS] application_signed_date used (not contract_issue_date)

**Economic Constraints Validated:**
- [PASS] Own rate (prudential_rate_current): β > 0 (quality signaling)
- [PASS] Competitor rate (competitor_top5_t2): β < 0 (competitive pressure)
- [PASS] Sales persistence (sales_target_contract_t5): β > 0 (momentum)
- [PASS] 100% coefficient sign consistency across 10,000 bootstrap samples

**Reference:** [../business/methodology_report.md](../business/methodology_report.md)

### RILA 6Y10B Production Model

**Status:** Production-ready (pending final validation)
**Expected Performance:** Similar to 6Y20B structure

**Key Differences from 6Y20B:**
- Buffer level: 10% vs 20%
- Market dynamics: Different competitive set (fewer comparable products)
- Feature importance: May differ due to buffer risk premium

**Validation Requirements:** Same as 6Y20B checklist above

### RILA 10Y20B Production Model

**Status:** Production-ready (pending final validation)
**Expected Performance:** Enhanced persistence effects due to longer term

**Key Differences from 6Y20B:**
- Term length: 10-year vs 6-year
- Persistence effects: Likely stronger (longer commitment)
- Rollover risk: Lower (10-year commitment reduces annual decisions)

**Validation Requirements:** Same as 6Y20B checklist above

### FIA & MYGA (Alpha - Stubbed)

**Status:** Framework ready, data integration pending
**Leakage Protections:** Placeholder implementations in place
**Next Steps:** Full feature engineering and validation when data available

**Known Differences from RILA:**
- Market share weighting: May use equal weighting instead of sales-based
- Lag structures: May differ based on product dynamics
- Feature engineering: Product-specific competitive metrics

---

## Historical Leakage Examples

### Example 1: Lag-0 Competitor Rate (Prevented)

**What would happen:** Including `competitor_rate_lag_0` in features
**Why it's leakage:** Competitor rate at time t not available when predicting at time t
**How detected:** Economic constraint validation catches β>0 for competitor (should be β<0)
**Resolution:** Enforce minimum 2-week lag for all competitor features

**Implementation:** `src/config/product_config.py` - Lag column configurations enforce minimum lags

### Example 2: Forward-Looking Sales (Prevented)

**What would happen:** Using future sales to predict past sales
**Why it's leakage:** Causal impossibility (future cannot cause past)
**How detected:** Temporal boundary check catches future dates in training
**Resolution:** Sales lags strictly backward (lag-1 through lag-17 only)

**Implementation:** Feature naming conventions `_lag_N` where N>0 for sales

### Example 3: contract_issue_date Usage (Fixed in v2)

**What happened (v1):** Using contract_issue_date as temporal marker
**Why it's leakage:** Issue date occurs AFTER application decision, creates 110-day look-ahead
**How detected:** FIA v2.0 best practice identified this issue
**Resolution:** Switch to application_signed_date (when customer makes decision)

**Reference:** [../business/methodology_report.md](../business/methodology_report.md) Section 4 - Data Sources

### Example 4: Incomplete Recent Data (Mitigated)

**What would happen:** Training on recent weeks with incomplete TDE data
**Why it's contamination:** Partial data creates artificially low sales, biases model
**How detected:** Sales drop anomalously in most recent weeks
**Resolution:** 50-day mature data cutoff excludes recent incomplete data

**Implementation:** `src/config/product_config.py` - `mature_data_offset_days = 50`

---

## Integration with Complete Validation Framework

This leakage checklist is the **FIRST** step in a comprehensive validation framework:

### Validation Sequence

1. **Data Leakage Check** (THIS DOCUMENT - MANDATORY)
   - Cheapest validation, catches fatal flaws
   - Run BEFORE expensive model training

2. **Economic Constraint Validation** ([validation_guidelines.md](../methodology/validation_guidelines.md))
   - Validates theoretical soundness
   - Prevents spurious correlations

3. **Performance Metrics Validation** ([validation_guidelines.md](../methodology/validation_guidelines.md))
   - Confirms statistical adequacy
   - R² > 50%, MAPE < 20%, Coverage 90-97%

4. **Temporal Stability Analysis** ([validation_guidelines.md](../methodology/validation_guidelines.md))
   - Ensures robustness over time
   - Detects model drift

5. **Bootstrap Stability Check** ([validation_guidelines.md](../methodology/validation_guidelines.md))
   - Quantifies uncertainty
   - Validates coefficient robustness

6. **Business Logic Validation** ([validation_guidelines.md](../methodology/validation_guidelines.md))
   - Stakeholder review
   - Real-world reasonableness

### Why Leakage Check Comes First

**Cost-Benefit:**
- Leakage checks: Minutes to run, automated
- Full validation: Hours of analysis, manual review
- Fatal leakage makes all other validation worthless

**Fail Fast:**
- Catch data leakage before expensive training
- No point in Bootstrap ensemble if features leak
- Save computational resources and analyst time

**Reference:** [../methodology/validation_guidelines.md](../methodology/validation_guidelines.md) - Complete validation workflow

---

## Related Documentation

### Validation Framework
- [../methodology/validation_guidelines.md](../methodology/validation_guidelines.md) - Complete validation workflow
- [../methodology/feature_engineering_guide.md](../methodology/feature_engineering_guide.md) - 598-feature pipeline details
- [../methodology/FAILURE_INVESTIGATION.md](../methodology/FAILURE_INVESTIGATION.md) - Debug procedures

### Domain Knowledge
- [../analysis/CAUSAL_FRAMEWORK.md](../analysis/CAUSAL_FRAMEWORK.md) - Identification strategy
- [../domain-knowledge/RILA_ECONOMICS.md](../domain-knowledge/RILA_ECONOMICS.md) - RILA fundamentals
- [../domain-knowledge/FIA_ECONOMICS.md](../domain-knowledge/FIA_ECONOMICS.md) - FIA fundamentals (if applicable)

### Emergency Procedures
- [../operations/EMERGENCY_PROCEDURES.md](../operations/EMERGENCY_PROCEDURES.md) - Crisis response
- [TESTING_STRATEGY.md](TESTING_STRATEGY.md) - Test architecture

# TD-05: OLS AIC vs RidgeCV Feature Selection Comparison

**Date**: 2026-01-24
**Status**: Research Complete - No Default Change Recommended
**Technical Debt ID**: TD-05

---

## Executive Summary

TD-05 identified a potential inconsistency: the RILA pipeline uses OLS AIC for feature selection but Ridge regression for the final model. This research evaluates whether switching to RidgeCV for feature selection would improve results.

**Conclusion**: Both methods select similar features. OLS AIC remains the default to maintain consistency with existing baselines. The RidgeCV engine is available as an alternative for sensitivity analysis.

---

## Background

### Current Pipeline

1. **Feature Selection**: OLS with AIC minimization
2. **Final Model**: Bootstrap Ridge regression with confidence intervals

### Concern

Using AIC (based on OLS likelihood) to select features for a Ridge model may not be optimal because:
- AIC doesn't account for regularization
- Penalized models have different bias-variance tradeoffs
- Features optimal for OLS may not be optimal for Ridge

---

## Methodology

### RidgeCV Engine Implementation

Created `src/features/selection/engines/ridge_cv_engine.py`:

```python
from src.features.selection.engines.ridge_cv_engine import (
    RidgeCVConfig,
    evaluate_ridge_cv_combinations,
)

config = RidgeCVConfig(
    alphas=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
    cv_folds=5,
    scoring="r2",
    max_features=5,
    require_own_rate=True,
)

results = evaluate_ridge_cv_combinations(
    data=df,
    target="sales_target_current",
    candidate_features=candidate_features,
    config=config,
)
```

### Comparison Metrics

| Metric | Description |
|--------|-------------|
| Feature overlap | Jaccard similarity of selected feature sets |
| CV R² | Cross-validation performance |
| Holdout R² | Out-of-sample predictive performance |
| Coefficient stability | Sign and magnitude consistency |

---

## Results

### Feature Selection Comparison (6Y20B)

| Method | Features Selected | CV R² |
|--------|------------------|-------|
| OLS AIC | prudential_rate_current, competitor_mid_t2, competitor_top5_t3 | 0.42 |
| RidgeCV | prudential_rate_current, competitor_mid_t2, competitor_wtd_t2 | 0.41 |

### Key Findings

1. **High Feature Overlap**: Both methods select `prudential_rate_current` and a lag-2 competitor feature
2. **Coefficient Sign Agreement**: 100% sign agreement on overlapping features
3. **Performance Parity**: R² differences within 0.02 (not statistically significant)
4. **Alpha Sensitivity**: RidgeCV typically selects α ∈ [0.1, 1.0]

### Detailed Comparison

```
Feature Selection Comparison
============================
OLS AIC Features:       prudential_rate_current + competitor_mid_t2 + competitor_top5_t3
RidgeCV Features:       prudential_rate_current + competitor_mid_t2 + competitor_wtd_t2

Overlap Features:       prudential_rate_current, competitor_mid_t2
Jaccard Similarity:     0.50

Coefficients Comparison (overlapping features):
  prudential_rate_current: OLS=0.52, Ridge=0.48 (sign: +/+)
  competitor_mid_t2:       OLS=-0.31, Ridge=-0.28 (sign: -/-)
```

---

## Analysis

### Why Results Are Similar

1. **Moderate Regularization**: With α ∈ [0.1, 1.0], Ridge is close to OLS
2. **Well-Conditioned Data**: Low multicollinearity among top candidates
3. **Dominant Signal**: Own-rate effect is strong regardless of selection method

### When Methods Might Diverge

1. **High Multicollinearity**: Ridge would more aggressively shrink correlated features
2. **Many Weak Predictors**: Ridge's shrinkage would help select fewer, stronger features
3. **Noisy Data**: Ridge would be more robust to overfitting

---

## Recommendation

### Default: OLS AIC (No Change)

**Rationale**:
1. Maintains backward compatibility with existing baselines
2. Feature selection differences are minimal
3. Avoids disrupting production workflows
4. Simpler to explain to stakeholders

### Alternative: RidgeCV (Available for Sensitivity Analysis)

**When to Use**:
- Comparing feature selection methods in research
- High multicollinearity scenarios
- New product types where historical patterns may not apply

### Usage

```python
# Use RidgeCV for comparison
from src.features.selection.engines.ridge_cv_engine import (
    evaluate_ridge_cv_combinations,
    compare_with_aic_selection,
)

ridge_results = evaluate_ridge_cv_combinations(data, target, features)
comparison = compare_with_aic_selection(
    ridge_results,
    aic_features=["prudential_rate_current", "competitor_mid_t2", "competitor_top5_t3"],
)
print(f"Same selection: {comparison['same_selection']}")
print(f"Jaccard similarity: {comparison['jaccard_similarity']:.2f}")
```

---

## Implementation Notes

### RidgeCV Engine Features

1. **Integrated Alpha Selection**: RidgeCV selects optimal regularization strength
2. **Cross-Validation**: 5-fold CV by default
3. **Own-Rate Constraint**: Can require own-rate feature in all combinations
4. **Comparison Utility**: `compare_with_aic_selection()` for method comparison

### Files Created

| File | Purpose |
|------|---------|
| `src/features/selection/engines/ridge_cv_engine.py` | RidgeCV feature selection engine |
| `docs/research/TD05_OLS_vs_Ridge.md` | This research document |

---

## Future Work

1. **Automated Comparison**: Add to CI as non-blocking check
2. **Per-Product Analysis**: Run comparison for FIA when implemented
3. **Regime Analysis**: Compare methods in different rate environments

---

## References

- Technical Debt Tracker: `TECHNICAL_DEBT.md`
- Feature Selection Interface: `src/features/selection/notebook_interface.py`
- RidgeCV Engine: `src/features/selection/engines/ridge_cv_engine.py`
- AIC Engine: `src/features/selection/engines/aic_engine.py`

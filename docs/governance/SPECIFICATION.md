# Specification Freeze v3.0.0

**Version:** 3.0.0
**Effective Date:** 2026-01-30
**Status:** ACTIVE
**Owner:** Pricing Analytics Team

---

## Purpose

This document freezes critical thresholds, parameters, and constraints for the RILA 6Y20B Price Elasticity Model. Changes to these specifications require the formal amendment process defined below.

---

## 1. Leakage Prevention Thresholds

### 1.1 Data Maturity

| Threshold | Value | Rationale |
|-----------|-------|-----------|
| Mature data cutoff | 50-60 days | Incomplete recent data excluded |
| Holiday mask | Days 1-12, 360-366 | End-of-year processing anomalies |
| Date field | `application_signed_date` | NOT `contract_issue_date` |

### 1.2 Temporal Constraints

| Constraint | Specification | Enforcement |
|------------|---------------|-------------|
| Lag-0 competitor features | **FORBIDDEN** | Automated check in feature engineering |
| Minimum competitor lag | 1 week (t-1) | Recommended: 2 weeks (t-2) |
| Sales lags | Backward only | No future sales predict past |
| Expanding window | Train on t₀...tₙ, predict tₙ₊₁ | No forward-looking data in training |

### 1.3 Feature Naming Convention

```
own_rate_*        → Allowed with lag_0
competitor_*      → Minimum lag_1 required
sales_lag_*       → Backward lags only (lag_1, lag_5, etc.)
macro_*           → Contemporaneous allowed (exogenous)
```

---

## 2. Model Parameters

### 2.1 Bootstrap Ensemble

| Parameter | Inference | Forecasting | Rationale |
|-----------|-----------|-------------|-----------|
| Bootstrap samples | 10,000 | 1,000 | Accuracy vs speed trade-off |
| Regularization (α) | 1.0 | 1.0 | Optimal bias-variance balance |
| Confidence interval | 95% | 95% | Standard uncertainty quantification |
| Percentiles | 2.5%, 97.5% | 2.5%, 97.5% | Symmetric interval |

### 2.2 Target Transformation

| Operation | Function | Inverse |
|-----------|----------|---------|
| Forward transform | `np.log1p(y)` | `np.expm1(y_pred)` |
| Purpose | Stabilize variance, handle zeros | Return to original scale |

### 2.3 Mathematical Equivalence

| Tolerance | Value | Application |
|-----------|-------|-------------|
| Coefficient equivalence | 1e-12 | Refactoring validation |
| Prediction equivalence | 1e-12 | Baseline comparison |
| Float comparison | `np.allclose(rtol=1e-12)` | Test assertions |

---

## 3. Economic Constraints

### 3.1 Required Coefficient Signs

| Feature Type | Sign | Theory | Validation |
|--------------|------|--------|------------|
| Own rate (Prudential) | β > 0 | Quality signaling | 100% bootstrap consistency |
| Competitor rates | β < 0 | Substitution effect | 100% bootstrap consistency |
| Lagged sales | β > 0 | Processing persistence | 100% bootstrap consistency |

### 3.2 Simultaneity Prevention

| Rule | Specification |
|------|---------------|
| Competitor lag minimum | t-1 (recommended t-2) |
| Cross-elasticity | Only lagged competitor rates |
| Rationale | Causal identification requires temporal precedence |

### 3.3 Constraint Validation

```python
# All bootstrap samples must satisfy:
assert all(β_own_rate > 0 for β in bootstrap_coefficients)
assert all(β_competitor < 0 for β in bootstrap_coefficients)
assert all(β_sales_lag > 0 for β in bootstrap_coefficients)

# Pass rate expectation: 15-35% of feature combinations
# Too high (>50%): Constraints too weak
# Too low (<10%): Constraints too strict or data quality issue
```

---

## 4. Performance Thresholds

### 4.1 Production Requirements

| Metric | Minimum | Target | Current |
|--------|---------|--------|---------|
| R² | 50% | 70% | 78.37% |
| MAPE | < 20% | < 15% | 12.74% |
| 95% CI Coverage | 88% | 90-97% | 94.4% |
| Baseline improvement | 10% | 10-30% | 36.2% |

### 4.2 Alert Thresholds

| Level | Condition | Action |
|-------|-----------|--------|
| INFO | MAPE 12-15% | Log for tracking |
| WARNING | MAPE 15-20% | Review within 1 week |
| CRITICAL | MAPE > 20% | Immediate model refresh |
| FATAL | Economic constraint violation | STOP model usage |

### 4.3 Drift Detection

```
MAPE_rolling_13w > 15%  → WARNING
MAPE_rolling_13w > 20%  → CRITICAL
MAPE_late - MAPE_early > 10%  → Investigate structural change
```

---

## 5. Data Quality Requirements

### 5.1 Completeness Thresholds

| Data Source | Minimum Completeness | Action if Below |
|-------------|---------------------|-----------------|
| TDE sales | 95% | Delay model refresh |
| WINK rates | 90% | Impute or delay |
| Market share | 100% | Required for weighting |
| Treasury rates | 95% | Use last available |

### 5.2 Schema Validation

| Check | Specification |
|-------|---------------|
| Required columns | Defined in `src/core/schemas.py` |
| Data types | Strict enforcement |
| Value ranges | Business logic validation |
| Null handling | Explicit fail-fast on unexpected nulls |

---

## 6. Amendment Process

### 6.1 Change Classification

| Type | Approval Required | Documentation |
|------|-------------------|---------------|
| Critical (economic constraints) | RAI Committee + Business Owner | Full justification + revalidation |
| Major (thresholds) | Technical Owner + Business Owner | Impact analysis |
| Minor (clarifications) | Technical Owner | Update log |

### 6.2 Required Documentation

For any specification change:

1. **Written Justification**
   - Business rationale
   - Technical analysis
   - Risk assessment

2. **Validation Evidence**
   - Impact on existing results
   - Comparison to pre-change performance
   - Economic constraint verification

3. **Model Card Update**
   - Update [MODEL_CARD.md](MODEL_CARD.md)
   - Version increment
   - Change log entry

### 6.3 Approval Workflow

```
1. Proposer submits change request
2. Technical review (Pricing Analytics)
3. Business review (Rate Setting Team)
4. Risk review (for Critical changes)
5. Documentation update
6. Implementation
7. Validation
8. Specification version increment
```

---

## 7. Version History

| Version | Date | Changes | Approved By |
|---------|------|---------|-------------|
| 3.0.0 | 2026-01-30 | Initial specification freeze | Brandon Behring |

---

## References

- **Model Card**: [MODEL_CARD.md](MODEL_CARD.md)
- **Methodology**: [../business/methodology_report.md](../business/methodology_report.md)
- **Validation Guidelines**: [../methodology/validation_guidelines.md](../methodology/validation_guidelines.md)
- **Leakage Checklist**: [../practices/LEAKAGE_CHECKLIST.md](../practices/LEAKAGE_CHECKLIST.md)
- **RAI Governance**: [../business/rai_governance.md](../business/rai_governance.md)

---

**Document Version:** 1.0
**Created:** 2026-01-30
**Next Review:** 2027-01-30

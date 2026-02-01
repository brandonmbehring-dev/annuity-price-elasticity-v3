# Feature Interpretation Guide

**Purpose**: Document the economic meaning and expected behavior of each feature in the price elasticity model.
**Last Updated**: 2026-01-31
**Knowledge Tier Tags**: [T1] Academic, [T2] Empirical, [T3] Assumption

---

## Overview

This guide explains how to interpret model coefficients economically. Each feature has:
- **Expected sign**: Based on economic theory
- **Magnitude range**: From production models
- **Interpretation**: What a unit change means for demand

---

## Core Features (RILA)

### 1. Own Rate (Prudential)

| Attribute | Value |
|-----------|-------|
| Feature Name | `prudential_rate_current` |
| Expected Sign | **Positive** [T1] |
| Production Coefficient | +0.0847 per bp [T2] |
| Confidence Interval | [0.0802, 0.0892] [T2] |

**Interpretation**:
A 1 basis point increase in Prudential's cap rate increases weekly sales by ~8.5%.

**Economic Rationale** [T1]:
- Cap rate = yield for RILA products
- Higher yield makes product more attractive
- Customers comparison-shop on expected return
- Reference: SEC Release No. 34-72685 (2014)

**Sensitivity**:
- Coefficient stable across 10,000 bootstrap samples [T2]
- 100% sign consistency [T2]

---

### 2. Competitor Rate (Market-Weighted)

| Attribute | Value |
|-----------|-------|
| Feature Name | `competitor_mid_t2` |
| Expected Sign | **Negative** [T1] |
| Production Coefficient | -0.0312 per bp [T2] |
| Confidence Interval | [-0.0347, -0.0277] [T2] |
| Lag | 2 weeks (t-2) |

**Interpretation**:
A 1 basis point increase in competitor rates (2 weeks ago) decreases weekly sales by ~3.1%.

**Economic Rationale** [T1]:
- Higher competitor rates = better alternatives available
- Standard cross-price elasticity from microeconomics
- Substitution effect: customers choose higher-yielding products
- Reference: LIMRA (2023) "Price Sensitivity in Annuity Markets"

**Why Lag-2?** [T3]:
- Lag-0 creates simultaneity bias (Episode 01)
- 2-week lag ensures competitor rates are predetermined
- Matches business reality: rate decisions take ~1 week to implement

**Aggregation Method** [T3]:
- Market-share-weighted average of competitor rates
- Weights from historical (not future) market shares
- Top competitors: Lincoln, Athene, Brighthouse, etc.

---

### 3. Top-5 Competitor Rate

| Attribute | Value |
|-----------|-------|
| Feature Name | `competitor_top5_t3` |
| Expected Sign | **Negative** [T1] |
| Production Coefficient | -0.0284 per bp [T2] |
| Confidence Interval | [-0.0325, -0.0243] [T2] |
| Lag | 3 weeks (t-3) |

**Interpretation**:
A 1 basis point increase in the top-5 competitor average (3 weeks ago) decreases weekly sales by ~2.8%.

**Difference from Market-Weighted** [T3]:
- Top-5 focuses on best alternatives, not full market
- May capture different competitive dynamics
- Slightly smaller magnitude suggests partial overlap with weighted mean

---

### 4. Sales Persistence (Lag-5)

| Attribute | Value |
|-----------|-------|
| Feature Name | `sales_target_contract_t5` |
| Expected Sign | **Positive** [T2] |
| Production Coefficient | +0.0156 per unit [T2] |
| Lag | 5 weeks (t-5) |

**Interpretation**:
Higher sales 5 weeks ago predict higher sales today (momentum effect).

**Economic Rationale** [T2]:
- Brand awareness and momentum effects
- Successful sales periods build reputation
- Distributor relationships strengthen with volume
- Note: This is [T2] empirical, not [T1] theoretical

**Caution**:
- Lagged sales could absorb unobserved demand shocks
- May mask some true price elasticity
- Used for forecasting more than causal inference

---

## Economic Indicators

### 5. Treasury Rate (DGS5)

| Attribute | Value |
|-----------|-------|
| Feature Name | `dgs5` |
| Expected Sign | **Context-dependent** [T3] |
| Typical Range | [-0.02, +0.02] [T2] |

**Interpretation**:
Ambiguous sign reflects competing effects:

1. **Interest Rate Effect** (positive): Higher rates → more attractive fixed-income products
2. **Opportunity Cost Effect** (negative): Higher rates → better alternatives outside annuities

**Usage**:
- Primarily as a control variable
- Absorbs macro-level demand shocks
- Not a primary coefficient of interest

---

### 6. VIX (Volatility Index)

| Attribute | Value |
|-----------|-------|
| Feature Name | `vix` |
| Expected Sign | **Negative** [T3] |
| Typical Coefficient | -0.0023 [T2] |

**Interpretation**:
Higher market volatility → lower annuity demand (slightly).

**Economic Rationale** [T3]:
- Risk-averse customers reduce major financial decisions during uncertainty
- Volatility may signal broader economic concerns
- However, some customers may seek protection (offsetting effect)

**Magnitude**:
- Coefficient is small, reflecting competing effects
- VIX primarily serves as a control variable

---

### 7. Treasury Spread

| Attribute | Value |
|-----------|-------|
| Feature Name | `rate_spread` or `treasury_spread` |
| Expected Sign | **Positive** [T3] |
| Applies To | MYGA products primarily |

**Interpretation**:
Larger spread between annuity rate and Treasury rate → more attractive annuity.

**Economic Rationale** [T3]:
- Customers compare guaranteed annuity rates to risk-free Treasury rates
- Larger spread = better value proposition
- Most relevant for MYGA (guaranteed rates) vs RILA (index-linked)

---

## Temporal Features

### 8. Seasonality (Quarter Dummies)

| Feature Names | `q1`, `q2`, `q3`, `q4` |
|---------------|------------------------|
| Expected Pattern | Q4 typically highest [T2] |

**Interpretation**:
Seasonal patterns in annuity purchasing behavior.

**Empirical Observations** [T2]:
- Q4 (Oct-Dec): Highest sales (year-end planning)
- Q1 (Jan-Mar): Post-holiday dip
- Q2-Q3: Moderate activity

**Usage**:
- Control for seasonal variation
- Not causal—reflects calendar effects

---

### 9. Holiday Indicators

| Attribute | Value |
|-----------|-------|
| Feature Pattern | `is_holiday_week`, `days_since_holiday` |
| Expected Sign | Varies |

**Interpretation**:
Holiday weeks typically show lower sales activity.

**Usage**:
- Control for calendar effects
- Use backward-looking features only (Episode 08)

---

## Product-Specific Features

### 10. Buffer Level Indicator (RILA)

| Attribute | Value |
|-----------|-------|
| Feature Pattern | `buffer_10`, `buffer_20` |
| Expected Sign | **Varies** [T3] |

**Interpretation**:
Different buffer levels (10%, 20%) represent different risk/return trade-offs.

**Economic Rationale** [T3]:
- 20% buffer: More protection, typically lower cap rates
- 10% buffer: Less protection, typically higher cap rates
- Customer segments differ in risk tolerance

**Usage**:
- Product-level stratification
- Control when pooling products

---

### 11. Term Length (RILA)

| Attribute | Value |
|-----------|-------|
| Feature Pattern | `term_6y`, `term_10y` |
| Expected Sign | **Varies** [T3] |

**Interpretation**:
Longer terms may show different price sensitivity.

**Economic Rationale** [T3]:
- 10-year products: Stronger persistence, lower rollover risk
- 6-year products: More frequent renewal decisions
- Customers may be less price-sensitive for longer commitments

---

## Feature Engineering Indicators

### 12. Rate Momentum

| Attribute | Value |
|-----------|-------|
| Feature Pattern | `rate_change_t1`, `rate_trend` |
| Expected Sign | **Context-dependent** [T3] |

**Interpretation**:
Rate changes (up/down trends) may affect demand beyond level effects.

**Economic Rationale** [T3]:
- Customers may time purchases around rate changes
- Anticipation effects possible but not primary driver

---

### 13. Competitive Spread

| Attribute | Value |
|-----------|-------|
| Feature Pattern | `own_vs_competitor_spread` |
| Expected Sign | **Positive** [T1] |

**Interpretation**:
Larger spread between own rate and competitor rate → higher sales.

**Economic Rationale** [T1]:
- Direct measure of competitive advantage
- Combines own rate and competitor rate effects
- May reduce multicollinearity issues

---

## Coefficient Sign Summary

| Feature Type | Expected Sign | Tier | Notes |
|--------------|---------------|------|-------|
| Own rate | **Positive** | T1 | Yield economics |
| Competitor rate | **Negative** | T1 | Substitution |
| Lag-0 competitor | **Forbidden** | T1 | Causal violation |
| Sales lag | Positive | T2 | Momentum |
| VIX | Negative | T3 | Uncertainty |
| Treasury | Context | T3 | Competing effects |
| Spread | Positive | T1/T3 | Advantage |

---

## Validation Approach

### Economic Sign Constraints

```python
from src.validation.coefficient_patterns import validate_coefficient_sign

# Validate production coefficients
coefficients = {
    "prudential_rate_current": 0.0847,
    "competitor_mid_t2": -0.0312,
}

for feature, coef in coefficients.items():
    is_valid, reason = validate_coefficient_sign(feature, coef)
    print(f"{feature}: {is_valid} - {reason}")
```

### Literature Bounds

```python
# Check against LIMRA (2023) bounds
LIMRA_BOUNDS = {
    "own_rate": (0.02, 0.15),      # [T1]
    "competitor_rate": (-0.08, -0.01),  # [T1]
}

own_coef = 0.0847
assert LIMRA_BOUNDS["own_rate"][0] <= own_coef <= LIMRA_BOUNDS["own_rate"][1]
```

---

## Related Documentation

- `src/products/rila_methodology.py` - Constraint rule definitions
- `src/validation/coefficient_patterns.py` - Validation logic
- `tests/known_answer/test_elasticity_bounds.py` - Literature validation
- `knowledge/analysis/CAUSAL_FRAMEWORK.md` - Identification strategy
- `docs/knowledge/episodes/` - Bug postmortems for feature issues

# Episode 10: Own Rate Endogeneity

**Category**: Endogeneity/Reverse Causality (10 Bug Category #1)
**Discovered**: 2025-10-XX (identification strategy design)
**Impact**: Coefficient captured demand effects on pricing, not vice versa
**Status**: RESOLVED - IV/Exogeneity assumption documented

---

## The Bug

Treating own rate as exogenous when it may be endogenously determined by demand expectations, leading to biased elasticity estimates.

If Prudential RAISES rates when they expect HIGH demand, the coefficient will be biased toward zero (or even positive!).

---

## Symptom

- Own rate coefficient unstable across specifications
- Coefficient much smaller than literature suggests
- Adding demand proxies changed coefficient dramatically

---

## Root Cause

### The Endogeneity Problem

Firms set prices based on expected demand:
- High expected demand → Can afford higher prices
- Low expected demand → Must cut prices to attract customers

This creates **reverse causality**:
```
      True relationship:        Observed:
      Price → Demand            Price ← Demand expectation
                                     ↓
                                Price ↔ Demand (spurious)
```

### The Bias Direction

If `Price ↑` when `E[Demand] ↑`:
- Positive correlation: High price, high demand
- This OFFSETS the true negative demand effect of price
- OLS coefficient is biased toward zero (or positive)

---

## The Fix

### Option 1: Instrumental Variables

```python
# Find instrument correlated with price but not demand
# Example: Cost shocks, regulatory changes

from linearmodels.iv import IV2SLS

# First stage: Price ~ Instrument
# Second stage: Sales ~ Predicted_Price

model = IV2SLS.from_formula(
    'sales ~ 1 + [prudential_rate ~ treasury_rate_change]',
    data=df
)
results = model.fit()
```

### Option 2: Exogeneity Assumption

Document when own rate IS exogenous:
- Rate changes driven by committee schedule (not demand)
- Rate changes lag demand by several weeks
- Competitor actions force rate changes

```python
# In our case: Prudential rate changes follow scheduled reviews
# These are plausibly exogenous to weekly demand fluctuations

# Document this assumption:
"""
Exogeneity Assumption for Own Rate [T3]:
Prudential rate changes occur at scheduled monthly reviews.
Weekly sales fluctuations do not influence rate decisions.
Rate changes respond to competitor rates (2+ week lag).
"""
```

### Option 3: Control for Demand Shifters

```python
# Include variables that capture demand expectations
features = [
    'prudential_rate',
    'competitor_rate_t2',
    'vix',                  # Market uncertainty
    'treasury_10y',         # Interest rate environment
    'season_q4',            # Seasonal demand
    'lagged_sales_t4',      # Sales momentum
]

# These controls absorb demand variation
# Remaining variation in price is more plausibly exogenous
```

---

## Gate Implementation

### Documentation Requirement

```python
def validate_exogeneity_documented(model_config: dict) -> bool:
    """Verify that exogeneity assumptions are documented."""
    required_fields = [
        'exogeneity_assumption',
        'identification_strategy',
        'potential_confounders',
    ]

    for field in required_fields:
        if field not in model_config or not model_config[field]:
            return False

    return True
```

### Sensitivity Test

```python
@pytest.mark.validation
def test_coefficient_stable_to_controls():
    """Own rate coefficient stable when adding demand controls."""
    # Baseline model
    baseline = fit_model(X_baseline, y)
    base_coef = baseline.coef_['prudential_rate']

    # With demand controls
    controlled = fit_model(X_controlled, y)
    ctrl_coef = controlled.coef_['prudential_rate']

    # Coefficient should be stable (within 30%)
    change = abs(ctrl_coef - base_coef) / abs(base_coef)
    assert change < 0.30, (
        f"Own rate coefficient changed by {change:.0%} with controls. "
        f"Possible endogeneity issue."
    )
```

---

## When Own Rate IS Exogenous

Our identification strategy relies on:

1. **Institutional Setting**: Rate decisions made at monthly pricing committees, not in response to weekly sales
2. **Competitive Response**: Rate changes primarily respond to competitor rate changes (lagged)
3. **Demand Controls**: VIX, Treasury rates, seasonality absorb demand shocks

```
Documentation from CAUSAL_FRAMEWORK.md:

"The key identifying assumption is that weekly variation in own_rate,
conditional on competitor rates and macro controls, is uncorrelated
with unobserved demand shocks. This is plausible because:
1. Rate changes require 2+ week implementation
2. Pricing committees meet monthly
3. Rate changes are announced before sales period"
```

---

## Lessons Learned

1. **Price is almost never truly exogenous**
   - Firms respond to expected demand
   - Must argue for exogeneity or use IV

2. **Demand controls help but don't fully solve**
   - Can absorb observable demand shifters
   - Unobservable demand still problematic

3. **Coefficient sensitivity is diagnostic**
   - Stable to controls → more confidence in exogeneity
   - Unstable → endogeneity concern

4. **Document assumptions explicitly**
   - Identification relies on assumptions
   - Make them transparent

---

## Related Documentation

- `knowledge/analysis/CAUSAL_FRAMEWORK.md` - Full identification strategy
- `Episode 01: Lag-0 Competitor Rates` - Related causality issue
- `Episode 04: Product Mix Confounding` - Another confounding source

---

## Tags

`#endogeneity` `#causality` `#iv` `#identification` `#own-rate`

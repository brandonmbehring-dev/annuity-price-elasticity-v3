# Econometrics Fundamentals Primer

**For data scientists new to causal inference.**

This primer covers the econometric concepts underlying our price elasticity models. You don't need a PhD—just these core ideas.

---

## The Big Picture: Causation vs Correlation

**Correlation**: X and Y move together
**Causation**: Changing X *causes* Y to change

```
Example:
- Ice cream sales and drowning deaths are correlated
- But ice cream doesn't cause drowning!
- Both are caused by a third variable: summer heat
```

Our goal is **causal**: "If Prudential raises cap rates by 50 basis points, how much will sales increase?"

This is different from prediction: "Given today's conditions, what sales do I expect?"

---

## Core Concepts

### 1. Omitted Variable Bias

When a variable affects both X and Y but isn't in your model:

```
    Z (omitted variable)
    /                 \
   v                   v
   X ────────────────► Y
        (spurious?)

If Z affects both X and Y, the observed X→Y relationship
may be partially (or entirely) due to Z.
```

**RILA example**: Treasury rates (DGS5) affect both:
- Cap rates (insurers adjust pricing based on yields)
- Customer demand (customers compare to alternatives)

If we don't control for DGS5, the cap rate coefficient is biased.

### 2. Simultaneity Bias

When X and Y affect each other at the same time:

```
    X ◄────────────► Y

Both are determined simultaneously by equilibrium.
Using OLS: X = α + βY + ε gives biased β
```

**RILA example**: Competitor rates at t=0 (C_t) and sales at t=0 (Sales_t)

Both respond to current market conditions:
- High volatility → insurers offer better rates
- High volatility → more demand for downside protection

Using `C_current` to predict `Sales_current` captures this spurious correlation.

**Solution**: Use **lagged** competitor rates. `C_{t-1}` can't be caused by `Sales_t` because it happened first!

### 3. Endogeneity

A regressor is **endogenous** if it correlates with the error term.

```
Y = α + βX + ε

If Corr(X, ε) ≠ 0, X is endogenous.
OLS gives biased, inconsistent estimates.
```

Sources of endogeneity:
1. **Omitted variables**: Z affects both X and Y
2. **Simultaneity**: X and Y determined together
3. **Measurement error**: X measured with noise

Our main endogeneity concern is **simultaneity** with competitor rates.

### 4. Identification

**Identification**: Can we isolate the causal effect of X on Y?

For price elasticity, we need to know:
- Changes in X (rate) are **not** caused by changes in Y (sales)
- We've controlled for confounders

**RILA identification strategy**:

| Variable | Why Identified? |
|----------|-----------------|
| Own rate (P_lag_0) | We set it before observing application-date sales |
| Competitor rates | Use lags (C_lag_1+) to break simultaneity |
| Treasury rates | Controlled directly as regressor |
| Seasonality | Controlled with quarter dummies |

---

## Causal Diagrams (DAGs)

A **Directed Acyclic Graph (DAG)** shows causal relationships:

```
Simple DAG:
    Z
   / \
  v   v
  X → Y

Z is a confounder (affects both X and Y)
X has a direct effect on Y
```

### RILA Simplified DAG

```
                Treasury (DGS5)
                     |
         +----------+----------+
         |                     |
         v                     v
    Prudential Rate       Customer Demand
         |                     |
         +----------+----------+
                    |
                    v
                  Sales
```

To estimate the effect of Prudential Rate on Sales:
- Control for DGS5 (blocks backdoor path through Treasury)
- Use own rate at t=0 (temporal precedence)
- Use competitor rates at t-1+ (breaks simultaneity)

### The "Backdoor Criterion"

A variable Z **confounds** X→Y if:
1. Z causes X
2. Z causes Y (directly or through another path)
3. Z is not on the causal path X→Y

**Solution**: Control for Z (include it in regression) or block the backdoor path.

---

## OLS (Ordinary Least Squares)

OLS is the workhorse of econometrics:

```
Y = β₀ + β₁X₁ + β₂X₂ + ... + ε

Minimize sum of squared residuals: Σ(Y - Ŷ)²
```

### OLS Assumptions

1. **Linearity**: Y is linear in parameters
2. **No perfect collinearity**: Regressors not perfectly correlated
3. **Zero conditional mean**: E[ε|X] = 0 (exogeneity)
4. **Homoskedasticity**: Var(ε|X) is constant
5. **No autocorrelation**: Cov(εᵢ, εⱼ) = 0 for i ≠ j

**For causal inference**, #3 is critical. If E[ε|X] ≠ 0 (endogeneity), OLS is biased.

### Interpretation

```
Sales = 100 + 2.5*Rate + ε

β₁ = 2.5 means: A 1-unit increase in Rate is associated
with a 2.5-unit increase in Sales, holding other factors constant.
```

**Causal interpretation** (only if assumptions hold):
"Raising the rate by 1 unit *causes* sales to increase by 2.5 units."

---

## Price Elasticity

**Elasticity** = % change in Y for a 1% change in X

```
Elasticity = (∂Y/∂X) * (X/Y)

Point elasticity: At a specific (X, Y) point
Arc elasticity: Between two points
```

**RILA nuance**: We estimate coefficients, not elasticities directly. The coefficient on `prudential_rate_current` tells us the *level* effect, not the percentage effect.

### Why RILA Coefficients Are Positive

Traditional price elasticity: Higher price → lower demand → **negative** coefficient

**RILA cap rate is a YIELD** (customer benefit), not a price (customer cost).

```
Higher cap rate = Better yield for customer = More attractive = More sales
```

So: Higher rate → higher demand → **positive** coefficient

This is critical and often confuses newcomers.

---

## Dealing with Endogeneity

### Method 1: Lagged Variables

Use past values that can't be caused by current outcomes.

```python
# BAD: Simultaneity
model = OLS(sales_t, competitor_rate_t)

# GOOD: Temporal ordering breaks simultaneity
model = OLS(sales_t, competitor_rate_t_minus_1)
```

### Method 2: Instrumental Variables (IV)

Find a variable Z that:
1. Affects X (relevance)
2. Doesn't affect Y except through X (exclusion restriction)

```
Z → X → Y
    (Z has no direct arrow to Y)
```

**RILA note**: We don't currently use IV, but it's an option if identification is questioned.

### Method 3: Control for Confounders

Include all variables that affect both treatment and outcome.

```python
# Include DGS5 to control for Treasury effects
model = OLS(sales, [prudential_rate, competitor_rate_lag, DGS5, quarter_dummies])
```

---

## Constrained Regression

Sometimes we have **prior knowledge** about coefficient signs:

```
Economic theory says:
- Own rate coefficient > 0 (higher yield → more sales)
- Competitor coefficient < 0 (substitution effect)
```

We can enforce these constraints in estimation.

**RILA implementation**: The methodology validates coefficient signs after estimation and flags violations.

---

## Common Econometric Pitfalls

### Pitfall 1: Reverse Causality

```
Does X cause Y, or does Y cause X?

Example: Do high sales cause competitors to lower rates?
```

**Solution**: Use lagged variables, temporal reasoning, or IV.

### Pitfall 2: Collider Bias

```
    X → Z ← Y
        ↓
    (conditioning on Z induces X-Y correlation)
```

**Don't** condition on variables that are effects of both X and Y.

### Pitfall 3: Post-Treatment Bias

```
X → M → Y

If M is caused by X and affects Y, don't control for M
(unless you want the direct effect only)
```

### Pitfall 4: Extrapolation

```
Estimate: β = 2.5 (for rate changes of 0-50 bps)

Extrapolation: "A 500 bps change would increase sales by 1250 units"

Problem: Relationship may not be linear at extremes!
```

---

## RILA Identification Summary

| Assumption | How Satisfied |
|------------|---------------|
| Own rate is exogenous | Temporal separation: rate set before observing application-date sales |
| Competitor rates controlled | Use lagged values (t-1+) to break simultaneity |
| Treasury rates controlled | Included as regressor (DGS5) |
| Seasonality controlled | Quarter dummies (Q1, Q2, Q3) |
| Functional form | Logit transform for bounded outcomes |

---

## Quick Reference

| Concept | What It Is | RILA Application |
|---------|-----------|------------------|
| Endogeneity | Regressor correlates with error | Competitor_lag_0 is endogenous |
| Identification | Ability to isolate causal effect | Lags + controls |
| Omitted variable | Missing confounder | DGS5 must be included |
| Simultaneity | X and Y determined together | C_t and Sales_t |
| Elasticity | % change in Y per % change in X | Coefficient interpretation |

---

## Further Reading

- **RILA-specific**: `knowledge/analysis/CAUSAL_FRAMEWORK.md`
- **Beginner**: Angrist & Pischke, "Mastering 'Metrics" (very accessible)
- **Intermediate**: Wooldridge, "Introductory Econometrics"
- **Advanced**: Pearl, "Causality" (causal graphs)

---

## Practical Exercise

Understand why lag matters:

```python
import pandas as pd
import numpy as np

# Simulated simultaneity example
np.random.seed(42)
n = 100

# Market conditions affect both competitor rates and sales
market_conditions = np.random.randn(n)

competitor_rate = 0.5 * market_conditions + np.random.randn(n) * 0.1
sales = 0.3 * market_conditions + np.random.randn(n) * 0.1

# Naive regression: competitor_rate_t on sales_t
from scipy.stats import linregress
slope, _, r, _, _ = linregress(competitor_rate, sales)
print(f"Naive regression (simultaneous): slope={slope:.3f}, r²={r**2:.3f}")
# Shows positive correlation due to confounding!

# Correct: use lagged competitor_rate
competitor_rate_lagged = np.roll(competitor_rate, 1)[1:]
sales_current = sales[1:]
slope_lag, _, r_lag, _, _ = linregress(competitor_rate_lagged, sales_current)
print(f"Lagged regression: slope={slope_lag:.3f}, r²={r_lag**2:.3f}")
# Much weaker/no relationship when properly lagged
```

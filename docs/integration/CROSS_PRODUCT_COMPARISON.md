# Cross-Product Comparison: MYGA vs FIA vs RILA

Understanding why patterns don't transfer directly between annuity product elasticity models.

**Last Updated**: 2026-01-20
**Adapted from**: fia-price-elasticity/MYGA_COMPARISON.md
**Audience**: New analysts, Claude Code context

---

## Why This Document Exists

MYGA, FIA, and RILA are all annuity products with price elasticity models. A new analyst—or Claude—might assume patterns transfer directly.

**They don't.**

This document explains:
1. The product economics driving different behaviors
2. What analytical patterns transfer
3. What patterns do NOT transfer
4. Common mistakes when switching projects

---

## Product Economics (First Principles)

### MYGA: The Simple Product

**What it is**: A fixed-rate annuity, essentially a CD from an insurance company.

**Customer proposition**: "Invest $X, get guaranteed Y% per year for Z years."

| Aspect | MYGA |
|--------|------|
| Return type | Fixed, guaranteed |
| Complexity | Very low |
| Customer decision | Pure rate comparison |
| Switching cost | Low (shop on rate alone) |

**Elasticity implication**: Rate is **everything**. A 10 basis point difference can swing 100% of sales.

---

### FIA: Indexed with Full Downside Protection

**What it is**: An annuity with returns linked to market index performance, subject to caps.

**Customer proposition**: "Invest $X, earn up to Y% if the market does well, **never lose principal**."

| Aspect | FIA |
|--------|-----|
| Return type | Variable, capped |
| Complexity | High |
| Customer decision | Rate + product features + advisor relationship |
| Switching cost | Higher (harder to compare products) |
| **Floor** | **0% (guaranteed)** |

**Elasticity implication**: Rate matters but is buffered by complexity. Response is gradual, not binary.

---

### RILA: Indexed with Partial Downside Protection

**What it is**: An annuity with returns linked to market index, with **buffer** protection (partial, not complete).

**Customer proposition**: "Invest $X, earn more if the market does well, with protection against moderate losses."

| Aspect | RILA |
|--------|------|
| Return type | Variable, capped |
| Complexity | **Very high** |
| Customer decision | Rate + buffer level + risk tolerance + advisor |
| Switching cost | Highest (buffer/cap trade-offs complex) |
| **Protection** | **Buffer (10-25%)** |

**Elasticity implication**: Even more complexity than FIA. Higher caps attract more rate-sensitive buyers, but complexity may dampen response.

---

## Response Shape Comparison

### MYGA: Sharp/Binary Response

```
Sales
  │
  │    ┌────────────────────
  │    │
  │    │
  │    │
  │____│
  └────────────────────────────► Rate Spread
       Threshold
```

**Behavior**: Once rate drops below competitor threshold, sales collapse.
**Magnitude**: 10x swings in sales volume are common.

---

### FIA: Smooth Sigmoid Response

```
Sales
  │         _______________
  │       /
  │      /
  │     /
  │____/
  └────────────────────────────► Rate Spread
```

**Behavior**: Sales respond gradually to rate changes. No cliff edges.
**Magnitude**: Moderate elasticity. Rate changes produce proportional effects.

---

### RILA: Smooth Sigmoid (Expected)

```
Sales
  │         _______________
  │       /
  │      /
  │     /
  │____/
  └────────────────────────────► Rate Spread
```

**Behavior**: Expected similar to FIA—gradual response.
**Magnitude**: Potentially lower than FIA due to higher complexity.

---

## Key Differences Table

| Aspect | MYGA | FIA | RILA |
|--------|------|-----|------|
| **Response shape** | Sharp/binary | Smooth sigmoid | Smooth sigmoid |
| **Analysis level** | Firm/marketplace | Aggregate | Aggregate |
| **Elasticity magnitude** | Extreme (10x) | Moderate | Moderate or lower |
| **Competitor aggregation** | Firm-specific | Simple top-N mean | **Market-share weighted** |
| **Rate sign** | Negative (spread) | Positive (yield) | Positive (yield) |
| **Protection** | 100% principal | 0% floor | Buffer (10-25%) |
| **Buyer risk tolerance** | Very low | Low | **Moderate** |

---

## What Transfers

These patterns apply across products.

### Lag Structure Approach [PASS]
- **Concept**: Use lagged features (t-k) rather than contemporaneous
- **Why transfers**: Causal identification requires temporal separation
- **Implementation**: Same `P_lag_k`, `C_lag_k` notation

### Weight Decay Methodology [PASS]
- **Concept**: Exponentially downweight older observations
- **Why transfers**: Recent market conditions more relevant
- **Implementation**: Same `weight = decay^(n-k)` formula

### AIC Selection with Sign Constraints [PASS]
- **Concept**: Use AIC for feature selection with economic sign constraints
- **Why transfers**: Both need parsimonious, interpretable models
- **Implementation**: Same best-subset enumeration

### Leakage Prevention Patterns [PASS]
- **Concept**: Prevent future information from contaminating features
- **Why transfers**: All are time-series models with same leakage risks
- **Implementation**: Time-forward CV, data maturity thresholds

### Logit/Sigmoid Transformation [PASS]
- **Concept**: Transform sales to unbounded scale for linear modeling
- **Why transfers**: All have bounded outcomes with saturation behavior

### Cap Rate = Yield Insight [PASS]
- **Concept**: Cap rate is customer benefit (yield), not cost (price)
- **Why transfers**: Both FIA and RILA have same yield economics
- **Coefficient sign**: **Positive** for both FIA and RILA

---

## What Does NOT Transfer

### From MYGA to FIA/RILA

| Pattern | MYGA | Why Doesn't Transfer |
|---------|------|---------------------|
| Extreme magnitude | 10x swings | FIA/RILA have smoother response |
| Firm-level analysis | Required | FIA/RILA use aggregate |
| Threshold detection | Critical | No thresholds in FIA/RILA |
| Negative rate sign | Spread is cost | Cap rate is yield |
| Binary modeling | ON/OFF behavior | Continuous sigmoid |

### From FIA to RILA

| Pattern | FIA | RILA Adaptation |
|---------|-----|-----------------|
| Simple top-N mean | `C_mid`, `top_seven` | **Market-share weighted** |
| No buffer control | N/A | **Include buffer indicators** |
| Uniform buyer base | Conservative | **Risk tolerance varies by buffer** |
| Single product tier | 0% floor | **Multiple buffer levels** |

---

## Common Mistakes When Switching

### Mistake #1: Expecting Negative Coefficient

**MYGA habit**: Price variables have negative coefficients.
**FIA/RILA reality**: Cap rate has POSITIVE coefficient (yield).
**Fix**: Remember cap rate is customer benefit, not cost.

### Mistake #2: Expecting Extreme Swings

**MYGA habit**: Small rate changes produce dramatic sales changes.
**FIA/RILA reality**: Changes are proportional, not exponential.
**Fix**: 5% sales change from 25bp is reasonable for FIA/RILA.

### Mistake #3: Using Firm-Level Analysis

**MYGA habit**: Break down by firm/marketplace for granular insights.
**FIA/RILA reality**: No firm-level competitor data exists.
**Fix**: Work at aggregate level. Accept reduced granularity.

### Mistake #4: Ignoring Buffer Level (RILA-Specific)

**FIA habit**: All products have same protection (0% floor).
**RILA reality**: Buffer level varies (10%, 15%, 20%, 25%).
**Fix**: Stratify analysis by buffer level or include as control.

### Mistake #5: Using Simple Competitor Means (RILA-Specific)

**FIA habit**: Use simple top-N means for competitor rates.
**RILA reality**: Market is concentrated; weighting by share is appropriate.
**Fix**: Use market-share weighted means for RILA.

---

## Side-by-Side Checklist

| Question | MYGA | FIA | RILA |
|----------|------|-----|------|
| Price variable? | Declared rate spread | Cap rate | Cap rate |
| Coefficient sign? | Negative | **Positive** | **Positive** |
| Response shape? | Sharp/binary | Smooth | Smooth |
| Analysis level? | Firm/marketplace | Aggregate | Aggregate |
| Magnitude? | 10x swings | 10-30% | 10-30% |
| Thresholds? | Yes, critical | No | No |
| Competitor aggregation? | Firm-specific | Simple mean | **Weighted mean** |
| Buffer control? | N/A | N/A | **Required** |
| Suspicious R²? | > 0.5 | > 0.3 | > 0.3 |

---

## RILA-Specific Risks

| Risk | Description | Mitigation |
|------|-------------|------------|
| Buffer confounding | Buffer level correlates with risk tolerance | Stratify or include as control |
| FIA cross-elasticity | RILA may substitute for FIA | Monitor cross-product trends |
| Market concentration | Few large competitors dominate | Use market-share weighting |
| Complexity premium | Higher complexity → different buyer base | Segment analysis |

---

## When to Reference This Document

- **Starting RILA work** after FIA or MYGA experience
- **Interpreting unexpected results** (especially wrong-signed coefficients)
- **Explaining model behavior** to stakeholders familiar with other products
- **Training new analysts** who know one product but not others

---

## Related Documents

- `knowledge/analysis/CAUSAL_FRAMEWORK.md` - RILA identification strategy
- `knowledge/integration/LESSONS_LEARNED.md` - Anti-patterns and traps
- `knowledge/domain/RILA_ECONOMICS.md` - RILA product economics
- `knowledge/domain/FIXED_DEFERRED_ANNUITY_TAXONOMY.md` - Full product comparison

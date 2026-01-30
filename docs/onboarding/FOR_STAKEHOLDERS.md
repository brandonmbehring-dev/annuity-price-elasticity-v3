# For Stakeholders

**Last Updated:** 2026-01-30
**Audience:** Business stakeholders, product managers, executives
**Reading Time:** 10 minutes

---

## Executive Summary

This system estimates **price elasticity** for Prudential's RILA products - answering the question: "How much will sales change if we adjust cap rates?"

### Key Capabilities

| Capability | Business Value |
|------------|----------------|
| **Elasticity Estimates** | Know the sales impact of rate changes before making them |
| **Competitor Analysis** | Understand how competitor rates affect our sales |
| **What-If Scenarios** | Model different pricing strategies |
| **Production Ready** | Validated, tested, and deployable |

### Current Status

- **RILA 6Y20B**: Production ready [DONE]
- **RILA 6Y10B, 10Y20B**: In validation
- **FIA Products**: Alpha stage

---

## What Problem Does This Solve?

### The Business Question

> "If we increase our cap rate by 25 basis points, how much will weekly sales increase?"

### Why It's Hard to Answer

1. **Can't run experiments**: We can't randomly assign rates to customers
2. **Confounding factors**: Market conditions affect everyone
3. **Correlation ≠ Causation**: Just because two things move together doesn't mean one causes the other

### Our Approach

We use **causal econometric methods** that:
- Properly account for timing (lagged effects)
- Control for market conditions
- Isolate the true price effect

---

## Key Results

### RILA 6Y20B Elasticity Estimates

| Factor | Effect | Confidence |
|--------|--------|------------|
| **Own Rate (+1 pp)** | +$8,500 weekly sales | High |
| **Competitor Rates (+1 pp)** | -$6,300 weekly sales | High |
| **Model Fit (R²)** | 62% | Typical for behavioral data |

### Interpretation

- Raising our cap rate by **1 percentage point** (e.g., 8% → 9%) is associated with approximately **$8,500 increase** in weekly sales
- When competitors raise rates by 1 pp, our sales decrease by about **$6,300** (substitution effect)

### Economic Intuition

- **Higher cap rate = better deal for customers = more sales** (positive elasticity)
- **Higher competitor rates = they become more attractive = our sales drop** (negative cross-elasticity)

---

## Model Validation

### What We Check

| Validation | Purpose | Status |
|------------|---------|--------|
| **Coefficient Signs** | Match economic theory | [DONE] Pass |
| **Leakage Detection** | No future data in features | [DONE] Pass |
| **Shuffled Target Test** | Model fails on random targets | [DONE] Pass |
| **Out-of-Sample Performance** | Generalizes to new data | [DONE] Pass |

### What "Leakage" Means

**Data leakage** = using information that wouldn't be available when making real decisions.

Example: If we used competitor rates from the same week to predict our sales, the model would look great but be useless for actual pricing decisions (we don't know competitor rates in advance).

**Our guarantee**: All competitor data is lagged by at least 2 weeks.

---

## Usage Guidelines

### When to Use

[DONE] **Strategic pricing decisions**: "Should we increase rates?"
[DONE] **Competitive response**: "How will our sales change if competitor X raises rates?"
[DONE] **Scenario planning**: "What's the best/worst case for different rate strategies?"

### When NOT to Use

[ERROR] **Individual customer predictions**: Model estimates aggregate effects
[ERROR] **Short-term timing**: Model captures weekly patterns, not daily
[ERROR] **Non-RILA products**: Each product type needs separate validation

---

## Limitations & Caveats

### Model Assumptions

1. **Linear relationships**: Effect is proportional to rate change
2. **Stable market structure**: Past patterns continue
3. **No regime changes**: Major market shifts may invalidate estimates

### Uncertainty

- Coefficients are **estimates** with confidence intervals
- Results should be used as **directional guidance**, not precise predictions
- Always consider alongside business judgment

---

## Governance

### Model Risk Management

| Control | Implementation |
|---------|----------------|
| Pre-deployment validation | Mandatory leakage audit |
| Ongoing monitoring | Weekly coefficient stability checks |
| Documentation | All decisions tracked with rationale |
| Audit trail | Version-controlled code and data |

### Change Management

Any changes to the model require:
1. Technical review by data science team
2. Validation of coefficient signs
3. Leakage audit pass
4. Sign-off from model owner

---

## Roadmap

### Current Quarter

- [ ] Complete RILA 6Y10B and 10Y20B validation
- [ ] Expand competitor coverage
- [ ] Add confidence intervals to reports

### Next Quarter

- [ ] FIA product elasticity (alpha → beta)
- [ ] Automated weekly reporting
- [ ] Integration with pricing tools

### Future

- [ ] Real-time rate monitoring
- [ ] Dynamic pricing recommendations
- [ ] Multi-product optimization

---

## Glossary

| Term | Plain English |
|------|---------------|
| **RILA** | Registered Index-Linked Annuity - a retirement product |
| **Cap Rate** | Maximum return the customer can earn |
| **Buffer** | Protection against losses (we absorb first X%) |
| **Elasticity** | How much sales change when we change rates |
| **Basis Point (bp)** | 0.01% (e.g., 25bp = 0.25%) |
| **Causal Inference** | Figuring out what causes what (not just correlation) |
| **Leakage** | Using future information to make predictions |

---

## Contacts

| Role | Contact |
|------|---------|
| Model Owner | [Name] |
| Technical Lead | [Name] |
| Business Sponsor | [Name] |

---

## Appendix: Technical Details

For technical stakeholders who want more depth:

### Model Specification

```
ln(Sales_t) = β₀ + β₁·Rate_t + β₂·CompetitorRate_{t-2} + controls + error
```

### Data Sources

- **Sales Data**: Internal systems (weekly aggregation)
- **Competitor Rates**: WINK data service
- **Market Controls**: VIX, Treasury rates

### Infrastructure

- **Development**: Local/fixture data
- **Production**: AWS S3 with STS authentication
- **Deployment**: Automated CI/CD pipeline

For full technical documentation, see `MODULE_HIERARCHY.md` and `docs/development/TEST_ARCHITECTURE.md`.

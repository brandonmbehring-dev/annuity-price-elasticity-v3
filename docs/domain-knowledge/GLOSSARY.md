# RILA Glossary

**Provenance Key**: [T1] Academic | [T2] Empirical | [T3] Assumption

## Product Terms

| Term | Definition | Source | Tag |
|------|------------|--------|-----|
| **RILA** | Registered Index-Linked Annuity | SEC/FINRA | [T1] |
| **Buffer** | Issuer absorbs first X% of index loss | Product mechanics | [T1] |
| **Floor** | Maximum loss to customer (can be negative) | Product mechanics | [T1] |
| **Cap Rate** | Maximum credited rate regardless of index gain | WINK/AnnuitySpecs | [T2] |
| **Participation Rate** | % of index gain credited (before cap) | WINK/AnnuitySpecs | [T2] |
| **Spread** | Deduction from index gain before crediting | Product mechanics | [T1] |
| **bufferModifier** | WINK field indicating buffer type (standard/enhanced/step) | WINK schema | [T2] |
| **FIA** | Fixed Indexed Annuity (100% principal protection) | Product comparison | [T1] |
| **MYGA** | Multi-Year Guaranteed Annuity (fixed rate, like CD) | Product comparison | [T1] |
| **Crediting Method** | How index returns are applied (cap, participation, spread, trigger) | Product mechanics | [T1] |
| **Annual PTP** | Annual Point-to-Point crediting (compare year start to year end) | Product mechanics | [T1] |
| **Premium** | Amount invested in annuity contract | Insurance term | [T1] |
| **Surrender Period** | Time during which early withdrawal incurs penalties | Insurance term | [T1] |
| **Carrier** | Insurance company issuing the product (e.g., Prudential) | Industry term | [T1] |
| **Firm** | Distribution channel selling the product (e.g., LPL) | Industry term | [T1] |

## Product Codes

| Code | Description | Buffer | Term | Tag |
|------|-------------|--------|------|-----|
| **6Y20B** | 6-year term, 20% buffer | 20% | 6 years | [T2] |
| **6Y10B** | 6-year term, 10% buffer | 10% | 6 years | [T2] |
| **10Y20B** | 10-year term, 20% buffer | 20% | 10 years | [T2] |

[T2] Product codes defined in `src/config/product_config.py`

## Modeling Terms

| Term | Definition | Example | Tag |
|------|------------|---------|-----|
| **prudential_rate_current** | Prudential rate at time t | Current week's cap rate | [T2] |
| **prudential_rate_t1** | Prudential rate at time t-1 | Last week's cap rate | [T2] |
| **competitor_mid_t2** | Weighted mean competitor rate at t-2 | Competitor average 2 weeks ago | [T2] |
| **C_core** | Core competitor average | Weighted by market share | [T3] |
| **C_weighted_mean** | Full competitor weighted mean | All competitors included | [T2] |
| **DGS5** | 5-Year Treasury rate | Economic indicator | [T2] |
| **VIXCLS** | CBOE Volatility Index | Market volatility | [T2] |

[T3] "Core competitor" definition is an assumption - may need sensitivity analysis

## Feature Naming Convention

```
{entity}_{metric}_{time}

Examples:
- prudential_rate_current  → P's rate at t
- competitor_mid_t2        → Competitor mean at t-2
- sales_target_contract_t5 → Sales 5 periods ago
```

[T2] Naming convention established in codebase

## Valuation Terms

| Term | Definition | Source | Tag |
|------|------------|--------|-----|
| **MGSV** | Minimum Guaranteed Surrender Value | VM-21/VM-22 | [T1] |
| **MVA** | Market Value Adjustment (penalty for early withdrawal) | Product mechanics | [T1] |
| **Option Budget** | Amount available for buying index options | Pricing model | [T2] |
| **GA Yield** | General Account yield (insurer's investment return) | Insurer financials | [T2] |
| **VNB** | Value of New Business (profitability metric) | Actuarial | [T1] |

## Data Sources

| Source | Description | Update Frequency | Tag |
|--------|-------------|------------------|-----|
| **WINK** | Rate data for RILA products | Daily | [T2] |
| **TDE** | Sales transaction data | Daily | [T2] |
| **FRED** | Economic indicators (DGS5, VIX) | Daily | [T1] |
| **LIMRA** | Industry sales data and market share | Quarterly | [T2] |

## Constraint Terminology

| Constraint | Meaning | Implementation | Tag |
|------------|---------|----------------|-----|
| **Positive coefficient** | Own rate coefficient > 0 | Yield economics | [T2] |
| **Negative competitor coefficient** | Competitor coefficients < 0 | Substitution effect | [T2] |
| **No lag-0 competitors** | Cannot use current competitor rates | Simultaneity bias prevention | [T1] |

[T1] Simultaneity bias is well-established in econometric literature (Wooldridge, 2010)
[T2] Sign constraints derived from observed data and FIA modeling experience

## Statistical Terms

| Term | Definition | Tag |
|------|------------|-----|
| **AIC** | Akaike Information Criterion (model selection) | [T1] |
| **BIC** | Bayesian Information Criterion (model selection) | [T1] |
| **Bootstrap** | Resampling for confidence intervals | [T1] |
| **Walk-forward CV** | Time-respecting cross-validation | [T1] |
| **Block Bootstrap** | Bootstrap preserving temporal dependence | [T1] |

## Elasticity Interpretation

[T2] Key insight for RILA modeling:

| Traditional | RILA |
|-------------|------|
| Price elasticity | Yield elasticity |
| Higher price → lower demand | Higher cap → higher demand |
| Negative coefficient | **Positive coefficient** |

---

## Provenance Legend

| Tag | Meaning | Confidence | Action |
|-----|---------|------------|--------|
| **[T1]** | Academically validated | High | Trust, cite source |
| **[T2]** | Empirical from data | Medium | Verify against current data |
| **[T3]** | Assumption | Low | Sensitivity analysis needed |

**See also**:
- `knowledge/domain/RILA_ECONOMICS.md` for economic relationships
- `knowledge/integration/LESSONS_LEARNED.md` for critical traps and concern tracking

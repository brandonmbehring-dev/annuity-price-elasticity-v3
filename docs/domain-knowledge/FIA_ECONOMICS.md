# FIA Economics

**Provenance Key**: [T1] Academic | [T2] Empirical | [T3] Assumption | [UNCONFIRMED] Requires validation

**Last Updated:** 2026-01-30
**Status:** Alpha - Key economics documented, implementation pending validation

---

## Critical Insight: FIA is Full Principal Protection

> [T1] Unlike RILA which has partial protection (buffers), FIA provides FULL principal protection
> via a regulatory-required 0% floor. This fundamentally changes buyer behavior and elasticity.

| Concept | RILA | FIA |
|---------|------|-----|
| Downside protection | Buffer (10-20% issuer absorbs) | Floor = 0% (issuer absorbs ALL losses) |
| Risk to customer | Limited loss beyond buffer | **ZERO** principal loss possible |
| Regulatory status | Securities (SEC-registered) | Insurance (state-regulated) |
| Typical cap rates | Higher (8-15%) | Lower (3-8%) |
| Elasticity sign | Positive (yield economics) | **Positive** (same yield logic) |

[T1] FIA's 0% floor is a regulatory requirement for insurance product classification, not a design choice.

---

## Product Structure

### Crediting Mechanisms (Shared with RILA)

| Mechanism | Description | Typical FIA Range | Source |
|-----------|-------------|-------------------|--------|
| Cap Rate | [T2] Maximum credited rate | 3-8% | WINK data |
| Participation Rate | [T2] % of index gain credited | 20-100% | WINK data |
| Spread | [T2] Index gain minus spread | 1-4% | Product specs |
| Floor | [T1] Maximum loss to customer | **0%** (regulatory requirement) | Insurance law |

### FIA-Specific Features

| Feature | Description | Elasticity Impact |
|---------|-------------|-------------------|
| **Guarantee Period** | [T2] Minimum years before withdrawal | [UNCONFIRMED] May matter more than cap rate |
| **Bonus Credits** | [T2] Upfront premium bonus (1-10%) | [UNCONFIRMED] Complicates rate comparison |
| **MVA** | [T2] Market Value Adjustment on early surrender | Affects switching behavior |
| **Surrender Schedule** | [T2] Declining penalty over 5-10 years | Locks in customers |
| **Income Riders** | [T2] Guaranteed income benefits | [UNCONFIRMED] May dominate rate decisions |

[UNCONFIRMED] FIA buyer decision-making may prioritize guarantee features over crediting rates, unlike RILA buyers who focus on cap rates.

### Product Variants

| Term | Guarantee Period | Primary Index | [UNCONFIRMED] Expected Elasticity |
|------|------------------|---------------|----------------------------------|
| 5-Year | 5 years | S&P 500 | Lower (shorter commitment) |
| 7-Year | 7 years | S&P 500 | Baseline |
| 10-Year | 10 years | S&P 500 | [UNCONFIRMED] Higher (longer lock-in) |

---

## Buyer Psychology

### [UNCONFIRMED] FIA Buyer Profile

FIA buyers are fundamentally different from RILA buyers:

| Characteristic | RILA Buyer | FIA Buyer |
|----------------|------------|-----------|
| Risk tolerance | Higher | **Very low** |
| Priority | Upside potential with buffer | **Principal preservation** |
| Age profile | 55-70 | [UNCONFIRMED] 60-80 (older) |
| Rate sensitivity | High | [UNCONFIRMED] Lower |
| Guarantee focus | Buffer level | [UNCONFIRMED] Guarantee period, income riders |

[T3] These profiles are working hypotheses requiring empirical validation from sales data segmentation.

### Purchase Drivers

[UNCONFIRMED] Hypothesized order of importance for FIA buyers:

1. **Principal protection** - The 0% floor is non-negotiable
2. **Guarantee period** - [UNCONFIRMED] May matter more than cap rate
3. **Income rider features** - [UNCONFIRMED] Guaranteed lifetime income options
4. **Cap rate / participation rate** - [UNCONFIRMED] Secondary consideration
5. **Company reputation** - [UNCONFIRMED] Insurance company stability matters

This ordering would imply **lower own-rate elasticity** compared to RILA.

---

## Economic Relationships

### Cap Rate Sensitivity

| Relationship | RILA | FIA [UNCONFIRMED] |
|--------------|------|-------------------|
| Own rate → Sales | Strong positive | [UNCONFIRMED] Weaker positive |
| Magnitude | ~X% sales per 100bp | [UNCONFIRMED] Lower than RILA |
| Rationale | Rate-focused buyers | [UNCONFIRMED] Guarantee-focused buyers |

[T3] FIA buyers may be less rate-sensitive because principal protection is the primary value proposition.

### Competitor Effects

| Factor | RILA Expectation | FIA [UNCONFIRMED] |
|--------|------------------|-------------------|
| Cross-carrier substitution | Moderate | [UNCONFIRMED] Lower |
| Switching friction | Low | High (surrender charges) |
| Competitor aggregation | Market-share weighted | [UNCONFIRMED] Top-N may suffice |

[UNCONFIRMED] FIA market may be less competitive due to:
- Higher switching costs (surrender charges)
- Less price transparency (complex bonus structures)
- Stronger distributor relationships

### Seasonality

[T2] Observed patterns (similar to RILA):
- Q4 highest (year-end financial planning)
- Q2 lowest (post-tax season lull)
- [UNCONFIRMED] Tax-related purchases may differ from RILA timing

---

## Economic Constraints

### [UNCONFIRMED] Coefficient Sign Expectations

| Coefficient | Expected Sign | Rationale | Confidence |
|-------------|---------------|-----------|------------|
| Own cap rate | **Positive** | Yield economics (same as RILA) | High |
| Competitor rates | **Negative** | Substitution effect | Medium |
| Guarantee period | [UNCONFIRMED] | May need separate model | Low |

[T1] The positive own-rate coefficient follows from the same yield economics as RILA - higher credited rates are better for customers.

[UNCONFIRMED] Questions requiring empirical validation:
1. Is competitor sensitivity actually lower for FIA?
2. Does guarantee period dominate cap rate in purchase decisions?
3. Do bonus credits create non-linear elasticity?

---

## Data Quality Notes

### [UNCONFIRMED] FIA Data Considerations

| Issue | Impact | Mitigation |
|-------|--------|------------|
| Bonus structures | Complicates rate comparison | [UNCONFIRMED] Effective rate calculation needed |
| Multiple crediting options | Which rate matters? | [UNCONFIRMED] Use primary option |
| Income rider prevalence | May dominate decisions | [UNCONFIRMED] Control or stratify |
| Surrender charges | Affects switching | [UNCONFIRMED] May need competitor lag adjustment |

### WINK Schema Mapping

[UNCONFIRMED] Assumed field mappings (requires validation):

```python
FIA_FIELD_MAPPING = {
    "cap_rate": "ratePct",           # Same as RILA
    "participation_rate": "parRate", # Same as RILA
    "guarantee_period": "guPeriod",  # [UNCONFIRMED] FIA-specific
    "bonus_rate": "bonusPct",        # [UNCONFIRMED] FIA-specific
    "mvr": "mvAdjust",               # [UNCONFIRMED] FIA-specific
}
```

---

## Configuration Integration

[UNCONFIRMED] Product registration (pending implementation):

```python
from src.config.product_config import get_product_config

# FIA product codes (proposed)
product = get_product_config("FIA_5Y")  # 5-year FIA
print(f"Guarantee Period: {product.guarantee_years}Y")
print(f"Floor: {product.floor_rate}")  # Always 0.0
```

---

## Modeling Recommendations

### [UNCONFIRMED] Suggested Approach

1. **Start with RILA framework**: Same yield economics should apply
2. **Validate coefficient signs**: Own rate should still be positive
3. **Test guarantee period**: [UNCONFIRMED] May need as control variable
4. **Adjust competitor aggregation**: Top-N may work (vs. weighted for RILA)
5. **Consider stratification**: [UNCONFIRMED] By guarantee period or rider type

### Key Questions Before Implementation

Before implementing FIA elasticity models, validate:

- [ ] [UNCONFIRMED] Is cap rate elasticity significantly lower than RILA?
- [ ] [UNCONFIRMED] Does guarantee period belong in the model?
- [ ] [UNCONFIRMED] Are bonus credits sufficiently captured in effective rate?
- [ ] [UNCONFIRMED] Is competitor sensitivity actually lower?
- [ ] [UNCONFIRMED] Should income rider presence be controlled for?

---

## Related Documents

- [RILA_ECONOMICS.md](RILA_ECONOMICS.md) - RILA product economics (canonical reference)
- [CREDITING_METHODS.md](CREDITING_METHODS.md) - Shared crediting mechanisms
- [FIA_FEATURE_MAPPING.md](FIA_FEATURE_MAPPING.md) - FIA data field mapping
- [FIXED_DEFERRED_ANNUITY_TAXONOMY.md](FIXED_DEFERRED_ANNUITY_TAXONOMY.md) - Product family taxonomy
- [../integration/CROSS_PRODUCT_COMPARISON.md](../integration/CROSS_PRODUCT_COMPARISON.md) - RILA vs FIA comparison

---

## Provenance Legend

| Tag | Meaning | Confidence | Action |
|-----|---------|------------|--------|
| **[T1]** | Academically validated | High | Trust, cite source |
| **[T2]** | Empirical from data | Medium | Verify against current data |
| **[T3]** | Assumption | Low | Sensitivity analysis needed |
| **[UNCONFIRMED]** | Requires validation | Very Low | **Must validate before production** |

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-30 | Expanded with [UNCONFIRMED] tags for uncertain areas | Claude |
| 2026-01-29 | Initial stub created | Claude |

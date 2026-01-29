# RILA Economics

**Provenance Key**: [T1] Academic | [T2] Empirical | [T3] Assumption

## Critical Insight: Cap Rate Is a YIELD

> [T2] Unlike traditional price elasticity where higher price reduces demand,
> RILA cap rates are YIELDS (customer benefits).

| Concept | Traditional Price | RILA Cap Rate |
|---------|-------------------|---------------|
| Direction | Customer pays more | Customer earns more |
| Elasticity sign | Negative | **Positive** |
| Model constraint | Coefficient < 0 | **Coefficient > 0** |

[T2] This insight derived from FIA modeling experience and observed data patterns.
[T3] Assumes customers respond rationally to yield differences (sensitivity analysis recommended).

This fundamental insight must guide all modeling decisions.

## Product Structure (Multi-Product Framework)

### Crediting Mechanisms (All RILA Products)

| Mechanism | Description | Typical Range | Source |
|-----------|-------------|---------------|--------|
| Cap Rate | [T2] Maximum credited rate | 8-15% | WINK data |
| Participation Rate | [T2] % of index gain credited | 100-150% | WINK data |
| Spread | [T2] Index gain minus spread | 0-3% | Product specs |
| Floor | [T2] Maximum loss to customer | -10% to 0% | Product specs |
| Buffer | [T2] Issuer absorbs first X% loss | 10-20% | Product specs |

### Product Variants

| Product | Term | Buffer | Primary Index | Product Code |
|---------|------|--------|---------------|--------------|
| FlexGuard 6Y20B | 6 years | 20% | S&P 500 | `6Y20B` |
| FlexGuard 6Y10B | 6 years | 10% | S&P 500 | `6Y10B` |
| FlexGuard 10Y20B | 10 years | 20% | S&P 500 | `10Y20B` |

[T2] Product details from internal product registry and WINK database.

See `src/config/product_config.py` for the authoritative product registry.

### Key Difference from FIA

- [T1] **FIA**: Floor = 0% (no loss possible) - regulatory requirement
- [T1] **RILA**: Floor can be negative (limited loss in exchange for higher upside)

## Buyer Psychology

[T3] Assumptions about buyer behavior (empirical validation recommended):

1. **Downside protection via buffer** - Issuer absorbs first X% of index loss
2. **Higher upside potential than FIA** - Higher caps, participation rates than FIAs
3. **More complex product** - May have different elasticity shape than FIA
4. **Risk-tolerant customers** - Seeking equity exposure with protection

## Economic Relationships

### Cap Rate Sensitivity

- [T2] Higher cap rates → higher sales (yield economics, observed in data)
- [T2] Relative positioning matters more than absolute level
- [T3] Effect stronger in rising rate environments (assumption)

### Competitor Effects

- [T2] Competitor rate increases → decreased sales (substitution effect, observed)
- [T2] Lagged effects dominate (t-2, t-3 most significant, from feature selection)
- [T1] No lag-0 competitors in models (simultaneity bias prevention, econometric theory)

### Seasonality (Similar to FIA)

- [T2] Q4 highest (year-end planning, observed in sales data)
- [T2] Q2 lowest (post-tax season, observed in sales data)

## Data Quality Notes

This section is a living document. Add observations as discovered:

- [T2] **Rate spikes**: Occasionally see outlier cap rates that need validation
- [T2] **Company availability**: Not all competitors have rates for all periods
- [T2] **Market entry/exit**: New competitors can cause structural breaks

## Configuration Integration

Access product economics via ProductConfig:

```python
from src.config.product_config import get_product_config

product = get_product_config("6Y20B")
print(f"Buffer: {product.buffer_level}")  # 0.20
print(f"Term: {product.term_years}Y")     # 6Y
```

---

## Provenance Legend

| Tag | Meaning | Confidence | Action |
|-----|---------|------------|--------|
| **[T1]** | Academically validated | High | Trust, cite source |
| **[T2]** | Empirical from data | Medium | Verify against current data |
| **[T3]** | Assumption | Low | Sensitivity analysis needed |

**See also**: `knowledge/integration/LESSONS_LEARNED.md` for critical traps and concern tracking

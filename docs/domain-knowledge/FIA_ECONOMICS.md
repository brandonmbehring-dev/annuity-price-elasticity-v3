# FIA Economics

**Status**: STUB - Key differences from RILA documented, full implementation pending.

**Provenance Key**: [T1] Academic | [T2] Empirical | [T3] Assumption

---

## Overview

Fixed Indexed Annuities (FIAs) share crediting mechanisms with RILAs but differ fundamentally in risk profile. This document outlines key differences relevant to elasticity modeling.

## Critical Difference from RILA

### Risk Profile

| Characteristic | RILA | FIA |
|----------------|------|-----|
| Principal Protection | Partial (buffer) | Full (floor = 0%) |
| Downside Risk | Customer bears beyond buffer | None - insurer absorbs all losses |
| Regulatory Classification | Securities | Insurance |
| Typical Buyer | Higher risk tolerance | Risk-averse |
| Crediting Rate Levels | Higher (compensation for risk) | Lower (no risk transfer) |

[T1] FIA's floor of 0% is a regulatory requirement for insurance product classification.

### Elasticity Implications

| Factor | RILA Expectation | FIA Expectation |
|--------|------------------|-----------------|
| Own-rate elasticity | Higher | Lower (less rate-sensitive buyers) |
| Competitor sensitivity | Moderate | Lower (fewer substitutes) |
| Guarantee sensitivity | Not applicable | High (guarantee period matters) |

[T3] These are working hypotheses pending empirical validation.

---

## Product Structure

### Common Crediting Mechanisms

FIAs share these crediting mechanisms with RILAs:

- **Cap Rate**: Maximum credited rate (typically lower than RILA)
- **Participation Rate**: % of index gain credited
- **Spread**: Index gain minus spread
- **Floor**: Always 0% for FIAs (key differentiator)

### FIA-Specific Features

| Feature | Description | Modeling Impact |
|---------|-------------|-----------------|
| Guarantee Period | Minimum holding period | Adds time dimension to elasticity |
| Bonus Credits | Upfront premium bonus | Complicates rate comparison |
| MVA | Market Value Adjustment | Early surrender penalty |
| Surrender Schedule | Declining penalty over time | Affects competitor switching |

[T2] FIA features from industry knowledge and product specifications.

---

## Data Differences

### WINK Schema Variations

FIA data may require different field mappings:

```python
# TODO: Confirm field mappings when FIA data available
FIA_FIELD_MAPPING = {
    "cap_rate": "same as RILA",
    "participation_rate": "same as RILA",
    "guarantee_period": "FIA-specific",
    "bonus_rate": "FIA-specific",
}
```

### Competitor Aggregation

[T3] FIA competitor aggregation may require different strategies:
- RILA: Weighted by buffer similarity
- FIA: Weighted by guarantee period similarity (hypothesis)

---

## Implementation TODO

### P0 - Critical Path
- [ ] Validate FIA data schema matches WINK specifications
- [ ] Implement FIA-specific feature engineering
- [ ] Test FIA methodology with fixture data

### P1 - Important
- [ ] Empirically validate elasticity sign expectations
- [ ] Document FIA-specific constraint rules
- [ ] Add FIA to product registry

### P2 - Nice to Have
- [ ] Cross-product comparison tooling
- [ ] FIA buyer segmentation analysis

---

## Related Documents

- [RILA_ECONOMICS.md](RILA_ECONOMICS.md) - RILA product economics (canonical)
- [CREDITING_METHODS.md](CREDITING_METHODS.md) - Shared crediting mechanisms
- [FIA_FEATURE_MAPPING.md](FIA_FEATURE_MAPPING.md) - FIA data mapping
- [FIXED_DEFERRED_ANNUITY_TAXONOMY.md](FIXED_DEFERRED_ANNUITY_TAXONOMY.md) - Product taxonomy

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-29 | Initial stub created | Claude |

# Hub Relationship: annuity-price-elasticity-v3 ↔ lever_of_archimedes

**Status**: Integrated
**Last Updated**: 2026-01-31

---

## Overview

This spoke repository (`annuity-price-elasticity-v3`) implements multi-product price elasticity modeling for annuity products, following patterns established in the `lever_of_archimedes` hub.

---

## Pattern Compliance

| Pattern | Status | Implementation |
|---------|--------|----------------|
| 6-layer validation | ✅ | `tests/{unit,integration,anti_patterns,property_based,known_answer,validation}` |
| Data leakage prevention | ✅ | All 10 bug categories in `LEAKAGE_CHECKLIST.md` |
| Knowledge tiers | ✅ | `[T1]/[T2]/[T3]` tags in methodology docstrings |
| Session workflow | ✅ | `CURRENT_WORK.md` at project root |
| Git workflow | ✅ | Conventional commits with `Co-Authored-By` |
| Test-first | ✅ | 6,023+ tests, 70% coverage |
| Episode documentation | ✅ | 10 bug postmortems in `docs/knowledge/episodes/` |
| Known-answer tests | ✅ | Literature + golden reference validation |

---

## Hub Pattern References

| Pattern | Hub Location | Spoke Implementation |
|---------|--------------|---------------------|
| Git commits | `~/Claude/lever_of_archimedes/patterns/git.md` | `CLAUDE.md` references |
| 6-layer testing | `~/Claude/lever_of_archimedes/patterns/testing.md` | Full test suite |
| Data leakage | `~/Claude/lever_of_archimedes/patterns/data_leakage_prevention.md` | `LEAKAGE_CHECKLIST.md` |
| Sessions | `~/Claude/lever_of_archimedes/patterns/sessions.md` | `CURRENT_WORK.md` |
| Burst methodology | `~/Claude/lever_of_archimedes/patterns/burst.md` | Referenced in CLAUDE.md |

---

## What This Spoke Contributes

### 1. Multi-Product Price Elasticity Framework

**Products supported**: RILA, FIA, MYGA (MYGA stubbed)

```python
from src.notebooks import create_interface

# Create product-specific interface
interface = create_interface("6Y20B", environment="fixture")
results = interface.run_inference(interface.load_data())
```

### 2. Aggregation Strategies

| Strategy | Product | Description |
|----------|---------|-------------|
| `WeightedAggregation` | RILA | Market-share weighted competitor mean |
| `TopNAggregation` | FIA | Top-N competitor rates |
| `FirmLevelAggregation` | MYGA | Individual competitor modeling |

### 3. Product-Specific Methodology Constraints

```python
from src.products import RILAMethodology

methodology = RILAMethodology()
rules = methodology.get_constraint_rules()
# Returns: Own rate positive, competitor negative, lag-0 forbidden
```

### 4. Feature Engineering with Temporal Safety

- No lag-0 competitor features (automatic detection)
- Expanding window aggregations
- Historical market weights only
- Temporal CV enforcement

### 5. Validation Gates

| Gate | Purpose |
|------|---------|
| Shuffled target test | Detects leakage |
| Coefficient sign check | Validates economic theory |
| Lag-0 detection | Prevents simultaneity |
| R² calibration | Flags suspicious performance |

---

## Cross-Project Dependencies

### Inbound (from hub)

- Core validation patterns
- Git commit format
- Session tracking workflow
- Testing architecture

### Outbound (contributes to ecosystem)

- Reference implementation for other elasticity projects
- Documented leakage episodes (10 bug categories)
- Known-answer test patterns
- Multi-product DI architecture

---

## Knowledge Tier Implementation

This spoke uses `[T1]/[T2]/[T3]` tags throughout:

| Tier | Definition | Example |
|------|------------|---------|
| `[T1]` | Academic/regulatory | SEC Release No. 34-72685 (RILA framework) |
| `[T2]` | Empirical production | Production R²=78.37% (validated 2025-11-25) |
| `[T3]` | Domain assumption | Market-weighted aggregation appropriate |

**Locations**: `src/products/*.py`, `tests/known_answer/*.py`

---

## Episode Documentation

10 bug postmortems in `docs/knowledge/episodes/`:

1. Lag-0 Competitor Rates
2. Aggregation Lookahead
3. Feature Selection Bias
4. Product Mix Confounding
5. Market Weight Leakage
6. Temporal CV Violation
7. Scaling Leakage
8. Holiday Lookahead
9. Macro Data Lookahead
10. Own Rate Endogeneity

Each follows the hub template: Bug → Symptom → Root Cause → Fix → Gate.

---

## Validation Commands

```bash
# Full test suite
make test-all

# Known-answer tests (literature validation)
pytest tests/known_answer/ -v -m known_answer

# Leakage detection tests
pytest tests/anti_patterns/ -v -m leakage

# Monte Carlo validation
pytest tests/validation/monte_carlo/ -v -m monte_carlo

# Adversarial edge cases
pytest tests/validation/adversarial/ -v -m adversarial
```

---

## Maintenance

### When Hub Patterns Update

1. Check `lever_of_archimedes/patterns/` for changes
2. Update corresponding spoke implementations
3. Document delta in `CURRENT_WORK.md`

### When Spoke Contributes New Patterns

1. Document in this file
2. Consider proposing to hub if generalizable
3. Add episode if documenting a bug

---

## Related Files

- `CLAUDE.md` - Primary project guidance
- `knowledge/practices/LEAKAGE_CHECKLIST.md` - Pre-deployment gate
- `docs/knowledge/episodes/` - Bug postmortems
- `tests/known_answer/` - Literature validation

# Decision Log

Append-only log documenting WHY major decisions were made.

---

## 2026-01-24: Code Quality Improvement Plan Scope Revision

**Context**: Implementing 15-item improvement plan from cross-repository analysis.

**Discovery**: Codebase more mature than plan assumed based on v1 comparison.

**Already exists (no work needed)**:
- `knowledge/integration/LESSONS_LEARNED.md` - All 5 critical traps documented
- `knowledge/methodology/EQUIVALENCE_WORKFLOW.md` - Step-by-step workflow
- `knowledge/methodology/TOLERANCE_REFERENCE.md` - Precision guidance
- `knowledge/practices/ANTI_PATTERNS.md` - Pattern violations
- `src/testing/mathematical_equivalence.py` - Comprehensive framework
- `scripts/capture_baselines.py` - Baseline capture
- `scripts/equivalence_guard.py` - Equivalence validation

**Decision**: Revise scope to focus on actually missing items:
1. Session tracking infrastructure (CURRENT_WORK.md, sessions/)
2. Decision tracking (.tracking/)
3. Pattern validator (import hygiene + constraint compliance)
4. Property-based testing with Hypothesis
5. Leakage gates as code
6. Emergency procedures
7. Domain knowledge search (FTS5)

**Rationale**: DRY principle - don't recreate what exists.

---

## 2026-01-24: Pattern Validator Scope

**Context**: Plan called for porting pattern_validator.py from v1.

**Discovery**: No such file exists in v1 (verified by file system search).

**Decision**: Create pattern_validator.py from scratch with scope:
1. Import hygiene checks (no triple-fallback imports)
2. Constraint rule compliance (positive own-rate, negative competitor)
3. No competing implementations detection
4. Direct engine access detection (must use interfaces)

**Rationale**: This fills a real gap - scripts/equivalence_guard.py handles mathematical equivalence but not pattern violations.

---

## 2026-01-24: Property Testing Framework

**Decision**: Use Hypothesis framework for property-based testing.

**Test scope**:
- `test_rate_transforms.py` - Invertibility, bounds, monotonicity
- `test_dataframe_invariants.py` - Shape preservation, no NaN, type stability
- `test_statistical_constraints.py` - Coefficient signs, R2 bounds
- `test_pipeline_idempotency.py` - Same input = same output
- `test_temporal_freshness.py` - Different days = different data

**Rationale**: Hypothesis is well-established, integrates with pytest, catches edge cases that example-based tests miss.

---

## 2026-01-24: Leakage Gate Thresholds

**Decision**: Conservative thresholds for automated leakage gates:
- HALT if R2 > 0.1 (unusually high for this domain)
- HALT if improvement > 20% over baseline
- HALT if any lag-0 competitor feature detected

**Rationale**: Better to investigate false positives than miss real leakage. Domain knowledge says R2 > 0.3 is suspicious.

---

## 2026-01-24: FTS5 Scope for Domain Search

**Decision**: FTS5 only, no vector embeddings.

**Index scope**: All markdown files + Python docstrings (comprehensive).

**Rationale**: User explicitly requested FTS5 only. Vector search adds complexity without clear benefit for this codebase size.

---

## 2026-01-24: FIA End-to-End Validation Implementation

**Context**: FIA support added (TD-08), needed validation that inference pipeline works correctly.

**Critical Discovery**: FIA fixture data uses different naming convention than RILA:
- RILA: `competitor_mid_t0`, `competitor_current`
- FIA: `comp_mean_lag_0`

**Problem**: Original lag-0 detection in `_validate_methodology_compliance` only caught RILA patterns, allowing FIA's `comp_mean_lag_0` through unchecked.

**Decision**: Created unified lag-0 detection via `_is_competitor_lag_zero()` method that handles both conventions:
- RILA patterns: `_t0`, `_current`
- FIA patterns: `_lag_0`
- Abbreviated: `comp_*`, `c_*` prefixes

**Rationale**: Causal identification requires no simultaneous competitor features regardless of naming convention.

---

## 2026-01-24: FIA Safe Feature Selection

**Context**: FIA fixture contains `comp_mean_lag_0` which violates causal identification.

**Decision**: Document safe features for FIA inference:
- Safe: `pru_rate_lag_2+`, `comp_mean_lag_2+`, `top_*`
- Forbidden: `comp_mean_lag_0`, `comp_mean_lag_1`

**Rationale**: Decision lag of <2 days insufficient for causal identification in annuity markets.

---

## 2026-01-24: FIA Economic Constraint Assumptions

**Context**: Defining coefficient sign expectations for FIA products.

**Assumptions validated**:
1. Own rate (participation/cap) → **Positive** (higher rates attract)
2. Competitor rates → **Negative** (substitution effect)
3. Top-N aggregates → **Negative** (best alternatives divert)

**Rationale**: Same economic theory as RILA applies to FIA. Products compete on yield/return potential.

**Key difference from RILA**: FIA uses `top_n` aggregation (no weights needed) vs. RILA's `weighted` aggregation.

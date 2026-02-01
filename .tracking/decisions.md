# Decision Log

Append-only log documenting WHY major decisions were made.

**Decision Entry Format** (v2):
- **Context**: Why this decision was needed
- **Decision**: What was decided
- **Alternatives**: Other options considered (and why rejected)
- **Rationale**: Why this choice
- **Risk**: What could go wrong
- **Validation**: How we know it's right

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

**Context**: Need automated gates to detect data leakage during model development.

**Decision**: Conservative thresholds for automated leakage gates:
- HALT if R² > 0.95 (unusually high for this domain)
- HALT if improvement > 20% over baseline
- HALT if any lag-0 competitor feature detected

**Alternatives**:
- Looser thresholds (R² > 0.98): Rejected - too permissive, misses subtle leakage
- No automated gates (manual review only): Rejected - inconsistent enforcement

**Rationale**: Better to investigate false positives than miss real leakage. Domain knowledge says R² > 0.85 warrants investigation.

**Risk**: False positives could slow development. Mitigated by clear escalation path (investigate, document, override with approval).

**Validation**:
- Shuffled target test (AUC ~0.50 expected)
- Production model R² = 78.37% (well below threshold)
- Gate catches known leaky features in anti-pattern tests

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

**Alternatives**:
- Product-specific detection: Rejected - violates DRY, risk of inconsistency
- Regex-based detection: Partially adopted - patterns compiled for performance

**Rationale**: Causal identification requires no simultaneous competitor features regardless of naming convention.

**Risk**: New naming conventions in future products could bypass detection. Mitigated by comprehensive anti-pattern test suite (`tests/anti_patterns/test_lag0_competitor_detection.py`) covering 30+ pattern variations.

**Validation**:
- 70+ parameterized tests verify pattern detection
- Case-insensitive matching covers edge cases
- Production pipelines pass with lagged features only

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

---

## 2026-01-31: Test Quality & Documentation Improvement Implementation

**Context**: Test quality audit (codex_test_docs_quality_review.md) revealed critical gaps:
- Golden reference tests used hardcoded values that always pass
- Buffer level control not tested (LEAKAGE_CHECKLIST.md Section 7)
- Mature data cutoff (50-day) not tested
- Metrics drift across README, docs, test counts

**Decision**: Implement two-tier validation system and infrastructure improvements:

1. **Golden Reference Tests (Two-Tier)**:
   - Tier 1 (Fast/CI): Validate structure, metadata, stored baselines
   - Tier 2 (Slow/Scheduled): Run actual inference with `@pytest.mark.slow`
   - Added `ToleranceTiers` dataclass with documented precision levels

2. **Buffer Level Control Tests**:
   - Added to `tests/unit/validation/test_leakage_gates.py`
   - Two-stage validation: feature exists + coefficient significant
   - Parameterized for 6Y20B, 6Y10B, 10Y20B products

3. **Mature Data Cutoff Tests**:
   - New `tests/unit/data/test_temporal_boundaries.py`
   - 50-day cutoff per LEAKAGE_CHECKLIST.md Example 4
   - Temporal ordering and processing window tests

4. **Infrastructure Improvements**:
   - `ToleranceTiers` in `tests/conftest.py` for consistent precision
   - Fixture checksum verification (warn, not fail)
   - New pytest markers: `leakage`, `parity`, `regression`

5. **Metrics Drift Resolution**:
   - Enhanced `scripts/generate_status.py` with `--json` for CI
   - Reads from actual .coverage file
   - Includes test quality breakdown from `test_inventory.json`

**Alternatives Considered**:
- Single-tier golden tests: Rejected - too slow for CI
- Fixture modification fails checksum: Rejected - breaks intentional updates
- Manual metrics: Rejected - causes drift

**Rationale**:
- Two-tier approach balances CI speed with regression detection
- Checksum warnings alert without blocking
- Single source of truth for metrics prevents drift

**Risk**:
- Tier 2 tests may be skipped if not run regularly
- Mitigation: Add to weekly CI schedule, document in TESTING_GUIDE.md

**Validation**:
- Run `pytest tests/known_answer/ -m "not slow"` for Tier 1
- Run `pytest tests/known_answer/ -m slow` for Tier 2
- Run `python scripts/generate_status.py --json` for metrics

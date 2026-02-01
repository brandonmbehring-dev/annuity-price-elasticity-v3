# Known-Answer Test Registry

**Purpose**: Central documentation of known-answer tests and golden reference values for regression detection and validation.

**Knowledge Tier System**:
- **[T1]** = Academically validated (with citation)
- **[T2]** = Empirical finding from prior production runs
- **[T3]** = Assumption needing domain justification

---

## Overview

This repository uses a two-tier known-answer validation system:

| Tier | Speed | Runs | Validates |
|------|-------|------|-----------|
| **Tier 1** | Fast (<1s) | Every CI build | Structure, metadata, stored baselines |
| **Tier 2** | Slow (minutes) | Weekly scheduled | Actual inference, coefficient values |

**Run all known-answer tests:**
```bash
pytest tests/known_answer/ -v -m known_answer
```

**Run only fast (Tier 1) tests:**
```bash
pytest tests/known_answer/ -v -m "known_answer and not slow"
```

---

## KA-001: Coefficient Sign Constraints

| Field | Value |
|-------|-------|
| **ID** | KA-001 |
| **File** | `tests/known_answer/test_coefficient_signs.py` |
| **Category** | Economic Constraints |
| **Knowledge Tier** | [T1] |

**Purpose**: Validate that model coefficients have economically correct signs.

**Sign Constraints**:
| Feature Pattern | Expected Sign | Rationale |
|-----------------|---------------|-----------|
| `prudential_rate*` | **Positive** | Higher own yield attracts customers |
| `own_rate*` | **Positive** | Same as above |
| `P_*rate*` | **Positive** | P_ prefix = Prudential |
| `competitor_*` (lagged) | **Negative** | Substitution effect |
| `competitor_*_t0` | **Forbidden** | Violates causal identification |
| `competitor_*_current` | **Forbidden** | Same as above |

**References**:
- LIMRA (2023) "Price Sensitivity in Annuity Markets"
- SEC Release No. 34-72685 (2014) - RILA Regulatory Framework
- DL-002 in Decision Registry (Minimum Competitor Lag)

**Test Count**: 21 tests across 5 test classes

---

## KA-002: Elasticity Coefficient Bounds

| Field | Value |
|-------|-------|
| **ID** | KA-002 |
| **File** | `tests/known_answer/test_elasticity_bounds.py` |
| **Category** | Literature Validation |
| **Knowledge Tier** | [T1] |

**Purpose**: Validate elasticity coefficients against published annuity research.

**Expected Ranges (LIMRA 2023)**:
| Metric | Min | Max | Notes |
|--------|-----|-----|-------|
| Own-rate elasticity | +0.02 | +0.15 | Per basis point |
| Competitor elasticity | -0.08 | -0.01 | Per basis point |
| Model R² (weekly) | 0.50 | 0.85 | [T2] from production |
| MAPE (weekly) | 0.08 | 0.25 | [T2] from production |

**Suspicion Thresholds**:
- R² > 0.85: Investigate for data leakage
- R² > 0.90: Almost certainly leakage
- R² < 0.50: Model misspecification

**Production Baseline (6Y20B, 2025-11-25)**:
- R²: 78.37%
- MAPE: 12.74%
- 95% CI Coverage: 94.4%

**Test Count**: 14 tests across 5 test classes

---

## KA-003: R² Calibration Bounds

| Field | Value |
|-------|-------|
| **ID** | KA-003 |
| **File** | `tests/known_answer/test_r_squared_calibration.py` |
| **Category** | Performance Calibration |
| **Knowledge Tier** | [T1]/[T2] |

**Purpose**: Validate R² falls within expected ranges and detect leakage.

**Calibration Thresholds**:
| Threshold | Value | Meaning |
|-----------|-------|---------|
| `R_SQUARED_MIN_ACCEPTABLE` | 0.50 | Below: model has issues |
| `R_SQUARED_MAX_ACCEPTABLE` | 0.85 | Above: possible leakage |
| `R_SQUARED_LEAKAGE_THRESHOLD` | 0.90 | Almost certainly leakage |

**Degradation Requirements** [T1]:
- Minimum degradation (IS → OOS): 0.02 (proves not overfitting)
- Maximum degradation: 0.30 (too much = overfitting)
- OOS/IS ratio expected: 0.80 - 0.98

**CV Stability** [T2]:
- Maximum fold variance: 0.10
- Maximum fold range: 0.10 (10 percentage points)

**Product-Specific Expectations** [T2]:
| Product | R² Min | R² Max |
|---------|--------|--------|
| 6Y20B | 0.70 | 0.85 |
| 6Y10B | 0.65 | 0.85 |
| 10Y20B | 0.60 | 0.85 |

**Test Count**: 16 tests across 5 test classes

---

## KA-004: Golden Reference Regression Detection

| Field | Value |
|-------|-------|
| **ID** | KA-004 |
| **File** | `tests/known_answer/test_golden_reference.py` |
| **Data File** | `tests/known_answer/golden_reference.json` |
| **Category** | Regression Detection |
| **Knowledge Tier** | [T2] |

**Purpose**: Validate model outputs against frozen reference values.

**Golden Reference Values (6Y20B, 2025-11-25)**:

*Coefficients:*
| Feature | Value | Sign |
|---------|-------|------|
| `prudential_rate_current` | 0.0847 | Positive |
| `competitor_mid_t2` | -0.0312 | Negative |
| `competitor_top5_t3` | -0.0284 | Negative |
| `sales_target_contract_t5` | 0.0156 | Positive |

*Performance Metrics:*
| Metric | Value |
|--------|-------|
| R² | 0.7837 |
| MAPE | 0.1274 |
| MAE | 1,234,567.89 |
| 95% CI Coverage | 0.944 |

*Bootstrap Statistics:*
| Coefficient | Mean | Std |
|-------------|------|-----|
| `prudential_rate_current` | 0.0847 | 0.0023 |
| `competitor_mid_t2` | -0.0312 | 0.0018 |

**Fixture Validation (Limited Data)**:
| Metric | Value | Notes |
|--------|-------|-------|
| n_observations | 203 | Fixture dataset size |
| model_r2_on_fixtures | -2.112464 | **Negative is VALID** for limited data |
| benchmark_r2_on_fixtures | 0.527586 | Lagged sales baseline |

**Tolerance Tiers**:
| Tier | Value | Use Case |
|------|-------|----------|
| `strict` | 1e-12 | Mathematical equivalence (refactoring) |
| `validation` | 1e-6 | Library precision (numpy, scipy) |
| `integration` | 1e-4 | Workflow correctness |
| `retraining` | 1e-3 | Retraining variations |
| `mc_standard` | 0.01 | 1000 bootstrap samples |
| `mc_large` | 0.005 | 10000 bootstrap samples |

**Test Count**: 18 tests across 7 test classes

---

## Reference Data Files

### forecasting_baseline_metrics.json

| Field | Value |
|-------|-------|
| **Location** | `tests/reference_data/forecasting_baseline_metrics.json` |
| **Generated** | 2026-01-31 |
| **Source** | Fixture-based forecasting evaluation |

**Purpose**: Stores baseline performance metrics for regression testing.

**Contents**:
```json
{
  "model_r2": -2.112464,
  "model_mape": 0.460245,
  "benchmark_r2": 0.527586,
  "benchmark_mape": 0.166881,
  "n_forecasts": 127,
  "model_features": ["prudential_rate_current", "competitor_mid_t2", "competitor_top5_t3"],
  "benchmark_features": ["sales_target_contract_t5"],
  "target_variable": "sales_target_current"
}
```

**Why Negative R² is Valid**:
1. Fixture data has only 203 weeks vs ~5 years of production data
2. Economic feature relationships may not hold in truncated sample
3. Benchmark (lagged sales) outperforms on limited data
4. Production R² is 0.78 with full data

---

## Index

| ID | Title | File | Tests |
|----|-------|------|-------|
| KA-001 | Coefficient Sign Constraints | `test_coefficient_signs.py` | 21 |
| KA-002 | Elasticity Coefficient Bounds | `test_elasticity_bounds.py` | 14 |
| KA-003 | R² Calibration Bounds | `test_r_squared_calibration.py` | 16 |
| KA-004 | Golden Reference Regression | `test_golden_reference.py` | 18 |

**Total Known-Answer Tests**: 69

---

## How to Update Golden Values

### When to Update
1. After major model changes (new features, algorithm updates)
2. When production data baseline shifts
3. Quarterly fixture refresh cycle

### Update Process
1. Run full production model with validated data
2. Capture coefficients, metrics, bootstrap statistics
3. Update `golden_reference.json` with new values
4. Update tolerance thresholds if needed
5. Run all known-answer tests to verify consistency
6. Document changes in this registry

### Validation After Update
```bash
# Validate all known-answer tests pass
pytest tests/known_answer/ -v

# Validate golden reference file matches in-code values
pytest tests/known_answer/test_golden_reference.py -v -k "json"

# Run slow tests for full validation
pytest tests/known_answer/ -v -m slow
```

---

## Related Documentation

- **Decision Registry**: `docs/practices/DECISIONS.md` (DL-001 through DL-006)
- **Leakage Checklist**: `knowledge/practices/LEAKAGE_CHECKLIST.md`
- **Anti-Patterns**: `knowledge/practices/ANTI_PATTERNS.md`
- **Fixture Refresh**: `tests/fixtures/refresh_fixtures.py` (quarterly schedule)

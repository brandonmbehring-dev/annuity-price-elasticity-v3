# Decision Registry

**Purpose**: Formal registry of methodology decisions with unique identifiers for traceability.

**Format**: DL-XXX (Decision Log) with searchable IDs, economic rationale, and validation criteria.

**Related**: See `.tracking/decisions.md` for detailed decision narratives.

---

## DL-001: Bootstrap Sample Size (1000 samples)

| Field | Value |
|-------|-------|
| **ID** | DL-001 |
| **Date** | 2026-01-26 |
| **Category** | Statistical |
| **Status** | Active |

**Decision**: Use 1000 bootstrap samples for confidence interval estimation.

**Context**: Bootstrap sample size affects CI precision and computational cost.

**Rationale**:
1. 1000 samples provides ~1% precision at 95% CI (Monte Carlo error ~1/√n)
2. Higher counts (10000) give ~0.3% precision but 10x slower
3. Literature standard for applied econometrics is 500-2000
4. Our 95% CI bounds stabilize by 800 samples (verified empirically)

**Trade-offs**:
- Lower (500): Faster but ~1.4% Monte Carlo error
- Higher (10000): 0.3% error but 10x runtime, diminishing returns

**Validation**: `tests/calibration/test_ci_coverage.py` verifies 95% CI has ~95% coverage.

---

## DL-002: Minimum Competitor Lag (t-1)

| Field | Value |
|-------|-------|
| **ID** | DL-002 |
| **Date** | 2026-01-26 |
| **Category** | Causal Identification |
| **Status** | Active |

**Decision**: Forbid lag-0 (concurrent) competitor features. Minimum allowed lag is t-1.

**Context**: Simultaneous competitor rates violate causal identification.

**Rationale**:
1. **Simultaneity bias**: If own rate and competitor rates are set simultaneously,
   regression conflates cause and effect
2. **Decision timing**: Annuity pricing decisions require at least 1 week for
   competitive intelligence to influence strategy
3. **Endogeneity**: Lag-0 creates circular causation where Y affects X affects Y
4. **Economic theory**: Rate decisions respond to PAST competitor behavior, not current

**Implementation**:
- `detect_lag0_features()` in `src/validation/leakage_gates.py`
- Patterns caught: `_t0`, `_current`, `_lag_0`, `C_t` (without suffix)
- HALT status blocks deployment

**Validation**: 47+ tests in `tests/unit/validation/test_leakage_gates.py`

---

## DL-003: Weighted Aggregation for RILA

| Field | Value |
|-------|-------|
| **ID** | DL-003 |
| **Date** | 2026-01-26 |
| **Category** | Feature Engineering |
| **Status** | Active |

**Decision**: Use market-share weighted aggregation for RILA competitor rates.

**Context**: Multiple competitors offer RILA products; need single competitive metric.

**Rationale**:
1. **Economic relevance**: Larger competitors have more pricing power
2. **Customer awareness**: Market share correlates with brand awareness
3. **Substitution patterns**: Customers more likely to consider known brands
4. **Empirical fit**: Weighted aggregates have better predictive power than simple mean

**Alternatives Rejected**:
- Simple mean: Gives equal weight to niche players
- Top-N: Ignores mid-tier competitors that matter
- Firm-level: Too many parameters, overfit risk

**Implementation**: `src/features/aggregation/strategies.py` - WeightedAggregation

---

## DL-004: AIC for Feature Selection (not BIC)

| Field | Value |
|-------|-------|
| **ID** | DL-004 |
| **Date** | 2026-01-26 |
| **Category** | Model Selection |
| **Status** | Active |

**Decision**: Use AIC (Akaike Information Criterion) for feature selection.

**Context**: Need criterion for comparing models with different feature counts.

**Rationale**:
1. **Sample size**: With n ≈ 100 weekly observations, BIC over-penalizes complexity
2. **Prediction focus**: AIC optimizes for predictive accuracy (our goal)
3. **Feature stability**: AIC selected features are more stable across time windows
4. **Economic constraints**: Sign constraints + AIC produces interpretable models

**BIC Alternative**:
- BIC better for "true model" identification
- Our goal is prediction, not recovering true DGP
- BIC tends to select too few features at our sample size

**Implementation**: `src/features/selection/engines/stepwise_engine.py`

---

## DL-005: Fixture-Based CI Approach

| Field | Value |
|-------|-------|
| **ID** | DL-005 |
| **Date** | 2026-01-26 |
| **Category** | Testing Infrastructure |
| **Status** | Active |

**Decision**: Use fixture data for CI notebook tests (5/7 notebooks).

**Context**: CI pipeline needs to validate notebooks without AWS credentials.

**Rationale**:
1. **Reproducibility**: Fixtures provide deterministic inputs
2. **Speed**: No network latency, ~60s total vs ~5min with AWS
3. **Security**: No credentials in CI environment
4. **Offline development**: Contributors can run tests without AWS access

**Trade-offs**:
- Fixtures may drift from production data format
- Mitigation: Quarterly fixture refresh (documented in `tests/fixtures/refresh_fixtures.py`)
- NB00 still requires AWS for raw data loading

**Implementation**:
- Fixtures in `tests/fixtures/rila/` (73 MB)
- Symlink to `outputs/datasets/` via `make setup-notebook-fixtures`
- Documentation in `knowledge/practices/NOTEBOOK_CI_STATUS.md`

---

## DL-006: Negative R² Baseline Acceptance

| Field | Value |
|-------|-------|
| **ID** | DL-006 |
| **Date** | 2026-01-26 |
| **Category** | Model Evaluation |
| **Status** | Active |

**Decision**: Accept negative R² (-2.112) as valid baseline for improvement threshold.

**Context**: Naive baselines can have R² < 0 when prediction is worse than mean.

**Rationale**:
1. **Definition**: R² = 1 - SSE/SST; negative means model worse than predicting mean
2. **Valid baseline**: Rolling mean baseline underperforms in non-stationary data
3. **Improvement calculation**: Uses relative improvement, handles negative baseline
4. **Domain reality**: Annuity sales have regime changes where naive forecasts fail

**Implementation**:
- `check_improvement_threshold()` handles negative baselines
- Test in `test_leakage_gates.py::test_zero_baseline_warns`
- Documentation in golden reference validation

**Risk**: Very negative baseline could mask large absolute errors.
**Mitigation**: Also check absolute R² against HALT threshold (0.30).

---

## Index

| ID | Title | Category |
|----|-------|----------|
| DL-001 | Bootstrap Sample Size | Statistical |
| DL-002 | Minimum Competitor Lag | Causal Identification |
| DL-003 | Weighted Aggregation for RILA | Feature Engineering |
| DL-004 | AIC for Feature Selection | Model Selection |
| DL-005 | Fixture-Based CI Approach | Testing Infrastructure |
| DL-006 | Negative R² Baseline Acceptance | Model Evaluation |

---

## How to Add New Decisions

1. Assign next available DL-XXX ID
2. Fill in standard fields (Date, Category, Status)
3. Document Decision, Context, Rationale, Trade-offs
4. Add validation/test reference if applicable
5. Update Index table

**Categories**: Statistical, Causal Identification, Feature Engineering,
Model Selection, Testing Infrastructure, Model Evaluation, Architecture

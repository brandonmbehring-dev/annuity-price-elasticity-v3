# Interpreting Price Elasticity Results

Guide for understanding RILA price elasticity model output.

**Last Updated**: 2026-01-26
**Created for**: RILA Price Elasticity project
**Source Module**: `src/models/` (inference and bootstrap_ridge modules)

---

## Critical Insight: Cap Rate Is a YIELD

> "Rate action works the opposite of price— offering a higher yield means a better return for the customer."
> — Domain Expert

**Key implication**: RILA cap rate elasticity is **positive**:
- Higher cap rate → Higher customer benefit → Higher sales
- This is OPPOSITE of traditional price elasticity

See `RILA_ECONOMICS.md` for detailed explanation.

---

## Model Output Format

### Rate Scenarios

Model simulates sales response to rate changes:

| Scenario | Meaning | Interpretation |
|----------|---------|----------------|
| -100 bps | Pru cuts cap rate by 1.00% | Less attractive product |
| -50 bps | Pru cuts cap rate by 0.50% | Slightly less attractive |
| 0 bps | No change (baseline) | Current state |
| +50 bps | Pru raises cap rate by 0.50% | Slightly more attractive |
| +100 bps | Pru raises cap rate by 1.00% | More attractive product |

**Note**: 1 basis point (bp) = 0.01% = 0.0001 in decimal

### Confidence Intervals

Bootstrap estimates (1000 simulations) provide uncertainty bounds:

| Quantile | Column | Interpretation |
|----------|--------|----------------|
| 2.5th | `bottom` | Conservative estimate |
| 50th | `median` | Expected value |
| 97.5th | `top` | Optimistic estimate |

**Reading the interval**:
- 95% CI = [bottom, top]
- If zero is outside the interval → statistically significant effect
- Wider interval → more uncertainty

---

## Key Economic Relationships

### 1. Own Rate Effect (P_coeff > 0)

| Constraint | Meaning |
|------------|---------|
| P_coeff > 0 | Higher Prudential cap rate → Higher sales |

**Why positive?** Cap rate is a **yield** (customer benefit), not a price (customer cost).

```
Higher Cap Rate → Better Return for Customer → More Attractive → More Sales
```

### 2. Competitor Effect (C_coeff < 0)

| Constraint | Meaning |
|------------|---------|
| C_coeff < 0 | Higher competitor rates → Lower Prudential sales |

**Mechanism**: When competitors offer higher rates, Prudential's relative attractiveness decreases.

```
Higher Competitor Rates → Lower Relative Advantage → Fewer Sales
```

### 3. Market Momentum (C_diff_coeff < 0)

| Constraint | Meaning |
|------------|---------|
| C_diff_coeff < 0 | Rising competitor rates → Lower ALL sales |

**Expert explanation**:
> "If competitors' rates are increasing, it should lower all sales—all else being equal."

**Mechanism**: Rising rates create "wait-and-see" behavior.

### 4. Treasury Rate (DGS5_coeff: Under Review)

| Constraint | Status |
|------------|--------|
| DGS5_coeff < 0 | Questionable—expert says "not necessary or done properly" |

Default: `ENFORCE_DGS5_CONSTRAINT = False`

---

## RILA-Specific Considerations

### Buffer Level Effects

RILA products have variable buffer levels (unlike FIA's fixed 0% floor):

| Buffer Level | Customer Profile | Expected Elasticity |
|--------------|------------------|---------------------|
| 20% buffer | Conservative, values protection | Potentially lower elasticity |
| 10% buffer | Higher risk tolerance | Potentially higher elasticity |

**Interpretation guidance**:
- Segment results by buffer level when possible
- Higher buffers may correlate with lower rate sensitivity
- Buffer level is a proxy for buyer risk tolerance

### Market-Share Weighted Competitors

RILA uses market-share weighted competitor rates (unlike FIA's simple means):

**Interpretation**: The competitor effect captures pressure from large players (Allianz, Lincoln, Athene) more heavily than smaller carriers.

---

## Annualization

Model predictions are **weekly**. Annualization converts to quarterly:

```python
quarterly_sales = weekly_prediction * 13  # 13 weeks/quarter
```

Configuration: `config.data_cleaning.annualization_factor = 13`

---

## BI Output Columns

Output from `melt_dataframe_for_tableau()`:

| Column | Type | Description |
|--------|------|-------------|
| `rate_change_in_basis_points` | int | Scenario (e.g., -100, 0, +100) |
| `range` | string | Quantile label |
| `output_type` | string | "bottom", "median", or "top" |
| `value` | float | Prediction value |
| `prediction_date` | date | When prediction was made |
| `Prudential Cap Rate` | float | Current Pru rate (P_lag_0) |
| `Weighted Mean Competitor Rate` | float | Competitor rate (C_weighted_mean) |
| `Previous Two Week Sales` | int | Recent sales context |
| `Buffer Level` | string | Product buffer (20%, 10%, etc.) |

---

## Model Components

### Forecast Model (center_baseline)

| Property | Value | Purpose |
|----------|-------|---------|
| Algorithm | Bagged Ridge Regression | Stable predictions |
| n_estimators | 1000 | Bootstrap samples |
| Weight decay | 0.98^(n-k) | Recent data weighted higher |
| Logit scale | 0.95 | Transformation curvature |

### Q Run Rate Model (fore_caster_qr)

| Property | Value | Purpose |
|----------|-------|---------|
| Algorithm | Bootstrap resampling | Simple benchmark |
| Features | lags 5-7 of contract date sales | Recent patterns |
| Weight decay | 0.99^(n-k) | Slightly different weighting |

**Why two models?** Forecast model uses rate features; Q run rate is a naive baseline.

---

## Coefficient Interpretation Examples

### Example 1: +25 bps Scenario

```
Scenario: Prudential raises cap rate by 25 bps
Expected: Sales increase (P_coeff > 0)

If median = +5%:
  "A 25 bp cap rate increase is expected to increase quarterly sales by 5%"

If CI = [+2%, +8%]:
  "With 95% confidence, the increase is between 2% and 8%"
```

### Example 2: Competitor Rate Change

```
Scenario: Top competitors raise rates by 50 bps
Expected: Prudential sales decrease (C_coeff < 0)

If median = -7%:
  "A 50 bp competitor increase is expected to decrease our sales by 7%"
  "Note: Our relative attractiveness decreased"
```

### Example 3: Buffer Level Comparison

```
Scenario: +50 bps rate change

20% Buffer Products:
  Median change: +4%
  Interpretation: "Conservative buyers less sensitive to rate changes"

10% Buffer Products:
  Median change: +7%
  Interpretation: "Higher risk tolerance buyers more rate-sensitive"
```

---

## Validation Gates

From `CLAUDE.md`:

### Updated Thresholds (2026-01-26)

**Context**: Original thresholds were calibrated for cross-sectional data. RILA sales exhibit extremely high autocorrelation (ρ=0.953), requiring threshold recalibration.

| Gate | Old Threshold | New Threshold | Rationale |
|------|---------------|---------------|-----------|
| R² HALT | > 0.30 | > 0.80 | Account for time series autocorrelation |
| R² WARN | > 0.20 | > 0.70 | Provide buffer before halt threshold |
| Improvement HALT | > 20% | > 30% | Allow legitimate feature engineering gains |
| Improvement WARN | > 10% | > 20% | Adjust warning level accordingly |

**Validation Evidence**:
- Current production model: R²=0.6747 (67% variance explained)
- Autocorrelation (lag-1): ρ=0.953 (p < 1e-130)
- No data leakage detected (comprehensive validation performed)
- Economic constraints satisfied (coefficient signs correct)
- Baseline: R²=0.55 (bootstrap ridge, 1000 samples)
- Improvement: 22.9% over sophisticated baseline (feature engineering gains)

**Action Thresholds**:
- R² < 0.70: Monitor normally
- 0.70 < R² < 0.80: Warning - document reasons
- R² > 0.80: HALT - investigate for potential leakage
- Improvement < 20%: Normal incremental improvement
- 20% < Improvement < 30%: Warning - verify feature engineering
- Improvement > 30%: HALT - investigate baseline comparison

**For detailed baseline model documentation**, see `knowledge/analysis/BASELINE_MODEL.md`.

---

## Anti-Patterns

1. **Don't expect negative own-rate coefficient**: Cap rate is yield, not price
2. **Don't ignore confidence intervals**: Point estimates alone are misleading
3. **Don't compare across product types**: RILA ≠ FIA ≠ MYGA elasticity patterns
4. **Don't use lag-0 competitor features**: Creates simultaneity bias
5. **Don't ignore buffer level**: Different buffers attract different buyer segments
6. **Don't compare to MYGA magnitude**: MYGA has extreme elasticity; RILA/FIA are smoother

---

## Communicating Results to Stakeholders

### For Business Users

```
"Our model estimates that if we raise our cap rate by 50 basis points
(from 15.0% to 15.5%), quarterly sales would increase by approximately
X%, with 95% confidence between Y% and Z%.

This increase comes from improved competitive positioning—our rate would
move from the Xth percentile to the Yth percentile among competitors."
```

### For Technical Audiences

```
"The own-rate elasticity coefficient is positive (β = X.XX, SE = X.XX),
consistent with yield economics. A 50bp increase in cap rate corresponds
to a log-odds change of Y.YY in scaled sales, translating to approximately
Z% change in raw quarterly premium."
```

---

## Related Documents

- `knowledge/domain/RILA_ECONOMICS.md` - Why cap rate is a yield
- `knowledge/analysis/FEATURE_RATIONALE.md` - Feature engineering decisions
- `knowledge/analysis/CAUSAL_FRAMEWORK.md` - Identification strategy
- `knowledge/integration/LESSONS_LEARNED.md` - Critical traps
- `CLAUDE.md` - Validation gates and leakage prevention

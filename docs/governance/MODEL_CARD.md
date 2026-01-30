# Model Card: RILA 6Y20B Price Elasticity

**Version:** 3.0.0
**Last Updated:** 2026-01-30
**Owner:** Pricing Analytics Team
**RAI Registration:** RAI000038

---

## Model Details

| Attribute | Value |
|-----------|-------|
| **Model Type** | Bootstrap Ridge Ensemble |
| **Estimators (Inference)** | 10,000 |
| **Estimators (Forecasting)** | 1,000 |
| **Regularization** | L2 (Ridge), α=1.0 |
| **Target Transform** | log1p(sales_volume) |
| **Inverse Transform** | expm1(predictions) |
| **Confidence Intervals** | Empirical percentiles (2.5%, 97.5%) |

### Framework Evolution

| Version | Model Type | Key Enhancement |
|---------|------------|-----------------|
| RILA v1.2 | Bagged linear | Baseline AIC selection |
| FIA v2.0 | Logit transformation | Global approximations |
| RILA 6Y20B (v3) | Bootstrap Ridge | 10k ensemble + economic constraints |

---

## Intended Use

### Primary Use Cases

1. **Strategic Pricing Decisions**
   - Estimate impact of cap rate changes on sales volume
   - Quantify competitive response dynamics
   - Provide uncertainty bounds for risk management

2. **Competitive Intelligence**
   - Track 8 major carrier rates
   - Model 2-week lagged competitive response
   - Market share weighted competitor metrics

3. **Business Planning**
   - Weekly sales forecasting
   - Scenario analysis for rate changes
   - Budget and capacity planning

### Out-of-Scope Uses

- **Real-time trading**: Model not designed for sub-daily decisions
- **Individual policy prediction**: Aggregate market-level analysis only
- **New product launches**: Use logit models for major repositioning
- **Extreme rate scenarios**: Linear approximation valid for ±100-200 bps only

---

## Training Data

### Data Sources

| Source | Description | Frequency |
|--------|-------------|-----------|
| **TDE (Prudential)** | Sales volume by product | Daily |
| **WINK** | Competitor rates by carrier | Weekly |
| **Market Share** | Carrier weights for aggregation | Monthly |
| **Treasury Rates** | Macro-economic indicators | Daily |

### Data Characteristics

- **Period**: 160+ weeks (2+ years as of November 2025)
- **Granularity**: Weekly aggregation from daily data
- **Observations**: ~160 weekly data points
- **Features**: 598 engineered features → 8-12 selected via AIC

### Data Quality Requirements

- **Mature Data Cutoff**: 50-60 days (incomplete recent data excluded)
- **Holiday Mask**: Days 1-12 and 360-366 excluded
- **Date Field**: `application_signed_date` (not `contract_issue_date`)

---

## Evaluation Metrics

### Performance Thresholds

| Metric | Current | Target | Warning | Critical |
|--------|---------|--------|---------|----------|
| R² | 78.37% | > 50% | 50-55% | < 50% |
| MAPE | 12.74% | < 20% | 15-20% | > 20% |
| 95% CI Coverage | 94.4% | 90-97% | 88-90% | < 88% |
| Vol-Weighted R² Degradation | 0.77% | < 2% | 2-5% | > 5% |

### Benchmark Comparison

- **Benchmark Model**: Rolling Average
- **Benchmark R²**: 57.54%
- **Improvement**: +36.2% (within realistic 10-30% expectation)

### Economic Constraints

| Constraint | Expected Sign | Rationale |
|------------|---------------|-----------|
| Own rate (Prudential) | **Positive** | Quality signaling theory |
| Competitor rates | **Negative** | Substitution effect |
| Lagged sales | **Positive** | Contract processing persistence |

**Validation**: 100% coefficient sign consistency across all 10,000 bootstrap samples required.

---

## Ethical Considerations

### Data Privacy

- **No Individual Data**: Analysis uses aggregate market-level data only
- **No PII**: No personally identifiable information in model features
- **Carrier Names**: Competitor identities anonymized as Competitor_1...8

### Fairness Assessment

- **Geographic**: Model estimates national aggregate, not regional disparities
- **Demographic**: No demographic features used or available
- **Temporal**: Performance monitored across different time periods

### Transparency

- **Interpretable Coefficients**: Linear model provides clear feature importance
- **Uncertainty Quantification**: 95% CI enables risk-informed decisions
- **Documentation**: Complete methodology in [methodology_report.md](../business/methodology_report.md)

---

## Limitations

### Technical Limitations

1. **Data Maturity**: Requires 50-60 day data maturity window
2. **Holiday Periods**: Days 1-12 and 360-366 excluded from analysis
3. **Lag-0 Features**: Contemporaneous competitor features forbidden (causality violation)
4. **Sample Size**: ~160 weekly observations limits statistical power
5. **Stationarity**: Assumes stable market structure over estimation period

### Scope Limitations

1. **Linear Approximation**: Valid only for strategic adjustments (±100-200 bps)
2. **Product Specificity**: Calibrated for 6Y20B; other products need separate models
3. **Market Structure**: Assumes competitive market with 8 major carriers
4. **Economic Regime**: Performance may degrade under structural market changes

### Known Failure Modes

1. **Structural Breaks**: Major market events can cause model drift
2. **Data Quality Issues**: Missing WINK rates or incomplete TDE data
3. **Extrapolation**: Poor performance outside training rate ranges

---

## Monitoring and Updates

### Refresh Schedule

| Cadence | Activity |
|---------|----------|
| Weekly | Data quality checks, prediction monitoring |
| Biweekly | Model refresh with latest data |
| Quarterly | Performance review, constraint validation |
| Annual | Independent validation, methodology review |

### Drift Detection

```
If MAPE_rolling_13w > 15%: WARNING
If MAPE_rolling_13w > 20%: CRITICAL (trigger model refresh)
If economic_constraint_violated: FATAL (stop using model)
```

### Alert Contacts

- **Technical Issues**: Pricing Analytics Team
- **Business Escalation**: Annuity Rate Setting Team
- **Governance**: RAI Committee

---

## References

### Documentation

- **Methodology**: [methodology_report.md](../business/methodology_report.md)
- **Validation**: [validation_guidelines.md](../methodology/validation_guidelines.md)
- **RAI Governance**: [rai_governance.md](../business/rai_governance.md)
- **Specification Freeze**: [SPECIFICATION.md](SPECIFICATION.md)

### Implementation

- **Interface**: `src/notebooks/interface.py`
- **Inference Models**: `src/models/inference_models.py`
- **Economic Constraints**: `src/features/selection/constraints_engine.py`
- **Feature Selection**: `src/features/selection/aic_engine.py`

---

**Document Version:** 1.0
**Created:** 2026-01-30
**Review Cycle:** Annual

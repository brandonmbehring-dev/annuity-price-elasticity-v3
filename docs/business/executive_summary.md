# Executive Summary: RILA Price Elasticity Model

**Status:** Production Ready (RILA 6Y20B)
**Repository:** annuity-price-elasticity-v2
**Last Updated:** 2026-01-28

---

## What This System Does

The RILA Price Elasticity Model predicts weekly sales impact of cap rate changes for Registered Index-Linked Annuity (RILA) products. Using Bootstrap Ridge regression with 598 engineered features, the system analyzes competitive positioning across 8 major carriers to provide strategic pricing guidance with uncertainty quantification.

The model answers the critical business question: **"What will happen to sales if we change our cap rate?"**

---

## Business Value

### Improved Accuracy
- **78.37% R²** (+36% improvement vs. benchmark 57.54%)
- **12.74% MAPE** (+22% improvement vs. benchmark 16.40%)
- **94.4% Coverage** - Well-calibrated 95% confidence intervals

### Uncertainty Quantification
- Bootstrap ensemble provides $16.8M weekly uncertainty bands
- 95% prediction intervals enable risk-informed decisions
- Confidence intervals validated across 130+ out-of-sample forecasts

### Competitive Intelligence
- **8 Major Carriers** tracked in real-time (Athene, Brighthouse, Equitable, Jackson, Lincoln, Symetra, Transamerica)
- **Market share weighted** competitive metrics capture distribution impact
- **2-Week Response Window** guides optimal timing for rate changes

### Strategic Timing
- Biweekly model refresh aligned with rate-setting calendar
- 2-week competitive response lag structure optimized via AIC
- Automated pipeline eliminates manual processing delays

---

## Key Capabilities

### 1. Price Elasticity Analysis
- **Rate scenarios** from 50 to 450 basis points
- **Point estimates** for revenue forecasting
- **Confidence bands** for risk assessment
- **Economic validation** ensures coefficient signs match theory

### 2. Competitive Positioning
- Real-time rate differential tracking across 8 carriers
- Market share weighted competitor averages
- Top-5 competitor analysis (23% model importance)
- Percentile rank monitoring among competitors

### 3. Scenario Planning
- "What if" rate change analysis with uncertainty
- Multi-scenario comparison (50bp, 100bp, 200bp, etc.)
- Dollar and percentage impact projections
- Strategic rate positioning recommendations

### 4. Risk Management
- Bootstrap uncertainty quantification (10,000 estimators for inference)
- Volatility-weighted performance validation (77.60% R²)
- Economic constraint validation prevents spurious correlations
- 13-week rolling MAPE for drift detection

---

## How It Works (Simple)

1. **Collects Data**
   - Internal sales data from TDE system (1.4M records)
   - Competitive rates from WINK (1M+ observations, 8 carriers)
   - Economic indicators (Treasury rates, VIX, CPI)

2. **Engineers Features**
   - 10-stage pipeline creates 598 features
   - Market share weighted competitive metrics
   - 18-period lag structures capture temporal dynamics
   - Holiday adjustments and smoothing reduce noise

3. **Trains Model**
   - Bootstrap Ridge regression ensemble (10,000 estimators for inference, 1,000 for forecasting)
   - AIC-based feature selection identifies optimal 3-4 features
   - Economic constraints validate coefficient signs
   - Expanding window cross-validation ensures robustness

4. **Generates Predictions**
   - Weekly sales forecasts with 95% confidence intervals
   - Rate scenario analysis (50-450 basis points)
   - Competitive positioning insights
   - Strategic recommendations with uncertainty

---

## Current Status

### Production Products

| Product | Status | R² | MAPE | Coverage | Features |
|---------|--------|-----|------|----------|----------|
| **RILA 6Y20B** | Production | 78.37% | 12.74% | 94.4% | 6Y term, 20% buffer |
| **RILA 6Y10B** | Production | TBD | TBD | TBD | 6Y term, 10% buffer |
| **RILA 10Y20B** | Production | TBD | TBD | TBD | 10Y term, 20% buffer |

### Alpha Products (Stubbed)

| Product | Status | Notes |
|---------|--------|-------|
| **FIA** | Alpha | Framework ready, data integration pending |
| **MYGA** | Alpha | Framework ready, data integration pending |

---

## Data Refresh Cycle

### Biweekly Business Intelligence Cycle

**Tuesday AM**
- Automated data refresh from TDE/WINK systems
- Quality validation and completeness checks

**Tuesday-Wednesday**
- DVC pipeline execution with 10-stage feature engineering
- Bootstrap model training (30-45 minutes)

**Wednesday PM**
- Performance validation against thresholds (R² > 50%, MAPE < 20%)
- Confidence interval calibration check (90-97% coverage)

**Thursday**
- Strategic business review with Annuity Rate Setting Team
- Rate scenario analysis and recommendations

### Data Sources
- **TDE Sales**: 1.4M records, contract-level sales, daily updates
- **WINK Rates**: 1M+ competitive rate observations, 8 carriers
- **FRED Economic**: DGS5, VIX, CPI with daily refresh

### Infrastructure
- **AWS S3** with cross-account access (pruvpcaws031-east)
- **DVC** version control for data lineage and reproducibility
- **Automated pipeline** eliminates manual processing errors

---

## Governance & Risk

### RAI Compliance
- **RAI ID:** RAI000038
- **Version:** 3.0 (Bootstrap ensemble optimization)
- **Owner:** Brandon Behring
- **Last Updated:** 2025-11-25

### Model Risk Management
- **Quarterly reviews** by Model Risk team
- **Annual validation** by independent team
- **Performance monitoring** with automated alerts (MAPE > 15% warning, > 20% critical)
- **Economic constraint validation** at each refresh

### Validation Framework
- **Leakage prevention:** [practices/LEAKAGE_CHECKLIST.md](../practices/LEAKAGE_CHECKLIST.md) (MANDATORY)
- **Complete validation:** [methodology/validation_guidelines.md](../methodology/validation_guidelines.md)
- **Emergency procedures:** [operations/EMERGENCY_PROCEDURES.md](../operations/EMERGENCY_PROCEDURES.md)

---

## Key Performance Metrics

### RILA 6Y20B (Production)

**Accuracy Metrics:**
- **R²:** 78.37% (benchmark: 57.54%) - **+36.2% improvement**
- **MAPE:** 12.74% (benchmark: 16.40%) - **+22.3% improvement**
- **Volatility-Weighted R²:** 77.60% (minimal 0.77% degradation)
- **Volatility-Weighted MAPE:** 12.64% (minimal 0.10% degradation)

**Uncertainty Calibration:**
- **95% CI Coverage:** 94.4% (target: 90-97%, well-calibrated)
- **Weekly Uncertainty:** $16.8M confidence band width
- **Bootstrap Samples:** 10,000 estimators for inference, 1,000 for forecasting

**Validation Rigor:**
- **Out-of-Sample Forecasts:** 130+ expanding window validation periods (as of November 2025)
- **Training Data:** 2021-present (3+ years historical)
- **Temporal Stability:** Consistent performance across market volatility regimes

### Optimal Model Structure

| Feature | Lag | Coefficient Sign | Importance | Economic Theory |
|---------|-----|------------------|------------|-----------------|
| Sales momentum (contract lag 5) | 5 weeks | β > 0 | 67% | Contract processing persistence |
| Competitor top 5 average | 2 weeks | β < 0 | 23% | Competitive market pressure |
| Prudential rate (current) | 0 weeks | β > 0 | 10% | Quality signaling effect |

**Economic Validation:**
- ✓ All coefficients have theoretically expected signs
- ✓ 100% sign consistency across 10,000 bootstrap samples
- ✓ All coefficients statistically significant at α = 0.05

---

## Strategic Applications

### 1. Rate-Setting Decisions
- Biweekly rate review informed by model scenarios
- Uncertainty-aware recommendations (not just point estimates)
- Competitive positioning analysis guides strategy

### 2. Revenue Forecasting
- Weekly sales projections with confidence intervals
- Budget planning with risk-adjusted forecasts
- Scenario stress testing for strategic planning

### 3. Competitive Intelligence
- Real-time monitoring of 8 major carriers
- Market share weighted impact analysis
- Early warning for competitive rate changes

### 4. Risk Assessment
- $16.8M weekly uncertainty quantification
- Downside scenario analysis
- Volatility-weighted performance validation

---

## Success Stories

### Improved Forecast Accuracy
**Before:** Benchmark model with 57.54% R², 16.40% MAPE
**After:** Bootstrap Ridge with 78.37% R², 12.74% MAPE
**Impact:** 36% improvement in explanatory power, 22% reduction in forecast error

### Uncertainty Quantification
**Before:** Point estimates only, no risk assessment
**After:** 95% bootstrap confidence intervals with 94.4% coverage
**Impact:** Risk-informed strategic decisions, $16.8M weekly uncertainty bands

### Competitive Intelligence Automation
**Before:** Manual competitive rate tracking and analysis
**After:** Automated 8-carrier monitoring with market share weighting
**Impact:** Real-time competitive positioning insights, 2-week response window

---

## For More Information

### Business Stakeholders
- **This document** - Executive overview (1 page)
- [methodology_report.md](methodology_report.md) - Complete technical methodology (1,631 lines)
- [rai_governance.md](rai_governance.md) - RAI000038 compliance (902 lines)

### Data Scientists
- [../onboarding/GETTING_STARTED.md](../onboarding/GETTING_STARTED.md) - Complete onboarding (2 hours)
- [../onboarding/COMMON_TASKS.md](../onboarding/COMMON_TASKS.md) - Practical examples
- [../methodology/feature_engineering_guide.md](../methodology/feature_engineering_guide.md) - 598-feature pipeline

### Developers
- [../architecture/MULTI_PRODUCT_DESIGN.md](../architecture/MULTI_PRODUCT_DESIGN.md) - System architecture
- [../development/MODULE_HIERARCHY.md](../development/MODULE_HIERARCHY.md) - Code organization
- [../development/CODING_STANDARDS.md](../development/CODING_STANDARDS.md) - Style guide

### Validators
- [../practices/LEAKAGE_CHECKLIST.md](../practices/LEAKAGE_CHECKLIST.md) - **MANDATORY** pre-deployment
- [../methodology/validation_guidelines.md](../methodology/validation_guidelines.md) - Complete validation workflow
- [../operations/EMERGENCY_PROCEDURES.md](../operations/EMERGENCY_PROCEDURES.md) - Crisis response

---

## Contact

**Technical Owner:** Brandon Behring (brandon.behring@prudential.com)
**Business Owner:** Annuity Rate Setting Team (annuity-rate-setting@prudential.com)
**Model Risk:** Annuities Model Risk (annuities-model-risk@prudential.com)

**Repository:** [annuity-price-elasticity-v2](https://github.com/prudential/annuity-price-elasticity-v2)
**Documentation:** `/docs/` directory with 17 subdirectories, 64+ markdown files
**RAI Registration:** RAI000038 (Version 3.0)

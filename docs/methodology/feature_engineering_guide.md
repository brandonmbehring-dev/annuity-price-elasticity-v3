# Feature Engineering Guide: 598 Features

**Repository:** annuity-price-elasticity-v3
**Last Updated:** 2026-01-28

---

## Overview

The RILA Price Elasticity model transforms raw sales and competitive data into 598 engineered features through a 10-stage pipeline. This guide explains the complete feature engineering process, from data extraction to final feature preparation.

**Pipeline Output:** 160 weekly observations × 598 features
**Execution Time:** ~45 seconds
**Peak Memory:** 2.1GB

---

## Pipeline Architecture

### 10-Stage Feature Engineering Pipeline

```
Stage 1: Data Extraction
  ↓ Cross-account S3 access (TDE, WINK, FRED)
Stage 2: Product Filtering
  ↓ FlexGuard 6Y20B identification (WINK Product ID 2979)
Stage 3: Sales Data Cleanup
  ↓ Quality validation, application_signed_date filtering
Stage 4: Time Series Creation
  ↓ Daily frequency, complete time index
Stage 5: WINK Rate Processing
  ↓ Forward-fill continuous rates, standardization
Stage 6: Market Share Weighting
  ↓ Quarterly weights applied to competitor rates
Stage 7: Data Integration
  ↓ Daily merge (sales + rates + economic indicators)
Stage 8: Competitive Features
  ↓ Rankings, percentiles, spreads, top-N analysis
Stage 9: Weekly Aggregation
  ↓ Business day adjustment, holiday normalization
Stage 10: Lag Engineering
  ↓ 18-period lag structures with directional controls
  ↓
Final Output: 598 features ready for AIC selection
```

**Reference Implementation:**
- Pipeline orchestration: `src/features/engineer.py`
- Configuration: `src/config/product_config.py`
- Data extraction: `src/data/extraction.py`
- Feature creation: `src/features/` modules

---

## Feature Categories

### 1. Own Rate Features (~50 features)

**Structure:** Prudential cap rate with transformations and lags

**Base Features:**
- `prudential_rate`: Current FlexGuard 6Y20B cap rate
- `prudential_rate_change`: Week-over-week rate change
- `prudential_rate_trend`: 4-week moving average trend

**Lag Structure:**
- `prudential_rate_lag_0` through `prudential_rate_lag_17`
- **Minimum lag:** 0 weeks (we control our own rate during rate-setting)
- **Direction:** Both forward and backward lags allowed
- **Optimal lag:** Lag-0 selected in optimal model (10% importance)

**Economic Theory:**
- **Quality Signaling:** β > 0 (higher rates → more sales)
- **6Y20B Enhancement:** 6-year commitment + 20% buffer amplifies signaling
- **Financial Strength:** Higher rates indicate superior product value

**Why Lag-0 Allowed:**
During rate-setting, we have perfect knowledge of our current rate. This is not data leakage because we control this variable.

---

### 2. Competitor Rate Features (~400 features)

**Structure:** 8 carriers × 9 lags × multiple aggregations

#### Individual Carrier Rates (8 × 18 = 144 features)

**Carriers Tracked:**
1. Athene [2772, 3409]
2. Brighthouse [2319]
3. Equitable [2286, 3282, 3853]
4. Jackson [3351, 3714]
5. Lincoln [2924]
6. Symetra [3263, 3751]
7. Transamerica [3495]
8. Allianz (if applicable)

**Feature Names:**
- `athene_rate_lag_0` through `athene_rate_lag_17`
- Similar for all 8 carriers

**Lag Structure:**
- **Minimum lag:** 2 weeks (rate-setting to effective date delay)
- **Direction:** Backward only (cannot use future competitor rates)
- **Optimal lag:** 2-week lag selected (23% importance)

#### Aggregated Competitor Features (8 × 18 = 144 features)

**Core Aggregations:**
- `competitor_mid`: Market share weighted mean of all competitors
- `competitor_core`: Core competitor group average
- `competitor_median`: Median rate (robust to outliers)
- `competitor_top3`: Mean of top 3 competitors
- `competitor_top5`: Mean of top 5 competitors **(optimal feature, 23% importance)**

**Market Share Weighting:**
```python
Competitive_Rate_t = Σ(Rate_i,t × MarketShare_i) / Σ(MarketShare_i)
```

**Why Market Share Matters:**
- Distribution is key to RILA sales success
- Competitors with wider distribution have greater competitive impact
- Previous quarter sales proxy for current distribution strength
- Unique to RILA products (FIA/MYGA may use equal weighting)

**Feature Names:**
- `competitor_top5_lag_2` through `competitor_top5_lag_17`
- Similar for all aggregation types

#### Competitive Spread Features (~100 features)

**Spread Calculations:**
- `spread_prudential_vs_top5`: Prudential rate - top 5 competitor average
- `spread_prudential_vs_median`: Prudential rate - median competitor
- `spread_prudential_vs_core`: Prudential rate - core group average

**Percentile Rankings:**
- `prudential_percentile_rank`: Prudential's rank among 9 carriers (0-100)
- `prudential_competitive_position`: Binary indicator (above/below median)

**Economic Theory:**
- **Competitive Pressure:** β < 0 for competitor rates (higher competitor rates → lower our sales)
- **Relative Positioning:** Customers compare rates across carriers
- **Cross-Elasticity:** Multi-firm modeling required for competitive markets

---

### 3. Sales Momentum Features (~50 features)

**Structure:** Lagged historical sales with transformations

**Base Features:**
- `sales`: Issue date aggregated sales (standard metric)
- `sales_by_contract_date`: Contract date aggregated sales (best practice from FIA v2.0)

**Lag Structure:**
- `sales_target_lag_1` through `sales_target_lag_17`
- `sales_target_contract_lag_1` through `sales_target_contract_lag_17`
- **Direction:** Backward only (historical sales, not future)
- **Optimal lag:** 5-week lag selected (67% importance)

**Transformations:**
- `sales_log`: Log-transformed sales (reduces skewness)
- `sales_normalized`: Min-max normalized [0, 1]
- `sales_change`: Week-over-week change
- `sales_pct_change`: Percentage change

**Economic Theory:**
- **Contract Processing Persistence:** β > 0 (recent sales momentum continues)
- **Distribution Lags:** Applications in progress create sales pipeline
- **6Y20B Enhancement:** Longer commitment amplifies persistence effects

**Why Lag-5 Optimal:**
AIC-based selection identified 5-week lag as optimal balance of:
- Captures medium-term momentum
- Avoids excessive autocorrelation
- Aligns with contract processing timelines

---

### 4. Economic Indicator Features (~30 features)

**Structure:** Macroeconomic context with lags

#### Treasury Rates (18 features)

**Base Feature:**
- `econ_treasury_5y`: 5-Year Treasury Constant Maturity Rate (DGS5)

**Lag Structure:**
- `treasury_5y_lag_0` through `treasury_5y_lag_17`
- **Direction:** Backward only (economic context)

**Business Context:**
- Interest rate environment affects annuity demand baseline
- Higher rates → more attractive fixed income alternatives
- RILA competitiveness vs traditional annuities

#### Market Volatility (18 features)

**Base Feature:**
- `market_volatility`: CBOE Volatility Index (VIX)

**Lag Structure:**
- `vix_lag_0` through `vix_lag_17`
- **Direction:** Backward only (market regime identification)

**Business Context:**
- Volatility affects customer risk preferences
- High VIX → increased demand for buffer protection
- 20% buffer becomes more valuable in volatile markets

**Volatility-Weighted Validation:**
The model maintains 77.60% R² under volatility weighting (only 0.77% degradation from standard 78.37%), demonstrating stability across market regimes.

---

### 5. Temporal Features (~68 features)

**Holiday Masks:**
- `holiday_mask_early`: Day of year < 14 (New Year's processing delays)
- `holiday_mask_late`: Day of year > 360 (Christmas slowdown)
- `holiday_mask_week`: Binary indicator for holiday weeks

**Business Day Adjustments:**
- Sales normalized by: `sales * 5 / business_days_in_week`
- Accounts for shortened trading weeks
- Memorial Day, Labor Day, Thanksgiving, Christmas, New Year's

**Seasonality Features:**
- `month`: Month indicator (1-12)
- `quarter`: Quarter indicator (Q1-Q4)
- `week_of_year`: Week indicator (1-52)
- `is_month_end`: Binary indicator for month-end weeks

**Why Exclude Day 1-13 and Day 361-365:**
Launch anomalies near FlexGuard introduction and year-end processing backlogs create noise that distorts model training.

---

## Feature Engineering Summary Table

| Category | Base Features | Lag Periods | Transformations | Total Features |
|----------|---------------|-------------|-----------------|----------------|
| **Own Rates** | 3 | 18 | 5 | ~50 |
| **Individual Carriers** | 8 | 18 | 1 | 144 |
| **Aggregated Competitors** | 8 | 18 | 1 | 144 |
| **Competitive Spreads** | ~50 | varies | 1 | ~100 |
| **Sales Momentum** | 2 | 18 | 5 | ~50 |
| **Economic Indicators** | 2 | 18 | 3 | ~36 |
| **Temporal** | ~15 | varies | multiple | ~68 |
| **Derived/Interaction** | ~3 | varies | multiple | ~6 |
| **TOTAL** | - | - | - | **598** |

---

## Implementation Details

### Market Share Weighting Implementation

```python
# Quarterly market share data
weights = df_ts_w[competitors].fillna(0).values * \
          df_ts_w[competitors_weight].fillna(0).values

# Weighted competitive rate
df_ts_w['C_weighted_mean'] = weights.sum(axis=1)

# Normalize weights
df_ts_w['weight_sum'] = df_ts_w[competitors_weight].sum(axis=1)
df_ts_w['C_weighted_mean'] = df_ts_w['C_weighted_mean'] / df_ts_w['weight_sum']
```

**Reference:** `src/features/aggregation/market_share_weighting.py`

### Lag Structure Optimization

**Directional Lag Control:**

```python
# Own rate: No minimum lag (we control it)
own_rate_lags = range(0, 18)  # 0-17

# Competitor rates: Minimum 2-week lag (rate-setting delay)
competitor_lags = range(2, 18)  # 2-17

# Sales: Backward only (historical momentum)
sales_lags = range(1, 18)  # 1-17

# Economic: Backward only (context)
econ_lags = range(0, 18)  # 0-17
```

**Reference:** `src/config/product_config.py` - Lag column configurations

### Weekly Aggregation

**Aggregation Dictionary:**
```python
{
    # Competitive features (mean rate for week)
    'C_weighted_mean': 'mean',
    'Prudential': 'mean',
    'competitor_top5': 'mean',

    # Sales data (sum for week)
    'sales': 'sum',
    'sales_by_contract_date': 'sum',

    # Economic indicators (last value for week)
    'DGS5': 'last',
    'VIXCLS': 'last',

    # CPI (monthly, forward-filled)
    'CPI': 'last'
}
```

**Reference:** `src/config/product_config.py` - Weekly aggregation configuration

### Smoothing

**2-Week Rolling Average:**
```python
# Sales smoothing
df_ts_w['sales_smooth'] = df_ts_w['sales'].rolling(window=2, min_periods=1).mean()

# Minimum weight time: 1 week
# Prevents causal information leakage from future to past
```

**Launch Anomaly Handling:**
Two anomalous weeks near FlexGuard launch (backlog processing) addressed via smoothing to prevent model distortion.

**Reference:** `src/features/preprocessing/smoothing.py`

### Forward Filling

**WINK Rate Interpolation:**
```python
# WINK only shows entries when rates change
# Create complete time index
date_range = pd.date_range(start=min_date, end=max_date, freq='D')

# Forward-fill rates to have effective rate defined for all dates
df_rates = df_rates.reindex(date_range, method='ffill')
```

**Why Forward-Fill:**
Competitor rates remain effective until explicitly changed. Forward-filling creates continuous rate time series for analysis.

**Reference:** `src/data/preprocessing.py` - WINK data processing

---

## Critical Design Decisions

### Why 598 Features?

**Engineering Philosophy:**
Generate comprehensive candidate feature set, then use AIC-based selection to identify optimal subset.

**Rationale:**
1. **Comprehensive Coverage:** All economically plausible relationships represented
2. **Lag Exploration:** Multiple lag periods test temporal hypotheses
3. **Aggregation Variants:** Different competitive metrics (top-3, top-5, median, mean)
4. **Transformations:** Linear, log, normalized, change metrics
5. **AIC Selection:** Reduces 598 → ~15 features via optimal subset search
6. **Economic Filtering:** Further reduces ~15 → 3-4 features via constraint validation

**Why Not More Features:**
- Diminishing returns beyond 18-period lags
- Computational cost increases exponentially
- Risk of spurious correlations rises

**Why Not Fewer Features:**
- May miss optimal lag structure
- Limited coverage of competitive dynamics
- Insufficient transformation variants

### Why AIC Selection from 598?

**Akaike Information Criterion (AIC):**
```
AIC = 2k - 2ln(L)
```
- k: Number of parameters
- L: Maximum likelihood
- **Lower AIC = Better** (balances fit and parsimony)

**Selection Process:**
1. Generate candidate feature combinations (793 combinations evaluated)
2. Fit Ridge regression for each combination
3. Calculate AIC for each model
4. Apply economic constraint filtering (193/793 pass = 24.3%)
5. Select lowest AIC among economically valid models

**Current Optimal Model: 3 Features**
| Feature | AIC Contribution | Economic Constraint |
|---------|------------------|---------------------|
| sales_target_contract_t5 | Largest reduction | β > 0 [PASS] |
| competitor_top5_t2 | Medium reduction | β < 0 [PASS] |
| prudential_rate_current | Small reduction | β > 0 [PASS] |

**Final AIC:** 5358.129

**Reference:** `src/features/selection/aic_engine.py`

---

## Feature Importance (Typical Production Model)

### Optimal Model Coefficients

| Rank | Feature | Lag | Importance | Coefficient Sign | Economic Theory |
|------|---------|-----|------------|------------------|-----------------|
| 1 | `sales_target_contract` | 5 weeks | 67% | β > 0 | Contract processing persistence |
| 2 | `competitor_top5` | 2 weeks | 23% | β < 0 | Competitive market pressure |
| 3 | `prudential_rate_current` | 0 weeks | 10% | β > 0 | Quality signaling effect |

### Interpretation

**Sales Momentum (67% importance):**
- 5-week lagged sales drives future sales
- Contract processing and distribution channel effects
- 6Y20B enhancement: Longer commitment amplifies persistence

**Competitive Response (23% importance):**
- Top 5 competitors' mean rate affects competitive position
- 2-week response window for competitive dynamics
- Market share weighted for distribution impact

**Quality Signaling (10% importance):**
- Current Prudential rate signals product value
- Higher rates indicate financial strength
- 6Y20B context: Enhanced signaling via 20% buffer + 6-year term

---

## Validation Checklist

### Pre-Deployment Feature Validation

- [ ] **No lag-0 competitor features** (prevents data leakage)
- [ ] **Expected coefficient signs** (quality signaling β>0, competitive pressure β<0, persistence β>0)
- [ ] **No future leakage** (all features use only historical data available at decision time)
- [ ] **Market share weights sum to 1.0** (normalization check)
- [ ] **Holiday mask covers 2 weeks** (day 1-13 and day 361-365 excluded)
- [ ] **Forward-filled rates continuous** (no gaps in WINK rate time series)
- [ ] **Business day adjustment applied** (sales normalized by trading days)
- [ ] **50-day mature data cutoff** (incomplete recent data excluded)

### Feature Quality Checks

- [ ] **No missing values in final feature matrix** (handle via forward-fill or median imputation)
- [ ] **No infinite values** (check log transformations, division by zero)
- [ ] **Reasonable ranges** (cap rates 0-50%, sales 0-$50M, VIX 10-80)
- [ ] **Temporal ordering preserved** (no look-ahead bias)
- [ ] **Lag structure validated** (minimum lags enforced by feature type)

**Reference:** [../practices/LEAKAGE_CHECKLIST.md](../practices/LEAKAGE_CHECKLIST.md) (MANDATORY)

---

## Common Pitfalls & Solutions

### Pitfall 1: Using Lag-0 Competitor Rates

**Problem:** Competitor rate at time t not available when making prediction at time t
**Solution:** Minimum 2-week lag for competitor features (rate-setting to effective date delay)
**Detection:** Economic constraint validation catches β>0 for competitor features (should be β<0)

### Pitfall 2: Forward-Looking Sales Lags

**Problem:** Using future sales to predict past sales (causal impossibility)
**Solution:** Sales lags strictly backward (lag-1 through lag-17 only)
**Detection:** Negative lag indicators in feature names

### Pitfall 3: Incomplete WINK Forward-Fill

**Problem:** Gaps in competitor rate time series create missing values
**Solution:** Forward-fill from last known rate until next rate change
**Detection:** `df.isna().sum()` on competitor features

### Pitfall 4: Holiday Week Contamination

**Problem:** Holiday processing delays create artificial sales drops
**Solution:** Business day adjustment (multiply by 5/business_days)
**Detection:** Anomalous sales drops during known holiday weeks

### Pitfall 5: Launch Anomaly Contamination

**Problem:** Product launch backlog creates unrealistic sales spikes
**Solution:** Exclude day 1-13 and smooth 2-week rolling average
**Detection:** Visual inspection of sales time series near launch

---

## Performance Characteristics

### Computational Efficiency

| Metric | Value | Notes |
|--------|-------|-------|
| **Execution Time** | ~45 seconds | Full 10-stage pipeline |
| **Peak Memory** | 2.1GB | 160 weeks × 598 features |
| **Lag Generation** | ~10 seconds | 18-period lag structures |
| **Market Share Weighting** | ~5 seconds | Quarterly weight application |
| **Weekly Aggregation** | ~8 seconds | Daily → weekly frequency |

### Scalability Considerations

**Current Capacity:**
- 160 weekly observations (3+ years historical)
- 598 features per observation
- 10,000 bootstrap samples for inference, 1,000 for forecasting

**Future Scaling:**
- Can handle 500+ weeks without performance degradation
- Bootstrap ensemble scales linearly with sample count
- Feature count can expand to 1,000+ with minor performance impact

---

## References

### Implementation
- **Pipeline orchestration:** `src/features/engineer.py`
- **Market share weighting:** `src/features/aggregation/market_share_weighting.py`
- **Lag structure:** `src/features/temporal/lag_engineering.py`
- **Smoothing:** `src/features/preprocessing/smoothing.py`
- **Holiday adjustment:** `src/features/preprocessing/holiday_normalization.py`

### Configuration
- **Product config:** `src/config/product_config.py`
- **Lag column configs:** `src/config/product_config.py` lines 53-92
- **Weekly aggregation:** `src/config/product_config.py` lines 95-143

### Related Documentation
- **Methodology:** [../business/methodology_report.md](../business/methodology_report.md)
- **Feature rationale:** [../analysis/FEATURE_RATIONALE.md](../analysis/FEATURE_RATIONALE.md)
- **Leakage prevention:** [../practices/LEAKAGE_CHECKLIST.md](../practices/LEAKAGE_CHECKLIST.md)
- **AIC selection:** `src/features/selection/aic_engine.py`

---

**Last Updated:** 2026-01-28
**Version:** 2.0 (v2 refactored architecture)

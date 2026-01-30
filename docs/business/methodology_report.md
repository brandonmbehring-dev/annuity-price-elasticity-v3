# RILA (FlexGuard) 6Y20B Methodology Report
## Price Elasticity Analysis - Bootstrap Ridge Regression Framework

**Product:** FlexGuard 6-Year Term, 20% Buffer (6Y20B)
**Applied Business Unit:** Annuities
**Market:** Retirement Strategies
**Report Generated:** 2026-01-28
**Analysis Framework:** Bootstrap Ridge Ensemble with Economic Constraint Validation
**Repository:** `annuity-price-elasticity-v3`

---

## Executive Summary

This comprehensive methodology document describes the price elasticity analysis framework for RILA FlexGuard 6Y20B product using Bootstrap Ridge regression ensemble modeling. The system provides automated competitive intelligence and strategic pricing guidance through enterprise-grade machine learning with comprehensive uncertainty quantification.

**Key Achievements:**
- **Model Performance**: 78.37% R² and 12.74% MAPE (22.3% improvement over benchmark)
- **Production Ready**: Bootstrap ensemble (10,000 estimators for inference, 1,000 for forecasting) with 95% confidence intervals
- **Competitive Intelligence**: Automated analysis across 8 major carriers with market share weighting
- **Strategic Impact**: 2-week competitive response window with $16.8M weekly uncertainty quantification
- **Infrastructure**: Cross-account AWS integration with complete DVC pipeline automation

---

## Table of Contents

1. [Objective](#objective)
2. [Background](#background)
   - What is Price Elasticity?
   - What are Annuities?
   - RILA 6Y20B Product Context
3. [Methodology](#methodology)
   - Model Framework Evolution
   - Bootstrap Ridge Regression Theory
   - Economic Constraint Validation
   - Inference vs Prediction Models
4. [Data Sources](#data-sources)
   - FAST Sales Data (TDE System)
   - WINK Competitive Rates
   - Economic Indicators
5. [Feature Engineering](#feature-engineering)
   - 598 Feature Pipeline
   - Market Share Weighting
   - Lag Structure Optimization
6. [Model Framework](#model-framework)
   - Bootstrap Ensemble Architecture
   - AIC-Based Feature Selection
   - Cross-Validation Methodology
7. [Model Performance & Validation](#model-performance--validation)
   - Performance Metrics
   - Economic Constraint Validation
   - Temporal Stability Analysis
8. [Results & Strategic Applications](#results--strategic-applications)
   - Price Elasticity Estimates
   - Strategic Pricing Recommendations
   - Competitive Analysis
9. [Conclusions & Recommendations](#conclusions--recommendations)
10. [Appendix: Alternative Approaches](#appendix-alternative-approaches)
    - Local Linear Approximations (RILA v1.2)
    - Global Logit Transformations (FIA v2.0)

---

## 1. Objective

### Applied Business Unit
**Annuities** | **Market:** Retirement Strategies

### Business Objectives

**Increased Efficiency:**  
With data pre-analysis and machine learning, the business can automate competitive analysis, effectively streamlining the rate-setting process by increasing efficiency and reducing the time and effort required to translate data and assumptions into pricing strategy.

**Improved Accuracy and Reliability:**  
Machine learning algorithms analyze large amounts of data and provide insights into pricing trends and customer behavior that may be missing in human experience. This helps improve the accuracy and reliability of rate-setting decisions, resulting in better outcomes for both growth and profitability.

**Better Decision Making:**  
Data-driven decisions in the rate-setting process become more accurate by using machine learning models to predict price elasticity and dynamic pricing compared to competitor rates.

**Continuous Improvement:**  
Machine learning models are trained on historical data to continually improve the accuracy of price optimization algorithms. This helps the business stay up-to-date with changing market conditions and customer preferences, ensuring that pricing strategies remain competitive over time.

### Strategic Context for RILA 6Y20B

The RILA 6Y20B product represents a strategic evolution from shorter-term RILA products:

**Product Architecture:**
- **6-Year Term**: Extended commitment period reducing rollover risk
- **20% Buffer**: Enhanced downside protection for customer security
- **S&P 500 Strategy**: Core index-linking strategy for competitive benchmarking
- **Multi-Year Positioning**: Strategic wealth-building vs. short-term tactical products

**Business Context:**

The Annuities Product Management team leverages past sales experience, feedback from distribution partners, and industry benchmarks during the rate-setting process. While these methods provide critical insight, they are not inclusive of all driving factors related to product price elasticity. The inclusion of data-driven assumptions provides greater accuracy for forecasting appropriate rate actions with the goal of maximizing value to the enterprise. This results in:

- Improved competitive positioning
- Enhanced forecasting capabilities
- Historically reliable data on sales impact due to pricing strategies
- Strategic decision support with uncertainty quantification

**Reference:** `src/config/config_builder.py:19` - Configuration system for FlexGuard 6Y20B

---

## 2. Background

### What is Price Elasticity?

Price elasticity measures the relationship between 'price' and sales volume for annuity products. In the context of RILA products, this relationship has unique characteristics that distinguish it from conventional pricing theory.

**Key Distinctions:**

We distinguish two different cases:
1. **Local Approximations**: Linear models when exploring prices near equilibrium values (small rate changes around current market rates)
2. **Global Approximations**: Sigmoid/Logit models when exploring prices from market entry or major product repositioning

**RILA-Specific Price Definition:**

For Annuities, the 'Price' in price elasticity refers to the benefit offered by the annuity. For RILA/FlexGuard 6Y20B, this is:
- **Primary Metric**: Cap Rate offered for 20% Buffer, 6-year term, S&P 500 strategy
- **Direction Reversal**: High cap rate means a more generous benefit (opposite of conventional pricing where low price = competitive offering)
- **Quality Signaling**: Higher rates signal superior product value and financial strength

**RILA 6Y20B Context:**

For FlexGuard 6Y20B, we use Local Approximations that are valid for small price changes around equilibrium. The linear framework is appropriate because:
- Product is well-established in the market
- Rate changes are strategic adjustments rather than major repositioning
- Linear models provide interpretability essential for business decision-making

**Theoretical Foundation:**
- Phillips, Robert. *Pricing and Revenue Optimization*, Stanford University Press, 2005. https://doi.org/10.1515/9780804781640

### What are Annuities?

**Registered Index-Linked Annuities (RILAs)** are tax-deferred retirement savings vehicles that offer:
- **Upside Participation**: Performance linked to market indices (S&P 500, Russell 2000)
- **Downside Protection**: Buffer protection against market losses (e.g., 20% buffer)
- **Tax Advantages**: Tax-deferred growth until withdrawal
- **Flexibility**: Multiple crediting strategies and term options

**FlexGuard 6Y20B Product Structure:**
- **Product Identification**: WINK Product ID 2979 (used for competitive rate tracking)
- **6-Year Maturity**: Multi-year strategic positioning for long-term wealth building
- **20% Downside Buffer**: Enhanced protection absorbs first 20% of index losses
- **Annual Crediting**: Point-to-point (PTP) performance measurement
- **Target Market**: Retirement investors seeking growth potential with risk management

### RILA 6Y20B Product Evolution

**From RILA v1.2 (1Y10B) to RILA 6Y20B:**

| Dimension | RILA v1.2 (1Y10B) | RILA 6Y20B |
|-----------|-------------------|------------|
| **Term Length** | 1-year (annual reset) | 6-year (extended commitment) |
| **Buffer Protection** | 10% downside buffer | 20% downside buffer |
| **Market Positioning** | Short-term tactical | Long-term strategic wealth building |
| **Competitive Dynamics** | Annual competitive pressure | Multi-year strategic positioning |
| **Customer Profile** | Tactical allocators | Strategic wealth builders |
| **Rollover Risk** | High (annual) | Low (6-year commitment) |

**Strategic Implications:**
- **Quality Signaling Amplification**: 6-year commitment amplifies quality signaling effects (β > 0 stronger)
- **Competitive Response Dynamics**: Multi-year positioning reduces short-term competitive volatility
- **Persistence Effects Enhanced**: Contract processing momentum more pronounced for long-term products
- **Buffer Risk Premium**: 20% vs 10% buffer creates differentiated risk premium competitive dynamics

**Reference:** README.md lines 61-132 - Evolution from RILA v1.2 to 6Y20B Framework

---

## 3. Methodology

### Visual Comparison: Local vs Global Approximations

The choice between local linear models and global logit models depends on the pricing context:

**Figure: Theoretical Price-Response Function Approaches**

![Theoretical Price-Response Functions](../notebooks/extracted_media/media/image1.png)

- **(A) Linear Price-Response**: Appropriate for strategic rate adjustments (±50-100 bps) around equilibrium. Used for RILA 6Y20B established product pricing.
- **(B) Constant Elasticity**: Power-law relationship assuming constant percentage response. Less common for RILA products.
- **(C) Logit Price-Response**: S-curve appropriate for market entry, product launches, or major repositioning across wide rate ranges.

**RILA 6Y20B Context**: We use the local linear approximation (A) because the product is well-established and rate changes are strategic adjustments rather than major repositioning.

**Reference:** Appendix Section A - Local Linear Approximations for detailed methodology.

---

### Model Framework Evolution

#### Original RILA v1.2 Methodology (Baseline)

**Model Type:** Bagged linear model with AIC feature selection
**Training Approach:** Time-holdout cross-validation with 2-week refresh cycles
**Feature Selection:** Best subset selection with manual lag structure optimization
**Uncertainty:** Standard statistical inference without ensemble uncertainty quantification

**Strengths:**
- Simple, interpretable linear framework
- Fast execution for weekly business cycles
- AIC-based feature selection prevents overfitting

**Limitations:**
- Single model estimates without robust uncertainty quantification
- Manual lag structure optimization required
- Limited handling of model variability

#### Enhanced RILA 6Y20B Bootstrap Framework (Current Implementation)

**Model Type:** Bootstrap Ridge regression ensemble (10,000 estimators for inference, 1,000 for forecasting) with L2 regularization (α=1.0)
**Training Approach:** Expanding window cross-validation with economic constraint validation
**Feature Selection:** AIC-based selection with automated economic constraint enforcement
**Uncertainty:** Empirical bootstrap confidence intervals for strategic risk management

**Key Enhancements:**
- **Bootstrap Ensemble**: 100 Ridge estimators provide robust uncertainty quantification
- **Economic Constraints**: Automated validation prevents spurious correlations
- **L2 Regularization**: Optimal bias-variance trade-off for stable predictions
- **Confidence Intervals**: 95% empirical bands for strategic decision support

**Performance Achievement:**
- **R² Improvement**: 78.37% vs 57.54% benchmark (36.2% improvement)
- **MAPE Improvement**: 12.74% vs 16.40% benchmark (22.3% improvement)
- **Bootstrap Coverage**: 94.4% of validation periods within 95% confidence bands

**Reference:** `src/models/bootstrap_ridge.py` - Bootstrap Ridge ensemble implementation

### Model Themes and Design Philosophy

#### Inference vs. Prediction: A Critical Distinction

This model is designed for **inference**, not just prediction. This distinction is fundamental to understanding our methodological choices:

**Prediction Models:**
- **Goal**: Maximize forecasting accuracy
- **Approach**: Black-box methods acceptable (XGBoost, neural networks)
- **Interpretation**: Secondary concern
- **Evaluation**: MAPE, R², forecast accuracy

**Inference Models (Our Approach):**
- **Goal**: Understand causal relationships between price and sales
- **Approach**: Interpretable parametric models required
- **Interpretation**: Primary concern for business decision-making
- **Evaluation**: Economic validity, coefficient signs, + forecast accuracy

**Why Linear Framework for Strategic Pricing:**

From James, G., Witten, D., Hastie, T. and Tibshirani, R. (2021) *An Introduction to Statistical Learning*:

> "Depending on whether our ultimate goal is prediction, inference, or a combination of the two, different methods for estimating f may be appropriate. Linear models allow for relatively simple and interpretable inference, but may not yield as accurate predictions as some other approaches. In contrast, some of the highly non-linear approaches can potentially provide quite accurate predictions for Y, but this comes at the expense of a less interpretable model for which inference is more challenging."

**Key Questions for Inference:**
1. **Which predictors are associated with the response?** Identifying the few important predictors among 598 engineered features
2. **What is the relationship between response and each predictor?** Understanding positive vs negative effects, magnitude of impacts
3. **Can the relationship be adequately summarized using a linear equation?** For strategic pricing near equilibrium, yes

**Why Not XGBoost or Neural Networks?**

While non-linear models can achieve higher predictive accuracy, they have critical downsides for our use case:
- **Less Interpretable**: Cannot explain to business stakeholders why a rate change affects sales
- **Hyper-Parameter Dependent**: Results highly sensitive to configuration choices
- **High Variance**: Inconsistent behavior under small perturbations of cap rates
- **No Clear Benefit**: Similar or worse performance on our inference-focused task

**Bootstrap Enhancement Benefits:**
- Adds robust uncertainty quantification without sacrificing interpretability
- Ensemble stability reduces single-model risk
- Economic constraints validated across bootstrap samples
- Confidence intervals enable risk-informed strategic decisions

**Reference:** README.md lines 508-524 - Inference vs Prediction distinction

### Exploratory Data Analysis: The Necessity of Cross-Elasticity Modeling

A critical exploratory analysis revealed an insight that fundamentally shaped our modeling approach: examining Prudential's cap rates in isolation demonstrates the WRONG trend. A naïve single-variable model would incorrectly suggest that increasing cap rates leads to decreased sales, directly contradicting business intuition and microeconomic theory. This counterintuitive result occurs because when Prudential raises rates, competitors often raise rates simultaneously due to market-wide factors (interest rate environment, volatility shifts). What matters is not absolute rate levels but relative competitive advantage - our rates compared to alternatives available to customers. This finding validates the absolute necessity of our multi-firm competitive modeling framework with cross-elasticity terms. Any attempt to create a model without incorporating competitor rates would fail to capture the relevant market behavior and produce misleading elasticity estimates for strategic pricing decisions.

### Bootstrap Ridge Regression Theory

**Objective Function:**

For each bootstrap sample, we minimize:

```
L(β) = ||y - Xβ||² + α||β||²
```

Where:
- `y`: Target variable (sales volume)
- `X`: Feature matrix (competitive rates, lags, economic indicators)
- `β`: Coefficient vector to estimate
- `α = 1.0`: L2 regularization parameter (optimal bias-variance trade-off)

**Bootstrap Ensemble:**
1. **Sample Size**: Optimized by use case - 10,000 estimators for inference analysis, 1,000 for time series forecasting
2. **Estimator Training**: Independent Ridge regression per bootstrap sample with replacement
3. **Prediction Aggregation**: Mean prediction across bootstrap ensemble
4. **Uncertainty Quantification**: Empirical percentiles (2.5%, 97.5%) from bootstrap distribution

**Advantages of Bootstrap Ridge:**
- **Stability**: L2 regularization prevents coefficient explosion
- **Uncertainty**: Empirical confidence intervals capture model uncertainty
- **Consistency**: Economic constraints validated across all bootstrap samples
- **Interpretability**: Linear coefficients remain interpretable

**Reference:** `src/models/bootstrap_ridge.py` - Implementation details

### Economic Constraint Framework

The model enforces microeconomic theory through automated constraint validation. All selected features must satisfy theoretical expectations:

**1. Quality Signaling Theory:**
- **Principle**: Higher own rates signal superior product value and financial strength
- **Mathematical Constraint**: β > 0 for Prudential rate effects
- **6Y20B Context**: Enhanced signaling effect due to 6-year commitment and 20% buffer
- **Validation**: Coefficient sign checked across all bootstrap estimators (10,000 for inference)

**2. Competitive Pressure Theory:**
- **Principle**: Competitor rate increases reduce relative competitive advantage
- **Mathematical Constraint**: β < 0 for competitive rate effects
- **Cross-Elasticity**: Multi-firm modeling required for competitive market dynamics
- **Validation**: Negative coefficients enforced for competitor features

**3. Contract Processing Persistence Theory:**
- **Principle**: Recent sales momentum continues due to processing and distribution lags
- **Mathematical Constraint**: β > 0 for lagged sales effects
- **6Y20B Enhancement**: Longer commitment amplifies persistence effects
- **Optimal Lag**: 5-week momentum effect identified through AIC optimization

**4. Macroeconomic Context:**
- **Principle**: Economic indicators affect overall annuity demand
- **Features**: DGS5 (5-year Treasury), VIX (volatility), CPI (inflation)
- **Validation**: Coefficients must have economically plausible signs and magnitudes

**Constraint Enforcement:**
- **Feature Selection**: 793 combinations → 193 economically valid (24.3% pass rate)
- **Bootstrap Validation**: 100% coefficient sign consistency across bootstrap samples
- **Performance**: Economic constraints improve out-of-sample stability

**Reference:** `src/features/selection/constraints_engine.py` - Constraint validation logic

### Model Selection Philosophy

**Try to predict as much of price variation with the least amount of complexity.**

**Key Principles:**
1. **Small Data Set**: Use (generalized) linear models based on microeconomic principles
2. **Prevent Overfitting**: AIC-based feature selection to identify key predictors
3. **Genuine Forecasts**: Evaluate accuracy on new data not used in training
4. **Economic Validity**: All features must satisfy theoretical constraints

**Feature Selection Process:**
- **Starting Point**: 598 engineered features with 18-period lag structures
- **AIC Optimization**: Identify combinations minimizing Akaike Information Criterion
- **Economic Filtering**: Enforce microeconomic constraints on coefficients
- **Optimal Model**: 3-4 features providing best balance of fit and parsimony

**Competitive Feature Focus:**

A dimensional reduction technique considers how Prudential's 20% buffer 6-year term S&P rates differ from the mean S&P rate for all 20% buffer 6-year term S&P rates. This is the rate used as a benchmark by Financial advisors.

**Why Not Expand Beyond Baseline Strategy?**
- Multiple strategies (different buffers, terms) are not orthogonal features
- Cannot be tuned independently due to collinearities
- Could use PCA or PLSR for expansion
- Currently not in scope - no clear benefit for strategic pricing decisions

**Reference:** `src/features/selection/aic_engine.py` - AIC calculation and ranking

---

## 4. Data Sources

### Enterprise Data Architecture

#### FAST Sales Data (TDE System)

**AWS Infrastructure:**
- **Role ARN**: `arn:aws:iam::159058241883:role/isg-usbie-annuity-CA-s3-sharing`
- **Bucket**: `pruvpcaws031-east-isg-ie-lake`
- **Source Path**: `access/ierpt/tde_sales_by_product_by_fund/`
- **Data Volume**: 1.4M records, 10+ parquet files updated daily

**Key Features:**
- `issdate`: Issue date for sales aggregation
- `premiumamount`: Premium amount for revenue analysis
- `productdescription`: Product identifier (`'FlexGuardSM'` for 6Y20B)
- Contract-level details for volume analysis

**Filters Applied:**
- `productdescription == 'FlexGuardSM'`: Select only FlexGuard products
- Exclude FlexGuard Income products
- Focus on 6-year term, 20% buffer variants

**Critical Data Source Improvement from FIA v2.0:**

**Best Practice Adopted:** Use `application_signed_date` instead of `contract_issue_date` to prevent time-ordering leakage. This ensures that only information available at decision time is used in the model, avoiding look-ahead bias.

**Modeling Data:** Aggregate sales data for weekly values
**Business Context:** Contract-level sales data for FlexGuard 6Y20B volume analysis

**Reference:** `src/data/extraction.py` - Cross-account S3 data loading

#### WINK Competitive Intelligence

**AWS Infrastructure:**
- **Bucket**: `pruvpcaws031-east-isg-ie-lake`
- **Source Path**: `access/ierpt/wink_ann_product_rates/`
- **Data Volume**: 1M+ competitive rate observations, daily rate updates
- **Market Coverage**: 8 major RILA carriers with product ID mapping

**Key Features:**
- `effectiveDate`: Rate effective date for competitive timing
- `capRate`: Competitive cap rates for comparative analysis
- `bufferRate`: Buffer protection levels
- `indexUsed`: Index type (S&P 500, Russell 2000)
- Competitor identifiers for carrier-specific analysis

**Competitive Filters Applied:**
- `indexUsed == 'S&P 500'`: Only S&P products for consistent comparison
- `annualFeeForIndexingMethod.isna()`: Only no-fee products (cap rates comparable)
- `capRate.notna()`: Has a defined cap rate
- `bufferRate == 0.20`: Only 20% buffer products (6Y20B comparable products)
- `indexCreditingFrequency == 'Annual'`: 1-year crediting period

**Competitor Product ID Mapping:**
```python
{
    'Prudential': [2979],           # FlexGuard 6Y20B products
    'Athene': [2772, 3409],         # Multiple product lines
    'Brighthouse': [2319],           # Single flagship RILA
    'Equitable': [2286, 3282, 3853], # Structured Capital Strategies
    'Jackson': [3351, 3714],         # Elite Access products
    'Lincoln': [2924],               # Lincoln OptiBlend
    'Symetra': [3263, 3751],         # Symetra Edge products
    'Transamerica': [3495]           # TransElite RILA
}
```

**Rate Interpolation:**
WINK only shows entries when a cap rate is updated. We merge the cap rate changes with a complete time-index, then forward-fill the data to have the effective rate defined for all dates where products were active.

**Reference:** `src/data/preprocessing.py` - WINK data processing

#### Data Source Clarifications for RILA 6Y20B

**Illustration Data Status:**
Illustration data is NOT used in the RILA 6Y20B model, following the FIA v2.0 best practice. By using `application_signed_date` instead of `contract_issue_date`, the need for illustration data as a predictive proxy is eliminated. The application date provides more accurate temporal alignment with customer decision-making, preventing time-ordering leakage and improving model validity.

**Competitive Coverage - Nationwide Exclusion:**
Note that Nationwide [3757] was tracked in RILA v1.2 (1Y10B) but is not included in 6Y20B competitive analysis. At the time of this analysis, Nationwide does not offer a comparable 6-year term, 20% buffer RILA product. The 8 carriers listed above represent the complete set of competitors with directly comparable 6Y20B product offerings.

**Training Data Temporal Scope:**
The model training data begins from **2021-01-01**. Data prior to this date is excluded because the analysis required waiting until FlexGuard had launched and achieved stable distribution across enough firms to establish meaningful competitive dynamics. The early launch phase involved non-stationary market expansion that would not be predictive of steady-state price elasticity relationships. This starting date ensures the model captures mature product competitive behavior rather than launch artifacts.

#### Data Maturation and Processing Windows

**50-Day Mature Data Cutoff:**

RILA 6Y20B implements a 50-day mature data cutoff to exclude recent data that may be incomplete in the TDE system. This design choice differs from FIA v2.0's 110-day processing window approach.

**Why RILA Doesn't Need FIA's 110-Day Window:**

FIA v2.0 determined that the 99th percentile time between `application_signed_date` and `contract_issue_date` was approximately 110 days. Their model used `contract_issue_date` as the primary temporal marker, requiring a long maturation window to account for applications in progress that hadn't yet been issued.

By using `application_signed_date` directly (best practice adopted from FIA v2.0), RILA 6Y20B eliminates this architectural consideration entirely. Applications are recorded immediately when contracts are initiated, not when they complete processing.

**RILA's 50-Day Cutoff Purpose:**
- Accounts for data settlement delays in TDE reporting systems
- Excludes preliminary/incomplete data artifacts
- Sufficient buffer given immediate application date availability
- No need to wait for contract issuance completion

**Implementation:** The 50-day cutoff is applied universally before model training as a data quality filter.

**Reference:** `src/config/config_builder.py:395` - `mature_data_offset_days` parameter

#### Economic Indicators

**Data Sources:**
- **DGS5** (5-Year Treasury Constant Maturity Rate): Federal Reserve Economic Data (FRED)
- **VIXCLS** (CBOE Volatility Index): Market volatility indicator
- **CPI** (Consumer Price Index): Inflation adjustment for real sales volume

**Integration:**
- **Frequency**: Daily data aggregated to weekly
- **Lag Structure**: Economic indicators use backward-looking lags only
- **Business Context**: Macroeconomic regime identification for strategic timing

**CPI Adjustment:**
Weekly sales are adjusted for inflation to analyze real sales volume trends independent of nominal price changes.

**Reference:** `src/data/economic_indicators.py` - Economic data integration

### Data Security & Governance

**Security Framework:**
- **Cross-Account Role**: STS assume-role with temporary credential refresh (2-hour windows)
- **MFA Integration**: Enterprise multi-factor authentication for sensitive data access
- **Audit Logging**: Complete CloudTrail integration for data access and processing audit trails
- **Data Classification**: Confidential business data with restricted access controls

**Data Governance:**
- **Data Lineage**: Complete tracking from source TDE/WINK systems through model outputs
- **Version Control**: DVC integration with S3-backed storage for data and model versioning
- **Business Continuity**: Cross-region backup and disaster recovery for critical model outputs
- **Regulatory Compliance**: SOX-compliant audit trails for financial model data and decisions

**Reference:** README.md lines 313-351 - AWS Infrastructure & Data Governance

---

## 5. Feature Engineering

### 598 Feature Engineering Pipeline

The feature engineering pipeline transforms raw sales and competitive data into a comprehensive feature matrix optimized for price elasticity modeling.

**10-Stage Pipeline Architecture:**

1. **Data Extraction**: Cross-account S3 access to TDE sales and WINK rates
2. **Product Filtering**: FlexGuard 6Y20B product identification
3. **Competitive Processing**: Market share weighted competitive metrics
4. **Temporal Aggregation**: Daily → Weekly frequency conversion
5. **Holiday Adjustment**: Business day normalization for seasonal effects
6. **Smoothing**: 2-week rolling average to reduce noise
7. **Lag Feature Creation**: 18-period lag structures for temporal dependencies
8. **Competitive Metrics**: Top-N competitor analysis and spread calculations
9. **Economic Integration**: DGS5, VIX, CPI feature integration
10. **Feature Validation**: Quality checks and missing data handling

**Performance Characteristics:**
- **Execution Time**: ~45 seconds
- **Peak Memory**: 2.1GB
- **Output**: 160 weekly observations × 598 features

**Reference:** `src/features/engineer.py` - Feature engineering implementation

### Core Feature Categories

#### 1. Sales Target Features (2 features × 18 lags = 36 features)

**Primary Targets:**
- `sales`: Issue date aggregated sales (standard metric)
- `sales_by_contract_date`: Contract date aggregated sales (best practice from FIA v2.0)

**Lag Structure:**
- **Direction**: Both forward and backward lags
- **Purpose**: Capture sales momentum and temporal dependencies
- **Optimal Lag**: 5-week lag identified through AIC optimization (67% feature importance)

**Feature Names:**
- `sales_target_t0` through `sales_target_t17`
- `sales_target_contract_t0` through `sales_target_contract_t17`

#### 2. Competitive Rate Features (8 features × 18 lags = 144 features)

**Core Competitive Metrics:**
- `prudential_rate`: Our own cap rate (quality signaling effect, lag 0 allowed)
- `competitor_mid`: Market share weighted mean of competitors
- `competitor_core`: Core competitor group average

**Enhanced Competitive Metrics:**
- `competitor_median`: Median competitor rate (robust to outliers)
- `competitor_1st`, `competitor_2nd`, `competitor_3rd`: Individual top competitors
- `competitor_top3`: Mean of top 3 competitors
- `competitor_top5`: Mean of top 5 competitors (optimal feature, 23% importance)

**Lag Structure:**
- **Direction**: Backward only (predictive context)
- **Minimum Lag**: 3 weeks for competitive features (rate setting to effective date delay)
- **Optimal Lag**: 2-week competitive response identified in optimal model

**Feature Names:**
- `prudential_rate_current` through `prudential_rate_t17`
- `competitor_top5_t2` through `competitor_top5_t17`
- Similar patterns for all competitive metrics

**Reference:** `src/config/config_builder.py:53-92` - Lag column configurations

#### 3. Economic Indicator Features (2 features × 18 lags = 36 features)

**Treasury Rates:**
- `econ_treasury_5y`: 5-year Treasury constant maturity rate
- **Business Context**: Interest rate environment affects annuity demand baseline
- **Lag Structure**: Backward only (economic context)

**Market Volatility:**
- `market_volatility`: VIX (CBOE Volatility Index)
- **Business Context**: Volatility affects customer risk preferences
- **Lag Structure**: Backward only (market regime identification)

**Total Economic Features:** 36 features (2 indicators × 18 lags)

#### 4. Individual Carrier Rates (8 carriers × 18 lags = 144 features)

**Individual Carrier Tracking:**
- Allianz, Athene, Brighthouse, Equitable, Jackson, Lincoln, Symetra, Transamerica
- **Purpose**: Carrier-specific competitive dynamics
- **Aggregation**: Combined into competitive metrics via market share weighting

#### 5. Competitive Spread Features (~100 features)

**Spread Calculations:**
- Prudential rate - competitor metrics (various combinations)
- Prudential percentile rank among competitors
- Competitive positioning indicators

#### 6. Derived Economic Features (~138 features)

**Market Context:**
- Treasury rate changes and trends
- Volatility regime indicators
- Inflation-adjusted metrics

**Feature Engineering Summary:**

| Category | Base Features | Lag Periods | Total Features |
|----------|---------------|-------------|----------------|
| Sales Targets | 2 | 18 | 36 |
| Competitive Rates | 8 | 18 | 144 |
| Economic Indicators | 2 | 18 | 36 |
| Individual Carriers | 8 | 18 | 144 |
| Competitive Spreads | ~50 | varies | ~100 |
| Derived Economic | ~70 | varies | ~138 |
| **Total** | - | - | **598** |

**Reference:** README.md lines 240-254 - Data pipeline performance

### Market Share Weighting

A critical innovation for RILA products is the use of quarterly market share data to weight competitor rates:

**Weighting Formula:**
```python
Competitive_Rate_t = Σ(Rate_i,t × MarketShare_i) / Σ(MarketShare_i)
```

**Rationale:**
- Distribution is key to RILA sales success
- Competitors with wider distribution have greater competitive impact
- Market share serves as proxy for distribution reach
- Previous quarter sales indicate current distribution strength

**Why Unique to RILA:**
This weighting approach is specific to RILA/FlexGuard products. For MYGA/FIA products, previous sales do not have the same distributional implications, and equal weighting or alternative approaches may be more appropriate.

**Implementation:**
```python
weights = (df_ts_w[competitors].fillna(0).values *
           df_ts_w[competitors_weight].fillna(0).values)
df_ts_w['C'] = weights.sum(axis=1)
```

**Reference:** `src/features/competitive_analysis.py` - Market share weighting implementation

### Lag Structure Optimization

**Lag Control by Feature Type:**

**Sales Features (Bidirectional):**
- **Forward Lags**: For forecasting future sales (target variable)
- **Backward Lags**: For momentum effects (contract processing persistence)
- **Optimal**: 5-week lag for sales_target_contract (67% importance)

**Prudential Rate (No Minimum Lag):**
- **Lag 0 Allowed**: We control our own rate during rate setting
- **Strategic Value**: Current rate signal (10% importance in optimal model)

**Competitive Rates (3-Week Minimum Lag):**
- **Minimum Lag**: 3 weeks accounts for rate setting to effective date delay
- **Optimal**: 2-week competitive response (23% importance in optimal model)
- **Rationale**: Can only use historical competitor rates, not future rates

**Economic Indicators (Backward Only):**
- **Backward Lags**: Market context information
- **Purpose**: Regime identification and demand baseline

**AIC-Based Lag Selection:**

Rather than manually selecting lags, we use best subset selection with AIC to find appropriate lag combinations:
- **Evaluation**: Top 5 lowest AIC scores examined
- **Stability**: Best choices show similar lag structure across rankings
- **Revalidation**: Lag structure should be reconsidered at each model version release

**Reference:** `src/config/config_builder.py:53-92` - Lag direction configurations

### Data Preprocessing Steps

#### Weekly Aggregation

**Aggregation Dictionary:**
```python
{
    # Competitive features
    'C_weighted_mean': 'mean',
    'Prudential': 'mean',

    # Sales data
    'sales': 'sum',
    'sales_by_contract_date': 'sum',

    # Economic indicators
    'DGS5': 'last',
    'VIXCLS': 'last'
}
```

#### Holiday Adjustment

Weekly sales are adjusted for holidays by multiplying by 5 divided by the number of business days for that week. This normalizes for shortened trading weeks due to market holidays.

#### Smoothing

**Sales Smoothing:**
- 2-week rolling average to reduce noise
- Minimum weight time when making prediction is one week
- Prevents causal information leakage from past to feature

**Launch Anomaly Handling:**
Two anomalous weeks near FlexGuard launch where a backlog of applications was processed were addressed by smoothing to prevent model distortion.

#### Forward Filling

WINK data only shows entries when cap rates change. We create a complete time index and forward-fill rates to have effective rates defined for all dates where products were active.

**Reference:** `src/config/config_builder.py:95-143` - Weekly aggregation configuration

---

## 6. Model Framework

### Bootstrap Ensemble Architecture

**Training Process:**

1. **Data Preparation:**
   - 160 weekly observations with 598 engineered features
   - Train/validation split using expanding window cross-validation
   - Economic constraint validation on candidate features

2. **Feature Selection:**
   - 793 feature combinations evaluated via AIC
   - 193 combinations pass economic constraint validation (24.3%)
   - Optimal model: AIC = 5358.129 with 3-4 features

3. **Bootstrap Sampling:**
   - Bootstrap samples drawn with replacement (10,000 for inference, 1,000 for forecasting)
   - Each sample maintains temporal ordering
   - Independent Ridge regression trained per sample

4. **Ensemble Prediction:**
   - Mean prediction across bootstrap ensemble
   - Empirical confidence intervals (2.5%, 97.5% percentiles)
   - Coefficient sign consistency validated

**Reference:** `src/models/bootstrap_ridge.py` - Bootstrap ensemble training

### Current Optimal Model Structure

**Economically-Validated 3-Feature Model:**

| Feature | Lag | Coefficient Sign | Importance | Economic Theory |
|---------|-----|-----------------|------------|-----------------|
| `sales_target_contract_t5` | 5 weeks | β > 0 | 67% | Contract processing persistence |
| `competitor_top5_t2` | 2 weeks | β < 0 | 23% | Competitive market pressure |
| `prudential_rate_current` | 0 weeks | β > 0 | 10% | Quality signaling effect |

**Model Interpretation:**

1. **Sales Persistence (67% importance)**:
   - 5-week lagged sales momentum drives future sales
   - Contract processing and distribution channel effects
   - 6Y20B enhancement: Longer commitment amplifies persistence

2. **Competitive Response (23% importance)**:
   - Top 5 competitors' mean rate affects our competitive position
   - 2-week response window for competitive dynamics
   - Market share weighted for distribution impact

3. **Quality Signaling (10% importance)**:
   - Current Prudential rate signals product value
   - Higher rates indicate financial strength
   - 6Y20B context: Enhanced signaling via 20% buffer and 6-year term

**Economic Constraint Validation:**
- [PASS] All coefficients have theoretically expected signs
- [PASS] 100% sign consistency across bootstrap samples
- [PASS] All coefficients statistically significant at α = 0.05

**Reference:** README.md lines 300-310 - Current optimal model structure

### AIC-Based Feature Selection

**Akaike Information Criterion (AIC):**

```
AIC = 2k - 2ln(L)
```

Where:
- `k`: Number of parameters (features + intercept)
- `L`: Maximum likelihood of the model
- **Lower AIC = Better**: Balances goodness-of-fit with model complexity

**Feature Selection Process:**

1. **Generate Combinations**: Create feature combinations from lag structures
2. **Fit Models**: Train Ridge regression for each combination
3. **Calculate AIC**: Compute AIC for each model
4. **Economic Filtering**: Validate coefficient signs against theory
5. **Rank & Select**: Choose lowest AIC among economically valid models

**Selection Statistics:**
- **Total Combinations**: 793 evaluated
- **Economic Pass Rate**: 193/793 (24.3%) satisfy constraints
- **Optimal Model**: AIC = 5358.129
- **Feature Count**: 3 features (parsimony principle)

**Why AIC vs Other Criteria:**
- **Prevents Overfitting**: Penalizes excessive parameters
- **Information-Theoretic**: Grounded in information theory
- **Asymptotic Efficiency**: Selects true model as sample size → ∞
- **Practical**: Widely used in econometrics and time series

**Reference:** `src/features/selection/aic_engine.py` - AIC calculation engine

### Cross-Validation Methodology

**Time Series Validation Protocol:**

**Expanding Window Approach:**
- **Training Set**: Expands with each validation period
- **Temporal Order**: Strictly maintained (no data leakage)
- **Forecast Horizon**: 1-week ahead predictions
- **Strategic Window**: 2-week competitive response modeling

**Validation Structure:**
```
Week 1-50: Initial training
Week 51: First validation forecast
Week 52-102: Expanded training
Week 103: Second validation forecast
...
Total: 130+ out-of-sample forecasts (as of November 2025; increases weekly)
```

**Note on Validation Period:** The out-of-sample forecast count reflects model performance as of November 2025. This count increases weekly as new data becomes available and the model continues to generate validated predictions. The reported metrics represent cumulative performance across all validation periods from 2023-04-02 onwards.

**Performance Metrics:**
- **R² (Coefficient of Determination)**: Variance explained by model
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
- **Bootstrap Coverage**: % of actuals within 95% confidence bands

**Validation Results:**
- **Early Period**: MAPE = 8.1% (excellent accuracy in stable markets)
- **Late Period**: MAPE = 20.1% (indicates model drift monitoring requirement)
- **Overall**: 94.4% bootstrap coverage (well-calibrated uncertainty)

**Production Threshold:**
- **MAPE < 20%**: Required for continued production use
- **Coverage ∈ [90%, 97%]**: Well-calibrated confidence intervals
- **Drift Monitoring**: 13-week rolling MAPE with automated alerts

**Reference:** `src/models/cross_validation.py` - Cross-validation framework

### Model Refresh Cycle

**Production Schedule:**

**Weekly Business Intelligence Cycle:**
1. **Tuesday AM**: Automated data refresh from TDE/WINK systems
2. **Tuesday-Wednesday**: DVC pipeline execution with quality validation
3. **Wednesday PM**: Bootstrap model refresh with performance validation
4. **Thursday**: Strategic business review with uncertainty-informed decisions

**Model Retraining:**
- **Frequency**: Every 2 weeks as new sales and competitor data accumulates
- **Data Window**: Currently using all historical data (considering 3-year sliding window)
- **Validation**: Performance gates must pass before production deployment

**Monitoring:**
- **13-Week Rolling MAPE**: Temporal performance tracking
- **Coefficient Sign Consistency**: Economic constraint validation
- **Bootstrap Coverage**: Confidence interval calibration

**Reference:** README.md lines 641-652 - Strategic business integration

---

## 7. Model Performance & Validation

### Performance Metrics

#### Bootstrap Ensemble Performance

**Comprehensive Standard and Volatility-Weighted Metrics:**

| Metric | Model (Standard) | Benchmark (Standard) | Improvement | Model (Vol-Weighted) | Benchmark (Vol-Weighted) |
|--------|------------------|----------------------|-------------|----------------------|--------------------------|
| **R²** | **78.37%** | 57.54% | **+36.2%** | **77.60%** | 57.23% |
| **MAPE** | **12.74%** | 16.40% | **-22.3%** | **12.64%** | 16.20% |
| **Coverage** | **94.4%** | N/A | Well-calibrated | **94.4%** | N/A |
| **Forecasts** | **130+** (Nov 2025) | 130+ | Grows weekly | **130+** | 130+ |

**Note:** R² and MAPE are the primary business-relevant metrics for strategic pricing decisions. MAE and RMSE are intentionally excluded to focus on percentage-based variance explanation (R²) and percentage-based forecast error (MAPE).

**Key Performance Highlights:**
- **78.37% R²**: Model explains 78.37% of sales volume variance
- **36.2% R² Improvement**: Substantial gain over rolling average benchmark
- **12.74% MAPE**: Average forecast error for strategic planning
- **22.3% MAPE Improvement**: Significant reduction in forecast error
- **94.4% Coverage**: Well-calibrated 95% confidence intervals (target: 90-97%)

**Forecasting Scope:**
- **Total Forecasts**: 130+ out-of-sample periods (as of November 2025; grows weekly with new data)
- **Date Range**: 2023-04-02 onwards (ongoing validation)
- **Validation Protocol**: Expanding window cross-validation

**Reference:** README.md lines 395-403 - Bootstrap ensemble performance

#### Volatility-Weighted Performance Analysis

**Why Volatility Weighting Matters for RILA Products:**

RILA products are inherently tied to market volatility through their index-linked structure and downside buffer protection. During high-volatility periods:
- **Customer Behavior**: Flight to safety vs. opportunity seeking behavior shifts
- **Competitive Dynamics**: Rate competition intensifies or becomes coordinated
- **Buffer Protection Value**: 20% downside buffer becomes more valuable in volatile markets
- **Distribution Focus**: Risk-adjusted products receive greater advisor attention

**Weighting Methodology:**

Each weekly forecast is weighted by the CBOE Volatility Index (VIX) for that period:
- **Formula**: Weight_t = VIX_t / mean(VIX)
- **Effect**: High-volatility weeks receive proportionally higher weight in performance metrics
- **Validation**: Ensures model performance is tested across full market regime spectrum

**Business Interpretation:**

The model maintains minimal degradation from standard to volatility-weighted metrics:
- **R² degradation**: Only 0.77% (78.37% → 77.60%)
- **MAPE degradation**: Only 0.10% (12.74% → 12.64%)

This consistency demonstrates the Bootstrap Ridge model maintains stable predictive accuracy regardless of market conditions, providing confidence for:
- Strategic planning across economic cycles
- Rate-setting decisions during market stress
- Long-term competitive positioning stability
- Risk-adjusted strategic decision support

**Reference:** README.md lines 216-219 - Volatility performance analysis

### Feature Selection Validation

**AIC-Based Selection with Economic Constraints:**

**Selection Statistics:**
- **Total Combinations Evaluated**: 793 feature combinations across lag structures
- **Economic Constraint Pass Rate**: 193/793 (24.3%) pass microeconomic validation
- **Optimal Model**: AIC = 5358.129 for economically-constrained 3-feature model
- **Bootstrap Stability**: 100% coefficient sign consistency across bootstrap samples

**Economic Validity:**
All selected features satisfy theoretical expectations:
- [PASS] Own rate effect (β > 0): Quality signaling validated
- [PASS] Competitive effect (β < 0): Competitive pressure validated
- [PASS] Persistence effect (β > 0): Contract processing momentum validated
- [PASS] Statistical significance: All coefficients significant at α = 0.05

**Stability Analysis:**
- All bootstrap samples yield correct coefficient signs (validated across 10,000 estimators for inference)
- No spurious correlations or theoretically implausible relationships
- Robust to perturbations in training data

**Reference:** README.md lines 405-412 - Feature selection validation

### Temporal Performance Analysis

**Cross-Validation Results (125 Out-of-Sample Forecasts):**

**Early Period Performance:**
- **MAPE**: 8.1%
- **Context**: Excellent accuracy during stable market conditions
- **Period**: Initial validation windows with established patterns

**Late Period Performance:**
- **MAPE**: 20.1%
- **Context**: Indicates potential model drift
- **Requirement**: Triggers model refresh consideration

**Production Threshold:**
- **MAPE < 20%**: Required for continued production use
- **Current Status**: Meeting production standards
- **Monitoring**: 13-week rolling MAPE with automated alert system

**Model Drift Monitoring:**
- **Approach**: Continuous validation with performance gates
- **Frequency**: Weekly performance assessment
- **Action**: Model refresh if MAPE exceeds 20% threshold

**Reference:** README.md lines 413-420 - Temporal performance analysis

### Economic Constraint Validation

**Microeconomic Theory Enforcement:**

**1. Own Rate Coefficient (Quality Signaling):**
- **Constraint**: β > 0
- **Feature**: `prudential_rate_current`
- **Status**: [PASS] Validated across all bootstrap samples
- **Interpretation**: Higher Prudential rates increase sales (quality signal)

**2. Competitive Rate Coefficient (Competitive Pressure):**
- **Constraint**: β < 0
- **Feature**: `competitor_top5_t2`
- **Status**: [PASS] Validated across all bootstrap samples
- **Interpretation**: Higher competitor rates reduce our sales (competitive disadvantage)

**3. Sales Persistence Coefficient (Contract Processing):**
- **Constraint**: β > 0
- **Feature**: `sales_target_contract_t5`
- **Status**: [PASS] Validated across all bootstrap samples
- **Interpretation**: Recent sales momentum continues due to processing lags

**4. Statistical Significance:**
- **All Coefficients**: Significant at α = 0.05 level
- **Confidence**: Strong statistical evidence for relationships
- **Robustness**: Results stable across bootstrap samples

**Enforcement Process:**
- **Pre-Selection**: Filter candidates to economically valid only
- **Post-Training**: Validate final model coefficients
- **Production**: Continuous monitoring of coefficient signs

**Reference:** README.md lines 421-428 - Economic constraint validation

### Performance Visualization

**Key Visualizations:**

1. **Bootstrap Forecast Plot**:
   - Point forecasts with 95% confidence intervals
   - Actual vs predicted sales comparison
   - Coverage visualization

2. **Cross-Validation Performance**:
   - Expanding window performance over time
   - MAPE evolution across validation periods
   - Model drift detection

3. **Competitive Spread Analysis**:
   - Sales vs competitive rate positioning
   - Validation of economic relationships
   - Strategic rate scenario impacts

4. **Volatility-Weighted Analysis**:
   - Performance across market regimes
   - Model stability demonstration
   - Regime-specific MAPE comparison

**Image References:**
- `../images/model_performance/model_performance_summary_metrics_latest.png`
- `../images/model_performance/model_performance_volatility_weighted_analysis_latest.png`
- `../images/data_pipeline/data_pipeline_sales_vs_competitive_spreads_latest.png`

---

## 8. Results & Strategic Applications

### Price Elasticity Estimates

**Strategic Rate Scenario Analysis:**

The Bootstrap Ridge model provides price elasticity estimates across rate scenarios from 0 to 450 basis points relative to current rates:

**Percentage Impact Analysis:**
- **0 bps**: Baseline (current rate positioning)
- **+100 bps**: Estimated sales impact with 95% confidence intervals
- **+200 bps**: Strategic rate increase scenario
- **+300 bps**: Aggressive rate positioning
- **+450 bps**: Maximum competitive rate scenario

**Dollar Impact Projections:**
- Point estimates for revenue forecasting
- 95% confidence bands for risk assessment
- $16.8M weekly uncertainty quantification

**Visualization:**
- Percentage impact plot shows relative sales changes
- Dollar impact plot shows revenue projections
- Both include 95% bootstrap confidence intervals

**Image References:**
- `../images/business_intelligence/business_intelligence_price_elasticity_confidence_intervals_pct_latest.png`
- `../images/business_intelligence/business_intelligence_price_elasticity_confidence_intervals_dollars_latest.png`

### Strategic Pricing Recommendations

**Decision Support Framework:**

**Revenue Forecasting:**
- Point estimates (bootstrap ensemble mean) for budget planning
- 95% confidence intervals for risk assessment
- Scenario analysis for strategic planning

**Competitive Analysis:**
- Market share weighted competitive intelligence
- 8-carrier competitive landscape monitoring
- Real-time competitive response tracking

**Risk Assessment:**
- $16.8M weekly uncertainty quantification
- Confidence interval coverage for downside protection
- Scenario stress testing for strategic decisions

**Strategic Timing:**
- 2-week competitive response window modeling
- Optimal rate change timing based on competitive dynamics
- Market regime consideration (volatility, interest rates)

**Reference:** README.md lines 146-153 - Strategic applications

### Competitive Analysis

**8-Carrier Market Intelligence:**

**Competitor Coverage:**
1. **Prudential** [2979]: FlexGuard 6Y20B (our product)
2. **Athene** [2772, 3409]: Multiple product lines
3. **Brighthouse** [2319]: Flagship RILA product
4. **Equitable** [2286, 3282, 3853]: Structured Capital Strategies
5. **Jackson** [3351, 3714]: Elite Access products
6. **Lincoln** [2924]: Lincoln OptiBlend
7. **Symetra** [3263, 3751]: Symetra Edge products
8. **Transamerica** [3495]: TransElite RILA

**Market Share Weighting:**
- Quarterly distribution data used as proxy for competitive influence
- Competitors with wider distribution weighted higher
- Top 5 competitors identified as most impactful (23% feature importance)

**Competitive Positioning:**
- Real-time rate differential tracking
- Percentile rank monitoring among competitors
- Strategic spread analysis for pricing decisions

**Reference:** README.md lines 88-100 - Competitive intelligence evolution

### Business Intelligence Outputs

**Weekly Deliverables:**

**1. Strategic Dashboards:**
- Price elasticity curves with confidence intervals
- Competitive rate positioning analysis
- Sales forecast with uncertainty quantification

**2. Tableau Integration:**
- BI team consumption-ready exports
- Interactive competitive analysis
- Historical performance tracking

**3. Executive Summaries:**
- Key metrics and trends
- Strategic recommendations
- Risk-adjusted forecasts

**4. Technical Reports:**
- Model performance validation
- Economic constraint verification
- Temporal stability assessment

**Export Formats:**
- **Visualizations**: High-resolution PNG (300 DPI)
- **Data**: CSV for BI team, Parquet for data science
- **Models**: Serialized bootstrap ensembles
- **Metadata**: JSON performance metrics and validation results

**Reference:** `src/visualization/business_communication.py` - Executive dashboards

### Operational Risk Management

**Model Risk Mitigation:**

**Bootstrap Ensemble:**
- 100-estimator ensemble reduces single-model risk
- Robust uncertainty quantification
- Consistent performance across market regimes

**Economic Constraints:**
- Theory-based validation prevents spurious correlations
- Coefficient signs enforced for interpretability
- Continuous monitoring of constraint satisfaction

**Performance Monitoring:**
- Weekly validation with automated alerts
- 13-week rolling MAPE tracking
- Production threshold enforcement (MAPE < 20%)

**Version Control:**
- Complete model version history via DVC
- Incident recovery capabilities
- Audit trail for regulatory compliance

**Business Continuity:**
- DVC automation eliminates manual errors
- Cross-account AWS security
- Complete data lineage tracking
- SOX-compliant audit trails

**Reference:** README.md lines 625-638 - Business continuity & risk management

---

## 9. Conclusions & Recommendations

### Key Findings

**Model Performance:**
- **78.37% R²**: Excellent explanatory power for strategic planning
- **12.74% MAPE**: Strong forecast accuracy for weekly business cycle
- **94.4% Coverage**: Well-calibrated confidence intervals for risk management
- **22.3% MAPE Improvement**: Substantial performance gain over benchmark

**Economic Validation:**
- All selected features satisfy microeconomic theory constraints
- Quality signaling effect confirmed (β > 0 for own rates)
- Competitive pressure validated (β < 0 for competitor rates)
- Contract persistence identified (5-week optimal lag)

**Strategic Insights:**
- Sales momentum (67% importance) dominates short-term forecasting
- Competitive response (23% importance) critical for strategic positioning
- Quality signaling (10% importance) enables premium rate positioning
- 2-week competitive response window guides timing decisions

### Recommendations

**Operational:**
1. **Continue Weekly Refresh Cycle**: Maintain Tuesday-Thursday business intelligence cycle
2. **Monitor Model Drift**: Track 13-week rolling MAPE, trigger refresh if > 20%
3. **Validate Economic Constraints**: Ensure coefficient signs remain theoretically sound
4. **Maintain Performance Gates**: R² > 50%, MAPE < 25%, Coverage ∈ [90%, 97%]

**Strategic:**
1. **Leverage Confidence Intervals**: Use 95% bands for risk-adjusted strategic decisions
2. **Competitive Intelligence**: Monitor top 5 competitors with market share weighting
3. **Timing Optimization**: Align rate changes with 2-week competitive response window
4. **Scenario Planning**: Use rate scenario analysis for strategic planning

**Methodological:**
1. **Consider 3-Year Sliding Window**: Evaluate if older data should be excluded
2. **Expand to Multi-Buffer Analysis**: Explore 10%, 15%, 20% buffer competitive dynamics
3. **Multi-Term Modeling**: Extend to 1-year, 3-year, 6-year term portfolio optimization
4. **Distribution Channel Integration**: Enhance market share weighting with channel data

**Infrastructure:**
1. **Maintain DVC Automation**: Continue pipeline automation for operational resilience
2. **Cross-Account Security**: Uphold enterprise security standards
3. **Documentation Currency**: Keep technical and business documentation current
4. **Quality Assurance**: Maintain comprehensive test suite for regression prevention

### Future Enhancements

**Short-Term (3-6 months):**
- Implement automated model drift detection and alerting
- Expand Tableau integration for self-service BI
- Develop mobile-friendly executive dashboards
- Enhance economic indicator integration (unemployment, consumer confidence)
- **Evaluate Decay Weighting Approach**: Consider implementing FIA v2.0's decay weighting factor (0.99) as an alternative to sliding time windows. This approach gives progressively more weight to recent observations while gradually reducing the influence of older data, which may improve model responsiveness to recent market dynamics without the discontinuity of hard time cutoffs.

**Medium-Term (6-12 months):**
- Multi-buffer portfolio optimization (10%, 15%, 20% buffers)
- Multi-term strategic modeling (1Y, 3Y, 6Y portfolio)
- Distribution channel data integration beyond market share proxy
- Advanced competitive intelligence with product feature analysis

**Long-Term (12+ months):**
- Real-time streaming data integration for intra-week updates
- Machine learning enhancement while maintaining interpretability
- Multi-product portfolio optimization (RILA + FIA + MYGA)
- Regulatory scenario stress testing framework

### Success Metrics

**Model Performance:**
- **R² ∈ [0.70, 0.85]**: Maintain high explanatory power
- **MAPE ∈ [10%, 20%]**: Strong forecast accuracy
- **Coverage ∈ [90%, 97%]**: Well-calibrated uncertainty
- **Stability**: Consistent performance across market regimes

**Business Impact:**
- **Decision Support**: 100% of rate decisions informed by model
- **Competitive Intelligence**: Real-time monitoring of 8 major carriers
- **Forecast Accuracy**: Weekly forecasts within 95% confidence bands
- **Strategic Planning**: Scenario analysis integrated into quarterly planning

**Operational Excellence:**
- **Automation**: 100% pipeline execution via DVC
- **Refresh Cycle**: Weekly updates maintained
- **Quality Gates**: 100% pass rate for economic constraints
- **Documentation**: Complete technical and business documentation maintained

### Conclusion

The RILA 6Y20B Price Elasticity Analysis System represents a mature, production-ready framework for strategic pricing decisions. The Bootstrap Ridge ensemble methodology provides robust uncertainty quantification while maintaining the interpretability essential for business decision-making and regulatory compliance.

The system's 78.37% R² and 22.3% MAPE improvement over benchmarks, combined with well-calibrated confidence intervals (94.4% coverage), demonstrates both statistical rigor and practical business value. Economic constraint validation ensures theoretical soundness, while the 2-week competitive response window enables optimal strategic timing.

The evolution from RILA v1.2's bagged linear models to the current Bootstrap Ridge framework represents a significant methodological advancement:
- Enhanced uncertainty quantification through ensemble methods
- Automated economic constraint validation
- Improved computational efficiency (2-stage vs 4-stage pipeline)
- Robust performance across market volatility regimes

For data scientists and analysts, this system exemplifies the balance between statistical sophistication and business interpretability. The comprehensive feature engineering (598 features), rigorous AIC-based selection (193/793 economically valid), and bootstrap ensemble approach (10,000 estimators for inference, 1,000 for forecasting) provide both accuracy and reliability for strategic decisions involving millions of dollars in weekly revenue.

**Production Status**: Phase 2B Production - Strategic Pricing Ready

---

## 10. Appendix: Alternative Approaches

This appendix documents alternative methodological approaches from the RILA v1.2 and FIA v2.0 documents. While the current implementation uses Bootstrap Ridge regression (local linear approximation), understanding these alternatives provides context for methodological evolution and potential future enhancements.

### Prediction vs Causal Inference for Prescriptive Pricing

**The Critical Distinction for Strategic Decisions:**

A fundamental methodological distinction separates **predictive forecasting** from **causal inference** in pricing models:

- **Predictive models** answer: "What will sales be next week?" (forecasting observed outcomes)
- **Causal inference** answers: "What if we change our cap rate by 50 bps?" (prescriptive recommendations for unobserved counterfactuals)

Strategic pricing decisions require prescriptive guidance based on causal understanding of how rate changes affect sales, not just accurate forecasts of what will happen under current conditions.

**Economic Constraints as Necessary Validation:**

For reliable causal inference in pricing models, economic constraints serve as **necessary validation** that the model captures true causal mechanisms rather than spurious correlations (Pearl, 2009; Imbens & Rubin, 2015). In our framework:

- **Quality Signaling** (β > 0): Higher own rates signal product value and financial strength
- **Competitive Pressure** (β < 0): Competitor rate increases reduce our relative advantage
- **Contract Persistence** (β > 0): Recent sales momentum continues due to processing lags

These theoretically-grounded constraints function as **identifying restrictions** that help distinguish structural causal parameters from reduced-form correlations. While not alone sufficient for establishing causality (randomized experiments remain the gold standard), economic constraints provide necessary validation for observational pricing data where experiments are impractical.

**Why Non-Linear Models Cannot Provide Causal Inference:**

Black-box models (XGBoost, neural networks) excel at prediction but cannot meaningfully enforce or validate economic constraints required for causal inference. While technically possible to add constraints via custom loss functions or constrained optimization:

1. This defeats the flexibility advantages that justify model complexity
2. Coefficient-level interpretability remains absent for "what if" business scenarios
3. No theoretical validation ensures the model captures causal mechanisms vs spurious correlations

High predictive accuracy does not guarantee reliable prescriptive recommendations. A model may perfectly forecast sales by learning spurious patterns (e.g., seasonality correlated with rate changes) while misidentifying the causal effect of rate adjustments.

**Alternative Approaches and Future Directions:**

Other causal inference identification strategies exist: randomized controlled trials (RCTs), instrumental variables (IV), regression discontinuity (RDD), and difference-in-differences (DID). For ongoing strategic pricing with observational data, structural models with economic constraints provide the most practical framework.

Future enhancements could explore **DoubleML** (double/debiased machine learning) techniques (Chernozhukov et al., 2018) that combine ML flexibility with causal inference properties for treatment effect estimation while maintaining theoretical rigor.

**Reference:** See Section 3 "Inference vs. Prediction: A Critical Distinction" for foundational discussion of why interpretable models with economic constraints are chosen for strategic inference despite potentially lower predictive accuracy.

---

### A. Local Linear Approximations (RILA v1.2 Approach)

**Original RILA v1.2 Methodology:**

**Model Type:** Bagged linear model with AIC feature selection

**Core Concept:**
Linear models are appropriate when exploring prices near equilibrium values (small rate changes around current market positioning). This is the foundation for the current Bootstrap Ridge approach.

**Mathematical Framework:**
```
Sales = β₀ + β₁(Prudential_Rate) + β₂(Competitor_Rate) + β₃(Sales_Lag) + ε
```

**Advantages:**
- **Interpretability**: Clear coefficient interpretation for business decisions
- **Simplicity**: Fast execution suitable for weekly business cycles
- **Stability**: Parametric approach stable under rate perturbations
- **Validation**: Economic constraints easily enforced and validated

**Limitations:**
- **Single Model**: No ensemble uncertainty quantification
- **Manual Tuning**: Lag structure required manual optimization
- **Limited Uncertainty**: Standard errors only, no bootstrap confidence intervals

**When to Use:**
- Strategic rate adjustments (±50-100 bps around equilibrium)
- Established products with stable competitive dynamics
- Weekly operational decisions with interpretability requirements

**Reference:** PEA_RILA_v1_2.docx - Original methodology section

### B. Global Logit Transformations (FIA v2.0 Approach)

**FIA v2.0 Enhancement:**

**Model Type:** Logit transformation for global approximations

**Core Concept:**
Sigmoid/Logit models are appropriate when exploring prices from market entry cutoffs or major product repositioning. Useful for large market shifts or new product launches.

**Mathematical Framework:**
```
logit(Sales_Normalized) = β₀ + β₁(Prudential_Rate) + β₂(Competitor_Rate) + ...
```

Where `Sales_Normalized` is scaled to [0, 1] range.

**Key Innovations from FIA v2.0:**
1. **application_signed_date**: Use application date instead of contract issue date to prevent time-ordering leakage
2. **Decay Weighting**: 0.99 decay factor for temporal emphasis on recent data
3. **Holiday Indicators**: Explicit holiday indicator variables vs. business day adjustment
4. **No Autoregressive Term**: Found unnecessary with proper temporal weighting

**Advantages:**
- **Global Validity**: Works across wide range of rate scenarios
- **Bounded Output**: Logit transformation ensures predictions in [0, 1]
- **Flexible**: Can model non-linear relationships near market boundaries
- **Launch Modeling**: Appropriate for new product introductions

**Limitations:**
- **Less Interpretable**: Logit coefficients harder to explain to business stakeholders
- **Complexity**: Additional transformation step adds computational overhead
- **Overkill for Established Products**: Linear sufficient for strategic adjustments

**When to Use:**
- New product launches with uncertain market response
- Major product repositioning (e.g., 10% buffer → 30% buffer)
- Exploring extreme rate scenarios (>200 bps from equilibrium)
- Market entry or exit decisions

**Reference:** PE_FIA_v2_0.docx - Global approximations methodology

**Clarification on "No Autoregressive Term":**
FIA v2.0's statement that "no autoregressive term" is needed requires technical clarification for RILA 6Y20B. The FIA v2.0 innovation eliminated explicit AR(p) time-series models (e.g., ARIMA structures with dedicated autoregressive parameters). However, RILA 6Y20B DOES use lagged sales features (`sales_target_contract_t5`), which constitute autoregressive modeling in the broader sense - using past values of the dependent variable as predictors. The distinction is that 6Y20B treats lagged sales as engineered features within the Ridge regression framework rather than as explicit autoregressive terms in a time-series model. Both approaches capture sales momentum and contract processing persistence; they differ in statistical formulation but serve the same economic purpose of modeling temporal dependence.

### C. Non-Linear Alternatives (Considered but Not Adopted)

**XGBoost / Gradient Boosting:**

**Why Considered:**
- High predictive accuracy on many datasets
- Automatic feature interaction detection
- Handles non-linear relationships

**Why Not Adopted:**
- **Less Interpretable**: Black-box model difficult to explain to business
- **Hyperparameter Sensitive**: Results highly dependent on tuning
- **High Variance**: Inconsistent behavior under small rate perturbations
- **No Clear Benefit**: Similar or worse performance on our inference task
- **Insufficient for Causal Inference**: While XGBoost excels at prediction, reliable causal inference for prescriptive pricing requires economic constraints (quality signaling β > 0, competitive pressure β < 0) that black-box models cannot meaningfully enforce. Though technically possible via custom loss functions, this approach: (1) defeats flexibility advantages, (2) lacks coefficient-level interpretability for "what if" scenarios, and (3) provides no theoretical validation that the model captures causal mechanisms vs spurious correlations. High predictive accuracy alone does not guarantee reliable prescriptive recommendations for strategic pricing decisions.

**When Might Be Appropriate:**

These methods are appropriate for **pure forecasting tasks** where predictive accuracy is the sole objective and causal inference for prescriptive recommendations is not required:

- Pure prediction tasks where interpretability is secondary
- Very large datasets (>1000 observations) with complex interactions
- Scenarios where model stability under perturbations is less critical

**Neural Networks:**

**Why Considered:**
- State-of-the-art performance on many prediction tasks
- Extreme flexibility in modeling complex relationships

**Why Not Adopted:**
- **Interpretability**: Even less interpretable than XGBoost
- **Data Requirements**: Require significantly more data (>5000 observations)
- **Overfitting Risk**: High risk with our limited historical record
- **Incompatible with Prescriptive Causal Inference**: Neural networks prioritize predictive accuracy but cannot provide the theoretically-grounded causal inference required for strategic pricing decisions. Economic constraints (quality signaling β > 0, competitive pressure β < 0) are necessary validation that models capture true causal mechanisms, not spurious correlations. While constrained neural networks exist technically, they sacrifice the flexibility that justifies their complexity while still lacking the coefficient-level interpretability essential for prescriptive "what if" recommendations to business stakeholders.
- **Computational Cost**: Training and inference much slower

**When Might Be Appropriate:**

These methods are appropriate for **pure forecasting tasks** where predictive accuracy is the sole objective and causal inference for prescriptive recommendations is not required:

- Massive datasets with rich feature sets
- Pure prediction tasks with no interpretability requirements
- Real-time pricing systems with streaming data

### D. Methodological Evolution Summary

**RILA v1.2 → FIA v2.0 → RILA 6Y20B Bootstrap:**

| Dimension | RILA v1.2 | FIA v2.0 | RILA 6Y20B (Current) |
|-----------|-----------|----------|---------------------|
| **Model Type** | Bagged linear | Logit transformation | Bootstrap Ridge |
| **Uncertainty** | Standard errors | Standard errors | Bootstrap CI (95%) |
| **Date Field** | contract_issue_date | application_signed_date | application_signed_date |
| **Lag Optimization** | Manual | Manual | AIC-based |
| **Economic Constraints** | Manual check | Manual check | Automated validation |
| **Ensemble Size** | Small bag | N/A | 10K inference / 1K forecast |
| **Regularization** | None | None | L2 (α=1.0) |
| **Pipeline Stages** | 4-stage | Similar to v1.2 | 2-stage optimized |

**Best Practices Adopted:**
- [PASS] **application_signed_date** from FIA v2.0 (prevents time-ordering leakage)
- [PASS] **Bootstrap ensemble** enhancement (robust uncertainty quantification)
- [PASS] **Automated economic constraints** (prevents spurious correlations)
- [PASS] **L2 regularization** (optimal bias-variance trade-off)
- [PASS] **DVC pipeline automation** (operational resilience)

**Best Practices NOT Adopted (with rationale):**
- [FAIL] **Logit transformation**: Unnecessary for established product strategic adjustments
- [FAIL] **Decay weighting**: Bootstrap ensemble provides better temporal weighting
- [FAIL] **Explicit holiday indicators**: Business day adjustment simpler and effective

### E. Comparative Performance Analysis

**Hypothetical Performance Comparison:**

| Approach | R² | MAPE | Interpretability | Training Time | Uncertainty |
|----------|-----|------|------------------|---------------|-------------|
| **RILA v1.2 (Bagged Linear)** | ~55% | ~17% | 5/5 | Fast (< 5 min) | Standard errors |
| **FIA v2.0 (Logit)** | ~58% | ~16% | 4/5 | Medium (~10 min) | Standard errors |
| **Bootstrap Ridge (Current)** | 78.37% | 12.74% | 5/5 | Medium (~30-45 min) | Bootstrap 95% CI |
| **XGBoost (hypothetical)** | ~80%? | ~12%? | 2/5 | Slow (~60 min) | Quantile regression |
| **Neural Network (hypothetical)** | ~82%? | ~11%? | 1/5 | Very slow (>2 hrs) | Dropout uncertainty |

**Key Takeaway:**
Bootstrap Ridge achieves strong performance (78.37% R²) while maintaining full interpretability (5-star rating). Non-linear alternatives might offer marginal accuracy gains but sacrifice the interpretability essential for business decision-making and regulatory compliance.

### F. When to Reconsider Alternative Approaches

**Triggers for Methodology Review:**

**1. Performance Degradation:**
- If MAPE exceeds 25% consistently for 13+ weeks
- If bootstrap coverage drops below 85%
- If R² falls below 50%

**2. Market Structure Changes:**
- New major competitor enters market
- Regulatory changes affecting product structure
- Major distribution channel shifts

**3. Product Portfolio Expansion:**
- Multi-product optimization needs (RILA + FIA + MYGA)
- Multi-buffer portfolio (10%, 15%, 20% simultaneous)
- Multi-term portfolio (1Y, 3Y, 6Y simultaneous)

**4. Data Availability Increases:**
- If observation count exceeds 500 weeks
- If granular distribution channel data becomes available
- If customer-level data enables segmentation modeling

**Recommendation:**
Continue with Bootstrap Ridge for current RILA 6Y20B strategic pricing. Revisit alternatives annually or when triggers above are encountered.

---

## Image Reference Guide

**For manual Word document finishing, embed the following images:**

### Methodology Section
- `notebooks/extracted_media/media/image1.png`
  - **Location**: In "Methodology" section after RILA 6Y20B evolution table
  - **Caption**: "Figure 1: Theoretical Price-Response Function Approaches - (A) Linear for local approximations, (B) Constant Elasticity, (C) Logit for global approximations"

### Data Pipeline Section
- `../images/data_pipeline/data_pipeline_sales_vs_competitive_spreads_latest.png`
  - **Location**: After "Data Sources" section
  - **Caption**: "Figure 2: FlexGuard Sales Performance vs Competitive Rate Positioning - Validation of economic relationships"

### Model Performance Section
- `../images/model_performance/model_performance_summary_metrics_latest.png`
  - **Location**: In "Model Performance & Validation" section
  - **Caption**: "Figure 3: Bootstrap Ridge Model Achievement: 78.37% R² and 12.74% MAPE"

- `../images/model_performance/model_performance_comprehensive_forecasting_analysis_latest.png`
  - **Location**: In "Model Performance & Validation" section after volatility analysis
  - **Caption**: "Figure 4: Comprehensive Time-Series Forecasting Analysis with Bootstrap Confidence Bands"

- `../images/model_performance/model_performance_volatility_weighted_analysis_latest.png`
  - **Location**: In "Temporal Performance Analysis" subsection
  - **Caption**: "Figure 5: Volatility-Weighted 13-Week Rolling MAPE Analysis - Consistent performance across market regimes"

### Business Intelligence Section
- `../images/business_intelligence/business_intelligence_price_elasticity_confidence_intervals_pct_latest.png`
  - **Location**: In "Results & Strategic Applications" section
  - **Caption**: "Figure 6a: Percentage Impact: Strategic rate scenarios with 95% confidence intervals"

- `../images/business_intelligence/business_intelligence_price_elasticity_confidence_intervals_dollars_latest.png`
  - **Location**: Next to percentage impact image
  - **Caption**: "Figure 6b: Dollar Impact: Revenue projections with bootstrap uncertainty quantification"

---

## References

1. Phillips, Robert. *Pricing and Revenue Optimization*. Stanford University Press, 2005. https://doi.org/10.1515/9780804781640

2. James, G., Witten, D., Hastie, T. and Tibshirani, R. *An Introduction to Statistical Learning: With Applications in R*. 2nd Edition, Springer, 2021. https://doi.org/10.1007/978-1-0716-1418-1

3. Pearl, Judea. *Causality: Models, Reasoning, and Inference*. 2nd Edition, Cambridge University Press, 2009. https://doi.org/10.1017/CBO9780511803161

4. Imbens, Guido W. and Rubin, Donald B. *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction*. Cambridge University Press, 2015. https://doi.org/10.1017/CBO9781139025751

5. Chernozhukov, Victor, Chetverikov, Denis, Demirer, Mert, Duflo, Esther, Hansen, Christian, Newey, Whitney, and Robins, James. "Double/debiased machine learning for treatment and structural parameters." *The Econometrics Journal*, 21(1):C1-C68, 2018. https://doi.org/10.1111/ectj.12097

6. **Internal Repository References:**
   - `README.md` - Comprehensive system overview and architecture
   - `src/config/config_builder.py` - Configuration system implementation
   - `src/models/bootstrap_ridge.py` - Bootstrap ensemble methodology
   - `src/features/selection/` - AIC-based feature selection with constraints
   - `notebooks/00_data_pipeline_refactored.ipynb` - Data extraction and feature engineering
   - `notebooks/01_price_elasticity_inference_refactored.ipynb` - Bootstrap modeling and analysis

7. **Source Documents:**
   - `PEA_RILA_v1_2 (1).docx` - Original RILA v1.2 methodology (1Y10B product)
   - `PE_FIA_v2_0 (1).docx` - FIA v2.0 enhancements and best practices

---

**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Repository:** `annuity-price-elasticity-v3`
**Contact:** Annuities Data Science Team
**Status:** Production-Ready (Phase 2B)


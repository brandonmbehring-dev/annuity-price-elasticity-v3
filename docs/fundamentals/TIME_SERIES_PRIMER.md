# Time Series Fundamentals Primer

**For data scientists new to time series analysis.**

This primer covers the essential concepts you'll need for the RILA price elasticity models. Read this alongside the domain documentation.

---

## What is Time Series Data?

Time series data is observations ordered by time. Our RILA data is **weekly**:

```
Week 1: Sales = $1.2M, Rate = 9.5%
Week 2: Sales = $1.1M, Rate = 9.3%
Week 3: Sales = $1.4M, Rate = 9.8%
...
```

**Key property**: Observations are NOT independent. Today's sales depend on yesterday's conditions.

---

## Core Concepts

### 1. Lags

A **lag** is a past value of a variable.

```
Time    Sales_t    Sales_lag1    Sales_lag2
t=1     100        -             -
t=2     110        100           -
t=3     105        110           100
t=4     120        105           110
```

In our notation:
- `_current` or `_t0` = value at time t (now)
- `_t1` = value at time t-1 (1 week ago)
- `_t2` = value at time t-2 (2 weeks ago)

**Why lags matter**: Customers don't respond instantly. A rate change takes time to affect sales.

### 2. Stationarity

A stationary series has **constant statistical properties** over time:
- Same mean
- Same variance
- Same autocorrelation structure

**Non-stationary example**: Sales trending upward over years

```
Year 1: Mean sales = $1M
Year 2: Mean sales = $1.2M    <- Mean is changing!
Year 3: Mean sales = $1.5M
```

**Why it matters**: Many statistical methods assume stationarity. If your data isn't stationary, you may need to transform it (differencing, detrending).

### 3. Autocorrelation

**Autocorrelation** measures how correlated a variable is with its own past values.

```
High autocorrelation: If sales were high last week, they're likely high this week
Low autocorrelation:  This week's sales are independent of last week's
```

In RILA data, we see positive autocorrelation—sales tend to persist.

**ACF (Autocorrelation Function)**: Shows correlation at each lag

```
Lag 1: 0.85  <- High: adjacent weeks are similar
Lag 2: 0.72
Lag 3: 0.61
Lag 4: 0.45  <- Decaying: more distant weeks less correlated
```

### 4. Seasonality

**Seasonality** is a repeating pattern at fixed intervals.

```
RILA Sales Seasonality:
- Q4: Highest (year-end planning)
- Q2: Lowest (post-tax season)
```

We control for seasonality with quarter dummy variables (Q1, Q2, Q3; Q4 is baseline).

---

## Time Series in RILA Models

### Lag Structure

We use lagged features to capture delayed responses:

| Feature | Lag | Interpretation |
|---------|-----|----------------|
| `prudential_rate_current` | 0 | Today's rate (treatment) |
| `competitor_mid_t2` | 2 | Competitor rate from 2 weeks ago |
| `competitor_mid_t3` | 3 | Competitor rate from 3 weeks ago |

**Why t-2 and t-3?** Feature selection found these lags most predictive. Customers need time to:
1. See rate changes
2. Compare options
3. Complete applications

### Why Lag-0 Competitors Are Problematic

This is a **causal** issue, not a time series issue. See `ECONOMETRICS_PRIMER.md`.

Short version: Competitor rates at t=0 and our sales at t=0 both respond to the same market conditions, creating spurious correlation.

### Handling Non-Stationarity

Our data may have:
- **Trends**: Gradual increase in sales over years
- **Level shifts**: Sudden changes (new product launch)

We handle this by:
1. Using **logit transformation** (bounds sales between 0 and max)
2. Including **time controls** (counter, quarter dummies)
3. Using **differenced features** when appropriate (C_diff = rate momentum)

---

## Key Diagnostics

### 1. Durbin-Watson Test

Tests for autocorrelation in residuals.

```
DW statistic:
  ~2.0: No autocorrelation (good)
  <1.5: Positive autocorrelation (common in time series)
  >2.5: Negative autocorrelation (rare)
```

**What to do**: If DW << 2, consider HAC standard errors (Newey-West).

### 2. Augmented Dickey-Fuller (ADF) Test

Tests for stationarity.

```
Null hypothesis: Series has a unit root (non-stationary)

p-value < 0.05: Reject null -> series is stationary
p-value > 0.05: Cannot reject -> may be non-stationary
```

### 3. Ljung-Box Test

Tests for significant autocorrelation at multiple lags.

```
p-value < 0.05: Significant autocorrelation present
p-value > 0.05: No significant autocorrelation
```

---

## Common Time Series Traps

### Trap 1: Using Future Information

```python
# WRONG: Uses future data to create rolling average
df['rolling_mean'] = df['sales'].rolling(window=4, center=True).mean()
#                                                 ^^^^^^ includes future!

# RIGHT: Only use past data
df['rolling_mean'] = df['sales'].rolling(window=4).mean()
```

In RILA models, we use **walk-forward validation** to prevent future leakage.

### Trap 2: Ignoring Autocorrelation in Standard Errors

```python
# Standard OLS assumes independent errors
# With autocorrelated data, standard errors are UNDERESTIMATED

# Solution: Use HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac

model = OLS(y, X).fit()
hac_cov = cov_hac(model)  # Robust standard errors
```

### Trap 3: Confusing Correlation with Causation

Time series makes this easy:
- Sales and competitor rates are correlated
- But correlation ≠ causation
- Both may be driven by underlying market conditions

**Solution**: Use lagged features and proper causal reasoning. See `ECONOMETRICS_PRIMER.md`.

---

## Walk-Forward Cross-Validation

Traditional CV shuffles data randomly. With time series, this leaks future information!

**Walk-forward CV** respects time ordering:

```
Fold 1: Train on [1-50], Test on [51-60]
Fold 2: Train on [1-60], Test on [61-70]
Fold 3: Train on [1-70], Test on [71-80]
...

Never: Train on [50-100], Test on [1-50]  <- Testing on past!
```

**RILA implementation**: We use `training_cutoff_date` to enforce this.

---

## Quick Reference

| Concept | What It Is | RILA Application |
|---------|-----------|------------------|
| Lag | Past value | `competitor_mid_t2` = 2 weeks ago |
| Stationarity | Constant statistics | Logit transform helps |
| Autocorrelation | Self-correlation | Sales persist week-to-week |
| Seasonality | Repeating pattern | Q4 highest, Q2 lowest |
| Walk-forward CV | Time-respecting validation | `training_cutoff_date` |

---

## Further Reading

- **RILA-specific**: `knowledge/analysis/CAUSAL_FRAMEWORK.md`
- **Feature engineering**: `knowledge/analysis/FEATURE_RATIONALE.md`
- **Academic**: Hyndman & Athanasopoulos, "Forecasting: Principles and Practice" (free online)

---

## Practical Exercise

Try this to understand autocorrelation:

```python
import pandas as pd
from statsmodels.stats.stattools import durbin_watson

# Load fixture data
df = pd.read_parquet("tests/fixtures/rila/final_weekly_dataset.parquet")

# Check autocorrelation of sales
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

plot_acf(df['sales'].dropna(), lags=20)
plt.title("Sales Autocorrelation")
plt.show()

# High ACF at lag 1-4 = sales are persistent (not random noise)
```

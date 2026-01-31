# Common Pitfalls and Best Practices

**Version**: 1.0.0 | **Last Updated**: 2026-01-31

This document describes the most common mistakes in annuity price elasticity modeling and how to avoid them. Each pitfall includes:
- **What goes wrong** and why
- **Don't/Do** code examples
- **Which validation gate catches it**
- **Business impact** if deployed

---

## Pitfall #1: Lag-0 Competitor Features

### The Problem

Using competitor rates from the same period (t=0) violates causal identification. When competitor rates at time t are used to predict sales at time t, the model cannot distinguish causation from correlation.

**Why This Happens**:
- Competitors may be responding to the SAME market signals as us
- Reverse causality: our pricing may influence competitor pricing
- Even if predictive, coefficients have no causal interpretation

### Don't

```python
# WRONG: Competitor rates at t=0 (contemporaneous)
features = [
    "prudential_rate_t0",       # Own rate at t=0 is OK
    "competitor_rate_t0",       # FORBIDDEN: Lag-0 competitor!
    "competitor_weighted_t0",   # FORBIDDEN: Lag-0 competitor!
]

df["competitor_mean"] = df.groupby("date")["competitor_rate"].transform("mean")
# This creates a lag-0 feature!
```

### Do

```python
# RIGHT: Competitor rates lagged by at least 2 weeks
features = [
    "prudential_rate_t0",       # Own rate at t=0 is OK
    "competitor_rate_t2",       # Proper 2-week lag
    "competitor_weighted_t3",   # Proper 3-week lag
]

# Create lagged competitor mean
df["competitor_mean_t2"] = (
    df.groupby("date")["competitor_rate"]
    .transform("mean")
    .shift(2)  # 2-week lag
)
```

### Gate Detection

```python
from src.validation.leakage_gates import detect_lag0_features

features = ["own_rate_t0", "C_lag0", "competitor_mean_t2"]
result = detect_lag0_features(features)

if result.status == GateStatus.HALT:
    raise ValueError(f"Lag-0 leakage: {result.message}")
# Output: Gate HALT - "C_lag0" detected as lag-0 competitor feature
```

### Business Impact

| If Deployed | Consequence |
|-------------|-------------|
| Coefficients biased | Elasticity estimates unusable for pricing |
| "What-if" scenarios invalid | Cannot predict response to rate changes |
| Regulatory risk | Pricing decisions based on invalid model |

---

## Pitfall #2: Train/Test Temporal Overlap

### The Problem

Time series data requires temporal splits, not random splits. When test data dates overlap with training data, the model learns patterns it shouldn't have access to.

**Why This Happens**:
- Scikit-learn's `train_test_split` defaults to random splitting
- Easy to forget temporal structure exists
- Random splits produce better (but misleading) metrics

### Don't

```python
from sklearn.model_selection import train_test_split

# WRONG: Random split ignores temporal structure
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Test data may include dates BEFORE training data!
```

### Do

```python
# RIGHT: Temporal split respects time ordering
df = df.sort_values("date")
cutoff_idx = int(len(df) * 0.8)

train_df = df.iloc[:cutoff_idx]
test_df = df.iloc[cutoff_idx:]

# Verify no overlap
assert train_df["date"].max() < test_df["date"].min()
```

### Gate Detection

```python
from src.validation.leakage_gates import check_temporal_boundary

result = check_temporal_boundary(
    train_dates=train_df["date"],
    test_dates=test_df["date"]
)

if result.status == GateStatus.HALT:
    raise ValueError(f"Temporal leakage: {result.message}")
```

### Business Impact

| If Deployed | Consequence |
|-------------|-------------|
| Overly optimistic metrics | Model performs worse in production |
| Hidden overfitting | Model memorizes instead of generalizing |
| Deployment failures | Real-world performance disappoints |

---

## Pitfall #3: Scaling on Full Dataset

### The Problem

Fitting a scaler (StandardScaler, MinMaxScaler) on the entire dataset leaks test set statistics into training. The model learns the scale of future data it shouldn't know about.

**Why This Happens**:
- More convenient to scale once at the beginning
- Scikit-learn pipelines make it easy to do correctly
- Easy mistake when manually preprocessing

### Don't

```python
from sklearn.preprocessing import StandardScaler

# WRONG: Scaler sees test data statistics
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit on ALL data including test

# Split AFTER scaling - test statistics already leaked!
X_train, X_test = X_scaled[:split], X_scaled[split:]
```

### Do

```python
from sklearn.preprocessing import StandardScaler

# RIGHT: Fit scaler only on training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X[:split])  # Fit ONLY on train
X_test = scaler.transform(X[split:])       # Transform test with train params

# Or use sklearn Pipeline (recommended)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

# Pipeline automatically fits scaler per CV fold
pipeline.fit(X_train, y_train)
```

### Gate Detection

This is hard to detect automatically. Use the shuffled target test:

```python
from src.validation.leakage_gates import run_shuffled_target_test

result = run_shuffled_target_test(model, X_train, y_train)
# If model works on shuffled targets, scaling leakage is possible
```

### Business Impact

| If Deployed | Consequence |
|-------------|-------------|
| Subtle overfitting | Hard to detect, consistent underperformance |
| Metric inflation | ~5-10% artificial R² improvement |
| Reproducibility issues | Production vs training divergence |

---

## Pitfall #4: Own-Rate Endogeneity

### The Problem

Our own rate at time t may be influenced by factors that also affect sales at time t. This creates a feedback loop that biases coefficient estimates.

**Why This Happens**:
- Rate-setters may observe demand signals before setting rates
- Promotional periods correlate with both rate changes and sales
- Seasonality affects both rate decisions and customer behavior

### Don't

```python
# WRONG: Assume own rate is purely exogenous
features = [
    "prudential_rate_t0",  # May be endogenous!
]

# Naive regression treats rate as exogenous
model = OLS(sales, features)
```

### Do

```python
# RIGHT: Use instrumental variables or control for confounders

# Option 1: Use lagged rate (reduces endogeneity)
features = [
    "prudential_rate_t1",  # Lagged own rate
]

# Option 2: Control for confounders
features = [
    "prudential_rate_t0",
    "treasury_10y_t0",     # Interest rate environment
    "vix_t0",              # Market volatility
    "quarter_indicator",   # Seasonality control
]

# Option 3: Instrumental variables (advanced)
# Use competitor rate changes as instruments
```

### Business Impact

| If Deployed | Consequence |
|-------------|-------------|
| Biased elasticity | May over/underestimate price sensitivity |
| Wrong direction | Coefficient sign may be incorrect |
| Bad pricing decisions | Rate changes have unexpected effects |

---

## Pitfall #5: Product Mix Confounding

### The Problem

Different buffer levels (6Y20B vs 6Y10B) have different customer populations. Pooling products without controls leads to confounded estimates.

**Why This Happens**:
- More data seems better
- Products share some features
- Risk profiles differ by buffer level

### Don't

```python
# WRONG: Pool all buffer levels without control
df_all = pd.concat([df_6y20b, df_6y10b, df_10y20b])

# Train single model on pooled data
model.fit(df_all[features], df_all["sales"])
# Coefficients confound buffer-level effects!
```

### Do

```python
# RIGHT: Option 1 - Stratify by product
models = {}
for product in ["6Y20B", "6Y10B", "10Y20B"]:
    df_product = df[df["product_code"] == product]
    models[product] = Ridge().fit(df_product[features], df_product["sales"])

# RIGHT: Option 2 - Include buffer indicators
df["buffer_20"] = (df["buffer_level"] == 20).astype(int)
features = base_features + ["buffer_20"]

# RIGHT: Option 3 - Interaction terms
df["rate_x_buffer20"] = df["prudential_rate"] * df["buffer_20"]
features = base_features + ["buffer_20", "rate_x_buffer20"]
```

### Gate Detection

Check coefficient signs per product:

```python
def test_coefficient_signs_by_product():
    for product in ["6Y20B", "6Y10B"]:
        result = fit_model(df[df["product"] == product])

        assert result.own_rate_coef > 0, (
            f"{product}: Own rate should be positive"
        )
        assert result.competitor_coef < 0, (
            f"{product}: Competitor rate should be negative"
        )
```

### Business Impact

| If Deployed | Consequence |
|-------------|-------------|
| Simpson's paradox | Aggregate results opposite of subgroup results |
| Wrong product recommendations | Elasticity varies by product |
| Mispricing risk | Products priced using wrong sensitivity |

---

## Summary: Validation Gate Coverage

| Pitfall | Detection Method | Gate Status |
|---------|-----------------|-------------|
| #1 Lag-0 competitors | `detect_lag0_features()` | HALT |
| #2 Temporal overlap | `check_temporal_boundary()` | HALT |
| #3 Scaling leakage | `run_shuffled_target_test()` | HALT if severe |
| #4 Endogeneity | Manual review + coefficient sign check | WARN |
| #5 Product confounding | Stratified coefficient validation | WARN |

---

## Running All Validation Gates

```python
from src.validation.leakage_gates import run_all_gates

report = run_all_gates(
    model=fitted_model,
    X=X_test,
    y=y_test,
    r_squared=model_r2,
    baseline_r_squared=naive_r2,
    feature_names=list(X.columns),
    train_dates=train_df["date"],
    test_dates=test_df["date"],
)

if not report.passed:
    raise ValueError(f"Validation failed:\n{report}")
```

---

## Quick Reference: Feature Naming Conventions

| Pattern | Meaning | Status |
|---------|---------|--------|
| `prudential_rate_t0` | Own rate at t=0 | ALLOWED |
| `competitor_rate_t0` | Competitor rate at t=0 | FORBIDDEN |
| `competitor_rate_t2` | Competitor rate at t-2 | ALLOWED |
| `C_lag0` | Competitor at lag 0 | FORBIDDEN |
| `C_lag_2` | Competitor at lag 2 | ALLOWED |
| `sales_momentum_t5` | Sales lagged 5 weeks | ALLOWED |
| `vix_t0` | Control variable at t=0 | ALLOWED |

---

## Next Steps

- **[Testing Strategy](../development/TESTING_STRATEGY.md)**: Full testing architecture
- **[Validation Evidence](../validation/VALIDATION_EVIDENCE.md)**: Concrete validation results
- **[Leakage Checklist](../../knowledge/practices/LEAKAGE_CHECKLIST.md)**: Pre-deployment gate
- **[Causal Framework](../../knowledge/analysis/CAUSAL_FRAMEWORK.md)**: Identification strategy

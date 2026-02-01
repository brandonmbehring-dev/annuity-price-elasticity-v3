# Episode 04: Product Mix Confounding

**Category**: Target/Outcome Leakage (10 Bug Category #5)
**Discovered**: 2025-11-XX (coefficient analysis)
**Impact**: Pooled models conflated product-specific effects
**Status**: RESOLVED - Product stratification implemented

---

## The Bug

Pooling multiple products (6Y20B, 6Y10B, 10Y20B) in a single model without controlling for product-specific effects, causing **Simpson's Paradox** in coefficient estimates.

The elasticity appeared different (and often wrong-signed) when products with different base sales volumes were combined.

---

## Symptom

**How it manifested:**
- Overall own-rate coefficient was NEGATIVE (should be positive)
- Coefficient sign flipped depending on which products were included
- Model performed well in-sample but predictions were nonsensical

**Red flags in output:**
```
Pooled Model:
  prudential_rate coefficient: -0.023  # WRONG SIGN!

By-Product Analysis:
  6Y20B: +0.085 (correct)
  6Y10B: +0.072 (correct)
  10Y20B: +0.091 (correct)
```

---

## Root Cause Analysis

### 1. Simpson's Paradox

When products have different:
- Base sales volumes (6Y20B >> 6Y10B)
- Rate levels (10Y products have lower rates)
- Seasonal patterns

The aggregate relationship can show the OPPOSITE direction of the within-product relationship.

### 2. Visual Example

```
                    Sales
                      ↑
                      |    ★★★ 6Y20B (high sales, high rates)
                      |    ★★★
                      |  ★★★
                      |
                      |         ◆◆◆ 10Y20B (low sales, low rates)
                      |        ◆◆◆
                      |       ◆◆◆
                      +─────────────────→ Rate

Within each product: Higher rate → Higher sales (positive slope)
Across products: 6Y20B has both higher rates AND higher sales
Overall (naive): Higher rate → Lower sales (negative slope!)
```

### 3. The Math

The pooled coefficient captures TWO effects:
1. **Within-product effect**: Rate changes for same product
2. **Between-product effect**: Differences across product types

If between-product variation dominates, the coefficient reflects product mix, not price elasticity.

---

## The Fix

### Before (Confounded) ❌

```python
# Pooled model across all products
df = load_all_products()  # 6Y20B, 6Y10B, 10Y20B combined

model = Ridge()
model.fit(df[features], df['sales'])
print(f"Rate coefficient: {model.coef_[0]}")  # Often wrong sign!
```

### After (Controlled) ✅

**Option 1: Product Stratification**

```python
# Separate model per product
results = {}
for product in ['6Y20B', '6Y10B', '10Y20B']:
    df_product = df[df['product'] == product]
    model = Ridge()
    model.fit(df_product[features], df_product['sales'])
    results[product] = model.coef_[0]

# Each coefficient has correct sign
```

**Option 2: Fixed Effects**

```python
# Include product indicators
df = pd.get_dummies(df, columns=['product'], drop_first=True)

# Now rate coefficient is WITHIN-product effect
features_with_fe = features + ['product_6Y10B', 'product_10Y20B']
model = Ridge()
model.fit(df[features_with_fe], df['sales'])
```

**Option 3: De-meaning (Within Transformation)**

```python
# Remove product means
df['sales_demeaned'] = df.groupby('product')['sales'].transform(
    lambda x: x - x.mean()
)
df['rate_demeaned'] = df.groupby('product')['prudential_rate'].transform(
    lambda x: x - x.mean()
)

# Estimate on demeaned data
model.fit(df[['rate_demeaned']], df['sales_demeaned'])
# Coefficient now captures within-product variation only
```

---

## Gate Implementation

### Detection Logic

```python
def detect_simpson_paradox(
    df: pd.DataFrame,
    rate_col: str,
    outcome_col: str,
    group_col: str,
) -> bool:
    """Detect if pooled vs stratified coefficients have opposite signs."""
    from scipy.stats import pearsonr

    # Pooled correlation
    pooled_corr, _ = pearsonr(df[rate_col], df[outcome_col])

    # Stratified correlations
    stratified_corrs = []
    for group in df[group_col].unique():
        group_df = df[df[group_col] == group]
        if len(group_df) > 10:
            corr, _ = pearsonr(group_df[rate_col], group_df[outcome_col])
            stratified_corrs.append(corr)

    # Check if signs differ
    if stratified_corrs:
        avg_stratified = np.mean(stratified_corrs)
        if np.sign(pooled_corr) != np.sign(avg_stratified):
            return True  # Simpson's paradox detected

    return False
```

### Test Coverage

```python
@pytest.mark.leakage
def test_no_simpson_paradox():
    """Pooled and stratified coefficients should have same sign."""
    df = load_multi_product_data()

    has_paradox = detect_simpson_paradox(
        df, 'prudential_rate', 'sales', 'product'
    )

    assert not has_paradox, (
        "Simpson's paradox detected! "
        "Pooled coefficient has opposite sign from within-product effects."
    )
```

---

## Verification

### Manual Verification

```python
# Check for paradox in your data
from scipy.stats import pearsonr

# Pooled
pooled_corr = pearsonr(df['rate'], df['sales'])[0]
print(f"Pooled correlation: {pooled_corr:.3f}")

# By product
for product in df['product'].unique():
    subset = df[df['product'] == product]
    corr = pearsonr(subset['rate'], subset['sales'])[0]
    print(f"{product} correlation: {corr:.3f}")

# If signs differ, you have Simpson's paradox
```

### Automated Check

```bash
# Run Simpson's paradox detection
pytest tests/anti_patterns/test_product_mix_confounding.py -v
```

---

## Impact Assessment

| Metric | Pooled Model | Stratified Model | Interpretation |
|--------|-------------|------------------|----------------|
| Own Rate Coef | -0.023 | +0.085 | Correct sign |
| Coefficient SE | 0.012 | 0.008 | More precise |
| Economic Validity | ❌ | ✅ | Matches theory |

**Key insight:** The stratified model gives economically sensible results that match microeconomic theory.

---

## Lessons Learned

1. **Never pool heterogeneous products naively**
   - Check for Simpson's paradox before pooling
   - Use fixed effects or stratification

2. **Correlation across groups ≠ correlation within groups**
   - Aggregate patterns can reverse individual patterns
   - Always verify within-group relationships

3. **Economic theory provides guardrails**
   - Negative own-rate coefficient violates theory
   - Use sign constraints to detect problems

4. **Fixed effects are your friend**
   - Product indicators control for level differences
   - Remaining variation is within-product

---

## Related Documentation

- `knowledge/analysis/CAUSAL_FRAMEWORK.md` - Identification strategy
- `src/products/` - Product-specific methodologies
- `Episode 01: Lag-0 Competitor Rates` - Another coefficient sign issue

---

## Tags

`#confounding` `#simpson-paradox` `#product-mix` `#fixed-effects` `#stratification`

# Feature Engineering Rationale

Domain expert explanations for RILA price elasticity feature engineering decisions.

**Last Updated**: 2026-01-20
**Adapted from**: FIA Price Elasticity project
**Source**: Expert Q&A session during planning + FIA lessons learned

---

## Critical Insight: Cap Rate Is a YIELD, Not a Price

> "Rate action works the opposite of price— offering a higher yield means a better return for the customer. We increase the savings rate the more desirable."

**Why this matters**: Explains the counterintuitive P_coeff > 0 constraint in AIC models.

| Traditional Price Elasticity | RILA Cap Rate Elasticity |
|------------------------------|-------------------------|
| Price = customer cost | Cap rate = customer benefit (yield) |
| Higher price → Lower demand | Higher cap rate → Higher sales |
| Negative coefficient expected | **Positive coefficient expected** |

This insight transfers directly from FIA and applies equally to RILA.

---

## 1. Lag Structure

### Current Implementation
- **Range**: 0-8 (9 lags)
- **Rationale**: FIA found lags >7 never selected by AIC

### Expert Assessment
> "I over did it and we shouldn't make that many lags since we don't use them anywhere."

### Actually Used Lags
| Lag | Variable | Where Used |
|-----|----------|------------|
| 0 | P_lag_0, C_lag_0 | Current rate (output metadata) |
| 1-2 | sales_lag_1, sales_lag_2 | Previous sales (output metadata) |
| 2-3 | C_weighted_mean_t2, C_weighted_mean_t3 | Competitor features |
| 3 | DGS5_lag_3 | Treasury rate control |
| 5-7 | sales_by_contract_date_lag_5/6/7 | Q run rate model features |

---

## 2. Coefficient Constraints (AIC Model Selection)

### P_coeff > 0 (Prudential Rate)
**Meaning**: Higher own cap rate → more sales

**Expert Explanation**: Cap rate is a yield (customer benefit), not a price (customer cost). Higher yield = more attractive product.

### C_coeff < 0 (Competitor Rates)
**Meaning**: Higher competitor rates → lower Prudential relative advantage

**Mechanism**: When competitors offer higher rates, Prudential's offering looks relatively less attractive even if absolute rate unchanged.

### C_diff_coeff < 0 (Rate Momentum)
**Expert Explanation**:
> "If competitors' rates are increasing it should lower all sales all else being equal."

**Mechanism**: Rising rates across the market create a "wait-and-see" effect where customers delay purchase decisions expecting further rate increases.

### DGS5_coeff < 0 (Treasury Rate)
**Expert Assessment**:
> "I don't think it is necessary or done properly."

**Recommendation**: Review necessity of this constraint. May remove in future refactoring.

---

## 3. Holiday Mask

### Current Implementation
- **Excluded**: `day_of_year < 14` OR `day_of_year > 360` (~18 days)
- **Effect**: ~5% of daily data excluded

### Expert Explanation
> "When doing things based on application date (rather than contract issue date) then there is a lot less sales the last two weeks of December— when doing a rolling average that is rolled out a little."

### Investigation Results

**Data Analysis**:

| Period | Days | Mean Sales | Pattern |
|--------|------|------------|---------|
| Late Dec (days 361+) | 16 | Near-zero | Offices closed |
| Early Jan (days 1-13) | 39 | Elevated | Catch-up processing |
| Non-holiday | ~340 | Normal | Baseline |

**Key Finding**: December year-end week has **$0 sales** (complete shutdown).

### Decision: Keep Binary Removal

**Rationale**:
1. Late December truly has $0 sales (offices closed) - not missing data
2. Early January catch-up creates artificial spikes (not representative demand)
3. 5% data loss is acceptable given data quality issues in those periods
4. Model should not try to "learn" from meaningless $0 weeks

---

## 4. Date Selection: application_signed_date

### Expert Explanation
> "Both reasons— but application date has its own problems because the holiday issue— the contract issue date is smoothed out."

### Tradeoff Analysis

| Date Type | Pros | Cons |
|-----------|------|------|
| application_signed_date | Captures customer decision point; closer to when they saw the rate | Holiday distortion; weekend clustering |
| contract_issue_date | Administrative smoothing reduces noise | Processing delays mask true demand signal |

### Current Choice
`application_signed_date` — prioritizes capturing the customer's response to rate information, accepting the holiday mask as necessary compensation.

---

## 5. Competitor Aggregation (RILA-Specific)

### Market-Share Weighted Mean

**Current Implementation**: RILA uses market-share weighted competitor means.

```python
C_weighted_mean = sum(rate_i * market_share_i) / sum(market_share_i)
```

### Why Different from FIA

| Project | Aggregation | Rationale |
|---------|-------------|-----------|
| FIA | Simple top-N mean | FIA market is fragmented |
| **RILA** | **Market-share weighted** | RILA market is concentrated |

**RILA-specific rationale**: The RILA market is dominated by a few large players (Allianz, Lincoln, Athene). Market-share weighting better reflects competitive pressure from these dominant players.

### Excluded Carriers

Certain carriers are excluded from competitor calculations:
- **Allianz**: Dominates market share; would overwhelm weighted mean
- **Trans**: Data quality issues

See `COMPETITIVE_ANALYSIS.md` for detailed exclusion rationale.

### Aggregations Available
- `C_weighted_mean`: Market-share weighted mean (primary)
- `C_mid`: Average of 2nd and 3rd highest competitors
- `top_5`, `top_7`, `top_10`: Simple means (context only)

---

## 6. Data Exclusions

### Channel Structural Breaks

**Pattern**: New distribution channel entries create volume spikes unrelated to rates.

**FIA Example**: J.P. Morgan channel launch in November 2024 created artificial spike.

**RILA Consideration**: Monitor for similar channel events. Current approach:
- Exclude channels with <6 months history, OR
- Add channel indicator variable

### Expert Explanation (from FIA)
> "They started much later so the sales are artificially larger when they come in— need to investigate a better way to deal with it."

### Decision: Channel Exclusion

**Rationale**:
1. Channel growth ≠ rate elasticity response
2. Exclusion prevents false positive rate sensitivity detection
3. After channel matures (12+ months), may revisit

---

## 7. External Data: Treasury Rates

### DGS5 (5-Year Treasury)
- **7-day rolling average** applied before use
- **Constraint**: DGS5_coeff < 0 (under review)

### Expert Assessment
> "I don't think it is necessary or done properly."

The Treasury rate relationship may need fundamental rethinking. Consider:
- Is 5-year the right maturity? (RILA is 6-year term)
- Is the constraint economically justified?
- Should rates be used level or spread form?

---

## 8. RILA-Specific Features

### Buffer Level Indicators

RILA products have variable buffer levels, unlike FIA's fixed 0% floor:

| Feature | Description | Use |
|---------|-------------|-----|
| `buffer_20pct` | Indicator for 20% buffer products | Primary product segment |
| `buffer_10pct` | Indicator for 10% buffer products | Higher risk tolerance segment |

### Why Buffer Matters

Different buffer levels attract different buyer segments:
- **20% buffer**: More conservative, lower caps
- **10% buffer**: Higher risk tolerance, higher caps

Buffer level may **moderate** the price elasticity relationship and should be included as control.

---

## 9. What Didn't Work (FIA Lessons)

### Momentum Features (pos_lag_*)
**What**: Position-based momentum features.
**Result**: Never selected by AIC.
**Expert assessment**: "Doesn't make a big impact."

### Lags > 7
**What**: Extended lag structure (up to lag 14).
**Result**: Never selected by AIC. Lags 8-14 added no predictive value.

### Alternative Competitor Aggregations (FIA context)
**What**: `C_median`, `C_top_3`, `C_top_5` instead of `C_mid`.
**Result**: In FIA, AIC consistently preferred `C_mid`.
**RILA note**: RILA uses market-share weighted means, which is different.

### Forward Stepwise Selection
**What**: Add features one at a time based on p-value.
**Result**: Can miss globally optimal combinations; order-dependent.
**Status**: Replaced with best-subset enumeration.

---

## 10. RILA-Specific Validation Needed

These patterns need RILA-specific testing:

| Item | FIA Pattern | RILA Validation |
|------|-------------|-----------------|
| Elasticity magnitude | Moderate | Confirm similar or lower |
| Lag structure | Lags ≤7 selected | Confirm same pattern |
| Sigmoid response | Smooth curve | Confirm not sharp like MYGA |
| Buffer effect | N/A | Does buffer level moderate elasticity? |
| Market-share weighting | Simple mean | Validate weighted mean is appropriate |

---

## Related Documents

- `knowledge/domain/RILA_ECONOMICS.md` - Product economics and buffer structure
- `knowledge/domain/COMPETITIVE_ANALYSIS.md` - Competitor selection and weighting
- `knowledge/practices/data_leakage_prevention.md` - Temporal leakage guards
- `CLAUDE.md` - Project-level modeling constraints and validation gates

# Lessons Learned: RILA Price Elasticity

Accumulated knowledge from development, including critical traps, FIA transfers, and RILA-specific discoveries.

**Last Updated**: 2026-01-20
**Adapted from**: FIA Price Elasticity LESSONS_LEARNED.md
**Purpose**: Prevent new analysts (and Claude) from repeating mistakes

---

## CRITICAL TRAPS (READ FIRST)

These issues have caused significant confusion in annuity elasticity projects. Read before touching the model.

---

### Trap #1: Cap Rate Sign (+)

**The mistake**: Expecting a negative coefficient on price (like traditional elasticity).

**The reality**: Cap rate is a **YIELD** (customer benefit), not a price (customer cost).

| Traditional Price | RILA Cap Rate |
|-------------------|--------------|
| Customer pays more | Customer earns more |
| Higher price → fewer sales | Higher cap → more sales |
| Negative coefficient | **Positive coefficient** |

**Code enforcement**: Require `p_coeff > 0` in model constraints.

**Expert quote**: "Rate action works the opposite of price—offering a higher yield means a better return for the customer."

→ See: `GLOSSARY.md` (Cap Rate), `CAUSAL_FRAMEWORK.md` (Section 8.2)

---

### Trap #2: Lag-0 Competitor Bias

**The mistake**: Using current competitor rates (`C_lag_0`) as a feature.

**The problem**: Simultaneity bias.

```
C_t (competitor rates) ←─┐
                         ├── Same market conditions
Sales_t ←────────────────┘

If C_t and Sales_t are jointly determined by market conditions,
using C_t to predict Sales_t creates spurious correlation.
```

**The solution**: Use lagged competitor rates (`C_lag_1+`).

**Why P_lag_0 is OK**: We control our own rate—it's the treatment variable. We set it before observing sales.

→ See: `CAUSAL_FRAMEWORK.md` (Section 3.3)

---

### Trap #3: Channel Structural Breaks

**The mistake**: Seeing sales spikes and attributing them to rate sensitivity.

**The reality**: New distribution channel entries create volume spikes unrelated to rates.

| Observation | Wrong Interpretation | Correct Interpretation |
|-------------|---------------------|------------------------|
| Sales spike | Rate change drove sales | New channel added volume |
| 10%+ increase | High elasticity | Distribution expansion |

**The solution**: Exclude channels with <6 months history or flag for investigation.

→ See: `FEATURE_RATIONALE.md` (Section 6)

---

### Trap #4: Ignoring Buffer Level (RILA-Specific)

**The mistake**: Treating all RILA products as homogeneous.

**The reality**: Different buffer levels attract different buyer segments.

| Buffer Level | Buyer Profile | Expected Elasticity |
|--------------|---------------|---------------------|
| 10% | Higher risk tolerance | Potentially more rate-sensitive |
| 20% | Moderate risk tolerance | Baseline |
| 25% | Conservative | Potentially less rate-sensitive |

**The solution**: Include buffer level as control or stratify analysis.

→ See: `CAUSAL_FRAMEWORK.md` (Section 4), `FIXED_DEFERRED_ANNUITY_TAXONOMY.md`

---

### Trap #5: Using Simple Competitor Means (RILA-Specific)

**The mistake**: Using FIA's simple top-N mean for competitor aggregation.

**The reality**: RILA market is more concentrated; market-share weighting is appropriate.

| Aggregation | FIA | RILA |
|-------------|-----|------|
| Top-5 mean | ✓ Used | ✗ Not recommended |
| Top-7 mean | ✓ Primary | ✗ Not recommended |
| **Market-share weighted** | ✗ Not used | **✓ Primary** |

**The solution**: Use market-share weighted competitor means.

→ See: `COMPETITIVE_ANALYSIS.md`, `FEATURE_RATIONALE.md` (Section 5)

---

## Transferred from FIA

These patterns were learned in FIA elasticity work and apply to RILA.

---

### Simpson's Paradox

**Pattern**: Aggregate and disaggregate analyses can give opposite conclusions.

**RILA relevance**: Buffer-level stratification may reveal different patterns than aggregate analysis.

**Guard**: Always consider whether aggregation level could flip conclusions.

---

### LEVEL vs CHANGE

**Pattern**: Customers see absolute rate levels, not changes.

**The mistake**: Using rate changes (Δrate) instead of rate levels.

**Correct approach**: Use absolute cap rate values, not differences.

**Exception**: `C_diff` (rate momentum) is intentionally a change—captures market direction.

---

### Future Leakage

**Pattern**: Accidentally including information from time t+h in features for predicting time t.

**Common sources**:
1. Using full-sample statistics (max, mean) computed over future data
2. Lag features computed incorrectly
3. Rolling averages that include future observations

**Guards**:
- Data maturity threshold (60 days)
- Time-forward CV (never train on future)
- Explicit lag verification in tests

→ See: `knowledge/practices/data_leakage_prevention.md`

---

### Reverse Causality

**Pattern**: Outcome influencing treatment rather than reverse.

**Mitigation**: Contract-issue-date lag creates identification window. Rate-setters observe sales with 19-76 day delay.

→ See: `CAUSAL_FRAMEWORK.md` (Section 3)

---

### Holiday Mask Necessity

**Pattern**: Application-date aggregation requires holiday exclusion.

**Problem**: Late December has $0 sales (offices closed). Early January has catch-up spikes.

**Solution**: Exclude days 1-12 and 360-366 from analysis (~5% of observations).

→ See: `FEATURE_RATIONALE.md` (Section 3)

---

### Weight Decay Bifurcation

**Pattern**: Different decay rates needed for different purposes.

| Purpose | Decay | Rationale |
|---------|-------|-----------|
| Forecast | 0.98 | React quickly to recent patterns |
| Monitoring | 0.99 | Smooth comparison with history |

---

## RILA-Specific Considerations

Knowledge unique to RILA, not directly from FIA.

---

### Buffer-Elasticity Interaction

**Hypothesis**: Elasticity may vary by buffer level.

| Buffer | Expected Pattern |
|--------|------------------|
| 10% | Higher risk tolerance buyers → possibly more rate-sensitive |
| 20% | Baseline buyer segment |
| 25% | Conservative buyers → possibly less rate-sensitive |

**Validation needed**: Test interaction between buffer level and rate coefficient.

---

### Market-Share Weighting

**Why RILA uses weighted means**:
- Top 3 carriers hold ~60% market share (vs ~35% for FIA)
- Allianz alone is ~35%
- Weighting reflects actual competitive pressure

**Allianz exclusion**: Including Allianz in weighted mean would dominate calculation. Better to exclude or treat as separate indicator.

---

### FIA Cross-Elasticity

**Risk**: RILA and FIA may be substitutes.

**Implication**: FIA rate changes may affect RILA sales (and vice versa).

**Current status**: Not controlled for. Future work may add FIA rate as feature.

---

### Regulatory Distinction

**Difference**: RILA is SEC-registered security; FIA is insurance-only.

**Implication**: RILA buyers receive prospectus, may be more informed about risk/return tradeoffs.

**Possible effect**: More sophisticated buyers may have different rate sensitivity.

---

## What Didn't Work (FIA Lessons Applied to RILA)

### Momentum Features (pos_lag_*)
**What**: Position-based momentum features.
**Result**: Never selected by AIC in FIA.
**RILA expectation**: Likely same outcome.

### Lags > 7
**What**: Extended lag structure (up to lag 14).
**Result**: Lags 8-14 never selected by AIC.
**RILA default**: Use `max_lag=8`.

### DGS5 Constraint
**What**: Enforcing `DGS5_coeff < 0`.
**Expert assessment**: "I don't think it is necessary or done properly."
**Status**: Under review for both FIA and RILA.

---

## Validation Gates

Quality checks before accepting model changes.

| Gate | Threshold | Action if Breached |
|------|-----------|-------------------|
| R² | > 0.30 (suspicious) | HALT - investigate leakage |
| Improvement | > 20% over baseline | HALT - investigate leakage |
| Coverage | < 80% test coverage | BLOCK commit |
| Parity | > 1e-6 difference | BLOCK commit |

**Rationale**: Unusually good results suggest data leakage, not genuine improvement.

→ See: `CLAUDE.md` (Validation Gates)

---

## Quick Reference: Do's and Don'ts

### DO

- Use P_lag_0 (own rate) as treatment
- Use C_lag_1+ (lagged competitor rates)
- Expect positive P coefficient
- Apply holiday mask
- Use time-forward CV
- Use market-share weighted competitor means
- Include buffer level as control
- Check out-of-sample performance

### DON'T

- Use C_lag_0 (simultaneity bias)
- Expect negative elasticity (cap rate is yield)
- Include untested channels (structural break)
- Trust in-sample R² alone
- Use lags > 7 (not selected by AIC)
- Use simple top-N means (RILA market is concentrated)
- Ignore buffer level variation
- Copy MYGA magnitude expectations

---

## Related Documents

- `knowledge/analysis/CAUSAL_FRAMEWORK.md` - Identification strategy
- `knowledge/analysis/FEATURE_RATIONALE.md` - Feature engineering decisions
- `knowledge/domain/GLOSSARY.md` - Term definitions
- `knowledge/integration/CROSS_PRODUCT_COMPARISON.md` - Cross-project patterns
- `knowledge/practices/data_leakage_prevention.md` - Leakage prevention

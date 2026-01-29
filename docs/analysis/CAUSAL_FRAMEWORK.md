# Causal Framework for RILA Price Elasticity

Formal causal framework documenting identification strategy, confounders, and assumptions.

**Last Updated**: 2026-01-26
**Adapted from**: FIA Price Elasticity project
**Status**: Active (RILA-specific validation ongoing)

---

## 1. Causal Question

> **What is the causal effect of Prudential's cap rate on RILA sales, controlling for competitive positioning and buffer level?**

### Formal Estimand

```
E[Sales_t | do(P_t = p), C_{t-1}, DGS5_t, Season_t, Buffer] - E[Sales_t | do(P_t = p'), C_{t-1}, DGS5_t, Season_t, Buffer]
```

Where:
- `P_t`: Prudential cap rate at time t (treatment)
- `Sales_t`: Weekly RILA sales (outcome, application-date-based)
- `C_{t-1}`: Competitor rates at t-1 (control, lagged to avoid simultaneity)
- `DGS5_t`: 5-year Treasury rate (confounder)
- `Season_t`: Quarterly seasonality indicators
- `Buffer`: Buffer level indicator (20%, 10%, etc.)

### Time Series Context

- **Frequency**: Weekly data
- **Aggregation**: Application-signed-date (not contract-issue-date)
- **Lag structure**: Own rate at t=0; competitor rates at t-1+
- **Outcome**: Aggregate RILA premium (not firm/channel-level)

---

## 2. Causal DAG

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                          Market Volatility (VIX)                            │
│                                   │                                         │
│                    ┌──────────────┴──────────────┐                          │
│                    ▼                             ▼                          │
│              Option Costs                   Customer Risk                   │
│              (hedging)                      Preferences                     │
│                    │                             │                          │
│                    ▼                             │                          │
│              ┌──────────┐                        │                          │
│              │   VNB    │     Treasury Rates     │                          │
│              │ Targets  │◄────── (DGS5) ─────────┤                          │
│              └────┬─────┘           │            │                          │
│                   │                 │            │                          │
│                   ▼                 ▼            ▼                          │
│           ┌──────────────┐    ┌──────────┐    ┌─────────────┐               │
│           │ Pru Cap Rate │───►│Competitor│    │  Customer   │               │
│           │   (P_t)      │    │ Rates    │    │ Alternatives│               │
│           │ [TREATMENT]  │    │ (C_t)    │    │             │               │
│           └──────┬───────┘    └────┬─────┘    └──────┬──────┘               │
│                  │                 │                 │                      │
│                  │                 │    Past Sales   │                      │
│                  │                 │    (lagged)     │                      │
│                  │                 │        │        │                      │
│                  └─────────────────┼────────┼────────┘                      │
│                                    ▼        │                               │
│       Buffer Level ────────► ┌───────────┐  │                               │
│       (10%, 20%)             │   SALES   │◄─┘                               │
│                              │  (weekly) │                                  │
│                              │ [OUTCOME] │                                  │
│                              └─────┬─────┘                                  │
│                                    ▲                                        │
│                                    │                                        │
│                               Seasonality                                   │
│                              (Q1, Q2, Q3)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Causal Pathways

| Pathway | Direction | Interpretation |
|---------|-----------|----------------|
| P → Sales | Direct | Treatment effect (what we estimate) |
| C → Sales | Direct | Competitive pressure (controlled) |
| DGS5 → P → Sales | Indirect | Treasury affects rate-setting |
| DGS5 → Sales | Direct | Treasury affects customer alternatives |
| Volatility → Option Costs → P | Indirect | Hedging costs affect rate-setting |
| Volatility → Customer Demand | Direct | Risk preferences affect RILA appeal |
| Buffer → Sales | Direct | Buffer level moderates risk tolerance |

### RILA-Specific Pathway: Buffer Level

Unlike FIA (0% floor), RILA has variable buffer levels:
- **Higher buffer (20%)**: More protection → Lower caps → Different buyer segment
- **Lower buffer (10%)**: Less protection → Higher caps → Higher risk tolerance buyers

Buffer level may **moderate** the price elasticity relationship.

---

## 3. Identification Strategy

### 3.1 Treatment: P_lag_0 (Own Cap Rate)

**Why P_lag_0 is valid as treatment:**

1. **We control it**: Prudential sets the cap rate. It's the intervention variable we can manipulate.

2. **Temporal separation**: Rate is set before observing that week's application-date sales.

3. **Delayed feedback**: Even if Prudential adjusts rates based on past sales, there's significant lag:
   - Sales observed via contract-issue-date (not application-date)
   - Contract issue date lags application date by 1-2+ weeks
   - This creates an identification window

### 3.2 Rate-Setting Mechanism

Prudential's cap rate is determined by:

| Input | Timing | Effect on Identification |
|-------|--------|-------------------------|
| **VNB targets** | Strategic planning | Exogenous to weekly sales |
| **Option costs** | Real-time market | Affects rate; confounder |
| **Earned rates** | Current portfolio | Exogenous to new sales |
| **Past sales** | Contract-issue-date (lagged) | Delayed feedback ✓ |
| **Competitor rates** | Rate-setting day | Observed, controlled |
| **Buffer level** | Product design | Stratification variable |

**Key insight**: The lag between application-date (customer decision) and contract-issue-date (when Prudential observes sales) creates the identification window.

### 3.3 Why C_lag_0 is NOT Valid

Competitor rates at t=0 are problematic:

- Competitors respond to the same market conditions driving sales
- C_t and Sales_t are jointly determined
- Creates simultaneity bias if used as regressor

**Solution**: Use C_lag_1+ (lagged competitor rates) to break simultaneity.

---

## 4. Confounders

### 4.1 Currently Controlled

| Confounder | How Controlled | Rationale |
|------------|----------------|-----------|
| **Treasury rates (DGS5)** | DGS5_lag_k features | Affects option costs → rate; affects customer alternatives → sales |
| **Competitor rates** | C_weighted_mean features | Affects competitive positioning → sales; observed by rate-setters |
| **Seasonality** | Q1, Q2, Q3 dummies | Q4 is baseline; captures tax-season and year-end effects |
| **Buffer level** | Stratification/indicator | Different buffer products have different buyer bases |

### 4.2 Open Investigation Items

| Potential Confounder | Status | Concern |
|---------------------|--------|---------|
| **Market volatility (VIX)** | PARTIALLY CONTROLLED | Affects option costs → rate; affects RILA appeal (downside protection value) |
| **Option costs directly** | NOT CONTROLLED | Could be better than DGS5 as proxy; see `annuity-pricing/` repo |
| **Equity returns** | NOT CONTROLLED | Affects customer alternatives |
| **FIA cross-elasticity** | NOT CONTROLLED | RILA and FIA may be substitutes |

**RILA-specific consideration**: VIX may be MORE important for RILA than FIA because RILA's value proposition is explicitly tied to downside protection.

---

## 5. Functional Form Assumptions

### 5.1 Logit/Sigmoid Transform

Sales are transformed via logit scaling:

```python
sales_scaled = 0.95 * sales / max(sales)
sales_logit = logit(sales_scaled)
```

**Assumptions embedded**:

| Assumption | Value | Status |
|------------|-------|--------|
| Saturation parameter | 0.95 | **Arbitrary** - should test empirically |
| Maximum sales | Global max | Uses full history |
| Response shape | Sigmoid | Empirically observed in FIA data; validate for RILA |

### 5.2 Expected Response Shape for RILA

Based on FIA experience and RILA product characteristics:
- **Expected**: Smooth sigmoid (gradual response like FIA)
- **Rationale**: RILA is complex product; complexity dampens rate sensitivity
- **Difference from MYGA**: MYGA has sharp/nearly binary response; RILA should be smoother

**Validation needed**: Confirm sigmoid response shape with RILA sales data.

---

## 6. Key Differences from FIA Elasticity

| Aspect | FIA | RILA | Implication |
|--------|-----|------|-------------|
| **Downside protection** | 0% floor (guaranteed) | Buffer (partial) | RILA buyers have higher risk tolerance |
| **Competitor weighting** | Simple top-N mean | **Market-share weighted** | Different aggregation method |
| **Buffer variation** | None | 10%, 15%, 20%, 25% | Additional control variable needed |
| **Product complexity** | Moderate | Higher | May dampen rate sensitivity |
| **Buyer segment** | Conservative accumulators | Moderate risk tolerance | Different elasticity expected |

### Expected Elasticity Differences

| Product | Expected Magnitude | Rationale |
|---------|-------------------|-----------|
| FIA | Moderate | Smooth response, product complexity |
| RILA | Similar or lower | Higher complexity, buyer self-selection |
| MYGA | Extreme (10x swings) | Simple product, direct rate comparison |

---

## 7. Threats to Validity

### 7.1 Internal Validity

| Threat | Severity | Mitigation |
|--------|----------|------------|
| **Reverse causality** | MEDIUM | Contract-issue-date lag creates identification window |
| **Omitted confounders** | HIGH | DGS5 controls some; volatility/options need investigation |
| **Buffer confounding** | MEDIUM | Stratify analysis by buffer level |
| **Measurement error** | LOW | Sales and rates from reliable sources (TDE, WINK) |

### 7.2 External Validity

| Threat | Severity | Notes |
|--------|----------|-------|
| **Rate regime changes** | MEDIUM | Model estimated in specific rate environment |
| **Product changes** | LOW | FlexGuard relatively stable |
| **FIA cannibalization** | MEDIUM | RILA may substitute for FIA; cross-product effects |

---

## 8. Estimation Approach

### 8.1 Current Approach: OLS with Controls

```
sales_logit_t = β₀ + β₁·P_lag_0 + β₂·C_weighted_lag_k + β₃·DGS5_lag_k + γ·Season + δ·Buffer + ε
```

**Why OLS is defensible**:
- Treatment (P_lag_0) is set before outcome observed (temporal separation)
- Contract-issue-date lag prevents immediate feedback
- Key confounders (DGS5, competitors, seasonality, buffer) are controlled

### 8.2 Coefficient Constraints

| Coefficient | Required Sign | Economic Rationale |
|-------------|---------------|-------------------|
| P_lag_0 | > 0 | Higher cap rate (yield) → more sales |
| C_weighted_mean | < 0 | Higher competitor rates → lower relative advantage |
| C_diff | < 0 | Rising market → wait-and-see effect |
| DGS5 | Under review | Expert questioned necessity |

---

## 9. Open Questions

### High Priority

1. **Volatility/option costs**: VIX may be especially important for RILA (downside protection value).
2. **Buffer interaction**: Does elasticity differ by buffer level (10% vs 20%)?
3. **FIA cross-elasticity**: Are RILA and FIA substitutes?

### Medium Priority

4. **Market-share weighting validation**: Confirm weighted mean is appropriate for RILA.
5. **Functional form testing**: Validate sigmoid response with RILA data.

### Low Priority

6. **Channel effects**: Are there channel-specific elasticity patterns?
7. **Term effects**: Does 6-year vs 10-year term affect sensitivity?

---

## 10. Limitations & Bias Risks

### 10.1 Known Uncontrolled Confounders

Despite the identification strategy, the following confounders are **not fully controlled** and may bias elasticity estimates:

| Uncontrolled Confounder | Expected Bias Direction | Rationale |
|------------------------|------------------------|-----------|
| **Option costs (direct)** | Overestimate magnitude | Option costs affect rate-setting AND customer value perception. Using DGS5 as proxy may miss option-specific variation. |
| **Equity returns** | Direction unclear | Strong equity markets affect customer alternatives AND RILA value proposition (downside protection less valuable in bull markets). |
| **FIA cross-elasticity** | Underestimate for RILA | RILA and FIA may be substitutes; ignoring this misses cross-product demand shifts. |

### 10.2 Endogeneity Concern: Own-Rate

The identification strategy claims P_lag_0 is exogenous because "rate is set before observing sales." However:

1. **Common shock problem**: Both P_t and Sales_t respond to the same market conditions (Treasury moves, VIX spikes, competitor actions)
2. **This creates omitted variable bias**, not resolved by temporal ordering alone
3. **VIX is partially controlled** via `market_volatility` features, but option costs are not

**Interpretation caveat**: Own-rate elasticity estimates should be interpreted as **upper bounds** on the true causal effect, since positive omitted variable bias is likely.

### 10.3 What Would Strengthen Identification

The following controls would strengthen causal identification but are not currently implemented:

| Future Control | Source | Benefit |
|---------------|--------|---------|
| **Option costs** | `annuity-pricing/` repo | Direct hedging cost input to rate-setting |
| **VNB targets** | Strategic planning data | Exogenous pricing pressure |
| **Earned rates** | ALM data | Portfolio return constraints |

### 10.4 Sensitivity Analysis Recommendation

Given the uncontrolled confounders, production deployment should include:

1. **Bounds analysis**: Compute elasticity under different assumptions about omitted variable bias
2. **Placebo tests**: Verify no effect from variables that shouldn't matter (e.g., competitor rates from unrelated product lines)
3. **Cross-validation with FIA**: Compare RILA and FIA elasticity for consistency

---

## 11. References

### Internal Documentation

- `knowledge/analysis/FEATURE_RATIONALE.md` - Feature engineering decisions
- `knowledge/domain/RILA_ECONOMICS.md` - Cap rate as yield
- `knowledge/integration/LESSONS_LEARNED.md` - Critical traps

### Cross-Project References

- `fia-price-elasticity/knowledge/analysis/CAUSAL_FRAMEWORK.md` - FIA causal framework
- `myga-elasticity-v2/docs/CAUSAL_METHODOLOGY.md` - MYGA causal framework (different!)

### Academic References

- Pearl, J. (2009). Causality: Models, Reasoning, and Inference
- Angrist & Pischke (2009). Mostly Harmless Econometrics
- Chernozhukov et al. (2018). Double/Debiased Machine Learning

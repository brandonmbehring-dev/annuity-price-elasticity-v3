# Fixed Deferred Annuity Product Taxonomy

**Purpose**: Comprehensive understanding of ALL fixed deferred annuity types to explain why different projects require different analytical approaches.

**Last Updated**: 2026-01-20
**Source**: Synthesized from domain expertise, LIMRA data, product documentation

---

## The Three Product Types

| Aspect | MYGA | FIA | RILA |
|--------|------|-----|------|
| **Full Name** | Multi-Year Guaranteed Annuity | Fixed Indexed Annuity | Registered Index-Linked Annuity |
| **Market Size (2024)** | ~$153B | ~$127B | ~$65B |
| **Regulatory Status** | Insurance product (state) | Insurance product (state) | **SEC-registered security** |
| **Principal Protection** | 100% guaranteed | 100% guaranteed (0% floor) | **Partial** (buffer absorbs first X%) |
| **Upside Potential** | Fixed rate only | Capped/limited index return | **Higher caps** than FIA |
| **Complexity** | Simple (like CD) | Moderate (crediting formulas) | High (buffer/floor + crediting) |
| **Target Buyer** | Risk-averse, rate shoppers | Conservative accumulators | **Moderate risk tolerance** |

---

## Product Mechanics Deep-Dive

### MYGA: The Simplest Product

```
Principal: $100,000
Rate: 5.00% (fixed for term)
Term: 5 years
Outcome: $127,628 guaranteed (no variability)
```

**Key Characteristics:**
- **Fixed rate declared upfront** (like a CD)
- **No market exposure** - returns independent of index
- **Sharp elasticity response** - customers compare rates directly, binary ON/OFF behavior
- **Extreme rate sensitivity** - small spread differences drive large share swings

**Why MYGA elasticity is DIFFERENT:**
1. Direct rate comparison (apples-to-apples)
2. Nearly binary purchasing behavior (threshold effects)
3. Firm-level analysis required (competitor sets vary by marketplace)
4. Rate = price paid BY customer (negative coefficient expected)

---

### FIA: Indexed with Full Downside Protection

```
Principal: $100,000
Index Return: +15% (S&P 500)
Participation Rate: 80%
Cap: 8%

Calculation:
  min(15% × 80%, 8%) = min(12%, 8%) = 8% credited

Index Return: -10%
Calculation:
  max(0%, -10%) = 0% credited (NO LOSS)
```

**Key Characteristics:**
- **0% floor** - cannot lose principal to market declines (insurance guarantee)
- **Multiple crediting methods**: cap, participation, spread, trigger
- **Smoother elasticity response** - gradual not binary
- **S&P 500 dominant** (~70% of index options)

**Why FIA elasticity is DIFFERENT from MYGA:**
1. Rate = YIELD (customer benefit) → **positive coefficient**
2. Smoother sigmoid response (not step function)
3. Aggregate analysis sufficient (not firm-level)
4. Multiple rate metrics matter (cap, participation, spread)

**FIA Crediting Methods:**

| Method | Formula | WINK Field |
|--------|---------|------------|
| **Cap** | min(index_return, cap) | `capRate` |
| **Participation** | index_return × participation_rate | `participationRate` |
| **Spread** | index_return - spread | `spreadRate` |
| **Trigger** | fixed_rate if index > 0 | `performanceTriggeredRate` |

---

### RILA: Indexed with Partial Downside Protection

```
Principal: $100,000
Buffer: 20% (insurer absorbs first 20% loss)
Cap: 15%

Index Return: +25%
Calculation:
  min(25%, 15%) = 15% credited

Index Return: -15%
Calculation:
  Insurer absorbs all 15% → Customer loss: 0%

Index Return: -30%
Calculation:
  Insurer absorbs first 20% → Customer loss: -10%
```

**Key Characteristics:**
- **Buffer absorbs FIRST X% of loss** (not floor limiting maximum loss)
- **Higher caps than FIA** (trade downside protection for upside)
- **SEC-registered** (sold with prospectus, not just insurance regulation)
- **Participation rates often >100%** (enhanced upside)

**Why RILA elasticity DIFFERS from FIA:**
1. Same yield economics (positive coefficient)
2. Buffer level affects rate sensitivity (20% buffer vs 10% buffer)
3. Buyer risk tolerance is HIGHER than FIA buyers
4. Product complexity may dampen rate response
5. Market-share weighted competitor rates (not simple top-N mean)

---

## Buffer vs Floor: Critical Distinction

```
BUFFER (RILA):                    FLOOR (FIA alternative):
─────────────────────────────     ─────────────────────────────
Index: -15%, Buffer: 10%          Index: -15%, Floor: -10%
Insurer absorbs: 10%              Customer loss: -10% (capped)
Customer loss: -5%

Index: -8%, Buffer: 10%           Index: -8%, Floor: -10%
Insurer absorbs: 8%               Customer loss: -8% (not capped)
Customer loss: 0%

Key: Buffer protects against      Key: Floor protects against
     MODERATE losses                   CATASTROPHIC losses
```

---

## Option Budget: Why Caps Vary

**Core Concept**: Insurance companies buy options to hedge index exposure. The "option budget" determines how much upside they can offer.

```
Option Budget Formula (simplified):
  Option_Budget = GA_Yield - Profit_Margin - Expenses

Where:
  GA_Yield = General account investment yield (~4-5%)
  Profit_Margin = Target return (~1-2%)
  Expenses = Administrative costs (~0.5%)

Result: ~2-3% annual option budget
```

**Higher option budget → Higher caps**

| Factor | Effect on Option Budget | Effect on Caps |
|--------|------------------------|----------------|
| Rising interest rates | ↑ GA yield | ↑ Higher caps possible |
| High VIX (volatility) | ↑ Option costs | ↓ Lower caps |
| Strong GA performance | ↑ Investment income | ↑ Higher caps |
| Competitive pressure | Squeeze margins | ↑ Aggressive caps |

---

## Why Each Product Requires Different Modeling

| Aspect | MYGA | FIA | RILA |
|--------|------|-----|------|
| **Analysis Level** | Firm/marketplace | Aggregate | Aggregate |
| **Elasticity Shape** | Sharp step-function | Smooth sigmoid | Smooth sigmoid |
| **Rate Variable** | Single declared rate | Cap rate primary | Cap rate + buffer |
| **Coefficient Sign** | Negative (price) | Positive (yield) | Positive (yield) |
| **Competitor Measure** | Firm-specific spreads | Top-N means | Market-share weighted |
| **Why Different** | Extreme sensitivity | Complex crediting | Buffer tradeoff |

---

## What Transfers Between Projects

### [DONE] TRANSFERS (Use Directly):
- WINK data extraction patterns
- TDE sales cleanup pipeline (with product filter changes)
- Seasonal patterns (Q4 high, Q2 low)
- Treasury rate effects (DGS5 as confounder)
- Lagged feature structure (avoid lag-0 competitors)
- Holiday mask approach (Dec/Jan exclusion)
- Mathematical equivalence validation framework

### [WARN] TRANSFERS WITH MODIFICATION:
- Cap rate = yield insight (applies to FIA and RILA, not MYGA)
- Competitor aggregation (FIA uses top-N mean; RILA uses market-share weighted)
- Feature names (different naming conventions per repo)
- Product filtering criteria (surrender period, buffer level)

### [ERROR] DOES NOT TRANSFER:
- MYGA firm-level analysis (FIA/RILA use aggregate)
- MYGA elasticity magnitude (FIA/RILA have smoother response)
- MYGA threshold detection (not applicable to FIA/RILA)
- Specific coefficient values (each product has different sensitivity)

---

## Market Context (LIMRA 2024)

**Sales Rankings:**
1. **MYGA**: $153B (simple, rate-sensitive market)
2. **FIA**: $127B (balanced risk/return)
3. **RILA**: $65B (growing, higher risk tolerance)

**Growth Trends:**
- RILA fastest growing segment (>20% YoY)
- FIA stable, mature market
- MYGA cyclical (rate-sensitive)

**Buyer Demographics:**

| Product | Typical Buyer | Risk Profile |
|---------|---------------|--------------|
| MYGA | Pre-retirees, conservative | Very low (CD-like) |
| FIA | Near-retirees, moderate | Low (principal protected) |
| RILA | Younger accumulators, engaged | **Moderate** (partial exposure) |

---

## Product Selection Decision Tree

```
Customer wants:

Principal protection guaranteed?
├── YES → How much return potential?
│   ├── Fixed rate acceptable → MYGA
│   └── Want market-linked upside → FIA
│
└── NO (willing to accept some loss) → RILA
    └── How much protection?
        ├── 25% buffer (conservative RILA)
        ├── 20% buffer (standard)
        ├── 15% buffer (moderate)
        └── 10% buffer (aggressive)
```

---

## Regulatory Distinction

| Aspect | MYGA/FIA | RILA |
|--------|----------|------|
| **Regulator** | State insurance commissioners | SEC + state |
| **Registration** | Insurance product only | SEC-registered security |
| **Sales document** | Brochure, illustration | **Prospectus required** |
| **Suitability** | Insurance standards | **FINRA standards** |
| **Who can sell** | Insurance agents | Registered representatives |

**Implication**: RILA buyers receive prospectus disclosure, may be more informed about risk/return tradeoffs.

---

## Related Documents

- `knowledge/domain/RILA_MECHANICS_DEEP.md` - RILA payoff details
- `knowledge/domain/CREDITING_METHODS.md` - Cap/participation mechanics
- `knowledge/integration/CROSS_PRODUCT_COMPARISON.md` - What transfers
- `knowledge/domain/COMPETITIVE_ANALYSIS.md` - Rate positioning

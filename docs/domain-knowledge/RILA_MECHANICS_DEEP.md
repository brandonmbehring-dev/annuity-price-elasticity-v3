# RILA (Registered Index-Linked Annuity) Mechanics - Deep Dive

**Tier**: L1 (Quick Reference) + L2 (Deep Dive)
**Domain**: Product Mechanics
**Last Updated**: 2026-01-20
**Adapted from**: annuity-pricing project

---

## Product Overview

RILA = "Buffered annuity" or "Buffer annuity" or "Structured annuity"

**Key features**:
- **Partial downside protection** (buffer or floor)
- **Limited upside participation** (cap, spread, or participation rate)
- **Index-linked returns** (S&P 500, Russell 2000, etc.)
- **No explicit guarantee of principal** (unlike FIA's 0% floor)
- **SEC-registered security** (unlike FIA which is insurance-only)

---

## Protection Mechanisms

### Buffer (Primary RILA Mechanism)

**Definition**: Insurer absorbs first X% of losses.

```
Buffer = 20%

Index return = -15%
→ Insurer absorbs first 15%
→ Client loss = 0%

Index return = -25%
→ Insurer absorbs first 20%
→ Client loss = -5%

Index return = +10%
→ No loss to absorb
→ Client gains (subject to cap)
```

**Common buffer levels**: 10%, 15%, 20%, 25%

### Floor (Alternative Mechanism)

**Definition**: Maximum loss client can experience.

```
Floor = -10%

Index return = -15%
→ Client loss capped at -10%

Index return = -8%
→ Client loss = -8% (not capped)
```

### Buffer vs Floor: Critical Distinction

| Aspect | Buffer | Floor |
|--------|--------|-------|
| **Protection type** | First X% of loss | Maximum loss |
| **Who benefits when** | Moderate declines | Catastrophic declines |
| **Example (-15% index)** | 20% buffer → 0% loss | -10% floor → -10% loss |
| **Example (-8% index)** | 20% buffer → 0% loss | -10% floor → -8% loss |

**Key insight**: Buffer protects against **moderate** losses; floor protects against **catastrophic** losses.

```
BUFFER (20%):                        FLOOR (-10%):
─────────────────────────────        ─────────────────────────────
Index: -15%                          Index: -15%
Insurer absorbs: 15%                 Customer loss: -10% (capped)
Customer loss: 0%

Index: -25%                          Index: -25%
Insurer absorbs: 20%                 Customer loss: -10% (capped)
Customer loss: -5%

Index: -8%                           Index: -8%
Insurer absorbs: 8%                  Customer loss: -8% (not capped)
Customer loss: 0%
```

---

## Upside Mechanisms

### Cap

**Definition**: Maximum return client can earn.

```
Cap = 15%

Index return = +20%
→ Client return = +15% (capped)

Index return = +10%
→ Client return = +10% (not capped)
```

### Participation Rate

**Definition**: Percentage of index return credited.

```
Participation rate = 100% (no reduction)
Participation rate = 150% (enhanced upside)

Index return = +10%, Participation = 150%
→ Client return = +15% (before cap)
```

**RILA note**: Unlike FIA, RILA participation rates often **exceed 100%** due to the trade-off with buffer protection.

### Spread (Fee)

**Definition**: Fixed percentage subtracted from return.

```
Spread = 2%

Index return = +15%
→ Client return = +13% (before cap)

Index return = +5%
→ Client return = +3%
```

---

## Payoff Formulas

### Buffer + Cap Strategy (Most Common RILA)

```python
def buffer_cap_payoff(index_return: float, buffer: float, cap: float) -> float:
    """
    Calculate RILA return with buffer protection and cap.

    Args:
        index_return: Index performance (e.g., -0.15 = -15%)
        buffer: Buffer level (e.g., 0.20 = 20%)
        cap: Cap rate (e.g., 0.15 = 15%)

    Returns:
        Client return after buffer and cap applied
    """
    if index_return >= 0:
        # Positive returns: apply cap only
        return min(index_return, cap)
    elif index_return >= -buffer:
        # Moderate negative: buffer absorbs all
        return 0.0
    else:
        # Large negative: buffer absorbs first X%, rest passes through
        return index_return + buffer
```

### Floor + Cap Strategy

```python
def floor_cap_payoff(index_return: float, floor: float, cap: float) -> float:
    """
    Calculate return with floor protection and cap.

    Args:
        index_return: Index performance
        floor: Floor level (e.g., -0.10 = -10% maximum loss)
        cap: Cap rate

    Returns:
        Client return after floor and cap applied
    """
    return max(floor, min(cap, index_return))
```

### Buffer + Participation + Cap

```python
def buffer_participation_cap_payoff(
    index_return: float,
    buffer: float,
    participation: float,
    cap: float
) -> float:
    """
    Calculate RILA return with buffer, participation, and cap.
    """
    if index_return >= 0:
        # Apply participation then cap to positive returns
        participated = index_return * participation
        return min(participated, cap)
    elif index_return >= -buffer:
        # Buffer absorbs moderate losses
        return 0.0
    else:
        # Pass through excess losses
        return index_return + buffer
```

---

## Option Decomposition

A RILA can be replicated with vanilla options:

### Buffer + Cap

| Component | Option | Strike |
|-----------|--------|--------|
| Base | Long Zero-Coupon Bond | - |
| Downside | Short put | 100% - buffer |
| Upside | Long call spread | 100% to cap |

**Intuition**: The insurer is selling downside protection (short put) and buying capped upside (call spread).

### Floor + Cap

| Component | Option | Strike |
|-----------|--------|--------|
| Base | Long Zero-Coupon Bond | - |
| Downside | Long put | 100% + floor |
| Upside | Short call | cap |

---

## Pricing Considerations

### Option Budget Framework

**Core Concept**: Insurance companies buy options to hedge index exposure. The "option budget" determines how much upside they can offer.

```
Option Budget (simplified):
  Option_Budget = GA_Yield - Profit_Margin - Expenses

Where:
  GA_Yield = General account investment yield (~4-5%)
  Profit_Margin = Target return (~1-2%)
  Expenses = Administrative costs (~0.5%)

Result: ~2-3% annual option budget
```

### Key Relationships

| Factor | Effect on Caps |
|--------|----------------|
| **Higher buffer** | Lower caps (more protection = less upside) |
| **Longer term** | Higher caps (time value favors insurer) |
| **Higher volatility** | Lower caps (options more expensive) |
| **Higher interest rates** | Higher caps (higher option budget) |

### Trade-off Example

```
Same option budget can provide:

Option A: 10% buffer + 18% cap
Option B: 20% buffer + 15% cap
Option C: 25% buffer + 12% cap

Customer chooses based on risk tolerance.
```

---

## Market Data Requirements

| Data | Source | Frequency |
|------|--------|-----------|
| S&P 500 options | CBOE, Bloomberg | Daily |
| VIX index | CBOE | Real-time |
| Treasury rates | FRED | Daily |
| Dividend yield | Estimated | Quarterly |

---

## WINK bufferModifier Field

WINK data includes a `bufferModifier` field indicating buffer type:

| Value | Meaning |
|-------|---------|
| `standard` | Standard buffer (insurer absorbs first X%) |
| `enhanced` | Enhanced buffer (additional protection features) |
| `step` | Step-rate buffer (varies by performance tier) |

**Filter recommendation**: Use `bufferModifier = 'standard'` for clean comparisons.

---

## FlexGuard Product Specifics

### Product Codes

| Code | Term | Buffer | Cap Range | Primary Index |
|------|------|--------|-----------|---------------|
| **6Y20B** | 6 years | 20% | 12-18% | S&P 500 |
| **6Y10B** | 6 years | 10% | 15-22% | S&P 500 |
| **10Y20B** | 10 years | 20% | 15-25% | S&P 500 |

### Rate-Setting Frequency

FlexGuard rates are typically set:
- **Weekly** for new business
- **Locked** at contract inception for duration

---

## Validation Notes

**Gap**: No open-source RILA pricing library exists.

**Cross-validation strategy**:
- Decompose to options, validate components against financepy
- Compare to published RILA rates from insurers
- Sensitivity analysis vs. market conditions

---

## Related Documents

- `knowledge/domain/RILA_ECONOMICS.md` - High-level economics
- `knowledge/domain/CREDITING_METHODS.md` - Cap/participation mechanics
- `knowledge/domain/WINK_SCHEMA.md` - Data dictionary
- `knowledge/domain/COMPETITIVE_ANALYSIS.md` - Rate positioning

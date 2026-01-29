# Crediting Methods Quick Reference

**Tier**: [T1] Academic + [T2] WINK Empirical
**Last Updated**: 2026-01-20
**Adapted from**: annuity-pricing project

---

## Overview [T1]

Annuity returns are linked to index performance but subject to limits. The insurer does NOT invest your money in the index; they track it and credit interest based on performance.

**Key Principle for RILA**: Unlike FIA's 0% floor, RILA allows negative returns (absorbed by buffer, then passed to customer).

---

## Method 1: Cap Rate [T1]

**Definition**: Maximum return client can earn, regardless of index performance.

### Formula

```python
def cap_crediting(index_return: float, cap: float, floor: float = None) -> float:
    """
    Cap rate crediting with optional floor. [T1]

    For FIA: floor = 0.0 (guaranteed)
    For RILA: floor = -buffer or can be negative

    Returns min of index return and cap, floored at floor.
    """
    if floor is not None:
        return min(max(index_return, floor), cap)
    return min(index_return, cap)  # RILA without floor
```

### Examples (15% Cap, 20% Buffer RILA)

| Index Return | Buffer Absorbed | Credited |
|--------------|-----------------|----------|
| +20% | 0% | +15% (capped) |
| +15% | 0% | +15% |
| +10% | 0% | +10% |
| -10% | 10% | 0% (buffer absorbs) |
| -25% | 20% | -5% (excess loss) |

### WINK Stats [T2]

| Metric | RILA Value | FIA Value |
|--------|------------|-----------|
| Median cap | 12-15% | 5-6% |
| High cap | 20%+ | 10%+ |
| Field | `capRate` | `capRate` |

**RILA caps are higher** because customers accept downside risk (buffer vs guaranteed floor).

---

## Method 2: Participation Rate [T1]

**Definition**: Percentage of index gain credited to your account.

### Formula

```python
def participation_crediting(
    index_return: float,
    participation: float,
    buffer: float = 0.0,
    cap: float = None
) -> float:
    """
    Participation rate crediting for RILA. [T1]

    Returns participation × index return, with buffer protection.
    """
    if index_return >= 0:
        participated = index_return * participation
        if cap:
            return min(participated, cap)
        return participated
    elif index_return >= -buffer:
        return 0.0  # Buffer absorbs
    else:
        return index_return + buffer  # Pass through excess loss
```

### Examples (150% Participation, 20% Buffer)

| Index Return | Participated | Final |
|--------------|--------------|-------|
| +10% | +15% (10% × 150%) | +15% |
| +5% | +7.5% (5% × 150%) | +7.5% |
| -10% | Buffer absorbs | 0% |
| -25% | Buffer absorbs 20% | -5% |

### WINK Stats [T2]

| Metric | RILA Value | FIA Value |
|--------|------------|-----------|
| Median | 80-120% | ~50% |
| High | 200%+ | 100%+ |
| Field | `participationRate` | `participationRate` |

**Note**: RILA participation rates often **exceed 100%** due to the buffer trade-off.

---

## Method 3: Spread (Margin) [T1]

**Definition**: Fee subtracted from index return before crediting.

### Formula

```python
def spread_crediting(
    index_return: float,
    spread: float,
    buffer: float = 0.0
) -> float:
    """
    Spread/margin crediting for RILA. [T1]

    Returns index return minus spread, with buffer protection.
    """
    if index_return >= 0:
        return max(index_return - spread, 0)
    elif index_return >= -buffer:
        return 0.0  # Buffer absorbs
    else:
        return index_return + buffer
```

### Examples (2% Spread, 20% Buffer)

| Index Return | After Spread | Final |
|--------------|--------------|-------|
| +10% | +8% (10% - 2%) | +8% |
| +5% | +3% (5% - 2%) | +3% |
| +1% | 0% (1% - 2% floored) | 0% |
| -10% | Buffer absorbs | 0% |

### WINK Stats [T2]

| Metric | Value |
|--------|-------|
| Typical | 1-3% |
| Field | `spreadRate` |

---

## Method 4: Performance Triggered [T1]

**Definition**: Fixed rate credited if index has ANY positive return.

### Formula

```python
def trigger_crediting(
    index_return: float,
    trigger_rate: float,
    buffer: float = 0.0
) -> float:
    """
    Performance triggered crediting. [T1]

    Returns fixed rate if index > 0, else 0 or buffer-adjusted loss.
    """
    if index_return > 0:
        return trigger_rate
    elif index_return >= -buffer:
        return 0.0
    else:
        return index_return + buffer
```

### Examples (8% Trigger Rate, 20% Buffer)

| Index Return | Credited |
|--------------|----------|
| +20% | +8% (trigger) |
| +0.1% | +8% (trigger) |
| 0% | 0% |
| -10% | 0% (buffer) |
| -25% | -5% (excess) |

### WINK Stats [T2]

| Metric | Value |
|--------|-------|
| Typical | 6-10% |
| Field | `performanceTriggeredRate` |
| Fill rate | Low (~5%) |

---

## Crediting Frequency [T2]

| Method | Description | WINK Field |
|--------|-------------|------------|
| **Annual PTP** | Compare year-start to year-end | `indexCreditingFrequency` |
| **Monthly PTP** | Sum of monthly returns (often with monthly cap) | |
| **Monthly Average** | Average of 12 month-end values | |
| **Term End Point** | Compare start to end of multi-year term | |

**RILA focus**: Most RILA products use **Annual PTP** or **Term End Point**.

---

## Index Used [T2]

Most common indices in WINK:

| Index | Notes |
|-------|-------|
| **S&P 500** | Most common (~70% of products) |
| Russell 2000 | Small cap exposure |
| NASDAQ-100 | Tech-heavy |
| MSCI EAFE | International developed |
| Proprietary/Vol-Control | Vendor data required |

**Crediting Basis**: Usually price return (excludes dividends).

---

## Combined Methods [T1]

Some products combine methods:

### Participation + Cap (Common for RILA)

```python
def participation_cap(
    index_return: float,
    participation: float,
    cap: float,
    buffer: float
) -> float:
    """Participation applied, then capped, with buffer protection."""
    if index_return >= 0:
        participated = index_return * participation
        return min(participated, cap)
    elif index_return >= -buffer:
        return 0.0
    else:
        return index_return + buffer
```

### Spread + Cap

```python
def spread_cap(
    index_return: float,
    spread: float,
    cap: float,
    buffer: float
) -> float:
    """Spread deducted, then capped, with buffer protection."""
    if index_return >= 0:
        after_spread = max(index_return - spread, 0)
        return min(after_spread, cap)
    elif index_return >= -buffer:
        return 0.0
    else:
        return index_return + buffer
```

---

## WINK Data Fields Summary [T2]

| Field | Method | RILA Fill Rate |
|-------|--------|----------------|
| `capRate` | Cap | High (~80%) |
| `participationRate` | Participation | High (~70%) |
| `spreadRate` | Spread | Moderate (~30%) |
| `performanceTriggeredRate` | Trigger | Low (~5%) |
| `indexUsed` | Index | High (~95%) |
| `indexingMethod` | Method type | High (~90%) |
| `indexCreditingFrequency` | Frequency | High (~90%) |
| `bufferLevel` | Buffer protection | High (~95%) |
| `bufferModifier` | Buffer type | Moderate (~60%) |

---

## Data Quality Notes [T2]

| Field | Issue | Fix |
|-------|-------|-----|
| `capRate` | max = 9999.99 | Clip to ≤ 0.30 (30%) |
| `performanceTriggeredRate` | max = 999 | Clip to ≤ 0.20 |
| `spreadRate` | max = 99.0 | Clip to ≤ 0.10 |
| `participationRate` | max = 999 | Clip to ≤ 3.00 (300%) |
| `bufferLevel` | Missing values | Require non-null |

---

## RILA vs FIA Crediting Comparison

| Aspect | FIA | RILA |
|--------|-----|------|
| **Floor** | 0% guaranteed | Buffer (partial protection) |
| **Typical Cap** | 5-8% | 12-18% |
| **Participation** | 50-100% | 80-200% |
| **Downside risk** | None | Beyond buffer |
| **Buyer profile** | Conservative | Moderate risk tolerance |

---

## Anti-Pattern Tests

```python
def test_buffer_protection():
    """RILA buffer absorbs moderate losses."""
    # Within buffer
    assert buffer_cap_payoff(-0.10, buffer=0.20, cap=0.15) == 0.0
    assert buffer_cap_payoff(-0.20, buffer=0.20, cap=0.15) == 0.0

    # Beyond buffer
    assert buffer_cap_payoff(-0.25, buffer=0.20, cap=0.15) == -0.05

def test_cap_limits_upside():
    """Cap method limits upside."""
    assert buffer_cap_payoff(0.20, buffer=0.20, cap=0.15) == 0.15
    assert buffer_cap_payoff(0.10, buffer=0.20, cap=0.15) == 0.10

def test_participation_scales():
    """Participation scales return."""
    # 150% participation on 10% return = 15%
    result = participation_crediting(0.10, participation=1.50, buffer=0.20)
    assert result == 0.15
```

---

## References

- Pacific Life: Understanding Annuity Crediting Methods
- WINK, Inc. AnnuitySpecs documentation
- Hardy (2003) "Investment Guarantees"
- `knowledge/domain/RILA_MECHANICS_DEEP.md` - Full RILA payoff formulas

# Episode 09: Macroeconomic Data Lookahead

**Category**: External Data Leakage (10 Bug Category #10)
**Discovered**: 2025-12-XX (data pipeline audit)
**Impact**: Revised macro data used instead of real-time vintages
**Status**: RESOLVED - Real-time vintage data sourced

---

## The Bug

Using REVISED macroeconomic data (GDP, unemployment, CPI) that wasn't available at the original prediction time, rather than the real-time vintage that was actually known.

---

## Symptom

- Macro features had unrealistic predictive power
- Model performed better on historical data than recent data
- Coefficient on GDP growth was suspiciously large and precise

---

## Root Cause

Economic agencies revise data:
- Initial GDP estimate: +2.1% (released Jan 30)
- First revision: +2.4% (released Feb 28)
- Final revision: +2.7% (released Mar 28)

If you're predicting for January and use the +2.7% figure:
- You're using information from 2 months in the future!

```python
# WRONG: Uses latest revision
df['gdp_growth'] = fred.get_series('GDP')  # Always returns latest revision

# At Jan 2024, this returns the FINAL revised value
# But at Jan 2024, only the initial estimate was known
```

---

## The Fix

```python
# CORRECT: Use real-time vintage data
from fredapi import Fred

def get_realtime_vintage(series_id: str, as_of_date: pd.Timestamp) -> float:
    """Get the value that was known at as_of_date."""
    # ALFRED (ArchivaL FRED) provides vintage data
    vintage = fred.get_series_as_of_date(
        series_id,
        observation_date=as_of_date,
        realtime_start=as_of_date,
        realtime_end=as_of_date
    )
    return vintage.iloc[-1] if len(vintage) > 0 else np.nan

# Apply row-wise
df['gdp_growth_realtime'] = df['date'].apply(
    lambda d: get_realtime_vintage('GDP', d)
)
```

**Alternative: Use lagged final values**
```python
# Use Q-2 GDP (definitely finalized by now)
df['gdp_growth_lag2q'] = df['gdp_final'].shift(2)
```

---

## Gate Implementation

```python
MACRO_SERIES_REVISED = ['GDP', 'UNRATE', 'CPI', 'PCE', 'PAYEMS']

def validate_macro_vintages(df: pd.DataFrame, macro_cols: List[str]) -> List[str]:
    """Check if macro data might be using revised values."""
    suspicious = []

    for col in macro_cols:
        # Check if values change over time (revisions would be constant)
        recent_std = df[df['date'] > '2024-01-01'][col].std()
        early_std = df[df['date'] < '2023-01-01'][col].std()

        # If recent data is much more volatile, might be using real-time
        # If identical volatility, might be using revised (less uncertainty)
        if abs(recent_std - early_std) < 0.001:
            suspicious.append(col)

    return suspicious
```

---

## Lessons Learned

1. **Economic data is revised, often multiple times**
   - Initial → First revision → Final
   - Can take 3+ months to finalize

2. **ALFRED provides vintage data**
   - Federal Reserve's ArchivaL FRED database
   - Shows what was known at each point in time

3. **When in doubt, use longer lags**
   - Q-2 GDP is definitely finalized
   - Current quarter GDP is always uncertain

4. **This applies to many data sources**
   - Government statistics (GDP, employment, inflation)
   - Corporate earnings (restated earnings)
   - Survey data (sampling revisions)

---

## Tags

`#macro` `#vintage` `#revisions` `#real-time` `#gdp`

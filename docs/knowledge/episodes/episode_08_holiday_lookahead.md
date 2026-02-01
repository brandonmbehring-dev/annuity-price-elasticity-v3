# Episode 08: Holiday/Calendar Lookahead

**Category**: Calendar Feature Leakage (10 Bug Category #9)
**Discovered**: 2025-12-XX (seasonal analysis)
**Impact**: Holiday flags computed with future calendar knowledge
**Status**: RESOLVED - Static calendar used

---

## The Bug

Creating holiday features or seasonal indicators that implicitly encode future calendar information, such as "days until next holiday" or "holiday week indicator" computed from the full calendar.

---

## Symptom

- Model suspiciously accurate around holidays
- "Days until" features worked too well on validation data
- Holiday effect magnitude unrealistically large

---

## Root Cause

```python
# WRONG: Uses future calendar knowledge
df['days_to_christmas'] = (pd.Timestamp('2025-12-25') - df['date']).dt.days

# At time t=June 2024, this feature "knows" Christmas 2025 will happen
# This is not leakage per se, but can create issues with validation
```

More subtle issue:
```python
# "Is holiday week" flag applied to entire dataset
# But training data shouldn't "know" which future weeks are holidays
```

---

## The Fix

```python
# CORRECT: Use fixed, pre-defined holiday calendar
from pandas.tseries.holiday import USFederalHolidayCalendar

cal = USFederalHolidayCalendar()
holidays = cal.holidays(start='2020-01-01', end='2030-12-31')

# Backward-looking features only
df['days_since_last_holiday'] = df['date'].apply(
    lambda d: (d - holidays[holidays <= d].max()).days
    if any(holidays <= d) else np.nan
)

# Don't use forward-looking:
# df['days_to_next_holiday']  # AVOID
```

---

## Gate Implementation

```python
FORBIDDEN_CALENDAR_PATTERNS = [
    r'days_to_',       # Forward-looking
    r'until_',         # Forward-looking
    r'next_holiday',   # Future knowledge
    r'upcoming_',      # Forward-looking
]

def detect_calendar_lookahead(feature_names: List[str]) -> List[str]:
    """Detect forward-looking calendar features."""
    forbidden = []
    for feature in feature_names:
        for pattern in FORBIDDEN_CALENDAR_PATTERNS:
            if re.search(pattern, feature, re.IGNORECASE):
                forbidden.append(feature)
    return forbidden
```

---

## Lessons Learned

1. **Calendar is deterministic but still problematic**
   - Everyone knows Christmas is Dec 25, but forward features can inflate performance
2. **Use backward-looking calendar features**
   - "Days since last holiday" is safe
   - "Days to next holiday" might inflate validation metrics
3. **Seasonal dummies are generally safe**
   - Month, quarter, day-of-week are known at prediction time
4. **Be cautious with fiscal calendars**
   - Corporate calendars may change and aren't universally known

---

## Tags

`#calendar` `#holidays` `#seasonal` `#forward-looking`

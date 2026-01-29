# FIA Feature Mapping

**Created**: 2026-01-24
**Status**: Active

Mapping between FIA and RILA feature naming conventions, and identification of safe vs. forbidden features for causal identification.

---

## Feature Naming Conventions

FIA and RILA use different naming patterns for rate features:

| Feature Type | RILA Pattern | FIA Pattern |
|--------------|--------------|-------------|
| Own rate (lagged) | `prudential_rate_t2` | `pru_rate_lag_2` |
| Competitor mean (lagged) | `competitor_mid_t2` | `comp_mean_lag_2` |
| Competitor (current) | `competitor_mid_current` | `comp_mean_lag_0` |
| Competitor (forbidden) | `competitor_*_t0` | `*_lag_0` |

### Pattern Explanation

- **RILA**: Uses `_tN` suffix where N is the lag in days (e.g., `_t2` = 2-day lag)
- **FIA**: Uses `_lag_N` suffix where N is the lag in days (e.g., `_lag_2` = 2-day lag)

---

## Safe Features (Causal Identification)

Features with **lag >= 2** are safe for causal identification:

### Own Rate Features (Positive Coefficient Expected)
```
pru_rate_lag_2
pru_rate_lag_3
pru_rate_lag_4
pru_rate_lag_5
...
pru_rate_lag_14
```

### Competitor Features (Negative Coefficient Expected)
```
comp_mean_lag_2
comp_mean_lag_3
comp_mean_lag_4
comp_mean_lag_5
...
comp_mean_lag_14
```

### Top-N Aggregates (Negative Coefficient Expected)
```
top_5
top_7
top_10
```

---

## Forbidden Features (Violate Causal ID)

**CRITICAL**: These features must NEVER be used in inference models.

### Lag-0 Competitor Features
```
comp_mean_lag_0    # FORBIDDEN - simultaneous rate
comp_mean_lag_1    # PROBLEMATIC - decision lag < 2 days
```

### Reason for Exclusion

1. **Simultaneity Bias**: Competitor rates at t=0 are determined simultaneously with own sales
2. **Reverse Causality**: Cannot distinguish cause from effect
3. **Causal Identification**: Requires temporal separation between treatment and outcome

### Detection Pattern

The validation catches these patterns:
- `comp_mean_lag_0` (FIA pattern)
- `comp_*_lag_0` (FIA variant)
- `competitor_*_t0` (RILA pattern)
- `competitor_*_current` (RILA variant)
- `c_*_lag_0` (Abbreviated pattern)

---

## FIA Fixture Data Summary

**Source**: `tests/fixtures/fia/test_features_full.parquet`

| Metric | Value |
|--------|-------|
| Rows | 1,136 |
| Columns | 102 |
| Date Range | 2022-11-10 to 2025-12-19 |
| Own Rate Lags | 15 (lag_0 through lag_14) |
| Competitor Lags | 15 (lag_0 through lag_14) |
| Top-N Aggregates | 3 (top_5, top_7, top_10) |

### Column Categories

1. **Sales Metrics**: count, premium, count_smooth, premium_smooth
2. **Own Rate**: Pru, pru_rate_lag_* (15 lags)
3. **Competitor Rates**: Individual companies + comp_mean_lag_* (15 lags)
4. **Macro Indicators**: DGS5, CPILFESL, DGS5_lag_* (15 lags)
5. **Top-N Aggregates**: top_5, top_7, top_10

---

## Economic Constraints

### Expected Coefficient Signs

| Feature Pattern | Expected Sign | Rationale |
|-----------------|---------------|-----------|
| `own_rate`, `pru_rate_*` | **Positive** | Higher rates attract customers |
| `competitor_*`, `comp_*` | **Negative** | Higher competitor rates divert sales |
| `top_*` | **Negative** | Best alternatives divert sales |

### Validation Rules

1. **OWN_RATE_POSITIVE**: Own rate coefficient must be positive
2. **COMPETITOR_NEGATIVE**: Competitor coefficients must be negative
3. **TOP_N_NEGATIVE**: Top-N aggregate coefficients must be negative
4. **NO_LAG_ZERO_COMPETITOR**: Lag-0 competitor features are forbidden

---

## Integration Notes

### Interface Usage
```python
from src.notebooks.interface import create_interface

interface = create_interface(
    "FIA5YR",
    environment="fixture",
    adapter_kwargs={"fixtures_dir": Path("tests/fixtures/fia")}
)

# Safe features for inference
safe_features = [
    "pru_rate_lag_2",
    "comp_mean_lag_2",
    "comp_mean_lag_3",
]

# Will raise ValueError for lag-0 competitors
interface._validate_methodology_compliance(data, features=safe_features)
```

### Methodology
- FIA uses `TopNAggregation` strategy (5 competitors by default)
- No weights required (unlike RILA's `WeightedAggregation`)
- No buffer level (FIA has floors, not buffers)

---

## References

- `src/products/fia_methodology.py` - FIA constraint rules
- `src/notebooks/interface.py` - Lag-0 detection (`_is_competitor_lag_zero`)
- `tests/test_fia_inference_e2e.py` - E2E validation tests
- `tests/fixtures/fia/README.md` - Fixture documentation

# Wiring Verification Prompts for Subagents

Verification prompts to ensure proper wiring of stub implementations to real code.

---

## Feature Selection Wiring (Phase 3)

### Verification Checklist

Verify that `UnifiedNotebookInterface.run_feature_selection()`:

1. **Actually calls** `production_feature_selection()` (not returning placeholder)
   ```python
   # BAD (placeholder):
   return {"selected_features": [], ...}

   # GOOD (wired):
   results = production_feature_selection(data=data, ...)
   return results
   ```

2. **Passes correct parameters** from config
   - `target_column` from `self._get_target_column()` or config
   - `candidate_features` from product configuration
   - `max_features` from config (default 3)

3. **Returns** `FeatureSelectionResults` dataclass (not dict)
   - Must be importable: `from src.features.selection_types import FeatureSelectionResults`
   - Attribute access: `results.selected_features` not `results["selected_features"]`

### Test Command
```python
from src.notebooks import create_interface
i = create_interface("6Y20B", environment="fixture")
df = i.load_data()
results = i.run_feature_selection(df)

# Verify real results (not placeholder)
assert results.selected_features != []  # Non-empty list
assert hasattr(results, 'selected_features')  # Dataclass attribute
```

---

## Inference Wiring (Phase 4)

### Verification Checklist

Verify that `UnifiedNotebookInterface.run_inference()`:

1. **Actually calls** `center_baseline()`, `rate_adjustments()`, `confidence_interval()`
   ```python
   # BAD (placeholder):
   return {"elasticity_point": 0.0, ...}

   # GOOD (wired):
   baseline_predictions, model = center_baseline(...)
   scenarios = rate_adjustments(...)
   ci = confidence_interval(...)
   return InferenceResults(...)
   ```

2. **Uses config-driven column names** (not hardcoded "Prudential")
   ```python
   # BAD:
   X_test[feature_name] = df_rates["Prudential"].iloc[-1]

   # GOOD:
   X_test[feature_name] = df_rates[self._config.own_rate_column].iloc[-1]
   ```

3. **Returns real elasticity values** (not 0.0)

### Test Command
```python
from src.notebooks import create_interface
i = create_interface("6Y20B", environment="fixture")
df = i.load_data()
results = i.run_inference(df)

# Verify real results (not placeholder)
assert results["elasticity_point"] != 0.0  # Non-zero elasticity
assert len(results["coefficients"]) > 0  # Has coefficients
```

---

## Forecasting Wiring (Phase 6)

### Verification Checklist

Verify that `UnifiedNotebookInterface.run_forecasting()`:

1. **Method exists** on the interface
   ```python
   assert hasattr(interface, 'run_forecasting')
   ```

2. **Actually calls** `run_forecasting_pipeline()`
   ```python
   from src.models.forecasting_orchestrator import run_forecasting_pipeline
   return run_forecasting_pipeline(data=data, config=config, ...)
   ```

3. **Returns** `ForecastingResults` dataclass
   - Has `predictions: pd.DataFrame`
   - Has `cv_scores: pd.DataFrame`
   - Has `benchmark_results: pd.DataFrame`

4. **Matches baseline outputs** at 1e-12 precision

### Test Command
```python
from src.notebooks import create_interface
i = create_interface("6Y20B", environment="fixture")
df = i.load_data()
results = i.run_forecasting(df)

# Verify real results
assert hasattr(results, 'predictions')
assert len(results.predictions) > 0  # Non-empty predictions
```

---

## Data Preparation Wiring (Phase 5)

### Verification Checklist

Verify that `_prepare_analysis_data()`:

1. **Aligns dates** between sales and rates DataFrames

2. **Computes competitor aggregates** using aggregation strategy
   ```python
   aggregated = self._aggregation.aggregate(rates_df, ...)
   ```

3. **Adds temporal features** (lags)

4. **Validates no leakage** (no lag-0 competitors)

### Test Command
```python
from src.notebooks import create_interface
i = create_interface("6Y20B", environment="fixture")
df = i.load_data()

# Verify proper preparation
assert 'competitor_mid_t2' in df.columns  # Lagged competitor feature
assert 'competitor_mid_t0' not in df.columns  # No lag-0 (leakage)
```

---

## Mathematical Equivalence Verification

### Both Layers Must Pass

**Layer 1: Interface → Direct Call**
```python
# Results from interface should match direct calls
from src.features.selection.notebook_interface import production_feature_selection

interface_results = interface.run_feature_selection(df)
direct_results = production_feature_selection(df, target_column=..., ...)

assert interface_results.selected_features == direct_results.selected_features
```

**Layer 2: Results → Baseline**
```python
# Results should match stored baselines at 1e-12
python scripts/equivalence_guard.py --baseline tests/baselines/pre_refactoring/
```

### Precision Requirement

All numeric comparisons must be within `1e-12` tolerance:
```python
import numpy as np
np.allclose(baseline_value, current_value, atol=1e-12)
```

---

## Quick Verification Script

```bash
#!/bin/bash
# Run all verification checks

echo "=== Feature Selection Wiring ==="
python -c "
from src.notebooks import create_interface
i = create_interface('6Y20B', environment='fixture')
df = i.load_data()
r = i.run_feature_selection(df)
print(f'Selected features: {r.selected_features}')
assert r.selected_features != [], 'FAIL: Still placeholder!'
print('PASS')
"

echo "=== Inference Wiring ==="
python -c "
from src.notebooks import create_interface
i = create_interface('6Y20B', environment='fixture')
df = i.load_data()
r = i.run_inference(df)
print(f'Elasticity: {r[\"elasticity_point\"]}')
assert r['elasticity_point'] != 0.0, 'FAIL: Still placeholder!'
print('PASS')
"

echo "=== Forecasting Wiring ==="
python -c "
from src.notebooks import create_interface
i = create_interface('6Y20B', environment='fixture')
df = i.load_data()
r = i.run_forecasting(df)
print(f'Predictions: {len(r.predictions)} rows')
print('PASS')
"

echo "=== Equivalence Guard ==="
python scripts/equivalence_guard.py --baseline tests/baselines/pre_refactoring/
```

---

## Common Issues and Fixes

### Issue: "module has no attribute 'selected_features'"
**Cause**: Still returning dict instead of dataclass
**Fix**: Import and return `FeatureSelectionResults` dataclass

### Issue: Elasticity is exactly 0.0
**Cause**: Placeholder return not replaced
**Fix**: Wire to `center_baseline()` and `rate_adjustments()`

### Issue: Missing 'run_forecasting' method
**Cause**: Method not added to interface
**Fix**: Add method in Phase 6

### Issue: Hardcoded "Prudential" still present
**Cause**: Genericity fix incomplete
**Fix**: Replace with `config.own_rate_column` (Phase 5.3)

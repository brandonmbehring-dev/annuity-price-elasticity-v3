# V2 Interface Wiring Migration Guide

**Date**: 2026-01-24
**Scope**: UnifiedNotebookInterface method implementations
**Breaking Changes**: Yes (type access patterns)

---

## Summary

This migration completes the wiring of `UnifiedNotebookInterface` stub methods to real implementations. The interface now provides a fully functional API for multi-product price elasticity analysis.

---

## What Changed

### 1. Feature Selection (`run_feature_selection()`)

**Before (Stub)**:
```python
def run_feature_selection(self, data, config) -> Dict:
    # Placeholder return
    return {
        "selected_features": [],
        "feature_importance": {},
        ...
    }
```

**After (Wired)**:
```python
def run_feature_selection(self, data, config) -> FeatureSelectionResults:
    results = production_feature_selection(
        data=data,
        target=target_column,
        features=candidate_features,
        max_features=max_features,
    )
    return results  # Returns dataclass, not dict
```

### 2. Inference (`run_inference()`)

**Before (Stub)**:
```python
def run_inference(self, data, config, features) -> Dict:
    return {
        "coefficients": {},
        "elasticity_point": 0.0,
        ...
    }
```

**After (Wired)**:
```python
def run_inference(self, data, config, features) -> InferenceResults:
    baseline_predictions, trained_model = center_baseline(
        sales_df=data,
        rates_df=data,
        features=features,
        ...
    )
    return {
        "coefficients": self._extract_model_coefficients(trained_model, features),
        "elasticity_point": elasticity_point,
        ...
    }
```

### 3. Forecasting (`run_forecasting()`) - NEW

**Added**:
```python
def run_forecasting(self, data, config) -> ForecastingResults:
    """Run time series forecasting pipeline."""
    pipeline_output = run_forecasting_pipeline(
        df=data,
        forecasting_config=forecasting_stage_config,
        ...
    )
    return ForecastingResults.from_pipeline_output(pipeline_output)
```

---

## Breaking Changes

### Type Access Patterns

**Before (Dict Access)**:
```python
# Feature selection returned dict
results = interface.run_feature_selection(df)
features = results["selected_features"]  # Dict access
importance = results["feature_importance"]
```

**After (Dataclass Access)**:
```python
# Feature selection returns FeatureSelectionResults dataclass
results = interface.run_feature_selection(df)
features = results.selected_features  # Attribute access
importance = results.feature_importance

# If dict access needed, use to_dict()
results_dict = results.to_dict()
```

### New Return Types

| Method | Old Return | New Return |
|--------|-----------|------------|
| `run_feature_selection()` | `Dict[str, Any]` | `FeatureSelectionResults` |
| `run_inference()` | `Dict[str, Any]` | `InferenceResults` (TypedDict) |
| `run_forecasting()` | N/A (new) | `ForecastingResults` |

---

## Migration Steps

### Step 1: Update Feature Selection Access

```python
# OLD
results = interface.run_feature_selection(df)
if results["selected_features"]:  # Dict access
    features = results["selected_features"]

# NEW
results = interface.run_feature_selection(df)
if results.selected_features:  # Attribute access
    features = results.selected_features
```

### Step 2: Update Inference Access

```python
# OLD (still works - TypedDict)
results = interface.run_inference(df)
elasticity = results["elasticity_point"]
coefficients = results["coefficients"]

# NEW (same - InferenceResults is TypedDict)
results = interface.run_inference(df)
elasticity = results["elasticity_point"]
coefficients = results["coefficients"]
```

### Step 3: Use New Forecasting Method

```python
# NEW
results = interface.run_forecasting(df)
print(f"MAPE Improvement: {results.mape_improvement:.1f}%")
print(f"Model Outperforms: {results.model_outperforms}")
print(results.summary())
```

---

## Backward Compatibility

### Feature Selection

The `FeatureSelectionResults` dataclass provides a `to_dict()` method for backward compatibility:

```python
results = interface.run_feature_selection(df)

# New style (recommended)
features = results.selected_features

# Legacy style (still works)
results_dict = results.to_dict()
features = results_dict["selected_features"]
```

### Inference

`InferenceResults` remains a TypedDict, so dict access continues to work.

### Forecasting

This is a new method with no backward compatibility concerns.

---

## Type Imports

### Before
```python
from src.core.types import FeatureSelectionResults  # TypedDict - DELETED
```

### After
```python
from src.features.selection_types import FeatureSelectionResults  # Dataclass
from src.models.forecasting_types import ForecastingResults  # New
```

---

## Verification

Run the following to verify the migration:

```python
from src.notebooks import create_interface

interface = create_interface("6Y20B", environment="fixture")

# 1. Feature selection returns dataclass
results = interface.run_feature_selection(df)
assert hasattr(results, 'selected_features')  # Attribute access
assert hasattr(results, 'to_dict')  # Dict conversion available

# 2. Inference returns real values
inf_results = interface.run_inference(df)
assert inf_results["coefficients"]  # Non-empty

# 3. Forecasting works
fc_results = interface.run_forecasting(df)
assert fc_results.n_forecasts > 0
```

---

## Files Changed

| File | Change |
|------|--------|
| `src/notebooks/interface.py` | Wired all methods, added `run_forecasting()` |
| `src/features/selection_types.py` | Added `to_dict()`, properties |
| `src/models/forecasting_types.py` | NEW - `ForecastingResults` dataclass |
| `src/core/types.py` | Removed `FeatureSelectionResults` TypedDict |
| `src/core/protocols.py` | Updated import source |
| `src/core/__init__.py` | Updated import source |

---

## Questions?

See `TECHNICAL_DEBT.md` for resolution details on:
- TD-09: UnifiedNotebookInterface Stubs
- TD-10: Hardcoded "Prudential" Column
- TD-11: MYGA Silent Failure

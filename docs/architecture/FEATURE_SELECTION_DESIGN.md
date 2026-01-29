# Feature Selection Architecture Guide

**Last Updated**: 2026-01-25
**Module Location**: `src/features/selection/`
**Total Scope**: ~18,000 lines across 56 files in 7 subdirectories

---

## 1. Purpose: Why Feature Selection is Complex

Feature selection for annuity price elasticity models requires:

1. **Multiple evaluation criteria**: AIC-based model comparison, bootstrap stability, economic constraint validation
2. **Statistical rigor**: Multiple testing corrections (Bonferroni, FDR), temporal validation
3. **Business constraints**: Positive own-rate coefficients, negative competitor coefficients, no lag-0 competitors
4. **Comprehensive reporting**: Dashboards, visualizations, export capabilities for stakeholder review

This complexity necessitates a modular architecture with clear separation of concerns.

---

## 2. Directory Map

```
src/features/selection/
├── __init__.py           # Public API exports (use this as entry point)
├── notebook_interface.py # Primary interface for notebooks (orchestrates interface/)
├── pipeline_orchestrator.py # Core pipeline coordination
│
├── engines/              # Core computational engines (atomic functions)
│   ├── aic_engine.py        # AIC-based model comparison
│   ├── bootstrap_engine.py  # Bootstrap stability analysis
│   ├── constraints_engine.py # Economic constraint validation
│   └── ridge_cv_engine.py   # Cross-validated Ridge alternative
│
├── enhancements/         # Advanced statistical extensions
│   ├── block_bootstrap_engine.py     # Time-series aware bootstrap
│   ├── out_of_sample_evaluation.py   # OOS validation
│   ├── temporal_validation_engine.py # Walk-forward validation
│   ├── multiple_testing/             # Multiple testing corrections
│   │   ├── bonferroni_engine.py
│   │   ├── fdr_engine.py
│   │   ├── multiple_testing_correction.py
│   │   ├── multiple_testing_types.py
│   │   └── search_space_reduction.py
│   └── statistical_constraints/      # Statistical validators
│       ├── constraint_analyzers.py
│       ├── constraint_types.py
│       ├── statistical_constraints_engine.py
│       └── statistical_validators.py
│
├── interface/            # Notebook interface components (split from notebook_interface.py)
│   ├── interface_config.py            # Configuration builders
│   ├── interface_dashboard.py         # Stability dashboards
│   ├── interface_dashboard_business.py
│   ├── interface_dashboard_dvc.py
│   ├── interface_dashboard_scoring.py
│   ├── interface_dashboard_validation.py
│   ├── interface_dashboard_viz.py
│   ├── interface_display.py           # HTML/display formatting
│   ├── interface_environment.py       # Import/setup utilities
│   ├── interface_execution.py         # Pipeline execution
│   ├── interface_export.py            # Export to MLflow/DVC
│   └── interface_validation.py        # Input validation
│
├── stability/            # Stability analysis components
│   ├── bootstrap_stability_analysis.py
│   ├── bootstrap_win_rate_analysis.py
│   ├── information_ratio_analysis.py
│   ├── stability_analysis.py
│   ├── stability_ir.py
│   ├── stability_visualizations.py
│   └── stability_win_rate.py
│
├── comparison/           # Model comparison utilities
│   ├── comparative_analysis.py
│   ├── comparison_business.py
│   └── comparison_metrics.py
│
├── visualization/        # Visualization and dual validation
│   ├── bootstrap_visualization_analysis.py
│   ├── bootstrap_visualization_detailed.py
│   └── dual_validation.py
│
└── support/              # Helper utilities
    ├── configuration_management.py
    ├── data_preprocessing.py
    ├── environment_setup.py
    ├── notebook_helpers.py
    ├── regression_diagnostics.py
    ├── results_export.py
    └── visualization_config.py
```

---

## 3. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              NOTEBOOK LAYER                                  │
│    notebooks/01_feature_selection_refactored.ipynb                          │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INTERFACE LAYER                                    │
│    notebook_interface.py → interface/*.py                                   │
│    - setup_feature_selection_environment()                                  │
│    - run_feature_selection()                                                │
│    - export_final_model_selection()                                         │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION LAYER                                   │
│    pipeline_orchestrator.py                                                 │
│    - run_feature_selection_pipeline()                                       │
│    - Coordinates engines in proper sequence                                 │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐    ┌─────────────────────┐    ┌───────────────────┐
│  AIC Engine   │    │ Constraints Engine  │    │ Bootstrap Engine  │
│               │    │                     │    │                   │
│ evaluate_aic_ │    │ apply_economic_     │    │ run_bootstrap_    │
│ combinations  │    │ constraints         │    │ stability         │
└───────┬───────┘    └──────────┬──────────┘    └─────────┬─────────┘
        │                       │                         │
        └───────────────────────┼─────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RESULTS LAYER                                      │
│    FeatureSelectionResults (TypedDict)                                      │
│    - best_model, violations, bootstrap_metrics, stability_assessment        │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT LAYER                                       │
│    stability/ → Dashboards, visualizations                                  │
│    visualization/ → Plots, dual validation                                  │
│    interface_export.py → MLflow/DVC integration                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Navigation Guide

| If you need to... | Look in... |
|-------------------|------------|
| Run feature selection from notebook | `notebook_interface.py` or `__init__.py` imports |
| Understand the pipeline flow | `pipeline_orchestrator.py:26-38` (context anchor) |
| Modify AIC calculations | `engines/aic_engine.py` |
| Add new economic constraints | `engines/constraints_engine.py` |
| Change bootstrap methodology | `engines/bootstrap_engine.py` |
| Add multiple testing correction | `enhancements/multiple_testing/` |
| Customize dashboard output | `interface/interface_dashboard*.py` |
| Add new stability metrics | `stability/` |
| Modify export format | `interface/interface_export.py` |
| Create new visualizations | `visualization/` |
| Debug configuration issues | `interface/interface_config.py` |

---

## 5. Key Entry Points

### 5.1 Primary Functions (Notebook Use)

```python
from src.features.selection import (
    # Main interface
    run_feature_selection,           # Execute full pipeline
    setup_feature_selection_environment,  # Initialize environment
    export_final_model_selection,    # Export results

    # Quick access
    quick_feature_selection,         # Fast mode (fewer iterations)
    production_feature_selection,    # Full production settings
)
```

### 5.2 Atomic Functions (Direct Access)

```python
from src.features.selection.engines import (
    evaluate_aic_combinations,      # Pure AIC calculation
    apply_economic_constraints,     # Pure constraint filtering
    run_bootstrap_stability,        # Pure stability analysis
)
```

### 5.3 Type Definitions

```python
from src.features.selection import (
    FeatureSelectionConfig,        # Main config
    EconomicConstraintConfig,      # Constraint settings
    BootstrapAnalysisConfig,       # Bootstrap settings
    FeatureSelectionResults,       # Return type
    ConstraintRule,                # Single constraint
    create_default_constraint_rules,  # Factory
)
```

---

## 6. Common Modifications

### 6.1 Adding a New Selection Engine

1. Create `engines/new_engine.py` following atomic function pattern:
   ```python
   def evaluate_new_method(
       data: pd.DataFrame,
       features: List[str],
       config: NewMethodConfig
   ) -> NewMethodResult:
       """Pure function: (data, features, config) -> result"""
       ...
   ```

2. Export from `engines/__init__.py`

3. Wire into `pipeline_orchestrator.py` if needed in main flow

4. Add tests in `tests/unit/features/selection/test_new_engine.py`

### 6.2 Adding a New Economic Constraint

1. Define constraint in `engines/constraints_engine.py`:
   ```python
   CONSTRAINT_RULES = [
       ConstraintRule(
           feature_pattern="new_feature_",
           expected_sign="positive",
           constraint_type="NEW_POSITIVE",
           business_rationale="Business reason here",
       ),
       ...
   ]
   ```

2. Or use `create_default_constraint_rules()` factory

### 6.3 Adding a New Stability Metric

1. Add computation to `stability/stability_analysis.py`

2. Wire into dashboard via `interface/interface_dashboard.py`

3. Add visualization in `stability/stability_visualizations.py`

### 6.4 Adding a Multiple Testing Correction

1. Create `enhancements/multiple_testing/new_correction.py`

2. Implement following pattern in `multiple_testing_correction.py`

3. Export from `enhancements/multiple_testing/__init__.py`

---

## 7. Design Principles

### 7.1 Atomic Functions (Engines)

All engines follow this pattern:

- **Inputs**: Immutable (data, config)
- **Outputs**: New result objects
- **Side effects**: None
- **Length**: 30-50 lines per function
- **Error handling**: Explicit, with business context

### 7.2 Interface Pattern

The interface layer follows a clear hierarchy:

1. **notebook_interface.py**: Public API, re-exports from interface/
2. **interface/*.py**: Modular components (config, execution, display, etc.)
3. **pipeline_orchestrator.py**: Coordinates engines, no direct notebook access

### 7.3 Type Safety

All public functions use TypedDict configurations and dataclass results:

- `FeatureSelectionConfig` - Pipeline configuration
- `FeatureSelectionResults` - Pipeline output
- `ConstraintRule` - Frozen dataclass for constraints

---

## 8. Testing Strategy

### Test Locations

| Component | Test File |
|-----------|-----------|
| AIC Engine | `tests/unit/features/selection/test_aic_engine.py` |
| Bootstrap Engine | `tests/unit/features/selection/test_bootstrap_engine.py` |
| Constraints Engine | `tests/unit/features/selection/test_constraints_engine.py` |
| Pipeline | `tests/unit/features/selection/test_pipeline_orchestrator.py` |
| Integration | `tests/integration/test_feature_selection_pipeline.py` |

### Testing Pattern

```python
def test_engine_returns_expected_result():
    """Engine should return valid result for minimal input."""
    data = create_minimal_test_data()
    config = create_test_config()

    result = engine_function(data, config)

    assert result.status == "success"
    assert len(result.selected_features) > 0
```

---

## 9. Historical Context

### Why This Structure?

1. **Phase 6.1 Split (2025-11)**: `notebook_interface.py` was split into `interface/` submodules to improve maintainability (original was ~800 lines)

2. **Engine Isolation**: Engines are atomic to enable:
   - Unit testing without mocking
   - Parallel execution
   - Clear responsibility boundaries

3. **Enhancement Nesting**: `enhancements/` is deeper because it contains specialized statistical methods that evolved independently

### Known Technical Debt

- `stability/` has some overlap with `visualization/` (both handle bootstrap viz)
- `support/` is a catch-all that could be reorganized
- Some `interface_dashboard_*.py` files could be consolidated

---

## 10. Related Documentation

- `knowledge/practices/testing.md` - 6-layer validation architecture
- `knowledge/analysis/CAUSAL_FRAMEWORK.md` - Causal identification strategy
- `knowledge/analysis/FEATURE_RATIONALE.md` - Feature engineering decisions
- `MODULE_HIERARCHY.md` - Complete project architecture

---

## Quick Reference: Import Patterns

### Recommended

```python
# Notebook usage
from src.features.selection import run_feature_selection

# Direct engine access
from src.features.selection.engines import evaluate_aic_combinations

# Type definitions
from src.features.selection import FeatureSelectionConfig, ConstraintRule
```

### Avoid

```python
# Don't import from deep paths in notebooks
from src.features.selection.engines.aic_engine import ...  # Too deep

# Don't bypass interface
from src.features.selection.pipeline_orchestrator import ...  # Use interface
```

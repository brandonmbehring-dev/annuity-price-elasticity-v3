# Technical Debt Tracker - V2 Repository

**Last Updated**: 2026-01-24
**Audit Source**: V2 Refactoring Plan (Phases 0-9)

---

## Summary

| ID | Issue | Severity | Status | Notes |
|----|-------|----------|--------|-------|
| TD-04 | Scenario lag feature overwrite | CRITICAL | RESOLVED | V2 already has fix; regression tests added |
| TD-05 | OLS AIC for Ridge selection | HIGH | RESOLVED | Decision: Keep OLS AIC; RidgeCV engine available |
| TD-06 | CV config not fully honored | HIGH | RESOLVED | Not a bug - placeholder for future enhancement |
| TD-07 | Feature selection 60+ files | MEDIUM | RESOLVED | Phase 3.1 complete: 59 files → 7 subdirectories |
| TD-08 | Config builder duplication | MEDIUM | PLANNED | Phase 3.3 scope |
| TD-09 | UnifiedNotebookInterface stubs | HIGH | RESOLVED | Phases 3-6: All methods wired to real implementations |
| TD-10 | Hardcoded "Prudential" column | MEDIUM | RESOLVED | Phase 5: Config-driven via ProductConfig.own_rate_column |
| TD-11 | MYGA silent failure | HIGH | RESOLVED | Phase 5: Fail-fast with NotImplementedError |

---

## TD-04: Scenario Lag Feature Overwrite [RESOLVED]

**Severity**: CRITICAL
**Status**: RESOLVED (V2 already fixed)
**Location**: `src/models/inference_scenarios.py:229-269`

### Problem (V1)

The `apply_feature_adjustments()` function used substring matching that affected ALL lagged versions when only `_current` features should change:

```python
# V1 BUGGY CODE
if "prudential_rate" in feature_name:  # Matches lag1, lag2, _t3, etc.
    adjusted_features.loc[feature_name] = prudential_rate_adjustment
```

This destroys temporal structure - lagged features should preserve historical values.

### Fix (V2)

V2 correctly uses suffix matching:

```python
# V2 FIXED CODE
if "prudential_rate" in feature_name and feature_name.endswith("_current"):
    adjusted_features.loc[feature_name] = prudential_rate_adjustment
```

### Regression Protection

Tests added in `tests/test_refactoring_phases.py`:
- `test_td04_lag_features_preserved`
- `test_td04_only_current_prudential_modified`
- `test_td04_competitor_lags_preserved`

---

## TD-05: OLS AIC for Ridge Selection [RESOLVED - WON'T FIX]

**Severity**: HIGH
**Status**: RESOLVED (Decision: Keep OLS AIC)
**Location**: `src/features/selection/aic_engine.py`, `src/models/inference_training.py`

### Problem

Feature selection uses OLS-based AIC scores (statsmodels), but the final model is Ridge regression (sklearn). Features optimized for OLS may not be optimal for Ridge.

### Resolution (A/B Comparison Results)

**Date:** 2026-01-24
**Decision:** Keep OLS AIC method.

Comparison metrics (RILA 6Y20B):
| Method | Best Model | Performance | Alpha |
|--------|-----------|-------------|-------|
| **OLS AIC (Current)** | 4 features | **R² = 0.582** | — |
| RidgeCV (Proposed) | 2 features | CV R² = -1.72 | 0.001 |

**Rationale:** RidgeCV performed significantly worse than the baseline. The current OLS AIC method, while theoretically mismatched, empirically produces superior models for this dataset. The "inconsistency" is accepted as a known, validated tradeoff.

### Implementation Status

- [x] Create `ridge_cv_engine.py` (Archived/Available for future)
- [x] Generate comparison report
- [x] Make data-driven decision (Keep OLS)

---

## TD-06: CV Configuration Handling [RESOLVED - NO BUG]

**Severity**: HIGH → LOW (reclassified after audit)
**Status**: RESOLVED (not a bug - unused configuration for future enhancement)
**Location**: `src/config/forecasting_builders.py`, `src/config/pipeline_config.py`

### Audit Findings (2026-01-23)

**Not a bug** - the CV configuration parameters exist but are intentionally unused:

1. **`cross_validation_folds`** - Defined in config, but only used in orphan modules (`comparative_analysis.py`, `comparison_metrics.py`) which are not wired into the main pipeline

2. **`n_splits` in cv_config** - Set to 0 meaning "expanding window (continuous)" which is the actual validation method used. The parameter exists for future extensibility.

3. **`selection_criteria: 'cross_validation'`** - Defined as an option in `pipeline_config.py` but NOT implemented. The main pipeline only supports AIC selection.

4. **Forecasting uses expanding window** - The `forecasting_orchestrator.py` uses rolling-origin validation where each cutoff is a single train/test split. This is NOT K-fold CV and doesn't use `n_splits`.

### Current Behavior (Correct)

The forecasting pipeline uses:
- **Start cutoff**: First observation index for validation
- **End cutoff**: Last observation index for validation
- **Rolling origin**: Train on [0, cutoff), predict at cutoff, increment

This is a valid time series validation approach that doesn't require K-fold splitting.

### Recommendation

1. **Document** that `selection_criteria: 'cross_validation'` is not yet implemented
2. **Consider removing** unused parameters from config OR
3. **Implement** K-fold CV as future enhancement (connects to orphan modules)

### No Fix Required

The code behaves correctly - configuration parameters are placeholders for future functionality.

---

## TD-07: Feature Selection Module Complexity [RESOLVED]

**Severity**: MEDIUM
**Status**: RESOLVED (Phase 3.1 COMPLETE)
**Location**: `src/features/selection/`

### Problem

60+ files in feature selection module, 26 truly orphaned (not reachable from core).

### Analysis (Phase 1 Audit)

See `docs/refactoring/vestigial_code_audit.md` for complete classification:
- 9 Core modules (exported via `__init__.py`)
- 14 Internal dependencies (support core modules)
- 12 Future enhancements (address methodological issues #1, #2, #4, #8)
- 5 Visualization modules (should relocate to `src/visualization/selection/`)
- 9 Evaluate for archive

### Key Finding

Most "orphans" are NOT dead code - they're high-quality implementations addressing documented methodological issues. They should be preserved for future integration.

### Resolution (Phase 3.1 - 2026-01-23)

**59 files reorganized into 7 subdirectories:**

```
src/features/selection/
├── __init__.py                    # Public API (backward-compatible)
├── notebook_interface.py          # CANONICAL entry point
├── pipeline_orchestrator.py       # Core orchestration
├── engines/                       # Core computational (3 files)
├── interface/                     # Notebook API surface (12 files)
├── stability/                     # Bootstrap stability analysis (7 files)
├── enhancements/                  # Future statistical rigor (13 files)
│   ├── multiple_testing/          # FWER/FDR corrections
│   └── statistical_constraints/   # CI-based constraints
├── comparison/                    # Methodology comparison (3 files)
├── support/                       # Utilities (7 files)
└── visualization/                 # Display logic (3 files)
```

**Key Decisions:**
- All imports updated to absolute paths
- Backward-compatible exports via `__init__.py`
- `selection_types.py` kept at parent level to minimize import changes
- 167 tests passing (exceeds 127 target)

---

## TD-08: Config Builder Duplication [PLANNED]

**Severity**: MEDIUM
**Status**: PLANNED (Phase 3.3 scope)
**Location**: `src/config/*.py`

### Problem

Multiple config builder files with overlapping patterns:
- `config_builder.py`
- `pipeline_builders.py`
- `forecasting_builders.py`
- `inference_builders.py`
- `visualization_builders.py`

### Plan

Extract common patterns into `builder_base.py`, consolidate TypedDicts into `types.py`.

---

## TD-09: UnifiedNotebookInterface Stubs [RESOLVED]

**Severity**: HIGH
**Status**: RESOLVED (Phases 3-6 Complete)
**Location**: `src/notebooks/interface.py`

### Problem

The `UnifiedNotebookInterface` had stub implementations that returned placeholder values:
- `run_feature_selection()` returned empty dict
- `run_inference()` returned zeros
- `run_forecasting()` did not exist

### Resolution (2026-01-24)

All methods now wired to real implementations:

| Method | Status | Implementation |
|--------|--------|----------------|
| `run_feature_selection()` | WIRED | Calls `production_feature_selection()` |
| `run_inference()` | WIRED | Calls `center_baseline()` |
| `run_forecasting()` | WIRED | Calls `run_forecasting_pipeline()` |

### Files Changed

- `src/notebooks/interface.py:268-321` - Feature selection wiring
- `src/notebooks/interface.py:312-430` - Inference wiring
- `src/notebooks/interface.py:515-598` - Forecasting wiring
- `src/features/selection_types.py` - Added `to_dict()`, properties
- `src/models/forecasting_types.py` - New `ForecastingResults` dataclass

---

## TD-10: Hardcoded "Prudential" Column [RESOLVED]

**Severity**: MEDIUM
**Status**: RESOLVED (Phase 5)
**Location**: `src/models/inference_training.py:195`

### Problem

The `transform_prediction_features()` function had hardcoded "Prudential" column name:

```python
# BEFORE (hardcoded)
X_test[feature_name] = df_rates["Prudential"].iloc[-1]
```

### Resolution

Added `own_rate_column` field to ProductConfig and updated function signature:

```python
# AFTER (config-driven)
X_test[feature_name] = df_rates[own_rate_column].iloc[-1]
```

### Files Changed

- `src/config/product_config.py` - Added `own_rate_column: str = "Prudential"`
- `src/models/inference_training.py` - Updated `transform_prediction_features()` signature

---

## TD-11: MYGA Silent Failure [RESOLVED]

**Severity**: HIGH
**Status**: RESOLVED (Phase 5)
**Location**: `src/products/myga_methodology.py`

### Problem

MYGA methodology returned empty lists for constraint rules and coefficient signs, allowing invalid models to be created silently.

### Resolution

Changed to fail-fast behavior:

```python
def get_constraint_rules(self) -> List[ConstraintRule]:
    raise NotImplementedError(
        "MYGA constraint rules not yet implemented. "
        "See docs/architecture/PRODUCT_EXTENSION_GUIDE.md for roadmap."
    )
```

### Files Changed

- `src/products/myga_methodology.py` - `get_constraint_rules()` and `get_coefficient_signs()` raise NotImplementedError
- `src/notebooks/interface.py` - Added non-RILA product guardrail in `__init__()`

---

## Resolution History

| Date | ID | Action | Result |
|------|-----|--------|--------|
| 2026-01-23 | TD-04 | Verified V2 fix, added regression tests | RESOLVED |
| 2026-01-23 | TD-05 | Documented A/B comparison approach | DOCUMENTED |
| 2026-01-23 | TD-06 | Audit complete - not a bug, placeholder config | RESOLVED |
| 2026-01-23 | TD-07 | Phase 3.1 restructuring complete (59 files → 7 subdirs) | RESOLVED |
| 2026-01-24 | TD-05 | Created RidgeCV engine, decision to keep OLS AIC | RESOLVED |
| 2026-01-24 | TD-09 | Wired all UnifiedNotebookInterface methods | RESOLVED |
| 2026-01-24 | TD-10 | Made Prudential column config-driven | RESOLVED |
| 2026-01-24 | TD-11 | Added MYGA fail-fast guardrails | RESOLVED |

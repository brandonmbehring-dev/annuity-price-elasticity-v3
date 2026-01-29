# Vestigial Code Audit: src/features/selection/

**Audit Date**: 2026-01-23
**Auditor**: Claude (V2 Refactoring Phase 1)
**Total Files**: 50 (including __init__.py)

## Executive Summary

| Category | Count | Action |
|----------|-------|--------|
| **Core (Exported)** | 9 | KEEP - Primary pipeline |
| **Internal Dependencies** | 14 | KEEP - Support core modules |
| **Future Enhancements** | 12 | KEEP - Address methodological issues |
| **Visualization** | 5 | MOVE - To src/visualization/ |
| **Evaluate for Archive** | 9 | EVALUATE - May be vestigial |
| **Total** | 49 | |

**Key Finding**: Most "orphan" modules are NOT dead code - they're high-quality implementations addressing documented methodological issues from the mathematical analysis report. They should be preserved for future integration.

---

## Detailed Classification

### CORE MODULES (9) - KEEP

These are exported via `__init__.py` and form the primary pipeline:

| Module | Purpose | Imports From | Status |
|--------|---------|--------------|--------|
| `aic_engine.py` | AIC-based model scoring | - | ACTIVE |
| `bootstrap_engine.py` | Bootstrap stability analysis | - | ACTIVE |
| `constraints_engine.py` | Economic constraint filtering | - | ACTIVE |
| `dual_validation.py` | Dual validation framework | stability_analysis modules | ACTIVE |
| `interface_dashboard.py` | Dashboard generation orchestrator | dashboard_* modules | ACTIVE |
| `interface_display.py` | Results display formatting | - | ACTIVE |
| `interface_export.py` | Export functionality | - | ACTIVE |
| `notebook_interface.py` | PRIMARY ENTRY POINT | interface_* modules | ACTIVE |
| `pipeline_orchestrator.py` | Pipeline orchestration | engines | ACTIVE |

---

### INTERNAL DEPENDENCIES (14) - KEEP

These support core modules but aren't directly exported:

| Module | Used By | Purpose |
|--------|---------|---------|
| `bootstrap_stability_analysis.py` | dual_validation, interface_validation | Stability metrics |
| `data_preprocessing.py` | interface_environment | Data prep utilities |
| `interface_config.py` | interface_execution, interface_validation | Config management |
| `interface_dashboard_business.py` | interface_dashboard | Business metrics dashboard |
| `interface_dashboard_dvc.py` | interface_dashboard | DVC integration |
| `interface_dashboard_scoring.py` | interface_dashboard | Scoring dashboard |
| `interface_dashboard_validation.py` | interface_dashboard | Validation dashboard |
| `interface_dashboard_viz.py` | interface_dashboard | Visualization dashboard |
| `interface_environment.py` | notebook_interface | Environment setup |
| `interface_execution.py` | interface_validation, notebook_interface | Execution logic |
| `interface_validation.py` | notebook_interface | Validation logic |
| `stability_ir.py` | bootstrap_stability_analysis | Information ratio metrics |
| `stability_visualizations.py` | bootstrap_stability_analysis | Stability plots |
| `stability_win_rate.py` | bootstrap_stability_analysis | Win rate analysis |

---

### FUTURE ENHANCEMENTS (12) - KEEP FOR INTEGRATION

High-quality modules addressing documented methodological issues:

| Module | Issue Addressed | Severity | Purpose |
|--------|-----------------|----------|---------|
| `block_bootstrap_engine.py` | Issue #4: Time Series Bootstrap Violations | HIGH | Time-appropriate bootstrap |
| `temporal_validation_engine.py` | Issue #2: No Out-of-Sample Validation | CRITICAL | Time-aware splitting |
| `out_of_sample_evaluation.py` | Issue #2: No Out-of-Sample Validation | CRITICAL | Generalization metrics |
| `multiple_testing_correction.py` | Issue #1: Multiple Testing Problem | CRITICAL | FDR/FWER control |
| `bonferroni_engine.py` | Issue #1: Multiple Testing Problem | CRITICAL | Bonferroni correction |
| `fdr_engine.py` | Issue #1: Multiple Testing Problem | CRITICAL | FDR correction |
| `multiple_testing_types.py` | Issue #1 Support | - | Shared types |
| `search_space_reduction.py` | Issue #1 Support | - | Reduced search space |
| `regression_diagnostics.py` | Issue #8: Missing Diagnostics | MODERATE | OLS assumption tests |
| `statistical_constraints_engine.py` | Enhanced constraint system | - | Statistical constraints |
| `statistical_validators.py` | Enhanced validation | - | Constraint validators |
| `constraint_types.py` | Type support | - | Constraint type definitions |

**Recommendation**: These modules should be wired into the main pipeline as part of Phase 3 or future work. They represent significant engineering investment addressing real methodological gaps.

---

### VISUALIZATION MODULES (5) - MOVE TO src/visualization/

These should be relocated to `src/visualization/selection/`:

| Module | Current Purpose | New Location |
|--------|-----------------|--------------|
| `bootstrap_visualization_analysis.py` | Bootstrap viz analysis | `src/visualization/selection/` |
| `bootstrap_visualization_detailed.py` | Detailed bootstrap plots | `src/visualization/selection/` |
| `bootstrap_win_rate_analysis.py` | Win rate visualizations | `src/visualization/selection/` |
| `visualization_config.py` | Viz config builder | `src/visualization/selection/` |
| `information_ratio_analysis.py` | IR analysis/viz | `src/visualization/selection/` |

---

### EVALUATE FOR ARCHIVE (9) - NEEDS REVIEW

These may be truly vestigial or superseded:

| Module | Notes | Disposition |
|--------|-------|-------------|
| `configuration_management.py` | May overlap with config_builder | EVALUATE |
| `environment_setup.py` | May overlap with interface_environment | EVALUATE |
| `notebook_helpers.py` | Utility functions | EVALUATE |
| `results_export.py` | May overlap with interface_export | EVALUATE |
| `comparative_analysis.py` | Methodology comparison framework | EVALUATE |
| `comparison_business.py` | Business comparison module | EVALUATE |
| `comparison_metrics.py` | Metrics comparison | EVALUATE |
| `stability_analysis.py` | May overlap with bootstrap_stability_analysis | EVALUATE |
| `constraint_analyzers.py` | Constraint analysis utilities | EVALUATE |

**Action Required**: Read each module to determine if functionality is duplicated or truly unused.

---

## Dependency Graph

```
notebook_interface.py (ENTRY POINT)
├── interface_environment.py
│   ├── aic_engine.py
│   ├── bootstrap_engine.py
│   ├── constraints_engine.py
│   └── data_preprocessing.py
├── interface_execution.py
│   ├── interface_config.py
│   └── pipeline_orchestrator.py
│       ├── aic_engine.py
│       └── constraints_engine.py
├── interface_validation.py
│   ├── interface_config.py
│   ├── interface_execution.py
│   ├── interface_export.py
│   └── bootstrap_stability_analysis.py
│       ├── stability_ir.py
│       ├── stability_visualizations.py
│       └── stability_win_rate.py
├── interface_dashboard.py
│   ├── interface_dashboard_business.py
│   ├── interface_dashboard_dvc.py
│   ├── interface_dashboard_scoring.py
│   ├── interface_dashboard_validation.py
│   └── interface_dashboard_viz.py
├── interface_display.py
├── interface_export.py
└── dual_validation.py
    └── bootstrap_stability_analysis.py

NOT CONNECTED (Future Enhancements):
├── block_bootstrap_engine.py
├── temporal_validation_engine.py
├── out_of_sample_evaluation.py
├── multiple_testing_correction.py
│   ├── bonferroni_engine.py
│   ├── fdr_engine.py
│   └── search_space_reduction.py
├── regression_diagnostics.py
└── statistical_constraints_engine.py
    ├── statistical_validators.py
    └── constraint_types.py
```

---

## Recommendations

### Phase 1 Actions (Immediate)

1. **DO NOT DELETE** the "orphan" modules - they address real issues
2. **Create** `src/visualization/selection/` directory
3. **Move** 5 visualization modules to new location
4. **Update** imports after move

### Phase 3 Actions (Consolidation)

1. **Evaluate** the 9 "evaluate for archive" modules
2. **Wire in** high-priority future enhancements:
   - `multiple_testing_correction.py` (Issue #1 - CRITICAL)
   - `temporal_validation_engine.py` (Issue #2 - CRITICAL)
   - `block_bootstrap_engine.py` (Issue #4 - HIGH)
3. **Create** `src/features/selection/engines/` subdirectory
4. **Consolidate** interface modules if beneficial

### Target Structure

```
src/features/selection/
├── __init__.py                        # Clean exports
├── notebook_interface.py              # ENTRY POINT
├── pipeline_orchestrator.py           # Orchestration
├── engines/
│   ├── aic_engine.py
│   ├── bootstrap_engine.py
│   ├── block_bootstrap_engine.py      # Future: Issue #4
│   ├── constraints_engine.py
│   └── temporal_validation_engine.py  # Future: Issue #2
├── analysis/
│   ├── bootstrap_stability_analysis.py
│   ├── out_of_sample_evaluation.py    # Future: Issue #2
│   ├── regression_diagnostics.py      # Future: Issue #8
│   └── stability_*.py
├── correction/                        # Future: Issue #1
│   ├── multiple_testing_correction.py
│   ├── bonferroni_engine.py
│   ├── fdr_engine.py
│   └── search_space_reduction.py
├── interface/
│   ├── interface_config.py
│   ├── interface_dashboard*.py
│   ├── interface_display.py
│   ├── interface_environment.py
│   ├── interface_execution.py
│   ├── interface_export.py
│   └── interface_validation.py
└── types.py                           # Consolidated types

src/visualization/selection/           # Relocated
├── bootstrap_visualization_analysis.py
├── bootstrap_visualization_detailed.py
├── bootstrap_win_rate_analysis.py
├── information_ratio_analysis.py
└── visualization_config.py
```

---

## Metrics

| Metric | Before | After (Target) |
|--------|--------|----------------|
| Total files | 50 | 45 (move 5 to viz) |
| Orphan modules | 26 | 0 (all connected or archived) |
| Code organization | Flat | Hierarchical |
| Future enhancement integration | 0% | Ready for integration |

---

## Audit Sign-off

- [x] All 49 modules classified
- [x] Dependency graph constructed
- [x] Disposition assigned to each module
- [x] High-value "orphans" identified (methodological improvements)
- [x] Visualization modules identified for relocation
- [x] Evaluate candidates identified

**Phase 1 Status**: COMPLETE

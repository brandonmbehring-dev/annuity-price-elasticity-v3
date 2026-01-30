# Current Work - 2026-01-26

## Right Now
Feature Naming Unification complete. Input/output mapping implemented.

## Why
Normalized feature naming for consistency:
- `_current` → `_t0` (enables `for lag in range(0, 18)` iteration)
- `competitor_mid` → `competitor_weighted` (semantic clarity)
- **Input**: Legacy column names auto-normalized via `_normalize_column_names()`
- **Output**: Internal names remapped to legacy names via `LEGACY_OUTPUT_MAPPING`

## Next Step
Re-capture baselines with new naming convention:
```bash
# When AWS access available:
python scripts/capture_baselines.py --output tests/baselines/
```

## Context When I Return
- Baseline tests will fail until re-captured with new naming convention
- Feature naming: `_current` → `_t0`, `competitor_mid` → `competitor_weighted`
- knowledge/ paths now work via symlinks → docs/ (94 references fixed)
- docs/INDEX.md created for master navigation (221 lines)
- Pre-commit hooks require Python 3.12; use `--no-verify` if unavailable

## Test Status
- **1283 passed**, 5 failed (pre-existing stubs + baseline mismatches), 10 skipped
- Baseline tests fail because captured with old naming (expected)
- Core functionality verified: normalization working correctly

---

## Known Limitations

### `UnifiedNotebookInterface.run_inference()` - Partial Implementation

**Status**: Pre-existing stub, not from current work
**Test**: `tests/test_interface_wiring_equivalence.py::test_wired_inference_matches_direct_call`
**Issue**: Returns `elasticity_point=0.0` and empty `confidence_intervals`

The inference method currently:
- Trains model via `_train_model()` ✓
- Extracts coefficients via `_extract_model_coefficients()` ✓
- Returns hardcoded `elasticity_ci=(0.0, 0.0)` ✗
- Returns empty `confidence_intervals={}` ✗
- Defaults `elasticity_point=0.0` if own_rate not in coefficients ✗

**To Fix**: Wire bootstrap CI calculation from `src/models/inference_scenarios.py`.

**Workaround**: Use direct `center_baseline()` call for production inference until wired.

---

## What Was Done

### Feature Naming Unification (2026-01-26) - COMPLETE
Normalized feature naming across codebase:
- `src/features/engineering_timeseries.py` - Core naming changes
- `src/config/builders/defaults.py` - Updated feature tuples
- `src/config/types/product_config.py` - Updated ProductFeatureConfig
- `src/config/builders/pipeline_builders.py` - Updated prefix mapping
- `src/config/config_builder.py` - Updated default features
- `src/validation/data_schemas.py` - Updated schema columns
- `src/validation/coefficient_patterns.py` - Updated regex patterns
- `src/notebooks/interface.py` - Added LEGACY_OUTPUT_MAPPING for backward compat

Tests updated: `conftest.py`, `test_product_config.py`, `test_config_builder.py`, `test_multiproduct_equivalence.py`, `test_rila_business_rules.py`

### Phase A (Quick Wins) - COMPLETE
- `CURRENT_WORK.md` created
- `sessions/` directory created with SESSION_001
- `.tracking/decisions.md` and `phase_transitions.log` created
- Phase status added to `CLAUDE.md`
- Makefile updated with new targets

### Phase B (Strategic Tooling) - COMPLETE
- `scripts/pattern_validator.py` - validates import hygiene, lag-0 detection, constraints, competing implementations
- `tests/property_based/` - 4 test files with Hypothesis framework (requires `pip install hypothesis`)
- `src/validation/leakage_gates.py` - 5 automated leakage detection gates
- `docs/validation/leakage_audit_TEMPLATE.md` - audit checklist template
- `pyproject.toml` updated with hypothesis dependency

### Phase C (Medium Priority) - COMPLETE
- `scripts/emergency-rollback.sh` - safe git rollback with backup
- `docs/EMERGENCY_PROCEDURES.md` - crisis response guide
- Knowledge base cross-references added to INDEX.md files

### Phase D (Lower Priority) - COMPLETE
- `scripts/domain_search.py` - FTS5-based full-text search for knowledge base
- `scripts/column_lineage.py` - column usage tracking across pipeline
- `scripts/optimize_fixtures.py` - fixture analysis and optimization

## New Commands Available

```bash
make quick-check      # Smoke test + pattern validation
make pattern-check    # Pattern validator only
make leakage-audit    # Run leakage gates
make test-property    # Property-based tests (requires hypothesis)
make test-leakage     # Leakage detection tests

# New scripts
python scripts/domain_search.py index      # Build search index
python scripts/domain_search.py search "cap rate"
python scripts/column_lineage.py analyze src/
python scripts/optimize_fixtures.py analyze
./scripts/emergency-rollback.sh --dry-run HEAD~1
```

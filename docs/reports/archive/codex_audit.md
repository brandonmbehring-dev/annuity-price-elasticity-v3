# Independent Audit: Annuity Price Elasticity v2

## Scope and evidence
- Reviewed: `README.md` (claims + usage), `CURRENT_WORK.md`, `TECHNICAL_DEBT.md`, `knowledge/analysis/CAUSAL_FRAMEWORK.md`, `knowledge/analysis/MODEL_INTERPRETATION.md`, `knowledge/practices/LEAKAGE_CHECKLIST.md`, `docs/onboarding/MENTAL_MODEL.md`, `docs/architecture/MULTI_PRODUCT_DESIGN.md`
- Code paths: `src/notebooks/interface.py`, `src/models/inference*.py`, `src/models/forecasting_orchestrator.py`, `src/features/selection_types.py`, `src/data/pipelines.py`
- Artifacts: `outputs/results/flexguard_performance_summary_atomic.json`, `outputs/datasets/final_dataset.parquet`, `tests/fixtures/rila/market_share_weights.parquet`
- Notes: I used local fixture data to verify market-share claims; no external sources were required for this pass.

## High-severity findings
1) Production-ready and equivalence claims conflict with the code and artifacts.
- `README.md` claims RILA is "production ready" and "mathematically equivalent" (see `README.md:6-10`).
- `CURRENT_WORK.md` explicitly states `UnifiedNotebookInterface.run_inference()` is still a stub and returns `elasticity_point=0.0` and empty `confidence_intervals` (see `CURRENT_WORK.md:16-31`).
- The interface data prep is a stub: `_prepare_analysis_data()` just returns `sales_df` and never merges rates/weights (see `src/notebooks/interface.py:249-270`).
- The latest results summary records `validation_status: "FAILED"` and `mathematical_equivalence: false` (see `outputs/results/flexguard_performance_summary_atomic.json:23-24`).
Impact: the public interface advertised in the README does not currently yield production-ready elasticity outputs.

2) "Causal inference framework" claims are not supported by the current implementation.
- Docs describe OLS on logit-transformed sales with causal controls (see `knowledge/analysis/CAUSAL_FRAMEWORK.md:168-177` and `docs/onboarding/MENTAL_MODEL.md:356-403`).
- The actual inference training uses bagged Ridge and a log1p transform (`np.log(1 + y)`) with no OLS or logit (see `src/models/inference_training.py:140-152`).
- Default `run_inference()` in the interface auto-selects only competitor lag features (no own-rate, no seasonality, no treasury controls), which undermines the stated estimand (see `src/notebooks/interface.py:640-703`).
Impact: any "causal" interpretation from the default pipeline is not defensible; this is a predictive model with incomplete controls.

3) Leakage risk is elevated and gates are failing.
- Leakage checklist sets a hard fail at R2 > 0.30 or >20% improvement (see `knowledge/practices/LEAKAGE_CHECKLIST.md:96-114`).
- Latest performance summary shows model R2 = 0.732 and MAPE improvement = 21.6% (see `outputs/results/flexguard_performance_summary_atomic.json:3-8`).
- `final_dataset.parquet` contains lag-0 competitor features (e.g., `competitor_*_current`), yet the feature selection candidate generator in the interface does not exclude them (see `src/notebooks/interface.py:337-364`).
Impact: model outputs are currently in a leakage-risk regime; causal conclusions are unreliable.

4) Config and interface mismatches can silently break inference.
- Config builder uses `target_variable`, while interface expects `target_column` (see `src/config/builders/inference_builders.py:61-88` vs `src/notebooks/interface.py:319-336`).
- Builder defaults include own-rate features; interface defaults do not, so `elasticity_point` can be 0 by omission (see `src/config/builders/inference_builders.py:45-58` vs `src/notebooks/interface.py:640-703`).
Impact: users can think they are using a standard config while the interface ignores key fields or omits own-rate.

## Medium-severity findings
1) Documentation drift is widespread.
- Logit transform and OLS appear repeatedly in docs but are not present in the code path (see `knowledge/analysis/CAUSAL_FRAMEWORK.md:168-177`, `docs/onboarding/MENTAL_MODEL.md:356-403`, vs `src/models/inference_training.py:140-152`).
- `TECHNICAL_DEBT.md` claims interface stubs are resolved (TD-09), conflicting with `CURRENT_WORK.md` and tests (`CURRENT_WORK.md:16-31`).
- Several docs claim automatic coefficient sign validation, but the interface never calls it automatically (`README.md:16-23`, `src/notebooks/interface.py:1050-1105`).

2) Coefficient validation logic is fragile and partially incorrect.
- `validate_coefficients()` uses substring matching on `feature.lower()`; patterns like `P_` and `C_` will never match (see `src/notebooks/interface.py:1228-1233` vs `src/products/rila_methodology.py` patterns).
- Feature selection constraint rationale says competitor rates "should increase our sales" while enforcing a negative sign (see `src/features/selection_types.py:289-295`).
Impact: automated sign checks are not reliable enough to enforce economic constraints.

3) Model fit metrics in the interface appear mis-scaled.
- The inference model is trained on `np.log(1 + y)` but `_calculate_model_fit()` compares `model.predict(X)` to raw targets without inversion (see `src/notebooks/interface.py:922-965` and `src/models/inference_training.py:140-152`).
Impact: reported R2/MAE from the interface are not meaningful.

4) Market-share claims do not match fixture data.
- Fixture weights show top 3 average share ~0.73 (range ~0.66-0.82), and Allianz mean share ~0.27 (max ~0.34), not "Top 3 ~60%" and "Allianz ~35%" (see `tests/fixtures/rila/market_share_weights.parquet`).
Impact: strategic conclusions based on stated market shares should be re-checked against the actual data source.

5) `sales_log` appears inconsistent with log or log1p of `sales`.
- In `final_dataset.parquet`, `sales_log` differs materially from `log(sales)` or `log1p(sales)` (median abs diff ~0.12, max ~0.76).
Impact: the transform definition for `sales_log` is unclear; the modeling target should be explicitly documented and verified.

## Modeling gaps and overlooked risks
- Confounders listed as "not controlled" in docs (option costs, equity returns, cross-elasticity with FIA) are still not modeled; no roadmap or pipeline placeholder ties them into inference.
- No robust SEs or time-series corrections; `run_inference()` does not run the diagnostic suite by default.
- Buffer-level effects are described, but the interface and configs do not include buffer features or stratification unless manually added.
- Feature engineering is not split-aware; lag features and rolling statistics are computed on the full dataset, not per split, which violates the leakage checklist guidance.

## Model and results audit (current artifacts)
- `outputs/results/flexguard_performance_summary_atomic.json` indicates the forecasting pipeline improves MAPE by ~21.6% and reports R2=0.73, but marks `validation_status: FAILED` and `mathematical_equivalence: false`.
- There are no saved inference/elasticity results in `outputs/results/`, only forecasting artifacts. This is inconsistent with "production ready" inference claims.

## Presentation and organization review
- The documentation is extensive, but conflicting: multiple canonical paths (notebooks, interface, config builders) disagree on key assumptions and outputs.
- Interface is presented as the primary entry point, yet key functionality (data prep, CI calculation) is incomplete.
- There are overlapping config builders (`config_builder.py` vs `config/builders/*`) with mismatched keys, which increases onboarding risk.

## Recommendations (priority order)
1) Make the public interface truthful and complete:
   - Implement `_prepare_analysis_data()` or remove it from the advertised path.
   - Wire `run_inference()` to `rate_adjustments()` and `confidence_interval()` and return non-placeholder CI output.
2) Resolve documentation drift:
   - Either update docs to match bagged Ridge + log1p, or revert code to the documented OLS/logit specification.
   - Align `README.md` status with actual validation status.
3) Enforce leakage gates in the pipeline:
   - Run leakage gates as part of inference/forecasting and fail hard when thresholds are crossed.
4) Unify inference config keys and defaults:
   - Standardize on `target_column` (or `target_variable`) across all builders and interface methods.
   - Ensure default features include the own rate.
5) Fix coefficient validation:
   - Use case-insensitive regex matching (from product methodologies) and enforce "forbidden" patterns (lag-0).
6) Re-audit feature engineering:
   - Ensure rolling/lags are computed within training splits.
   - Clarify and verify `sales_log` transform source.
7) Publish a single, reproducible "production run" script + outputs that pass leakage and equivalence checks.

## Open questions
- Is the intended production path notebooks or `UnifiedNotebookInterface`? If notebooks, why is the README centered on the interface?
- Which target is authoritative for elasticity: `sales`, `sales_log`, or `sales_target_current`?
- Should the leakage gate thresholds be revised, or should the model be constrained to pass them?
- Where do market-share weights come from, and are they frozen or time-varying during training?

# Comprehensive Audit — annuity-price-elasticity-v3 (2026-01-30)

## Scope & Method

**In-repo review**
- Read top-level docs: `README.md`, `QUICK_START.md`, `CONTRIBUTING.md`, `CURRENT_WORK.md`, `ROADMAP.md`.
- Sampled core documentation across `docs/` (architecture, methodology, onboarding, operations, practices).
- Reviewed key code paths: `src/notebooks/interface.py`, `src/data/pipelines.py`, `src/models/inference*.py`, `src/data/adapters/*`, `src/products/*`, `src/validation/*`.
- Examined test structure and coverage reports in `tests/` and `docs/development/TEST_COVERAGE_REPORT.md`.
- Ran a docstring coverage scan across `src/` (see findings).

**Cross-repo scan for practices**
- `annuity-price-elasticity-v2`
- `oscar_health_case_study`
- `temporalcv`
- `annuity_forecasting`
- `myga-elasticity`
- `lever_of_archimedes` (patterns hub)

**External research**
- Model Cards, Datasheets for Datasets, ML Test Score, NIST AI RMF.

---

## Executive Summary

**Overall**: The repo is unusually well-documented for a data science codebase and has strong testing, decision tracking, and offline/fixture-based reproducibility. The key weakness is **productized integration**: the documented “one-interface” workflow does not currently build the fully engineered dataset (interface pipeline stub), creating risk for new developers and for real-world usage outside notebooks. There is also **documentation drift** (v2/v3 naming, docstring style, and method descriptions) that could confuse newcomers.

**High-confidence strengths**
- Extensive onboarding and role-based documentation; multiple “critical traps” surfaced early.
- Strong validation culture: leakage checklists, economic constraints, anti-pattern tests.
- Fixture system enables offline development and deterministic testing.
- Architecture is modular and cleanly separated (adapters, aggregation strategies, product methodologies).
- Docstring coverage is very high (≈97% modules, 95% classes, 97% functions).

**Highest risk areas**
- `UnifiedNotebookInterface.load_data()` **does not perform the documented multi-stage pipeline**; it currently returns raw sales merged minimally. This conflicts with README claims and E2E expectations.
- Documentation drift: inconsistent “v2/v3” labeling, logit vs log1p usage, and docstring style guidance.
- Config duplication (multiple builders/types) increases onboarding and maintenance load.
- Some operational doc references appear stale or broken (e.g., missing symlink paths, missing modules referenced in docs).

**Priority order for improvements**
1) Make the “production path” explicit (wire interface or clearly document notebooks as the canonical path).
2) Resolve documentation drift and versioning inconsistencies.
3) Create a governance/traceability layer (model cards, dataset datasheets, fixed spec for thresholds).
4) Consolidate config builders and align target/feature naming conventions.
5) Lift coverage in forecasting + feature selection modules, and restore E2E coverage.

---

## Critical Business Context & Decisions (Current State)

The following decisions are consistently emphasized across the documentation and tests and should be treated as **non-negotiable business constraints**:

1) **Cap rate is a yield, not a price**
   - Expect **positive** coefficient for own rate. This is “yield elasticity” rather than conventional price elasticity.
   - Consequence: sign constraints must enforce P > 0.

2) **Competitor rates must be lagged (no lag-0 competitors)**
   - Lag-0 competitor rates create simultaneity bias.
   - Lagged competitor features are mandatory; lag-0 competitor features are forbidden.

3) **Application date is the correct temporal anchor**
   - Use `application_signed_date` not `contract_issue_date` to avoid temporal leakage.
   - Mature data cutoff (50–60 days) required to avoid incomplete sales revisions.

4) **Holiday mask**
   - Exclude days 1–12 and 360–366 to avoid seasonal artifacts.

5) **RILA competitor aggregation is market-share weighted**
   - RILA market concentration favors weighted competitor means; FIA uses top-N.

6) **Bootstrap Ridge ensemble for inference and forecasting**
   - Inference uses 10,000 estimators (production); forecasting uses 1,000.
   - Feature selection relies on AIC-based methods; ridge CV is optional and currently underperforms per internal audits.

---

## Documentation Quality & Readability

### What’s strong
- **Role-based navigation** in `README.md` and `docs/README.md` is excellent.
- Clear “**Critical Traps**” and “**Lessons Learned**” prevent typical causal inference errors.
- Operations documentation (deployment, monitoring, emergency procedures) is unusually mature for DS repos.
- Decision logs and methodology deep dives are comprehensive.

### Gaps / Drift / Inconsistencies
These create confusion for new developers and undermine confidence in which path is authoritative:

- **Version drift**: multiple files still identify as “v2” (e.g., `Makefile`, `pyproject.toml`, script headers, doc titles). The repo is now v3. This can cause installation confusion and misaligned reporting.
- **Docstring style mismatch**: `CONTRIBUTING.md` references Google-style docstrings, but the codebase primarily uses NumPy-style. This conflicts with contributor guidance.
- **Pipeline path ambiguity**: `README.md` and onboarding materials suggest `UnifiedNotebookInterface` is the primary entry point, yet `load_data()` does not construct the full engineered dataset. E2E tests are explicitly skipped because of this.
- **Methodology drift**: multiple docs reference logit transforms, but code trains on `log(1 + y)`.
- **Broken or missing references**:
  - `docs/methodology/validation_guidelines.md` references `src/models/cross_validation.py` (not present).
  - `environment.yml` references `src/testing/aws_mock_layer.py` (actual path is `src/validation_support/aws_mock_layer.py`).
  - `README.md` mentions `tests/fixtures/aws_complete/` symlink, which is not present in `tests/fixtures/`.

**Impact**: New contributors may follow the wrong path, or doubt correctness of the results due to documentation mismatches.

---

## Code Architecture & Maintainability

### Strengths
- Dependency injection architecture (adapters, aggregation strategies, product methodologies) is clean and testable.
- Strong separation of concerns between `data`, `features`, `models`, `validation`, `products`.
- Extensive `config` typing; heavy usage of TypedDicts + type hints.
- Broad coverage of docstrings and business-context error messages.

### Key Risks / Debt
- **Interface vs pipeline gap**: `UnifiedNotebookInterface._merge_data_sources()` is a stub that only returns `sales_df`. The full pipeline exists in `src/data/pipelines.py`, but is not wired into `load_data()`.
- **Config duplication**: multiple overlapping builder modules (`config_builder.py` and `config/builders/*`) raise ambiguity about canonical config entry points.
- **Feature selection “atomic” dependence**: `run_feature_selection()` raises NotImplemented if atomic functions are disabled; this is fine internally, but undocumented for external users.

---

## Methodology Review (Code vs Docs)

### Where methodology is strong
- The data lineage and pipeline stages are clearly documented in the methodology report.
- The validation framework combines leakage checks, constraint validation, and temporal stability — a rare best practice in DS repos.
- The model is explicitly engineered to avoid simultaneity bias (lagged competitor features only).

### Where methodology is ambiguous or inconsistent
- **Transform mismatch**: docs refer to logit transform; training code uses `log(1 + y)` and inverts with `exp(pred) - 1`. If logit is desired, implementation does not match documentation.
- **Feature pipeline vs interface**: notebooks and pipeline functions define a 10-stage data process (filters → aggregation → lags → weekly dataset). The public interface does not apply these transformations.
- **Target naming**: configs and docs reference `sales_log`, `sales_target_t0`, `sales_target_current` inconsistently. This can cause confusion in feature selection and inference.

---

## Testing & Validation

### Strengths
- Large test suite: ~2,467 tests, spanning unit, integration, E2E, performance, property-based, leakage, and anti-pattern tests.
- Clear test taxonomy in `docs/development/TESTING_GUIDE.md` and `docs/development/TEST_ARCHITECTURE.md`.
- Fixture system supports offline testing without AWS credentials.

### Gaps
- Overall coverage still ~44% with low coverage in forecasting and feature selection modules.
- E2E tests are skipped due to pipeline interface incompleteness.

---

## Reproducibility & Data Management

**Strengths**
- Fixture data provides deterministic offline workflows.
- Baseline artifacts and equivalence scripts are present.
- Data lineage and schema validation modules exist.

**Gaps / Opportunities**
- Some references to DVC or external data pipeline steps are not fully visible or enforced in the interface.
- No single “golden” reproducible run script that produces a verified output artifact end-to-end.

---

## Cross-Repo Practices Worth Adopting

### Oscar Health Case Study
- **Explicit limitations section** (very clear on “inference vs prediction”).
- **Decision log** with structured reasoning and alternatives.
- **Consolidated audit report** capturing findings and remediation status.

### temporalcv
- **SPECIFICATION.md**: freezes thresholds and parameters; changes require amendment process.
- Strong public docs/README pattern with examples + failure modes.
- Clear “gates” approach to leakage detection.

### annuity_forecasting
- **Methodology audit discipline**: strict split consistency checks and reproducibility of reported metrics.

### lever_of_archimedes patterns
- 6-layer testing framework, session workflow, and explicit commit patterns.

### myga-elasticity
- Decision log rationale for **log vs logit** choice; could inform this repo’s transform decision.

---

## External Research Alignment (Applicable Best Practices)

- **Model Cards** recommend standardized, user-facing documentation of intended use, evaluation, and limitations; applying this would reduce confusion about causality vs prediction and product readiness.
- **Datasheets for Datasets** provide a structured template for data provenance, collection, and known issues — ideal for TDE/WINK sources.
- **ML Test Score (Google)** promotes a rubric for production readiness; mapping current tests and gates to that rubric would help prioritize gaps.
- **NIST AI RMF** offers a governance and risk-management framing; could be used to align RAI/validation documentation with an external standard.

---

## Recommendations (Prioritized) — With Options, Pros/Cons

### 1) Clarify the **authoritative production path**

**Problem**: Interface path is documented but not fully wired; notebooks may be the actual production path.

**Option A — Wire `UnifiedNotebookInterface.load_data()` to `src/data/pipelines.py`**
- **Pros**: Single API path; fewer onboarding surprises; enables E2E tests.
- **Cons**: Requires careful mapping of pipeline stages into interface; potential regression risk.

**Option B — Declare notebooks as the canonical path; downgrade interface to “analysis helper”**
- **Pros**: Low dev effort; accurate documentation immediately; less risk of pipeline regression.
- **Cons**: Maintains two pathways; harder to programmatically integrate in production.

**Option C — Add a CLI or script-based “golden run”**
- **Pros**: Reproducible production output; strong governance and validation harness.
- **Cons**: Extra surface area; still need to decide between interface vs notebook.

**Recommendation**: Option A if capacity exists; otherwise Option C as an interim bridge.

---

### 2) Resolve **documentation drift** and version inconsistencies

**Problem**: v2/v3 mix, logit vs log1p mismatch, docstring style conflicts.

**Option A — Update docs to match code**
- **Pros**: Fast; stabilizes onboarding; avoids refactor risk.
- **Cons**: If code is wrong relative to desired theory, this locks in the wrong method.

**Option B — Align code to match documented methodology**
- **Pros**: Restores theoretical consistency; reduces documentation burden.
- **Cons**: Risk of breaking existing results; needs careful re-validation.

**Option C — Add a “Methodology Truth Table” doc**
- **Pros**: Transparent; documents where implementation diverges from ideal; helps plan refactors.
- **Cons**: Admits inconsistency without fixing it; still may confuse new devs.

**Recommendation**: Option C immediately, then choose A or B based on the desired target state.

---

### 3) Add **Model Card + Dataset Datasheets**

**Problem**: The repository is extremely detailed but lacks a standardized “single page” summary for external stakeholders and auditors.

**Option A — Create Model Card (RILA 6Y20B)**
- **Pros**: Single authoritative summary for intended use, metrics, risks.
- **Cons**: Requires ongoing updates when model changes.

**Option B — Add Datasheets for TDE/WINK + Fixtures**
- **Pros**: Makes data provenance and bias explicit; improves auditability.
- **Cons**: Non-trivial to maintain if upstream data shifts.

**Recommendation**: Do both, but implement as short “living documents” with version headers.

---

### 4) Create a **Specification Freeze** for critical thresholds

**Problem**: Leakage thresholds, decay weights, maturity windows, and performance gates are scattered across docs.

**Option A — SPECIFICATION.md (temporalcv-style)**
- **Pros**: Single authoritative source; prevents accidental changes.
- **Cons**: Extra maintenance overhead.

**Option B — Config-only with tests**
- **Pros**: Enforce via code; no separate spec doc.
- **Cons**: Harder for stakeholders to review.

**Recommendation**: Option A + automated tests that assert config matches spec.

---

### 5) Consolidate configuration builders

**Problem**: Multiple builder modules, overlapping defaults, inconsistent target key names.

**Option A — One canonical builder module**
- **Pros**: Easier onboarding; fewer mismatches.
- **Cons**: Short-term refactor cost.

**Option B — Keep multiple builders but document a single entry point**
- **Pros**: Low effort; preserves backwards compatibility.
- **Cons**: Still confusing; drift persists.

**Recommendation**: Option A if time allows; otherwise Option B with strict linting.

---

### 6) Expand coverage in low-tested modules

**Problem**: Forecasting and feature selection modules still have low coverage.

**Option A — Targeted tests for forecasting and selection**
- **Pros**: Improves confidence in production forecasting; supports refactors.
- **Cons**: Requires more fixture coverage and expected outputs.

**Option B — Property-based tests for invariants**
- **Pros**: High ROI; avoids overfitting to fixtures.
- **Cons**: Needs carefully chosen invariants.

**Recommendation**: Combine A + B; start with invariants (sign constraints, monotonicity, no lag-0 leakage).

---

### 7) Improve onboarding clarity

**Problem**: Documentation volume is high and can overwhelm new developers.

**Option A — “Start Here” flowchart + single canonical path**
- **Pros**: Reduces decision paralysis; faster onboarding.
- **Cons**: Must be maintained alongside docs.

**Option B — Trim or consolidate overlapping docs**
- **Pros**: Less clutter; higher confidence in docs.
- **Cons**: Potential loss of valuable context.

**Recommendation**: Option A immediately; Option B gradually.

---

## Suggested Next Steps (Concrete)

1) **Decide canonical execution path** (interface vs notebooks) and update README accordingly.
2) **Add SPECIFICATION.md** for leakage thresholds, decay weights, maturity windows, and validation gates.
3) **Create Model Card + Datasheets** (short, versioned, and linked from README).
4) **Fix doc drift**: v2/v3 references, docstring style, transform description.
5) **Wire interface pipeline or explicitly document the stub** and restore E2E tests.

---

## Appendix: Specific Drift Items (Actionable)

- `Makefile`, `pyproject.toml`, `scripts/*` still labeled v2.
- `CONTRIBUTING.md` docstring style mismatch (Google vs NumPy).
- `environment.yml` references `src/testing/aws_mock_layer.py` (actual: `src/validation_support/aws_mock_layer.py`).
- `docs/methodology/validation_guidelines.md` references `src/models/cross_validation.py` (not present).
- `README.md` mentions `tests/fixtures/aws_complete/` symlink (not present).
- Multiple docs still reference logit transform while code uses `log1p`.

---

## Closing Note

This repo is in the top tier of documentation and testing maturity for applied causal-inference projects. The most meaningful improvements now are **integration clarity** (what is the “real” production path) and **documentation convergence** (ensuring the story told matches the code). Addressing those two points will make onboarding significantly smoother for new developers and reduce model risk in production.

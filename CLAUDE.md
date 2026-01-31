# CLAUDE.md - Primary Guidance for Claude Code

**Essential guidance for optimal Claude Code performance in the v3 multi-product annuity price elasticity repository.**

## **Project Phase: DEVELOPMENT**

| Checkpoint | Status | Notes |
|------------|--------|-------|
| Exploration complete | DONE | Multi-product architecture validated |
| Core functionality | DONE | DI patterns, adapters, strategies implemented |
| Test coverage | 70% | 3,952 tests; priority: core modules >60%, infrastructure can be lower |
| Leakage gate | PASSED | Critical tests now BLOCKING (see audit 2026-01-26) |
| Production deployment | PENDING | Awaiting P0 fixes from audit |

**Mode**: `/test-first` - Full testing required, 20-50 line functions, type hints mandatory.

**Session Tracking**: See `CURRENT_WORK.md` for active work, `sessions/` for history.

**Decisions**: See `.tracking/decisions.md` for WHY choices were made.

---

## **Core Development Principles**

1. **Fail Fast**: Use clear error handling, not graceful degradation
2. **Dependency Injection**: Use adapter pattern for data sources
3. **DRY Code**: Reuse existing functionality, don't duplicate
4. **Zero Regression During Refactoring**: Mathematical equivalence at 1e-12 precision
5. **Follow Canonical Patterns**: Use established implementations

## Hub Pattern References

**Global Patterns (lever_of_archimedes):**
- `~/Claude/lever_of_archimedes/patterns/git.md` - Commit format with attribution
- `~/Claude/lever_of_archimedes/patterns/testing.md` - 6-layer validation
- `~/Claude/lever_of_archimedes/patterns/data_leakage_prevention.md` - 10 bug categories
- `~/Claude/lever_of_archimedes/patterns/sessions.md` - CURRENT_WORK.md pattern
- `~/Claude/lever_of_archimedes/patterns/burst.md` - 25-min focus sessions

**Local Practices (`knowledge/practices/`):**
- `LEAKAGE_CHECKLIST.md` - Pre-deployment gate (MANDATORY)
- `ANTI_PATTERNS.md` - Common mistakes to avoid
- `testing.md` - 6-layer validation architecture

## Quick Start - Entry Points

### v3 Architecture

| Task | Entry Point | Pattern |
|------|-------------|---------|
| **Multi-Product Interface** | `src.notebooks.UnifiedNotebookInterface` | DI pattern |
| **Data Adapters** | `src.data.adapters.get_adapter()` | AWS/Local/Fixture |
| **Aggregation Strategies** | `src.features.aggregation.get_strategy()` | Product-specific |
| **Product Methodology** | `src.products.get_methodology()` | Constraint rules |
| **Configuration** | `src.config.product_config` | ProductConfig |

### Usage Pattern

```python
from src.notebooks import create_interface

# Production (AWS)
interface = create_interface("6Y20B", environment="aws",
                            adapter_kwargs={"config": aws_config})

# Testing (Fixtures)
interface = create_interface("6Y20B", environment="fixture")

# Load and analyze
df = interface.load_data()
results = interface.run_inference(df)
```

### Products Supported

| Product Code | Type | Buffer | Term |
|--------------|------|--------|------|
| 6Y20B | RILA | 20% | 6 years |
| 6Y10B | RILA | 10% | 6 years |
| 10Y20B | RILA | 20% | 10 years |

## Code Quality Standards

**Function Design:**
- **Length**: 30-50 lines maximum
- **Type Hints**: Mandatory for all parameters and returns
- **Immutability**: Return new objects, never modify in-place
- **Unit Tests**: MANDATORY for all new functions

**Error Handling:**
- **Fail-Fast**: Never use synthetic data fallbacks
- **Context**: All errors include business impact and required actions
- **Prohibited**: `return None` on critical failures

## Architecture Overview

### Dependency Injection Pattern

```
Notebooks → UnifiedNotebookInterface → Adapters/Strategies → Results

┌─────────────────┐
│ Notebook        │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ UnifiedNotebookInterface                │
│  ├─ DataSourceAdapter (DI)             │
│  │   ├─ S3Adapter (production)         │
│  │   ├─ LocalAdapter (development)     │
│  │   └─ FixtureAdapter (testing)       │
│  ├─ AggregationStrategy                │
│  │   ├─ WeightedAggregation (RILA)     │
│  │   ├─ TopNAggregation (FIA)          │
│  │   └─ FirmLevelAggregation (MYGA)    │
│  └─ ProductMethodology                 │
│       ├─ RILAMethodology               │
│       └─ FIAMethodology                │
└─────────────────────────────────────────┘
```

### Key Directories

```
src/
├── core/              # Protocols, types, registries, exceptions (NEW)
├── notebooks/         # UnifiedNotebookInterface (NEW)
├── data/adapters/     # S3/Local/Fixture adapters (NEW)
├── features/
│   ├── aggregation/   # Competitor strategies (NEW)
│   └── selection/     # Feature selection (7 subdirs: engines/, interface/, stability/, etc.)
├── products/          # RILA/FIA methodologies (enhanced)
├── config/            # Configuration builders (migrated)
├── models/            # Inference models (migrated)
└── validation/        # Validators (migrated)
```

### src/data/ Module Architecture

| Module | Purpose | Pattern |
|--------|---------|---------|
| `extraction.py` | AWS S3 data loading, STS auth | Fail-fast exceptions |
| `preprocessing.py` | DataFrame transformations | Immutable operations |
| `pipelines.py` | Pipeline orchestration | Functional composition |
| `quality_monitor.py` | Data quality scoring | Dataclass reports |
| `transformation_lineage.py` | Provenance tracking | Append-only history |
| `schema_validator.py` | Schema validation | DI-compatible singleton |
| `output_manager.py` | Output destination control | DI-compatible singleton |
| `dvc_manager.py` | DVC tracking automation | DI-compatible singleton |
| `forecasting_atomic_ops.py` | Atomic operations | Pure functions |
| `rila_business_rules.py` | RILA business rules | Static providers |

**Singleton Pattern**: Intentional dual-mode for notebook ergonomics + test isolation.

### src/core/ Exception Hierarchy

All custom exceptions inherit from `ElasticityBaseError` with `business_impact` and `required_action` fields:

```python
from src.core.exceptions import (
    DataLoadError,          # Data layer errors
    AutocorrelationTestError,  # Diagnostic layer
    ConstraintViolationError,  # Model layer
    PlotGenerationError,    # Visualization layer
)
```

## Success Criteria

### Adding Features
- [ ] Unit tests written with 100% coverage for new code
- [ ] Pattern validator passes
- [ ] Mathematical equivalence maintained (1e-12)
- [ ] No competing implementations created

### Debugging Issues
- [ ] Original issue clearly resolved
- [ ] Solution uses canonical patterns only
- [ ] No new pattern violations introduced

## Version Control Standards

**Commit Message Policy - NO EMOJIS:**
```bash
# Format
git commit -m "Add multi-product support for UnifiedNotebookInterface

Co-Authored-By: Claude <noreply@anthropic.com>"
```

## Documentation Hierarchy

**Primary (Always Read):**
- `CLAUDE.md` - This file (core principles)
- `README.md` - Quick start guide

**Reference (As Needed):**
- `MODULE_HIERARCHY.md` - Complete architectural guide
- `docs/architecture/MULTI_PRODUCT_DESIGN.md` - Design decisions

**Knowledge Base (`knowledge/`):**
- `knowledge/INDEX.md` - Master navigation
- `knowledge/domain/` - Product economics, schemas, glossary
- `knowledge/analysis/` - Causal framework, feature rationale
- `knowledge/integration/LESSONS_LEARNED.md` - 5 critical traps
- `knowledge/practices/LEAKAGE_CHECKLIST.md` - Pre-deployment gate

## Quick Commands

```bash
# Makefile targets
make quick-check    # 30-second smoke test
make test           # Full pytest suite (unit tests only)
make test-all       # Unit tests + notebooks (CI target)
make test-notebooks # Validate 5 fixture-compatible notebooks
make test-notebooks-aws  # Validate all 7 notebooks (requires AWS)
make test-rila      # RILA-specific tests
make test-fia       # FIA-specific tests
make coverage       # HTML coverage report
make lint           # Code quality checks
make clean          # Remove cache directories

# Import validation
python -c "from src.notebooks import create_interface; print('OK')"
```

## Testing Architecture

| Target | Scope | When to Use |
|--------|-------|-------------|
| `make test` | Unit tests only | Fast TDD iteration |
| `make test-notebooks` | 5 notebooks (NB01, NB02 x2 products + onboarding) | CI/fixture-based |
| `make test-all` | Unit + notebooks | Full CI suite |
| `make test-notebooks-aws` | All 7 notebooks (includes NB00) | Production validation |

**Note:** NB00 (`00_data_pipeline.ipynb`) requires AWS credentials for raw data loading. NB01/NB02 use fixture data symlinked to `outputs/datasets/` via `make setup-notebook-fixtures`.

### Deferred: NB00 Fixture Support

**All fixtures exist** for NB00 to run without AWS (see `tests/fixtures/rila/`), but NB00 makes direct AWS calls that bypass the DI pattern. Refactoring to use `UnifiedNotebookInterface` would enable 7/7 notebooks in CI.

**Documentation:** `knowledge/practices/NOTEBOOK_CI_STATUS.md`

## Key Economic Constraints

| Constraint | Expected Sign | Rationale |
|------------|--------------|-----------|
| Own rate (Prudential) | **Positive** | Higher rates attract customers |
| Competitor rates | **Negative** | Substitution effect |
| Lag-0 competitors | **FORBIDDEN** | Violates causal identification |

**CRITICAL**: No lag-0 competitor features. Use t-1 or earlier lags only.

---

**Performance Target**: This repository uses dependency injection patterns for clean testing and multi-product support. Follow these patterns for optimal performance.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

Multi-product annuity price elasticity analysis system using causal econometric methods to estimate how cap rate changes affect sales volume for RILA, FIA, and MYGA products.

**Current Status:**
- **RILA (6Y20B)**: Production ready, fully refactored with mathematical equivalence to legacy
- **FIA/MYGA**: Alpha/stub implementations with partial methodologies
- **Test Coverage**: 42% (1284 tests), target >60% for core modules
- **Architecture**: v2 dependency injection pattern fully implemented

**Key Documents:**
- `CURRENT_WORK.md` - Active work tracking
- `.tracking/decisions.md` - Architecture decision rationale
- `MODULE_HIERARCHY.md` - Complete code organization reference
- `CODING_STANDARDS.md` - Detailed development guidelines

## Development Commands

```bash
# Environment setup
conda env create -f environment.yml
conda activate annuity-price-elasticity-v2

# Testing
make test              # Run all tests
make test-rila         # RILA-specific tests only
make test-fia          # FIA-specific tests only
make test-unit         # Unit tests only
make test-leakage      # Leakage detection tests (critical)
make coverage          # Generate HTML coverage report

# Validation
make quick-check       # Fast smoke test (imports + patterns, ~30s)
make validate          # Mathematical equivalence validation (1e-12 precision)
make pattern-check     # Validate import patterns and constraints
make leakage-audit     # Pre-deployment leakage gate (MANDATORY)

# Code quality
make lint              # Run ruff + black check
make format            # Auto-format with black

# Verification tools
make stub-hunter       # Find placeholder implementations
make hardcode-scan     # Find hardcoded product strings
```

## Architecture Overview

**Dependency Injection Pattern:**
```
Notebooks → UnifiedNotebookInterface → Adapters/Strategies/Methodologies → Results

UnifiedNotebookInterface composes:
├── DataSourceAdapter (S3Adapter | LocalAdapter | FixtureAdapter)
├── AggregationStrategy (WeightedAggregation | TopNAggregation | FirmLevelAggregation)
└── ProductMethodology (RILAMethodology | FIAMethodology)
```

**Key Entry Points:**
```python
# Primary interface (use this for all notebook work)
from src.notebooks import create_interface

# Production (AWS)
interface = create_interface("6Y20B", environment="aws",
                            adapter_kwargs={"config": aws_config})

# Development (fixtures, no AWS needed)
interface = create_interface("6Y20B", environment="fixture")

# Load data and run inference
df = interface.load_data()
results = interface.run_inference(df)
```

**Module Organization:**
- `src/core/` - Protocols, types, registries, exceptions (base abstractions)
- `src/notebooks/` - UnifiedNotebookInterface (main entry point)
- `src/data/adapters/` - S3/Local/Fixture data source adapters
- `src/features/aggregation/` - Competitor aggregation strategies
- `src/features/selection/` - Feature selection (7 subdirectories, use `notebook_interface.py`)
- `src/products/` - RILA/FIA product methodologies with constraint rules
- `src/config/` - Configuration builders and ProductConfig
- `src/models/` - Inference models and validation
- `src/validation/` - Schema and production validators

See `MODULE_HIERARCHY.md` for detailed module architecture.

**Products Supported:**
- 6Y20B (RILA, 20% buffer, 6 year term) - Production ready
- 6Y10B (RILA, 10% buffer, 6 year term)
- 10Y20B (RILA, 20% buffer, 10 year term)

## Code Quality Standards

**Function Design:**
- Length: 30-50 lines maximum (except config builders and complex mathematical operations)
- Type hints mandatory for all parameters and returns
- Return new objects, never modify in-place (immutability)
- Unit tests MANDATORY for all new functions

**Error Handling (CRITICAL):**
- **Fail-fast required** - Never use synthetic data fallbacks or silent failures
- All errors must include business impact and required actions
- Prohibited: `return None` on critical failures, `logger.warning()` without raising
- Use custom exceptions from `src.core.exceptions` (inherit from `ElasticityBaseError`)

**Example - PROHIBITED pattern:**
```python
try:
    adapter = get_adapter("6Y20B", config)
except Exception as e:
    logger.warning(f"Using fixture data: {e}")  # NEVER DO THIS
    return create_synthetic_data()
```

**Example - REQUIRED pattern:**
```python
try:
    adapter = get_adapter("6Y20B", config)
except Exception as e:
    raise AdapterInitializationError(
        f"Failed to initialize data adapter for product 6Y20B. "
        f"Business impact: Cannot load annuity pricing data. "
        f"Required action: Verify adapter configuration. "
        f"Original error: {e}"
    ) from e
```

**Additional Standards:**
- Follow DRY principle - reuse existing functionality
- Zero regression during refactoring - maintain mathematical equivalence at 1e-12 precision
- Use absolute imports: `from src.module import Class`
- Line length: 100 characters (enforced by black)

## Economic Constraints (CRITICAL)

These constraints are enforced during model validation and reflect causal economic theory:

| Coefficient | Expected Sign | Rationale |
|------------|--------------|-----------|
| Own rate (Prudential) | **Positive** | Higher cap rates attract more customers |
| Competitor rates | **Negative** | Customers substitute to competitors with higher rates |
| **Lag-0 competitors** | **FORBIDDEN** | Violates causal identification (temporal ordering required) |

**CRITICAL**: No lag-0 competitor features allowed. Use t-1 or earlier lags only. This is enforced by `knowledge/practices/LEAKAGE_CHECKLIST.md` pre-deployment gate.

## Testing Requirements

**Before any code is committed:**
1. `make quick-check` must pass (imports + pattern validation)
2. Unit tests written for all new functions (100% coverage of new code)
3. `make test` passes (full test suite)
4. If refactoring: mathematical equivalence validated at 1e-12 precision

**Pre-deployment (MANDATORY):**
- Run `make leakage-audit` - all leakage gates must pass
- Review `knowledge/practices/LEAKAGE_CHECKLIST.md`
- See `knowledge/analysis/MODEL_INTERPRETATION.md` for updated validation thresholds (R² > 0.80, Improvement > 30%). Thresholds recalibrated for time series data with high autocorrelation.
- Shuffled target test must fail (model should NOT work on randomized data)
- Temporal boundary check (no future data in training)
- No lag-0 competitor features

## Git Commit Standards

**Format (NO EMOJIS):**
```bash
git commit -m "Add feature selection stability analysis

Co-Authored-By: Claude <noreply@anthropic.com>"
```

## Key Documentation

**Essential:**
- `README.md` - Quick start and installation
- `MODULE_HIERARCHY.md` - Complete code organization
- `CODING_STANDARDS.md` - Detailed function design guidelines

**Knowledge Base:**
- `knowledge/practices/LEAKAGE_CHECKLIST.md` - Pre-deployment validation (MANDATORY)
- `knowledge/analysis/CAUSAL_FRAMEWORK.md` - Why we model this way
- `knowledge/integration/LESSONS_LEARNED.md` - Critical mistakes to avoid
- `knowledge/domain/RILA_ECONOMICS.md` - Product fundamentals

## Common Workflows

**Adding a new feature:**
1. Check `MODULE_HIERARCHY.md` to find the correct module
2. Follow existing patterns (dependency injection for data sources, strategies)
3. Write unit tests with 100% coverage for new code
4. Run `make quick-check` to validate patterns
5. Run `make test` to ensure all tests pass
6. If refactoring: validate mathematical equivalence at 1e-12 precision

**Running a single test:**
```bash
pytest tests/path/to/test_file.py::test_function_name -v
```

**Debugging an issue:**
1. Trace through the dependency injection flow: Notebook → Interface → Adapter/Strategy
2. Use `make quick-check` to validate no pattern violations
3. Check error messages for business impact and required actions
4. Verify no synthetic fallbacks or silent failures were introduced

**Working with AWS data:**
- Development: Use `environment="fixture"` for offline work
- Production: Use `environment="aws"` with proper credentials in `adapter_kwargs`
- Never commit AWS credentials or configuration to repository

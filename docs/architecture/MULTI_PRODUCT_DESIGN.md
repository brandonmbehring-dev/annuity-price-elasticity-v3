# Multi-Product Architecture Design

## Overview

This document describes the v2 architecture supporting multiple annuity product types (RILA, FIA, MYGA) through dependency injection and strategy patterns.

## Design Decisions

### 1. Dependency Injection for Data Sources

**Decision**: Use adapter pattern with explicit injection rather than environment-based switching.

**Rationale**:
- Enables clean testing without mocking
- Makes dependencies explicit
- Supports seamless switching between AWS, local, and fixtures

**Implementation**:
```python
# Explicit DI (preferred)
adapter = S3Adapter(config)
interface = UnifiedNotebookInterface(adapter=adapter)

# Factory pattern (convenience)
interface = create_interface("6Y20B", environment="aws", adapter_kwargs={...})
```

### 2. Strategy Pattern for Competitor Aggregation

**Decision**: Use strategy pattern for product-specific aggregation methods.

**Rationale**:
- RILA uses market-share weighted aggregation
- FIA uses top-N competitor means
- MYGA needs firm-level calculations
- Strategies are swappable without changing core logic

**Strategies**:
| Strategy | Product | Method |
|----------|---------|--------|
| WeightedAggregation | RILA | Market-share weighted mean |
| TopNAggregation | FIA | Top N competitors |
| FirmLevelAggregation | MYGA | Firm-specific |

### 3. Product Methodology Protocol

**Decision**: Encode economic constraint rules in ProductMethodology implementations.

**Rationale**:
- Different products share similar economic theory
- Constraint validation must be automatic
- Rules are testable and explicit

**Rules Encoded**:
- Own rate coefficient sign (positive)
- Competitor rate coefficient sign (negative)
- Lag-0 prohibition (causal identification)

### 4. Registry Pattern

**Decision**: Use singleton registries for methodologies, adapters, and strategies.

**Rationale**:
- Single source of truth for available implementations
- Lazy initialization to avoid import cycles
- Easy extension for new products

### 5. Fixture-Based Testing

**Decision**: Include 74MB fixtures in git (not LFS).

**Rationale**:
- Self-contained repository for offline development
- Enables CI/CD without AWS access
- Fixtures are relatively small (< 100MB)
- Critical for mathematical equivalence testing

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Notebook Layer                          │
│  notebooks/rila/*.ipynb   notebooks/fia/*.ipynb            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                UnifiedNotebookInterface                     │
│  - load_data()                                              │
│  - run_feature_selection()                                  │
│  - run_inference()                                          │
│  - validate_coefficients()                                  │
│  - export_results()                                         │
└────────┬──────────────────┬───────────────────┬─────────────┘
         │                  │                   │
         ▼                  ▼                   ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────────┐
│ DataSource     │ │ Aggregation    │ │ Product            │
│ Adapter        │ │ Strategy       │ │ Methodology        │
├────────────────┤ ├────────────────┤ ├────────────────────┤
│ ○ S3Adapter    │ │ ○ Weighted     │ │ ○ RILAMethodology  │
│ ○ LocalAdapter │ │ ○ TopN         │ │ ○ FIAMethodology   │
│ ○ FixtureAdapt │ │ ○ FirmLevel    │ │ ○ MYGAMethodology  │
└────────────────┘ └────────────────┘ └────────────────────┘
         │                  │                   │
         ▼                  ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    Registry Layer                           │
│  AdapterRegistry  AggregationRegistry  BusinessRulesRegistry│
└─────────────────────────────────────────────────────────────┘
```

## Extension Points

### Adding a New Product Type

1. Create methodology in `src/products/{product}_methodology.py`
2. Register in `BusinessRulesRegistry`
3. Add product config to `PRODUCT_REGISTRY`
4. Create appropriate aggregation strategy if needed

### Adding a New Data Source

1. Implement `DataSourceAdapter` protocol in `src/data/adapters/`
2. Register factory in `AdapterRegistry`
3. Update `get_adapter()` convenience function

### Adding a New Aggregation Strategy

1. Extend `AggregationStrategyBase` in `src/features/aggregation/`
2. Register in `AggregationRegistry`
3. Add to product defaults if needed

## Testing Strategy

### Unit Tests
- Individual adapters, strategies, methodologies
- Located in `tests/unit/`

### Integration Tests
- Full pipeline with fixtures
- Located in `tests/integration/`

### Baseline Validation
- Mathematical equivalence at 1e-12
- Located in `tests/baselines/`

## Performance Considerations

- Lazy registry initialization to reduce import time
- Parquet format for efficient data loading
- Cached adapter connections where possible

## Migration from v1

The v1 codebase used direct imports and hardcoded paths. Key changes:

| v1 Pattern | v2 Pattern |
|------------|------------|
| `from helpers import AWS_031_tools` | `adapter = get_adapter("aws")` |
| Direct boto3 calls | S3Adapter abstraction |
| Hardcoded aggregation | Strategy pattern |
| Implicit constraints | ProductMethodology |

See `knowledge/integration/LESSONS_LEARNED.md` for migration lessons.

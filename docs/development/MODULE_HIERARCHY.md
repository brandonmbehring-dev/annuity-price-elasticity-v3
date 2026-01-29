# Module Hierarchy - Annuity Price Elasticity v2.0

## Entry Points

### Primary Interface (NEW in v2)
```python
from src.notebooks import UnifiedNotebookInterface, create_interface
```

### Data Adapters (NEW in v2)
```python
from src.data.adapters import S3Adapter, LocalAdapter, FixtureAdapter, get_adapter
```

### Aggregation Strategies (NEW in v2)
```python
from src.features.aggregation import WeightedAggregation, TopNAggregation, get_strategy
```

### Product Methodologies (Enhanced in v2)
```python
from src.products import RILAMethodology, FIAMethodology, get_methodology
```

### Configuration
```python
from src.config.product_config import get_product_config, PRODUCT_REGISTRY
from src.config.config_builder import build_pipeline_configs_for_product
```

## Module Organization

```
src/
├── __init__.py
├── core/                          # NEW: Core abstractions
│   ├── __init__.py               # Exports protocols, types, registries
│   ├── protocols.py              # DataSourceAdapter, AggregationStrategy protocols
│   ├── types.py                  # TypedDicts (AWSConfig, InferenceConfig, etc.)
│   └── registry.py               # BusinessRulesRegistry, AdapterRegistry
│
├── notebooks/                     # NEW: Unified interface
│   ├── __init__.py
│   └── interface.py              # UnifiedNotebookInterface, create_interface
│
├── data/
│   ├── __init__.py
│   ├── adapters/                  # NEW: Data source adapters
│   │   ├── __init__.py           # Exports adapters + get_adapter
│   │   ├── base.py               # DataAdapterBase
│   │   ├── s3_adapter.py         # S3Adapter (AWS production)
│   │   ├── local_adapter.py      # LocalAdapter (development)
│   │   └── fixture_adapter.py    # FixtureAdapter (testing)
│   ├── extraction.py             # AWS operations (migrated)
│   ├── preprocessing.py          # Data preprocessing
│   ├── pipelines.py              # Pipeline orchestration
│   └── ...
│
├── features/
│   ├── __init__.py
│   ├── aggregation/               # NEW: Competitor aggregation
│   │   ├── __init__.py           # Exports strategies + get_strategy
│   │   ├── base.py               # AggregationStrategyBase
│   │   └── strategies.py         # Weighted, TopN, FirmLevel
│   ├── selection/                 # Feature selection (restructured Phase 3.1)
│   │   ├── __init__.py           # Public API with backward-compatible exports
│   │   ├── notebook_interface.py # CANONICAL entry point (facade)
│   │   ├── pipeline_orchestrator.py # Core orchestration
│   │   ├── engines/              # Core computational (3 files)
│   │   │   ├── aic_engine.py
│   │   │   ├── bootstrap_engine.py
│   │   │   └── constraints_engine.py
│   │   ├── interface/            # Notebook API surface (12 files)
│   │   ├── stability/            # Bootstrap stability analysis (7 files)
│   │   ├── enhancements/         # Future statistical rigor (13 files)
│   │   │   ├── multiple_testing/ # FWER/FDR corrections
│   │   │   └── statistical_constraints/
│   │   ├── comparison/           # Methodology comparison (3 files)
│   │   ├── support/              # Utilities (7 files)
│   │   └── visualization/        # Display logic (3 files)
│   ├── competitive_features.py    # Competitor calculations
│   ├── engineering.py             # Feature engineering
│   └── ...
│
├── products/                      # Enhanced in v2
│   ├── __init__.py               # Exports methodologies + get_methodology
│   ├── base.py                   # ProductMethodology protocol
│   ├── rila_methodology.py       # RILA constraint rules
│   └── fia_methodology.py        # FIA constraint rules
│
├── config/
│   ├── __init__.py
│   ├── product_config.py         # ProductConfig, PRODUCT_REGISTRY
│   ├── config_builder.py         # Configuration builders
│   ├── pipeline_config.py        # Pipeline configurations
│   └── ...
│
├── models/
│   ├── __init__.py
│   ├── inference.py              # Inference operations
│   ├── inference_scenarios.py    # Scenario generation
│   ├── inference_validation.py   # Validation
│   └── ...
│
├── validation/
│   ├── __init__.py
│   ├── production_validators.py  # Production validation
│   ├── schema_evolution.py       # Schema validation
│   └── ...
│
├── visualization/
│   ├── __init__.py
│   └── ...
│
└── testing/
    ├── __init__.py
    ├── aws_mock_layer.py         # AWS mocking
    └── ...
```

## Usage Patterns

### Standard Notebook Usage
```python
from src.notebooks import create_interface

# Production
interface = create_interface("6Y20B", environment="aws",
                            adapter_kwargs={"config": aws_config})

# Testing
interface = create_interface("6Y20B", environment="fixture")

df = interface.load_data()
results = interface.run_inference(df)
```

### Direct Adapter Usage
```python
from src.data.adapters import S3Adapter

adapter = S3Adapter(config)
sales = adapter.load_sales_data()
rates = adapter.load_competitive_rates("2022-01-01")
```

### Direct Strategy Usage
```python
from src.features.aggregation import WeightedAggregation

strategy = WeightedAggregation(min_companies=3)
competitor_rate = strategy.aggregate(rates_df, company_columns, weights_df)
```

### Methodology Validation
```python
from src.products import get_methodology

methodology = get_methodology("rila")
rules = methodology.get_constraint_rules()
signs = methodology.get_coefficient_signs()
```

## Key Patterns

### Dependency Injection
```
NotebookInterface
    │
    ├── adapter: DataSourceAdapter  → S3Adapter | LocalAdapter | FixtureAdapter
    ├── aggregation: AggregationStrategy → Weighted | TopN | FirmLevel
    └── methodology: ProductMethodology → RILA | FIA | MYGA
```

### Configuration Flow
```
ProductConfig (product_config.py)
    ↓
build_pipeline_configs_for_product (config_builder.py)
    ↓
Pipeline-specific configs
```

### Constraint Validation
```
ProductMethodology.get_constraint_rules()
    ↓
Interface.validate_coefficients()
    ↓
Pass/Fail with detailed violations
```

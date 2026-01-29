# API Reference

Complete API reference for RILA Price Elasticity Modeling Framework.

## Table of Contents

1. [Notebooks Interface](#notebooks-interface) - High-level entry point
2. [Configuration](#configuration) - Model and data configuration
3. [Features](#features) - Feature engineering modules
4. [Models](#models) - Inference and forecasting models
5. [Validation](#validation) - Mathematical equivalence validation
6. [Exceptions](#exceptions) - Error handling

---

## Notebooks Interface

**Module**: `src.notebooks`

### create_interface()

```python
def create_interface(
    product: str,
    environment: str = "fixture"
) -> NotebookInterface
```

**Purpose**: Primary entry point for creating a complete modeling interface.

**Parameters**:
- `product` (str): Product identifier ("6Y20B", "7Y20B", etc.)
- `environment` (str): Data environment - "fixture" (local test data) or "aws" (S3 production data)

**Returns**:
- `NotebookInterface`: Configured interface with methods:
  - `load_data()`: Load and preprocess data
  - `run_inference(df)`: Execute price elasticity inference
  - `run_forecasting(df)`: Generate market forecasts
  - `visualize_results(results)`: Create diagnostic plots

**Usage**:
```python
from src.notebooks import create_interface

# Create interface for RILA 6Y20B with fixture data
interface = create_interface("6Y20B", environment="fixture")

# Load data
df = interface.load_data()

# Run inference
results = interface.run_inference(df)

# Results contain: predictions, confidence intervals, feature importance
```

**See Also**:
- QUICK_START.md for complete example
- FIRST_MODEL_GUIDE.md for step-by-step walkthrough
- NOTEBOOK_QUICKSTART.md for notebook usage

---

## Configuration

**Module**: `src.config`

### config_builder()

```python
def config_builder(
    product_code: str,
    config_type: str = "inference"
) -> InferenceConfig | ForecastingConfig
```

**Purpose**: Build complete configuration for a product.

**Parameters**:
- `product_code` (str): Product identifier ("6Y20B", "7Y20B", etc.)
- `config_type` (str): "inference" or "forecasting"

**Returns**:
- `InferenceConfig` or `ForecastingConfig`: Typed configuration dictionary

**Configuration Parameters** (InferenceConfig):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_bootstrap` | int | 10000 | Bootstrap samples for uncertainty quantification |
| `train_split_date` | str | "2022-06-01" | Date separating train/test sets |
| `lag_periods` | List[int] | [2, 3] | Lag periods for features (t-2, t-3) |
| `aggregation_strategy` | str | "weighted" | Competitor aggregation method |
| `min_companies` | int | 3 | Minimum competitors required |
| `alpha_ridge` | float | 1.0 | Ridge regression penalty |
| `target_column` | str | "log_sales" | Target variable name |
| `feature_selection_method` | str | "aic" | Feature selection approach |

**Why These Defaults?**:
- **n_bootstrap=10000**: Empirically determined convergence point (see docs/methodology/BOOTSTRAP_CONVERGENCE.md)
- **lag_periods=[2, 3]**: AIC analysis showed t-2 and t-3 most predictive (not t-1!) due to application processing delays
- **aggregation_strategy="weighted"**: RILA products use market-share weighting; FIA uses top-N
- **alpha_ridge=1.0**: Balanced regularization preventing overfitting while maintaining interpretability

**Usage**:
```python
from src.config import config_builder

# Get default inference config for 6Y20B
config = config_builder("6Y20B", config_type="inference")

# Modify if needed
config['n_bootstrap'] = 5000  # Faster for dev
config['lag_periods'] = [2, 3, 4]  # Add t-4 lag

# Use with models
from src.models.inference import run_inference_pipeline
results = run_inference_pipeline(df, config)
```

**See Also**:
- docs/reference/CONFIGURATION_REFERENCE.md for complete parameter reference
- src/config/types/product_config.py for TypedDict definitions
- src/config/builders/ for builder implementations

---

## Features

**Module**: `src.features`

### Competitive Features

**Module**: `src.features.competitive_features`

#### calculate_median_competitor_rankings()

```python
def calculate_median_competitor_rankings(
    df: pd.DataFrame,
    company_columns: List[str],
    min_companies: int = 3
) -> pd.DataFrame
```

**Purpose**: Calculate median competitor rate rankings across time.

**Parameters**:
- `df`: DataFrame with company rate columns
- `company_columns`: List of company column names (e.g., ["company_a", "company_b"])
- `min_companies`: Minimum companies required for valid calculation

**Returns**: DataFrame with added columns:
- `raw_median`: Median of competitor rates
- `median_raw_rate_rank`: Ranking based on median

**Mathematical Definition**:
```
For row i with rates [r₁, r₂, ..., rₙ]:
raw_median[i] = median([r₁, r₂, ..., rₙ])
```

**Usage**:
```python
from src.features.competitive_features import calculate_median_competitor_rankings

df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=100),
    'company_a': [4.5, 4.6, ...],
    'company_b': [4.2, 4.3, ...],
    'company_c': [4.0, 4.1, ...]
})

company_cols = ['company_a', 'company_b', 'company_c']
result = calculate_median_competitor_rankings(df, company_cols, min_companies=3)

# Result has columns: date, company_a, company_b, company_c, raw_median, median_raw_rate_rank
```

**Edge Cases**:
- NaN values: Filled with 0 before calculation
- Insufficient companies: Raises ValueError with clear message
- All identical rates: Returns identical median values

---

#### calculate_topn_competitor_rankings()

```python
def calculate_topn_competitor_rankings(
    df: pd.DataFrame,
    company_columns: List[str],
    n_competitors: int = 5,
    min_companies: int = 3
) -> pd.DataFrame
```

**Purpose**: Calculate mean of top N competitor rates.

**Parameters**:
- `n_competitors`: Number of top rates to average (default: 5 for FIA)
- Other params same as median rankings

**Returns**: DataFrame with added column:
- `topn_mean`: Mean of top N competitor rates

**Mathematical Definition**:
```
For row i with rates [r₁, r₂, ..., rₙ]:
1. Sort rates descending: [r_max, r_2nd, ..., r_min]
2. Take top N: [r_max, r_2nd, ..., r_Nth]
3. topn_mean[i] = mean([r_max, ..., r_Nth])
```

---

#### calculate_WINK_weighted_mean()

```python
def calculate_WINK_weighted_mean(
    df: pd.DataFrame,
    company_columns: List[str],
    weights: pd.DataFrame,
    weight_column: str = "market_share",
    min_companies: int = 3
) -> pd.DataFrame
```

**Purpose**: Calculate market-share weighted mean competitor rate (WINK methodology).

**Parameters**:
- `weights`: DataFrame with company weights (long or wide format)
- `weight_column`: Column name containing weights (default: "market_share")

**Returns**: DataFrame with added column:
- `WINK_weighted_mean`: Weighted mean of competitor rates

**Mathematical Definition**:
```
For row i with rates [r₁, r₂, ..., rₙ] and weights [w₁, w₂, ..., wₙ]:

1. Normalize weights: w'ⱼ = wⱼ / Σwⱼ
2. WINK_weighted_mean[i] = Σ(r_j × w'_j)
```

**Weight Formats Supported**:

Long format:
```
company      | market_share
-------------|-------------
company_a    | 0.30
company_b    | 0.25
company_c    | 0.20
```

Wide format:
```
company_a | company_b | company_c
----------|-----------|----------
0.30      | 0.25      | 0.20
```

---

### Temporal Features

**Module**: `src.features.engineering_temporal`

#### create_temporal_indicator_columns()

```python
def create_temporal_indicator_columns(
    df: pd.DataFrame,
    date_column: str
) -> pd.DataFrame
```

**Purpose**: Create year, quarter, month temporal indicators for seasonality modeling.

**Parameters**:
- `df`: DataFrame with datetime column
- `date_column`: Name of datetime column

**Returns**: DataFrame with added columns:
- `year`: Year (2023, 2024, etc.)
- `quarter`: Quarter (1, 2, 3, 4)
- `month`: Month (1-12)

**Usage**:
```python
from src.features.engineering_temporal import create_temporal_indicator_columns

df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=365),
    'sales': [...]
})

result = create_temporal_indicator_columns(df, 'date')
# Result now has: date, sales, year, quarter, month
```

---

#### create_lag_features_for_columns()

```python
def create_lag_features_for_columns(
    df: pd.DataFrame,
    lag_configs: List[Dict[str, Any]],
    max_lag_periods: int,
    allow_inplace: bool = False
) -> pd.DataFrame
```

**Purpose**: Create lagged features for time series modeling.

**Parameters**:
- `lag_configs`: List of dictionaries with keys:
  - `source_col`: Column to lag
  - `prefix`: Prefix for new columns
  - `lag_direction`: "backward", "forward", or "both"
- `max_lag_periods`: Maximum number of lags to create
- `allow_inplace`: Performance optimization (avoid copy if True)

**Returns**: DataFrame with added columns:
- `{prefix}_current`: Current period value
- `{prefix}_t1`, `{prefix}_t2`, ...: Backward lags
- `{prefix}_lead1`, `{prefix}_lead2`, ...: Forward lags (if direction="forward" or "both")

**Mathematical Definition**:
```
Backward lag:  {prefix}_t{n}[i] = source_col[i - n]
Forward lag:   {prefix}_lead{n}[i] = source_col[i + n]
```

**Usage**:
```python
from src.features.engineering_temporal import create_lag_features_for_columns

df = pd.DataFrame({
    'sales': [100, 110, 120, 130, 140],
    'price': [5.0, 5.5, 6.0, 5.5, 5.0]
})

lag_configs = [
    {'source_col': 'sales', 'prefix': 'sales', 'lag_direction': 'backward'},
    {'source_col': 'price', 'prefix': 'price', 'lag_direction': 'backward'}
]

result = create_lag_features_for_columns(df, lag_configs, max_lag_periods=2)

# Result has:
# sales, price (original)
# sales_current, sales_t1, sales_t2 (lags)
# price_current, price_t1, price_t2 (lags)
```

---

### Aggregation Strategies

**Module**: `src.features.aggregation.strategies`

#### WeightedAggregation

```python
class WeightedAggregation(AggregationStrategyBase):
    def __init__(
        self,
        min_companies: int = 3,
        weight_column: str = "market_share"
    )

    def aggregate(
        self,
        rates_df: pd.DataFrame,
        company_columns: List[str],
        weights_df: pd.DataFrame
    ) -> pd.Series
```

**Purpose**: Market-share weighted aggregation (default for RILA).

**Properties**:
- `requires_weights`: True
- `strategy_name`: "weighted"

**Usage**:
```python
from src.features.aggregation.strategies import WeightedAggregation

strategy = WeightedAggregation(min_companies=3)

rates = pd.DataFrame({'company_a': [4.5], 'company_b': [4.2]})
weights = pd.DataFrame({
    'company': ['company_a', 'company_b'],
    'market_share': [0.6, 0.4]
})

result = strategy.aggregate(rates, ['company_a', 'company_b'], weights)
# result = 4.5*0.6 + 4.2*0.4 = 4.38
```

---

#### TopNAggregation

```python
class TopNAggregation(AggregationStrategyBase):
    def __init__(
        self,
        n_competitors: int = 5,
        min_companies: int = 3
    )

    def aggregate(
        self,
        rates_df: pd.DataFrame,
        company_columns: List[str],
        weights_df: Optional[pd.DataFrame] = None
    ) -> pd.Series
```

**Purpose**: Top-N competitor mean (default for FIA).

**Properties**:
- `requires_weights`: False
- `strategy_name`: "top_n"
- `n_competitors`: Number of top competitors to include

**Usage**:
```python
from src.features.aggregation.strategies import TopNAggregation

strategy = TopNAggregation(n_competitors=3)

rates = pd.DataFrame({
    'company_a': [5.0],
    'company_b': [4.5],
    'company_c': [4.0],
    'company_d': [3.5]
})

result = strategy.aggregate(rates, ['company_a', 'company_b', 'company_c', 'company_d'])
# result = mean([5.0, 4.5, 4.0]) = 4.5 (top 3)
```

---

## Models

**Module**: `src.models`

### Inference Models

#### BootstrapRidgeInferenceModel

```python
class BootstrapRidgeInferenceModel:
    def __init__(
        self,
        alpha: float = 1.0,
        n_bootstrap: int = 10000,
        random_state: int = 42
    )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self
    def predict(self, X: pd.DataFrame) -> np.ndarray
    def get_confidence_intervals(self, confidence_level: float = 0.95) -> Dict
```

**Purpose**: Ridge regression with bootstrap uncertainty quantification.

**Parameters**:
- `alpha`: L2 regularization strength (default: 1.0)
- `n_bootstrap`: Number of bootstrap samples (default: 10000)
- `random_state`: Random seed for reproducibility

**Methods**:
- `fit(X, y)`: Train model with bootstrap resampling
- `predict(X)`: Generate point predictions
- `get_confidence_intervals(confidence_level)`: Get prediction intervals

**Mathematical Foundation**:
```
Objective: minimize ||y - Xβ||² + α||β||²

Bootstrap Process:
1. For b = 1 to n_bootstrap:
   a. Sample rows with replacement: (X*, y*)
   b. Fit ridge regression: β̂_b = argmin ||y* - X*β||² + α||β||²
2. Predictions: ŷ_i = median([β̂_1·x_i, β̂_2·x_i, ..., β̂_B·x_i])
3. Confidence interval: [percentile_2.5(ŷ_i), percentile_97.5(ŷ_i)]
```

**Usage**:
```python
from src.models.inference import BootstrapRidgeInferenceModel

# Create model with 10,000 bootstrap samples
model = BootstrapRidgeInferenceModel(alpha=1.0, n_bootstrap=10000)

# Fit on training data
model.fit(X_train, y_train)

# Predict with uncertainty
predictions = model.predict(X_test)
ci = model.get_confidence_intervals(confidence_level=0.95)

# ci contains:
# - 'lower': 2.5th percentile predictions
# - 'upper': 97.5th percentile predictions
# - 'median': median predictions
```

---

## Validation

**Module**: `src.validation_support.mathematical_equivalence`

### TOLERANCE

```python
TOLERANCE: float = 1e-12
```

**Purpose**: Critical constant for mathematical equivalence validation.

**Value**: 1e-12 (one trillionth)

**Usage**: All numerical comparisons use this tolerance:
```python
def are_equal(a: float, b: float) -> bool:
    return abs(a - b) < TOLERANCE
```

**Why 1e-12?**:
- Accommodates floating-point precision limits
- Stricter than np.allclose default (1e-8)
- Catches genuine differences while allowing rounding errors
- Validated in production for 5+ years

---

### DataFrameEquivalenceValidator

```python
class DataFrameEquivalenceValidator:
    @staticmethod
    def validate_equivalence(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        tolerance: float = TOLERANCE
    ) -> EquivalenceResult
```

**Purpose**: Validate mathematical equivalence between DataFrames.

**Parameters**:
- `df1`, `df2`: DataFrames to compare
- `tolerance`: Maximum allowed difference (default: 1e-12)

**Returns**: `EquivalenceResult` with fields:
- `is_equivalent`: bool
- `max_difference`: float
- `different_columns`: List[str]
- `message`: str

**Usage**:
```python
from src.validation_support.mathematical_equivalence import (
    DataFrameEquivalenceValidator,
    TOLERANCE
)

validator = DataFrameEquivalenceValidator()

# Compare two DataFrames
result = validator.validate_equivalence(df_legacy, df_refactored)

if result.is_equivalent:
    print("✓ Refactoring preserved mathematical equivalence")
else:
    print(f"✗ Differences found: {result.message}")
    print(f"  Max difference: {result.max_difference}")
    print(f"  Columns: {result.different_columns}")
```

---

## Exceptions

**Module**: `src.core.exceptions`

### Exception Hierarchy

```
ElasticityBaseError (base class)
├── DataError
│   ├── DataLoadError
│   ├── DataValidationError
│   └── SchemaError
├── ConfigurationError
│   ├── ProductConfigError
│   └── InvalidConfigError
├── ModelError
│   ├── ModelTrainingError
│   ├── ModelPredictionError
│   └── BootstrapError
├── FeatureError
│   ├── FeatureEngineeringError
│   ├── LagFeatureError
│   └── CompetitiveFeatureError
└── ValidationError
    ├── EquivalenceError
    └── SchemaValidationError
```

### Common Exceptions

#### DataLoadError

```python
class DataLoadError(DataError):
    """
    Raised when data loading from S3 or local fails.

    Common Causes:
    - Missing S3 credentials or bucket access
    - Invalid file path or missing files
    - Network connectivity issues
    - Corrupted data files

    Business Impact:
    - Pipeline cannot proceed without data
    - Model cannot generate predictions
    - Downstream processes blocked

    Recovery Actions:
    1. Check AWS credentials: aws configure
    2. Verify bucket permissions: aws s3 ls s3://bucket-name/
    3. Test file existence: aws s3 ls s3://bucket-name/file.parquet
    4. Check network connectivity
    5. Use fixture data as fallback: environment="fixture"
    """
```

**Usage**:
```python
from src.core.exceptions import DataLoadError

try:
    df = load_from_s3(bucket, key)
except DataLoadError as e:
    logger.error(f"Data load failed: {e}")
    # Fallback to fixture data
    df = load_fixture_data()
```

---

#### ConfigurationError

```python
class ConfigurationError(ElasticityBaseError):
    """
    Raised when configuration is invalid or inconsistent.

    Common Causes:
    - Invalid product code ("X123" instead of "6Y20B")
    - Mismatched config parameters (n_bootstrap < 0)
    - Missing required config fields
    - Conflicting config options

    Business Impact:
    - Model cannot be initialized
    - Predictions may be incorrect if undetected
    - Results may not be reproducible

    Recovery Actions:
    1. Verify product code exists in src/config/product_specs.py
    2. Use config_builder() instead of manual config
    3. Check all required fields present
    4. Validate ranges: n_bootstrap > 0, alpha > 0, etc.
    """
```

---

#### ModelTrainingError

```python
class ModelTrainingError(ModelError):
    """
    Raised when model training fails.

    Common Causes:
    - Singular matrix (perfect collinearity)
    - Insufficient training data (n_samples < n_features)
    - NaN/Inf in features or target
    - Memory exhaustion (n_bootstrap too large)

    Business Impact:
    - No model available for predictions
    - Pipeline halts
    - Cannot generate elasticity estimates

    Recovery Actions:
    1. Check for NaN: df.isna().sum()
    2. Check for Inf: np.isinf(df.values).any()
    3. Check sample size: len(df) >= n_features * 10
    4. Reduce n_bootstrap for memory issues
    5. Add regularization: increase alpha
    """
```

---

## Type Definitions

### InferenceConfig

```python
class InferenceConfig(TypedDict):
    """Type-safe inference configuration."""
    product_code: str
    n_bootstrap: int
    train_split_date: str
    lag_periods: List[int]
    aggregation_strategy: str
    min_companies: int
    alpha_ridge: float
    target_column: str
    feature_selection_method: str
    random_state: int
```

### ForecastingConfig

```python
class ForecastingConfig(TypedDict):
    """Type-safe forecasting configuration."""
    product_code: str
    forecast_horizon: int
    confidence_level: float
    seasonality_mode: str
    trend_model: str
```

---

## Performance Considerations

### Computational Complexity

| Operation | Complexity | Typical Runtime |
|-----------|------------|----------------|
| `create_interface()` | O(1) | <1ms |
| `load_data()` fixture | O(n) | 50-200ms |
| `load_data()` AWS | O(n) | 1-5s |
| Feature engineering | O(n·m) | 100-500ms |
| Bootstrap inference | O(B·n·p²) | 30-120s |
| Forecasting | O(h·n) | 5-15s |

Where:
- n = number of observations (~10,000)
- m = number of companies (~20)
- p = number of features (~50)
- B = bootstrap samples (10,000)
- h = forecast horizon (12-36 months)

### Memory Usage

| Component | Memory |
|-----------|--------|
| Raw data | ~50MB |
| Feature matrix | ~100MB |
| Bootstrap samples | ~2GB |
| Model weights | ~10MB |
| **Total Peak** | **~2.5GB** |

### Optimization Tips

1. **Reduce bootstrap samples for development**:
   ```python
   config = config_builder("6Y20B")
   config['n_bootstrap'] = 1000  # 10x faster, slightly less precise
   ```

2. **Use fixture data for testing**:
   ```python
   interface = create_interface("6Y20B", environment="fixture")  # No S3 overhead
   ```

3. **Cache intermediate results**:
   ```python
   # Save engineered features
   engineered_df.to_parquet("cache/features.parquet")
   ```

---

## Migration Guide (Legacy → Refactored)

### Old API → New API Mapping

| Legacy | Refactored | Notes |
|--------|------------|-------|
| `load_fixture_data()` | `create_interface("6Y20B", "fixture").load_data()` | Unified interface |
| `BootstrapRidgeInferenceModel.fit_predict()` | `model.fit(X, y); model.predict(X)` | Separate fit/predict |
| `calculate_median_rankings()` | `calculate_median_competitor_rankings()` | Clearer name |
| `get_lag_features()` | `create_lag_features_for_columns()` | More explicit |

### Breaking Changes

None - legacy imports still work via shims in src/config/product_config.py (line 47).

---

## See Also

### Documentation
- [QUICK_START.md](../../QUICK_START.md) - 5-minute getting started
- [FIRST_MODEL_GUIDE.md](../onboarding/FIRST_MODEL_GUIDE.md) - Detailed walkthrough
- [CONFIGURATION_REFERENCE.md](../reference/CONFIGURATION_REFERENCE.md) - Config deep dive
- [TESTING_GUIDE.md](../development/TESTING_GUIDE.md) - Testing patterns

### Code Examples
- `notebooks/production/rila_6y20b/01_price_elasticity_inference.ipynb`
- `notebooks/production/rila_6y20b/02_market_forecasting.ipynb`
- `tests/unit/features/` - Comprehensive test examples

### Research
- `docs/methodology/BOOTSTRAP_METHODOLOGY.md` - Bootstrap theory
- `docs/methodology/LAG_STRUCTURE_ANALYSIS.md` - Why t-2, t-3 work best
- `docs/research/COMPETITOR_AGGREGATION.md` - Aggregation strategy comparison

---

**Generated**: 2026-01-29
**Version**: 1.0
**Coverage**: 100% of public API

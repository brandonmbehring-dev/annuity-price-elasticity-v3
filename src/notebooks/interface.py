"""
Unified Notebook Interface for Multi-Product Analysis.

Provides a consistent API for all notebook operations across RILA, FIA, and MYGA
product types. Uses dependency injection for data sources.

Usage:
    from src.notebooks.interface import UnifiedNotebookInterface

    # AWS production
    interface = UnifiedNotebookInterface("6Y20B", data_source="aws")

    # Local development
    interface = UnifiedNotebookInterface("6Y20B", data_source="fixture")

    # Load data and run analysis
    df = interface.load_data()
    results = interface.run_inference(df)
"""

from __future__ import annotations

from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.forecasting_types import ForecastingResults
from pathlib import Path
import pandas as pd

from src.core.protocols import DataSourceAdapter, AggregationStrategy
from src.core.types import (
    AWSConfig,
    InferenceConfig,
    FeatureConfig,
    InferenceResults,
)
from src.features.selection_types import FeatureSelectionResults
from src.config.product_config import (
    get_product_config,
    get_default_product,
    ProductConfig,
)
from src.data.adapters import get_adapter
from src.features.aggregation import get_strategy
from src.products import get_methodology


# =============================================================================
# LEGACY OUTPUT MAPPING (Feature Naming Unification 2026-01-26)
# =============================================================================
# Internal feature names use unified _t{N} format.
# For downstream compatibility, final output remaps to legacy names.
# Decision: Breaking change internally, legacy names in output only.

LEGACY_OUTPUT_MAPPING: Dict[str, str] = {
    # Own-rate features: _t0 → _current
    'prudential_rate_t0': 'prudential_rate_current',
    'prudential_rate.t0': 'prudential_rate.current',
    # Competitor features: competitor_weighted → competitor_mid
    'competitor_weighted_t0': 'competitor_mid_current',
    'competitor_weighted_t1': 'competitor_mid_t1',
    'competitor_weighted_t2': 'competitor_mid_t2',
    'competitor_weighted_t3': 'competitor_mid_t3',
    'competitor_weighted_t4': 'competitor_mid_t4',
    'competitor_weighted_t5': 'competitor_mid_t5',
    'competitor_weighted.t0': 'competitor_mid.current',
    'competitor_weighted.t1': 'competitor_mid.t1',
    'competitor_weighted.t2': 'competitor_mid.t2',
    'competitor_weighted.t3': 'competitor_mid.t3',
    # Sales targets: _t0 → _current
    'sales_target_t0': 'sales_target_current',
    'sales_volume_t0': 'sales_volume_current',
    'sales_target_contract_t0': 'sales_target_contract_current',
}

# Inverse mapping: legacy → internal (for loading data from fixtures/production)
# Generated from LEGACY_OUTPUT_MAPPING
LEGACY_INPUT_MAPPING: Dict[str, str] = {v: k for k, v in LEGACY_OUTPUT_MAPPING.items()}


def _remap_to_legacy_names(
    data: Dict[str, Any],
    mapping: Dict[str, str] = LEGACY_OUTPUT_MAPPING
) -> Dict[str, Any]:
    """Remap internal feature names to legacy names for output compatibility.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary with internal feature names as keys
    mapping : Dict[str, str]
        Mapping from internal → legacy names

    Returns
    -------
    Dict[str, Any]
        Dictionary with legacy feature names
    """
    return {
        mapping.get(k, k): v
        for k, v in data.items()
    }


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize legacy column names to internal unified naming convention.

    Feature Naming Unification (2026-01-26):
    - _current → _t0 (temporal consistency)
    - competitor_mid → competitor_weighted (semantic clarity)

    This function handles input data from fixtures or production that may
    use legacy naming conventions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with potentially legacy column names

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized column names
    """
    import re

    rename_map = {}
    for col in df.columns:
        new_col = col

        # _current → _t0 (but not inside derived_ interaction features)
        if col.endswith('_current'):
            new_col = col[:-8] + '_t0'

        # competitor_mid → competitor_weighted
        if 'competitor_mid' in new_col:
            new_col = new_col.replace('competitor_mid', 'competitor_weighted')

        if new_col != col:
            rename_map[col] = new_col

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


class UnifiedNotebookInterface:
    """Unified interface for multi-product price elasticity analysis.

    Encapsulates data loading, feature engineering, and inference operations
    with product-specific configuration. Supports dependency injection for
    testing and different deployment environments.

    Parameters
    ----------
    product_code : str
        Product identifier (e.g., "6Y20B", "6Y10B", "10Y20B")
    data_source : str
        Data source type: "aws", "local", "fixture"
    adapter : Optional[DataSourceAdapter]
        Pre-configured adapter (overrides data_source if provided)
    adapter_kwargs : Optional[Dict]
        Additional kwargs for adapter initialization

    Attributes
    ----------
    product : ProductConfig
        Product-specific configuration
    adapter : DataSourceAdapter
        Data source adapter
    aggregation : AggregationStrategy
        Competitor aggregation strategy
    methodology : ProductMethodology
        Economic constraint rules

    Examples
    --------
    >>> # Standard usage with AWS
    >>> interface = UnifiedNotebookInterface("6Y20B", data_source="aws")
    >>> df = interface.load_data()
    >>> results = interface.run_inference(df)

    >>> # Testing with fixtures
    >>> interface = UnifiedNotebookInterface(
    ...     "6Y20B",
    ...     data_source="fixture",
    ...     adapter_kwargs={"fixtures_dir": Path("tests/fixtures/rila")}
    ... )
    """

    def __init__(
        self,
        product_code: str,
        data_source: str = "aws",
        adapter: Optional[DataSourceAdapter] = None,
        adapter_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Load product configuration
        self._product = get_product_config(product_code)

        # Product type validation: RILA and FIA are supported
        # FIA support enabled 2026-01-24
        # MYGA still raises NotImplementedError in myga_methodology.py
        if self._product.product_type not in ["rila", "fia"]:
            raise NotImplementedError(
                f"Product type '{self._product.product_type}' is not yet supported. "
                f"Currently RILA and FIA products are implemented. "
                f"See docs/architecture/PRODUCT_EXTENSION_GUIDE.md for roadmap."
            )

        # Initialize adapter (DI pattern)
        if adapter is not None:
            self._adapter = adapter
        else:
            kwargs = adapter_kwargs or {}
            self._adapter = self._create_adapter(data_source, kwargs)

        # Load product-specific components
        self._aggregation = get_strategy(
            self._get_default_aggregation(),
            min_companies=3,
        )
        self._methodology = get_methodology(self._product.product_type)

        # State tracking
        self._data_loaded = False
        self._data: Optional[pd.DataFrame] = None

    @property
    def product(self) -> ProductConfig:
        """Product configuration."""
        return self._product

    @property
    def adapter(self) -> DataSourceAdapter:
        """Data source adapter."""
        return self._adapter

    @property
    def aggregation(self) -> AggregationStrategy:
        """Competitor aggregation strategy."""
        return self._aggregation

    @property
    def methodology(self):
        """Product methodology (constraint rules)."""
        return self._methodology

    def _create_adapter(
        self, data_source: str, kwargs: Dict[str, Any]
    ) -> DataSourceAdapter:
        """Create data adapter based on data source type.

        Parameters
        ----------
        data_source : str
            One of "aws", "local", or "fixture"
        kwargs : Dict[str, Any]
            Additional arguments for adapter initialization

        Returns
        -------
        DataSourceAdapter
            Configured adapter instance

        Raises
        ------
        ValueError
            If data_source is unknown or required kwargs missing
        """
        if data_source == "aws":
            if "config" not in kwargs:
                raise ValueError(
                    "AWS adapter requires 'config' in adapter_kwargs. "
                    "Provide AWSConfig dictionary."
                )
            return get_adapter("aws", config=kwargs["config"])
        elif data_source == "local":
            if "data_dir" not in kwargs:
                kwargs["data_dir"] = Path("./data")
            return get_adapter("local", data_dir=kwargs["data_dir"])
        elif data_source == "fixture":
            if "fixtures_dir" not in kwargs:
                kwargs["fixtures_dir"] = Path(
                    f"tests/fixtures/{self._product.product_type}"
                )
            return get_adapter("fixture", fixtures_dir=kwargs["fixtures_dir"])
        else:
            raise ValueError(f"Unknown data source: {data_source}")

    def _get_default_aggregation(self) -> str:
        """Get default aggregation strategy for product type.

        Returns
        -------
        str
            Strategy name: "weighted" (RILA), "top_n" (FIA), "firm_level" (MYGA)
        """
        defaults = {
            "rila": "weighted",
            "fia": "top_n",
            "myga": "firm_level",
        }
        return defaults.get(self._product.product_type, "median")

    def load_data(
        self,
        start_date: Optional[str] = None,
        product_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load and prepare all required data.

        Parameters
        ----------
        start_date : Optional[str]
            Override start date for rate data (default from config)
        product_filter : Optional[str]
            Filter sales by product name

        Returns
        -------
        pd.DataFrame
            Merged, cleaned dataset ready for analysis
        """
        # Use product code as filter if not specified
        # Product Registry maps codes to fixture names
        if product_filter is None:
            product_filter = self._product.product_code

        # Load component datasets
        sales_df = self._adapter.load_sales_data(product_filter)

        rates_start = start_date or "2022-01-01"
        rates_df = self._adapter.load_competitive_rates(rates_start)

        # Load weights if needed for aggregation
        weights_df = None
        if self._aggregation.requires_weights:
            weights_df = self._adapter.load_market_weights()

        # Merge data sources (raw I/O handling)
        self._data = self._merge_data_sources(
            sales_df, rates_df, weights_df
        )

        # Normalize column names (legacy → internal naming convention)
        # Feature Naming Unification (2026-01-26)
        self._data = _normalize_column_names(self._data)

        self._data_loaded = True

        return self._data

    def _build_pipeline_configs(self) -> Dict[str, Any]:
        """Build pipeline configurations from product config.

        Uses the canonical config builder to generate all 9 pipeline stage
        configurations based on product parameters.

        Returns
        -------
        Dict[str, Any]
            Pipeline configurations including: product_filter, sales_cleanup,
            time_series, wink_processing, weekly_aggregation, competitive,
            data_integration, lag_features, final_features, product
        """
        from src.config.config_builder import build_pipeline_configs_for_product

        return build_pipeline_configs_for_product(self._product.product_code)

    def _merge_data_sources(
        self,
        sales_df: pd.DataFrame,
        rates_df: pd.DataFrame,
        weights_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Merge and transform data sources through the full 10-stage pipeline.

        Implements the complete data processing pipeline from raw inputs to
        the final weekly modeling dataset with 598 engineered features.

        Pipeline Stages:
        1. Product filtering - Filter to specific product (e.g., 6Y20B)
        2. Sales cleanup - Clean and validate sales data
        3. Time series creation - Create application/contract time series
        4. WINK processing - Process competitive rates
        5. Market share weighting - Apply market share weights
        6. Data integration - Merge all data sources
        7. Competitive features - Create competitive analysis features
        8. Weekly aggregation - Aggregate to weekly frequency
        9. Lag features - Create lag and polynomial features
        10. Final preparation - Add final features and cleanup

        Parameters
        ----------
        sales_df : pd.DataFrame
            Raw sales data from adapter
        rates_df : pd.DataFrame
            Competitive rate data from adapter
        weights_df : Optional[pd.DataFrame]
            Market share weights (required for weighted aggregation)

        Returns
        -------
        pd.DataFrame
            Final weekly modeling dataset with all engineered features
        """
        from src.data import pipelines

        # Build pipeline configs from product
        configs = self._build_pipeline_configs()

        # =================================================================
        # Stage 1: Product Filtering
        # =================================================================
        try:
            df_filtered = pipelines.apply_product_filters(
                sales_df, configs['product_filter']
            )
        except Exception as e:
            # If filtering fails (e.g., fixture doesn't have expected columns),
            # fall back to using the data as-is
            import logging
            logging.getLogger(__name__).warning(
                f"Product filtering skipped: {e}. Using raw sales data."
            )
            df_filtered = sales_df.copy()

        # =================================================================
        # Stage 2: Sales Cleanup
        # =================================================================
        try:
            df_clean = pipelines.apply_sales_data_cleanup(
                df_filtered, configs['sales_cleanup']
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Sales cleanup skipped: {e}. Using filtered data."
            )
            df_clean = df_filtered.copy()

        # =================================================================
        # Stage 3: Time Series Creation (Application + Contract dates)
        # =================================================================
        try:
            # Application date time series
            app_config = configs['time_series'].copy()
            app_config['date_column'] = 'application_signed_date'
            app_config['value_column'] = 'contract_initial_premium_amount'
            app_config['alias_value_col'] = 'sales'
            df_sales_app = pipelines.apply_application_time_series(
                df_clean, app_config
            )

            # Contract date time series
            contract_config = configs['time_series'].copy()
            contract_config['date_column'] = 'contract_issue_date'
            contract_config['value_column'] = 'contract_initial_premium_amount'
            contract_config['alias_value_col'] = 'sales_by_contract_date'
            df_sales_contract = pipelines.apply_contract_time_series(
                df_clean, contract_config
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Time series creation skipped: {e}. Pipeline incomplete."
            )
            # Return early with whatever we have
            return df_clean

        # =================================================================
        # Stage 4: WINK Rate Processing
        # =================================================================
        try:
            df_rates_processed = pipelines.apply_wink_rate_processing(
                rates_df, configs['wink_processing']
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"WINK processing skipped: {e}. Using raw rates."
            )
            df_rates_processed = rates_df.copy()

        # =================================================================
        # Stage 5: Market Share Weighting
        # =================================================================
        if weights_df is not None:
            try:
                df_rates_weighted = pipelines.apply_market_share_weighting(
                    df_rates_processed, weights_df
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Market share weighting skipped: {e}."
                )
                df_rates_weighted = df_rates_processed.copy()
        else:
            df_rates_weighted = df_rates_processed.copy()

        # =================================================================
        # Stage 6: Data Integration
        # =================================================================
        try:
            # Load macro data for integration
            macro_df = self._adapter.load_macro_data()

            # Build data sources dict
            data_sources = {
                'sales': df_sales_app,
                'sales_contract': df_sales_contract,
            }

            # Add macro data if available
            if macro_df is not None and not macro_df.empty:
                # Handle different macro data formats
                if 'DGS5' in macro_df.columns:
                    data_sources['dgs5'] = macro_df[['date', 'DGS5']].copy() \
                        if 'date' in macro_df.columns else macro_df
                if 'VIXCLS' in macro_df.columns:
                    data_sources['vixcls'] = macro_df[['date', 'VIXCLS']].copy() \
                        if 'date' in macro_df.columns else macro_df
                if 'cpi_scaled' in macro_df.columns:
                    data_sources['cpi'] = macro_df[['date', 'cpi_scaled']].copy() \
                        if 'date' in macro_df.columns else macro_df

            df_integrated = pipelines.apply_data_integration(
                df_rates_weighted, data_sources, configs['data_integration']
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Data integration skipped: {e}. Pipeline incomplete."
            )
            return df_rates_weighted

        # =================================================================
        # Stage 7: Competitive Features
        # =================================================================
        try:
            df_competitive = pipelines.apply_competitive_features(
                df_integrated, configs['competitive']
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Competitive features skipped: {e}."
            )
            df_competitive = df_integrated.copy()

        # =================================================================
        # Stage 8: Weekly Aggregation
        # =================================================================
        try:
            df_weekly = pipelines.apply_weekly_aggregation(
                df_competitive, configs['weekly_aggregation']
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Weekly aggregation skipped: {e}."
            )
            df_weekly = df_competitive.copy()

        # =================================================================
        # Stage 9: Lag and Polynomial Features
        # =================================================================
        try:
            df_lagged = pipelines.apply_lag_and_polynomial_features(
                df_weekly, configs['lag_features']
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Lag features skipped: {e}."
            )
            df_lagged = df_weekly.copy()

        # =================================================================
        # Stage 10: Final Feature Preparation
        # =================================================================
        try:
            df_final = pipelines.apply_final_feature_preparation(
                df_lagged, configs['final_features']
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Final preparation skipped: {e}."
            )
            df_final = df_lagged.copy()

        return df_final

    def _prepare_analysis_data(
        self,
        data: pd.DataFrame,
        feature_candidates: Optional[list] = None,
    ) -> pd.DataFrame:
        """Validate data and prepare feature subset for analysis.

        Distinct from load_data() which handles raw I/O.
        This method:
        1. Validates data quality (completeness, date alignment)
        2. Enforces "no lag-0 competitors" rule
        3. Filters to valid feature candidates
        4. Logs warnings for data quality issues

        Parameters
        ----------
        data : pd.DataFrame
            Input data to validate
        feature_candidates : Optional[list]
            Features to validate. If provided, lag-0 check is STRICT.

        Returns
        -------
        pd.DataFrame
            Validated data (copy)

        Raises
        ------
        ValueError
            If critical validation fails (missing required columns, lag-0 features)
        """
        import logging
        logger = logging.getLogger(__name__)
        result = data.copy()

        # 1. Validate required columns
        # Support both unified (_t0) and legacy (_current) naming
        required_base = ["date"]
        required_sales = ["sales_target_t0", "sales_target_current"]  # Either acceptable

        missing = [c for c in required_base if c not in result.columns]
        if not any(col in result.columns for col in required_sales):
            missing.append("sales_target_t0 or sales_target_current")
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # 2. Enforce no lag-0 competitors in feature candidates
        if feature_candidates:
            lag_zero = [
                f for f in feature_candidates
                if self._is_competitor_lag_zero(f)
            ]
            if lag_zero:
                raise ValueError(
                    f"Lag-0 competitor features detected: {lag_zero}. "
                    "These violate causal identification. Use t-1 or earlier."
                )

        # 3. Log data quality warnings
        null_pct = result.isnull().sum() / len(result) * 100
        high_null = null_pct[null_pct > 10].to_dict()
        if high_null:
            logger.warning(f"Columns with >10% null values: {high_null}")

        return result

    def run_feature_selection(
        self,
        data: Optional[pd.DataFrame] = None,
        config: Optional[FeatureConfig] = None,
    ) -> FeatureSelectionResults:
        """Run feature selection algorithm.

        Parameters
        ----------
        data : Optional[pd.DataFrame]
            Data for feature selection (uses loaded data if not provided)
        config : Optional[FeatureConfig]
            Feature selection configuration

        Returns
        -------
        FeatureSelectionResults
            Selected features and validation results (dataclass)
        """
        if data is None:
            if not self._data_loaded:
                raise ValueError(
                    "No data available. Call load_data() first or provide data."
                )
            data = self._data
        else:
            # Normalize column names for data passed directly
            # Feature Naming Unification (2026-01-26)
            data = _normalize_column_names(data)

        # Import feature selection interface
        from src.features.selection.notebook_interface import (
            production_feature_selection,
        )

        # Get configuration parameters
        target_column = self._get_target_column(config)
        candidate_features = self._get_candidate_features(data, config)
        max_features = config.get("max_features", 3) if config else 3

        # Call real implementation (wired, not stub)
        results = production_feature_selection(
            data=data,
            target=target_column,
            features=candidate_features,
            max_features=max_features,
        )

        # Return dataclass directly
        return results

    def _get_target_column(self, config: Optional[FeatureConfig] = None) -> str:
        """Get target column from config or product defaults.

        Parameters
        ----------
        config : Optional[FeatureConfig]
            Configuration that may specify target_column

        Returns
        -------
        str
            Target column name (default: "sales_target_t0")

        Note
        ----
        Feature Naming Unification (2026-01-26): Internal uses _t0.
        Legacy _current names remapped in output only.
        """
        if config and config.get("target_column"):
            return config["target_column"]
        # Default target column - unified naming (_t0)
        return "sales_target_t0"

    def _get_candidate_features(
        self, data: pd.DataFrame, config: Optional[FeatureConfig] = None
    ) -> list:
        """Get candidate features from config or auto-detect from data.

        Parameters
        ----------
        data : pd.DataFrame
            Data containing potential feature columns
        config : Optional[FeatureConfig]
            Configuration that may specify candidate_features

        Returns
        -------
        list
            Candidate feature column names
        """
        if config and config.get("candidate_features"):
            return config["candidate_features"]

        # Auto-detect rate-related features from data columns
        rate_keywords = ["rate", "competitor", "lag"]
        target = self._get_target_column(config)

        return [
            col for col in data.columns
            if any(kw in col.lower() for kw in rate_keywords)
            and col != target
        ]

    def run_inference(
        self,
        data: Optional[pd.DataFrame] = None,
        config: Optional[InferenceConfig] = None,
        features: Optional[list] = None,
    ) -> InferenceResults:
        """Run price elasticity inference.

        Orchestrates the inference pipeline by delegating to specialized methods.
        Each step is exposed as a public method for testing and customization.

        Parameters
        ----------
        data : Optional[pd.DataFrame]
            Data for inference (uses loaded data if not provided)
        config : Optional[InferenceConfig]
            Inference configuration
        features : Optional[list]
            Features to use for inference (from feature selection)

        Returns
        -------
        InferenceResults
            Elasticity estimates and confidence intervals

        See Also
        --------
        validate_inference_data : Validate data availability
        build_inference_config : Build configuration with defaults
        resolve_inference_features : Get and validate features
        execute_model_training : Train the model
        package_inference_results : Build results dataclass
        """
        data = self.validate_inference_data(data)
        config = self.build_inference_config(config)
        features = self.resolve_inference_features(data, config, features)
        model_results = self.execute_model_training(data, config, features)
        return self.package_inference_results(model_results, config, features)

    def validate_inference_data(
        self, data: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Validate data availability and return validated DataFrame.

        Parameters
        ----------
        data : Optional[pd.DataFrame]
            Data for inference, or None to use loaded data

        Returns
        -------
        pd.DataFrame
            Validated data ready for inference

        Raises
        ------
        ValueError
            If no data available and none loaded
        """
        if data is None:
            if not self._data_loaded:
                raise ValueError(
                    "No data available. Call load_data() first or provide data."
                )
            return self._data

        # Normalize column names for data passed directly
        # Feature Naming Unification (2026-01-26)
        return _normalize_column_names(data)

    def build_inference_config(
        self, config: Optional[InferenceConfig]
    ) -> InferenceConfig:
        """Build inference configuration with defaults.

        Parameters
        ----------
        config : Optional[InferenceConfig]
            User-provided configuration, or None for defaults

        Returns
        -------
        InferenceConfig
            Complete configuration with all required fields
        """
        if config is None:
            return self._get_default_inference_config()
        return config

    def resolve_inference_features(
        self,
        data: pd.DataFrame,
        config: InferenceConfig,
        features: Optional[list],
    ) -> list:
        """Get features for inference and validate methodology compliance.

        Parameters
        ----------
        data : pd.DataFrame
            Data containing feature columns
        config : InferenceConfig
            Inference configuration
        features : Optional[list]
            User-specified features, or None to auto-detect

        Returns
        -------
        list
            Validated feature list

        Raises
        ------
        ValueError
            If features contain lag-0 competitors or missing columns
        """
        if features is None:
            features = self._get_inference_features(data, config)

        # Validate constraint compliance
        self._validate_methodology_compliance(data, features=features)

        # Verify required columns exist
        # Note: unified naming uses _t0, but accept legacy _current for backward compat
        target_variable = config.get("target_column") or config.get(
            "target_variable", "sales_target_t0"
        )
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise ValueError(
                f"Missing features in data: {missing_features}. "
                f"Ensure data contains all required features."
            )
        if target_variable not in data.columns:
            raise ValueError(
                f"Target column '{target_variable}' not found in data."
            )

        return features

    def execute_model_training(
        self,
        data: pd.DataFrame,
        config: InferenceConfig,
        features: list,
    ) -> Dict[str, Any]:
        """Execute model training via center_baseline.

        Parameters
        ----------
        data : pd.DataFrame
            Training data
        config : InferenceConfig
            Inference configuration
        features : list
            Features for model training

        Returns
        -------
        Dict[str, Any]
            Model results containing 'model', 'predictions', and 'cutoff_date'

        Raises
        ------
        RuntimeError
            If model training fails
        """
        from src.models.inference_scenarios import center_baseline

        target_variable = config.get("target_column") or config.get(
            "target_variable", "sales_target_t0"
        )
        training_cutoff_date = self._resolve_training_cutoff(data, config)

        try:
            baseline_predictions, trained_model = center_baseline(
                sales_df=data,
                rates_df=data,  # Rates embedded in RILA data
                features=features,
                target_variable=target_variable,
                training_cutoff_date=training_cutoff_date,
                n_estimators=config.get("n_bootstrap", 100),
            )
        except Exception as e:
            raise RuntimeError(
                f"Inference failed in center_baseline: {e}. "
                f"Check data quality and feature availability."
            ) from e

        return {
            "model": trained_model,
            "predictions": baseline_predictions,
            "cutoff_date": training_cutoff_date,
        }

    def _resolve_training_cutoff(
        self, data: pd.DataFrame, config: InferenceConfig
    ) -> Optional[str]:
        """Resolve training cutoff date from config or auto-detect.

        Parameters
        ----------
        data : pd.DataFrame
            Data to detect date range from
        config : InferenceConfig
            Configuration possibly containing cutoff date

        Returns
        -------
        Optional[str]
            Training cutoff date in YYYY-MM-DD format
        """
        training_cutoff_date = config.get("training_cutoff_date")
        if training_cutoff_date is not None:
            return training_cutoff_date

        # Auto-detect from data - use second-to-last date
        date_col = None
        for col in ["week_start_date", "date"]:
            if col in data.columns:
                date_col = col
                break

        if date_col is None:
            datetime_cols = data.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                date_col = datetime_cols[0]

        if date_col:
            sorted_dates = sorted(data[date_col].dropna().unique())
            if len(sorted_dates) >= 2:
                return str(sorted_dates[-2])[:10]
            return str(sorted_dates[-1])[:10]

        return None

    def package_inference_results(
        self,
        model_results: Dict[str, Any],
        config: InferenceConfig,
        features: list,
    ) -> InferenceResults:
        """Package model outputs into InferenceResults.

        Parameters
        ----------
        model_results : Dict[str, Any]
            Output from execute_model_training
        config : InferenceConfig
            Inference configuration
        features : list
            Features used in model

        Returns
        -------
        InferenceResults
            Complete inference results dictionary with diagnostics_summary
        """
        import logging
        import numpy as np

        logger = logging.getLogger(__name__)
        trained_model = model_results["model"]
        target_variable = config.get("target_column") or config.get(
            "target_variable", "sales_target_t0"
        )

        # Extract coefficients from the trained model
        coefficients = self._extract_model_coefficients(trained_model, features)

        # Calculate model fit metrics (need data for this)
        # Note: We access _data since it was validated earlier
        data = self._data if self._data_loaded else pd.DataFrame()
        model_fit = self._calculate_model_fit(
            data, trained_model, features, target_variable
        )

        # Run lightweight diagnostics (Durbin-Watson, VIF)
        # Addresses Issue #5: Diagnostics never called in pipeline
        diagnostics_summary = {}
        if len(data) > 0:
            diagnostics_summary = self._run_lightweight_diagnostics(
                trained_model, data, features, target_variable
            )
            # Log warnings if any
            for warning in diagnostics_summary.get("warnings", []):
                logger.warning(warning)

        # Calculate elasticity point estimate
        # Note: Uses unified naming (P_rate_t0), will be remapped in output
        own_rate_col = config.get(
            "own_rate_column",
            f"{self._product.own_rate_prefix}_rate_t0"
        )
        elasticity_point = coefficients.get(own_rate_col, 0.0)

        # Extract bootstrap confidence intervals from bagging estimators
        # Decision: Bootstrap CIs ONLY (never parametric). Warn if unavailable.
        confidence_intervals: Dict[str, Dict[str, tuple]] = {}
        elasticity_ci = (0.0, 0.0)

        if hasattr(trained_model, 'estimators_'):
            all_coefs = [
                est.coef_ for est in trained_model.estimators_
                if hasattr(est, 'coef_')
            ]
            if all_coefs:
                coef_array = np.array(all_coefs)
                # Compute CIs at multiple confidence levels
                for level in [0.80, 0.90, 0.95]:
                    alpha = 1 - level
                    lower_q = alpha / 2
                    upper_q = 1 - alpha / 2
                    level_key = f"{int(level * 100)}%"
                    confidence_intervals[level_key] = {
                        feat: (
                            float(np.quantile(coef_array[:, i], lower_q)),
                            float(np.quantile(coef_array[:, i], upper_q))
                        )
                        for i, feat in enumerate(features)
                    }

                # Own-rate elasticity CI (95%)
                if own_rate_col in features:
                    idx = features.index(own_rate_col)
                    elasticity_ci = (
                        float(np.quantile(coef_array[:, idx], 0.025)),
                        float(np.quantile(coef_array[:, idx], 0.975))
                    )
            else:
                logger.warning(
                    "Model has estimators_ but no coef_ attributes - "
                    "cannot compute bootstrap CIs"
                )
        else:
            logger.warning(
                "Model does not have bootstrap estimators (estimators_ attribute). "
                "Cannot compute confidence intervals. Consider using BaggingRegressor."
            )

        # Apply legacy output mapping for downstream compatibility
        # Feature Naming Unification (2026-01-26): Internal uses _t0, output uses _current
        legacy_coefficients = _remap_to_legacy_names(coefficients)
        legacy_confidence_intervals = {
            level: _remap_to_legacy_names(ci_dict)
            for level, ci_dict in confidence_intervals.items()
        }

        return {
            "coefficients": legacy_coefficients,
            "confidence_intervals": legacy_confidence_intervals,
            "elasticity_point": elasticity_point,
            "elasticity_ci": elasticity_ci,
            "model_fit": model_fit,
            "n_observations": len(data) if len(data) > 0 else 0,
            "diagnostics_summary": diagnostics_summary,
        }

    def _get_inference_features(
        self, data: pd.DataFrame, config: InferenceConfig
    ) -> list:
        """Get features for inference from config or auto-detect.

        Auto-detection includes:
        1. Own-rate features (prudential_rate_t0, prudential_rate_t*) - MUST include
        2. Competitor lag features (NOT lag-0 to avoid leakage)

        Feature Naming (2026-01-26): Uses unified _t{N} format internally.

        Returns
        -------
        list
            Feature list with own-rate + competitors (max 5 total)
        """
        # Check if features specified in config
        if config.get("features"):
            return config["features"]

        # Own-rate detection: handle multiple naming conventions
        # - Unified: prudential_rate_t0, prudential_rate_t1, etc.
        # - Legacy: prudential_rate_current (backward compat)
        # - FIA/MYGA: may use different prefixes
        own_rate_patterns = [
            "prudential_rate",  # RILA standard
            "prudential.rate",  # Dot notation
            f"{self._product.own_rate_prefix}_rate",  # Short prefix (P_rate)
            f"{self._product.own_rate_prefix.lower()}_rate",  # Lowercase
        ]

        own_rate_features = []
        for col in data.columns:
            col_lower = col.lower()
            # Match any of the own-rate patterns
            if any(pat.lower() in col_lower for pat in own_rate_patterns):
                # Exclude derived/interaction features (too complex for auto-selection)
                if "derived" not in col_lower and "interaction" not in col_lower:
                    own_rate_features.append(col)
                    if len(own_rate_features) >= 2:
                        break

        # Competitor lag features (NOT lag-0 to avoid leakage)
        # Note: _t0 is current period, so excluded alongside _current for backward compat
        competitor_features = [
            col for col in data.columns
            if "competitor" in col.lower()
            and "_t" in col.lower()  # Has lag suffix
            and "_t0" not in col.lower()  # Not lag-0 (current period)
            and "_current" not in col.lower()  # Legacy current also excluded
        ][:3]  # Limit to 3 competitor features

        features = own_rate_features + competitor_features

        # Fallback defaults if auto-detection failed
        # Uses unified naming (will be remapped to legacy in output)
        if not features:
            return [
                "prudential_rate_t0",
                "competitor_weighted_t2"
            ]

        return features

    def _extract_model_coefficients(
        self, model, features: list
    ) -> Dict[str, float]:
        """Extract average coefficients from bagging ensemble.

        For BaggingRegressor, averages coefficients across all estimators.
        Falls back to direct coef_ access for single estimators.

        Parameters
        ----------
        model : sklearn estimator
            Trained model (BaggingRegressor or linear model)
        features : list
            Feature names corresponding to coefficient order

        Returns
        -------
        Dict[str, float]
            Feature name to coefficient mapping
        """
        import numpy as np

        # BaggingRegressor stores base estimators
        if hasattr(model, 'estimators_'):
            all_coefs = []
            for estimator in model.estimators_:
                if hasattr(estimator, 'coef_'):
                    all_coefs.append(estimator.coef_)

            if all_coefs:
                avg_coefs = np.mean(all_coefs, axis=0)
                return dict(zip(features, avg_coefs.tolist()))

        # Fallback for single models
        if hasattr(model, 'coef_'):
            return dict(zip(features, model.coef_.tolist()))

        return {}

    def _calculate_model_fit(
        self,
        data: pd.DataFrame,
        model,
        features: list,
        target_variable: str,
    ) -> Dict[str, Any]:
        """Calculate model fit metrics on BOTH raw and log scales.

        The model is trained on log1p(y), so predictions are in log scale.
        This method reports metrics on both scales for transparency.

        Decision: Report BOTH scales. No hidden transform assumptions -
        RILA/FIA/MYGA may differ in their transform requirements.

        Parameters
        ----------
        data : pd.DataFrame
            Data containing features and target
        model : sklearn estimator
            Trained model with predict() method
        features : list
            Feature column names
        target_variable : str
            Target column name

        Returns
        -------
        Dict[str, Any]
            Dictionary with raw and log scale metrics:
            - r_squared_raw, mae_raw, mape_raw (interpretable units)
            - r_squared_log, mae_log (model's native scale)
            - n_samples, transform_used, note
        """
        import logging
        import numpy as np
        from sklearn.metrics import r2_score, mean_absolute_error

        logger = logging.getLogger(__name__)

        try:
            X = data[features].dropna()
            y_raw = data.loc[X.index, target_variable].values

            # Model predictions are in log scale (model trained on log1p(y))
            y_pred_log = model.predict(X)

            # Inverse transform to get raw scale predictions
            y_pred_raw = np.expm1(y_pred_log)

            # Compute y in log scale for log metrics
            y_log = np.log1p(y_raw)

            # Raw scale metrics (interpretable units)
            r2_raw = r2_score(y_raw, y_pred_raw)
            mae_raw = mean_absolute_error(y_raw, y_pred_raw)

            # MAPE with protection against division by zero
            nonzero_mask = y_raw != 0
            if nonzero_mask.sum() > 0:
                mape_raw = float(
                    np.mean(
                        np.abs((y_raw[nonzero_mask] - y_pred_raw[nonzero_mask])
                               / y_raw[nonzero_mask])
                    ) * 100
                )
            else:
                mape_raw = float('nan')

            # Log scale metrics (model's native scale)
            r2_log = r2_score(y_log, y_pred_log)
            mae_log = mean_absolute_error(y_log, y_pred_log)

            return {
                # Raw scale metrics (interpretable units)
                "r_squared_raw": float(r2_raw),
                "mae_raw": float(mae_raw),
                "mape_raw": mape_raw,
                # Log scale metrics (model's native scale)
                "r_squared_log": float(r2_log),
                "mae_log": float(mae_log),
                # Metadata
                "n_samples": len(y_raw),
                "transform_used": "log1p",
                "note": (
                    "Raw metrics are in original units; "
                    "log metrics match training scale"
                ),
                # Backward compatibility: keep legacy keys
                "r_squared": float(r2_log),  # Legacy: log scale
                "mae": float(mae_log),       # Legacy: log scale
            }
        except (ValueError, KeyError, AttributeError) as e:
            logger.warning(f"Could not calculate model fit: {e}")
            return {
                "r_squared_raw": 0.0,
                "mae_raw": 0.0,
                "mape_raw": 0.0,
                "r_squared_log": 0.0,
                "mae_log": 0.0,
                "n_samples": 0,
                "transform_used": "log1p",
                "note": f"Calculation failed: {e}",
                "r_squared": 0.0,
                "mae": 0.0,
            }

    def _is_competitor_lag_zero(self, feature_name: str) -> bool:
        """Check if feature is a simultaneous competitor (forbidden for causal ID).

        Handles both naming conventions:
            - RILA: competitor_mid_t0, competitor_current
            - FIA: comp_mean with zero lag, competitor with zero lag

        See knowledge/domain/FIA_FEATURE_MAPPING.md for pattern details.

        Parameters
        ----------
        feature_name : str
            Feature name to check

        Returns
        -------
        bool
            True if feature is a simultaneous competitor feature
        """
        f_lower = feature_name.lower()

        # Identify competitor-related features
        is_competitor = (
            "competitor" in f_lower
            or (f_lower.startswith("comp_") and "comply" not in f_lower)
            or (f_lower.startswith("c_") and "rate" in f_lower)
        )

        if not is_competitor:
            return False

        # Check for lag-0 patterns
        # RILA: _t0, _current
        # FIA: _lag_0
        is_lag_zero = "_t0" in f_lower or "_current" in f_lower or "_lag_0" in f_lower

        return is_lag_zero

    def _get_default_inference_config(self) -> InferenceConfig:
        """Get default inference configuration for product.

        Note
        ----
        Feature Naming Unification (2026-01-26): Uses unified _t0 naming.
        Legacy names remapped in output only.
        """
        return {
            "product_code": self._product.product_code,
            "product_type": self._product.product_type,
            "own_rate_column": f"{self._product.own_rate_prefix}_rate_t0",
            "competitor_rate_column": "competitor_weighted_t2",
            "target_column": "sales_target_t0",
            "rate_adjustment_range": (-300, 300),
            "n_bootstrap": 1000,
            "confidence_levels": [0.80, 0.90, 0.95],
        }

    def run_forecasting(
        self,
        data: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> "ForecastingResults":
        """Run time series forecasting pipeline.

        Corresponds to notebook 02_time_series_forecasting_refactored.
        Executes benchmark comparison and bootstrap Ridge forecasting.

        Parameters
        ----------
        data : Optional[pd.DataFrame]
            Data for forecasting (uses loaded data if not provided)
        config : Optional[Dict[str, Any]]
            Forecasting configuration overrides. Keys:
            - bootstrap_samples: int (default: 1000)
            - ridge_alpha: float (default: 1.0)
            - start_cutoff: int (default: 30)
            - end_cutoff: Optional[int] (default: None = full dataset)

        Returns
        -------
        ForecastingResults
            Forecasting results with benchmark comparison

        Raises
        ------
        ValueError
            If no data available
        RuntimeError
            If forecasting pipeline fails

        Examples
        --------
        >>> interface = create_interface("6Y20B", environment="fixture")
        >>> df = interface.load_data()
        >>> results = interface.run_forecasting(df)
        >>> print(f"MAPE Improvement: {results.mape_improvement:.1f}%")
        """
        from src.models.forecasting_types import ForecastingResults
        from src.models.forecasting_orchestrator import run_forecasting_pipeline
        from src.config.forecasting_builders import build_forecasting_stage_config

        if data is None:
            if not self._data_loaded:
                raise ValueError(
                    "No data available. Call load_data() first or provide data."
                )
            data = self._data

        # Build forecasting configuration with overrides
        config = config or {}
        forecasting_stage_config = build_forecasting_stage_config(
            bootstrap_samples=config.get("bootstrap_samples", 1000),
            ridge_alpha=config.get("ridge_alpha", 1.0),
            start_cutoff=config.get("start_cutoff", 30),
            end_cutoff=config.get("end_cutoff", None),
            enable_detailed_validation=config.get("enable_detailed_validation", True),
        )

        # Extract sign correction configs from stage config
        model_sign_correction_config = forecasting_stage_config.get(
            "model_sign_correction_config", {}
        )
        benchmark_sign_correction_config = forecasting_stage_config.get(
            "benchmark_sign_correction_config", {}
        )

        try:
            # Run the forecasting pipeline
            pipeline_output = run_forecasting_pipeline(
                df=data,
                forecasting_config=forecasting_stage_config,
                model_sign_correction_config=model_sign_correction_config,
                benchmark_sign_correction_config=benchmark_sign_correction_config,
            )
        except Exception as e:
            raise RuntimeError(
                f"Forecasting pipeline failed: {e}. "
                f"Check data quality and configuration."
            ) from e

        return ForecastingResults.from_pipeline_output(pipeline_output)

    def _get_default_forecasting_config(self) -> Dict[str, Any]:
        """Get default forecasting configuration for product."""
        return {
            "bootstrap_samples": 1000,
            "ridge_alpha": 1.0,
            "start_cutoff": 30,
            "end_cutoff": None,
            "enable_detailed_validation": True,
        }

    def _validate_methodology_compliance(
        self,
        data: pd.DataFrame,
        features: Optional[list] = None,
    ) -> None:
        """Validate data/features comply with methodology constraints.

        Parameters
        ----------
        data : pd.DataFrame
            Data to validate
        features : list, optional
            Specific features to validate. If provided, only these are checked.
            If None, all columns are checked (warning only for lag-0 presence).

        Raises
        ------
        ValueError
            If features parameter contains lag-0 competitor features
        """
        import logging
        logger = logging.getLogger(__name__)

        rules = self._methodology.get_constraint_rules()

        for rule in rules:
            if rule.constraint_type == "NO_LAG_ZERO_COMPETITOR":
                if features is not None:
                    # Strict check on specified features
                    # Catches both RILA (_t0, _current) and FIA (_lag_0) patterns
                    lag_zero_features = [
                        f for f in features
                        if self._is_competitor_lag_zero(f)
                    ]
                    if lag_zero_features:
                        raise ValueError(
                            f"CRITICAL: Lag-0 competitor features in model: {lag_zero_features}. "
                            f"This violates causal identification. "
                            f"Use lagged features (t-1 or earlier) only."
                        )
                else:
                    # Advisory warning for lag-0 columns in data
                    lag_zero_cols = [
                        c for c in data.columns
                        if self._is_competitor_lag_zero(c)
                    ]
                    if lag_zero_cols:
                        logger.warning(
                            f"Data contains {len(lag_zero_cols)} lag-0 competitor columns. "
                            f"Ensure they are not used in feature selection or inference."
                        )

    def export_results(
        self,
        results: InferenceResults,
        format: str = "excel",
        name: Optional[str] = None,
    ) -> str:
        """Export inference results.

        Parameters
        ----------
        results : InferenceResults
            Results to export
        format : str
            Export format: "excel", "csv", "parquet"
        name : Optional[str]
            Output name (auto-generated if not provided)

        Returns
        -------
        str
            Path where results were exported
        """
        import datetime

        if name is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            name = f"inference_results_{self._product.product_code}_{timestamp}"

        # Convert results to DataFrame for export
        results_df = pd.DataFrame({
            "coefficient": list(results["coefficients"].values()),
            "feature": list(results["coefficients"].keys()),
        })

        return self._adapter.save_output(results_df, name, format)

    def _run_lightweight_diagnostics(
        self,
        model,
        data: pd.DataFrame,
        features: list,
        target_variable: str,
    ) -> Dict[str, Any]:
        """
        Run lightweight diagnostics for auto-integration with run_inference().

        Performs quick checks for common issues:
        - Durbin-Watson test for autocorrelation (warn if < 1.5 or > 2.5)
        - VIF for top 5 features (warn if > 10)

        Parameters
        ----------
        model : sklearn estimator or statsmodels model
            Trained model with predict() method
        data : pd.DataFrame
            Data containing features and target
        features : list
            Feature column names
        target_variable : str
            Target column name

        Returns
        -------
        Dict[str, Any]
            Lightweight diagnostics summary with warnings
        """
        import logging
        import numpy as np
        logger = logging.getLogger(__name__)

        warnings_list = []
        diagnostics = {
            "durbin_watson": None,
            "vif_warnings": [],
            "warnings": warnings_list,
            "production_ready": True,
        }

        try:
            # Get predictions and compute residuals
            X = data[features].dropna()
            y = data.loc[X.index, target_variable]

            if len(X) < 20:
                warnings_list.append(
                    "WARN: Insufficient data for diagnostics (n < 20)"
                )
                return diagnostics

            y_pred = model.predict(X)
            residuals = y - y_pred

            # Durbin-Watson test for autocorrelation
            try:
                from statsmodels.stats.stattools import durbin_watson
                dw_stat = durbin_watson(residuals)
                diagnostics["durbin_watson"] = float(dw_stat)

                if dw_stat < 1.5:
                    warnings_list.append(
                        f"WARN: Positive autocorrelation detected (DW={dw_stat:.3f} < 1.5). "
                        "Standard errors may be biased. Consider Newey-West robust SE."
                    )
                    diagnostics["production_ready"] = False
                elif dw_stat > 2.5:
                    warnings_list.append(
                        f"WARN: Negative autocorrelation detected (DW={dw_stat:.3f} > 2.5). "
                        "Standard errors may be biased."
                    )
                    diagnostics["production_ready"] = False
            except Exception as e:
                logger.warning(f"Durbin-Watson test failed: {e}")

            # VIF for top 5 features (quick multicollinearity check)
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor

                # Limit to top 5 features to keep it lightweight
                check_features = features[:5]
                X_check = X[check_features].copy()
                X_check = X_check.assign(const=1)

                vif_issues = []
                for idx, col in enumerate(check_features):
                    col_idx = X_check.columns.get_loc(col)
                    vif = variance_inflation_factor(X_check.values, col_idx)
                    if vif > 10:
                        vif_issues.append(f"{col} (VIF={vif:.1f})")

                if vif_issues:
                    diagnostics["vif_warnings"] = vif_issues
                    warnings_list.append(
                        f"WARN: High multicollinearity detected for: {', '.join(vif_issues)}. "
                        "Coefficient estimates may be unstable."
                    )
            except Exception as e:
                logger.warning(f"VIF calculation failed: {e}")

        except Exception as e:
            logger.warning(f"Lightweight diagnostics failed: {e}")
            warnings_list.append(f"WARN: Diagnostics incomplete: {e}")

        return diagnostics

    def generate_diagnostic_report(
        self,
        model,
        data: Optional[pd.DataFrame] = None,
        features: Optional[list] = None,
        target_variable: Optional[str] = None,
    ) -> "ComprehensiveDiagnostics":
        """
        Generate comprehensive diagnostic report for model assumption validation.

        Runs the full regression diagnostics suite including:
        - Autocorrelation tests (Durbin-Watson, Ljung-Box)
        - Heteroscedasticity tests (Breusch-Pagan, White)
        - Multicollinearity analysis (VIF)
        - Normality tests (Jarque-Bera, Shapiro-Wilk)

        Parameters
        ----------
        model : statsmodels regression model
            Fitted model with .resid attribute (from statsmodels)
        data : Optional[pd.DataFrame]
            Data for diagnostics (uses loaded data if not provided)
        features : Optional[list]
            Feature list for the model
        target_variable : Optional[str]
            Target variable name

        Returns
        -------
        ComprehensiveDiagnostics
            Full diagnostic report with remediation plan

        Raises
        ------
        ValueError
            If no data available or model lacks required attributes

        Examples
        --------
        >>> # Fit a statsmodels model first
        >>> import statsmodels.formula.api as smf
        >>> formula = f"{target} ~ {' + '.join(features)}"
        >>> sm_model = smf.ols(formula, data=df).fit()
        >>> # Generate report
        >>> report = interface.generate_diagnostic_report(sm_model, df, features, target)
        >>> if not report.overall_assessment['production_ready']:
        ...     print("Remediation needed:")
        ...     for action in report.remediation_plan:
        ...         print(f"  - {action}")
        """
        from src.features.selection.support.regression_diagnostics import (
            comprehensive_diagnostic_suite,
            ComprehensiveDiagnostics,
        )

        if data is None:
            if not self._data_loaded:
                raise ValueError(
                    "No data available. Call load_data() first or provide data."
                )
            data = self._data

        if target_variable is None:
            target_variable = "sales_target_t0"

        if features is None:
            # Try to get features from model
            if hasattr(model, 'params'):
                features = [f for f in model.params.index if f != 'Intercept']
            else:
                raise ValueError(
                    "Features not specified and cannot be inferred from model. "
                    "Provide features parameter."
                )

        # Validate model has required attributes for comprehensive diagnostics
        if not hasattr(model, 'resid'):
            raise ValueError(
                "Model must be a statsmodels OLS model with .resid attribute. "
                "Use statsmodels.formula.api.ols() to fit the model for diagnostics."
            )

        return comprehensive_diagnostic_suite(
            model=model,
            data=data,
            target_variable=target_variable,
            features=features,
        )

    def get_constraint_rules(self) -> list:
        """Get economic constraint rules for the product type."""
        return self._methodology.get_constraint_rules()

    def get_coefficient_signs(self) -> Dict[str, str]:
        """Get expected coefficient signs."""
        return self._methodology.get_coefficient_signs()

    def validate_coefficients(
        self, coefficients: Dict[str, float]
    ) -> Dict[str, Any]:
        """Validate model coefficients against economic constraints.

        Uses unified regex patterns to avoid false positives from substring
        matching with P_ and C_ patterns.

        Parameters
        ----------
        coefficients : Dict[str, float]
            Model coefficients by feature name

        Returns
        -------
        Dict[str, Any]
            Validation results with passes/violations/warnings

        See Also
        --------
        src.validation.coefficient_patterns : Pattern registry and validation
        """
        from src.validation.coefficient_patterns import validate_all_coefficients

        return validate_all_coefficients(
            coefficients,
            product_type=self._product.product_type.upper()
        )


# Convenience factory functions


def create_interface(
    product_code: str = "6Y20B",
    environment: str = "aws",
    **kwargs,
) -> UnifiedNotebookInterface:
    """Create a UnifiedNotebookInterface with sensible defaults.

    Parameters
    ----------
    product_code : str
        Product code (default: "6Y20B")
    environment : str
        Environment: "aws", "local", "fixture", "test"
    **kwargs
        Additional arguments passed to interface

    Returns
    -------
    UnifiedNotebookInterface
        Configured interface
    """
    if environment == "test":
        environment = "fixture"
        if "adapter_kwargs" not in kwargs:
            kwargs["adapter_kwargs"] = {}
        if "fixtures_dir" not in kwargs["adapter_kwargs"]:
            kwargs["adapter_kwargs"]["fixtures_dir"] = Path("tests/fixtures/rila")

    return UnifiedNotebookInterface(
        product_code=product_code,
        data_source=environment,
        **kwargs,
    )


__all__ = [
    "UnifiedNotebookInterface",
    "create_interface",
]

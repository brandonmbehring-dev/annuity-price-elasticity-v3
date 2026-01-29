"""
Core Protocols for Annuity Price Elasticity v2.

Defines Protocol interfaces for dependency injection patterns.
Enables clean separation between business logic and data sources.

Architecture Pattern: Dependency Injection
    NotebookInterface(adapter=S3Adapter(config))
    NotebookInterface(adapter=LocalAdapter(data_dir))
    NotebookInterface(adapter=FixtureAdapter(fixtures_dir))

Usage:
    from src.core.protocols import DataSourceAdapter, AggregationStrategy
"""

from typing import Protocol, Optional, List, Dict, Any, runtime_checkable
from pathlib import Path
import pandas as pd

from src.core.types import (
    AWSConfig,
    InferenceConfig,
    FeatureConfig,
    InferenceResults,
)
from src.features.selection_types import FeatureSelectionResults


# =============================================================================
# DATA SOURCE ADAPTER PROTOCOL
# =============================================================================


@runtime_checkable
class DataSourceAdapter(Protocol):
    """Protocol for data source abstraction (Dependency Injection pattern).

    This protocol defines the interface that all data source implementations
    must follow. Enables seamless switching between AWS, local, and fixture
    data sources without changing business logic.

    Implementations:
        - S3Adapter: Production AWS S3 data access
        - LocalAdapter: Local filesystem for development
        - FixtureAdapter: Test fixtures for validation

    Examples
    --------
    >>> # Production usage
    >>> adapter = S3Adapter(config)
    >>> interface = NotebookInterface(adapter=adapter)
    >>> df = interface.load_data()

    >>> # Test usage
    >>> adapter = FixtureAdapter(fixtures_path)
    >>> interface = NotebookInterface(adapter=adapter)
    >>> df = interface.load_data()
    """

    def load_sales_data(
        self, product_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """Load sales data from the data source.

        Parameters
        ----------
        product_filter : Optional[str]
            Filter by product name (e.g., "FlexGuard 6Y20B")

        Returns
        -------
        pd.DataFrame
            Sales data with required columns

        Raises
        ------
        DataLoadError
            If data loading fails
        """
        ...

    def load_competitive_rates(
        self, start_date: str
    ) -> pd.DataFrame:
        """Load competitive rate data (WINK).

        Parameters
        ----------
        start_date : str
            Start date for rate data (ISO format: YYYY-MM-DD)

        Returns
        -------
        pd.DataFrame
            Competitive rate data with company columns
        """
        ...

    def load_market_weights(self) -> pd.DataFrame:
        """Load market share weights for competitor aggregation.

        Returns
        -------
        pd.DataFrame
            Market weights by company and time period
        """
        ...

    def load_macro_data(self) -> pd.DataFrame:
        """Load macroeconomic indicator data.

        Returns
        -------
        pd.DataFrame
            Macro indicators (interest rates, indices, etc.)
        """
        ...

    def save_output(
        self, df: pd.DataFrame, name: str, format: str = "parquet"
    ) -> str:
        """Save output to the data destination.

        Parameters
        ----------
        df : pd.DataFrame
            Data to save
        name : str
            Output name/identifier
        format : str
            Output format: "parquet", "csv", or "excel"

        Returns
        -------
        str
            Path/URI where data was saved
        """
        ...

    @property
    def source_type(self) -> str:
        """Return the adapter type identifier.

        Returns
        -------
        str
            One of: "aws", "local", "fixture"
        """
        ...


# =============================================================================
# AGGREGATION STRATEGY PROTOCOL
# =============================================================================


@runtime_checkable
class AggregationStrategy(Protocol):
    """Protocol for competitor rate aggregation strategies.

    Different products require different aggregation approaches:
    - RILA: Market-share weighted means
    - FIA: Simple top-N competitor means
    - MYGA: Firm-level aggregation

    Implementations:
        - WeightedAggregation: Market-share weighted (RILA default)
        - TopNAggregation: Top N competitors by rate
        - FirmLevelAggregation: Firm-specific calculations

    Examples
    --------
    >>> strategy = WeightedAggregation(weights_df)
    >>> competitor_rate = strategy.aggregate(rates_df, company_columns)
    """

    def aggregate(
        self,
        rates_df: pd.DataFrame,
        company_columns: List[str],
        weights_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Aggregate competitor rates using the strategy.

        Parameters
        ----------
        rates_df : pd.DataFrame
            DataFrame containing company rate columns
        company_columns : List[str]
            List of company column names to aggregate
        weights_df : Optional[pd.DataFrame]
            Market weights (required for weighted aggregation)

        Returns
        -------
        pd.Series
            Aggregated competitor rate series
        """
        ...

    @property
    def requires_weights(self) -> bool:
        """Whether this strategy requires market weights.

        Returns
        -------
        bool
            True if weights_df must be provided
        """
        ...

    @property
    def strategy_name(self) -> str:
        """Return the strategy identifier.

        Returns
        -------
        str
            Strategy name: "weighted", "top_n", "firm_level"
        """
        ...


# =============================================================================
# PRODUCT METHODOLOGY PROTOCOL
# =============================================================================


@runtime_checkable
class ProductMethodology(Protocol):
    """Protocol for product-specific methodology rules.

    Defines economic constraint rules and coefficient expectations
    for different product types (RILA, FIA, MYGA).

    Adapted from src/products/base.py in v1.

    Examples
    --------
    >>> methodology = RILAMethodology()
    >>> rules = methodology.get_constraint_rules()
    >>> signs = methodology.get_coefficient_signs()
    """

    def get_constraint_rules(self) -> List[Dict[str, Any]]:
        """Get economic constraint rules for this product type.

        Returns
        -------
        List[Dict[str, Any]]
            List of constraint rule definitions containing:
            - feature_pattern: str (regex pattern)
            - expected_sign: str ("positive" or "negative")
            - constraint_type: str (identifier)
            - business_rationale: str (explanation)
            - strict: bool (fail vs warn)
        """
        ...

    def get_coefficient_signs(self) -> Dict[str, str]:
        """Get expected coefficient signs by feature pattern.

        Returns
        -------
        Dict[str, str]
            Mapping of feature pattern to expected sign
        """
        ...

    def supports_regime_detection(self) -> bool:
        """Check if product type supports regime detection.

        Returns
        -------
        bool
            True if regime detection is appropriate
        """
        ...

    @property
    def product_type(self) -> str:
        """Return the product type identifier.

        Returns
        -------
        str
            Product type: "rila", "fia", or "myga"
        """
        ...


# =============================================================================
# NOTEBOOK INTERFACE PROTOCOL
# =============================================================================


@runtime_checkable
class NotebookInterfaceProtocol(Protocol):
    """Protocol for unified notebook interface.

    Provides consistent API for all notebook operations across
    product types. Uses dependency injection for data sources.

    Examples
    --------
    >>> interface = UnifiedNotebookInterface("6Y20B", adapter=adapter)
    >>> df = interface.load_data()
    >>> results = interface.run_inference(df, model_config)
    """

    def load_data(self) -> pd.DataFrame:
        """Load and prepare all required data.

        Returns
        -------
        pd.DataFrame
            Merged, cleaned dataset ready for analysis
        """
        ...

    def run_feature_selection(
        self, data: pd.DataFrame, config: Optional[FeatureConfig] = None
    ) -> FeatureSelectionResults:
        """Run feature selection algorithm.

        Parameters
        ----------
        data : pd.DataFrame
            Prepared data for feature selection
        config : Optional[FeatureConfig]
            Feature selection configuration

        Returns
        -------
        FeatureSelectionResults
            Selected features and validation results
        """
        ...

    def run_inference(
        self, data: pd.DataFrame, config: Optional[InferenceConfig] = None
    ) -> InferenceResults:
        """Run price elasticity inference.

        Parameters
        ----------
        data : pd.DataFrame
            Prepared data for inference
        config : Optional[InferenceConfig]
            Inference configuration

        Returns
        -------
        InferenceResults
            Elasticity estimates and confidence intervals
        """
        ...

    def export_results(
        self, results: InferenceResults, format: str = "excel"
    ) -> str:
        """Export inference results.

        Parameters
        ----------
        results : InferenceResults
            Results to export
        format : str
            Export format: "excel", "csv", "parquet"

        Returns
        -------
        str
            Path where results were exported
        """
        ...


__all__ = [
    "DataSourceAdapter",
    "AggregationStrategy",
    "ProductMethodology",
    "NotebookInterfaceProtocol",
]

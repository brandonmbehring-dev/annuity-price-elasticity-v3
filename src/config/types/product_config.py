"""
RILA Product Configuration

Multi-product abstraction layer for RILA/FIA/MYGA price elasticity modeling.
Pattern adapted from FIA price elasticity project.

Supports:
- RILA products (buffer_level required)
- FIA products (buffer_level = None, yield-based)
- MYGA products (buffer_level = None, fixed rate)

Usage:
    from src.config.product_config import get_product_config, PRODUCT_REGISTRY

    # Get product config
    product = get_product_config("6Y20B")
    buffer = product.buffer_level  # 0.20

    # Get default product
    default = get_default_product()  # Returns 6Y20B

    # Get product IDs for WINK processing
    from src.config.product_config import get_wink_product_ids
    ids = get_wink_product_ids()
    pipeline_ids = ids.pipeline_ids  # For WINK rate processing (current)
    metadata_ids = ids.metadata_ids  # For inference metadata (historical)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# =============================================================================
# WINK PRODUCT ID MAPPINGS (Single Source of Truth)
# =============================================================================


@dataclass(frozen=True)
class WinkProductIds:
    """WINK database product ID mappings.

    Both pipeline_ids and metadata_ids are valid for different contexts:
    - pipeline_ids: For WINK rate processing (current product IDs)
    - metadata_ids: For inference metadata (historical product IDs)

    This consolidation ensures there is a single source of truth for all
    product ID mappings, eliminating duplicate definitions.

    Attributes
    ----------
    pipeline_ids : Dict[str, Tuple[int, ...]]
        Company to product ID mapping for pipeline processing (current WINK data)
    metadata_ids : Dict[str, Tuple[int, ...]]
        Company to product ID mapping for inference metadata (historical data)

    Examples
    --------
    >>> ids = get_wink_product_ids()
    >>> ids.pipeline_ids['Prudential']
    (2979,)
    >>> ids.metadata_ids['Athene']
    (2772, 3409)
    """
    pipeline_ids: Dict[str, Tuple[int, ...]]
    metadata_ids: Dict[str, Tuple[int, ...]]


# Canonical WINK product ID definitions
_WINK_PIPELINE_IDS: Dict[str, Tuple[int, ...]] = {
    "Prudential": (2979,),
    "Allianz": (2162, 3699),
    "Athene": (3409,),
    "Brighthouse": (2319, 4149),
    "Equitable": (3282,),
    "Jackson": (3351, 4491),
    "Lincoln": (2358, 4058),
    "Symetra": (3263,),
    "Trans": (3495,),
}

_WINK_METADATA_IDS: Dict[str, Tuple[int, ...]] = {
    "Prudential": (2979,),
    "Allianz": (2162, 3699),
    "Athene": (2772, 3409),
    "Brighthouse": (2319,),
    "Equitable": (2286, 3282, 3853),
    "Jackson": (3714, 3351),
    "Lincoln": (2924,),
    "Symetra": (3263, 3751),
    "Trans": (3495,),
}


def get_wink_product_ids() -> WinkProductIds:
    """Get canonical WINK product ID mappings.

    Returns
    -------
    WinkProductIds
        Product ID mappings for both pipeline and metadata contexts

    Examples
    --------
    >>> ids = get_wink_product_ids()
    >>> ids.pipeline_ids['Prudential']
    (2979,)
    """
    return WinkProductIds(
        pipeline_ids=_WINK_PIPELINE_IDS,
        metadata_ids=_WINK_METADATA_IDS
    )


def get_pipeline_product_ids_as_lists() -> Dict[str, list]:
    """Get pipeline product IDs as lists (backward compatibility).

    Returns dict with list values for compatibility with existing code
    that expects List[int] instead of Tuple[int, ...].

    Returns
    -------
    Dict[str, List[int]]
        Pipeline product IDs with list values
    """
    return {k: list(v) for k, v in _WINK_PIPELINE_IDS.items()}


def get_metadata_product_ids_as_lists() -> Dict[str, list]:
    """Get metadata product IDs as lists (backward compatibility).

    Returns dict with list values for compatibility with existing code
    that expects List[int] instead of Tuple[int, ...].

    Returns
    -------
    Dict[str, List[int]]
        Metadata product IDs with list values
    """
    return {k: list(v) for k, v in _WINK_METADATA_IDS.items()}


# =============================================================================
# PRODUCT DATE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ProductDateConfig:
    """Product-specific date configuration for analysis windows.

    These date ranges define the temporal boundaries for different
    stages of the analysis pipeline.

    Attributes
    ----------
    rate_analysis_start_date : str
        Start date for WINK rate processing (ISO format: YYYY-MM-DD)
    analysis_start_date : str
        Start date for data integration analysis
    feature_analysis_start_date : str
        Start date for feature engineering and modeling
    data_filter_start_date : str
        Start date for raw data filtering

    Examples
    --------
    >>> config = ProductDateConfig()
    >>> config.rate_analysis_start_date
    '2018-06-21'
    >>> config.feature_analysis_start_date
    '2022-01-01'
    """
    rate_analysis_start_date: str = "2018-06-21"
    analysis_start_date: str = "2021-01-01"
    feature_analysis_start_date: str = "2022-01-01"
    data_filter_start_date: str = "2018-01-01"


def get_default_date_config() -> ProductDateConfig:
    """Get default date configuration for RILA analysis.

    Returns
    -------
    ProductDateConfig
        Default date configuration

    Examples
    --------
    >>> config = get_default_date_config()
    >>> config.rate_analysis_start_date
    '2018-06-21'
    """
    return ProductDateConfig()


# =============================================================================
# PRODUCT FEATURE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ProductFeatureConfig:
    """Product-specific feature configuration for modeling.

    Defines the base and candidate features for each product type.
    Base features are always included; candidate features are evaluated
    by feature selection algorithms.

    Feature Naming Unification (2026-01-26):
    - All temporal suffixes use _t{N} format: _t0, _t1, _t2, etc.
    - Previous _current suffix normalized to _t0
    - competitor_mid renamed to competitor_weighted for semantic clarity

    Attributes
    ----------
    base_features : Tuple[str, ...]
        Features always included in models (e.g., own rate)
    candidate_features : Tuple[str, ...]
        Features to evaluate for selection
    target_variable : str
        Default target variable for modeling
    analysis_start_date : str
        Default start date for feature analysis

    Examples
    --------
    >>> config = get_default_feature_config()
    >>> 'prudential_rate_t0' in config.base_features
    True
    """
    base_features: Tuple[str, ...] = ("prudential_rate_t0",)
    candidate_features: Tuple[str, ...] = (
        # Competitor rate features with time lags (competitor_weighted = weighted mean)
        "competitor_weighted_t2",
        "competitor_weighted_t3",
        "competitor_weighted_t4",
        "competitor_weighted_t5",
        "competitor_top5_t2",
        "competitor_top5_t3",
        "competitor_top5_t4",
        "competitor_top5_t5",
        # Prudential rate features (t0 = current period)
        "prudential_rate_t0",
        "prudential_rate_t1",
        "prudential_rate_t2",
        "prudential_rate_t3",
    )
    target_variable: str = "sales_target_t0"
    analysis_start_date: str = "2022-08-01"


def get_default_feature_config() -> ProductFeatureConfig:
    """Get default feature configuration for RILA analysis.

    Returns
    -------
    ProductFeatureConfig
        Default feature configuration

    Examples
    --------
    >>> config = get_default_feature_config()
    >>> len(config.candidate_features)
    12
    """
    return ProductFeatureConfig()


def get_feature_config_for_product_type(product_type: str) -> ProductFeatureConfig:
    """Get feature configuration appropriate for a product type.

    Currently all product types use the same default features.
    This function exists for future extensibility when different
    product types may need different feature sets.

    Parameters
    ----------
    product_type : str
        Product type: "rila", "fia", or "myga"

    Returns
    -------
    ProductFeatureConfig
        Feature configuration for the product type

    Examples
    --------
    >>> config = get_feature_config_for_product_type("rila")
    >>> 'prudential_rate_t0' in config.base_features
    True
    """
    # All product types currently use default features
    # Extend here when FIA/MYGA need different features
    return ProductFeatureConfig()


# =============================================================================
# PRODUCT CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ProductConfig:
    """Immutable configuration for a RILA/FIA/MYGA product variant.

    This dataclass defines product-specific parameters for price elasticity
    analysis. Supports multiple product types with appropriate validation.

    Product Types:
    - rila: Registered Index-Linked Annuity (requires buffer_level)
    - fia: Fixed Index Annuity (buffer_level = None, yield-based)
    - myga: Multi-Year Guaranteed Annuity (buffer_level = None, fixed rate)

    Attributes
    ----------
    name : str
        Human-readable product name (e.g., "FlexGuard 6Y20B")
    product_code : str
        Short identifier used in data and exports (e.g., "6Y20B")
    product_type : str
        Product category: "rila", "fia", or "myga"
    rate_column : str
        Column name for rate data in WINK dataset
    own_rate_prefix : str
        Prefix for Prudential rate features (P = Prudential)
    competitor_rate_prefix : str
        Prefix for competitor rate features (C = Competitor)
    buffer_level : Optional[float]
        Buffer percentage as decimal (0.20 = 20%). Required for RILA,
        None for FIA/MYGA products.
    term_years : int
        Product term in years
    primary_index : str
        Primary index for product (e.g., "SP500")
    max_lag : int
        Maximum lag periods for feature engineering
    competitor_count : int
        Number of competitors to include in analysis

    Examples
    --------
    >>> # RILA product (requires buffer)
    >>> config = ProductConfig(
    ...     name="FlexGuard 6Y20B",
    ...     product_code="6Y20B",
    ...     buffer_level=0.20,
    ...     term_years=6
    ... )
    >>> config.buffer_level
    0.2

    >>> # FIA product (no buffer)
    >>> config = ProductConfig(
    ...     name="FIA Example",
    ...     product_code="FIA_EX",
    ...     product_type="fia",
    ...     buffer_level=None,
    ...     term_years=6
    ... )
    >>> config.buffer_level is None
    True
    """

    # Product identification
    name: str
    product_code: str
    product_type: str = "rila"

    # Rate configuration
    rate_column: str = "capRate"
    own_rate_prefix: str = "P"
    own_rate_column: str = "Prudential"  # Column name in WINK rate data
    competitor_rate_prefix: str = "C"

    # Product-specific parameters (buffer_level optional for FIA/MYGA)
    buffer_level: Optional[float] = 0.20
    term_years: int = 6
    primary_index: str = "SP500"

    # Feature engineering defaults
    max_lag: int = 8
    competitor_count: int = 7

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises
        ------
        ValueError
            If validation fails for product type constraints
        """
        # Validate product_type
        valid_types = {"rila", "fia", "myga"}
        if self.product_type not in valid_types:
            raise ValueError(
                f"product_type must be one of {valid_types}: {self.product_type}"
            )

        # Validate buffer_level based on product type
        if self.product_type == "rila":
            if self.buffer_level is None:
                raise ValueError(
                    "buffer_level is required for RILA products"
                )
            if not 0 < self.buffer_level <= 1:
                raise ValueError(
                    f"buffer_level must be in (0, 1] for RILA: {self.buffer_level}"
                )
        elif self.buffer_level is not None:
            # FIA/MYGA should have None buffer_level, but allow it for flexibility
            # with a warning-level validation (no raise)
            pass

        # Validate term_years (applies to all product types)
        if self.term_years < 1:
            raise ValueError(
                f"term_years must be positive: {self.term_years}"
            )


# =============================================================================
# PRODUCT REGISTRY - Add new products here (single source of truth)
# =============================================================================

PRODUCT_REGISTRY: Dict[str, ProductConfig] = {
    # ==========================================================================
    # RILA Products (Registered Index-Linked Annuities)
    # ==========================================================================
    "6Y20B": ProductConfig(
        name="FlexGuard 6Y20B",
        product_code="6Y20B",
        buffer_level=0.20,
        term_years=6,
    ),
    "6Y10B": ProductConfig(
        name="FlexGuard 6Y10B",
        product_code="6Y10B",
        buffer_level=0.10,
        term_years=6,
    ),
    "10Y20B": ProductConfig(
        name="FlexGuard 10Y20B",
        product_code="10Y20B",
        buffer_level=0.20,
        term_years=10,
    ),
    # ==========================================================================
    # FIA Products (Fixed Indexed Annuities) - Added 2026-01-24
    # ==========================================================================
    "FIA5YR": ProductConfig(
        name="PruSecure FIA 5-Year",
        product_code="FIA5YR",
        product_type="fia",
        buffer_level=None,  # FIA uses participation rates, not buffers
        term_years=5,
        own_rate_column="Prudential",
    ),
    "FIA7YR": ProductConfig(
        name="PruSecure FIA 7-Year",
        product_code="FIA7YR",
        product_type="fia",
        buffer_level=None,
        term_years=7,
        own_rate_column="Prudential",
    ),
    "FIACA5YR": ProductConfig(
        name="PruSecure FIA Cap 5-Year",
        product_code="FIACA5YR",
        product_type="fia",
        buffer_level=None,
        term_years=5,
        own_rate_column="Prudential",
    ),
    "FIACA7YR": ProductConfig(
        name="PruSecure FIA Cap 7-Year",
        product_code="FIACA7YR",
        product_type="fia",
        buffer_level=None,
        term_years=7,
        own_rate_column="Prudential",
    ),
    # Future products: add here
    # "1Y10B": ProductConfig(...),
    "1Y10B": ProductConfig(
        name="FlexGuard 1Y10B",
        product_code="1Y10B",
        buffer_level=0.10,
        term_years=1,
    ),
}


def get_product_config(product_code: str) -> ProductConfig:
    """Get configuration for a product by code.

    Parameters
    ----------
    product_code : str
        Product identifier (e.g., "6Y20B", "6Y10B", "10Y20B")

    Returns
    -------
    ProductConfig
        Configuration for the specified product

    Raises
    ------
    KeyError
        If product_code is not in PRODUCT_REGISTRY

    Examples
    --------
    >>> config = get_product_config("6Y20B")
    >>> config.buffer_level
    0.2
    >>> config.term_years
    6
    """
    if product_code not in PRODUCT_REGISTRY:
        available = ", ".join(PRODUCT_REGISTRY.keys())
        raise KeyError(f"Unknown product: {product_code}. Available: {available}")
    return PRODUCT_REGISTRY[product_code]


def get_default_product() -> ProductConfig:
    """Get the default product configuration (6Y20B).

    Returns
    -------
    ProductConfig
        Configuration for the default product (6Y20B)

    Examples
    --------
    >>> config = get_default_product()
    >>> config.product_code
    '6Y20B'
    """
    return PRODUCT_REGISTRY["6Y20B"]


# =============================================================================
# COMPETITOR CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class CompetitorConfig:
    """Competitor company configuration for RILA analysis.

    Centralizes all competitor-related constants to avoid hardcoding
    company names throughout the codebase.

    Attributes
    ----------
    rila_competitors : Tuple[str, ...]
        Full list of RILA competitor companies for weighted mean calculation
    core_competitors : Tuple[str, ...]
        Core competitors for C_core calculation (3-company average)
    own_company : str
        Own company column name in rate data

    Examples
    --------
    >>> config = get_competitor_config()
    >>> 'Allianz' in config.rila_competitors
    True
    >>> config.own_company
    'Prudential'
    """
    rila_competitors: Tuple[str, ...] = (
        "Allianz",
        "Athene",
        "Brighthouse",
        "Equitable",
        "Jackson",
        "Lincoln",
        "Symetra",
        "Trans",
    )
    core_competitors: Tuple[str, ...] = (
        "Brighthouse",
        "Equitable",
        "Lincoln",
    )
    own_company: str = "Prudential"


def get_competitor_config() -> CompetitorConfig:
    """Get competitor configuration for RILA analysis.

    Returns
    -------
    CompetitorConfig
        Competitor configuration with company lists

    Examples
    --------
    >>> config = get_competitor_config()
    >>> len(config.rila_competitors)
    8
    """
    return CompetitorConfig()


# =============================================================================
# DEFAULT CONSTANTS (avoid magic numbers in code)
# =============================================================================


# Model training defaults
DEFAULT_MAX_FEATURES = 3
DEFAULT_BOOTSTRAP_SAMPLES = 100
DEFAULT_CV_FOLDS = 5
DEFAULT_RIDGE_ALPHA = 1.0

# Forecasting defaults
DEFAULT_START_CUTOFF = 30
DEFAULT_CONFIDENCE_LEVELS = (0.80, 0.90, 0.95)


class DiagnosticSeverity:
    """Diagnostic severity levels for regression diagnostics.

    Use these constants instead of magic strings.
    """
    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"
    POOR = "POOR"

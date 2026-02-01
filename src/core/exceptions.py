"""
Exception Hierarchy for Annuity Price Elasticity v2.

Centralized exception definitions with business context for fail-fast
error handling. All exceptions include business_impact and required_action
fields for clear diagnostics.

Usage:
    from src.core.exceptions import (
        DataLoadError,
        AutocorrelationTestError,
        ConstraintViolationError,
    )

    raise DataLoadError(
        "Failed to load S3 data",
        business_impact="Cannot run inference pipeline",
        required_action="Verify AWS credentials and bucket access"
    )
"""


class ElasticityBaseError(Exception):
    """Base exception with business context for all elasticity errors.

    Attributes
    ----------
    message : str
        Technical error description
    business_impact : str
        Business consequence of this error
    required_action : str
        What the user should do to resolve
    """

    def __init__(
        self,
        message: str,
        business_impact: str | None = None,
        required_action: str | None = None,
    ):
        self.message = message
        self.business_impact = business_impact or "Unknown business impact"
        self.required_action = required_action or "Contact development team"
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error with business context."""
        return (
            f"{self.message}\n"
            f"  Business Impact: {self.business_impact}\n"
            f"  Required Action: {self.required_action}"
        )


# =============================================================================
# DATA LAYER EXCEPTIONS
# =============================================================================


class DataError(ElasticityBaseError):
    """Base class for data-related errors.

    Raised when data loading, validation, or schema operations fail. All data layer
    exceptions inherit from this class to enable granular error handling.

    Common Causes
    -------------
    - Data source unavailability (S3, database, file system)
    - Data quality issues (missing values, invalid formats)
    - Schema mismatches between expected and actual data

    Business Impact
    ---------------
    Pipeline cannot proceed without valid data, blocking all downstream analysis
    and model training operations.

    Recovery Actions
    ----------------
    - Verify data source accessibility
    - Check data quality and preprocessing steps
    - Validate schema matches expected structure

    Examples
    --------
    >>> try:
    ...     df = load_data_from_source()
    ... except DataError as e:
    ...     print(f"Data error: {e.message}")
    ...     print(f"Impact: {e.business_impact}")
    """

    pass


class DataLoadError(DataError):
    """Raised when data cannot be loaded from source.

    Examples: S3 access failure, missing files, network issues.
    """

    def __init__(
        self,
        message: str,
        source: str | None = None,
        business_impact: str | None = None,
        required_action: str | None = None,
    ):
        self.source = source
        business_impact = business_impact or "Pipeline cannot proceed without data"
        required_action = required_action or "Verify data source accessibility"
        super().__init__(message, business_impact, required_action)


class DataValidationError(DataError):
    """Raised when data fails validation checks.

    Examples: Missing columns, invalid dtypes, out-of-range values.
    """

    def __init__(
        self,
        message: str,
        validation_type: str | None = None,
        business_impact: str | None = None,
        required_action: str | None = None,
    ):
        self.validation_type = validation_type
        business_impact = business_impact or "Invalid data may produce incorrect results"
        required_action = required_action or "Review data quality and preprocessing"
        super().__init__(message, business_impact, required_action)


class DataSchemaError(DataError):
    """Raised when data schema doesn't match expected structure."""

    def __init__(
        self,
        message: str,
        expected_schema: str | None = None,
        actual_schema: str | None = None,
        business_impact: str | None = None,
        required_action: str | None = None,
    ):
        self.expected_schema = expected_schema
        self.actual_schema = actual_schema
        business_impact = business_impact or "Data structure mismatch prevents processing"
        required_action = required_action or "Verify data source schema matches expected"
        super().__init__(message, business_impact, required_action)


class PipelineStageError(DataError):
    """Raised when a data pipeline stage fails.

    Used for fail-fast error handling in the 10-stage merge pipeline.
    Includes stage number and name for clear diagnostics.
    """

    def __init__(
        self,
        message: str,
        stage_number: int,
        stage_name: str,
        business_impact: str | None = None,
        required_action: str | None = None,
    ):
        self.stage_number = stage_number
        self.stage_name = stage_name
        business_impact = (
            business_impact or f"Pipeline halted at stage {stage_number}: {stage_name}"
        )
        required_action = required_action or "Check data quality and configuration for this stage"
        super().__init__(
            f"Stage {stage_number} ({stage_name}) failed: {message}",
            business_impact,
            required_action,
        )


# =============================================================================
# DIAGNOSTIC LAYER EXCEPTIONS
# =============================================================================


class DiagnosticError(ElasticityBaseError):
    """Base class for regression diagnostic errors.

    Raised when statistical diagnostic tests fail or cannot be computed. Diagnostic
    tests validate regression assumptions (autocorrelation, heteroscedasticity,
    normality, multicollinearity) required for reliable inference.

    When Raised
    -----------
    - Durbin-Watson, Ljung-Box, or other autocorrelation tests fail
    - Breusch-Pagan or White heteroscedasticity tests fail
    - Jarque-Bera or other normality tests fail
    - VIF analysis detects severe multicollinearity
    - Insufficient data for diagnostic computation

    Common Causes
    -------------
    - Time series autocorrelation in residuals
    - Non-constant variance in residuals (heteroscedasticity)
    - Non-normal residual distributions
    - Highly correlated predictors
    - Insufficient sample size for statistical tests

    Business Impact
    ---------------
    Cannot verify regression assumptions, which may invalidate confidence intervals
    and significance tests. Predictions may still be accurate, but statistical
    reliability cannot be guaranteed.

    Recovery Actions
    ----------------
    - Review residual plots for diagnostic patterns
    - Consider robust standard errors
    - Apply variance-stabilizing transformations
    - Remove or combine correlated features
    - Increase sample size if possible

    Examples
    --------
    >>> try:
    ...     diagnostics = run_regression_diagnostics(model, data)
    ... except DiagnosticError as e:
    ...     print(f"Diagnostic failed: {e.message}")
    ...     # Proceed with caution or apply robust methods
    """

    pass


class AutocorrelationTestError(DiagnosticError):
    """Raised when autocorrelation test fails or cannot be computed.

    Examples: Durbin-Watson test failure, Ljung-Box computation error.
    """

    def __init__(
        self,
        message: str,
        test_name: str | None = None,
        business_impact: str | None = None,
        required_action: str | None = None,
    ):
        self.test_name = test_name
        business_impact = business_impact or "Cannot verify time series assumptions"
        required_action = required_action or "Check residuals for sufficient length"
        super().__init__(message, business_impact, required_action)


class HeteroscedasticityTestError(DiagnosticError):
    """Raised when heteroscedasticity test fails.

    Examples: Breusch-Pagan test failure, White test computation error.
    """

    def __init__(
        self,
        message: str,
        test_name: str | None = None,
        business_impact: str | None = None,
        required_action: str | None = None,
    ):
        self.test_name = test_name
        business_impact = business_impact or "Standard errors may be unreliable"
        required_action = required_action or "Consider robust standard errors"
        super().__init__(message, business_impact, required_action)


class NormalityTestError(DiagnosticError):
    """Raised when normality test fails.

    Examples: Jarque-Bera test failure, insufficient sample size.
    """

    def __init__(
        self,
        message: str,
        test_name: str | None = None,
        business_impact: str | None = None,
        required_action: str | None = None,
    ):
        self.test_name = test_name
        business_impact = business_impact or "Confidence intervals may be unreliable"
        required_action = required_action or "Check for outliers or consider bootstrap"
        super().__init__(message, business_impact, required_action)


class MulticollinearityError(DiagnosticError):
    """Raised when VIF analysis fails or finds severe multicollinearity."""

    def __init__(
        self,
        message: str,
        vif_values: dict | None = None,
        business_impact: str | None = None,
        required_action: str | None = None,
    ):
        self.vif_values = vif_values
        business_impact = business_impact or "Coefficient estimates may be unstable"
        required_action = required_action or "Consider removing correlated features"
        super().__init__(message, business_impact, required_action)


# =============================================================================
# MODEL LAYER EXCEPTIONS
# =============================================================================


class ModelError(ElasticityBaseError):
    """Base class for model-related errors.

    Raised when model training, prediction, or validation operations fail. All model
    layer exceptions inherit from this class to enable granular error handling.

    When Raised
    -----------
    - Model training fails to converge
    - Economic constraints are violated
    - Feature selection produces invalid results
    - Prediction operations fail

    Common Causes
    -------------
    - Insufficient or poor quality training data
    - Violated economic theory constraints (wrong coefficient signs)
    - Numerical instability in optimization
    - Invalid feature sets or parameter configurations

    Business Impact
    ---------------
    Cannot generate reliable price elasticity predictions, blocking strategic
    pricing decisions and revenue forecasting.

    Recovery Actions
    ----------------
    - Review training data quality and completeness
    - Verify feature engineering and selection
    - Check economic constraints configuration
    - Adjust model hyperparameters
    - Scale or normalize features

    Examples
    --------
    >>> try:
    ...     model = train_elasticity_model(data, config)
    ... except ModelError as e:
    ...     print(f"Model error: {e.message}")
    ...     # Review data quality or adjust configuration
    """

    pass


class ConstraintViolationError(ModelError):
    """Raised when economic constraints are violated.

    Examples: Wrong coefficient signs, forbidden lag-0 features.
    """

    def __init__(
        self,
        message: str,
        constraint_type: str | None = None,
        feature_name: str | None = None,
        expected_sign: str | None = None,
        actual_sign: str | None = None,
        business_impact: str | None = None,
        required_action: str | None = None,
    ):
        self.constraint_type = constraint_type
        self.feature_name = feature_name
        self.expected_sign = expected_sign
        self.actual_sign = actual_sign
        business_impact = business_impact or "Model violates economic theory"
        required_action = required_action or "Review feature selection and data quality"
        super().__init__(message, business_impact, required_action)


class ModelConvergenceError(ModelError):
    """Raised when model fails to converge."""

    def __init__(
        self,
        message: str,
        n_iterations: int | None = None,
        business_impact: str | None = None,
        required_action: str | None = None,
    ):
        self.n_iterations = n_iterations
        business_impact = business_impact or "Cannot produce reliable estimates"
        required_action = required_action or "Check data scaling or reduce model complexity"
        super().__init__(message, business_impact, required_action)


class FeatureSelectionError(ModelError):
    """Raised when feature selection fails."""

    def __init__(
        self,
        message: str,
        n_features_attempted: int | None = None,
        business_impact: str | None = None,
        required_action: str | None = None,
    ):
        self.n_features_attempted = n_features_attempted
        business_impact = business_impact or "Cannot identify optimal feature set"
        required_action = required_action or "Review candidate features and constraints"
        super().__init__(message, business_impact, required_action)


# =============================================================================
# VISUALIZATION LAYER EXCEPTIONS
# =============================================================================


class VisualizationError(ElasticityBaseError):
    """Base class for visualization errors.

    Raised when plot generation or export operations fail. All visualization layer
    exceptions inherit from this class to enable granular error handling.

    When Raised
    -----------
    - Plot generation fails due to data incompatibility
    - File export fails due to permissions or disk space
    - Invalid visualization parameters or configurations
    - Missing required visualization libraries

    Common Causes
    -------------
    - Incompatible data formats for plot types
    - File system permission issues
    - Insufficient disk space
    - Missing matplotlib, seaborn, or other visualization dependencies
    - Invalid file paths or formats

    Business Impact
    ---------------
    Cannot generate visual outputs for executive presentations, strategic planning,
    or regulatory documentation. Analysis results remain valid but cannot be
    effectively communicated.

    Recovery Actions
    ----------------
    - Verify data format compatibility with plot type
    - Check file permissions and disk space
    - Validate output paths exist
    - Ensure visualization libraries installed
    - Try alternative export formats

    Examples
    --------
    >>> try:
    ...     fig = generate_price_elasticity_plot(results)
    ...     export_plot(fig, "output.png")
    ... except VisualizationError as e:
    ...     print(f"Visualization failed: {e.message}")
    ...     # Export to CSV as fallback
    """

    pass


class PlotGenerationError(VisualizationError):
    """Raised when plot generation fails."""

    def __init__(
        self,
        message: str,
        plot_type: str | None = None,
        business_impact: str | None = None,
        required_action: str | None = None,
    ):
        self.plot_type = plot_type
        business_impact = business_impact or "Cannot generate visual output"
        required_action = required_action or "Check data compatibility with plot type"
        super().__init__(message, business_impact, required_action)


class ExportError(VisualizationError):
    """Raised when export to file fails."""

    def __init__(
        self,
        message: str,
        export_format: str | None = None,
        file_path: str | None = None,
        business_impact: str | None = None,
        required_action: str | None = None,
    ):
        self.export_format = export_format
        self.file_path = file_path
        business_impact = business_impact or "Results cannot be saved or shared"
        required_action = required_action or "Check file permissions and disk space"
        super().__init__(message, business_impact, required_action)


# =============================================================================
# CONFIGURATION EXCEPTIONS
# =============================================================================


class ConfigurationError(ElasticityBaseError):
    """Base class for configuration errors.

    Raised when configuration is invalid, incomplete, or incompatible. All configuration
    layer exceptions inherit from this class to enable granular error handling.

    When Raised
    -----------
    - Configuration parameters fail validation
    - Product type not found or not registered
    - Builder configuration incompatible with data
    - Missing required configuration fields

    Common Causes
    -------------
    - Typos in product codes or parameter names
    - Invalid parameter values (out of range, wrong type)
    - Product not registered in configuration registry
    - Configuration schema version mismatches
    - Missing required configuration files

    Business Impact
    ---------------
    Pipeline cannot run with invalid configuration, blocking all analysis and
    model training operations. Configuration errors must be fixed before proceeding.

    Recovery Actions
    ----------------
    - Review configuration parameter values
    - Verify product codes match registered products
    - Check configuration schema documentation
    - Validate all required fields present
    - Use configuration validation tools

    Examples
    --------
    >>> try:
    ...     config = load_product_config("INVALID_PRODUCT")
    ... except ConfigurationError as e:
    ...     print(f"Configuration error: {e.message}")
    ...     print(f"Available products: {e.available_products}")
    """

    pass


class InvalidConfigError(ConfigurationError):
    """Raised when configuration is invalid."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        business_impact: str | None = None,
        required_action: str | None = None,
    ):
        self.config_key = config_key
        business_impact = business_impact or "Pipeline cannot run with invalid config"
        required_action = required_action or "Review configuration parameters"
        super().__init__(message, business_impact, required_action)


class ProductNotFoundError(ConfigurationError):
    """Raised when product type is not registered."""

    def __init__(
        self,
        message: str,
        product_type: str | None = None,
        available_products: list | None = None,
        business_impact: str | None = None,
        required_action: str | None = None,
    ):
        self.product_type = product_type
        self.available_products = available_products
        business_impact = business_impact or f"Product '{product_type}' not supported"
        required_action = required_action or f"Use one of: {available_products}"
        super().__init__(message, business_impact, required_action)


__all__ = [
    # Base
    "ElasticityBaseError",
    # Data layer
    "DataError",
    "DataLoadError",
    "DataValidationError",
    "DataSchemaError",
    "PipelineStageError",
    # Diagnostic layer
    "DiagnosticError",
    "AutocorrelationTestError",
    "HeteroscedasticityTestError",
    "NormalityTestError",
    "MulticollinearityError",
    # Model layer
    "ModelError",
    "ConstraintViolationError",
    "ModelConvergenceError",
    "FeatureSelectionError",
    # Visualization layer
    "VisualizationError",
    "PlotGenerationError",
    "ExportError",
    # Configuration layer
    "ConfigurationError",
    "InvalidConfigError",
    "ProductNotFoundError",
]

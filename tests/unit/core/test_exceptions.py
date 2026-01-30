"""
Unit tests for src/core/exceptions.py.

Tests validate exception construction, inheritance, and attribute handling
for the complete exception hierarchy.
"""

import pytest

from src.core.exceptions import (
    # Base
    ElasticityBaseError,
    # Data layer
    DataError,
    DataLoadError,
    DataValidationError,
    DataSchemaError,
    # Diagnostic layer
    DiagnosticError,
    AutocorrelationTestError,
    HeteroscedasticityTestError,
    NormalityTestError,
    MulticollinearityError,
    # Model layer
    ModelError,
    ConstraintViolationError,
    ModelConvergenceError,
    FeatureSelectionError,
    # Visualization layer
    VisualizationError,
    PlotGenerationError,
    ExportError,
    # Configuration layer
    ConfigurationError,
    InvalidConfigError,
    ProductNotFoundError,
)


# =============================================================================
# ElasticityBaseError Tests
# =============================================================================


class TestElasticityBaseError:
    """Tests for ElasticityBaseError base class."""

    def test_construction_with_all_params(self):
        """ElasticityBaseError should accept message, business_impact, required_action."""
        err = ElasticityBaseError(
            message="Test error",
            business_impact="Cannot proceed",
            required_action="Fix the issue"
        )

        assert err.message == "Test error"
        assert err.business_impact == "Cannot proceed"
        assert err.required_action == "Fix the issue"

    def test_construction_with_defaults(self):
        """ElasticityBaseError should have sensible defaults."""
        err = ElasticityBaseError("Simple error")

        assert err.message == "Simple error"
        assert err.business_impact == "Unknown business impact"
        assert err.required_action == "Contact development team"

    def test_format_message_includes_all_fields(self):
        """_format_message should include message, impact, and action."""
        err = ElasticityBaseError(
            message="Error message",
            business_impact="Impact description",
            required_action="Action required"
        )

        formatted = err._format_message()

        assert "Error message" in formatted
        assert "Business Impact: Impact description" in formatted
        assert "Required Action: Action required" in formatted

    def test_str_representation(self):
        """str(exception) should include formatted message."""
        err = ElasticityBaseError(
            message="Test error",
            business_impact="Test impact",
            required_action="Test action"
        )

        str_repr = str(err)

        assert "Test error" in str_repr
        assert "Test impact" in str_repr
        assert "Test action" in str_repr

    def test_can_be_raised_and_caught(self):
        """ElasticityBaseError should be raiseable and catchable."""
        with pytest.raises(ElasticityBaseError) as exc_info:
            raise ElasticityBaseError("Test raise")

        assert "Test raise" in str(exc_info.value)

    def test_inherits_from_exception(self):
        """ElasticityBaseError should inherit from Exception."""
        assert issubclass(ElasticityBaseError, Exception)


# =============================================================================
# Data Layer Exception Tests
# =============================================================================


class TestDataError:
    """Tests for DataError base class."""

    def test_construction(self):
        """DataError should construct properly."""
        err = DataError("Data error occurred")
        assert err.message == "Data error occurred"

    def test_inherits_from_base(self):
        """DataError should inherit from ElasticityBaseError."""
        assert issubclass(DataError, ElasticityBaseError)

    def test_catchable_by_parent(self):
        """DataError should be catchable as ElasticityBaseError."""
        with pytest.raises(ElasticityBaseError):
            raise DataError("Test")


class TestDataLoadError:
    """Tests for DataLoadError."""

    def test_construction_with_source(self):
        """DataLoadError should accept source attribute."""
        err = DataLoadError(
            message="Failed to load data",
            source="s3://bucket/path/data.parquet"
        )

        assert err.message == "Failed to load data"
        assert err.source == "s3://bucket/path/data.parquet"

    def test_default_business_impact(self):
        """DataLoadError should have specific default impact."""
        err = DataLoadError("Load failed")
        assert "Pipeline cannot proceed" in err.business_impact

    def test_default_required_action(self):
        """DataLoadError should have specific default action."""
        err = DataLoadError("Load failed")
        assert "Verify data source" in err.required_action

    def test_custom_impact_overrides_default(self):
        """Custom business_impact should override default."""
        err = DataLoadError(
            "Load failed",
            business_impact="Custom impact"
        )
        assert err.business_impact == "Custom impact"

    def test_inherits_from_data_error(self):
        """DataLoadError should inherit from DataError."""
        assert issubclass(DataLoadError, DataError)


class TestDataValidationError:
    """Tests for DataValidationError."""

    def test_construction_with_validation_type(self):
        """DataValidationError should accept validation_type attribute."""
        err = DataValidationError(
            message="Validation failed",
            validation_type="schema"
        )

        assert err.validation_type == "schema"

    def test_default_messages(self):
        """DataValidationError should have specific defaults."""
        err = DataValidationError("Validation failed")
        assert "Invalid data" in err.business_impact
        assert "Review data quality" in err.required_action

    def test_inherits_from_data_error(self):
        """DataValidationError should inherit from DataError."""
        assert issubclass(DataValidationError, DataError)


class TestDataSchemaError:
    """Tests for DataSchemaError."""

    def test_construction_with_schema_info(self):
        """DataSchemaError should accept schema attributes."""
        err = DataSchemaError(
            message="Schema mismatch",
            expected_schema="v2.0",
            actual_schema="v1.0"
        )

        assert err.expected_schema == "v2.0"
        assert err.actual_schema == "v1.0"

    def test_default_messages(self):
        """DataSchemaError should have specific defaults."""
        err = DataSchemaError("Schema mismatch")
        assert "structure mismatch" in err.business_impact
        assert "schema matches" in err.required_action

    def test_inherits_from_data_error(self):
        """DataSchemaError should inherit from DataError."""
        assert issubclass(DataSchemaError, DataError)


# =============================================================================
# Diagnostic Layer Exception Tests
# =============================================================================


class TestDiagnosticError:
    """Tests for DiagnosticError base class."""

    def test_construction(self):
        """DiagnosticError should construct properly."""
        err = DiagnosticError("Diagnostic failed")
        assert err.message == "Diagnostic failed"

    def test_inherits_from_base(self):
        """DiagnosticError should inherit from ElasticityBaseError."""
        assert issubclass(DiagnosticError, ElasticityBaseError)


class TestAutocorrelationTestError:
    """Tests for AutocorrelationTestError."""

    def test_construction_with_test_name(self):
        """AutocorrelationTestError should accept test_name attribute."""
        err = AutocorrelationTestError(
            message="Durbin-Watson test failed",
            test_name="durbin_watson"
        )

        assert err.test_name == "durbin_watson"

    def test_default_messages(self):
        """AutocorrelationTestError should have specific defaults."""
        err = AutocorrelationTestError("Test failed")
        assert "time series assumptions" in err.business_impact
        assert "residuals" in err.required_action

    def test_inherits_from_diagnostic_error(self):
        """AutocorrelationTestError should inherit from DiagnosticError."""
        assert issubclass(AutocorrelationTestError, DiagnosticError)


class TestHeteroscedasticityTestError:
    """Tests for HeteroscedasticityTestError."""

    def test_construction_with_test_name(self):
        """HeteroscedasticityTestError should accept test_name attribute."""
        err = HeteroscedasticityTestError(
            message="Breusch-Pagan test failed",
            test_name="breusch_pagan"
        )

        assert err.test_name == "breusch_pagan"

    def test_default_messages(self):
        """HeteroscedasticityTestError should have specific defaults."""
        err = HeteroscedasticityTestError("Test failed")
        assert "Standard errors" in err.business_impact
        assert "robust" in err.required_action

    def test_inherits_from_diagnostic_error(self):
        """HeteroscedasticityTestError should inherit from DiagnosticError."""
        assert issubclass(HeteroscedasticityTestError, DiagnosticError)


class TestNormalityTestError:
    """Tests for NormalityTestError."""

    def test_construction_with_test_name(self):
        """NormalityTestError should accept test_name attribute."""
        err = NormalityTestError(
            message="Jarque-Bera test failed",
            test_name="jarque_bera"
        )

        assert err.test_name == "jarque_bera"

    def test_default_messages(self):
        """NormalityTestError should have specific defaults."""
        err = NormalityTestError("Test failed")
        assert "Confidence intervals" in err.business_impact
        assert "outliers" in err.required_action

    def test_inherits_from_diagnostic_error(self):
        """NormalityTestError should inherit from DiagnosticError."""
        assert issubclass(NormalityTestError, DiagnosticError)


class TestMulticollinearityError:
    """Tests for MulticollinearityError."""

    def test_construction_with_vif_values(self):
        """MulticollinearityError should accept vif_values attribute."""
        vif = {'feature_a': 15.5, 'feature_b': 12.3}
        err = MulticollinearityError(
            message="High VIF detected",
            vif_values=vif
        )

        assert err.vif_values == vif

    def test_default_messages(self):
        """MulticollinearityError should have specific defaults."""
        err = MulticollinearityError("VIF too high")
        assert "Coefficient estimates" in err.business_impact or "unstable" in err.business_impact
        assert "correlated features" in err.required_action

    def test_inherits_from_diagnostic_error(self):
        """MulticollinearityError should inherit from DiagnosticError."""
        assert issubclass(MulticollinearityError, DiagnosticError)


# =============================================================================
# Model Layer Exception Tests
# =============================================================================


class TestModelError:
    """Tests for ModelError base class."""

    def test_construction(self):
        """ModelError should construct properly."""
        err = ModelError("Model error")
        assert err.message == "Model error"

    def test_inherits_from_base(self):
        """ModelError should inherit from ElasticityBaseError."""
        assert issubclass(ModelError, ElasticityBaseError)


class TestConstraintViolationError:
    """Tests for ConstraintViolationError."""

    def test_construction_with_all_attributes(self):
        """ConstraintViolationError should accept all constraint attributes."""
        err = ConstraintViolationError(
            message="Wrong sign for competitor coefficient",
            constraint_type="sign_constraint",
            feature_name="competitor_weighted_t2",
            expected_sign="negative",
            actual_sign="positive"
        )

        assert err.constraint_type == "sign_constraint"
        assert err.feature_name == "competitor_weighted_t2"
        assert err.expected_sign == "negative"
        assert err.actual_sign == "positive"

    def test_default_messages(self):
        """ConstraintViolationError should have specific defaults."""
        err = ConstraintViolationError("Constraint violated")
        assert "economic theory" in err.business_impact
        assert "feature selection" in err.required_action

    def test_inherits_from_model_error(self):
        """ConstraintViolationError should inherit from ModelError."""
        assert issubclass(ConstraintViolationError, ModelError)


class TestModelConvergenceError:
    """Tests for ModelConvergenceError."""

    def test_construction_with_n_iterations(self):
        """ModelConvergenceError should accept n_iterations attribute."""
        err = ModelConvergenceError(
            message="Model did not converge",
            n_iterations=10000
        )

        assert err.n_iterations == 10000

    def test_default_messages(self):
        """ModelConvergenceError should have specific defaults."""
        err = ModelConvergenceError("Convergence failed")
        assert "reliable estimates" in err.business_impact
        assert "scaling" in err.required_action or "complexity" in err.required_action

    def test_inherits_from_model_error(self):
        """ModelConvergenceError should inherit from ModelError."""
        assert issubclass(ModelConvergenceError, ModelError)


class TestFeatureSelectionError:
    """Tests for FeatureSelectionError."""

    def test_construction_with_n_features(self):
        """FeatureSelectionError should accept n_features_attempted attribute."""
        err = FeatureSelectionError(
            message="Feature selection failed",
            n_features_attempted=50
        )

        assert err.n_features_attempted == 50

    def test_default_messages(self):
        """FeatureSelectionError should have specific defaults."""
        err = FeatureSelectionError("Selection failed")
        assert "optimal feature set" in err.business_impact
        assert "candidate features" in err.required_action

    def test_inherits_from_model_error(self):
        """FeatureSelectionError should inherit from ModelError."""
        assert issubclass(FeatureSelectionError, ModelError)


# =============================================================================
# Visualization Layer Exception Tests
# =============================================================================


class TestVisualizationError:
    """Tests for VisualizationError base class."""

    def test_construction(self):
        """VisualizationError should construct properly."""
        err = VisualizationError("Visualization failed")
        assert err.message == "Visualization failed"

    def test_inherits_from_base(self):
        """VisualizationError should inherit from ElasticityBaseError."""
        assert issubclass(VisualizationError, ElasticityBaseError)


class TestPlotGenerationError:
    """Tests for PlotGenerationError."""

    def test_construction_with_plot_type(self):
        """PlotGenerationError should accept plot_type attribute."""
        err = PlotGenerationError(
            message="Failed to generate heatmap",
            plot_type="heatmap"
        )

        assert err.plot_type == "heatmap"

    def test_default_messages(self):
        """PlotGenerationError should have specific defaults."""
        err = PlotGenerationError("Plot failed")
        assert "visual output" in err.business_impact
        assert "data compatibility" in err.required_action

    def test_inherits_from_visualization_error(self):
        """PlotGenerationError should inherit from VisualizationError."""
        assert issubclass(PlotGenerationError, VisualizationError)


class TestExportError:
    """Tests for ExportError."""

    def test_construction_with_export_info(self):
        """ExportError should accept export_format and file_path attributes."""
        err = ExportError(
            message="Export failed",
            export_format="png",
            file_path="/output/plot.png"
        )

        assert err.export_format == "png"
        assert err.file_path == "/output/plot.png"

    def test_default_messages(self):
        """ExportError should have specific defaults."""
        err = ExportError("Export failed")
        assert "cannot be saved" in err.business_impact or "shared" in err.business_impact
        assert "permissions" in err.required_action or "disk space" in err.required_action

    def test_inherits_from_visualization_error(self):
        """ExportError should inherit from VisualizationError."""
        assert issubclass(ExportError, VisualizationError)


# =============================================================================
# Configuration Layer Exception Tests
# =============================================================================


class TestConfigurationError:
    """Tests for ConfigurationError base class."""

    def test_construction(self):
        """ConfigurationError should construct properly."""
        err = ConfigurationError("Config error")
        assert err.message == "Config error"

    def test_inherits_from_base(self):
        """ConfigurationError should inherit from ElasticityBaseError."""
        assert issubclass(ConfigurationError, ElasticityBaseError)


class TestInvalidConfigError:
    """Tests for InvalidConfigError."""

    def test_construction_with_config_key(self):
        """InvalidConfigError should accept config_key attribute."""
        err = InvalidConfigError(
            message="Invalid config value",
            config_key="n_samples"
        )

        assert err.config_key == "n_samples"

    def test_default_messages(self):
        """InvalidConfigError should have specific defaults."""
        err = InvalidConfigError("Config invalid")
        assert "Pipeline cannot run" in err.business_impact
        assert "configuration parameters" in err.required_action

    def test_inherits_from_configuration_error(self):
        """InvalidConfigError should inherit from ConfigurationError."""
        assert issubclass(InvalidConfigError, ConfigurationError)


class TestProductNotFoundError:
    """Tests for ProductNotFoundError."""

    def test_construction_with_product_info(self):
        """ProductNotFoundError should accept product_type and available_products."""
        err = ProductNotFoundError(
            message="Product not found",
            product_type="INVALID",
            available_products=["6Y20B", "6Y10B", "10Y20B"]
        )

        assert err.product_type == "INVALID"
        assert err.available_products == ["6Y20B", "6Y10B", "10Y20B"]

    def test_default_business_impact_includes_product(self):
        """ProductNotFoundError default impact should include product type."""
        err = ProductNotFoundError(
            "Not found",
            product_type="UNKNOWN"
        )
        assert "UNKNOWN" in err.business_impact

    def test_default_action_includes_available(self):
        """ProductNotFoundError default action should include available products."""
        available = ["6Y20B", "6Y10B"]
        err = ProductNotFoundError(
            "Not found",
            product_type="INVALID",
            available_products=available
        )
        assert "6Y20B" in err.required_action or str(available) in err.required_action

    def test_inherits_from_configuration_error(self):
        """ProductNotFoundError should inherit from ConfigurationError."""
        assert issubclass(ProductNotFoundError, ConfigurationError)


# =============================================================================
# Inheritance Chain Verification Tests
# =============================================================================


class TestInheritanceChains:
    """Verify complete inheritance hierarchy."""

    @pytest.mark.parametrize("exception_class,parent_class", [
        # Data layer
        (DataError, ElasticityBaseError),
        (DataLoadError, DataError),
        (DataValidationError, DataError),
        (DataSchemaError, DataError),
        # Diagnostic layer
        (DiagnosticError, ElasticityBaseError),
        (AutocorrelationTestError, DiagnosticError),
        (HeteroscedasticityTestError, DiagnosticError),
        (NormalityTestError, DiagnosticError),
        (MulticollinearityError, DiagnosticError),
        # Model layer
        (ModelError, ElasticityBaseError),
        (ConstraintViolationError, ModelError),
        (ModelConvergenceError, ModelError),
        (FeatureSelectionError, ModelError),
        # Visualization layer
        (VisualizationError, ElasticityBaseError),
        (PlotGenerationError, VisualizationError),
        (ExportError, VisualizationError),
        # Configuration layer
        (ConfigurationError, ElasticityBaseError),
        (InvalidConfigError, ConfigurationError),
        (ProductNotFoundError, ConfigurationError),
    ])
    def test_inheritance(self, exception_class, parent_class):
        """Verify each exception inherits from expected parent."""
        assert issubclass(exception_class, parent_class)

    @pytest.mark.parametrize("exception_class", [
        DataLoadError,
        DataValidationError,
        DataSchemaError,
        AutocorrelationTestError,
        HeteroscedasticityTestError,
        NormalityTestError,
        MulticollinearityError,
        ConstraintViolationError,
        ModelConvergenceError,
        FeatureSelectionError,
        PlotGenerationError,
        ExportError,
        InvalidConfigError,
        ProductNotFoundError,
    ])
    def test_all_catchable_as_base(self, exception_class):
        """All exceptions should be catchable as ElasticityBaseError."""
        assert issubclass(exception_class, ElasticityBaseError)


# =============================================================================
# Exception Catching By Layer Tests
# =============================================================================


class TestExceptionCatchingByLayer:
    """Test catching exceptions by their layer parent class."""

    def test_catch_all_data_errors(self):
        """All data layer exceptions should be catchable as DataError."""
        data_exceptions = [DataLoadError, DataValidationError, DataSchemaError]

        for exc_class in data_exceptions:
            with pytest.raises(DataError):
                raise exc_class("Test error")

    def test_catch_all_diagnostic_errors(self):
        """All diagnostic layer exceptions should be catchable as DiagnosticError."""
        diagnostic_exceptions = [
            AutocorrelationTestError,
            HeteroscedasticityTestError,
            NormalityTestError,
            MulticollinearityError,
        ]

        for exc_class in diagnostic_exceptions:
            with pytest.raises(DiagnosticError):
                raise exc_class("Test error")

    def test_catch_all_model_errors(self):
        """All model layer exceptions should be catchable as ModelError."""
        model_exceptions = [
            ConstraintViolationError,
            ModelConvergenceError,
            FeatureSelectionError,
        ]

        for exc_class in model_exceptions:
            with pytest.raises(ModelError):
                raise exc_class("Test error")

    def test_catch_all_visualization_errors(self):
        """All visualization layer exceptions should be catchable as VisualizationError."""
        viz_exceptions = [PlotGenerationError, ExportError]

        for exc_class in viz_exceptions:
            with pytest.raises(VisualizationError):
                raise exc_class("Test error")

    def test_catch_all_configuration_errors(self):
        """All configuration layer exceptions should be catchable as ConfigurationError."""
        config_exceptions = [InvalidConfigError, ProductNotFoundError]

        for exc_class in config_exceptions:
            with pytest.raises(ConfigurationError):
                raise exc_class("Test error")

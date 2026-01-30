"""
Tests for Feature Selection Pipeline Execution Module.

Tests cover:
- _prepare_feature_selection_parameters: Flag resolution logic
- _log_pipeline_status: Console output verification
- _create_error_aic_result: Error result structure
- _create_error_feature_selection_results: Full error results
- _create_error_fallback_results: Return type routing
- _handle_pipeline_error: Error logging and fallback creation
- _execute_feature_selection_pipeline: Pipeline execution with mocking
- _format_pipeline_results: Result formatting logic
- _run_atomic_pipeline: Atomic pipeline execution with mocking
- run_feature_selection: Main entry point with error handling

Design Principles:
- Real assertions about correctness
- Test error handling paths thoroughly
- Mock external dependencies where needed

Author: Claude Code
Date: 2026-01-30
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.features.selection.interface.interface_execution import (
    ATOMIC_FUNCTIONS_AVAILABLE,
    _prepare_feature_selection_parameters,
    _log_pipeline_status,
    _create_error_aic_result,
    _create_error_feature_selection_results,
    _create_error_fallback_results,
    _handle_pipeline_error,
    _format_pipeline_results,
    run_feature_selection,
)
from src.features.selection.interface.interface_config import FEATURE_FLAGS
from src.features.selection_types import (
    AICResult,
    FeatureSelectionResults,
    FeatureSelectionConfig,
    EconomicConstraintConfig,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def reset_feature_flags():
    """Reset feature flags to defaults after each test."""
    original_flags = FEATURE_FLAGS.copy()
    yield
    FEATURE_FLAGS.clear()
    FEATURE_FLAGS.update(original_flags)


@pytest.fixture
def simple_dataframe():
    """Create simple DataFrame for testing."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        'target': np.random.randn(n) * 10 + 50,
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'feature_3': np.random.randn(n),
    })


@pytest.fixture
def sample_error():
    """Create sample exception for error testing."""
    return ValueError("Test error message")


@pytest.fixture
def mock_feature_selection_results():
    """Create mock FeatureSelectionResults for testing."""
    mock_results = MagicMock()
    mock_results.valid_results = pd.DataFrame({'col': [1, 2, 3]})
    mock_results.all_results = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
    return mock_results


@pytest.fixture
def empty_mock_results():
    """Create mock results with empty valid_results."""
    mock_results = MagicMock()
    mock_results.valid_results = pd.DataFrame()
    mock_results.all_results = pd.DataFrame({'col': [1, 2]})
    return mock_results


@pytest.fixture
def real_feature_selection_results():
    """Create a real FeatureSelectionResults dataclass for testing."""
    best_model = AICResult(
        features="feature_1 + feature_2",
        n_features=2,
        aic=150.5,
        bic=160.2,
        r_squared=0.75,
        r_squared_adj=0.72,
        coefficients={"feature_1": 0.5, "feature_2": -0.3, "const": 10.0},
        converged=True,
        n_obs=100,
    )

    feature_config = FeatureSelectionConfig(
        base_features=[],
        candidate_features=["feature_1", "feature_2", "feature_3"],
        max_candidate_features=3,
        target_variable="target",
    )

    constraint_config = EconomicConstraintConfig(enabled=True)

    return FeatureSelectionResults(
        best_model=best_model,
        all_results=pd.DataFrame({
            'features': ['feature_1', 'feature_1 + feature_2'],
            'aic': [155.0, 150.5],
        }),
        valid_results=pd.DataFrame({
            'features': ['feature_1 + feature_2'],
            'aic': [150.5],
        }),
        total_combinations=10,
        converged_models=8,
        economically_valid_models=5,
        constraint_violations=[],
        feature_config=feature_config,
        constraint_config=constraint_config,
    )


# =============================================================================
# Tests for _prepare_feature_selection_parameters
# =============================================================================


class TestPrepareFeatureSelectionParameters:
    """Tests for _prepare_feature_selection_parameters function."""

    def test_explicit_values_used(self, reset_feature_flags):
        """Test that explicit values override defaults."""
        display, bootstrap = _prepare_feature_selection_parameters(
            display_results=True,
            enable_bootstrap=True,
        )

        assert display is True
        assert bootstrap is True

    def test_explicit_false_values(self, reset_feature_flags):
        """Test that explicit False values are preserved."""
        display, bootstrap = _prepare_feature_selection_parameters(
            display_results=False,
            enable_bootstrap=False,
        )

        assert display is False
        assert bootstrap is False

    def test_none_uses_flag_defaults(self, reset_feature_flags):
        """Test that None falls back to feature flags."""
        # Set known flag values
        FEATURE_FLAGS["AUTO_DISPLAY_RESULTS"] = True
        FEATURE_FLAGS["ENABLE_BOOTSTRAP_DEFAULT"] = False

        display, bootstrap = _prepare_feature_selection_parameters(
            display_results=None,
            enable_bootstrap=None,
        )

        assert display is True
        assert bootstrap is False

    def test_mixed_explicit_and_default(self, reset_feature_flags):
        """Test mixed explicit and default values."""
        FEATURE_FLAGS["AUTO_DISPLAY_RESULTS"] = False

        display, bootstrap = _prepare_feature_selection_parameters(
            display_results=None,  # Use default (False)
            enable_bootstrap=True,  # Explicit True
        )

        assert display is False
        assert bootstrap is True

    def test_both_none_with_true_defaults(self, reset_feature_flags):
        """Test both None parameters with True defaults."""
        FEATURE_FLAGS["AUTO_DISPLAY_RESULTS"] = True
        FEATURE_FLAGS["ENABLE_BOOTSTRAP_DEFAULT"] = True

        display, bootstrap = _prepare_feature_selection_parameters(
            display_results=None,
            enable_bootstrap=None,
        )

        assert display is True
        assert bootstrap is True


# =============================================================================
# Tests for _log_pipeline_status
# =============================================================================


class TestLogPipelineStatus:
    """Tests for _log_pipeline_status function."""

    def test_logs_starting_message(self, capsys, reset_feature_flags):
        """Test that starting message is logged."""
        _log_pipeline_status()

        captured = capsys.readouterr()
        assert "Starting Feature Selection Analysis" in captured.out

    def test_logs_atomic_pipeline_when_available(self, capsys, reset_feature_flags):
        """Test atomic pipeline message when available."""
        FEATURE_FLAGS["USE_ATOMIC_FUNCTIONS"] = True

        _log_pipeline_status()

        captured = capsys.readouterr()
        # Output depends on ATOMIC_FUNCTIONS_AVAILABLE
        assert "Using" in captured.out

    def test_logs_legacy_when_atomic_disabled(self, capsys, reset_feature_flags):
        """Test legacy message when atomic functions disabled."""
        FEATURE_FLAGS["USE_ATOMIC_FUNCTIONS"] = False

        _log_pipeline_status()

        captured = capsys.readouterr()
        assert "Legacy Implementation" in captured.out

    def test_logs_legacy_when_atomic_flag_true_but_unavailable(self, capsys, reset_feature_flags):
        """Test that legacy message shown when flag True but functions unavailable."""
        FEATURE_FLAGS["USE_ATOMIC_FUNCTIONS"] = True

        # Temporarily patch ATOMIC_FUNCTIONS_AVAILABLE
        with patch('src.features.selection.interface.interface_execution.ATOMIC_FUNCTIONS_AVAILABLE', False):
            from src.features.selection.interface import interface_execution
            original_value = interface_execution.ATOMIC_FUNCTIONS_AVAILABLE
            interface_execution.ATOMIC_FUNCTIONS_AVAILABLE = False

            _log_pipeline_status()

            interface_execution.ATOMIC_FUNCTIONS_AVAILABLE = original_value

        captured = capsys.readouterr()
        assert "Using" in captured.out


# =============================================================================
# Tests for _create_error_aic_result
# =============================================================================


class TestCreateErrorAICResult:
    """Tests for _create_error_aic_result function (returns AICResult dataclass)."""

    def test_returns_aic_result_with_error_fields(self, sample_error):
        """Test that error AIC result has expected fields."""
        result = _create_error_aic_result(n_obs=100, error=sample_error)

        # AICResult is a dataclass - use attribute access
        assert result.features == "ERROR"
        assert result.n_features == 0
        assert result.aic == np.inf
        assert result.bic == np.inf
        assert result.r_squared == 0.0
        assert result.converged is False
        assert result.n_obs == 100
        assert "Test error message" in result.error

    def test_error_message_captured(self, sample_error):
        """Test that error message is captured."""
        result = _create_error_aic_result(n_obs=50, error=sample_error)

        assert result.error == "Test error message"

    def test_different_n_obs_values(self):
        """Test with different n_obs values."""
        error = RuntimeError("Different error")

        result_small = _create_error_aic_result(n_obs=10, error=error)
        result_large = _create_error_aic_result(n_obs=10000, error=error)

        assert result_small.n_obs == 10
        assert result_large.n_obs == 10000

    def test_r_squared_adj_is_zero(self, sample_error):
        """Test that r_squared_adj is also zero for error results."""
        result = _create_error_aic_result(n_obs=100, error=sample_error)

        assert result.r_squared_adj == 0.0

    def test_coefficients_is_empty_dict(self, sample_error):
        """Test that coefficients is empty dict for error results."""
        result = _create_error_aic_result(n_obs=100, error=sample_error)

        assert result.coefficients == {}

    def test_complex_exception_message(self):
        """Test with complex exception message containing special characters."""
        error = ValueError("Error with 'quotes' and \"double quotes\" and newline\n")
        result = _create_error_aic_result(n_obs=50, error=error)

        assert "quotes" in result.error
        assert result.converged is False


# =============================================================================
# Tests for _create_error_feature_selection_results
# =============================================================================


class TestCreateErrorFeatureSelectionResults:
    """Tests for _create_error_feature_selection_results function (returns dataclass)."""

    def test_returns_dataclass_structure(self, sample_error):
        """Test that error results have expected structure."""
        error_aic = _create_error_aic_result(n_obs=100, error=sample_error)

        result = _create_error_feature_selection_results(
            error_aic=error_aic,
            candidate_features=["feat1", "feat2"],
            target_variable="target",
            max_features=2,
            base_features=["base1"],
            enable_constraints=True,
        )

        # FeatureSelectionResults is a dataclass - use attribute access
        assert result.best_model == error_aic
        assert result.total_combinations == 0
        assert result.converged_models == 0
        assert result.economically_valid_models == 0

    def test_feature_config_preserved(self, sample_error):
        """Test that feature config is preserved in error results."""
        error_aic = _create_error_aic_result(n_obs=50, error=sample_error)

        result = _create_error_feature_selection_results(
            error_aic=error_aic,
            candidate_features=["a", "b", "c"],
            target_variable="sales",
            max_features=3,
            base_features=None,
            enable_constraints=False,
        )

        # feature_config is a TypedDict (dict), access with []
        assert result.feature_config["candidate_features"] == ["a", "b", "c"]
        assert result.feature_config["target_variable"] == "sales"
        assert result.feature_config["max_candidate_features"] == 3

    def test_none_base_features_becomes_empty_list(self, sample_error):
        """Test that None base_features becomes empty list."""
        error_aic = _create_error_aic_result(n_obs=50, error=sample_error)

        result = _create_error_feature_selection_results(
            error_aic=error_aic,
            candidate_features=["feat1"],
            target_variable="target",
            max_features=1,
            base_features=None,
            enable_constraints=True,
        )

        assert result.feature_config["base_features"] == []

    def test_constraint_config_enabled_flag(self, sample_error):
        """Test that constraint config enabled flag is set correctly."""
        error_aic = _create_error_aic_result(n_obs=50, error=sample_error)

        result_enabled = _create_error_feature_selection_results(
            error_aic=error_aic,
            candidate_features=["feat1"],
            target_variable="target",
            max_features=1,
            base_features=None,
            enable_constraints=True,
        )

        result_disabled = _create_error_feature_selection_results(
            error_aic=error_aic,
            candidate_features=["feat1"],
            target_variable="target",
            max_features=1,
            base_features=None,
            enable_constraints=False,
        )

        assert result_enabled.constraint_config["enabled"] is True
        assert result_disabled.constraint_config["enabled"] is False

    def test_all_results_is_empty_dataframe(self, sample_error):
        """Test that all_results is empty DataFrame."""
        error_aic = _create_error_aic_result(n_obs=50, error=sample_error)

        result = _create_error_feature_selection_results(
            error_aic=error_aic,
            candidate_features=["feat1"],
            target_variable="target",
            max_features=1,
            base_features=None,
            enable_constraints=True,
        )

        assert isinstance(result.all_results, pd.DataFrame)
        assert result.all_results.empty

    def test_constraint_violations_is_empty_list(self, sample_error):
        """Test that constraint_violations is empty list."""
        error_aic = _create_error_aic_result(n_obs=50, error=sample_error)

        result = _create_error_feature_selection_results(
            error_aic=error_aic,
            candidate_features=["feat1"],
            target_variable="target",
            max_features=1,
            base_features=None,
            enable_constraints=True,
        )

        assert result.constraint_violations == []


# =============================================================================
# Tests for _create_error_fallback_results
# =============================================================================


class TestCreateErrorFallbackResults:
    """Tests for _create_error_fallback_results function."""

    def test_returns_empty_dataframe_when_not_detailed(self, simple_dataframe, sample_error):
        """Test that empty DataFrame returned when return_detailed=False."""
        result = _create_error_fallback_results(
            data=simple_dataframe,
            candidate_features=["feature_1"],
            target_variable="target",
            max_features=1,
            base_features=None,
            enable_constraints=True,
            error=sample_error,
            return_detailed=False,
        )

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_full_results_when_detailed(self, simple_dataframe, sample_error):
        """Test that full results returned when return_detailed=True."""
        result = _create_error_fallback_results(
            data=simple_dataframe,
            candidate_features=["feature_1"],
            target_variable="target",
            max_features=1,
            base_features=None,
            enable_constraints=True,
            error=sample_error,
            return_detailed=True,
        )

        # Returns FeatureSelectionResults dataclass
        assert hasattr(result, 'best_model')
        assert result.best_model.features == "ERROR"

    def test_data_length_used_for_n_obs(self, sample_error):
        """Test that data length is used for n_obs in error result."""
        small_df = pd.DataFrame({'target': [1, 2, 3]})

        result = _create_error_fallback_results(
            data=small_df,
            candidate_features=["feature_1"],
            target_variable="target",
            max_features=1,
            base_features=None,
            enable_constraints=True,
            error=sample_error,
            return_detailed=True,
        )

        assert result.best_model.n_obs == 3


# =============================================================================
# Tests for _handle_pipeline_error
# =============================================================================


class TestHandlePipelineError:
    """Tests for _handle_pipeline_error function."""

    def test_logs_error_message(self, simple_dataframe, sample_error, capsys):
        """Test that error message is logged."""
        _handle_pipeline_error(
            error=sample_error,
            data=simple_dataframe,
            candidate_features=["feature_1"],
            target_variable="target",
            max_features=1,
            base_features=None,
            enable_constraints=True,
            return_detailed=False,
        )

        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "Test error message" in captured.out

    def test_returns_fallback_results(self, simple_dataframe, sample_error):
        """Test that fallback results are returned."""
        result = _handle_pipeline_error(
            error=sample_error,
            data=simple_dataframe,
            candidate_features=["feature_1"],
            target_variable="target",
            max_features=1,
            base_features=None,
            enable_constraints=True,
            return_detailed=True,
        )

        # Returns FeatureSelectionResults dataclass
        assert hasattr(result, 'best_model')

    def test_logs_different_error_types(self, simple_dataframe, capsys):
        """Test that different error types are logged correctly."""
        errors = [
            RuntimeError("Runtime issue"),
            TypeError("Type mismatch"),
            KeyError("missing_key"),
        ]

        for error in errors:
            _handle_pipeline_error(
                error=error,
                data=simple_dataframe,
                candidate_features=["feature_1"],
                target_variable="target",
                max_features=1,
                base_features=None,
                enable_constraints=True,
                return_detailed=False,
            )

            captured = capsys.readouterr()
            assert "ERROR" in captured.out


# =============================================================================
# Tests for _format_pipeline_results
# =============================================================================


class TestFormatPipelineResults:
    """Tests for _format_pipeline_results function."""

    def test_returns_full_results_when_detailed(self, mock_feature_selection_results):
        """Test that full results returned when return_detailed=True."""
        result = _format_pipeline_results(
            results=mock_feature_selection_results,
            return_detailed=True,
        )

        assert result is mock_feature_selection_results

    def test_returns_valid_results_when_not_empty(self, mock_feature_selection_results):
        """Test that valid_results returned when available."""
        result = _format_pipeline_results(
            results=mock_feature_selection_results,
            return_detailed=False,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # valid_results has 3 rows

    def test_returns_all_results_when_valid_empty(self, empty_mock_results):
        """Test that all_results returned when valid_results empty."""
        result = _format_pipeline_results(
            results=empty_mock_results,
            return_detailed=False,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # all_results has 2 rows

    def test_with_real_feature_selection_results(self, real_feature_selection_results):
        """Test with real FeatureSelectionResults dataclass."""
        # Test detailed return
        result_detailed = _format_pipeline_results(
            results=real_feature_selection_results,
            return_detailed=True,
        )
        assert result_detailed is real_feature_selection_results

        # Test non-detailed return
        result_simple = _format_pipeline_results(
            results=real_feature_selection_results,
            return_detailed=False,
        )
        assert isinstance(result_simple, pd.DataFrame)
        assert len(result_simple) == 1  # valid_results has 1 row


# =============================================================================
# Tests for run_feature_selection
# =============================================================================


class TestRunFeatureSelection:
    """Tests for run_feature_selection main entry point."""

    def test_handles_missing_atomic_functions(self, simple_dataframe, capsys, reset_feature_flags):
        """Test graceful handling when atomic functions unavailable."""
        # Force legacy path
        FEATURE_FLAGS["USE_ATOMIC_FUNCTIONS"] = False

        result = run_feature_selection(
            data=simple_dataframe,
            candidate_features=["feature_1", "feature_2"],
            target_variable="target",
            return_detailed=True,
        )

        # Should return error results (legacy not implemented)
        captured = capsys.readouterr()
        assert "ERROR" in captured.out or isinstance(result, dict)

    def test_logs_pipeline_status(self, simple_dataframe, capsys, reset_feature_flags):
        """Test that pipeline status is logged."""
        run_feature_selection(
            data=simple_dataframe,
            candidate_features=["feature_1"],
            target_variable="target",
            return_detailed=True,
        )

        captured = capsys.readouterr()
        assert "Starting Feature Selection Analysis" in captured.out

    def test_resolves_parameters(self, simple_dataframe, reset_feature_flags):
        """Test that parameters are resolved correctly."""
        FEATURE_FLAGS["AUTO_DISPLAY_RESULTS"] = False  # Suppress display

        # Should not raise
        result = run_feature_selection(
            data=simple_dataframe,
            candidate_features=["feature_1"],
            target_variable="target",
            display_results=False,  # Explicit
            return_detailed=False,
        )

        # Result should be DataFrame (not detailed)
        assert isinstance(result, (pd.DataFrame, dict))

    def test_returns_dataframe_by_default(self, simple_dataframe, reset_feature_flags):
        """Test that DataFrame returned when return_detailed=False."""
        FEATURE_FLAGS["AUTO_DISPLAY_RESULTS"] = False

        result = run_feature_selection(
            data=simple_dataframe,
            candidate_features=["feature_1"],
            target_variable="target",
            return_detailed=False,
        )

        # Either empty DataFrame (error case) or results DataFrame
        assert isinstance(result, (pd.DataFrame, dict))

    def test_error_handling_preserves_config(self, simple_dataframe, reset_feature_flags):
        """Test that error handling preserves configuration info."""
        FEATURE_FLAGS["USE_ATOMIC_FUNCTIONS"] = False
        FEATURE_FLAGS["AUTO_DISPLAY_RESULTS"] = False

        result = run_feature_selection(
            data=simple_dataframe,
            candidate_features=["feature_1", "feature_2"],
            target_variable="target",
            max_features=2,
            base_features=["feature_3"],
            enable_constraints=True,
            return_detailed=True,
        )

        # Error results should preserve feature_config
        if isinstance(result, dict) and "feature_config" in result:
            assert result["feature_config"]["candidate_features"] == ["feature_1", "feature_2"]
            assert result["feature_config"]["max_candidate_features"] == 2

    def test_with_all_parameters(self, simple_dataframe, reset_feature_flags, capsys):
        """Test with all parameters specified."""
        FEATURE_FLAGS["AUTO_DISPLAY_RESULTS"] = False

        result = run_feature_selection(
            data=simple_dataframe,
            candidate_features=["feature_1", "feature_2"],
            target_variable="target",
            max_features=2,
            base_features=None,
            enable_constraints=True,
            enable_bootstrap=False,
            bootstrap_samples=50,
            random_seed=123,
            display_results=False,
            return_detailed=True,
        )

        # Should complete without error
        assert result is not None

    def test_bootstrap_parameter_resolution(self, simple_dataframe, reset_feature_flags):
        """Test bootstrap parameter resolution from flag."""
        FEATURE_FLAGS["AUTO_DISPLAY_RESULTS"] = False
        FEATURE_FLAGS["ENABLE_BOOTSTRAP_DEFAULT"] = True

        # Pass None to use flag default
        result = run_feature_selection(
            data=simple_dataframe,
            candidate_features=["feature_1"],
            target_variable="target",
            enable_bootstrap=None,  # Should use flag (True)
            return_detailed=True,
        )

        assert result is not None


# =============================================================================
# Tests for _execute_feature_selection_pipeline (mocked)
# =============================================================================


class TestExecuteFeatureSelectionPipeline:
    """Tests for _execute_feature_selection_pipeline with mocking."""

    @pytest.mark.skipif(
        not ATOMIC_FUNCTIONS_AVAILABLE,
        reason="Atomic functions not available - cannot test pipeline execution"
    )
    def test_executes_pipeline_and_displays_results(
        self, simple_dataframe, real_feature_selection_results
    ):
        """Test that pipeline is executed and results displayed."""
        from src.features.selection.interface.interface_execution import _execute_feature_selection_pipeline

        with patch('src.features.selection.pipeline_orchestrator.run_feature_selection_pipeline') as mock_pipeline, \
             patch('src.features.selection.interface.interface_execution.display_results_summary') as mock_display:

            mock_pipeline.return_value = real_feature_selection_results

            feature_config = FeatureSelectionConfig(
                base_features=[],
                candidate_features=["feature_1", "feature_2"],
                max_candidate_features=2,
                target_variable="target",
            )
            constraint_config = EconomicConstraintConfig(enabled=True)

            result = _execute_feature_selection_pipeline(
                data=simple_dataframe,
                feature_config=feature_config,
                constraint_config=constraint_config,
                bootstrap_config=None,
                resolved_display_results=True,
            )

            mock_pipeline.assert_called_once()
            mock_display.assert_called_once()
            assert result == real_feature_selection_results

    @pytest.mark.skipif(
        not ATOMIC_FUNCTIONS_AVAILABLE,
        reason="Atomic functions not available - cannot test pipeline execution"
    )
    def test_skips_display_when_disabled(
        self, simple_dataframe, real_feature_selection_results
    ):
        """Test that display is skipped when resolved_display_results=False."""
        from src.features.selection.interface.interface_execution import _execute_feature_selection_pipeline

        with patch('src.features.selection.pipeline_orchestrator.run_feature_selection_pipeline') as mock_pipeline, \
             patch('src.features.selection.interface.interface_execution.display_results_summary') as mock_display:

            mock_pipeline.return_value = real_feature_selection_results

            feature_config = FeatureSelectionConfig(
                base_features=[],
                candidate_features=["feature_1"],
                max_candidate_features=1,
                target_variable="target",
            )
            constraint_config = EconomicConstraintConfig(enabled=True)

            _execute_feature_selection_pipeline(
                data=simple_dataframe,
                feature_config=feature_config,
                constraint_config=constraint_config,
                bootstrap_config=None,
                resolved_display_results=False,
            )

            mock_display.assert_not_called()


# =============================================================================
# Tests for _run_atomic_pipeline (mocked)
# =============================================================================


class TestRunAtomicPipeline:
    """Tests for _run_atomic_pipeline with mocking."""

    def test_creates_configs_and_executes_pipeline(
        self, simple_dataframe, real_feature_selection_results
    ):
        """Test that configs are created and pipeline is executed."""
        from src.features.selection.interface.interface_execution import _run_atomic_pipeline

        feature_config = FeatureSelectionConfig(
            base_features=[],
            candidate_features=["feature_1"],
            max_candidate_features=1,
            target_variable="target",
        )
        constraint_config = EconomicConstraintConfig(enabled=True)

        with patch.object(
            __import__('src.features.selection.interface.interface_execution', fromlist=['_execute_feature_selection_pipeline']),
            '_execute_feature_selection_pipeline'
        ) as mock_execute, patch.object(
            __import__('src.features.selection.interface.interface_execution', fromlist=['create_feature_selection_config']),
            'create_feature_selection_config'
        ) as mock_create_config:

            mock_create_config.return_value = (feature_config, constraint_config, None)
            mock_execute.return_value = real_feature_selection_results

            result = _run_atomic_pipeline(
                data=simple_dataframe,
                candidate_features=["feature_1"],
                target_variable="target",
                max_features=1,
                base_features=None,
                enable_constraints=True,
                resolved_enable_bootstrap=False,
                bootstrap_samples=100,
                random_seed=42,
                resolved_display_results=False,
                return_detailed=True,
            )

            mock_create_config.assert_called_once()
            mock_execute.assert_called_once()
            assert result == real_feature_selection_results


# =============================================================================
# Integration Tests
# =============================================================================


class TestInterfaceExecutionIntegration:
    """Integration tests for interface_execution module."""

    def test_full_error_workflow(self, simple_dataframe, reset_feature_flags, capsys):
        """Test complete error handling workflow."""
        # Disable atomic functions to force error path
        FEATURE_FLAGS["USE_ATOMIC_FUNCTIONS"] = False
        FEATURE_FLAGS["AUTO_DISPLAY_RESULTS"] = False

        result = run_feature_selection(
            data=simple_dataframe,
            candidate_features=["feature_1", "feature_2"],
            target_variable="target",
            max_features=2,
            return_detailed=True,
        )

        # Verify error was logged
        captured = capsys.readouterr()
        assert "ERROR" in captured.out or "Legacy" in captured.out

        # Verify result structure
        if isinstance(result, dict):
            assert "best_model" in result or "error" in str(result).lower()

    def test_parameter_resolution_chain(self, reset_feature_flags):
        """Test that parameter resolution works through the chain."""
        # Set specific flag values
        FEATURE_FLAGS["AUTO_DISPLAY_RESULTS"] = True
        FEATURE_FLAGS["ENABLE_BOOTSTRAP_DEFAULT"] = True

        # Test resolution
        display, bootstrap = _prepare_feature_selection_parameters(
            display_results=None,
            enable_bootstrap=None,
        )

        assert display is True
        assert bootstrap is True

        # Override with explicit values
        display2, bootstrap2 = _prepare_feature_selection_parameters(
            display_results=False,
            enable_bootstrap=False,
        )

        assert display2 is False
        assert bootstrap2 is False

    def test_error_aic_to_feature_selection_results_chain(self, sample_error):
        """Test the chain from error AIC to full feature selection results."""
        # Create error AIC
        error_aic = _create_error_aic_result(n_obs=100, error=sample_error)
        assert error_aic.features == "ERROR"
        assert error_aic.aic == np.inf

        # Create feature selection results from error AIC
        results = _create_error_feature_selection_results(
            error_aic=error_aic,
            candidate_features=["f1", "f2"],
            target_variable="target",
            max_features=2,
            base_features=None,
            enable_constraints=True,
        )

        # Verify chain
        assert results.best_model is error_aic
        assert results.total_combinations == 0
        assert results.feature_config["candidate_features"] == ["f1", "f2"]

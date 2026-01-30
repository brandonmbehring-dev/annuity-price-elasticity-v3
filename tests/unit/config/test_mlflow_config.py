"""
Tests for src.config.mlflow_config module.

Comprehensive tests for MLflow configuration and safe wrapper functions including:
- setup_environment_for_notebooks() - Environment configuration
- setup_mlflow_experiment() - Experiment creation and setup
- safe_mlflow_log_param() - Safe parameter logging
- safe_mlflow_log_metric() - Safe metric logging
- end_mlflow_experiment() - Run termination
- safe_mlflow_log_schema_validation() - Schema validation logging
- safe_mlflow_log_config_validation() - Config validation logging

Uses hybrid mock strategy:
- monkeypatch MLFLOW_AVAILABLE for unavailable scenarios
- @patch for call verification when MLflow is "available"

Coverage Target: 80%+ of mlflow_config.py
"""

import warnings
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config.mlflow_config import (
    MLFLOW_AVAILABLE,
    end_mlflow_experiment,
    safe_mlflow_log_config_validation,
    safe_mlflow_log_metric,
    safe_mlflow_log_param,
    safe_mlflow_log_schema_validation,
    setup_environment_for_notebooks,
    setup_mlflow_experiment,
)


# =============================================================================
# SESSION-SCOPED SETUP TO ENSURE CLEAN MLFLOW STATE
# =============================================================================


@pytest.fixture(autouse=True)
def ensure_clean_mlflow_state():
    """Ensure clean MLflow state before and after each test."""
    if MLFLOW_AVAILABLE:
        import mlflow
        # End any existing runs before test
        while mlflow.active_run() is not None:
            mlflow.end_run()

    yield

    if MLFLOW_AVAILABLE:
        import mlflow
        # Clean up after test
        while mlflow.active_run() is not None:
            mlflow.end_run()


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Standard DataFrame for schema validation tests."""
    return pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=100),
        "sales": [1000.0] * 100,
        "price": [5.0] * 100,
    })


@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    """Empty DataFrame for edge case tests."""
    return pd.DataFrame()


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Standard config dict for validation tests."""
    return {
        "model_type": "elastic_net",
        "alpha": 0.1,
        "l1_ratio": 0.5,
        "max_iter": 1000,
    }


# =============================================================================
# SETUP_ENVIRONMENT_FOR_NOTEBOOKS TESTS
# =============================================================================


class TestSetupEnvironmentForNotebooks:
    """Tests for setup_environment_for_notebooks()."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        result = setup_environment_for_notebooks()

        assert isinstance(result, dict)

    def test_contains_mlflow_available_key(self):
        """Should contain mlflow_available key."""
        result = setup_environment_for_notebooks()

        assert "mlflow_available" in result
        assert isinstance(result["mlflow_available"], bool)

    def test_contains_tracking_uri_key(self):
        """Should contain tracking_uri key."""
        result = setup_environment_for_notebooks()

        assert "tracking_uri" in result

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_tracking_uri_populated_when_mlflow_available(self):
        """tracking_uri should be populated when MLflow is available."""
        result = setup_environment_for_notebooks()

        assert result["mlflow_available"] is True
        assert result["tracking_uri"] is not None

    def test_returns_false_when_mlflow_unavailable(self, monkeypatch):
        """Should return mlflow_available=False when MLflow missing."""
        monkeypatch.setattr("src.config.mlflow_config.MLFLOW_AVAILABLE", False)

        result = setup_environment_for_notebooks()

        assert result["mlflow_available"] is False
        assert result["tracking_uri"] is None


# =============================================================================
# SETUP_MLFLOW_EXPERIMENT TESTS
# =============================================================================


class TestSetupMLflowExperiment:
    """Tests for setup_mlflow_experiment()."""

    def test_returns_none_when_mlflow_unavailable(self, monkeypatch):
        """Should return None when MLflow unavailable."""
        monkeypatch.setattr("src.config.mlflow_config.MLFLOW_AVAILABLE", False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = setup_mlflow_experiment("test_experiment")

            assert result is None
            assert len(w) == 1
            assert "MLflow not available" in str(w[0].message)

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_creates_experiment_with_valid_name(self):
        """Should create experiment with valid name."""
        result = setup_mlflow_experiment("test_experiment_valid")

        assert result is not None

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_returns_experiment_id(self):
        """Should return experiment_id string."""
        result = setup_mlflow_experiment("test_experiment_id")

        assert result is not None
        # experiment_id should be a string (may be numeric string)
        assert isinstance(result, str)

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_sets_custom_tracking_uri(self):
        """Should set custom tracking_uri when provided."""
        import mlflow

        original_uri = mlflow.get_tracking_uri()

        # Use a file-based URI for testing
        test_uri = "file:///tmp/mlflow_test"
        result = setup_mlflow_experiment("test_uri_experiment", tracking_uri=test_uri)

        # Restore original URI
        mlflow.set_tracking_uri(original_uri)

        assert result is not None

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_handles_existing_experiment(self):
        """Should handle already existing experiment gracefully."""
        # Create experiment twice
        setup_mlflow_experiment("duplicate_experiment_test")
        result = setup_mlflow_experiment("duplicate_experiment_test")

        # Should return same experiment_id, not crash
        assert result is not None

    def test_tags_parameter_accepted(self, monkeypatch):
        """Should accept tags parameter without crashing."""
        # Even with MLflow unavailable, the function signature should work
        monkeypatch.setattr("src.config.mlflow_config.MLFLOW_AVAILABLE", False)

        result = setup_mlflow_experiment(
            "test_with_tags", tags={"version": "1.0", "author": "test"}
        )

        assert result is None  # MLflow unavailable

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_handles_mlflow_exception(self):
        """Should handle MLflow exceptions gracefully."""
        with patch("src.config.mlflow_config.mlflow.set_experiment") as mock_set:
            mock_set.side_effect = Exception("MLflow error")

            # Should raise or return None depending on implementation
            # Current implementation lets exception propagate
            with pytest.raises(Exception):
                setup_mlflow_experiment("test_error_experiment")


# =============================================================================
# SAFE_MLFLOW_LOG_PARAM TESTS
# =============================================================================


class TestSafeMlflowLogParam:
    """Tests for safe_mlflow_log_param()."""

    def test_returns_false_when_mlflow_unavailable(self, monkeypatch):
        """Should return False when MLflow unavailable."""
        monkeypatch.setattr("src.config.mlflow_config.MLFLOW_AVAILABLE", False)

        result = safe_mlflow_log_param("test_key", "test_value")

        assert result is False

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_returns_true_on_success(self):
        """Should return True on successful log."""
        import mlflow

        # Start a run so we can log params
        with mlflow.start_run():
            result = safe_mlflow_log_param("test_param", "test_value")

        assert result is True

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_returns_false_with_warning_on_exception(self):
        """Should return False with warning on exception."""
        with patch("src.config.mlflow_config.mlflow.log_param") as mock_log:
            mock_log.side_effect = Exception("Log failed")

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = safe_mlflow_log_param("key", "value")

                assert result is False
                assert len(w) == 1
                assert "Failed to log param" in str(w[0].message)

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_handles_none_value(self):
        """Should handle None value."""
        import mlflow

        with mlflow.start_run():
            # None should be convertible to string
            result = safe_mlflow_log_param("null_param", None)

        assert result is True

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_handles_numeric_value(self):
        """Should handle numeric values."""
        import mlflow

        with mlflow.start_run():
            result = safe_mlflow_log_param("numeric_param", 42)

        assert result is True


# =============================================================================
# SAFE_MLFLOW_LOG_METRIC TESTS
# =============================================================================


class TestSafeMlflowLogMetric:
    """Tests for safe_mlflow_log_metric()."""

    def test_returns_false_when_mlflow_unavailable(self, monkeypatch):
        """Should return False when MLflow unavailable."""
        monkeypatch.setattr("src.config.mlflow_config.MLFLOW_AVAILABLE", False)

        result = safe_mlflow_log_metric("test_metric", 0.95)

        assert result is False

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_returns_true_on_success(self):
        """Should return True on successful log."""
        import mlflow

        with mlflow.start_run():
            result = safe_mlflow_log_metric("accuracy", 0.95)

        assert result is True

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_returns_false_with_warning_on_exception(self):
        """Should return False with warning on exception."""
        with patch("src.config.mlflow_config.mlflow.log_metric") as mock_log:
            mock_log.side_effect = Exception("Metric log failed")

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = safe_mlflow_log_metric("metric", 0.5)

                assert result is False
                assert len(w) == 1
                assert "Failed to log metric" in str(w[0].message)

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_step_parameter_works(self):
        """Should accept and use step parameter."""
        import mlflow

        with mlflow.start_run():
            result1 = safe_mlflow_log_metric("loss", 1.0, step=0)
            result2 = safe_mlflow_log_metric("loss", 0.5, step=1)
            result3 = safe_mlflow_log_metric("loss", 0.25, step=2)

        assert result1 is True
        assert result2 is True
        assert result3 is True

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_handles_integer_metric(self):
        """Should handle integer metric values."""
        import mlflow

        with mlflow.start_run():
            result = safe_mlflow_log_metric("count", 100)

        assert result is True


# =============================================================================
# END_MLFLOW_EXPERIMENT TESTS
# =============================================================================


class TestEndMlflowExperiment:
    """Tests for end_mlflow_experiment()."""

    def test_returns_false_when_mlflow_unavailable(self, monkeypatch):
        """Should return False when MLflow unavailable."""
        monkeypatch.setattr("src.config.mlflow_config.MLFLOW_AVAILABLE", False)

        result = end_mlflow_experiment()

        assert result is False

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_returns_true_on_success(self):
        """Should return True on success."""
        import mlflow

        mlflow.start_run()
        result = end_mlflow_experiment()

        assert result is True

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_returns_false_with_warning_on_exception(self):
        """Should return False with warning on exception."""
        with patch("src.config.mlflow_config.mlflow.end_run") as mock_end:
            mock_end.side_effect = Exception("End run failed")

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = end_mlflow_experiment()

                assert result is False
                assert len(w) == 1
                assert "Failed to end MLflow run" in str(w[0].message)

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_works_when_no_active_run(self):
        """Should work gracefully when no active run."""
        import mlflow

        # Ensure no active run
        mlflow.end_run()

        result = end_mlflow_experiment()

        # Should return True (end_run is idempotent in MLflow)
        assert result is True


# =============================================================================
# SAFE_MLFLOW_LOG_SCHEMA_VALIDATION TESTS
# =============================================================================


class TestSafeMlflowLogSchemaValidation:
    """Tests for safe_mlflow_log_schema_validation()."""

    def test_returns_dict_with_correct_keys(self, sample_dataframe):
        """Should return dict with correct keys."""
        result = safe_mlflow_log_schema_validation(
            sample_dataframe, "test_schema", validation_strict=True
        )

        assert isinstance(result, dict)
        assert "schema_name" in result
        assert "validation_strict" in result
        assert "row_count" in result
        assert "column_count" in result
        assert "logged_to_mlflow" in result
        assert "mlflow_status" in result

    def test_row_count_populated_from_dataframe(self, sample_dataframe):
        """Should populate row_count from DataFrame."""
        result = safe_mlflow_log_schema_validation(sample_dataframe, "test")

        assert result["row_count"] == 100

    def test_column_count_populated_from_dataframe(self, sample_dataframe):
        """Should populate column_count from DataFrame."""
        result = safe_mlflow_log_schema_validation(sample_dataframe, "test")

        assert result["column_count"] == 3

    def test_logged_to_mlflow_false_when_unavailable(
        self, sample_dataframe, monkeypatch
    ):
        """logged_to_mlflow should be False when MLflow unavailable."""
        monkeypatch.setattr("src.config.mlflow_config.MLFLOW_AVAILABLE", False)

        result = safe_mlflow_log_schema_validation(sample_dataframe, "test")

        assert result["logged_to_mlflow"] is False
        assert result["mlflow_status"] == "mlflow_not_available"

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_logged_to_mlflow_true_when_available(self, sample_dataframe):
        """logged_to_mlflow should be True when MLflow works."""
        import mlflow

        with mlflow.start_run():
            result = safe_mlflow_log_schema_validation(sample_dataframe, "test")

        assert result["logged_to_mlflow"] is True
        assert result["mlflow_status"] == "success"

    def test_works_with_empty_dataframe(self, empty_dataframe):
        """Should work with empty DataFrame."""
        result = safe_mlflow_log_schema_validation(empty_dataframe, "empty_schema")

        assert result["row_count"] == 0
        assert result["column_count"] == 0

    def test_schema_name_in_result(self, sample_dataframe):
        """Should include schema_name in result."""
        result = safe_mlflow_log_schema_validation(
            sample_dataframe, "my_custom_schema"
        )

        assert result["schema_name"] == "my_custom_schema"

    def test_validation_strict_in_result(self, sample_dataframe):
        """Should include validation_strict flag in result."""
        result_strict = safe_mlflow_log_schema_validation(
            sample_dataframe, "test", validation_strict=True
        )
        result_loose = safe_mlflow_log_schema_validation(
            sample_dataframe, "test", validation_strict=False
        )

        assert result_strict["validation_strict"] is True
        assert result_loose["validation_strict"] is False


# =============================================================================
# SAFE_MLFLOW_LOG_CONFIG_VALIDATION TESTS
# =============================================================================


class TestSafeMlflowLogConfigValidation:
    """Tests for safe_mlflow_log_config_validation()."""

    def test_returns_dict_with_correct_keys(self, sample_config):
        """Should return dict with correct keys."""
        result = safe_mlflow_log_config_validation(
            sample_config, "model_config", validation_passed=True
        )

        assert isinstance(result, dict)
        assert "config_name" in result
        assert "validation_passed" in result
        assert "config_keys" in result
        assert "logged_to_mlflow" in result
        assert "mlflow_status" in result

    def test_config_keys_extracted_from_dict(self, sample_config):
        """Should extract config_keys from dict."""
        result = safe_mlflow_log_config_validation(sample_config, "test")

        assert set(result["config_keys"]) == {"model_type", "alpha", "l1_ratio", "max_iter"}

    def test_validation_passed_reflected(self, sample_config):
        """Should reflect validation_passed correctly."""
        result_passed = safe_mlflow_log_config_validation(
            sample_config, "test", validation_passed=True
        )
        result_failed = safe_mlflow_log_config_validation(
            sample_config, "test", validation_passed=False
        )

        assert result_passed["validation_passed"] is True
        assert result_failed["validation_passed"] is False

    def test_logged_to_mlflow_flag_correct_when_unavailable(
        self, sample_config, monkeypatch
    ):
        """logged_to_mlflow should be False when MLflow unavailable."""
        monkeypatch.setattr("src.config.mlflow_config.MLFLOW_AVAILABLE", False)

        result = safe_mlflow_log_config_validation(sample_config, "test")

        assert result["logged_to_mlflow"] is False
        assert result["mlflow_status"] == "mlflow_not_available"

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_logged_to_mlflow_true_when_works(self, sample_config):
        """logged_to_mlflow should be True when logging succeeds."""
        import mlflow

        with mlflow.start_run():
            result = safe_mlflow_log_config_validation(sample_config, "test")

        assert result["logged_to_mlflow"] is True
        assert result["mlflow_status"] == "success"

    def test_handles_non_dict_config_gracefully(self):
        """Should handle non-dict config gracefully."""
        result = safe_mlflow_log_config_validation("not_a_dict", "test")

        assert result["config_keys"] == []

    def test_handles_empty_dict(self):
        """Should handle empty dict config."""
        result = safe_mlflow_log_config_validation({}, "empty_config")

        assert result["config_keys"] == []
        assert result["config_name"] == "empty_config"

    def test_config_name_in_result(self, sample_config):
        """Should include config_name in result."""
        result = safe_mlflow_log_config_validation(sample_config, "my_config_name")

        assert result["config_name"] == "my_config_name"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestMLflowIntegration:
    """Integration tests verifying full workflow."""

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_full_workflow(self, sample_dataframe, sample_config):
        """Test complete MLflow workflow: setup, log, end."""
        import mlflow

        # 1. Setup environment
        env_result = setup_environment_for_notebooks()
        assert env_result["mlflow_available"] is True

        # 2. Setup experiment
        exp_id = setup_mlflow_experiment("test_full_workflow")
        assert exp_id is not None

        # 3. Start run and log
        with mlflow.start_run():
            # Log params
            assert safe_mlflow_log_param("test_param", "value") is True

            # Log metrics
            assert safe_mlflow_log_metric("accuracy", 0.95) is True

            # Log schema validation
            schema_result = safe_mlflow_log_schema_validation(
                sample_dataframe, "test_schema"
            )
            assert schema_result["logged_to_mlflow"] is True

            # Log config validation
            config_result = safe_mlflow_log_config_validation(
                sample_config, "test_config"
            )
            assert config_result["logged_to_mlflow"] is True

        # 4. End experiment (after context manager already ended run)
        # This should still return True (idempotent)
        assert end_mlflow_experiment() is True

    def test_all_functions_work_when_mlflow_unavailable(
        self, sample_dataframe, sample_config, monkeypatch
    ):
        """All functions should work gracefully when MLflow unavailable."""
        monkeypatch.setattr("src.config.mlflow_config.MLFLOW_AVAILABLE", False)

        # Environment setup
        env_result = setup_environment_for_notebooks()
        assert env_result["mlflow_available"] is False

        # Experiment setup
        exp_result = setup_mlflow_experiment("test")
        assert exp_result is None

        # Log param
        assert safe_mlflow_log_param("key", "value") is False

        # Log metric
        assert safe_mlflow_log_metric("metric", 0.5) is False

        # End experiment
        assert end_mlflow_experiment() is False

        # Schema validation
        schema_result = safe_mlflow_log_schema_validation(sample_dataframe, "test")
        assert schema_result["logged_to_mlflow"] is False

        # Config validation
        config_result = safe_mlflow_log_config_validation(sample_config, "test")
        assert config_result["logged_to_mlflow"] is False

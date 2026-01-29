"""Tests for forecasting_orchestrator module.

Covers validation helpers, result initialization, and orchestration logic.
Target: 14% → 50% coverage
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.models.forecasting_orchestrator import (
    # Validation functions
    _validate_features_not_empty,
    _validate_features_exist_in_dataframe,
    _validate_target_variable,
    _validate_cutoff_bounds,
    validate_forecasting_inputs,
    # Helper functions
    _initialize_forecasting_results,
    _report_progress,
    _finalize_forecasting_results,
    # Config extraction
    _extract_pipeline_config,
    _print_phase_results,
    _calculate_comparison_metrics,
    # Main orchestrators (will be tested with mocks)
    run_benchmark_forecasting,
    run_bootstrap_ridge_forecasting,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_df():
    """Create sample time series dataframe."""
    dates = pd.date_range("2024-01-01", periods=150, freq="W")
    np.random.seed(42)
    return pd.DataFrame({
        "date": dates,
        "sales_target_current": np.random.uniform(100, 500, 150),
        "sales_target_contract_t5": np.random.uniform(80, 450, 150),
        "prudential_rate_current": np.random.uniform(3.0, 5.0, 150),
        "competitor_mid_t2": np.random.uniform(2.5, 4.5, 150),
        "competitor_top5_t3": np.random.uniform(2.5, 4.5, 150),
        "economic_indicator": np.random.uniform(0.5, 1.5, 150),
    })


@pytest.fixture
def model_features():
    """Standard model feature list."""
    return ["prudential_rate_current", "competitor_mid_t2", "competitor_top5_t3"]


@pytest.fixture
def benchmark_features():
    """Standard benchmark feature list."""
    return ["sales_target_contract_t5"]


@pytest.fixture
def sample_forecasting_config():
    """Sample forecasting configuration."""
    return {
        'cv_config': {
            'start_cutoff': 20,
            'end_cutoff': 150,
        },
        'bootstrap_model_config': {
            'alpha': 1.0,
            'positive_constraint': True,
        },
        'performance_monitoring_config': {
            'progress_reporting_interval': 25,
        },
        'forecasting_config': {
            'n_bootstrap_samples': 100,
            'random_state': 42,
        },
        'model_features': ['prudential_rate_current', 'competitor_mid_t2'],
        'benchmark_features': ['sales_target_contract_t5'],
        'target_variable': 'sales_target_current',
    }


# =============================================================================
# TESTS: _validate_features_not_empty (Lines 65-70)
# =============================================================================


class TestValidateFeaturesNotEmpty:
    """Tests for _validate_features_not_empty validator."""

    def test_valid_features_pass(self, model_features):
        """Valid non-empty feature list passes."""
        _validate_features_not_empty(model_features)  # No exception

    def test_empty_list_raises(self):
        """Empty list raises ValueError (Line 66)."""
        with pytest.raises(ValueError, match="Feature list is empty"):
            _validate_features_not_empty([])

    def test_none_raises(self):
        """None raises ValueError (Line 65)."""
        with pytest.raises(ValueError, match="Feature list is empty"):
            _validate_features_not_empty(None)


# =============================================================================
# TESTS: _validate_features_exist_in_dataframe (Lines 91-98)
# =============================================================================


class TestValidateFeaturesExistInDataframe:
    """Tests for _validate_features_exist_in_dataframe validator."""

    def test_valid_features_pass(self, sample_df, model_features):
        """Valid features in DataFrame pass."""
        _validate_features_exist_in_dataframe(sample_df, model_features)

    def test_missing_features_raises(self, sample_df):
        """Missing features raise KeyError (Lines 93-98)."""
        bad_features = ["prudential_rate_current", "nonexistent_feature"]
        with pytest.raises(KeyError, match="Missing required features"):
            _validate_features_exist_in_dataframe(sample_df, bad_features)

    def test_all_missing_raises(self, sample_df):
        """All missing features raise KeyError."""
        bad_features = ["completely_fake_1", "completely_fake_2"]
        with pytest.raises(KeyError, match="Missing required features"):
            _validate_features_exist_in_dataframe(sample_df, bad_features)


# =============================================================================
# TESTS: _validate_target_variable (Lines 119-129)
# =============================================================================


class TestValidateTargetVariable:
    """Tests for _validate_target_variable validator."""

    def test_valid_target_passes(self, sample_df):
        """Valid target variable passes."""
        _validate_target_variable(sample_df, "sales_target_current")

    def test_empty_target_allowed(self, sample_df):
        """Empty target string returns early (Line 120)."""
        _validate_target_variable(sample_df, "")  # No exception
        _validate_target_variable(sample_df, None)  # No exception

    def test_missing_target_raises(self, sample_df):
        """Missing target raises KeyError."""
        with pytest.raises(KeyError):
            _validate_target_variable(sample_df, "nonexistent_target")


# =============================================================================
# TESTS: _validate_cutoff_bounds (Lines 152-167)
# =============================================================================


class TestValidateCutoffBounds:
    """Tests for _validate_cutoff_bounds validator."""

    def test_valid_bounds_pass(self):
        """Valid cutoff bounds pass."""
        _validate_cutoff_bounds(n_obs=150, start_cutoff=20, end_cutoff=150)

    def test_end_cutoff_exceeds_raises(self):
        """end_cutoff > n_obs raises ValueError (Lines 153-157)."""
        with pytest.raises(ValueError, match="end_cutoff.*exceeds dataset size"):
            _validate_cutoff_bounds(n_obs=100, start_cutoff=20, end_cutoff=150)

    def test_single_forecast_raises(self):
        """R² requires at least 2 forecasts (Lines 163-167)."""
        with pytest.raises(ValueError, match="R² calculation requires at least 2 forecasts"):
            _validate_cutoff_bounds(n_obs=150, start_cutoff=50, end_cutoff=51)

    def test_zero_forecasts_allowed(self):
        """Zero forecasts allowed (no R² calculated)."""
        # start >= end means zero forecasts, which is valid
        _validate_cutoff_bounds(n_obs=150, start_cutoff=100, end_cutoff=100)

    def test_many_forecasts_pass(self):
        """Many forecasts pass."""
        _validate_cutoff_bounds(n_obs=150, start_cutoff=20, end_cutoff=120)


# =============================================================================
# TESTS: validate_forecasting_inputs (Lines 200-203)
# =============================================================================


class TestValidateForecastingInputs:
    """Tests for main validate_forecasting_inputs orchestrator."""

    def test_valid_inputs_pass(self, sample_df, model_features):
        """Valid inputs pass all validation."""
        validate_forecasting_inputs(
            df=sample_df,
            start_cutoff=20,
            end_cutoff=150,
            features=model_features,
            target_variable="sales_target_current"
        )

    def test_cascades_feature_validation(self, sample_df):
        """Cascades to feature validation (Line 200)."""
        with pytest.raises(ValueError, match="Feature list is empty"):
            validate_forecasting_inputs(
                df=sample_df,
                start_cutoff=20,
                end_cutoff=150,
                features=[],
                target_variable="sales_target_current"
            )

    def test_cascades_feature_existence(self, sample_df):
        """Cascades to feature existence validation (Line 201)."""
        with pytest.raises(KeyError, match="Missing required features"):
            validate_forecasting_inputs(
                df=sample_df,
                start_cutoff=20,
                end_cutoff=150,
                features=["nonexistent"],
                target_variable="sales_target_current"
            )

    def test_cascades_cutoff_validation(self, sample_df, model_features):
        """Cascades to cutoff validation (Line 203)."""
        with pytest.raises(ValueError, match="end_cutoff.*exceeds dataset size"):
            validate_forecasting_inputs(
                df=sample_df,
                start_cutoff=20,
                end_cutoff=200,  # Exceeds 150
                features=model_features,
                target_variable="sales_target_current"
            )


# =============================================================================
# TESTS: _initialize_forecasting_results (Lines 251-259)
# =============================================================================


class TestInitializeForecastingResults:
    """Tests for _initialize_forecasting_results helper."""

    def test_returns_correct_structure(self):
        """Returns dict with all required keys."""
        results = _initialize_forecasting_results()

        expected_keys = ['dates', 'y_true', 'y_predict', 'abs_pct_error',
                         'bootstrap_predictions', 'cutoffs', 'errors']
        assert set(results.keys()) == set(expected_keys)

    def test_lists_empty(self):
        """All list values are empty."""
        results = _initialize_forecasting_results()

        assert results['dates'] == []
        assert results['y_true'] == []
        assert results['y_predict'] == []
        assert results['abs_pct_error'] == []
        assert results['cutoffs'] == []
        assert results['errors'] == []

    def test_bootstrap_predictions_empty_dict(self):
        """Bootstrap predictions is empty dict."""
        results = _initialize_forecasting_results()
        assert results['bootstrap_predictions'] == {}


# =============================================================================
# TESTS: _report_progress (Lines 282-285)
# =============================================================================


class TestReportProgress:
    """Tests for _report_progress helper."""

    def test_reports_at_interval(self, capsys):
        """Reports progress at specified intervals (Line 282)."""
        abs_pct_errors = [0.10, 0.15, 0.12, 0.08, 0.20]

        # At interval of 5, should print at index 4 (5th forecast)
        _report_progress(cutoff_idx=4, n_forecasts=100,
                        abs_pct_errors=abs_pct_errors, progress_interval=5)

        captured = capsys.readouterr()
        assert "Progress:" in captured.out
        assert "5.0%" in captured.out  # 5/100 = 5%
        assert "MAPE:" in captured.out

    def test_silent_between_intervals(self, capsys):
        """No output between intervals."""
        abs_pct_errors = [0.10]

        # At index 2 with interval 5, should not print
        _report_progress(cutoff_idx=2, n_forecasts=100,
                        abs_pct_errors=abs_pct_errors, progress_interval=5)

        captured = capsys.readouterr()
        assert captured.out == ""


# =============================================================================
# TESTS: _finalize_forecasting_results (Lines 289-300)
# =============================================================================


class TestFinalizeForecastingResults:
    """Tests for _finalize_forecasting_results helper."""

    def test_calculates_metrics_on_success(self):
        """Calculates metrics when success_count > 0 (Lines 290-297)."""
        results = {
            'y_true': [100, 150, 200, 180, 220],
            'y_predict': [105, 145, 195, 185, 215],
        }

        _finalize_forecasting_results(results, success_count=5)

        assert 'metrics' in results
        assert 'r2_score' in results['metrics']
        assert 'mape' in results['metrics']
        assert results['n_forecasts'] == 5

    def test_default_metrics_on_zero_success(self):
        """Returns default metrics when success_count = 0 (Lines 298-300)."""
        results = {'y_true': [], 'y_predict': []}

        _finalize_forecasting_results(results, success_count=0)

        assert results['metrics']['r2_score'] == 0.0
        assert results['metrics']['mape'] == 100.0
        assert results['n_forecasts'] == 0


# =============================================================================
# TESTS: _extract_pipeline_config (Lines 493-506)
# =============================================================================


class TestExtractPipelineConfig:
    """Tests for _extract_pipeline_config helper."""

    def test_extracts_all_fields(self, sample_forecasting_config):
        """Extracts all configuration fields correctly."""
        config = _extract_pipeline_config(sample_forecasting_config, n_observations=150)

        assert config['start_cutoff'] == 20
        assert config['end_cutoff'] == 150
        assert config['n_bootstrap_samples'] == 100
        assert config['random_state'] == 42
        assert config['target_variable'] == 'sales_target_current'
        assert 'model_features' in config
        assert 'benchmark_features' in config

    def test_uses_n_observations_when_end_cutoff_none(self):
        """Uses n_observations when end_cutoff is None."""
        config = {
            'cv_config': {'start_cutoff': 20, 'end_cutoff': None},
            'bootstrap_model_config': {},
            'performance_monitoring_config': {},
            'forecasting_config': {'n_bootstrap_samples': 100, 'random_state': 42},
        }

        result = _extract_pipeline_config(config, n_observations=200)
        assert result['end_cutoff'] == 200


# =============================================================================
# TESTS: _print_phase_results (Lines 511-514)
# =============================================================================


class TestPrintPhaseResults:
    """Tests for _print_phase_results helper."""

    def test_prints_phase_summary(self, capsys):
        """Prints formatted phase results."""
        results = {
            'n_forecasts': 130,
            'metrics': {'r2_score': 0.782598, 'mape': 12.74}
        }

        _print_phase_results("Model", results)

        captured = capsys.readouterr()
        assert "Model Complete" in captured.out
        assert "130" in captured.out
        assert "0.782598" in captured.out
        assert "12.74" in captured.out


# =============================================================================
# TESTS: _calculate_comparison_metrics (Lines 519-528)
# =============================================================================


class TestCalculateComparisonMetrics:
    """Tests for _calculate_comparison_metrics helper."""

    def test_calculates_improvements(self):
        """Calculates MAPE and R² improvements correctly."""
        benchmark_results = {
            'metrics': {'r2_score': 0.575, 'mape': 16.40}
        }
        model_results = {
            'metrics': {'r2_score': 0.783, 'mape': 12.74}
        }

        comparison = _calculate_comparison_metrics(benchmark_results, model_results)

        # MAPE improvement: (16.40 - 12.74) / 16.40 * 100 ≈ 22.3%
        assert comparison['mape_improvement_pct'] > 20
        # R² improvement: (0.783 - 0.575) / 0.575 * 100 ≈ 36.2%
        assert comparison['r2_improvement_pct'] > 35
        assert comparison['model_outperforms'] is True

    def test_model_underperforms(self):
        """Identifies when model underperforms benchmark."""
        benchmark_results = {
            'metrics': {'r2_score': 0.80, 'mape': 10.0}
        }
        model_results = {
            'metrics': {'r2_score': 0.75, 'mape': 15.0}  # Worse
        }

        comparison = _calculate_comparison_metrics(benchmark_results, model_results)

        assert comparison['model_outperforms'] is False
        assert comparison['mape_improvement_pct'] < 0

    def test_handles_zero_benchmark_r2(self):
        """Handles zero benchmark R² without division by zero."""
        benchmark_results = {
            'metrics': {'r2_score': 0.0, 'mape': 20.0}
        }
        model_results = {
            'metrics': {'r2_score': 0.5, 'mape': 15.0}
        }

        # Should not raise
        comparison = _calculate_comparison_metrics(benchmark_results, model_results)
        assert np.isfinite(comparison['r2_improvement_pct'])


# =============================================================================
# TESTS: run_benchmark_forecasting (with mocks)
# =============================================================================


class TestRunBenchmarkForecasting:
    """Tests for run_benchmark_forecasting orchestrator."""

    @patch("src.models.forecasting_orchestrator.resample")
    @patch("src.models.forecasting_orchestrator.extract_test_target_contract_date_atomic")
    @patch("src.models.forecasting_orchestrator.calculate_prediction_error_atomic")
    def test_runs_for_all_cutoffs(
        self, mock_error, mock_extract, mock_resample,
        sample_df, benchmark_features
    ):
        """Runs benchmark forecasting for all cutoffs."""
        mock_extract.return_value = 150.0
        mock_resample.return_value = pd.Series([140, 145, 150, 155])
        mock_error.return_value = {'percentage_error': 5.0}

        results = run_benchmark_forecasting(
            df=sample_df,
            start_cutoff=20,
            end_cutoff=25,  # 5 forecasts
            benchmark_features=benchmark_features,
            n_bootstrap_samples=10,
            progress_interval=100  # Suppress progress output
        )

        assert 'metrics' in results
        assert 'n_forecasts' in results
        assert results['n_forecasts'] <= 5

    def test_validation_failure_raises(self, sample_df):
        """Validation failure raises error."""
        with pytest.raises(ValueError, match="Feature list is empty"):
            run_benchmark_forecasting(
                df=sample_df,
                start_cutoff=20,
                end_cutoff=30,
                benchmark_features=[],  # Invalid
                n_bootstrap_samples=10
            )


# =============================================================================
# TESTS: run_bootstrap_ridge_forecasting (with mocks)
# =============================================================================


class TestRunBootstrapRidgeForecasting:
    """Tests for run_bootstrap_ridge_forecasting orchestrator."""

    @patch("src.models.forecasting_orchestrator.validate_bootstrap_predictions_atomic")
    @patch("src.models.forecasting_orchestrator.predict_bootstrap_ensemble_atomic")
    @patch("src.models.forecasting_orchestrator.fit_bootstrap_ensemble_atomic")
    @patch("src.models.forecasting_orchestrator.prepare_cutoff_data_complete")
    @patch("src.models.forecasting_orchestrator.calculate_prediction_error_atomic")
    def test_runs_for_all_cutoffs(
        self, mock_error, mock_prepare, mock_fit, mock_predict, mock_validate,
        sample_df, model_features
    ):
        """Runs Ridge forecasting for all cutoffs."""
        # Mock data preparation
        mock_prepare.return_value = {
            'X_train': np.random.rand(10, 3),
            'y_train': np.random.uniform(100, 500, 10),
            'X_test': np.random.rand(1, 3),
            'y_test': 200.0,
            'weights': np.ones(10)
        }
        mock_fit.return_value = [MagicMock() for _ in range(10)]
        mock_predict.return_value = np.random.uniform(4.5, 6.5, 10)  # Log scale
        mock_validate.return_value = {'all_positive': True, 'count_matches': True}
        mock_error.return_value = {'percentage_error': 8.0}

        results = run_bootstrap_ridge_forecasting(
            df=sample_df,
            start_cutoff=20,
            end_cutoff=25,  # 5 forecasts
            model_features=model_features,
            target_variable="sales_target_current",
            sign_correction_config={},
            bootstrap_config={'positive_constraint': True},
            forecasting_config={'n_bootstrap_samples': 10},
            progress_interval=100
        )

        assert 'metrics' in results
        assert 'n_forecasts' in results

    def test_validation_failure_raises(self, sample_df, model_features):
        """Validation failure raises error."""
        with pytest.raises(ValueError, match="end_cutoff.*exceeds"):
            run_bootstrap_ridge_forecasting(
                df=sample_df,
                start_cutoff=20,
                end_cutoff=200,  # Exceeds dataset size
                model_features=model_features,
                target_variable="sales_target_current",
                sign_correction_config={},
                bootstrap_config={},
                forecasting_config={'n_bootstrap_samples': 10}
            )


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestOrchestratorIntegration:
    """Integration tests for orchestrator coordination."""

    def test_initialize_and_finalize_workflow(self):
        """Test initialize → process → finalize workflow."""
        # 1. Initialize
        results = _initialize_forecasting_results()
        assert len(results['y_true']) == 0

        # 2. Simulate adding results
        results['y_true'].extend([100, 150, 200])
        results['y_predict'].extend([105, 145, 210])
        results['abs_pct_error'].extend([0.05, 0.033, 0.05])
        results['dates'].extend(['2024-01-01', '2024-01-08', '2024-01-15'])
        results['cutoffs'].extend([20, 21, 22])

        # 3. Finalize
        _finalize_forecasting_results(results, success_count=3)

        assert results['n_forecasts'] == 3
        assert 'r2_score' in results['metrics']
        assert results['metrics']['mape'] > 0

    def test_comparison_workflow(self):
        """Test benchmark → model → comparison workflow."""
        benchmark = {
            'metrics': {'r2_score': 0.575, 'mape': 16.4},
            'n_forecasts': 129
        }
        model = {
            'metrics': {'r2_score': 0.783, 'mape': 12.7},
            'n_forecasts': 129
        }

        comparison = _calculate_comparison_metrics(benchmark, model)

        assert comparison['model_outperforms'] is True
        assert comparison['mape_improvement_pct'] > 20

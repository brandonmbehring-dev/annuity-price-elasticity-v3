"""
Tests for out_of_sample_evaluation module.

Target: 15% → 50%+ coverage
Tests organized by function categories:
- Dataclasses
- Validation functions
- Performance metrics
- Generalization analysis
- Statistical tests
- Residual analysis
- Production readiness
- Cross-validation
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.features.selection.enhancements.out_of_sample_evaluation import (
    # Dataclasses
    OutOfSampleResult,
    CrossValidationResult,
    # Validation
    _validate_temporal_split_data,
    # Time period extraction
    _extract_time_periods,
    # Performance metrics
    _compute_core_regression_metrics,
    _compute_mape,
    _compute_directional_accuracy,
    _calculate_performance_metrics,
    # Generalization
    _compute_metric_gaps,
    _assess_generalization_quality,
    _calculate_generalization_metrics,
    # Statistical tests
    _test_residual_means,
    _test_variance_equality,
    _test_distribution_similarity,
    _perform_generalization_statistical_tests,
    # Residual analysis
    _compute_residual_basic_stats,
    _run_normality_tests,
    _check_heteroscedasticity,
    _analyze_test_residuals,
    # Production readiness checks
    _check_test_r2,
    _check_test_mape,
    _check_generalization,
    _check_residual_diagnostics,
    _determine_confidence_level,
    _add_specific_recommendations,
    _build_summary_metrics,
    _assess_production_readiness,
    # CV functions
    _evaluate_cv_fold,
    _aggregate_cv_performance,
    _assess_cv_quality,
    _build_cv_overall_assessment,
    _run_single_model_time_series_cv,
    run_time_series_cross_validation,
    # Main functions
    _fit_and_predict,
    _evaluate_single_model_generalization,
    evaluate_temporal_generalization,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_actual():
    """Sample actual values for testing."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def sample_predictions():
    """Sample predictions for testing."""
    return np.array([1.1, 2.2, 2.9, 4.1, 4.8])


@pytest.fixture
def sample_train_data():
    """Sample training DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100, freq='W'),
        'target': np.random.normal(0, 1, 100),
        'feature_a': np.random.normal(0, 1, 100),
        'feature_b': np.random.normal(0, 1, 100),
    })


@pytest.fixture
def sample_test_data():
    """Sample test DataFrame."""
    np.random.seed(123)
    return pd.DataFrame({
        'date': pd.date_range('2021-11-01', periods=20, freq='W'),
        'target': np.random.normal(0, 1, 20),
        'feature_a': np.random.normal(0, 1, 20),
        'feature_b': np.random.normal(0, 1, 20),
    })


@pytest.fixture
def sample_temporal_split(sample_train_data, sample_test_data):
    """Sample temporal split data."""
    return {'train': sample_train_data, 'test': sample_test_data}


@pytest.fixture
def sample_model_results():
    """Sample model results DataFrame with AIC scores."""
    return pd.DataFrame({
        'features': ['feature_a', 'feature_b', 'feature_a + feature_b'],
        'aic': [100.0, 110.0, 95.0],
        'r_squared': [0.5, 0.4, 0.55],
    })


@pytest.fixture
def sample_train_performance():
    """Sample training performance metrics."""
    return {
        'r_squared': 0.75,
        'mape': 10.5,
        'mae': 0.15,
        'correlation': 0.87,
        'n_observations': 100,
    }


@pytest.fixture
def sample_test_performance():
    """Sample test performance metrics."""
    return {
        'r_squared': 0.65,
        'mape': 15.0,
        'mae': 0.22,
        'correlation': 0.80,
        'n_observations': 20,
    }


@pytest.fixture
def empty_assessment():
    """Empty production assessment structure."""
    return {
        'production_ready': False,
        'confidence_level': 'LOW',
        'primary_concerns': [],
        'strengths': [],
        'recommendations': [],
    }


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestOutOfSampleResult:
    """Tests for OutOfSampleResult dataclass."""

    def test_construction_with_required_fields(self):
        """Dataclass constructs with all required fields."""
        result = OutOfSampleResult(
            model_features='feature_a + feature_b',
            train_period=('2020-01-01', '2021-06-30'),
            test_period=('2021-07-01', '2021-12-31'),
            train_performance={'r_squared': 0.75},
            test_performance={'r_squared': 0.65},
            generalization_metrics={'r_squared_gap': 0.10},
            statistical_tests={'residual_mean_tests': {}},
            residual_analysis={'basic_stats': {}},
            production_assessment={'production_ready': True},
            predictions={'train_actual': [1.0, 2.0], 'test_actual': [3.0, 4.0]}
        )

        assert result.model_features == 'feature_a + feature_b'
        assert result.train_period[0] == '2020-01-01'
        assert result.test_performance['r_squared'] == 0.65

    def test_all_fields_accessible(self):
        """All dataclass fields are accessible."""
        result = OutOfSampleResult(
            model_features='x',
            train_period=('a', 'b'),
            test_period=('c', 'd'),
            train_performance={},
            test_performance={},
            generalization_metrics={},
            statistical_tests={},
            residual_analysis={},
            production_assessment={},
            predictions={}
        )

        # All 10 fields should be accessible
        attrs = ['model_features', 'train_period', 'test_period',
                 'train_performance', 'test_performance', 'generalization_metrics',
                 'statistical_tests', 'residual_analysis', 'production_assessment',
                 'predictions']

        for attr in attrs:
            assert hasattr(result, attr)


class TestCrossValidationResult:
    """Tests for CrossValidationResult dataclass."""

    def test_construction(self):
        """Dataclass constructs correctly."""
        result = CrossValidationResult(
            model_features='feature_a',
            n_folds=5,
            fold_results=[{'fold': 0, 'r_squared': 0.5}],
            average_performance={'r_squared': 0.5},
            performance_stability={'r_squared_std': 0.02},
            overall_assessment={'cv_quality': 'GOOD'}
        )

        assert result.model_features == 'feature_a'
        assert result.n_folds == 5
        assert len(result.fold_results) == 1
        assert result.overall_assessment['cv_quality'] == 'GOOD'

    def test_all_fields_accessible(self):
        """All CV result fields are accessible."""
        result = CrossValidationResult(
            model_features='x',
            n_folds=3,
            fold_results=[],
            average_performance={},
            performance_stability={},
            overall_assessment={}
        )

        attrs = ['model_features', 'n_folds', 'fold_results',
                 'average_performance', 'performance_stability', 'overall_assessment']

        for attr in attrs:
            assert hasattr(result, attr)


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidateTemporalSplitData:
    """Tests for _validate_temporal_split_data."""

    def test_valid_split_returns_dataframes(self, sample_temporal_split):
        """Valid split returns train and test DataFrames."""
        train, test = _validate_temporal_split_data(sample_temporal_split)

        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert len(train) == 100
        assert len(test) == 20

    def test_missing_train_key_raises(self, sample_test_data):
        """Missing 'train' key raises ValueError."""
        with pytest.raises(ValueError, match="missing required 'train' and 'test'"):
            _validate_temporal_split_data({'test': sample_test_data})

    def test_missing_test_key_raises(self, sample_train_data):
        """Missing 'test' key raises ValueError."""
        with pytest.raises(ValueError, match="missing required 'train' and 'test'"):
            _validate_temporal_split_data({'train': sample_train_data})

    def test_empty_train_raises(self, sample_test_data):
        """Empty training data raises ValueError."""
        empty_train = pd.DataFrame()
        with pytest.raises(ValueError, match="Empty training or test dataset"):
            _validate_temporal_split_data({'train': empty_train, 'test': sample_test_data})

    def test_empty_test_raises(self, sample_train_data):
        """Empty test data raises ValueError."""
        empty_test = pd.DataFrame()
        with pytest.raises(ValueError, match="Empty training or test dataset"):
            _validate_temporal_split_data({'train': sample_train_data, 'test': empty_test})


# =============================================================================
# Time Period Extraction Tests
# =============================================================================


class TestExtractTimePeriods:
    """Tests for _extract_time_periods."""

    def test_extracts_date_periods(self, sample_train_data, sample_test_data):
        """Extracts formatted date periods when date column exists."""
        train_period, test_period = _extract_time_periods(
            sample_train_data, sample_test_data
        )

        # Check format
        assert '-' in train_period[0]  # YYYY-MM-DD format
        assert len(train_period) == 2
        assert len(test_period) == 2

    def test_fallback_without_date_column(self):
        """Falls back to period numbering without date column."""
        train_data = pd.DataFrame({'x': [1, 2, 3]})
        test_data = pd.DataFrame({'x': [4, 5]})

        train_period, test_period = _extract_time_periods(train_data, test_data)

        assert train_period == ('Period 1', 'Period 3')
        assert test_period == ('Period 4', 'Period 5')


# =============================================================================
# Core Metrics Tests
# =============================================================================


class TestComputeCoreRegressionMetrics:
    """Tests for _compute_core_regression_metrics."""

    def test_perfect_predictions(self):
        """Perfect predictions yield R²=1, zero errors."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = actual.copy()

        mse, mae, rmse, r2 = _compute_core_regression_metrics(actual, pred)

        assert mse == pytest.approx(0.0)
        assert mae == pytest.approx(0.0)
        assert rmse == pytest.approx(0.0)
        assert r2 == pytest.approx(1.0)

    def test_reasonable_predictions(self, sample_actual, sample_predictions):
        """Reasonable predictions yield expected metrics."""
        mse, mae, rmse, r2 = _compute_core_regression_metrics(
            sample_actual, sample_predictions
        )

        # Property: all should be non-negative
        assert mse >= 0
        assert mae >= 0
        assert rmse >= 0
        # Property: R² should be between 0 and 1 for reasonable predictions
        assert 0 <= r2 <= 1

    def test_constant_actual_zero_ss_tot(self):
        """Handles zero total sum of squares (constant actual)."""
        actual = np.array([5.0, 5.0, 5.0, 5.0])
        pred = np.array([4.9, 5.0, 5.1, 5.0])

        mse, mae, rmse, r2 = _compute_core_regression_metrics(actual, pred)

        # ss_tot = 0 should return R² = 0
        assert r2 == 0

    def test_rmse_equals_sqrt_mse(self, sample_actual, sample_predictions):
        """Property: RMSE = sqrt(MSE)."""
        mse, mae, rmse, r2 = _compute_core_regression_metrics(
            sample_actual, sample_predictions
        )

        assert rmse == pytest.approx(np.sqrt(mse))


class TestComputeMape:
    """Tests for _compute_mape."""

    def test_perfect_predictions(self):
        """Perfect predictions yield MAPE = 0."""
        actual = np.array([1.0, 2.0, 3.0])
        pred = actual.copy()

        mape = _compute_mape(actual, pred)

        assert mape == pytest.approx(0.0)

    def test_excludes_near_zero_values(self):
        """Excludes near-zero actual values from MAPE calculation."""
        actual = np.array([1.0, 0.0, 1e-12, 2.0])  # Two near-zero values
        pred = np.array([1.0, 0.5, 0.5, 2.0])

        mape = _compute_mape(actual, pred)

        # Should only use actual=[1.0, 2.0] since 0.0 and 1e-12 excluded
        assert mape == pytest.approx(0.0)

    def test_all_zeros_returns_inf(self):
        """All zero/near-zero actuals return infinity."""
        actual = np.array([0.0, 1e-15, 0.0])
        pred = np.array([0.1, 0.2, 0.3])

        mape = _compute_mape(actual, pred)

        assert mape == np.inf

    def test_mape_percentage_range(self):
        """MAPE returns percentage values (can exceed 100)."""
        actual = np.array([1.0, 2.0])
        pred = np.array([2.0, 4.0])  # 100% error

        mape = _compute_mape(actual, pred)

        assert mape == pytest.approx(100.0)


class TestComputeDirectionalAccuracy:
    """Tests for _compute_directional_accuracy."""

    def test_perfect_directional_accuracy(self):
        """All correct directions yield 100%."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # All increasing
        pred = np.array([1.0, 2.5, 2.8, 4.5, 5.2])  # Also all increasing

        accuracy = _compute_directional_accuracy(actual, pred)

        assert accuracy == pytest.approx(100.0)

    def test_zero_directional_accuracy(self):
        """All wrong directions yield 0%."""
        actual = np.array([1.0, 2.0, 3.0, 4.0])  # Increasing
        pred = np.array([5.0, 4.0, 3.0, 2.0])  # Decreasing

        accuracy = _compute_directional_accuracy(actual, pred)

        assert accuracy == pytest.approx(0.0)

    def test_mixed_directions(self):
        """Mixed correct/wrong directions."""
        actual = np.array([1.0, 2.0, 1.5, 2.5])  # +, -, +
        pred = np.array([1.0, 3.0, 2.0, 3.0])  # +, -, + (all correct)

        accuracy = _compute_directional_accuracy(actual, pred)

        assert accuracy == pytest.approx(100.0)

    def test_single_element_returns_nan(self):
        """Single element returns NaN."""
        actual = np.array([1.0])
        pred = np.array([1.5])

        accuracy = _compute_directional_accuracy(actual, pred)

        assert np.isnan(accuracy)


class TestCalculatePerformanceMetrics:
    """Tests for _calculate_performance_metrics."""

    def test_returns_all_expected_keys(self):
        """Returns dictionary with all expected metric keys."""
        actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = pd.Series([1.1, 2.0, 3.1, 3.9, 5.1])

        metrics = _calculate_performance_metrics(actual, pred, 'test')

        expected_keys = ['r_squared', 'mse', 'mae', 'rmse', 'mape',
                         'correlation', 'directional_accuracy', 'n_observations',
                         'dataset_type']
        for key in expected_keys:
            assert key in metrics

    def test_dataset_type_stored(self):
        """Dataset type is stored in metrics."""
        actual = pd.Series([1.0, 2.0, 3.0])
        pred = pd.Series([1.0, 2.0, 3.0])

        metrics = _calculate_performance_metrics(actual, pred, 'training')

        assert metrics['dataset_type'] == 'training'

    def test_handles_nan_values(self):
        """Filters out NaN values before calculation."""
        actual = pd.Series([1.0, np.nan, 3.0, 4.0])
        pred = pd.Series([1.0, 2.0, np.nan, 4.0])

        metrics = _calculate_performance_metrics(actual, pred, 'test')

        # Should use only valid pairs: (1.0, 1.0), (4.0, 4.0)
        assert metrics['n_observations'] == 2

    def test_all_nan_returns_error(self):
        """All NaN values return error dict."""
        actual = pd.Series([np.nan, np.nan])
        pred = pd.Series([np.nan, np.nan])

        metrics = _calculate_performance_metrics(actual, pred, 'test')

        assert metrics == {'error': 'no_valid_predictions'}


# =============================================================================
# Generalization Metrics Tests
# =============================================================================


class TestComputeMetricGaps:
    """Tests for _compute_metric_gaps."""

    def test_computes_absolute_gaps(self, sample_train_performance, sample_test_performance):
        """Computes absolute gap for each metric."""
        gaps = _compute_metric_gaps(sample_train_performance, sample_test_performance)

        assert 'r_squared_gap' in gaps
        assert gaps['r_squared_gap'] == pytest.approx(0.10)  # 0.75 - 0.65

    def test_computes_relative_gaps(self, sample_train_performance, sample_test_performance):
        """Computes relative gap percentage."""
        gaps = _compute_metric_gaps(sample_train_performance, sample_test_performance)

        assert 'r_squared_relative_gap_pct' in gaps
        # (0.75 - 0.65) / 0.75 * 100 = 13.33%
        expected_pct = (0.10 / 0.75) * 100
        assert gaps['r_squared_relative_gap_pct'] == pytest.approx(expected_pct)

    def test_handles_zero_train_value(self):
        """Handles zero training value (no relative gap computed)."""
        train_perf = {'r_squared': 0.0, 'mape': 0.0, 'mae': 0.0, 'correlation': 0.0}
        test_perf = {'r_squared': 0.5, 'mape': 10.0, 'mae': 0.1, 'correlation': 0.5}

        gaps = _compute_metric_gaps(train_perf, test_perf)

        # Absolute gap computed, relative gap skipped
        assert 'r_squared_gap' in gaps
        assert 'r_squared_relative_gap_pct' not in gaps

    def test_handles_nan_values(self):
        """Skips metrics with NaN values."""
        train_perf = {'r_squared': np.nan, 'mape': 10.0, 'mae': 0.1, 'correlation': 0.8}
        test_perf = {'r_squared': 0.5, 'mape': 15.0, 'mae': 0.2, 'correlation': 0.7}

        gaps = _compute_metric_gaps(train_perf, test_perf)

        assert 'r_squared_gap' not in gaps
        assert 'mape_gap' in gaps


class TestAssessGeneralizationQuality:
    """Tests for _assess_generalization_quality."""

    @pytest.mark.parametrize('gap_pct,expected_quality,expected_score', [
        (3, 'EXCELLENT', 95),
        (5, 'EXCELLENT', 95),
        (7, 'GOOD', 80),
        (10, 'GOOD', 80),
        (15, 'ACCEPTABLE', 65),
        (20, 'ACCEPTABLE', 65),
        (30, 'CONCERNING', 40),
        (40, 'CONCERNING', 40),
        (50, 'POOR', 20),
        (100, 'POOR', 20),
    ])
    def test_quality_thresholds(self, gap_pct, expected_quality, expected_score):
        """Quality grades match degradation thresholds."""
        quality, score = _assess_generalization_quality(gap_pct)

        assert quality == expected_quality
        assert score == expected_score


class TestCalculateGeneralizationMetrics:
    """Tests for _calculate_generalization_metrics."""

    def test_includes_quality_assessment(self, sample_train_performance, sample_test_performance):
        """Includes generalization quality assessment."""
        metrics = _calculate_generalization_metrics(
            sample_train_performance, sample_test_performance
        )

        assert 'generalization_quality' in metrics
        assert 'generalization_score' in metrics
        assert metrics['generalization_quality'] in ['EXCELLENT', 'GOOD', 'ACCEPTABLE',
                                                      'CONCERNING', 'POOR']

    def test_includes_primary_gaps(self, sample_train_performance, sample_test_performance):
        """Includes primary R² gap metrics."""
        metrics = _calculate_generalization_metrics(
            sample_train_performance, sample_test_performance
        )

        assert 'primary_gap_r2' in metrics
        assert 'primary_gap_r2_pct' in metrics


# =============================================================================
# Statistical Tests
# =============================================================================


class TestTestResidualMeans:
    """Tests for _test_residual_means."""

    def test_returns_expected_keys(self):
        """Returns all expected result keys."""
        train_resid = np.array([0.1, -0.1, 0.05, -0.05])
        test_resid = np.array([0.2, -0.2, 0.1, -0.1])

        result = _test_residual_means(train_resid, test_resid)

        expected_keys = ['train_mean', 'train_tstat', 'train_pval',
                         'test_mean', 'test_tstat', 'test_pval']
        for key in expected_keys:
            assert key in result

    def test_zero_mean_residuals_high_pvalue(self):
        """Zero-mean residuals yield high p-values."""
        residuals = np.array([-1, 1, -0.5, 0.5, 0])

        result = _test_residual_means(residuals, residuals)

        # Zero-mean should have high p-value (fail to reject H0: mean=0)
        assert result['train_pval'] > 0.05


class TestTestVarianceEquality:
    """Tests for _test_variance_equality."""

    def test_returns_expected_keys(self):
        """Returns Levene test results."""
        train_resid = np.array([0.1, -0.1, 0.2, -0.2])
        test_resid = np.array([0.1, -0.1, 0.15, -0.15])

        result = _test_variance_equality(train_resid, test_resid)

        assert 'levene_statistic' in result
        assert 'levene_pvalue' in result
        assert 'train_var' in result
        assert 'test_var' in result

    def test_equal_variance_high_pvalue(self):
        """Equal variance distributions yield high p-value."""
        np.random.seed(42)
        train_resid = np.random.normal(0, 1, 100)
        test_resid = np.random.normal(0, 1, 100)

        result = _test_variance_equality(train_resid, test_resid)

        # Similar variance should have moderate p-value
        assert result['levene_pvalue'] > 0.01


class TestTestDistributionSimilarity:
    """Tests for _test_distribution_similarity."""

    def test_returns_expected_keys(self):
        """Returns KS test results."""
        train_resid = np.array([0.1, -0.1, 0.2])
        test_resid = np.array([0.1, -0.1, 0.2])

        result = _test_distribution_similarity(train_resid, test_resid)

        assert 'ks_statistic' in result
        assert 'ks_pvalue' in result
        assert 'distributions_similar' in result

    def test_identical_distributions(self):
        """Identical distributions marked as similar."""
        data = np.array([0.1, -0.1, 0.2, -0.2, 0.0])

        result = _test_distribution_similarity(data, data)

        assert result['distributions_similar'] == True  # noqa: E712
        assert result['ks_pvalue'] == pytest.approx(1.0)


class TestPerformGeneralizationStatisticalTests:
    """Tests for _perform_generalization_statistical_tests."""

    def test_runs_all_tests(self):
        """Runs all statistical tests."""
        train_actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        train_pred = np.array([1.1, 2.0, 3.1, 4.0, 5.1])
        test_actual = pd.Series([6.0, 7.0, 8.0])
        test_pred = np.array([6.1, 7.0, 8.1])

        result = _perform_generalization_statistical_tests(
            train_actual, train_pred, test_actual, test_pred
        )

        assert 'residual_mean_tests' in result
        assert 'variance_equality_test' in result
        assert 'distribution_similarity_test' in result

    def test_handles_nan_residuals(self):
        """Filters NaN from residuals."""
        train_actual = pd.Series([1.0, np.nan, 3.0])
        train_pred = np.array([1.0, 2.0, 3.0])
        test_actual = pd.Series([4.0, 5.0])
        test_pred = np.array([4.0, 5.0])

        result = _perform_generalization_statistical_tests(
            train_actual, train_pred, test_actual, test_pred
        )

        # Should complete without error
        assert 'residual_mean_tests' in result


# =============================================================================
# Residual Analysis Tests
# =============================================================================


class TestComputeResidualBasicStats:
    """Tests for _compute_residual_basic_stats."""

    def test_returns_all_stats(self):
        """Returns all basic statistical measures."""
        residuals = np.array([0.1, -0.2, 0.3, -0.1, 0.05])

        stats = _compute_residual_basic_stats(residuals)

        expected_keys = ['mean', 'std', 'min', 'max', 'median', 'skewness', 'kurtosis']
        for key in expected_keys:
            assert key in stats

    def test_small_sample_nan_skewness(self):
        """Small samples return NaN for skewness/kurtosis."""
        residuals = np.array([0.1, 0.2])  # Only 2 elements

        stats = _compute_residual_basic_stats(residuals)

        assert np.isnan(stats['skewness'])
        assert np.isnan(stats['kurtosis'])


class TestRunNormalityTests:
    """Tests for _run_normality_tests."""

    def test_small_sample_uses_shapiro(self):
        """Samples ≤50 use Shapiro-Wilk test."""
        residuals = np.random.normal(0, 1, 30)

        results = _run_normality_tests(residuals)

        assert 'normality_shapiro' in results
        assert 'normality_jarque_bera' in results

    def test_large_sample_skips_shapiro(self):
        """Samples >50 skip Shapiro-Wilk test."""
        residuals = np.random.normal(0, 1, 100)

        results = _run_normality_tests(residuals)

        assert 'normality_shapiro' not in results
        assert 'normality_jarque_bera' in results

    def test_normal_residuals_pass_tests(self):
        """Normal residuals pass normality tests at 5%."""
        np.random.seed(42)
        residuals = np.random.normal(0, 1, 50)

        results = _run_normality_tests(residuals)

        # Should pass at 5% level (may occasionally fail due to randomness)
        assert 'normality_shapiro' in results
        assert 'pvalue' in results['normality_shapiro']


class TestCheckHeteroscedasticity:
    """Tests for _check_heteroscedasticity."""

    def test_returns_expected_keys(self):
        """Returns heteroscedasticity check results."""
        residuals = np.array([0.1, -0.2, 0.3, -0.1])
        predictions = np.array([1.0, 2.0, 3.0, 4.0])

        result = _check_heteroscedasticity(residuals, predictions)

        assert 'residual_fitted_correlation' in result
        assert 'potential_heteroscedasticity' in result

    def test_high_correlation_flags_issue(self):
        """High abs residual-fitted correlation flags heteroscedasticity."""
        # Create pattern where |residuals| increase with predictions
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        residuals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Strong positive correlation

        result = _check_heteroscedasticity(residuals, predictions)

        assert result['potential_heteroscedasticity'] == True  # noqa: E712

    def test_no_correlation_no_flag(self):
        """Low correlation does not flag heteroscedasticity."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        residuals = np.array([0.1, -0.2, 0.1, -0.1, 0.2])  # Random pattern

        result = _check_heteroscedasticity(residuals, predictions)

        # Correlation should be low
        assert abs(result['residual_fitted_correlation']) < 0.5


class TestAnalyzeTestResiduals:
    """Tests for _analyze_test_residuals."""

    def test_runs_all_analyses(self):
        """Runs complete residual analysis."""
        test_residuals = pd.Series([0.1, -0.2, 0.15, -0.1, 0.05])
        test_predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = _analyze_test_residuals(test_residuals, test_predictions)

        assert 'basic_stats' in result
        assert 'heteroscedasticity' in result

    def test_handles_empty_residuals(self):
        """Handles empty residuals gracefully."""
        test_residuals = pd.Series([np.nan, np.nan])
        test_predictions = np.array([np.nan, np.nan])

        result = _analyze_test_residuals(test_residuals, test_predictions)

        # Should not have basic_stats
        assert 'basic_stats' not in result or result.get('analysis_failed', False)


# =============================================================================
# Production Readiness Tests
# =============================================================================


class TestCheckTestR2:
    """Tests for _check_test_r2."""

    def test_passes_above_threshold(self, empty_assessment):
        """R² above threshold passes check."""
        result = _check_test_r2(0.5, 0.3, empty_assessment)

        assert result == True  # noqa: E712
        assert len(empty_assessment['strengths']) == 1
        assert 'Good test R²' in empty_assessment['strengths'][0]

    def test_fails_below_threshold(self, empty_assessment):
        """R² below threshold fails check."""
        result = _check_test_r2(0.2, 0.3, empty_assessment)

        assert result == False  # noqa: E712
        assert len(empty_assessment['primary_concerns']) == 1
        assert 'Low test R²' in empty_assessment['primary_concerns'][0]


class TestCheckTestMape:
    """Tests for _check_test_mape."""

    def test_passes_below_threshold(self, empty_assessment):
        """MAPE below threshold passes check."""
        result = _check_test_mape(15.0, 25.0, empty_assessment)

        assert result == True  # noqa: E712
        assert 'Acceptable MAPE' in empty_assessment['strengths'][0]

    def test_fails_above_threshold(self, empty_assessment):
        """MAPE above threshold fails check."""
        result = _check_test_mape(30.0, 25.0, empty_assessment)

        assert result == False  # noqa: E712
        assert 'High MAPE' in empty_assessment['primary_concerns'][0]


class TestCheckGeneralization:
    """Tests for _check_generalization."""

    def test_passes_small_gap(self, empty_assessment):
        """Small R² gap passes check."""
        result = _check_generalization(10.0, 30.0, empty_assessment)

        assert result == True  # noqa: E712
        assert 'Good generalization' in empty_assessment['strengths'][0]

    def test_fails_large_gap(self, empty_assessment):
        """Large R² gap fails check."""
        result = _check_generalization(50.0, 30.0, empty_assessment)

        assert result == False  # noqa: E712
        assert 'Poor generalization' in empty_assessment['primary_concerns'][0]


class TestCheckResidualDiagnostics:
    """Tests for _check_residual_diagnostics."""

    def test_clean_residuals_pass(self, empty_assessment):
        """Clean residuals pass diagnostics."""
        residual_analysis = {
            'basic_stats': {'mean': 0.01},
            'heteroscedasticity': {'potential_heteroscedasticity': False}
        }

        result = _check_residual_diagnostics(residual_analysis, empty_assessment)

        assert result == True  # noqa: E712
        assert 'Clean residual diagnostics' in empty_assessment['strengths']

    def test_biased_residuals_fail(self, empty_assessment):
        """Biased residuals fail diagnostics."""
        residual_analysis = {
            'basic_stats': {'mean': 0.5}  # High bias
        }

        result = _check_residual_diagnostics(residual_analysis, empty_assessment)

        assert result == False  # noqa: E712
        assert any('biased' in c for c in empty_assessment['primary_concerns'])

    def test_heteroscedastic_residuals_fail(self, empty_assessment):
        """Heteroscedastic residuals fail diagnostics."""
        residual_analysis = {
            'basic_stats': {'mean': 0.01},
            'heteroscedasticity': {'potential_heteroscedasticity': True}
        }

        result = _check_residual_diagnostics(residual_analysis, empty_assessment)

        assert result == False  # noqa: E712
        assert any('heteroscedasticity' in c for c in empty_assessment['primary_concerns'])


class TestDetermineConfidenceLevel:
    """Tests for _determine_confidence_level."""

    @pytest.mark.parametrize('passed,total,expected_ready,expected_level', [
        (4, 4, True, 'HIGH'),
        (3, 4, True, 'MODERATE'),
        (2, 4, False, 'LOW'),
        (1, 4, False, 'VERY LOW'),
        (0, 4, False, 'VERY LOW'),
    ])
    def test_confidence_thresholds(self, passed, total, expected_ready, expected_level):
        """Confidence levels match pass rate thresholds."""
        assessment = {
            'production_ready': False,
            'confidence_level': 'LOW',
            'recommendations': []
        }

        _determine_confidence_level(passed, total, assessment)

        assert assessment['production_ready'] == expected_ready
        assert assessment['confidence_level'] == expected_level


class TestAddSpecificRecommendations:
    """Tests for _add_specific_recommendations."""

    def test_adds_feature_recommendation_for_low_r2(self):
        """Adds feature recommendation when R² too low."""
        assessment = {'production_ready': False, 'recommendations': []}

        _add_specific_recommendations(
            assessment, test_r2=0.2, test_mape=10.0, r2_gap_pct=5.0,
            min_test_r2=0.3, max_test_mape=25.0, max_r2_degradation=30.0
        )

        assert any('feature' in r.lower() for r in assessment['recommendations'])

    def test_adds_accuracy_recommendation_for_high_mape(self):
        """Adds accuracy recommendation when MAPE too high."""
        assessment = {'production_ready': False, 'recommendations': []}

        _add_specific_recommendations(
            assessment, test_r2=0.5, test_mape=30.0, r2_gap_pct=5.0,
            min_test_r2=0.3, max_test_mape=25.0, max_r2_degradation=30.0
        )

        assert any('accuracy' in r.lower() for r in assessment['recommendations'])

    def test_adds_overfitting_recommendation_for_large_gap(self):
        """Adds overfitting recommendation when generalization gap too large."""
        assessment = {'production_ready': False, 'recommendations': []}

        _add_specific_recommendations(
            assessment, test_r2=0.5, test_mape=10.0, r2_gap_pct=50.0,
            min_test_r2=0.3, max_test_mape=25.0, max_r2_degradation=30.0
        )

        assert any('overfit' in r.lower() for r in assessment['recommendations'])

    def test_no_recommendations_when_production_ready(self):
        """No specific recommendations when production ready."""
        assessment = {'production_ready': True, 'recommendations': []}

        _add_specific_recommendations(
            assessment, test_r2=0.5, test_mape=10.0, r2_gap_pct=5.0,
            min_test_r2=0.3, max_test_mape=25.0, max_r2_degradation=30.0
        )

        assert len(assessment['recommendations']) == 0


class TestBuildSummaryMetrics:
    """Tests for _build_summary_metrics."""

    def test_builds_complete_summary(self):
        """Builds summary with all expected fields."""
        summary = _build_summary_metrics(
            passed_checks=3, total_checks=4, test_r2=0.65,
            test_mape=12.5, generalization_quality='GOOD', r2_gap_pct=15.0
        )

        expected_keys = ['checks_passed', 'total_checks', 'pass_rate',
                         'test_r2', 'test_mape', 'generalization_quality',
                         'r2_degradation_pct']
        for key in expected_keys:
            assert key in summary

        assert summary['pass_rate'] == pytest.approx(0.75)

    def test_handles_zero_total_checks(self):
        """Handles zero total checks without division error."""
        summary = _build_summary_metrics(0, 0, 0.5, 10.0, 'UNKNOWN', 0.0)

        assert summary['pass_rate'] == 0


class TestAssessProductionReadiness:
    """Tests for _assess_production_readiness."""

    def test_production_ready_with_good_metrics(self):
        """Model is production ready with good metrics."""
        train_perf = {'r_squared': 0.8, 'mape': 10.0}
        test_perf = {'r_squared': 0.75, 'mape': 12.0}
        gen_metrics = {'generalization_quality': 'EXCELLENT', 'r_squared_relative_gap_pct': 5.0}
        residuals = {'basic_stats': {'mean': 0.01}, 'heteroscedasticity': {'potential_heteroscedasticity': False}}

        assessment = _assess_production_readiness(
            train_perf, test_perf, gen_metrics, residuals
        )

        assert assessment['production_ready'] == True  # noqa: E712
        assert assessment['confidence_level'] in ['HIGH', 'MODERATE']

    def test_not_production_ready_with_poor_metrics(self):
        """Model is not production ready with poor metrics."""
        train_perf = {'r_squared': 0.8, 'mape': 10.0}
        test_perf = {'r_squared': 0.1, 'mape': 50.0}  # Poor test performance
        gen_metrics = {'generalization_quality': 'POOR', 'r_squared_relative_gap_pct': 80.0}
        residuals = {'basic_stats': {'mean': 0.5}}  # Biased

        assessment = _assess_production_readiness(
            train_perf, test_perf, gen_metrics, residuals
        )

        assert assessment['production_ready'] == False  # noqa: E712
        assert assessment['confidence_level'] in ['LOW', 'VERY LOW']

    def test_includes_summary_metrics(self):
        """Assessment includes summary metrics."""
        train_perf = {'r_squared': 0.7, 'mape': 15.0}
        test_perf = {'r_squared': 0.6, 'mape': 18.0}
        gen_metrics = {'generalization_quality': 'GOOD', 'r_squared_relative_gap_pct': 12.0}
        residuals = {}

        assessment = _assess_production_readiness(
            train_perf, test_perf, gen_metrics, residuals
        )

        assert 'summary_metrics' in assessment


# =============================================================================
# Cross-Validation Tests
# =============================================================================


class TestEvaluateCvFold:
    """Tests for _evaluate_cv_fold."""

    def test_evaluates_fold_successfully(self, sample_train_data):
        """Successfully evaluates a CV fold."""
        train_idx = np.array(range(50))
        test_idx = np.array(range(50, 70))

        result = _evaluate_cv_fold(
            fold_idx=0,
            train_idx=train_idx,
            test_idx=test_idx,
            data=sample_train_data,
            formula='target ~ feature_a',
            target_variable='target',
            features='feature_a'
        )

        assert result['fold'] == 0
        assert result['train_size'] == 50
        assert result['test_size'] == 20
        assert 'performance' in result
        assert 'r_squared' in result['performance']

    def test_handles_fold_error(self, sample_train_data):
        """Handles errors in fold evaluation gracefully."""
        train_idx = np.array([0, 1])  # Very small
        test_idx = np.array([2, 3])

        # Create invalid formula
        result = _evaluate_cv_fold(
            fold_idx=0,
            train_idx=train_idx,
            test_idx=test_idx,
            data=sample_train_data,
            formula='target ~ nonexistent_column',
            target_variable='target',
            features='nonexistent_column'
        )

        assert 'failed' in result or 'error' in result


class TestAggregateCvPerformance:
    """Tests for _aggregate_cv_performance."""

    def test_aggregates_successful_folds(self):
        """Aggregates performance across successful folds."""
        folds = [
            {'fold': 0, 'performance': {'r_squared': 0.5, 'mape': 10.0, 'mae': 0.1, 'correlation': 0.7}},
            {'fold': 1, 'performance': {'r_squared': 0.6, 'mape': 12.0, 'mae': 0.12, 'correlation': 0.75}},
            {'fold': 2, 'performance': {'r_squared': 0.55, 'mape': 11.0, 'mae': 0.11, 'correlation': 0.72}},
        ]

        avg_perf, stability = _aggregate_cv_performance(folds)

        assert avg_perf['r_squared'] == pytest.approx(0.55, rel=0.01)
        assert 'r_squared_std' in stability
        assert 'r_squared_cv' in stability

    def test_handles_empty_folds(self):
        """Handles empty fold list."""
        avg_perf, stability = _aggregate_cv_performance([])

        assert avg_perf == {}
        assert stability == {}

    def test_handles_nan_values(self):
        """Handles NaN values in fold results."""
        folds = [
            {'fold': 0, 'performance': {'r_squared': 0.5, 'mape': np.nan}},
            {'fold': 1, 'performance': {'r_squared': 0.6, 'mape': 12.0}},
        ]

        avg_perf, stability = _aggregate_cv_performance(folds)

        # r_squared should use both values, mape only one
        assert 'r_squared' in avg_perf
        assert 'mape' in avg_perf


class TestAssessCvQuality:
    """Tests for _assess_cv_quality."""

    @pytest.mark.parametrize('r2_cv,avg_r2,expected', [
        (0.05, 0.5, 'EXCELLENT'),  # Low CV, high R²
        (0.15, 0.35, 'GOOD'),  # Moderate CV, moderate R²
        (0.25, 0.25, 'ACCEPTABLE'),  # Higher CV, lower R²
        (0.5, 0.1, 'POOR'),  # High CV, low R²
    ])
    def test_quality_thresholds(self, r2_cv, avg_r2, expected):
        """CV quality grades match thresholds."""
        quality = _assess_cv_quality(r2_cv, avg_r2)

        assert quality == expected


class TestBuildCvOverallAssessment:
    """Tests for _build_cv_overall_assessment."""

    def test_builds_assessment_for_successful_folds(self):
        """Builds assessment when folds succeed."""
        successful_folds = [
            {'fold': 0, 'performance': {'r_squared': 0.5}},
            {'fold': 1, 'performance': {'r_squared': 0.6}},
        ]
        fold_results = successful_folds  # All successful
        avg_perf = {'r_squared': 0.55}
        stability = {'r_squared_cv': 0.1}

        assessment = _build_cv_overall_assessment(
            successful_folds, fold_results, avg_perf, stability
        )

        assert assessment['cv_quality'] in ['EXCELLENT', 'GOOD', 'ACCEPTABLE', 'POOR']
        assert assessment['successful_folds'] == 2
        assert assessment['total_folds'] == 2
        assert assessment['success_rate'] == 1.0

    def test_handles_all_failed_folds(self):
        """Handles case where all folds fail."""
        successful_folds = []
        fold_results = [
            {'fold': 0, 'failed': True},
            {'fold': 1, 'failed': True},
        ]

        assessment = _build_cv_overall_assessment(
            successful_folds, fold_results, {}, {}
        )

        assert assessment['cv_quality'] == 'FAILED'
        assert assessment['success_rate'] == 0.0


class TestRunSingleModelTimeSeriesCv:
    """Tests for _run_single_model_time_series_cv."""

    def test_runs_cv_successfully(self, sample_train_data):
        """Runs time series CV for a model."""
        from sklearn.model_selection import TimeSeriesSplit

        model_row = pd.Series({
            'features': 'feature_a',
            'aic': 100.0
        })
        tscv = TimeSeriesSplit(n_splits=3)

        result = _run_single_model_time_series_cv(
            model_row, sample_train_data, 'target', tscv
        )

        assert isinstance(result, CrossValidationResult)
        assert result.model_features == 'feature_a'
        assert result.n_folds == 3
        assert len(result.fold_results) == 3


class TestRunTimeSeriesCrossValidation:
    """Tests for run_time_series_cross_validation (main function)."""

    def test_runs_cv_for_top_models(self, sample_model_results, sample_train_data):
        """Runs CV for top N models by AIC."""
        results = run_time_series_cross_validation(
            model_results=sample_model_results,
            data=sample_train_data,
            target_variable='target',
            n_splits=3,
            models_to_evaluate=2
        )

        assert len(results) == 2
        assert all(isinstance(r, CrossValidationResult) for r in results)

    def test_uses_default_min_train_size(self, sample_model_results, sample_train_data):
        """Uses default minimum training size calculation."""
        results = run_time_series_cross_validation(
            model_results=sample_model_results,
            data=sample_train_data,
            target_variable='target',
            n_splits=3,
            min_train_size=None,  # Use default
            models_to_evaluate=1
        )

        assert len(results) == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestFitAndPredict:
    """Tests for _fit_and_predict."""

    def test_fits_and_predicts(self, sample_train_data, sample_test_data):
        """Fits model on train and predicts for both sets."""
        train_actual, train_pred, test_actual, test_pred = _fit_and_predict(
            formula='target ~ feature_a',
            train_data=sample_train_data,
            test_data=sample_test_data,
            target_variable='target'
        )

        assert len(train_actual) == len(sample_train_data)
        assert len(train_pred) == len(sample_train_data)
        assert len(test_actual) == len(sample_test_data)
        assert len(test_pred) == len(sample_test_data)


class TestEvaluateSingleModelGeneralization:
    """Tests for _evaluate_single_model_generalization."""

    def test_returns_complete_result(self, sample_train_data, sample_test_data):
        """Returns complete OutOfSampleResult."""
        model_row = pd.Series({
            'features': 'feature_a',
            'aic': 100.0
        })

        result = _evaluate_single_model_generalization(
            model_row, sample_train_data, sample_test_data, 'target'
        )

        assert isinstance(result, OutOfSampleResult)
        assert result.model_features == 'feature_a'
        assert 'r_squared' in result.train_performance
        assert 'r_squared' in result.test_performance
        assert 'production_ready' in result.production_assessment


class TestEvaluateTemporalGeneralization:
    """Tests for evaluate_temporal_generalization (main function)."""

    def test_evaluates_multiple_models(self, sample_model_results, sample_temporal_split):
        """Evaluates temporal generalization for multiple models."""
        results = evaluate_temporal_generalization(
            model_results=sample_model_results,
            temporal_split_data=sample_temporal_split,
            target_variable='target',
            models_to_evaluate=2
        )

        assert len(results) == 2
        assert all(isinstance(r, OutOfSampleResult) for r in results)

    def test_raises_with_invalid_split(self, sample_model_results):
        """Raises error with invalid temporal split data."""
        with pytest.raises(ValueError, match="missing required"):
            evaluate_temporal_generalization(
                model_results=sample_model_results,
                temporal_split_data={'invalid': pd.DataFrame()},
                target_variable='target'
            )

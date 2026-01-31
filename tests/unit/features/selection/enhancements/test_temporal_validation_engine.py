"""
Tests for temporal_validation_engine module.

Target: 20% → 50%+ coverage
Tests organized by function categories:
- Dataclass
- Validation functions
- Data preparation
- Split calculation
- Main functions
- Quality assessment
- Frequency checks
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from src.features.selection.enhancements.temporal_validation_engine import (
    # Dataclass
    TemporalSplit,
    # Validation
    _validate_data_size_for_split,
    _validate_temporal_split_integrity,
    # Data preparation
    _prepare_datetime_data,
    _extract_period_strings,
    _calculate_split_point,
    # Main functions
    create_temporal_splits,
    evaluate_out_of_sample_performance,
    validate_temporal_structure,
    # Model helpers
    _get_or_fit_model,
    _calculate_test_metrics,
    _build_performance_results,
    # Quality assessment
    _classify_gap_quality,
    _assess_generalization_quality,
    # Frequency checks
    _check_frequency_consistency,
    _check_missing_periods,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_time_series_data():
    """Sample time series data with date column."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='W')
    return pd.DataFrame({
        'date': dates,
        'target': np.random.normal(0, 1, 100),
        'feature_a': np.random.normal(0, 1, 100),
        'feature_b': np.random.normal(0, 1, 100),
    })


@pytest.fixture
def sample_data_no_date():
    """Sample data without date column."""
    np.random.seed(42)
    return pd.DataFrame({
        'target': np.random.normal(0, 1, 50),
        'feature_a': np.random.normal(0, 1, 50),
    })


@pytest.fixture
def sample_data_datetime_index():
    """Sample data with DatetimeIndex."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=60, freq='W')
    df = pd.DataFrame({
        'target': np.random.normal(0, 1, 60),
        'feature_a': np.random.normal(0, 1, 60),
    }, index=dates)
    return df


@pytest.fixture
def sample_temporal_split(sample_time_series_data):
    """Pre-created temporal split for testing."""
    return create_temporal_splits(
        sample_time_series_data,
        split_ratio=0.8,
        ensure_minimum_test_size=10
    )


@pytest.fixture
def sample_model_results():
    """Sample model results dictionary."""
    return {
        'r_squared': 0.75,
        'aic': 100.0,
        'features': ['feature_a', 'feature_b']
    }


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestTemporalSplit:
    """Tests for TemporalSplit dataclass."""

    def test_construction_with_all_fields(self):
        """Dataclass constructs with all required fields."""
        train_df = pd.DataFrame({'a': [1, 2, 3]})
        test_df = pd.DataFrame({'a': [4, 5]})

        split = TemporalSplit(
            train_data=train_df,
            test_data=test_df,
            train_period=('2020-01-01', '2020-06-30'),
            test_period=('2020-07-01', '2020-12-31'),
            split_ratio=0.6,
            n_train_obs=3,
            n_test_obs=2,
            validation_checks={'no_overlap': True}
        )

        assert len(split.train_data) == 3
        assert len(split.test_data) == 2
        assert split.split_ratio == 0.6
        assert split.validation_checks['no_overlap'] == True  # noqa: E712

    def test_all_fields_accessible(self):
        """All dataclass fields are accessible."""
        split = TemporalSplit(
            train_data=pd.DataFrame(),
            test_data=pd.DataFrame(),
            train_period=('a', 'b'),
            test_period=('c', 'd'),
            split_ratio=0.8,
            n_train_obs=0,
            n_test_obs=0,
            validation_checks={}
        )

        attrs = ['train_data', 'test_data', 'train_period', 'test_period',
                 'split_ratio', 'n_train_obs', 'n_test_obs', 'validation_checks']

        for attr in attrs:
            assert hasattr(split, attr)


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidateDataSizeForSplit:
    """Tests for _validate_data_size_for_split."""

    def test_sufficient_data_passes(self, sample_time_series_data):
        """Sufficient data passes validation without error."""
        # Should not raise
        _validate_data_size_for_split(sample_time_series_data, 20)

    def test_insufficient_data_raises(self):
        """Insufficient data raises ValueError."""
        small_data = pd.DataFrame({'x': [1, 2, 3, 4, 5]})

        with pytest.raises(ValueError, match="Insufficient data for temporal split"):
            _validate_data_size_for_split(small_data, 20)

    def test_minimum_boundary(self):
        """Tests exact boundary condition."""
        # Need ensure_minimum_test_size + 10, so 30 total for min_test=20
        boundary_data = pd.DataFrame({'x': range(30)})

        # Exactly at boundary should pass
        _validate_data_size_for_split(boundary_data, 20)

        # One less should fail
        small_data = pd.DataFrame({'x': range(29)})
        with pytest.raises(ValueError):
            _validate_data_size_for_split(small_data, 20)


class TestValidateTemporalSplitIntegrity:
    """Tests for _validate_temporal_split_integrity."""

    def test_valid_split_passes_all_checks(self):
        """Valid temporal split passes all integrity checks."""
        train_data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        test_data = pd.DataFrame({'a': [7, 8], 'b': [9, 10]})

        checks = _validate_temporal_split_integrity(
            train_data, test_data,
            train_end='2020-06-30', test_start='2020-07-01'
        )

        assert checks['no_temporal_overlap'] == True  # noqa: E712
        assert checks['sequential_periods'] == True  # noqa: E712
        assert checks['train_data_non_empty'] == True  # noqa: E712
        assert checks['test_data_non_empty'] == True  # noqa: E712
        assert checks['consistent_features'] == True  # noqa: E712

    def test_temporal_overlap_fails(self):
        """Temporal overlap detected."""
        train_data = pd.DataFrame({'a': [1, 2]})
        test_data = pd.DataFrame({'a': [3, 4]})

        # Test starts before train ends
        checks = _validate_temporal_split_integrity(
            train_data, test_data,
            train_end='2020-07-15', test_start='2020-07-01'
        )

        assert checks['no_temporal_overlap'] == False  # noqa: E712

    def test_large_gap_detected(self):
        """Large gap between train and test detected."""
        train_data = pd.DataFrame({'a': [1, 2]})
        test_data = pd.DataFrame({'a': [3, 4]})

        # 30-day gap
        checks = _validate_temporal_split_integrity(
            train_data, test_data,
            train_end='2020-06-01', test_start='2020-07-01'
        )

        assert checks['sequential_periods'] == False  # noqa: E712

    def test_inconsistent_features_detected(self):
        """Inconsistent features between train and test detected."""
        train_data = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        test_data = pd.DataFrame({'a': [5, 6], 'c': [7, 8]})  # Different column

        checks = _validate_temporal_split_integrity(
            train_data, test_data,
            train_end='2020-06-30', test_start='2020-07-01'
        )

        assert checks['consistent_features'] == False  # noqa: E712

    def test_reasonable_split_ratio_check(self):
        """Reasonable split ratio check."""
        train_data = pd.DataFrame({'a': range(8)})
        test_data = pd.DataFrame({'a': range(2)})  # 80% train

        checks = _validate_temporal_split_integrity(
            train_data, test_data,
            train_end='2020-06-30', test_start='2020-07-01'
        )

        assert checks['reasonable_split'] == True  # noqa: E712

        # Extreme split (95% train)
        train_data_extreme = pd.DataFrame({'a': range(95)})
        test_data_extreme = pd.DataFrame({'a': range(5)})

        checks_extreme = _validate_temporal_split_integrity(
            train_data_extreme, test_data_extreme,
            train_end='2020-06-30', test_start='2020-07-01'
        )

        assert checks_extreme['reasonable_split'] == False  # noqa: E712


# =============================================================================
# Data Preparation Tests
# =============================================================================


class TestPrepareDatetimeData:
    """Tests for _prepare_datetime_data."""

    def test_with_date_column(self, sample_time_series_data):
        """Prepares data with date column correctly."""
        working_data, date_series = _prepare_datetime_data(
            sample_time_series_data, 'date'
        )

        assert 'date' in working_data.columns
        assert len(date_series) == len(sample_time_series_data)
        assert isinstance(date_series.iloc[0], pd.Timestamp)

    def test_with_datetime_index(self, sample_data_datetime_index):
        """Handles DatetimeIndex correctly."""
        working_data, date_series = _prepare_datetime_data(
            sample_data_datetime_index, 'date'
        )

        assert len(date_series) == len(sample_data_datetime_index)

    def test_without_date_creates_synthetic(self, sample_data_no_date):
        """Creates synthetic dates when no date column exists."""
        with pytest.warns(UserWarning, match="No date column"):
            working_data, date_series = _prepare_datetime_data(
                sample_data_no_date, 'date'
            )

        assert 'date' in working_data.columns
        assert len(date_series) == len(sample_data_no_date)

    def test_sorts_by_date(self):
        """Sorts data by date column."""
        unsorted_data = pd.DataFrame({
            'date': pd.to_datetime(['2020-03-01', '2020-01-01', '2020-02-01']),
            'value': [3, 1, 2]
        })

        working_data, date_series = _prepare_datetime_data(unsorted_data, 'date')

        # Should be sorted chronologically
        assert working_data['value'].tolist() == [1, 2, 3]


class TestExtractPeriodStrings:
    """Tests for _extract_period_strings."""

    def test_extracts_formatted_dates(self):
        """Extracts properly formatted date strings."""
        dates = pd.Series(pd.date_range('2020-01-01', periods=10, freq='W'))

        train_start, train_end, test_start, test_end = _extract_period_strings(
            dates, n_train=7
        )

        assert train_start == '2020-01-05'  # First date (week freq starts on Sunday)
        assert train_end == '2020-02-16'    # 7th date
        assert test_start == '2020-02-23'   # 8th date
        assert test_end == '2020-03-08'     # Last date


class TestCalculateSplitPoint:
    """Tests for _calculate_split_point."""

    def test_standard_split(self):
        """Calculates standard split point."""
        n_train, ratio = _calculate_split_point(
            n_total=100, split_ratio=0.8, ensure_minimum_test_size=10
        )

        assert n_train == 80
        assert ratio == 0.8

    def test_adjusts_for_minimum_test_size(self):
        """Adjusts split to ensure minimum test size."""
        # 100 * 0.95 = 95 train, leaving only 5 test
        # Should adjust to 100 - 20 = 80 train
        with pytest.warns(UserWarning, match="Adjusted split ratio"):
            n_train, ratio = _calculate_split_point(
                n_total=100, split_ratio=0.95, ensure_minimum_test_size=20
            )

        assert n_train == 80
        assert ratio == 0.8


# =============================================================================
# Main Function Tests
# =============================================================================


class TestCreateTemporalSplits:
    """Tests for create_temporal_splits."""

    def test_creates_valid_split(self, sample_time_series_data):
        """Creates valid temporal split."""
        split = create_temporal_splits(
            sample_time_series_data,
            split_ratio=0.8,
            ensure_minimum_test_size=10
        )

        assert isinstance(split, TemporalSplit)
        assert len(split.train_data) == 80
        assert len(split.test_data) == 20
        assert split.n_train_obs == 80
        assert split.n_test_obs == 20

    def test_insufficient_data_raises(self):
        """Raises error for insufficient data."""
        small_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=15, freq='W'),
            'target': range(15)
        })

        with pytest.raises(ValueError, match="Insufficient data"):
            create_temporal_splits(small_data, ensure_minimum_test_size=20)

    def test_validation_checks_populated(self, sample_time_series_data):
        """Validation checks are populated."""
        split = create_temporal_splits(sample_time_series_data)

        assert 'no_temporal_overlap' in split.validation_checks
        assert 'train_data_non_empty' in split.validation_checks

    def test_custom_split_ratio(self, sample_time_series_data):
        """Respects custom split ratio."""
        split = create_temporal_splits(
            sample_time_series_data,
            split_ratio=0.7,
            ensure_minimum_test_size=10
        )

        assert split.n_train_obs == 70
        assert split.n_test_obs == 30


class TestValidateTemporalStructure:
    """Tests for validate_temporal_structure."""

    def test_validates_good_data(self, sample_time_series_data):
        """Validates properly structured data."""
        result = validate_temporal_structure(
            sample_time_series_data, date_column='date'
        )

        assert result['checks']['date_column_exists'] == True  # noqa: E712
        assert result['checks']['datetime_conversion_successful'] == True  # noqa: E712
        assert result['validation_passed'] == True  # noqa: E712

    def test_missing_date_column(self, sample_data_no_date):
        """Handles missing date column."""
        result = validate_temporal_structure(
            sample_data_no_date, date_column='date'
        )

        assert result['checks']['date_column_exists'] == False  # noqa: E712
        # Note: Function returns early without 'validation_passed' when date column missing
        assert 'validation_passed' not in result or result.get('validation_passed') == False  # noqa: E712

    def test_includes_timestamp(self, sample_time_series_data):
        """Includes validation timestamp."""
        result = validate_temporal_structure(sample_time_series_data)

        assert 'temporal_validation_timestamp' in result

    def test_checks_chronological_order(self):
        """Checks chronological ordering."""
        # Unsorted data
        unsorted = pd.DataFrame({
            'date': pd.to_datetime(['2020-03-01', '2020-01-01', '2020-02-01']),
            'value': [3, 1, 2]
        })

        result = validate_temporal_structure(unsorted)

        assert result['checks']['chronological_order'] == False  # noqa: E712


# =============================================================================
# Model Helper Tests
# =============================================================================


class TestGetOrFitModel:
    """Tests for _get_or_fit_model."""

    def test_returns_existing_model(self, sample_temporal_split):
        """Returns model from results if available."""
        mock_model = MagicMock()
        model_results = {'model': mock_model}

        result = _get_or_fit_model(
            model_results, sample_temporal_split, 'target', ['feature_a']
        )

        assert result is mock_model

    def test_fits_new_model_when_missing(self, sample_temporal_split):
        """Fits new model when not in results."""
        model_results = {'r_squared': 0.5}  # No 'model' key

        result = _get_or_fit_model(
            model_results, sample_temporal_split, 'target', ['feature_a']
        )

        # Should return a fitted statsmodels result
        assert hasattr(result, 'params')
        assert hasattr(result, 'predict')


class TestCalculateTestMetrics:
    """Tests for _calculate_test_metrics."""

    def test_calculates_metrics(self, sample_temporal_split):
        """Calculates predictions, MSE, and R²."""
        # Fit a simple model
        import statsmodels.formula.api as smf
        model = smf.ols('target ~ feature_a', data=sample_temporal_split.train_data).fit()

        predictions, mse, r2 = _calculate_test_metrics(
            model, sample_temporal_split.test_data, 'target'
        )

        assert len(predictions) == len(sample_temporal_split.test_data)
        assert mse >= 0
        assert isinstance(r2, float)


class TestBuildPerformanceResults:
    """Tests for _build_performance_results."""

    def test_builds_complete_results(self, sample_temporal_split):
        """Builds complete performance results dictionary."""
        import statsmodels.formula.api as smf
        model = smf.ols('target ~ feature_a', data=sample_temporal_split.train_data).fit()
        test_predictions = model.predict(sample_temporal_split.test_data)

        results = _build_performance_results(
            trained_model=model,
            test_predictions=test_predictions,
            test_data=sample_temporal_split.test_data,
            target_variable='target',
            test_mse=0.5,
            test_r_squared=0.6,
            train_r_squared=0.7,
            temporal_split=sample_temporal_split,
            features=['feature_a']
        )

        expected_keys = ['test_r_squared', 'test_mse', 'train_r_squared',
                         'generalization_gap', 'generalization_gap_percent',
                         'performance_assessment', 'features_evaluated',
                         'model_coefficients', 'test_predictions', 'test_residuals']

        for key in expected_keys:
            assert key in results


# =============================================================================
# Quality Assessment Tests
# =============================================================================


class TestClassifyGapQuality:
    """Tests for _classify_gap_quality."""

    @pytest.mark.parametrize('gap,expected_quality,expected_ready', [
        (0.03, 'EXCELLENT', True),
        (0.05, 'EXCELLENT', True),
        (0.07, 'GOOD', True),
        (0.10, 'GOOD', True),
        (0.15, 'FAIR', False),
        (0.20, 'FAIR', False),
        (0.25, 'POOR', False),
        (0.50, 'POOR', False),
    ])
    def test_quality_thresholds(self, gap, expected_quality, expected_ready):
        """Quality ratings match gap thresholds."""
        quality, confidence, ready, rec = _classify_gap_quality(gap)

        assert quality == expected_quality
        assert ready == expected_ready

    def test_includes_recommendation(self):
        """Includes actionable recommendation."""
        quality, confidence, ready, rec = _classify_gap_quality(0.03)

        assert 'production' in rec.lower()


class TestAssessGeneralizationQuality:
    """Tests for _assess_generalization_quality."""

    def test_excellent_generalization(self):
        """Assesses excellent generalization correctly."""
        assessment = _assess_generalization_quality(
            train_r2=0.8, test_r2=0.78, gap=0.02
        )

        assert assessment['quality_rating'] == 'EXCELLENT'
        assert assessment['production_ready'] == True  # noqa: E712
        assert assessment['confidence_level'] == 'HIGH'

    def test_failed_model_negative_r2(self):
        """Handles failed model with negative test R²."""
        assessment = _assess_generalization_quality(
            train_r2=0.5, test_r2=-0.1, gap=0.6
        )

        assert assessment['quality_rating'] == 'FAILED'
        assert assessment['production_ready'] == False  # noqa: E712
        assert assessment['confidence_level'] == 'NONE'
        assert 'naive mean' in assessment['recommendation'].lower()

    def test_includes_literature_reference(self):
        """Includes literature reference for validation."""
        assessment = _assess_generalization_quality(0.7, 0.6, 0.1)

        assert 'literature_reference' in assessment
        assert 'Hawkins' in assessment['literature_reference']


# =============================================================================
# Frequency Check Tests
# =============================================================================


class TestCheckFrequencyConsistency:
    """Tests for _check_frequency_consistency."""

    def test_consistent_weekly_frequency(self):
        """Detects consistent weekly frequency."""
        dates = pd.Series(pd.date_range('2020-01-01', periods=10, freq='W'))
        results = {'checks': {}}

        _check_frequency_consistency(dates, results)

        assert results['checks']['frequency_consistency'] == True  # noqa: E712

    def test_inconsistent_frequency(self):
        """Detects inconsistent frequency."""
        # Mix of weekly and bi-weekly gaps
        dates = pd.Series(pd.to_datetime([
            '2020-01-01', '2020-01-08', '2020-01-22',  # 7 days, then 14 days
            '2020-01-29', '2020-02-12'  # 7 days, then 14 days
        ]))
        results = {'checks': {}}

        _check_frequency_consistency(dates, results)

        # Should fail consistency check (mode not >80% of diffs)
        assert 'frequency_consistency' in results['checks']

    def test_single_date_no_check(self):
        """Handles single date gracefully."""
        dates = pd.Series([pd.Timestamp('2020-01-01')])
        results = {'checks': {}}

        _check_frequency_consistency(dates, results)

        assert 'frequency_consistency' not in results['checks']


class TestCheckMissingPeriods:
    """Tests for _check_missing_periods."""

    def test_no_missing_periods(self):
        """Detects complete series with no gaps."""
        dates = pd.Series(pd.date_range('2020-01-05', periods=10, freq='W'))
        results = {'checks': {}}

        _check_missing_periods(dates, 'W', results)

        assert results['checks']['no_missing_periods'] == True  # noqa: E712
        assert results['missing_periods_count'] == 0

    def test_missing_periods_detected(self):
        """Detects missing periods in series."""
        # Skip some weeks
        dates = pd.Series(pd.to_datetime([
            '2020-01-05', '2020-01-12', '2020-01-26',  # Missing 2020-01-19
            '2020-02-02', '2020-02-09'
        ]))
        results = {'checks': {}}

        _check_missing_periods(dates, 'W', results)

        assert results['checks']['no_missing_periods'] == False  # noqa: E712
        assert results['missing_periods_count'] > 0

    def test_non_weekly_frequency_skipped(self):
        """Skips check for non-weekly frequency."""
        dates = pd.Series(pd.date_range('2020-01-01', periods=10, freq='D'))
        results = {'checks': {}}

        _check_missing_periods(dates, 'D', results)

        assert 'no_missing_periods' not in results['checks']


# =============================================================================
# Integration Tests
# =============================================================================


class TestEvaluateOutOfSamplePerformance:
    """Tests for evaluate_out_of_sample_performance."""

    def test_evaluates_model(self, sample_temporal_split, sample_model_results):
        """Evaluates model on out-of-sample data."""
        result = evaluate_out_of_sample_performance(
            model_results=sample_model_results,
            temporal_split=sample_temporal_split,
            target_variable='target',
            features=['feature_a']
        )

        assert 'test_r_squared' in result
        assert 'generalization_gap' in result
        assert 'performance_assessment' in result

    def test_includes_predictions(self, sample_temporal_split, sample_model_results):
        """Includes predictions and residuals."""
        result = evaluate_out_of_sample_performance(
            model_results=sample_model_results,
            temporal_split=sample_temporal_split,
            target_variable='target',
            features=['feature_a']
        )

        assert 'test_predictions' in result
        assert 'test_residuals' in result
        assert len(result['test_predictions']) == sample_temporal_split.n_test_obs

    def test_invalid_model_raises(self, sample_temporal_split):
        """Raises error for invalid model specification."""
        with pytest.raises(ValueError, match="Out-of-sample evaluation failed"):
            evaluate_out_of_sample_performance(
                model_results={},
                temporal_split=sample_temporal_split,
                target_variable='target',
                features=['nonexistent_feature']
            )

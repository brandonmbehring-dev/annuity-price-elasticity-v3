"""
Unit Tests for Production Validators
=====================================

Tests for production-grade validation functions used to ensure
data quality and pipeline stability.

Coverage target: 60%

Author: Claude Code
Date: 2026-01-30
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from src.validation.production_validators import (
    extract_validation_metrics,
    validate_schema_stability,
    validate_growth_patterns,
    validate_date_range_progression,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'sales': np.random.rand(100) * 1000,
        'product_code': ['6Y20B'] * 100,
        'premium': np.random.rand(100) * 10000,
    })


@pytest.fixture
def sample_metrics():
    """Create sample ValidationMetrics."""
    return {
        'run_date': '2023-06-01T10:00:00',
        'record_count': 100,
        'column_count': 4,
        'column_names': ['date', 'sales', 'product_code', 'premium'],
        'min_date': '2023-01-01',
        'max_date': '2023-04-10',
        'date_range_days': 99,
        'aggregation_ratio': None,
    }


@pytest.fixture
def growth_config():
    """Create sample GrowthConfig."""
    return {
        'min_growth_pct': -10.0,
        'max_growth_pct': 50.0,
        'warn_on_shrinkage': True,
        'warn_on_high_growth': True,
    }


# =============================================================================
# EXTRACT METRICS TESTS
# =============================================================================


class TestExtractValidationMetrics:
    """Tests for extract_validation_metrics function."""

    def test_extracts_basic_metrics(self, sample_df):
        """Should extract record count, column count, and column names."""
        metrics = extract_validation_metrics(sample_df)

        assert metrics['record_count'] == 100
        assert metrics['column_count'] == 4
        assert 'date' in metrics['column_names']
        assert 'sales' in metrics['column_names']

    def test_extracts_date_range(self, sample_df):
        """Should extract date range when date column provided."""
        metrics = extract_validation_metrics(sample_df, date_column='date')

        assert metrics['min_date'] is not None
        assert metrics['max_date'] is not None
        assert metrics['date_range_days'] == 99

    def test_handles_missing_date_column(self, sample_df):
        """Should handle gracefully when date column doesn't exist."""
        metrics = extract_validation_metrics(sample_df, date_column='nonexistent')

        assert metrics['min_date'] is None
        assert metrics['max_date'] is None
        assert metrics['date_range_days'] is None

    def test_handles_no_date_column_specified(self, sample_df):
        """Should work without date column specification."""
        metrics = extract_validation_metrics(sample_df)

        assert metrics['min_date'] is None
        assert metrics['record_count'] == 100

    def test_includes_run_date(self, sample_df):
        """Should include run_date in metrics."""
        metrics = extract_validation_metrics(sample_df)

        assert 'run_date' in metrics
        # Should be ISO format
        datetime.fromisoformat(metrics['run_date'])


# =============================================================================
# VALIDATE SCHEMA STABILITY TESTS
# =============================================================================


class TestValidateSchemaStability:
    """Tests for validate_schema_stability function."""

    def test_passes_on_first_run(self, sample_metrics):
        """Should pass when no previous metrics (first run)."""
        is_valid, issues = validate_schema_stability(
            sample_metrics, None, strict=True
        )

        assert is_valid is True
        assert len(issues) == 0

    def test_passes_with_identical_schema(self, sample_metrics):
        """Should pass when schema is identical."""
        previous = sample_metrics.copy()

        is_valid, issues = validate_schema_stability(
            sample_metrics, previous, strict=True
        )

        assert is_valid is True
        assert len(issues) == 0

    def test_raises_on_missing_columns_strict(self, sample_metrics):
        """Should raise ValueError in strict mode when columns missing."""
        previous = sample_metrics.copy()
        current = sample_metrics.copy()
        current['column_names'] = ['date', 'sales']  # Missing product_code, premium

        with pytest.raises(ValueError) as exc_info:
            validate_schema_stability(current, previous, strict=True)

        assert "Schema mismatch" in str(exc_info.value)
        assert "Missing columns" in str(exc_info.value)

    def test_warns_on_missing_columns_non_strict(self, sample_metrics):
        """Should return warnings in non-strict mode when columns missing."""
        previous = sample_metrics.copy()
        current = sample_metrics.copy()
        current['column_names'] = ['date', 'sales']  # Missing columns

        is_valid, issues = validate_schema_stability(
            current, previous, strict=False
        )

        assert is_valid is False
        assert len(issues) > 0
        assert 'Missing columns' in issues[0]

    def test_detects_added_columns(self, sample_metrics):
        """Should detect newly added columns."""
        previous = sample_metrics.copy()
        current = sample_metrics.copy()
        current['column_names'] = sample_metrics['column_names'] + ['new_feature']

        is_valid, issues = validate_schema_stability(
            current, previous, strict=False
        )

        assert is_valid is False
        assert 'Added columns' in issues[0]

    def test_raises_for_missing_critical_columns(self, sample_metrics):
        """Should raise ValueError when critical columns are missing."""
        current = sample_metrics.copy()
        critical_columns = ['required_feature']

        with pytest.raises(ValueError) as exc_info:
            validate_schema_stability(
                current, None, critical_columns=critical_columns
            )

        assert "Missing critical columns" in str(exc_info.value)
        assert "required_feature" in str(exc_info.value)

    def test_passes_when_critical_columns_present(self, sample_metrics):
        """Should pass when all critical columns are present."""
        current = sample_metrics.copy()
        critical_columns = ['date', 'sales']

        is_valid, issues = validate_schema_stability(
            current, None, critical_columns=critical_columns
        )

        assert is_valid is True


# =============================================================================
# VALIDATE GROWTH PATTERNS TESTS
# =============================================================================


class TestValidateGrowthPatterns:
    """Tests for validate_growth_patterns function."""

    def test_passes_on_first_run(self, sample_metrics, growth_config):
        """Should pass when no previous metrics (first run)."""
        is_valid, warnings, growth_pct = validate_growth_patterns(
            sample_metrics, None, growth_config
        )

        assert is_valid is True
        assert len(warnings) == 0
        assert growth_pct is None

    def test_passes_with_normal_growth(self, sample_metrics, growth_config):
        """Should pass with growth within acceptable range."""
        previous = sample_metrics.copy()
        previous['record_count'] = 95  # 5% growth

        current = sample_metrics.copy()
        current['record_count'] = 100

        is_valid, warnings, growth_pct = validate_growth_patterns(
            current, previous, growth_config
        )

        assert is_valid is True
        assert len(warnings) == 0
        assert growth_pct is not None
        assert growth_pct > 0

    def test_warns_on_data_shrinkage(self, sample_metrics, growth_config):
        """Should warn when data shrinks below threshold."""
        previous = sample_metrics.copy()
        previous['record_count'] = 100

        current = sample_metrics.copy()
        current['record_count'] = 80  # 20% shrinkage

        is_valid, warnings, growth_pct = validate_growth_patterns(
            current, previous, growth_config
        )

        assert is_valid is True  # Still valid, just warning
        assert len(warnings) > 0
        assert 'shrinkage' in warnings[0].lower()
        assert growth_pct < 0

    def test_warns_on_excessive_growth(self, sample_metrics, growth_config):
        """Should warn when growth exceeds maximum threshold."""
        previous = sample_metrics.copy()
        previous['record_count'] = 100

        current = sample_metrics.copy()
        current['record_count'] = 200  # 100% growth

        is_valid, warnings, growth_pct = validate_growth_patterns(
            current, previous, growth_config
        )

        assert is_valid is True  # Still valid, just warning
        assert len(warnings) > 0
        assert 'growth' in warnings[0].lower()

    def test_respects_warn_flags(self, sample_metrics):
        """Should respect warn_on_shrinkage and warn_on_high_growth flags."""
        config = {
            'min_growth_pct': -10.0,
            'max_growth_pct': 50.0,
            'warn_on_shrinkage': False,  # Don't warn
            'warn_on_high_growth': False,  # Don't warn
        }

        previous = sample_metrics.copy()
        previous['record_count'] = 100

        current = sample_metrics.copy()
        current['record_count'] = 50  # 50% shrinkage - should not warn

        is_valid, warnings, growth_pct = validate_growth_patterns(
            current, previous, config
        )

        assert len(warnings) == 0


# =============================================================================
# VALIDATE DATE RANGE PROGRESSION TESTS
# =============================================================================


class TestValidateDateRangeProgression:
    """Tests for validate_date_range_progression function."""

    def test_passes_on_first_run(self, sample_metrics):
        """Should pass when no previous metrics (first run)."""
        is_valid, warnings = validate_date_range_progression(
            sample_metrics, None
        )

        assert is_valid is True
        assert len(warnings) == 0

    def test_passes_when_no_dates(self):
        """Should pass when no date information available."""
        metrics = {
            'min_date': None,
            'max_date': None,
        }

        is_valid, warnings = validate_date_range_progression(
            metrics, None
        )

        assert is_valid is True

    def test_passes_with_normal_progression(self, sample_metrics):
        """Should pass when date range progresses normally."""
        previous = sample_metrics.copy()
        previous['min_date'] = '2023-01-01'
        previous['max_date'] = '2023-03-31'

        current = sample_metrics.copy()
        current['min_date'] = '2023-01-01'  # Same start
        current['max_date'] = '2023-04-30'  # Later end

        is_valid, warnings = validate_date_range_progression(
            current, previous
        )

        assert is_valid is True
        assert len(warnings) == 0

    def test_warns_on_start_date_shift(self, sample_metrics):
        """Should warn when start date shifts significantly."""
        previous = sample_metrics.copy()
        previous['min_date'] = '2023-01-01'
        previous['max_date'] = '2023-03-31'

        current = sample_metrics.copy()
        current['min_date'] = '2023-03-01'  # Shifted 60 days forward
        current['max_date'] = '2023-04-30'

        is_valid, warnings = validate_date_range_progression(
            current, previous, max_start_shift_days=30
        )

        assert is_valid is True  # Still valid, just warning
        assert len(warnings) > 0
        assert 'shifted' in warnings[0].lower()

    def test_warns_on_end_date_regression(self, sample_metrics):
        """Should warn when end date regresses (data outdated)."""
        previous = sample_metrics.copy()
        previous['min_date'] = '2023-01-01'
        previous['max_date'] = '2023-04-30'

        current = sample_metrics.copy()
        current['min_date'] = '2023-01-01'
        current['max_date'] = '2023-03-31'  # Earlier than previous

        is_valid, warnings = validate_date_range_progression(
            current, previous
        )

        assert is_valid is True  # Still valid, just warning
        assert len(warnings) > 0
        assert 'regression' in warnings[0].lower()


# =============================================================================
# METADATA OPERATIONS TESTS
# =============================================================================


class TestLoadPreviousValidationMetadata:
    """Tests for load_previous_validation_metadata function."""

    def test_returns_none_when_file_not_exists(self, tmp_path):
        """Should return None when no metadata file exists."""
        from src.validation.production_validators import load_previous_validation_metadata

        result = load_previous_validation_metadata(
            checkpoint_name='test_checkpoint',
            version=1,
            project_root=str(tmp_path)
        )

        assert result is None

    def test_loads_existing_metadata(self, tmp_path, sample_metrics):
        """Should load metadata from existing file."""
        from src.validation.production_validators import (
            load_previous_validation_metadata,
            save_validation_metadata
        )

        # Save metadata first
        save_validation_metadata(
            sample_metrics,
            checkpoint_name='test_checkpoint',
            version=1,
            project_root=str(tmp_path)
        )

        # Load it back
        result = load_previous_validation_metadata(
            checkpoint_name='test_checkpoint',
            version=1,
            project_root=str(tmp_path)
        )

        assert result is not None
        assert result['record_count'] == sample_metrics['record_count']
        assert result['column_count'] == sample_metrics['column_count']


class TestSaveValidationMetadata:
    """Tests for save_validation_metadata function."""

    def test_saves_to_correct_path(self, tmp_path, sample_metrics):
        """Should save to outputs/metadata directory."""
        from src.validation.production_validators import save_validation_metadata

        path = save_validation_metadata(
            sample_metrics,
            checkpoint_name='test_checkpoint',
            version=1,
            project_root=str(tmp_path)
        )

        assert path.exists()
        assert 'test_checkpoint_v1_metadata.json' in str(path)
        assert 'outputs/metadata' in str(path)

    def test_creates_parent_directories(self, tmp_path, sample_metrics):
        """Should create parent directories if they don't exist."""
        from src.validation.production_validators import save_validation_metadata

        path = save_validation_metadata(
            sample_metrics,
            checkpoint_name='new_checkpoint',
            version=2,
            project_root=str(tmp_path)
        )

        assert path.exists()

    def test_saves_valid_json(self, tmp_path, sample_metrics):
        """Should save valid JSON that can be parsed."""
        from src.validation.production_validators import save_validation_metadata
        import json

        path = save_validation_metadata(
            sample_metrics,
            checkpoint_name='test_checkpoint',
            version=1,
            project_root=str(tmp_path)
        )

        with open(path) as f:
            loaded = json.load(f)

        assert loaded['record_count'] == sample_metrics['record_count']


# =============================================================================
# BUSINESS RULES VALIDATION TESTS
# =============================================================================


class TestCheckPositivePremiums:
    """Tests for _check_positive_premiums function."""

    def test_passes_when_no_negative_values(self):
        """Should pass when all premium values are positive."""
        from src.validation.production_validators import _check_positive_premiums

        df = pd.DataFrame({'premium': [100, 200, 300]})
        rule = {'column': 'premium', 'threshold': 0}

        result = _check_positive_premiums(df, rule, raise_on_error=False)

        assert result is None

    def test_fails_when_threshold_exceeded(self):
        """Should return error when negative count exceeds threshold."""
        from src.validation.production_validators import _check_positive_premiums

        df = pd.DataFrame({'premium': [-100, -200, 300]})
        rule = {'column': 'premium', 'threshold': 1}

        result = _check_positive_premiums(df, rule, raise_on_error=False)

        assert result is not None
        assert 'negative premium' in result.lower()

    def test_raises_when_raise_on_error_true(self):
        """Should raise DataValidationError when raise_on_error=True."""
        from src.validation.production_validators import _check_positive_premiums
        from src.core.exceptions import DataValidationError

        df = pd.DataFrame({'premium': [-100, -200, 300]})
        rule = {'column': 'premium', 'threshold': 0}

        with pytest.raises(DataValidationError):
            _check_positive_premiums(df, rule, raise_on_error=True)


class TestCheckDateConsistency:
    """Tests for _check_date_consistency function."""

    def test_passes_when_dates_consistent(self):
        """Should pass when app date precedes contract date."""
        from src.validation.production_validators import _check_date_consistency

        df = pd.DataFrame({
            'app_date': pd.to_datetime(['2023-01-01', '2023-02-01']),
            'contract_date': pd.to_datetime(['2023-01-15', '2023-02-15'])
        })
        rule = {'app_col': 'app_date', 'contract_col': 'contract_date'}

        result = _check_date_consistency(df, rule, raise_on_error=False)

        assert result is None

    def test_fails_when_dates_inconsistent(self):
        """Should fail when app date after contract date."""
        from src.validation.production_validators import _check_date_consistency

        df = pd.DataFrame({
            'app_date': pd.to_datetime(['2023-02-01', '2023-03-01']),
            'contract_date': pd.to_datetime(['2023-01-15', '2023-02-15'])
        })
        rule = {'app_col': 'app_date', 'contract_col': 'contract_date'}

        result = _check_date_consistency(df, rule, raise_on_error=False)

        assert result is not None
        assert 'inconsistencies' in result or 'application date' in result.lower()


class TestCheckValidBufferRates:
    """Tests for _check_valid_buffer_rates function."""

    def test_passes_when_all_values_valid(self):
        """Should pass when all buffer rates are in valid set."""
        from src.validation.production_validators import _check_valid_buffer_rates

        df = pd.DataFrame({'buffer_rate': [10, 20, 10, 20]})
        rule = {'column': 'buffer_rate', 'valid_values': [10, 20]}

        result = _check_valid_buffer_rates(df, rule, raise_on_error=False)

        assert result is None

    def test_fails_when_invalid_values_present(self):
        """Should fail when invalid buffer rates found."""
        from src.validation.production_validators import _check_valid_buffer_rates

        df = pd.DataFrame({'buffer_rate': [10, 20, 15, 25]})
        rule = {'column': 'buffer_rate', 'valid_values': [10, 20]}

        result = _check_valid_buffer_rates(df, rule, raise_on_error=False)

        assert result is not None
        assert 'invalid buffer rates' in result.lower()

    def test_skips_when_column_missing(self):
        """Should skip validation when column doesn't exist."""
        from src.validation.production_validators import _check_valid_buffer_rates

        df = pd.DataFrame({'other_column': [1, 2, 3]})
        rule = {'column': 'buffer_rate', 'valid_values': [10, 20]}

        result = _check_valid_buffer_rates(df, rule, raise_on_error=False)

        assert result is None


class TestCheckAggregationRatio:
    """Tests for _check_aggregation_ratio function."""

    def test_passes_when_ratio_in_range(self):
        """Should pass when aggregation ratio is within expected range."""
        from src.validation.production_validators import _check_aggregation_ratio

        rule = {'expected_min': 5.0, 'expected_max': 10.0, 'actual': 7.5}

        result = _check_aggregation_ratio(rule, raise_on_error=False)

        assert result is None

    def test_fails_when_ratio_out_of_range(self):
        """Should fail when aggregation ratio outside expected range."""
        from src.validation.production_validators import _check_aggregation_ratio

        rule = {'expected_min': 5.0, 'expected_max': 10.0, 'actual': 15.0}

        result = _check_aggregation_ratio(rule, raise_on_error=False)

        assert result is not None
        assert 'aggregation ratio' in result.lower()


class TestCheckSalesPreservation:
    """Tests for _check_sales_preservation function."""

    def test_passes_when_totals_match(self):
        """Should pass when sales totals are within tolerance."""
        from src.validation.production_validators import _check_sales_preservation

        rule = {'daily_total': 100000, 'weekly_total': 100500, 'tolerance_pct': 1.0}

        result = _check_sales_preservation(rule, raise_on_error=False)

        assert result is None

    def test_fails_when_totals_differ(self):
        """Should fail when sales totals differ beyond tolerance."""
        from src.validation.production_validators import _check_sales_preservation

        rule = {'daily_total': 100000, 'weekly_total': 110000, 'tolerance_pct': 1.0}

        result = _check_sales_preservation(rule, raise_on_error=False)

        assert result is not None
        assert 'differ' in result.lower()


class TestValidateBusinessRules:
    """Tests for validate_business_rules function."""

    def test_returns_empty_list_when_no_rules(self, sample_df):
        """Should return empty list when no rules provided."""
        from src.validation.production_validators import validate_business_rules

        result = validate_business_rules(sample_df, rules=None)

        assert result == []

    def test_collects_violations(self, sample_df):
        """Should collect all violations when raise_on_error=False."""
        from src.validation.production_validators import validate_business_rules

        # Create DataFrame with violations
        df = sample_df.copy()
        df['premium'] = [-100] * len(df)

        rules = {
            'positive_premiums': {'column': 'premium', 'threshold': 0}
        }

        result = validate_business_rules(df, rules, raise_on_error=False)

        assert len(result) > 0

    def test_checks_multiple_rules(self, sample_df):
        """Should check all provided rules."""
        from src.validation.production_validators import validate_business_rules

        rules = {
            'aggregation_ratio': {'expected_min': 1, 'expected_max': 2, 'actual': 5},
        }

        result = validate_business_rules(sample_df, rules, raise_on_error=False)

        assert len(result) > 0


# =============================================================================
# STATUS DETERMINATION TESTS
# =============================================================================


class TestDetermineValidationStatus:
    """Tests for _determine_validation_status function."""

    def test_returns_baseline_on_first_run(self):
        """Should return BASELINE when no previous metrics."""
        from src.validation.production_validators import _determine_validation_status

        result = _determine_validation_status(
            previous_metrics=None, issues=[], warnings=[]
        )

        assert result == 'BASELINE'

    def test_returns_failed_with_issues(self, sample_metrics):
        """Should return FAILED when issues present."""
        from src.validation.production_validators import _determine_validation_status

        result = _determine_validation_status(
            previous_metrics=sample_metrics, issues=['Critical issue'], warnings=[]
        )

        assert result == 'FAILED'

    def test_returns_warnings_with_warnings_only(self, sample_metrics):
        """Should return WARNINGS when only warnings present."""
        from src.validation.production_validators import _determine_validation_status

        result = _determine_validation_status(
            previous_metrics=sample_metrics, issues=[], warnings=['Some warning']
        )

        assert result == 'WARNINGS'

    def test_returns_passed_when_clean(self, sample_metrics):
        """Should return PASSED when no issues or warnings."""
        from src.validation.production_validators import _determine_validation_status

        result = _determine_validation_status(
            previous_metrics=sample_metrics, issues=[], warnings=[]
        )

        assert result == 'PASSED'


class TestPrintValidationSummary:
    """Tests for _print_validation_summary function."""

    def test_prints_baseline_message(self, sample_metrics, capsys):
        """Should print baseline message on first run."""
        from src.validation.production_validators import _print_validation_summary

        _print_validation_summary(
            checkpoint_name='test',
            status='BASELINE',
            issues=[],
            warnings=[],
            current_metrics=sample_metrics,
            previous_metrics=None,
            growth_pct=None,
            date_column=None
        )

        captured = capsys.readouterr()
        assert 'baseline' in captured.out.lower()

    def test_prints_fail_with_issues(self, sample_metrics, capsys):
        """Should print FAIL and issues when failed."""
        from src.validation.production_validators import _print_validation_summary

        _print_validation_summary(
            checkpoint_name='test',
            status='FAILED',
            issues=['Critical issue'],
            warnings=[],
            current_metrics=sample_metrics,
            previous_metrics=sample_metrics,
            growth_pct=None,
            date_column=None
        )

        captured = capsys.readouterr()
        assert 'FAIL' in captured.out
        assert 'Critical issue' in captured.out

    def test_prints_warnings(self, sample_metrics, capsys):
        """Should print warnings when present."""
        from src.validation.production_validators import _print_validation_summary

        _print_validation_summary(
            checkpoint_name='test',
            status='WARNINGS',
            issues=[],
            warnings=['Warning message'],
            current_metrics=sample_metrics,
            previous_metrics=sample_metrics,
            growth_pct=None,
            date_column=None
        )

        captured = capsys.readouterr()
        assert 'WARN' in captured.out
        assert 'Warning message' in captured.out

    def test_prints_growth_when_available(self, sample_metrics, capsys):
        """Should print growth percentage when available."""
        from src.validation.production_validators import _print_validation_summary

        _print_validation_summary(
            checkpoint_name='test',
            status='PASSED',
            issues=[],
            warnings=[],
            current_metrics=sample_metrics,
            previous_metrics=sample_metrics,
            growth_pct=5.5,
            date_column=None
        )

        captured = capsys.readouterr()
        assert 'growth' in captured.out.lower()

    def test_prints_date_range_when_available(self, sample_metrics, capsys):
        """Should print date range when date column provided."""
        from src.validation.production_validators import _print_validation_summary

        _print_validation_summary(
            checkpoint_name='test',
            status='PASSED',
            issues=[],
            warnings=[],
            current_metrics=sample_metrics,
            previous_metrics=None,
            growth_pct=None,
            date_column='date'
        )

        captured = capsys.readouterr()
        assert 'Date range' in captured.out


# =============================================================================
# RUN PRODUCTION VALIDATION CHECKPOINT TESTS
# =============================================================================


class TestRunProductionValidationCheckpoint:
    """Tests for run_production_validation_checkpoint function."""

    @pytest.fixture
    def validation_config(self, tmp_path, growth_config):
        """Create standard validation config."""
        return {
            'checkpoint_name': 'test_checkpoint',
            'version': 1,
            'project_root': str(tmp_path),
            'strict_schema': False,
            'growth_config': growth_config,
            'critical_columns': None
        }

    def test_returns_validation_result(self, sample_df, validation_config):
        """Should return ValidationResult object."""
        from src.validation.production_validators import (
            run_production_validation_checkpoint,
            ValidationResult
        )

        result = run_production_validation_checkpoint(
            df=sample_df,
            config=validation_config
        )

        assert isinstance(result, ValidationResult)
        assert result.checkpoint_name == 'test_checkpoint'

    def test_baseline_status_on_first_run(self, sample_df, validation_config):
        """Should return BASELINE status on first run."""
        from src.validation.production_validators import run_production_validation_checkpoint

        result = run_production_validation_checkpoint(
            df=sample_df,
            config=validation_config
        )

        assert result.status == 'BASELINE'

    def test_passed_status_on_second_run(self, sample_df, validation_config):
        """Should return PASSED status on subsequent identical run."""
        from src.validation.production_validators import run_production_validation_checkpoint

        # First run - establishes baseline
        run_production_validation_checkpoint(
            df=sample_df,
            config=validation_config
        )

        # Second run - should pass
        result = run_production_validation_checkpoint(
            df=sample_df,
            config=validation_config
        )

        assert result.status == 'PASSED'

    def test_includes_current_metrics(self, sample_df, validation_config):
        """Should include current metrics in result."""
        from src.validation.production_validators import run_production_validation_checkpoint

        result = run_production_validation_checkpoint(
            df=sample_df,
            config=validation_config
        )

        assert result.current_metrics is not None
        assert result.current_metrics['record_count'] == len(sample_df)

    def test_saves_metadata(self, sample_df, validation_config, tmp_path):
        """Should save metadata to file."""
        from src.validation.production_validators import run_production_validation_checkpoint

        result = run_production_validation_checkpoint(
            df=sample_df,
            config=validation_config
        )

        assert result.metadata_path.exists()

    def test_validates_business_rules(self, sample_df, validation_config):
        """Should validate business rules when provided."""
        from src.validation.production_validators import run_production_validation_checkpoint

        # Create DataFrame with bad data
        df = sample_df.copy()
        df['premium'] = [-100] * len(df)

        business_rules = {
            'positive_premiums': {'column': 'premium', 'threshold': 0}
        }

        result = run_production_validation_checkpoint(
            df=df,
            config=validation_config,
            business_rules=business_rules
        )

        assert len(result.warnings) > 0 or result.status != 'PASSED'

    def test_extracts_date_metrics(self, sample_df, validation_config):
        """Should extract date metrics when date column provided."""
        from src.validation.production_validators import run_production_validation_checkpoint

        result = run_production_validation_checkpoint(
            df=sample_df,
            config=validation_config,
            date_column='date'
        )

        assert result.current_metrics['min_date'] is not None
        assert result.current_metrics['max_date'] is not None


# =============================================================================
# VALIDATION STAGE HELPER TESTS
# =============================================================================


class TestValidateSchemaAndRules:
    """Tests for _validate_schema_and_rules function."""

    @pytest.fixture
    def validation_config(self, tmp_path, growth_config):
        """Create standard validation config."""
        return {
            'checkpoint_name': 'test',
            'version': 1,
            'project_root': str(tmp_path),
            'strict_schema': False,
            'growth_config': growth_config,
            'critical_columns': None
        }

    def test_returns_empty_on_first_run(self, sample_df, sample_metrics, validation_config):
        """Should return empty lists on first run."""
        from src.validation.production_validators import _validate_schema_and_rules

        issues, warnings = _validate_schema_and_rules(
            df=sample_df,
            current_metrics=sample_metrics,
            previous_metrics=None,
            config=validation_config,
            business_rules=None
        )

        assert issues == []
        assert warnings == []


class TestValidatePatterns:
    """Tests for _validate_patterns function."""

    @pytest.fixture
    def validation_config(self, tmp_path, growth_config):
        """Create standard validation config."""
        return {
            'checkpoint_name': 'test',
            'version': 1,
            'project_root': str(tmp_path),
            'strict_schema': False,
            'growth_config': growth_config,
            'critical_columns': None
        }

    def test_returns_none_growth_on_first_run(self, sample_metrics, validation_config):
        """Should return None growth percentage on first run."""
        from src.validation.production_validators import _validate_patterns

        warnings, growth_pct = _validate_patterns(
            current_metrics=sample_metrics,
            previous_metrics=None,
            config=validation_config,
            date_column=None
        )

        assert growth_pct is None

    def test_calculates_growth_with_previous(self, sample_metrics, validation_config):
        """Should calculate growth when previous metrics available."""
        from src.validation.production_validators import _validate_patterns

        previous = sample_metrics.copy()
        previous['record_count'] = 90

        warnings, growth_pct = _validate_patterns(
            current_metrics=sample_metrics,
            previous_metrics=previous,
            config=validation_config,
            date_column=None
        )

        assert growth_pct is not None
        assert growth_pct > 0


# =============================================================================
# SUMMARY TEST
# =============================================================================


def test_production_validators_coverage_summary():
    """Summary of production validators test coverage.

    Tested Functions:
    - extract_metrics: Basic extraction, date handling, run_date
    - validate_schema_stability: First run, identical, missing, added, critical
    - validate_growth_patterns: First run, normal, shrinkage, excessive growth
    - validate_date_range_progression: First run, normal, shift, regression
    - load_previous_validation_metadata: Not exists, exists
    - save_validation_metadata: Path, directories, JSON validity
    - Business rule checks: positive_premiums, date_consistency, buffer_rates, etc.
    - validate_business_rules: Empty, collect, multiple rules
    - _determine_validation_status: All status values
    - _print_validation_summary: All statuses, growth, dates
    - run_production_validation_checkpoint: Full integration
    - _validate_schema_and_rules: First run
    - _validate_patterns: First run, with previous

    Coverage Target: 75%+
    """
    pass

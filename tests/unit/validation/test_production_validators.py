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
# SUMMARY TEST
# =============================================================================


def test_production_validators_coverage_summary():
    """Summary of production validators test coverage.

    Tested Functions:
    - extract_metrics: Basic extraction, date handling, run_date
    - validate_schema_stability: First run, identical, missing, added, critical
    - validate_growth_patterns: First run, normal, shrinkage, excessive growth
    - validate_date_range_progression: First run, normal, shift, regression

    Coverage Target: 60%
    """
    pass

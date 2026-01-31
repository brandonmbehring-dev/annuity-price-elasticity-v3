"""
Tests for Multiple Testing Types Module.

Tests cover:
- MultipleTestingResults dataclass
- Field validation
- Data structure integrity

Design Principles:
- Simple unit tests for dataclass
- Verify all fields work correctly
- Test default values and type handling

Author: Claude Code
Date: 2026-01-30
"""

import pytest
import pandas as pd
import numpy as np

from src.features.selection.enhancements.multiple_testing.multiple_testing_types import (
    MultipleTestingResults,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        'model_id': ['m1', 'm2', 'm3'],
        'aic': [100.0, 105.0, 95.0],
        'r_squared': [0.8, 0.75, 0.82],
    })


@pytest.fixture
def sample_correction_impact():
    """Sample correction impact dict."""
    return {
        'models_rejected': 50,
        'models_retained': 10,
        'rejection_rate_percent': 83.3,
        'retention_rate_percent': 16.7,
        'correction_stringency': 'HIGH',
        'business_impact': 'Test impact',
    }


@pytest.fixture
def sample_statistical_power():
    """Sample statistical power dict."""
    return {
        'statistical_power': 0.65,
        'type_ii_error_rate': 0.35,
        'power_interpretation': 'MODERATE',
        'effect_size_assumed': 0.1,
    }


# =============================================================================
# Tests for MultipleTestingResults Dataclass
# =============================================================================


class TestMultipleTestingResults:
    """Tests for MultipleTestingResults dataclass."""

    def test_creation_with_all_fields(
        self, sample_dataframe, sample_correction_impact, sample_statistical_power
    ):
        """Test dataclass creation with all required fields."""
        result = MultipleTestingResults(
            method='bonferroni',
            original_alpha=0.05,
            corrected_alpha=0.001,
            n_tests=100,
            significant_models=sample_dataframe,
            rejected_models=sample_dataframe,
            correction_impact=sample_correction_impact,
            statistical_power=sample_statistical_power,
        )

        assert result.method == 'bonferroni'
        assert result.original_alpha == 0.05
        assert result.corrected_alpha == 0.001
        assert result.n_tests == 100

    def test_method_field_values(
        self, sample_dataframe, sample_correction_impact, sample_statistical_power
    ):
        """Test that method field accepts valid method names."""
        valid_methods = ['bonferroni', 'fdr_bh', 'reduced_space']

        for method in valid_methods:
            result = MultipleTestingResults(
                method=method,
                original_alpha=0.05,
                corrected_alpha=0.001,
                n_tests=50,
                significant_models=sample_dataframe,
                rejected_models=sample_dataframe,
                correction_impact=sample_correction_impact,
                statistical_power=sample_statistical_power,
            )
            assert result.method == method

    def test_alpha_values(
        self, sample_dataframe, sample_correction_impact, sample_statistical_power
    ):
        """Test that alpha values are stored correctly."""
        result = MultipleTestingResults(
            method='bonferroni',
            original_alpha=0.05,
            corrected_alpha=0.00005,
            n_tests=1000,
            significant_models=sample_dataframe,
            rejected_models=sample_dataframe,
            correction_impact=sample_correction_impact,
            statistical_power=sample_statistical_power,
        )

        assert result.original_alpha == 0.05
        assert result.corrected_alpha == 0.00005
        assert result.corrected_alpha < result.original_alpha

    def test_dataframe_fields(
        self, sample_dataframe, sample_correction_impact, sample_statistical_power
    ):
        """Test that DataFrame fields work correctly."""
        significant = sample_dataframe.head(1)
        rejected = sample_dataframe.tail(2)

        result = MultipleTestingResults(
            method='fdr_bh',
            original_alpha=0.05,
            corrected_alpha=0.01,
            n_tests=50,
            significant_models=significant,
            rejected_models=rejected,
            correction_impact=sample_correction_impact,
            statistical_power=sample_statistical_power,
        )

        assert isinstance(result.significant_models, pd.DataFrame)
        assert isinstance(result.rejected_models, pd.DataFrame)
        assert len(result.significant_models) == 1
        assert len(result.rejected_models) == 2

    def test_empty_dataframes(
        self, sample_correction_impact, sample_statistical_power
    ):
        """Test that empty DataFrames are handled."""
        empty_df = pd.DataFrame()

        result = MultipleTestingResults(
            method='bonferroni',
            original_alpha=0.05,
            corrected_alpha=0.0001,
            n_tests=500,
            significant_models=empty_df,
            rejected_models=empty_df,
            correction_impact=sample_correction_impact,
            statistical_power=sample_statistical_power,
        )

        assert len(result.significant_models) == 0
        assert len(result.rejected_models) == 0

    def test_correction_impact_dict(
        self, sample_dataframe, sample_statistical_power
    ):
        """Test that correction_impact dict is stored correctly."""
        impact = {
            'models_rejected': 80,
            'models_retained': 20,
            'rejection_rate_percent': 80.0,
        }

        result = MultipleTestingResults(
            method='bonferroni',
            original_alpha=0.05,
            corrected_alpha=0.0005,
            n_tests=100,
            significant_models=sample_dataframe,
            rejected_models=sample_dataframe,
            correction_impact=impact,
            statistical_power=sample_statistical_power,
        )

        assert result.correction_impact['models_rejected'] == 80
        assert result.correction_impact['rejection_rate_percent'] == 80.0

    def test_statistical_power_dict(
        self, sample_dataframe, sample_correction_impact
    ):
        """Test that statistical_power dict is stored correctly."""
        power = {
            'statistical_power': 0.8,
            'type_ii_error_rate': 0.2,
        }

        result = MultipleTestingResults(
            method='fdr_bh',
            original_alpha=0.05,
            corrected_alpha=0.01,
            n_tests=100,
            significant_models=sample_dataframe,
            rejected_models=sample_dataframe,
            correction_impact=sample_correction_impact,
            statistical_power=power,
        )

        assert result.statistical_power['statistical_power'] == 0.8
        assert result.statistical_power['type_ii_error_rate'] == 0.2

    def test_n_tests_integer(
        self, sample_dataframe, sample_correction_impact, sample_statistical_power
    ):
        """Test that n_tests is stored as integer."""
        result = MultipleTestingResults(
            method='bonferroni',
            original_alpha=0.05,
            corrected_alpha=0.0005,
            n_tests=100,
            significant_models=sample_dataframe,
            rejected_models=sample_dataframe,
            correction_impact=sample_correction_impact,
            statistical_power=sample_statistical_power,
        )

        assert isinstance(result.n_tests, int)
        assert result.n_tests == 100


class TestMultipleTestingResultsEdgeCases:
    """Edge case tests for MultipleTestingResults."""

    def test_very_small_corrected_alpha(self, sample_dataframe):
        """Test handling of very small corrected alpha."""
        result = MultipleTestingResults(
            method='bonferroni',
            original_alpha=0.05,
            corrected_alpha=1e-10,  # Very small
            n_tests=50000000,
            significant_models=sample_dataframe,
            rejected_models=sample_dataframe,
            correction_impact={},
            statistical_power={},
        )

        assert result.corrected_alpha == 1e-10

    def test_large_n_tests(self, sample_dataframe):
        """Test handling of large n_tests."""
        result = MultipleTestingResults(
            method='bonferroni',
            original_alpha=0.05,
            corrected_alpha=0.0000001,
            n_tests=1000000,
            significant_models=sample_dataframe,
            rejected_models=sample_dataframe,
            correction_impact={},
            statistical_power={},
        )

        assert result.n_tests == 1000000

    def test_empty_dicts(self, sample_dataframe):
        """Test with empty correction_impact and statistical_power dicts."""
        result = MultipleTestingResults(
            method='fdr_bh',
            original_alpha=0.05,
            corrected_alpha=0.01,
            n_tests=50,
            significant_models=sample_dataframe,
            rejected_models=sample_dataframe,
            correction_impact={},
            statistical_power={},
        )

        assert result.correction_impact == {}
        assert result.statistical_power == {}

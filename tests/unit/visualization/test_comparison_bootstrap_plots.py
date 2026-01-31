"""
Tests for src.visualization.comparison_bootstrap_plots module.

Tests bootstrap distribution visualizations:
- Dict-format bootstrap placeholder
- NamedTuple bootstrap histogram

Target coverage: 80%+
"""

from collections import namedtuple
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.visualization.comparison_bootstrap_plots import (
    plot_dict_format_bootstrap,
    plot_namedtuple_bootstrap_histogram,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_axes():
    """Create a mock matplotlib axes for testing."""
    ax = MagicMock(spec=plt.Axes)
    ax.transAxes = 'transAxes_mock'
    return ax


@pytest.fixture
def default_colors():
    """Default color scheme for plots."""
    return {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'tertiary': '#2ca02c',
        'highlight': '#d62728',
        'light_gray': '#f0f0f0',
    }


@pytest.fixture
def sample_dict_result():
    """Sample dict-format bootstrap result."""
    return {
        'features': 'feat_a+feat_b',
        'aic': 100.5,
        'r_squared': 0.85,
        'aic_stability_cv': 0.05,
    }


@pytest.fixture
def sample_namedtuple_result():
    """Sample namedtuple-format bootstrap result with samples."""
    BootstrapResult = namedtuple(
        'BootstrapResult',
        ['model_features', 'bootstrap_samples', 'stability_metrics',
         'confidence_intervals', 'original_aic']
    )
    return BootstrapResult(
        model_features='feat_a+feat_b+feat_c',
        bootstrap_samples=[
            {'aic': 100.0, 'r_squared': 0.85},
            {'aic': 101.5, 'r_squared': 0.84},
            {'aic': 99.0, 'r_squared': 0.86},
            {'aic': 102.0, 'r_squared': 0.83},
            {'aic': 98.5, 'r_squared': 0.87},
        ],
        stability_metrics={'aic_cv': 0.015, 'successful_fit_rate': 0.95},
        confidence_intervals={95: (98.0, 102.5)},
        original_aic=100.5
    )


@pytest.fixture
def namedtuple_result_no_samples():
    """Namedtuple result with no bootstrap samples."""
    BootstrapResult = namedtuple(
        'BootstrapResult',
        ['model_features', 'bootstrap_samples', 'stability_metrics',
         'confidence_intervals', 'original_aic']
    )
    return BootstrapResult(
        model_features='empty_model',
        bootstrap_samples=[],
        stability_metrics={'aic_cv': 0},
        confidence_intervals={},
        original_aic=100.0
    )


@pytest.fixture
def namedtuple_result_no_aic():
    """Namedtuple result with samples but no AIC values."""
    BootstrapResult = namedtuple(
        'BootstrapResult',
        ['model_features', 'bootstrap_samples', 'stability_metrics',
         'confidence_intervals', 'original_aic']
    )
    return BootstrapResult(
        model_features='no_aic_model',
        bootstrap_samples=[
            {'r_squared': 0.85},  # No AIC
            {'r_squared': 0.84},
        ],
        stability_metrics={'aic_cv': 0},
        confidence_intervals={},
        original_aic=100.0
    )


# =============================================================================
# DICT FORMAT BOOTSTRAP TESTS
# =============================================================================


class TestPlotDictFormatBootstrap:
    """Tests for plot_dict_format_bootstrap function."""

    def test_basic_dict_plot(self, mock_axes, sample_dict_result, default_colors):
        """Test basic dict format placeholder plot."""
        cv = 0.05

        plot_dict_format_bootstrap(mock_axes, sample_dict_result, cv, default_colors)

        # Verify text was called with correct content
        mock_axes.text.assert_called_once()
        call_args = mock_axes.text.call_args
        text_content = call_args[0][2]

        assert 'feat_a+feat_b' in text_content
        assert '0.0500' in text_content  # CV formatted
        assert 'not available' in text_content

    def test_dict_plot_with_missing_features(self, mock_axes, default_colors):
        """Test dict format with missing features key."""
        result = {'aic': 100.0}  # Missing 'features'
        cv = 0.1

        plot_dict_format_bootstrap(mock_axes, result, cv, default_colors)

        call_args = mock_axes.text.call_args
        text_content = call_args[0][2]
        assert 'Unknown' in text_content

    def test_dict_plot_positioning(self, mock_axes, sample_dict_result, default_colors):
        """Test text positioning at center."""
        cv = 0.05

        plot_dict_format_bootstrap(mock_axes, sample_dict_result, cv, default_colors)

        call_args = mock_axes.text.call_args
        # Check x, y coordinates are at center (0.5, 0.5)
        assert call_args[0][0] == 0.5
        assert call_args[0][1] == 0.5

    def test_dict_plot_uses_light_gray_box(self, mock_axes, sample_dict_result, default_colors):
        """Test that light gray background box is used."""
        cv = 0.05

        plot_dict_format_bootstrap(mock_axes, sample_dict_result, cv, default_colors)

        call_args = mock_axes.text.call_args
        bbox = call_args[1]['bbox']
        assert bbox['facecolor'] == default_colors['light_gray']

    def test_dict_plot_high_cv_value(self, mock_axes, sample_dict_result, default_colors):
        """Test with high CV value formatting."""
        cv = 1.2345

        plot_dict_format_bootstrap(mock_axes, sample_dict_result, cv, default_colors)

        call_args = mock_axes.text.call_args
        text_content = call_args[0][2]
        assert '1.2345' in text_content


# =============================================================================
# NAMEDTUPLE BOOTSTRAP HISTOGRAM TESTS
# =============================================================================


class TestPlotNamedtupleBootstrapHistogram:
    """Tests for plot_namedtuple_bootstrap_histogram function."""

    def test_basic_histogram(self, mock_axes, sample_namedtuple_result, default_colors):
        """Test basic histogram plot with valid samples."""
        rank = 1

        plot_namedtuple_bootstrap_histogram(
            mock_axes, sample_namedtuple_result, rank, default_colors
        )

        # Verify histogram was created
        mock_axes.hist.assert_called_once()
        # Verify vertical line for original AIC
        mock_axes.axvline.assert_called_once()
        # Verify CI shading
        mock_axes.axvspan.assert_called_once()
        # Verify labels
        mock_axes.set_xlabel.assert_called_with('AIC Score')
        mock_axes.set_ylabel.assert_called_with('Density')
        mock_axes.legend.assert_called_once()
        mock_axes.grid.assert_called_once()

    def test_histogram_with_no_samples(
        self, mock_axes, namedtuple_result_no_samples, default_colors
    ):
        """Test histogram with empty bootstrap samples."""
        rank = 1

        plot_namedtuple_bootstrap_histogram(
            mock_axes, namedtuple_result_no_samples, rank, default_colors
        )

        # Should show placeholder text, not histogram
        mock_axes.text.assert_called_once()
        mock_axes.hist.assert_not_called()

        call_args = mock_axes.text.call_args
        text_content = call_args[0][2]
        assert 'No bootstrap samples' in text_content

    def test_histogram_with_no_aic_samples(
        self, mock_axes, namedtuple_result_no_aic, default_colors
    ):
        """Test histogram when samples have no AIC values."""
        rank = 2

        plot_namedtuple_bootstrap_histogram(
            mock_axes, namedtuple_result_no_aic, rank, default_colors
        )

        # Should show placeholder text, not histogram
        mock_axes.text.assert_called_once()
        mock_axes.hist.assert_not_called()

        call_args = mock_axes.text.call_args
        text_content = call_args[0][2]
        assert 'No valid AIC samples' in text_content

    def test_histogram_title_includes_rank(
        self, mock_axes, sample_namedtuple_result, default_colors
    ):
        """Test histogram title includes stability rank."""
        rank = 3

        plot_namedtuple_bootstrap_histogram(
            mock_axes, sample_namedtuple_result, rank, default_colors
        )

        mock_axes.set_title.assert_called_once()
        call_args = mock_axes.set_title.call_args
        title = call_args[0][0]
        assert '#3' in title

    def test_histogram_model_name_truncated(
        self, mock_axes, sample_namedtuple_result, default_colors
    ):
        """Test long model names are truncated in title."""
        rank = 1

        plot_namedtuple_bootstrap_histogram(
            mock_axes, sample_namedtuple_result, rank, default_colors
        )

        call_args = mock_axes.set_title.call_args
        title = call_args[0][0]
        # Model name should be truncated to 20 chars + '...'
        assert '...' in title

    def test_histogram_statistics_text(
        self, mock_axes, sample_namedtuple_result, default_colors
    ):
        """Test statistics text box is added."""
        rank = 1

        plot_namedtuple_bootstrap_histogram(
            mock_axes, sample_namedtuple_result, rank, default_colors
        )

        # text called twice: once for stats, but actually only once for stats
        # (no_samples and no_aic cases call text once each)
        # For valid samples, text is called once for statistics
        text_calls = mock_axes.text.call_args_list
        assert len(text_calls) >= 1

        # Check stats text includes mean, std, cv
        stats_call = text_calls[0]
        text_content = stats_call[0][2]
        assert 'Mean:' in text_content
        assert 'Std:' in text_content
        assert 'CV:' in text_content

    def test_histogram_without_95_ci(self, mock_axes, default_colors):
        """Test histogram when 95% CI is not present."""
        BootstrapResult = namedtuple(
            'BootstrapResult',
            ['model_features', 'bootstrap_samples', 'stability_metrics',
             'confidence_intervals', 'original_aic']
        )
        result = BootstrapResult(
            model_features='model',
            bootstrap_samples=[{'aic': 100.0}, {'aic': 101.0}, {'aic': 99.0}],
            stability_metrics={'aic_cv': 0.01},
            confidence_intervals={90: (99.0, 101.0)},  # Only 90% CI, not 95%
            original_aic=100.0
        )
        rank = 1

        plot_namedtuple_bootstrap_histogram(mock_axes, result, rank, default_colors)

        # axvspan should not be called without 95% CI
        mock_axes.axvspan.assert_not_called()

    def test_histogram_bins_calculation(
        self, mock_axes, sample_namedtuple_result, default_colors
    ):
        """Test histogram bins are calculated correctly."""
        rank = 1

        plot_namedtuple_bootstrap_histogram(
            mock_axes, sample_namedtuple_result, rank, default_colors
        )

        hist_call = mock_axes.hist.call_args
        # With 5 samples: min(15, 5//2 + 1) = min(15, 3) = 3 bins
        assert hist_call[1]['bins'] == 3

    def test_histogram_uses_correct_colors(
        self, mock_axes, sample_namedtuple_result, default_colors
    ):
        """Test histogram uses correct color scheme."""
        rank = 1

        plot_namedtuple_bootstrap_histogram(
            mock_axes, sample_namedtuple_result, rank, default_colors
        )

        # Histogram uses primary color
        hist_call = mock_axes.hist.call_args
        assert hist_call[1]['color'] == default_colors['primary']

        # Original AIC line uses highlight color
        axvline_call = mock_axes.axvline.call_args
        assert axvline_call[1]['color'] == default_colors['highlight']

        # CI shading uses tertiary color
        axvspan_call = mock_axes.axvspan.call_args
        assert axvspan_call[1]['color'] == default_colors['tertiary']


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases for bootstrap plots."""

    def test_dict_with_empty_features(self, mock_axes, default_colors):
        """Test dict with empty string features."""
        result = {'features': ''}
        cv = 0.1

        plot_dict_format_bootstrap(mock_axes, result, cv, default_colors)

        # Should not crash
        mock_axes.text.assert_called_once()

    def test_namedtuple_with_none_samples_list(self, mock_axes, default_colors):
        """Test namedtuple where samples is None."""
        BootstrapResult = namedtuple(
            'BootstrapResult',
            ['model_features', 'bootstrap_samples', 'stability_metrics',
             'confidence_intervals', 'original_aic']
        )
        result = BootstrapResult(
            model_features='model',
            bootstrap_samples=None,
            stability_metrics={'aic_cv': 0},
            confidence_intervals={},
            original_aic=100.0
        )
        rank = 1

        plot_namedtuple_bootstrap_histogram(mock_axes, result, rank, default_colors)

        # Should show placeholder
        mock_axes.text.assert_called_once()
        mock_axes.hist.assert_not_called()

    def test_namedtuple_with_mixed_sample_types(self, mock_axes, default_colors):
        """Test samples with non-dict elements mixed in."""
        BootstrapResult = namedtuple(
            'BootstrapResult',
            ['model_features', 'bootstrap_samples', 'stability_metrics',
             'confidence_intervals', 'original_aic']
        )
        result = BootstrapResult(
            model_features='model',
            bootstrap_samples=[
                {'aic': 100.0},
                'not_a_dict',  # Should be skipped
                {'aic': 101.0},
                None,  # Should be skipped
            ],
            stability_metrics={'aic_cv': 0.01},
            confidence_intervals={95: (99.0, 101.0)},
            original_aic=100.0
        )
        rank = 1

        plot_namedtuple_bootstrap_histogram(mock_axes, result, rank, default_colors)

        # Should still create histogram with valid samples
        mock_axes.hist.assert_called_once()
        hist_call = mock_axes.hist.call_args
        aic_values = hist_call[0][0]
        assert len(aic_values) == 2  # Only two valid AIC values

    def test_namedtuple_with_missing_cv_metric(self, mock_axes, default_colors):
        """Test when stability_metrics doesn't have aic_cv."""
        BootstrapResult = namedtuple(
            'BootstrapResult',
            ['model_features', 'bootstrap_samples', 'stability_metrics',
             'confidence_intervals', 'original_aic']
        )
        result = BootstrapResult(
            model_features='model',
            bootstrap_samples=[{'aic': 100.0}, {'aic': 101.0}],
            stability_metrics={},  # Missing aic_cv
            confidence_intervals={95: (99.0, 101.0)},
            original_aic=100.0
        )
        rank = 1

        plot_namedtuple_bootstrap_histogram(mock_axes, result, rank, default_colors)

        # Should use default 0 for CV
        text_call = mock_axes.text.call_args
        text_content = text_call[0][2]
        assert 'CV: 0.0000' in text_content

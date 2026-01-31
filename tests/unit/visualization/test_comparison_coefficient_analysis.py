"""
Tests for src.visualization.comparison_coefficient_analysis module.

Tests coefficient analysis visualizations:
- Coefficient heatmap
- Sign consistency scatter
- Economic constraint validation
- Coefficient uncertainty quantification

Target coverage: 70%+
"""

from typing import Any, Dict
from unittest.mock import MagicMock, patch, PropertyMock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualization.comparison_coefficient_analysis import (
    plot_coefficient_heatmap,
    plot_coefficient_uncertainty,
    plot_economic_constraint_validation,
    plot_sign_consistency_scatter,
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
        'success': '#2ca02c',
        'light_gray': '#f0f0f0',
    }


@pytest.fixture
def sample_top_models():
    """Sample top models DataFrame with coefficients."""
    return pd.DataFrame({
        'features': ['feat_a+feat_b', 'feat_c+feat_d', 'feat_e'],
        'aic': [100.5, 105.2, 98.7],
        'r_squared': [0.85, 0.82, 0.87],
        'n_features': [2, 2, 1],
        'coefficients': [
            {'const': 1.0, 'prudential_rate': 0.5, 'competitor_avg': -0.3},
            {'const': 0.8, 'prudential_rate': 0.4, 'competitor_rate': -0.2},
            {'const': 1.2, 'prudential_rate': -0.1},
        ]
    })


@pytest.fixture
def sample_top_models_no_coefficients():
    """Top models without coefficients column."""
    return pd.DataFrame({
        'features': ['model_a', 'model_b'],
        'aic': [100.0, 102.0],
        'r_squared': [0.85, 0.83],
    })


@pytest.fixture
def sample_coefficient_stability():
    """Sample coefficient stability data."""
    return {
        'feat_a+feat_b': {
            'prudential_rate': {'sign_consistency': 0.95, 'mean': 0.5, 'cv': 0.1},
            'competitor_avg': {'sign_consistency': 0.90, 'mean': -0.3, 'cv': 0.15},
        },
        'feat_c+feat_d': {
            'prudential_rate': {'sign_consistency': 0.88, 'mean': 0.4, 'cv': 0.2},
            'competitor_rate': {'sign_consistency': 0.92, 'mean': -0.2, 'cv': 0.12},
        },
    }


# =============================================================================
# COEFFICIENT HEATMAP TESTS
# =============================================================================


class TestPlotCoefficientHeatmap:
    """Tests for plot_coefficient_heatmap function."""

    @patch('src.visualization.comparison_coefficient_analysis.plt')
    def test_basic_heatmap(self, mock_plt, mock_axes, sample_top_models, default_colors):
        """Test basic coefficient heatmap creation."""
        mock_colorbar = MagicMock()
        mock_plt.colorbar.return_value = mock_colorbar

        plot_coefficient_heatmap(mock_axes, sample_top_models, default_colors)

        # Verify imshow was called for heatmap
        mock_axes.imshow.assert_called_once()
        # Verify title was set
        mock_axes.set_title.assert_called_once()
        # Verify colorbar was created
        mock_plt.colorbar.assert_called_once()

    def test_heatmap_no_coefficients(
        self, mock_axes, sample_top_models_no_coefficients, default_colors
    ):
        """Test heatmap shows placeholder when no coefficients."""
        plot_coefficient_heatmap(mock_axes, sample_top_models_no_coefficients, default_colors)

        # Should show placeholder text
        mock_axes.text.assert_called_once()
        call_args = mock_axes.text.call_args
        text_content = call_args[0][2]
        assert 'Not Available' in text_content

    @patch('src.visualization.comparison_coefficient_analysis.plt')
    def test_heatmap_truncates_model_names(
        self, mock_plt, mock_axes, default_colors
    ):
        """Test that long model names are truncated."""
        mock_plt.colorbar.return_value = MagicMock()

        models = pd.DataFrame({
            'features': ['this_is_a_very_long_feature_name_that_exceeds_limit'],
            'coefficients': [{'feat': 0.5}],
        })

        plot_coefficient_heatmap(mock_axes, models, default_colors)

        # Verify ytick labels were set
        mock_axes.set_yticklabels.assert_called_once()
        labels = mock_axes.set_yticklabels.call_args[0][0]
        # Name should be truncated to 25 chars
        assert len(labels[0]) == 25

    @patch('src.visualization.comparison_coefficient_analysis.plt')
    def test_heatmap_empty_coefficients(self, mock_plt, mock_axes, default_colors):
        """Test heatmap with empty coefficients dict."""
        models = pd.DataFrame({
            'features': ['model'],
            'coefficients': [{}],
        })

        plot_coefficient_heatmap(mock_axes, models, default_colors)

        # Should return early without plotting
        mock_axes.imshow.assert_not_called()

    @patch('src.visualization.comparison_coefficient_analysis.plt')
    def test_heatmap_excludes_const(
        self, mock_plt, mock_axes, sample_top_models, default_colors
    ):
        """Test that const coefficient is excluded from heatmap."""
        mock_plt.colorbar.return_value = MagicMock()

        plot_coefficient_heatmap(mock_axes, sample_top_models, default_colors)

        # Check xticklabels don't include 'const'
        call_args = mock_axes.set_xticklabels.call_args
        labels = call_args[0][0]
        assert 'const' not in labels

    @patch('src.visualization.comparison_coefficient_analysis.plt')
    def test_heatmap_with_small_matrix_annotations(self, mock_plt, mock_axes, default_colors):
        """Test text annotations are added for small matrices."""
        mock_plt.colorbar.return_value = MagicMock()

        # Small matrix: 2 models, 2 features
        models = pd.DataFrame({
            'features': ['model_a', 'model_b'],
            'coefficients': [
                {'feat1': 0.5, 'feat2': -0.3},
                {'feat1': 0.4, 'feat2': -0.2},
            ],
        })

        plot_coefficient_heatmap(mock_axes, models, default_colors)

        # Should add text annotations for each cell (2x2 = 4 calls)
        assert mock_axes.text.call_count == 4


# =============================================================================
# SIGN CONSISTENCY SCATTER TESTS
# =============================================================================


class TestPlotSignConsistencyScatter:
    """Tests for plot_sign_consistency_scatter function."""

    def test_basic_scatter(
        self, mock_axes, sample_coefficient_stability, sample_top_models, default_colors
    ):
        """Test basic sign consistency scatter plot."""
        plot_sign_consistency_scatter(
            mock_axes, sample_coefficient_stability, sample_top_models, default_colors
        )

        # Verify scatter was created
        mock_axes.scatter.assert_called_once()
        # Verify labels
        mock_axes.set_xlabel.assert_called()
        mock_axes.set_ylabel.assert_called()
        mock_axes.set_title.assert_called()

    def test_scatter_no_stability_data(self, mock_axes, sample_top_models, default_colors):
        """Test scatter with empty stability data."""
        plot_sign_consistency_scatter(mock_axes, {}, sample_top_models, default_colors)

        # Should show placeholder
        mock_axes.text.assert_called_once()
        call_args = mock_axes.text.call_args
        text_content = call_args[0][2]
        assert 'Not Available' in text_content

    def test_scatter_threshold_lines(
        self, mock_axes, sample_coefficient_stability, sample_top_models, default_colors
    ):
        """Test threshold lines are added."""
        plot_sign_consistency_scatter(
            mock_axes, sample_coefficient_stability, sample_top_models, default_colors
        )

        # Should have two vertical threshold lines
        assert mock_axes.axvline.call_count == 2

    def test_scatter_legend(
        self, mock_axes, sample_coefficient_stability, sample_top_models, default_colors
    ):
        """Test legend is created."""
        plot_sign_consistency_scatter(
            mock_axes, sample_coefficient_stability, sample_top_models, default_colors
        )

        mock_axes.legend.assert_called_once()

    def test_scatter_with_few_features_annotates(
        self, mock_axes, sample_coefficient_stability, sample_top_models, default_colors
    ):
        """Test annotations are added when few features."""
        plot_sign_consistency_scatter(
            mock_axes, sample_coefficient_stability, sample_top_models, default_colors
        )

        # Should have annotations for <= 8 features
        assert mock_axes.annotate.call_count > 0


# =============================================================================
# ECONOMIC CONSTRAINT VALIDATION TESTS
# =============================================================================


class TestPlotEconomicConstraintValidation:
    """Tests for plot_economic_constraint_validation function."""

    def test_basic_constraint_plot(self, mock_axes, sample_top_models, default_colors):
        """Test basic economic constraint validation plot."""
        plot_economic_constraint_validation(mock_axes, sample_top_models, default_colors)

        # Verify horizontal bar chart
        mock_axes.barh.assert_called_once()
        # Verify labels
        mock_axes.set_xlabel.assert_called()
        mock_axes.set_title.assert_called()

    def test_constraint_plot_xlim(self, mock_axes, sample_top_models, default_colors):
        """Test x-axis limit is set to [0, 1]."""
        plot_economic_constraint_validation(mock_axes, sample_top_models, default_colors)

        mock_axes.set_xlim.assert_called_with(0, 1)

    def test_constraint_plot_color_coding(self, mock_axes, sample_top_models, default_colors):
        """Test color coding based on pass rate."""
        plot_economic_constraint_validation(mock_axes, sample_top_models, default_colors)

        # Check barh was called with colors
        call_args = mock_axes.barh.call_args
        colors_used = call_args[1]['color']
        # Should have different colors based on pass rates
        assert len(colors_used) == 3

    def test_constraint_plot_text_labels(self, mock_axes, sample_top_models, default_colors):
        """Test percentage labels are added to bars."""
        # Create mock bar
        mock_bar = MagicMock()
        mock_bar.get_y.return_value = 0
        mock_bar.get_height.return_value = 0.8
        mock_axes.barh.return_value = [mock_bar, mock_bar, mock_bar]

        plot_economic_constraint_validation(mock_axes, sample_top_models, default_colors)

        # Should add text for each bar
        assert mock_axes.text.call_count == 3

    def test_constraint_plot_empty_models(self, mock_axes, default_colors):
        """Test with empty models DataFrame."""
        empty_models = pd.DataFrame({
            'features': [],
            'coefficients': []
        })

        plot_economic_constraint_validation(mock_axes, empty_models, default_colors)

        # Should return early
        mock_axes.barh.assert_not_called()


# =============================================================================
# COEFFICIENT UNCERTAINTY TESTS
# =============================================================================


class TestPlotCoefficientUncertainty:
    """Tests for plot_coefficient_uncertainty function."""

    def test_basic_uncertainty_plot(
        self, mock_axes, sample_coefficient_stability, sample_top_models, default_colors
    ):
        """Test basic coefficient uncertainty plot."""
        plot_coefficient_uncertainty(
            mock_axes, sample_coefficient_stability, sample_top_models, default_colors
        )

        # Verify bar chart
        mock_axes.bar.assert_called()
        # Verify labels
        mock_axes.set_xlabel.assert_called()
        mock_axes.set_ylabel.assert_called()
        mock_axes.set_title.assert_called()

    def test_uncertainty_no_stability_data(self, mock_axes, sample_top_models, default_colors):
        """Test uncertainty with empty stability data."""
        plot_coefficient_uncertainty(mock_axes, {}, sample_top_models, default_colors)

        # Should show placeholder
        mock_axes.text.assert_called_once()
        call_args = mock_axes.text.call_args
        text_content = call_args[0][2]
        assert 'Not Available' in text_content

    def test_uncertainty_threshold_line(
        self, mock_axes, sample_coefficient_stability, sample_top_models, default_colors
    ):
        """Test threshold line is added."""
        plot_coefficient_uncertainty(
            mock_axes, sample_coefficient_stability, sample_top_models, default_colors
        )

        # Should have horizontal threshold line at 0.1
        mock_axes.axhline.assert_called_once_with(
            0.1, color=default_colors['success'], linestyle='--', alpha=0.7
        )

    def test_uncertainty_legend(
        self, mock_axes, sample_coefficient_stability, sample_top_models, default_colors
    ):
        """Test legend is created."""
        plot_coefficient_uncertainty(
            mock_axes, sample_coefficient_stability, sample_top_models, default_colors
        )

        mock_axes.legend.assert_called_once()

    def test_uncertainty_truncates_feature_names(
        self, mock_axes, sample_top_models, default_colors
    ):
        """Test long feature names are truncated."""
        stability = {
            'feat_a+feat_b': {
                'very_long_feature_name_exceeding_limit': {
                    'cv': 0.1, 'mean': 0.5, 'sign_consistency': 0.9
                }
            }
        }

        plot_coefficient_uncertainty(mock_axes, stability, sample_top_models, default_colors)

        # Check xticklabels include truncated names
        if mock_axes.set_xticklabels.called:
            call_args = mock_axes.set_xticklabels.call_args
            labels = call_args[0][0]
            # Truncated to 10 chars + '...'
            for label in labels:
                assert len(label) <= 13 or '...' in label


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases across coefficient analysis functions."""

    @patch('src.visualization.comparison_coefficient_analysis.plt')
    def test_heatmap_single_model(self, mock_plt, mock_axes, default_colors):
        """Test heatmap with single model."""
        mock_plt.colorbar.return_value = MagicMock()

        models = pd.DataFrame({
            'features': ['single_model'],
            'coefficients': [{'feat': 0.5}],
        })

        plot_coefficient_heatmap(mock_axes, models, default_colors)

        # Should still create heatmap
        mock_axes.imshow.assert_called_once()

    def test_scatter_empty_extracted_data(self, mock_axes, sample_top_models, default_colors):
        """Test scatter when no matching models in stability data."""
        # Stability data with non-matching model names
        stability = {
            'non_matching_model': {
                'feat': {'sign_consistency': 0.9, 'mean': 0.5, 'cv': 0.1}
            }
        }

        plot_sign_consistency_scatter(mock_axes, stability, sample_top_models, default_colors)

        # Should return early without scatter (or minimal plot)
        # The function returns early if features_analyzed is empty

    def test_uncertainty_empty_unique_features(self, mock_axes, sample_top_models, default_colors):
        """Test uncertainty when no unique features."""
        stability = {
            'feat_a+feat_b': {}  # No features
        }

        plot_coefficient_uncertainty(mock_axes, stability, sample_top_models, default_colors)

        # Should return early
        mock_axes.bar.assert_not_called()

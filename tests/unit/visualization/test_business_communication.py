"""
Unit tests for src/visualization/business_communication.py.

Tests validate the BusinessCommunicationPlots class and related functions
for generating business-focused stakeholder visualizations.

Target: 60%+ coverage for business_communication.py
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_aic_results():
    """Sample AIC results DataFrame for testing."""
    np.random.seed(42)
    n_models = 30
    return pd.DataFrame({
        'model_id': range(n_models),
        'aic': np.random.uniform(500, 700, n_models),
        'r_squared': np.random.uniform(0.3, 0.9, n_models),
        'n_features': np.random.randint(2, 8, n_models),
        'economically_valid': np.random.choice([True, False], n_models, p=[0.6, 0.4])
    })


@pytest.fixture
def sample_final_model():
    """Sample final model selection."""
    return {
        'model_id': 5,
        'selected_model': {
            'features': 'prudential_rate_t0, competitor_weighted_t2',
            'r_squared': 0.72,
            'n_features': 4
        }
    }


@pytest.fixture
def sample_analysis_results(sample_aic_results, sample_final_model):
    """Complete analysis results dictionary."""
    return {
        'aic_results': sample_aic_results,
        'final_model': sample_final_model,
        'bootstrap_results': {
            'block_bootstrap_results': [
                {'aic_stability_cv': 0.05},
                {'aic_stability_cv': 0.08},
                {'aic_stability_cv': 0.03}
            ]
        }
    }


@pytest.fixture
def empty_analysis_results():
    """Empty analysis results for edge case testing."""
    return {
        'aic_results': pd.DataFrame(),
        'final_model': None
    }


# =============================================================================
# CLASS INITIALIZATION TESTS
# =============================================================================


class TestBusinessCommunicationPlotsInit:
    """Tests for BusinessCommunicationPlots class initialization."""

    def test_init_default_colors(self):
        """Should initialize with default color scheme."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()

        assert 'primary' in plotter.colors
        assert 'secondary' in plotter.colors
        assert 'success' in plotter.colors
        assert 'warning' in plotter.colors
        assert 'danger' in plotter.colors

    def test_init_custom_colors(self):
        """Should accept custom company colors."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        custom_colors = {
            'primary': '#FF0000',
            'secondary': '#00FF00',
            'success': '#0000FF',
            'warning': '#FFFF00',
            'danger': '#FF00FF'
        }

        plotter = BusinessCommunicationPlots(company_colors=custom_colors)

        assert plotter.colors['primary'] == '#FF0000'
        assert plotter.colors['success'] == '#0000FF'

    def test_init_sets_plot_config(self):
        """Should initialize plot configuration."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()

        assert 'executive_figure_size' in plotter.plot_config
        assert 'standard_figure_size' in plotter.plot_config
        assert 'compact_figure_size' in plotter.plot_config

    def test_init_sets_business_terms(self):
        """Should initialize business terminology mapping."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()

        assert 'aic' in plotter.business_terms
        assert 'r_squared' in plotter.business_terms
        assert plotter.business_terms['aic'] == 'Model Quality Score'


# =============================================================================
# HELPER METHOD TESTS
# =============================================================================


class TestGetImpactAssessment:
    """Tests for _get_impact_assessment helper method."""

    def test_strong_impact_for_high_performance(self):
        """Should return STRONG for performance >= 60."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        level, color, description = plotter._get_impact_assessment(75)

        assert level == 'STRONG'
        assert color == plotter.colors['success']
        assert 'High predictive' in description

    def test_moderate_impact_for_medium_performance(self):
        """Should return MODERATE for performance 45-60."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        level, color, description = plotter._get_impact_assessment(52)

        assert level == 'MODERATE'
        assert color == plotter.colors['warning']

    def test_limited_impact_for_low_performance(self):
        """Should return LIMITED for performance < 45."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        level, color, description = plotter._get_impact_assessment(30)

        assert level == 'LIMITED'
        assert color == plotter.colors['danger']


class TestCategorizeFeatures:
    """Tests for _categorize_features helper method."""

    def test_categorizes_competitor_features(self):
        """Should categorize competitor features."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        categories = plotter._categorize_features('competitor_weighted_t2')

        assert categories['Competitive Intelligence'] == 1

    def test_categorizes_prudential_features(self):
        """Should categorize internal pricing features."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        categories = plotter._categorize_features('prudential_rate_t0')

        assert categories['Internal Pricing'] == 1

    def test_categorizes_economic_features(self):
        """Should categorize economic indicator features."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        categories = plotter._categorize_features('treasury_10y, econ_index')

        assert categories['Economic Indicators'] == 1

    def test_handles_multiple_categories(self):
        """Should handle features spanning multiple categories."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        features = 'competitor_weighted, prudential_rate, treasury_yield'
        categories = plotter._categorize_features(features)

        assert categories['Competitive Intelligence'] == 1
        assert categories['Internal Pricing'] == 1
        assert categories['Economic Indicators'] == 1


class TestGenerateRecommendations:
    """Tests for _generate_recommendations helper method."""

    def test_generates_recommendations_with_model(self, sample_final_model):
        """Should generate recommendations when model exists."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        recommendations = plotter._generate_recommendations(sample_final_model)

        assert len(recommendations) > 0
        assert any('RECOMMENDED' in r or 'CONDITIONAL' in r for r in recommendations)

    def test_generates_not_recommended_without_model(self):
        """Should generate NOT RECOMMENDED when no model."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        recommendations = plotter._generate_recommendations(None)

        assert any('NOT RECOMMENDED' in r for r in recommendations)

    def test_includes_competitor_advice_when_relevant(self, sample_final_model):
        """Should include competitor monitoring advice."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        recommendations = plotter._generate_recommendations(sample_final_model)

        # Features include 'competitor_weighted_t2'
        assert any('competitive landscape' in r.lower() for r in recommendations)


class TestComputeRiskAssessment:
    """Tests for _compute_risk_assessment helper method."""

    def test_returns_three_lists(self, sample_analysis_results, sample_final_model):
        """Should return risk factors, levels, and colors."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        factors, levels, colors = plotter._compute_risk_assessment(
            sample_analysis_results, sample_final_model
        )

        assert isinstance(factors, list)
        assert isinstance(levels, list)
        assert isinstance(colors, list)
        assert len(factors) == len(levels) == len(colors)

    def test_includes_economic_logic_factor(self, sample_analysis_results, sample_final_model):
        """Should include economic logic in risk factors."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        factors, levels, colors = plotter._compute_risk_assessment(
            sample_analysis_results, sample_final_model
        )

        assert 'Economic Logic' in factors

    def test_includes_feature_dependency_factor(self, sample_analysis_results, sample_final_model):
        """Should include feature dependency in risk factors."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        factors, levels, colors = plotter._compute_risk_assessment(
            sample_analysis_results, sample_final_model
        )

        assert 'Feature Dependency' in factors


class TestComputeReadinessScores:
    """Tests for _compute_readiness_scores helper method."""

    def test_returns_four_scores(self, sample_final_model):
        """Should return four readiness scores."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        scores = plotter._compute_readiness_scores(sample_final_model)

        assert len(scores) == 4
        assert all(isinstance(s, (int, float)) for s in scores)

    def test_returns_lower_scores_without_model(self):
        """Should return lower scores when no model."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        scores_no_model = plotter._compute_readiness_scores(None)
        scores_with_model = plotter._compute_readiness_scores({
            'selected_model': {'r_squared': 0.8, 'n_features': 3}
        })

        # Scores without model should generally be lower
        assert sum(scores_no_model) < sum(scores_with_model)


class TestComputeResponsivenessScores:
    """Tests for _compute_responsiveness_scores helper method."""

    def test_high_score_for_competitor_features(self):
        """Should return high score when competitor features present."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        scores = plotter._compute_responsiveness_scores('competitor_weighted')

        assert scores[0] == 85  # Competitor pricing score

    def test_low_score_without_competitor_features(self):
        """Should return low score without competitor features."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        scores = plotter._compute_responsiveness_scores('treasury_yield')

        assert scores[0] == 30  # Low competitor score


class TestComputePositioningScores:
    """Tests for _compute_positioning_scores helper method."""

    def test_high_score_for_combined_features(self):
        """Should return high score when competitor + prudential present."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        scores = plotter._compute_positioning_scores('competitor, prudential')

        assert scores[1] == 80  # Competitive differentiation


class TestGenerateCompetitiveRecommendations:
    """Tests for _generate_competitive_recommendations helper method."""

    def test_pass_for_competitor_features(self):
        """Should show PASS for competitor features."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        recommendations = plotter._generate_competitive_recommendations('competitor_weighted')

        assert any('[PASS]' in r for r in recommendations)

    def test_warn_without_competitor_features(self):
        """Should show WARN without competitor features."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        recommendations = plotter._generate_competitive_recommendations('treasury_yield')

        assert any('[WARN]' in r for r in recommendations)


# =============================================================================
# PLOT METHOD TESTS
# =============================================================================


class TestPlotBusinessOutcomes:
    """Tests for _plot_business_outcomes method."""

    def test_turns_axis_off(self, sample_final_model):
        """Should turn off axis."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        mock_ax = MagicMock()

        plotter._plot_business_outcomes(mock_ax, sample_final_model)

        mock_ax.axis.assert_called_with('off')

    def test_handles_no_model(self):
        """Should handle missing model gracefully."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        mock_ax = MagicMock()

        # Should not raise
        plotter._plot_business_outcomes(mock_ax, None)
        mock_ax.axis.assert_called_with('off')


class TestPlotSelectionFunnel:
    """Tests for _plot_selection_funnel method."""

    def test_creates_horizontal_bars(self, sample_aic_results, sample_final_model):
        """Should create horizontal bar chart."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        mock_ax = MagicMock()

        plotter._plot_selection_funnel(mock_ax, sample_aic_results, sample_final_model)

        mock_ax.barh.assert_called()

    def test_handles_empty_results(self):
        """Should handle empty AIC results."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        mock_ax = MagicMock()

        # Should not raise
        plotter._plot_selection_funnel(mock_ax, pd.DataFrame(), None)


class TestPlotRiskAssessment:
    """Tests for _plot_risk_assessment method."""

    def test_creates_horizontal_bars(self):
        """Should create horizontal bars for risk factors."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        mock_ax = MagicMock()

        risk_factors = ['Model Stability', 'Economic Logic']
        risk_levels = ['LOW RISK', 'HIGH RISK']
        risk_colors = [plotter.colors['success'], plotter.colors['danger']]

        plotter._plot_risk_assessment(mock_ax, risk_factors, risk_levels, risk_colors)

        mock_ax.barh.assert_called_once()

    def test_handles_empty_factors(self):
        """Should handle empty risk factors."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        mock_ax = MagicMock()

        # Should not raise
        plotter._plot_risk_assessment(mock_ax, [], [], [])


class TestPlotRecommendations:
    """Tests for _plot_recommendations method."""

    def test_turns_axis_off_and_adds_text(self):
        """Should turn axis off and add recommendation text."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        mock_ax = MagicMock()
        recommendations = ['Recommendation 1', 'Recommendation 2']

        plotter._plot_recommendations(mock_ax, recommendations)

        mock_ax.axis.assert_called_with('off')
        mock_ax.text.assert_called_once()


# =============================================================================
# DASHBOARD CREATION TESTS
# =============================================================================


class TestCreateExecutiveSummaryDashboard:
    """Tests for create_executive_summary_dashboard method."""

    @patch('src.visualization.business_communication.plt')
    def test_returns_figure(self, mock_plt, sample_analysis_results):
        """Should return matplotlib Figure."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_fig.add_gridspec.return_value = MagicMock()
        mock_plt.figure.return_value = mock_fig

        plotter = BusinessCommunicationPlots()
        result = plotter.create_executive_summary_dashboard(sample_analysis_results)

        assert result == mock_fig

    @patch('src.visualization.business_communication.plt')
    def test_saves_to_path_when_provided(self, mock_plt, sample_analysis_results, tmp_path):
        """Should save figure when save_path provided."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_fig.add_gridspec.return_value = MagicMock()
        mock_plt.figure.return_value = mock_fig

        plotter = BusinessCommunicationPlots()
        save_path = tmp_path / "test_dashboard.png"
        plotter.create_executive_summary_dashboard(sample_analysis_results, save_path=save_path)

        mock_fig.savefig.assert_called_once()


class TestCreateModelPerformanceComparison:
    """Tests for create_model_performance_comparison method."""

    @patch('src.visualization.business_communication.plt')
    def test_returns_figure(self, mock_plt, sample_aic_results, sample_final_model):
        """Should return matplotlib Figure."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        mock_fig = MagicMock()
        # Create mock axes with proper return values for pie chart
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_ax3 = MagicMock()
        mock_ax4 = MagicMock()

        # Mock pie return: (wedges, texts, autotexts)
        mock_text = MagicMock()
        mock_ax1.pie.return_value = ([MagicMock()], [mock_text], [mock_text])

        mock_axes = [[mock_ax1, mock_ax2], [mock_ax3, mock_ax4]]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_plt.colorbar.return_value = MagicMock()

        plotter = BusinessCommunicationPlots()
        result = plotter.create_model_performance_comparison(
            sample_aic_results, sample_final_model
        )

        assert result == mock_fig


class TestCreateCompetitiveAnalysisVisualization:
    """Tests for create_competitive_analysis_visualization method."""

    @patch('src.visualization.business_communication.plt')
    def test_returns_figure(self, mock_plt, sample_analysis_results):
        """Should return matplotlib Figure."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        mock_fig = MagicMock()
        mock_axes = [[MagicMock(), MagicMock()], [MagicMock(), MagicMock()]]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_plt.subplot.return_value = MagicMock()

        plotter = BusinessCommunicationPlots()
        result = plotter.create_competitive_analysis_visualization(sample_analysis_results)

        assert result == mock_fig

    @patch('src.visualization.business_communication.plt')
    def test_handles_no_model(self, mock_plt, empty_analysis_results):
        """Should handle missing model gracefully."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_axes = [[mock_ax, mock_ax], [mock_ax, mock_ax]]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        plotter = BusinessCommunicationPlots()
        result = plotter.create_competitive_analysis_visualization(empty_analysis_results)

        # Should still return figure
        assert result == mock_fig


# =============================================================================
# STANDALONE FUNCTION TESTS
# =============================================================================


class TestCreateBusinessCommunicationReport:
    """Tests for create_business_communication_report standalone function."""

    @patch('src.visualization.business_communication.plt')
    def test_returns_dict_of_paths(self, mock_plt, sample_analysis_results, tmp_path):
        """Should return dictionary of report file paths."""
        from src.visualization.business_communication import create_business_communication_report

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_axes = [[mock_ax, mock_ax], [mock_ax, mock_ax]]
        mock_fig.add_subplot.return_value = mock_ax
        mock_fig.add_gridspec.return_value = MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_plt.subplot.return_value = mock_ax
        mock_plt.colorbar.return_value = MagicMock()

        result = create_business_communication_report(
            sample_analysis_results,
            tmp_path
        )

        assert isinstance(result, dict)
        assert 'executive_summary' in result

    @patch('src.visualization.business_communication.plt')
    def test_creates_output_directory(self, mock_plt, sample_analysis_results, tmp_path):
        """Should create output directory if not exists."""
        from src.visualization.business_communication import create_business_communication_report

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_axes = [[mock_ax, mock_ax], [mock_ax, mock_ax]]
        mock_fig.add_subplot.return_value = mock_ax
        mock_fig.add_gridspec.return_value = MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_plt.subplot.return_value = mock_ax
        mock_plt.colorbar.return_value = MagicMock()

        output_dir = tmp_path / "new_output_dir"
        create_business_communication_report(sample_analysis_results, output_dir)

        assert output_dir.exists()

    @patch('src.visualization.business_communication.plt')
    def test_uses_custom_file_prefix(self, mock_plt, sample_analysis_results, tmp_path):
        """Should use custom file prefix in output filenames."""
        from src.visualization.business_communication import create_business_communication_report

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_axes = [[mock_ax, mock_ax], [mock_ax, mock_ax]]
        mock_fig.add_subplot.return_value = mock_ax
        mock_fig.add_gridspec.return_value = MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_plt.subplot.return_value = mock_ax
        mock_plt.colorbar.return_value = MagicMock()

        result = create_business_communication_report(
            sample_analysis_results,
            tmp_path,
            file_prefix='custom_prefix'
        )

        # Check that paths use custom prefix
        if 'executive_summary' in result:
            assert 'custom_prefix' in str(result['executive_summary'])


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_features_string(self):
        """Should handle empty features string."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        categories = plotter._categorize_features('')

        # All categories should be 0
        assert all(v == 0 for v in categories.values())

    def test_none_model_in_risk_assessment(self, sample_analysis_results):
        """Should handle None model in risk assessment."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        factors, levels, colors = plotter._compute_risk_assessment(
            sample_analysis_results, None
        )

        # Should still return lists
        assert isinstance(factors, list)
        assert 'Economic Logic' in factors

    def test_missing_bootstrap_results(self, sample_final_model):
        """Should handle missing bootstrap results."""
        from src.visualization.business_communication import BusinessCommunicationPlots

        plotter = BusinessCommunicationPlots()
        analysis_results = {
            'aic_results': pd.DataFrame({'aic': [500, 600]}),
            'final_model': sample_final_model
            # No bootstrap_results
        }

        factors, levels, colors = plotter._compute_risk_assessment(
            analysis_results, sample_final_model
        )

        # Should still work without bootstrap
        assert isinstance(factors, list)
        assert len(factors) > 0

    @patch('src.visualization.business_communication.plt')
    def test_report_handles_errors_gracefully(self, mock_plt, sample_analysis_results, tmp_path):
        """Should handle errors in report generation."""
        from src.visualization.business_communication import create_business_communication_report

        mock_plt.figure.side_effect = Exception("Plot error")

        result = create_business_communication_report(
            sample_analysis_results,
            tmp_path
        )

        # Should return dict even on error
        assert isinstance(result, dict)

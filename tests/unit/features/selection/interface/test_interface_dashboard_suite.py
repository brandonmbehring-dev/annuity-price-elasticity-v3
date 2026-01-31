"""
Tests for Interface Dashboard Module Suite.

Tests cover:
- interface_dashboard_validation.py: Input validation and config extraction
- interface_dashboard_scoring.py: Win rate + IR scoring system
- interface_dashboard_business.py: Business recommendations
- interface_dashboard_viz.py: Dashboard visualizations
- interface_dashboard.py: Main orchestrator

Design Principles:
- Pure function tests for scoring and business logic
- Mock matplotlib for visualization tests
- Validate business rules and grade thresholds

Author: Claude Code
Date: 2026-01-30
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from typing import Dict, List, Any

# Import validation functions
from src.features.selection.interface.interface_dashboard_validation import (
    validate_dashboard_inputs,
    extract_dashboard_config,
    process_advanced_stability_results,
)

# Import scoring functions
from src.features.selection.interface.interface_dashboard_scoring import (
    extract_base_model_info,
    normalize_ir_to_score,
    compute_stability_grade,
    build_comprehensive_result,
    create_comprehensive_scoring_system,
)

# Import business functions
from src.features.selection.interface.interface_dashboard_business import (
    get_empty_recommendation,
    determine_recommendation_when_same,
    determine_recommendation_when_different,
    build_alternative_models_list,
    build_grade_distribution,
    build_best_model_summary,
    generate_final_recommendations,
)

# Import visualization functions
from src.features.selection.interface.interface_dashboard_viz import (
    plot_winrate_vs_ir_scatter,
    plot_composite_score_distribution,
    plot_winrate_rankings,
    plot_ir_rankings,
    plot_aic_vs_stability,
    plot_recommendation_summary,
    create_comprehensive_dashboard_visualizations,
    create_dashboard_visualizations_safe,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_win_rate_result():
    """Sample win rate result dictionary."""
    return {
        'model': 'Model_1',
        'features': ['own_rate_t1', 'competitor_weighted_t2'],
        'original_aic': 1234.5,
        'median_bootstrap_aic': 1240.2,
        'win_rate_pct': 85.5,
    }


@pytest.fixture
def sample_ir_result():
    """Sample information ratio result dictionary."""
    return {
        'model_name': 'Model_1',
        'information_ratio': 1.25,
        'success_rate': 72.0,
    }


@pytest.fixture
def sample_comprehensive_scores():
    """Sample comprehensive scores for testing."""
    return [
        {
            'model_name': 'Model_1',
            'features': 'own_rate_t1 + comp_t2',
            'original_aic': 1200,
            'win_rate_score': 90.0,
            'ir_score': 81.25,
            'composite_score': 85.6,
            'stability_grade': 'A+ (Excellent)',
            'in_sample_win_rate': 88.0,
            'out_sample_win_rate': 86.0,
        },
        {
            'model_name': 'Model_2',
            'features': 'own_rate_t1',
            'original_aic': 1180,  # Best AIC
            'win_rate_score': 75.0,
            'ir_score': 62.5,
            'composite_score': 68.8,
            'stability_grade': 'B+ (Good)',
            'in_sample_win_rate': 72.0,
            'out_sample_win_rate': 70.0,
        },
        {
            'model_name': 'Model_3',
            'features': 'own_rate_t1 + treasury_t1',
            'original_aic': 1220,
            'win_rate_score': 70.0,
            'ir_score': 56.25,
            'composite_score': 63.1,
            'stability_grade': 'B+ (Good)',
            'in_sample_win_rate': 68.0,
            'out_sample_win_rate': 65.0,
        },
        {
            'model_name': 'Model_4',
            'features': 'comp_t2',
            'original_aic': 1250,
            'win_rate_score': 55.0,
            'ir_score': 43.75,
            'composite_score': 49.4,
            'stability_grade': 'C (Below Average)',
            'in_sample_win_rate': 52.0,
            'out_sample_win_rate': 50.0,
        },
    ]


@pytest.fixture
def mock_axes():
    """Mock matplotlib axes for visualization tests."""
    ax = MagicMock()
    ax.scatter.return_value = MagicMock()
    ax.hist.return_value = (MagicMock(), MagicMock(), MagicMock())
    ax.barh.return_value = MagicMock()
    return ax


# =============================================================================
# Tests for interface_dashboard_validation.py
# =============================================================================


class TestValidateDashboardInputs:
    """Tests for validate_dashboard_inputs."""

    def test_valid_inputs_pass(self):
        """Valid inputs do not raise."""
        bootstrap_results = [MagicMock(), MagicMock()]
        config = {'models_to_analyze': 10}

        # Should not raise
        validate_dashboard_inputs(bootstrap_results, config)

    def test_empty_bootstrap_results_raises(self):
        """Empty bootstrap results raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_dashboard_inputs([], {'key': 'value'})

        assert 'No bootstrap results' in str(exc_info.value)

    def test_non_dict_config_raises(self):
        """Non-dictionary config raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_dashboard_inputs([MagicMock()], "not a dict")

        assert 'must be a dictionary' in str(exc_info.value)


class TestExtractDashboardConfig:
    """Tests for extract_dashboard_config."""

    def test_extracts_default_values(self, capsys):
        """Uses defaults when config is empty."""
        config = {}
        bootstrap_results = [MagicMock()] * 20

        n_models, create_viz, weights = extract_dashboard_config(config, bootstrap_results)

        assert n_models == 15  # Default
        assert create_viz is True  # Default
        assert weights['win_rate_weight'] == 0.5
        assert weights['information_ratio_weight'] == 0.5

    def test_extracts_custom_values(self, capsys):
        """Uses provided config values."""
        config = {
            'models_to_analyze': 10,
            'create_visualizations': False,
            'integration_weights': {
                'win_rate_weight': 0.6,
                'information_ratio_weight': 0.4
            }
        }
        bootstrap_results = [MagicMock()] * 20

        n_models, create_viz, weights = extract_dashboard_config(config, bootstrap_results)

        assert n_models == 10
        assert create_viz is False
        assert weights['win_rate_weight'] == 0.6

    def test_caps_models_to_bootstrap_count(self, capsys):
        """Caps models_to_analyze at bootstrap_results length."""
        config = {'models_to_analyze': 100}
        bootstrap_results = [MagicMock()] * 5  # Only 5 results

        n_models, _, _ = extract_dashboard_config(config, bootstrap_results)

        assert n_models == 5  # Capped to bootstrap count


class TestProcessAdvancedStabilityResults:
    """Tests for process_advanced_stability_results."""

    def test_processes_valid_results(self, capsys):
        """Processes results with both win_rate and IR."""
        advanced_results = {
            'win_rate_results': [{'model': 'A'}],
            'information_ratio_results': [{'model_name': 'A'}]
        }
        results = {}

        process_advanced_stability_results(advanced_results, results)

        assert 'win_rate_results' in results
        assert 'information_ratio_results' in results

    def test_raises_when_win_rate_missing(self):
        """Raises RuntimeError when win_rate_results missing."""
        advanced_results = {
            'information_ratio_results': [{'model_name': 'A'}]
        }

        with pytest.raises(RuntimeError) as exc_info:
            process_advanced_stability_results(advanced_results, {})

        assert 'Win Rate Analysis' in str(exc_info.value)

    def test_raises_when_ir_missing(self, capsys):
        """Raises RuntimeError when information_ratio_results missing."""
        advanced_results = {
            'win_rate_results': [{'model': 'A'}]
        }

        with pytest.raises(RuntimeError) as exc_info:
            process_advanced_stability_results(advanced_results, {})

        assert 'Information Ratio Analysis' in str(exc_info.value)


# =============================================================================
# Tests for interface_dashboard_scoring.py
# =============================================================================


class TestExtractBaseModelInfo:
    """Tests for extract_base_model_info."""

    def test_extracts_all_fields(self, sample_win_rate_result):
        """Extracts model_name, features, AICs."""
        info = extract_base_model_info(sample_win_rate_result)

        assert info['model_name'] == 'Model_1'
        assert info['features'] == ['own_rate_t1', 'competitor_weighted_t2']
        assert info['original_aic'] == 1234.5
        assert info['median_bootstrap_aic'] == 1240.2


class TestNormalizeIrToScore:
    """Tests for normalize_ir_to_score."""

    @pytest.mark.parametrize("ir_value,expected_score", [
        (-2, 0),      # Minimum clamp
        (0, 50),      # Middle
        (2, 100),     # Maximum clamp
        (1, 75),      # Above zero
        (-1, 25),     # Below zero
        (-3, 0),      # Below minimum (clamped)
        (3, 100),     # Above maximum (clamped)
    ])
    def test_normalization_mapping(self, ir_value, expected_score):
        """IR values map correctly to [0, 100] scale."""
        score = normalize_ir_to_score(ir_value)
        assert score == expected_score

    def test_score_always_in_range(self):
        """Score is always between 0 and 100."""
        for ir in np.linspace(-5, 5, 100):
            score = normalize_ir_to_score(ir)
            assert 0 <= score <= 100


class TestComputeStabilityGrade:
    """Tests for compute_stability_grade."""

    @pytest.mark.parametrize("score,expected_grade", [
        (95, "A+ (Excellent)"),
        (80, "A+ (Excellent)"),
        (79, "A (Very Good)"),
        (70, "A (Very Good)"),
        (69, "B+ (Good)"),
        (60, "B+ (Good)"),
        (59, "B (Average)"),
        (50, "B (Average)"),
        (49, "C (Below Average)"),
        (0, "C (Below Average)"),
    ])
    def test_grade_thresholds(self, score, expected_grade):
        """Scores map to correct letter grades."""
        grade = compute_stability_grade(score)
        assert grade == expected_grade


class TestBuildComprehensiveResult:
    """Tests for build_comprehensive_result."""

    def test_builds_complete_result(self):
        """Builds result with all required fields."""
        base_info = {'model_name': 'Test', 'features': ['f1'], 'original_aic': 1000}
        ir_data = {'success_rate': 70.0}

        result = build_comprehensive_result(
            base_info=base_info,
            win_rate_score=80.0,
            ir_score=75.0,
            ir_value=1.0,
            composite_score=77.5,
            stability_grade="A (Very Good)",
            ir_data=ir_data
        )

        assert result['model_name'] == 'Test'
        assert result['win_rate_score'] == 80.0
        assert result['ir_score'] == 75.0
        assert result['composite_score'] == 77.5
        assert result['stability_grade'] == "A (Very Good)"
        assert result['individual_metrics']['success_rate'] == 70.0


class TestCreateComprehensiveScoringSystem:
    """Tests for create_comprehensive_scoring_system."""

    def test_creates_sorted_scores(self, sample_win_rate_result, sample_ir_result):
        """Creates scores sorted by composite_score (descending)."""
        win_rate_results = [sample_win_rate_result]
        ir_results = [sample_ir_result]
        weights = {'win_rate_weight': 0.5, 'information_ratio_weight': 0.5}

        scores = create_comprehensive_scoring_system(win_rate_results, ir_results, weights)

        assert len(scores) == 1
        assert 'composite_score' in scores[0]
        assert 'stability_grade' in scores[0]

    def test_handles_missing_ir_data(self, sample_win_rate_result):
        """Handles models with no IR data (defaults to 0)."""
        win_rate_results = [sample_win_rate_result]
        ir_results = []  # No IR data
        weights = {'win_rate_weight': 0.5, 'information_ratio_weight': 0.5}

        scores = create_comprehensive_scoring_system(win_rate_results, ir_results, weights)

        assert len(scores) == 1
        # IR defaults to 0, normalized to 50
        assert scores[0]['ir_score'] == 50.0

    def test_applies_weights_correctly(self, sample_win_rate_result, sample_ir_result):
        """Weights are correctly applied to composite score."""
        win_rate_results = [sample_win_rate_result]
        ir_results = [sample_ir_result]
        weights = {'win_rate_weight': 0.7, 'information_ratio_weight': 0.3}

        scores = create_comprehensive_scoring_system(win_rate_results, ir_results, weights)

        # win_rate_pct = 85.5, ir_score = (1.25 + 2) * 25 = 81.25
        expected_composite = 0.7 * 85.5 + 0.3 * 81.25
        assert abs(scores[0]['composite_score'] - expected_composite) < 0.01


# =============================================================================
# Tests for interface_dashboard_business.py
# =============================================================================


class TestGetEmptyRecommendation:
    """Tests for get_empty_recommendation."""

    def test_returns_empty_structure(self):
        """Returns recommendation with no-models message."""
        rec = get_empty_recommendation()

        assert 'No models available' in rec['primary_recommendation']
        assert rec['confidence_level'] == 'None'
        assert rec['alternative_models'] == []


class TestDetermineRecommendationWhenSame:
    """Tests for determine_recommendation_when_same."""

    def test_returns_optimal_balance_recommendation(self):
        """Returns Optimal Balance recommendation."""
        best_overall = {'model_name': 'Model_A'}

        primary, confidence, rationale = determine_recommendation_when_same(best_overall)

        assert 'Model_A' in primary
        assert 'Optimal Balance' in primary
        assert confidence == 'Very High'


class TestDetermineRecommendationWhenDifferent:
    """Tests for determine_recommendation_when_different."""

    def test_stability_leader_when_large_stability_advantage(self):
        """Recommends stability leader when stability advantage > 10 and AIC cost < 5."""
        best_overall = {
            'model_name': 'Stable_Model',
            'composite_score': 85,
            'original_aic': 1203
        }
        best_aic = {
            'model_name': 'AIC_Model',
            'composite_score': 70,
            'original_aic': 1200
        }

        primary, confidence, rationale = determine_recommendation_when_different(
            best_overall, best_aic
        )

        assert 'Stability Leader' in primary
        assert 'Stable_Model' in primary
        assert confidence == 'High'

    def test_aic_leader_when_large_aic_advantage(self):
        """Recommends AIC leader when AIC cost > 10."""
        best_overall = {
            'model_name': 'Stable_Model',
            'composite_score': 85,
            'original_aic': 1220  # 20 points worse
        }
        best_aic = {
            'model_name': 'AIC_Model',
            'composite_score': 75,
            'original_aic': 1200
        }

        primary, confidence, rationale = determine_recommendation_when_different(
            best_overall, best_aic
        )

        assert 'AIC Leader' in primary
        assert 'AIC_Model' in primary
        assert confidence == 'Moderate'

    def test_balanced_choice_when_close(self):
        """Recommends balanced choice when metrics are close."""
        best_overall = {
            'model_name': 'Overall_Model',
            'composite_score': 80,
            'original_aic': 1205
        }
        best_aic = {
            'model_name': 'AIC_Model',
            'composite_score': 75,
            'original_aic': 1200
        }

        primary, confidence, rationale = determine_recommendation_when_different(
            best_overall, best_aic
        )

        assert 'Balanced Choice' in primary


class TestBuildAlternativeModelsList:
    """Tests for build_alternative_models_list."""

    def test_returns_top_alternatives_above_threshold(self, sample_comprehensive_scores):
        """Returns models 2-4 with score >= 60."""
        alternatives = build_alternative_models_list(sample_comprehensive_scores)

        # Model_2 (68.8) and Model_3 (63.1) qualify, Model_4 (49.4) doesn't
        assert len(alternatives) == 2
        assert any('Model_2' in alt for alt in alternatives)
        assert any('Model_3' in alt for alt in alternatives)

    def test_returns_empty_when_no_good_alternatives(self):
        """Returns empty list when no good alternatives."""
        scores = [
            {'model_name': 'Best', 'composite_score': 90},
            {'model_name': 'Bad1', 'composite_score': 40},
            {'model_name': 'Bad2', 'composite_score': 30},
        ]

        alternatives = build_alternative_models_list(scores)

        assert alternatives == []


class TestBuildGradeDistribution:
    """Tests for build_grade_distribution."""

    def test_counts_grades_correctly(self, sample_comprehensive_scores):
        """Counts grades in correct buckets."""
        distribution = build_grade_distribution(sample_comprehensive_scores)

        # Model_1: 85.6 -> A+, Model_2: 68.8 -> B+, Model_3: 63.1 -> B+, Model_4: 49.4 -> C
        assert distribution['A+'] == 1
        assert distribution['B+'] == 2
        assert distribution['C'] == 1


class TestBuildBestModelSummary:
    """Tests for build_best_model_summary."""

    def test_extracts_summary_fields(self, sample_comprehensive_scores):
        """Extracts all required summary fields."""
        summary = build_best_model_summary(sample_comprehensive_scores[0])

        assert summary['model_name'] == 'Model_1'
        assert summary['composite_score'] == 85.6
        assert summary['win_rate_score'] == 90.0
        assert 'model_features' in summary


class TestGenerateFinalRecommendations:
    """Tests for generate_final_recommendations."""

    def test_returns_empty_when_no_scores(self):
        """Returns empty recommendation when no scores."""
        rec = generate_final_recommendations([], [])

        assert 'No models available' in rec['primary_recommendation']

    def test_generates_complete_recommendation(self, sample_comprehensive_scores):
        """Generates recommendation with all required fields."""
        rec = generate_final_recommendations(sample_comprehensive_scores, [])

        assert 'primary_recommendation' in rec
        assert 'confidence_level' in rec
        assert 'rationale' in rec
        assert 'alternative_models' in rec
        assert 'performance_summary' in rec

    def test_recommends_optimal_when_best_equals_aic(self):
        """Recommends optimal balance when best overall equals best AIC."""
        scores = [
            {
                'model_name': 'Best',
                'composite_score': 90,
                'original_aic': 1000,  # Best AIC
                'features': 'f1',
                'stability_grade': 'A+',
                'win_rate_score': 90,
                'ir_score': 90,
            },
            {
                'model_name': 'Other',
                'composite_score': 70,
                'original_aic': 1100,
                'features': 'f2',
                'stability_grade': 'A',
                'win_rate_score': 70,
                'ir_score': 70,
            },
        ]

        rec = generate_final_recommendations(scores, [])

        assert 'Optimal Balance' in rec['primary_recommendation']
        assert rec['confidence_level'] == 'Very High'


# =============================================================================
# Tests for interface_dashboard_viz.py
# =============================================================================


class TestPlotWinrateVsIrScatter:
    """Tests for plot_winrate_vs_ir_scatter."""

    def test_creates_scatter_plot(self, mock_axes, sample_comprehensive_scores):
        """Creates scatter plot with correct data."""
        with patch('src.features.selection.interface.interface_dashboard_viz.plt') as mock_plt:
            mock_plt.colorbar.return_value = MagicMock()

            composite_vals = plot_winrate_vs_ir_scatter(mock_axes, sample_comprehensive_scores)

            mock_axes.scatter.assert_called_once()
            mock_axes.set_xlabel.assert_called()
            mock_axes.set_ylabel.assert_called()
            assert len(composite_vals) == len(sample_comprehensive_scores)


class TestPlotCompositeScoreDistribution:
    """Tests for plot_composite_score_distribution."""

    def test_creates_histogram(self, mock_axes):
        """Creates histogram with mean line."""
        composite_vals = [80, 75, 70, 65, 60]

        plot_composite_score_distribution(mock_axes, composite_vals)

        mock_axes.hist.assert_called_once()
        mock_axes.axvline.assert_called_once()
        mock_axes.legend.assert_called_once()


class TestPlotWinrateRankings:
    """Tests for plot_winrate_rankings."""

    def test_creates_horizontal_bar_chart(self, mock_axes, sample_comprehensive_scores):
        """Creates horizontal bar chart for top 10."""
        with patch('src.features.selection.interface.interface_dashboard_viz.plt') as mock_plt:
            mock_plt.cm.viridis.return_value = np.array([[0, 0, 0, 1]] * 4)

            plot_winrate_rankings(mock_axes, sample_comprehensive_scores)

            mock_axes.barh.assert_called_once()
            mock_axes.set_yticks.assert_called()


class TestPlotIrRankings:
    """Tests for plot_ir_rankings."""

    def test_creates_ir_ranking_chart(self, mock_axes, sample_comprehensive_scores):
        """Creates IR ranking horizontal bar chart."""
        with patch('src.features.selection.interface.interface_dashboard_viz.plt') as mock_plt:
            mock_plt.cm.plasma.return_value = np.array([[0, 0, 0, 1]] * 4)

            plot_ir_rankings(mock_axes, sample_comprehensive_scores)

            mock_axes.barh.assert_called_once()


class TestPlotAicVsStability:
    """Tests for plot_aic_vs_stability."""

    def test_creates_aic_stability_scatter(self, mock_axes, sample_comprehensive_scores):
        """Creates AIC vs stability scatter plot."""
        composite_vals = [85.6, 68.8, 63.1, 49.4]

        with patch('src.features.selection.interface.interface_dashboard_viz.plt') as mock_plt:
            mock_plt.colorbar.return_value = MagicMock()

            plot_aic_vs_stability(mock_axes, sample_comprehensive_scores, composite_vals)

            mock_axes.scatter.assert_called_once()
            mock_plt.colorbar.assert_called_once()


class TestPlotRecommendationSummary:
    """Tests for plot_recommendation_summary."""

    def test_creates_text_summary(self, mock_axes, sample_comprehensive_scores):
        """Creates text summary box."""
        plot_recommendation_summary(mock_axes, sample_comprehensive_scores)

        mock_axes.axis.assert_called_with('off')
        mock_axes.text.assert_called_once()


class TestCreateComprehensiveDashboardVisualizations:
    """Tests for create_comprehensive_dashboard_visualizations."""

    def test_creates_dashboard_figure(self, sample_comprehensive_scores):
        """Creates comprehensive dashboard with all plots."""
        with patch('src.features.selection.interface.interface_dashboard_viz.plt') as mock_plt:
            mock_fig = MagicMock()
            mock_plt.figure.return_value = mock_fig
            mock_gs = MagicMock()
            mock_fig.add_gridspec.return_value = mock_gs
            mock_ax = MagicMock()
            mock_ax.scatter.return_value = MagicMock()
            mock_ax.hist.return_value = (MagicMock(), MagicMock(), MagicMock())
            mock_ax.barh.return_value = MagicMock()
            mock_fig.add_subplot.return_value = mock_ax
            mock_plt.colorbar.return_value = MagicMock()
            mock_plt.cm.viridis.return_value = np.array([[0, 0, 0, 1]] * 10)
            mock_plt.cm.plasma.return_value = np.array([[0, 0, 0, 1]] * 10)

            win_rate = []
            ir_results = []
            config = {'fig_width': 16, 'fig_height': 12}

            visualizations = create_comprehensive_dashboard_visualizations(
                win_rate, ir_results, sample_comprehensive_scores, config
            )

            assert 'comprehensive_dashboard' in visualizations

    def test_handles_visualization_error_gracefully(self, capsys):
        """Handles errors without raising."""
        with patch('src.features.selection.interface.interface_dashboard_viz.plt') as mock_plt:
            mock_plt.figure.side_effect = Exception("Plot error")

            visualizations = create_comprehensive_dashboard_visualizations(
                [], [], [], {}
            )

            # Should return empty dict on error
            assert visualizations == {}
            captured = capsys.readouterr()
            assert 'WARNING' in captured.out


class TestCreateDashboardVisualizationsSafe:
    """Tests for create_dashboard_visualizations_safe."""

    def test_updates_results_with_visualizations(self, sample_comprehensive_scores, capsys):
        """Updates results dict with visualizations."""
        with patch(
            'src.features.selection.interface.interface_dashboard_viz.'
            'create_comprehensive_dashboard_visualizations'
        ) as mock_create:
            mock_create.return_value = {'dashboard': MagicMock()}

            results = {
                'win_rate_results': [],
                'information_ratio_results': []
            }
            config = {}

            create_dashboard_visualizations_safe(results, sample_comprehensive_scores, config)

            assert 'visualizations' in results
            assert 'dashboard' in results['visualizations']

    def test_handles_error_gracefully(self, sample_comprehensive_scores, capsys):
        """Handles errors and sets empty visualizations."""
        with patch(
            'src.features.selection.interface.interface_dashboard_viz.'
            'create_comprehensive_dashboard_visualizations'
        ) as mock_create:
            mock_create.side_effect = Exception("Visualization error")

            results = {
                'win_rate_results': [],
                'information_ratio_results': []
            }

            create_dashboard_visualizations_safe(results, sample_comprehensive_scores, {})

            assert results['visualizations'] == {}
            captured = capsys.readouterr()
            assert 'WARNING' in captured.out

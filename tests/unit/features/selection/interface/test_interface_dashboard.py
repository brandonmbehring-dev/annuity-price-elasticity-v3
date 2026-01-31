"""
Tests for interface_dashboard module.

Target: 20% → 60%+ coverage
Tests organized by function categories:
- Dashboard analysis pipeline execution
- Main dashboard generation orchestration
- Module exports
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any

from src.features.selection.interface.interface_dashboard import (
    _execute_dashboard_analysis_pipeline,
    generate_comprehensive_stability_dashboard,
)


# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class MockBootstrapResult:
    """Mock bootstrap result for testing."""
    model_name: str
    model_features: str
    bootstrap_aics: List[float]
    original_aic: float
    stability_assessment: str


@pytest.fixture
def sample_bootstrap_results():
    """Sample bootstrap results for testing."""
    return [
        MockBootstrapResult(
            model_name='Model_1',
            model_features='feature_a + feature_b',
            bootstrap_aics=[100.0, 102.0, 98.0],
            original_aic=100.0,
            stability_assessment='Stable'
        ),
        MockBootstrapResult(
            model_name='Model_2',
            model_features='feature_c + feature_d',
            bootstrap_aics=[110.0, 115.0, 105.0],
            original_aic=110.0,
            stability_assessment='Moderate'
        ),
    ]


@pytest.fixture
def sample_config():
    """Sample dashboard configuration."""
    return {
        'models_to_analyze': 10,
        'create_visualizations': False,
        'integration_weights': {
            'win_rate_weight': 0.5,
            'information_ratio_weight': 0.5,
        },
        'create_dvc_checkpoint': False,
    }


@pytest.fixture
def mock_advanced_analysis():
    """Mock advanced stability analysis function."""
    mock_fn = MagicMock()
    mock_fn.return_value = {
        'win_rate_results': [
            {'model_name': 'Model_1', 'win_rate': 0.65},
            {'model_name': 'Model_2', 'win_rate': 0.55},
        ],
        'information_ratio_results': [
            {'model_name': 'Model_1', 'information_ratio': 0.52},
            {'model_name': 'Model_2', 'information_ratio': 0.35},
        ],
    }
    return mock_fn


# =============================================================================
# Dashboard Analysis Pipeline Tests
# =============================================================================


class TestExecuteDashboardAnalysisPipeline:
    """Tests for _execute_dashboard_analysis_pipeline."""

    def test_returns_tuple_of_three(
        self, sample_bootstrap_results, sample_config, mock_advanced_analysis
    ):
        """Returns tuple of (win_rate, ir, scores)."""
        with patch(
            'src.features.selection.interface.interface_dashboard.create_comprehensive_scoring_system',
            return_value=[]
        ):
            result = _execute_dashboard_analysis_pipeline(
                sample_bootstrap_results,
                sample_config,
                {'win_rate_weight': 0.5, 'information_ratio_weight': 0.5},
                mock_advanced_analysis,
            )

            assert isinstance(result, tuple)
            assert len(result) == 3

    def test_calls_advanced_analysis(
        self, sample_bootstrap_results, sample_config, mock_advanced_analysis
    ):
        """Calls advanced stability analysis function."""
        with patch(
            'src.features.selection.interface.interface_dashboard.create_comprehensive_scoring_system',
            return_value=[]
        ):
            _execute_dashboard_analysis_pipeline(
                sample_bootstrap_results,
                sample_config,
                {'win_rate_weight': 0.5, 'information_ratio_weight': 0.5},
                mock_advanced_analysis,
            )

            mock_advanced_analysis.assert_called_once()

    def test_extracts_win_rate_results(
        self, sample_bootstrap_results, sample_config, mock_advanced_analysis
    ):
        """Extracts win rate results from analysis."""
        with patch(
            'src.features.selection.interface.interface_dashboard.create_comprehensive_scoring_system',
            return_value=[]
        ):
            win_rate, _, _ = _execute_dashboard_analysis_pipeline(
                sample_bootstrap_results,
                sample_config,
                {'win_rate_weight': 0.5, 'information_ratio_weight': 0.5},
                mock_advanced_analysis,
            )

            assert len(win_rate) == 2
            assert win_rate[0]['model_name'] == 'Model_1'

    def test_extracts_ir_results(
        self, sample_bootstrap_results, sample_config, mock_advanced_analysis
    ):
        """Extracts information ratio results from analysis."""
        with patch(
            'src.features.selection.interface.interface_dashboard.create_comprehensive_scoring_system',
            return_value=[]
        ):
            _, ir_results, _ = _execute_dashboard_analysis_pipeline(
                sample_bootstrap_results,
                sample_config,
                {'win_rate_weight': 0.5, 'information_ratio_weight': 0.5},
                mock_advanced_analysis,
            )

            assert len(ir_results) == 2

    def test_creates_comprehensive_scores(
        self, sample_bootstrap_results, sample_config, mock_advanced_analysis
    ):
        """Creates comprehensive scores via scoring system."""
        expected_scores = [{'model': 'Model_1', 'score': 0.9}]

        with patch(
            'src.features.selection.interface.interface_dashboard.create_comprehensive_scoring_system',
            return_value=expected_scores
        ) as mock_scoring:
            _, _, scores = _execute_dashboard_analysis_pipeline(
                sample_bootstrap_results,
                sample_config,
                {'win_rate_weight': 0.5, 'information_ratio_weight': 0.5},
                mock_advanced_analysis,
            )

            mock_scoring.assert_called_once()
            assert scores == expected_scores

    def test_uses_default_models_to_analyze(
        self, sample_bootstrap_results, mock_advanced_analysis
    ):
        """Uses default models_to_analyze of 15."""
        config = {}  # No models_to_analyze specified

        with patch(
            'src.features.selection.interface.interface_dashboard.create_comprehensive_scoring_system',
            return_value=[]
        ):
            _execute_dashboard_analysis_pipeline(
                sample_bootstrap_results,
                config,
                {'win_rate_weight': 0.5, 'information_ratio_weight': 0.5},
                mock_advanced_analysis,
            )

            call_config = mock_advanced_analysis.call_args[0][1]
            assert call_config['models_to_analyze'] == 15


# =============================================================================
# Main Dashboard Generation Tests
# =============================================================================


class TestGenerateComprehensiveStabilityDashboard:
    """Tests for generate_comprehensive_stability_dashboard."""

    def test_validates_inputs(self, sample_bootstrap_results, sample_config):
        """Validates inputs before proceeding."""
        with patch(
            'src.features.selection.interface.interface_dashboard.validate_dashboard_inputs'
        ) as mock_validate, \
             patch(
                 'src.features.selection.interface.interface_dashboard.import_advanced_stability_analysis'
             ) as mock_import, \
             patch(
                 'src.features.selection.interface.interface_dashboard.extract_dashboard_config',
                 return_value=(10, False, {'win_rate_weight': 0.5, 'information_ratio_weight': 0.5})
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard._execute_dashboard_analysis_pipeline',
                 return_value=([], [], [])
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.generate_final_recommendations',
                 return_value={}
             ):

            mock_import.return_value = MagicMock()
            generate_comprehensive_stability_dashboard(
                sample_bootstrap_results, sample_config
            )

            mock_validate.assert_called_once_with(
                sample_bootstrap_results, sample_config
            )

    def test_returns_dict(self, sample_bootstrap_results, sample_config):
        """Returns dictionary with results."""
        with patch(
            'src.features.selection.interface.interface_dashboard.validate_dashboard_inputs'
        ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.import_advanced_stability_analysis',
                 return_value=MagicMock()
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.extract_dashboard_config',
                 return_value=(10, False, {'win_rate_weight': 0.5, 'information_ratio_weight': 0.5})
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard._execute_dashboard_analysis_pipeline',
                 return_value=([], [], [])
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.generate_final_recommendations',
                 return_value={}
             ):

            result = generate_comprehensive_stability_dashboard(
                sample_bootstrap_results, sample_config
            )

            assert isinstance(result, dict)

    def test_contains_win_rate_results(self, sample_bootstrap_results, sample_config):
        """Result contains win_rate_results."""
        with patch(
            'src.features.selection.interface.interface_dashboard.validate_dashboard_inputs'
        ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.import_advanced_stability_analysis',
                 return_value=MagicMock()
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.extract_dashboard_config',
                 return_value=(10, False, {'win_rate_weight': 0.5, 'information_ratio_weight': 0.5})
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard._execute_dashboard_analysis_pipeline',
                 return_value=([{'model': 'A'}], [], [])
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.generate_final_recommendations',
                 return_value={}
             ):

            result = generate_comprehensive_stability_dashboard(
                sample_bootstrap_results, sample_config
            )

            assert 'win_rate_results' in result

    def test_contains_ir_results(self, sample_bootstrap_results, sample_config):
        """Result contains information_ratio_results."""
        with patch(
            'src.features.selection.interface.interface_dashboard.validate_dashboard_inputs'
        ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.import_advanced_stability_analysis',
                 return_value=MagicMock()
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.extract_dashboard_config',
                 return_value=(10, False, {'win_rate_weight': 0.5, 'information_ratio_weight': 0.5})
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard._execute_dashboard_analysis_pipeline',
                 return_value=([], [{'model': 'A', 'ir': 0.5}], [])
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.generate_final_recommendations',
                 return_value={}
             ):

            result = generate_comprehensive_stability_dashboard(
                sample_bootstrap_results, sample_config
            )

            assert 'information_ratio_results' in result

    def test_contains_comprehensive_scores(self, sample_bootstrap_results, sample_config):
        """Result contains comprehensive_scores."""
        with patch(
            'src.features.selection.interface.interface_dashboard.validate_dashboard_inputs'
        ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.import_advanced_stability_analysis',
                 return_value=MagicMock()
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.extract_dashboard_config',
                 return_value=(10, False, {'win_rate_weight': 0.5, 'information_ratio_weight': 0.5})
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard._execute_dashboard_analysis_pipeline',
                 return_value=([], [], [{'model': 'A', 'score': 0.9}])
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.generate_final_recommendations',
                 return_value={}
             ):

            result = generate_comprehensive_stability_dashboard(
                sample_bootstrap_results, sample_config
            )

            assert 'comprehensive_scores' in result

    def test_contains_final_recommendations(self, sample_bootstrap_results, sample_config):
        """Result contains final_recommendations."""
        with patch(
            'src.features.selection.interface.interface_dashboard.validate_dashboard_inputs'
        ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.import_advanced_stability_analysis',
                 return_value=MagicMock()
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.extract_dashboard_config',
                 return_value=(10, False, {'win_rate_weight': 0.5, 'information_ratio_weight': 0.5})
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard._execute_dashboard_analysis_pipeline',
                 return_value=([], [], [])
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.generate_final_recommendations',
                 return_value={'best_model': 'Model_1'}
             ):

            result = generate_comprehensive_stability_dashboard(
                sample_bootstrap_results, sample_config
            )

            assert 'final_recommendations' in result
            assert result['final_recommendations']['best_model'] == 'Model_1'

    def test_creates_visualizations_when_requested(
        self, sample_bootstrap_results
    ):
        """Creates visualizations when create_visualizations=True."""
        config = {'create_visualizations': True}

        with patch(
            'src.features.selection.interface.interface_dashboard.validate_dashboard_inputs'
        ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.import_advanced_stability_analysis',
                 return_value=MagicMock()
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.extract_dashboard_config',
                 return_value=(10, True, {'win_rate_weight': 0.5, 'information_ratio_weight': 0.5})
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard._execute_dashboard_analysis_pipeline',
                 return_value=([], [], [])
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.create_dashboard_visualizations_safe'
             ) as mock_viz, \
             patch(
                 'src.features.selection.interface.interface_dashboard.generate_final_recommendations',
                 return_value={}
             ):

            generate_comprehensive_stability_dashboard(
                sample_bootstrap_results, config
            )

            mock_viz.assert_called_once()

    def test_skips_visualizations_when_not_requested(
        self, sample_bootstrap_results
    ):
        """Skips visualizations when create_visualizations=False."""
        config = {'create_visualizations': False}

        with patch(
            'src.features.selection.interface.interface_dashboard.validate_dashboard_inputs'
        ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.import_advanced_stability_analysis',
                 return_value=MagicMock()
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.extract_dashboard_config',
                 return_value=(10, False, {'win_rate_weight': 0.5, 'information_ratio_weight': 0.5})
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard._execute_dashboard_analysis_pipeline',
                 return_value=([], [], [])
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.create_dashboard_visualizations_safe'
             ) as mock_viz, \
             patch(
                 'src.features.selection.interface.interface_dashboard.generate_final_recommendations',
                 return_value={}
             ):

            generate_comprehensive_stability_dashboard(
                sample_bootstrap_results, config
            )

            mock_viz.assert_not_called()

    def test_creates_dvc_checkpoint_when_requested(self, sample_bootstrap_results):
        """Creates DVC checkpoint when create_dvc_checkpoint=True."""
        config = {'create_dvc_checkpoint': True}

        with patch(
            'src.features.selection.interface.interface_dashboard.validate_dashboard_inputs'
        ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.import_advanced_stability_analysis',
                 return_value=MagicMock()
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.extract_dashboard_config',
                 return_value=(10, False, {'win_rate_weight': 0.5, 'information_ratio_weight': 0.5})
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard._execute_dashboard_analysis_pipeline',
                 return_value=([], [], [])
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.generate_final_recommendations',
                 return_value={}
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.create_dashboard_dvc_checkpoint'
             ) as mock_dvc:

            generate_comprehensive_stability_dashboard(
                sample_bootstrap_results, config
            )

            mock_dvc.assert_called_once()

    def test_handles_dvc_checkpoint_failure(self, sample_bootstrap_results, capsys):
        """Handles DVC checkpoint failure gracefully with warning."""
        config = {'create_dvc_checkpoint': True}

        with patch(
            'src.features.selection.interface.interface_dashboard.validate_dashboard_inputs'
        ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.import_advanced_stability_analysis',
                 return_value=MagicMock()
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.extract_dashboard_config',
                 return_value=(10, False, {'win_rate_weight': 0.5, 'information_ratio_weight': 0.5})
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard._execute_dashboard_analysis_pipeline',
                 return_value=([], [], [])
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.generate_final_recommendations',
                 return_value={}
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.create_dashboard_dvc_checkpoint',
                 side_effect=Exception("DVC failed")
             ):

            # Should not raise, just warn
            result = generate_comprehensive_stability_dashboard(
                sample_bootstrap_results, config
            )

            captured = capsys.readouterr()
            assert 'WARNING' in captured.out
            assert 'DVC checkpoint creation failed' in captured.out

    def test_raises_runtime_error_on_pipeline_failure(self, sample_bootstrap_results, sample_config):
        """Raises RuntimeError when pipeline fails."""
        with patch(
            'src.features.selection.interface.interface_dashboard.validate_dashboard_inputs'
        ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.import_advanced_stability_analysis',
                 return_value=MagicMock()
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.extract_dashboard_config',
                 return_value=(10, False, {'win_rate_weight': 0.5, 'information_ratio_weight': 0.5})
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard._execute_dashboard_analysis_pipeline',
                 side_effect=Exception("Pipeline failed")
             ):

            with pytest.raises(RuntimeError, match="dashboard generation failed"):
                generate_comprehensive_stability_dashboard(
                    sample_bootstrap_results, sample_config
                )

    def test_prints_success_message(self, sample_bootstrap_results, sample_config, capsys):
        """Prints success message on completion."""
        with patch(
            'src.features.selection.interface.interface_dashboard.validate_dashboard_inputs'
        ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.import_advanced_stability_analysis',
                 return_value=MagicMock()
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.extract_dashboard_config',
                 return_value=(10, False, {'win_rate_weight': 0.5, 'information_ratio_weight': 0.5})
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard._execute_dashboard_analysis_pipeline',
                 return_value=([], [], [])
             ), \
             patch(
                 'src.features.selection.interface.interface_dashboard.generate_final_recommendations',
                 return_value={}
             ):

            generate_comprehensive_stability_dashboard(
                sample_bootstrap_results, sample_config
            )

            captured = capsys.readouterr()
            assert 'SUCCESS' in captured.out


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_main_function_exported(self):
        """Main function is in __all__."""
        from src.features.selection.interface import interface_dashboard

        assert 'generate_comprehensive_stability_dashboard' in interface_dashboard.__all__

    def test_validation_functions_exported(self):
        """Validation functions are in __all__."""
        from src.features.selection.interface import interface_dashboard

        assert 'validate_dashboard_inputs' in interface_dashboard.__all__
        assert 'extract_dashboard_config' in interface_dashboard.__all__

    def test_scoring_functions_exported(self):
        """Scoring functions are in __all__."""
        from src.features.selection.interface import interface_dashboard

        assert 'create_comprehensive_scoring_system' in interface_dashboard.__all__

    def test_viz_functions_exported(self):
        """Visualization functions are in __all__."""
        from src.features.selection.interface import interface_dashboard

        assert 'create_comprehensive_dashboard_visualizations' in interface_dashboard.__all__

    def test_business_functions_exported(self):
        """Business functions are in __all__."""
        from src.features.selection.interface import interface_dashboard

        assert 'generate_final_recommendations' in interface_dashboard.__all__

    def test_dvc_functions_exported(self):
        """DVC functions are in __all__."""
        from src.features.selection.interface import interface_dashboard

        assert 'create_dashboard_dvc_checkpoint' in interface_dashboard.__all__

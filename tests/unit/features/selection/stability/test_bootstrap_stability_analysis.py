"""
Tests for bootstrap_stability_analysis module (orchestrator).

Target: 50% → 90%+ coverage
Tests organized by function categories:
- Input validation
- Configuration extraction
- Header printing
- Result packaging
- Component runners (win rate, IR, visualizations)
- Main public API
"""

from typing import Any, Dict, List
from dataclasses import dataclass
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest

from src.features.selection.stability.bootstrap_stability_analysis import (
    run_advanced_stability_analysis,
    _validate_analysis_inputs,
    _extract_analysis_config,
    _print_analysis_header,
    _package_analysis_results,
    _run_win_rate_analysis,
    _run_information_ratio_analysis,
    _create_visualizations_if_enabled,
)


# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class MockBootstrapResult:
    """Mock bootstrap result dataclass for testing."""
    model_name: str
    model_features: str
    bootstrap_aics: List[float]
    bootstrap_r2_values: List[float]
    original_aic: float
    original_r2: float
    aic_stability_coefficient: float
    r2_stability_coefficient: float
    confidence_intervals: Dict[str, Dict[str, float]]
    successful_fits: int
    total_attempts: int
    stability_assessment: str


@pytest.fixture
def mock_bootstrap_result():
    """Create a single mock bootstrap result."""
    return MockBootstrapResult(
        model_name='Model_1',
        model_features='A + B + C',
        bootstrap_aics=list(np.random.normal(100.0, 1.0, 100)),
        bootstrap_r2_values=list(np.random.normal(0.80, 0.02, 100)),
        original_aic=100.0,
        original_r2=0.80,
        aic_stability_coefficient=0.005,
        r2_stability_coefficient=0.08,
        confidence_intervals={
            'aic': {'lower': 98.0, 'upper': 102.0},
            'r2': {'lower': 0.75, 'upper': 0.85}
        },
        successful_fits=100,
        total_attempts=100,
        stability_assessment='STABLE'
    )


@pytest.fixture
def mock_bootstrap_results():
    """Create a list of mock bootstrap results for testing."""
    results = []
    features_list = ['A + B', 'A + C', 'B + C', 'A + B + C', 'A']
    assessments = ['STABLE', 'STABLE', 'MODERATE', 'STABLE', 'UNSTABLE']

    for i, (features, assessment) in enumerate(zip(features_list, assessments)):
        results.append(MockBootstrapResult(
            model_name=f'Model_{i + 1}',
            model_features=features,
            bootstrap_aics=list(np.random.normal(100 + i, 1, 100)),
            bootstrap_r2_values=list(np.random.normal(0.8 - i * 0.05, 0.02, 100)),
            original_aic=100.0 + i,
            original_r2=0.80 - i * 0.05,
            aic_stability_coefficient=0.005 + i * 0.001,
            r2_stability_coefficient=0.08 + i * 0.01,
            confidence_intervals={
                'aic': {'lower': 98.0 + i, 'upper': 102.0 + i},
                'r2': {'lower': 0.75 - i * 0.05, 'upper': 0.85 - i * 0.05}
            },
            successful_fits=100,
            total_attempts=100,
            stability_assessment=assessment
        ))

    return results


@pytest.fixture
def default_config():
    """Default analysis configuration."""
    return {
        'enable_win_rate_analysis': True,
        'enable_information_ratio': True,
        'create_visualizations': False,
        'models_to_analyze': 15
    }


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestValidateAnalysisInputs:
    """Tests for _validate_analysis_inputs."""

    def test_passes_with_valid_inputs(self, mock_bootstrap_results, default_config):
        """Validation passes with valid inputs."""
        # Should not raise
        _validate_analysis_inputs(mock_bootstrap_results, default_config)

    def test_raises_on_empty_results(self, default_config):
        """Raises ValueError for empty bootstrap results."""
        with pytest.raises(ValueError, match="No bootstrap results"):
            _validate_analysis_inputs([], default_config)

    def test_raises_on_string_config(self, mock_bootstrap_results):
        """Raises ValueError for string config."""
        with pytest.raises(ValueError, match="Configuration must be a dictionary"):
            _validate_analysis_inputs(mock_bootstrap_results, "invalid")

    def test_raises_on_none_config(self, mock_bootstrap_results):
        """Raises ValueError for None config."""
        with pytest.raises(ValueError, match="Configuration must be a dictionary"):
            _validate_analysis_inputs(mock_bootstrap_results, None)

    def test_raises_on_list_config(self, mock_bootstrap_results):
        """Raises ValueError for list config."""
        with pytest.raises(ValueError, match="Configuration must be a dictionary"):
            _validate_analysis_inputs(mock_bootstrap_results, [1, 2, 3])

    def test_accepts_single_result(self, mock_bootstrap_result, default_config):
        """Accepts list with single result."""
        _validate_analysis_inputs([mock_bootstrap_result], default_config)


# =============================================================================
# Configuration Extraction Tests
# =============================================================================


class TestExtractAnalysisConfig:
    """Tests for _extract_analysis_config."""

    def test_uses_defaults_for_empty_config(self):
        """Uses defaults for missing keys."""
        config = {}
        n_results = 20

        enable_wr, enable_ir, create_vis, n_models = _extract_analysis_config(config, n_results)

        assert enable_wr is True  # Default
        assert enable_ir is True  # Default
        assert create_vis is True  # Default
        assert n_models == 15  # Default (min of 15 and 20)

    def test_respects_custom_values(self):
        """Respects custom configuration values."""
        config = {
            'enable_win_rate_analysis': False,
            'enable_information_ratio': False,
            'create_visualizations': False,
            'models_to_analyze': 5
        }
        n_results = 20

        enable_wr, enable_ir, create_vis, n_models = _extract_analysis_config(config, n_results)

        assert enable_wr is False
        assert enable_ir is False
        assert create_vis is False
        assert n_models == 5

    def test_limits_models_to_available(self):
        """Limits n_models to available results."""
        config = {'models_to_analyze': 100}
        n_results = 10

        _, _, _, n_models = _extract_analysis_config(config, n_results)

        assert n_models == 10  # min(100, 10)

    def test_handles_mixed_config(self):
        """Handles partially specified config."""
        config = {
            'enable_win_rate_analysis': False,
            # enable_information_ratio uses default
            'models_to_analyze': 8
        }
        n_results = 15

        enable_wr, enable_ir, create_vis, n_models = _extract_analysis_config(config, n_results)

        assert enable_wr is False
        assert enable_ir is True  # Default
        assert create_vis is True  # Default
        assert n_models == 8


# =============================================================================
# Header Printing Tests
# =============================================================================


class TestPrintAnalysisHeader:
    """Tests for _print_analysis_header."""

    def test_prints_models_to_analyze(self, capsys):
        """Prints number of models to analyze."""
        _print_analysis_header(n_models=10, enable_win_rate=True, enable_ir=True)

        captured = capsys.readouterr()
        assert "10" in captured.out

    def test_prints_win_rate_enabled(self, capsys):
        """Prints Win Rate Analysis status when enabled."""
        _print_analysis_header(n_models=5, enable_win_rate=True, enable_ir=False)

        captured = capsys.readouterr()
        assert "Enabled" in captured.out
        assert "Win Rate" in captured.out

    def test_prints_win_rate_disabled(self, capsys):
        """Prints Win Rate Analysis status when disabled."""
        _print_analysis_header(n_models=5, enable_win_rate=False, enable_ir=True)

        captured = capsys.readouterr()
        assert "Disabled" in captured.out

    def test_prints_ir_enabled(self, capsys):
        """Prints IR Analysis status when enabled."""
        _print_analysis_header(n_models=5, enable_win_rate=False, enable_ir=True)

        captured = capsys.readouterr()
        assert "Enabled" in captured.out
        assert "Information Ratio" in captured.out

    def test_prints_ir_disabled(self, capsys):
        """Prints IR Analysis status when disabled."""
        _print_analysis_header(n_models=5, enable_win_rate=True, enable_ir=False)

        captured = capsys.readouterr()
        lines = captured.out.split('\n')
        # IR line should show Disabled
        assert any("Information Ratio" in line and "Disabled" in line for line in lines)

    def test_prints_starting_message(self, capsys):
        """Prints starting message."""
        _print_analysis_header(n_models=3, enable_win_rate=True, enable_ir=True)

        captured = capsys.readouterr()
        assert "Starting advanced stability analysis" in captured.out


# =============================================================================
# Result Packaging Tests
# =============================================================================


class TestPackageAnalysisResults:
    """Tests for _package_analysis_results."""

    def test_includes_all_components(self):
        """Includes all provided components."""
        win_rate_results = [{'model': 'Model_1', 'win_rate_pct': 50.0}]
        ir_results = [{'model': 'Model_1', 'information_ratio': 0.5}]
        visualizations = {'chart': 'mock_figure'}

        result = _package_analysis_results(win_rate_results, ir_results, visualizations)

        assert 'win_rate_results' in result
        assert 'information_ratio_results' in result
        assert 'visualizations' in result
        assert result['win_rate_results'] == win_rate_results
        assert result['information_ratio_results'] == ir_results
        assert result['visualizations'] == visualizations

    def test_handles_none_win_rate(self):
        """Handles None win_rate_results."""
        ir_results = [{'model': 'Model_1', 'information_ratio': 0.5}]
        visualizations = {}

        result = _package_analysis_results(None, ir_results, visualizations)

        assert 'win_rate_results' not in result
        assert 'information_ratio_results' in result

    def test_handles_none_ir(self):
        """Handles None ir_results."""
        win_rate_results = [{'model': 'Model_1', 'win_rate_pct': 50.0}]
        visualizations = {}

        result = _package_analysis_results(win_rate_results, None, visualizations)

        assert 'win_rate_results' in result
        assert 'information_ratio_results' not in result

    def test_handles_all_none(self):
        """Handles all None components."""
        result = _package_analysis_results(None, None, {})

        assert 'win_rate_results' not in result
        assert 'information_ratio_results' not in result
        assert 'visualizations' in result
        assert result['visualizations'] == {}

    def test_always_includes_visualizations(self):
        """Always includes visualizations key even if empty."""
        result = _package_analysis_results(None, None, {})

        assert 'visualizations' in result


# =============================================================================
# Win Rate Analysis Runner Tests
# =============================================================================


class TestRunWinRateAnalysis:
    """Tests for _run_win_rate_analysis."""

    def test_returns_none_when_disabled(self, mock_bootstrap_results):
        """Returns None when disabled."""
        result = _run_win_rate_analysis(mock_bootstrap_results, n_models=5, enable_win_rate=False)

        assert result is None

    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_bootstrap_win_rates')
    def test_calls_calculate_function(self, mock_calculate, mock_bootstrap_results):
        """Calls calculate_bootstrap_win_rates when enabled."""
        mock_calculate.return_value = [{'model': 'Model_1', 'win_rate_pct': 50.0}]

        _run_win_rate_analysis(mock_bootstrap_results, n_models=5, enable_win_rate=True)

        mock_calculate.assert_called_once()

    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_bootstrap_win_rates')
    def test_slices_to_n_models(self, mock_calculate, mock_bootstrap_results):
        """Slices results to n_models."""
        mock_calculate.return_value = [{'model': 'Model_1'}]
        n_models = 3

        _run_win_rate_analysis(mock_bootstrap_results, n_models=n_models, enable_win_rate=True)

        # Should pass only first 3 results
        call_args = mock_calculate.call_args[0][0]
        assert len(call_args) == n_models

    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_bootstrap_win_rates')
    def test_returns_results(self, mock_calculate, mock_bootstrap_results):
        """Returns results from calculate function."""
        expected = [{'model': 'Model_1', 'win_rate_pct': 50.0}]
        mock_calculate.return_value = expected

        result = _run_win_rate_analysis(mock_bootstrap_results, n_models=5, enable_win_rate=True)

        assert result == expected

    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_bootstrap_win_rates')
    def test_prints_completion_message(self, mock_calculate, mock_bootstrap_results, capsys):
        """Prints completion message."""
        mock_calculate.return_value = [{'model': 'M1'}, {'model': 'M2'}]

        _run_win_rate_analysis(mock_bootstrap_results, n_models=5, enable_win_rate=True)

        captured = capsys.readouterr()
        assert "Win Rate Analysis complete" in captured.out
        assert "2 models" in captured.out

    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_bootstrap_win_rates')
    def test_raises_runtime_error_on_failure(self, mock_calculate, mock_bootstrap_results):
        """Raises RuntimeError on analysis failure."""
        mock_calculate.side_effect = Exception("Analysis failed")

        with pytest.raises(RuntimeError, match="Win Rate Analysis failed"):
            _run_win_rate_analysis(mock_bootstrap_results, n_models=5, enable_win_rate=True)


# =============================================================================
# Information Ratio Analysis Runner Tests
# =============================================================================


class TestRunInformationRatioAnalysis:
    """Tests for _run_information_ratio_analysis."""

    def test_returns_none_when_disabled(self, mock_bootstrap_results):
        """Returns None when disabled."""
        result = _run_information_ratio_analysis(mock_bootstrap_results, n_models=5, enable_ir=False)

        assert result is None

    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_information_ratio_analysis')
    def test_calls_calculate_function(self, mock_calculate, mock_bootstrap_results):
        """Calls calculate_information_ratio_analysis when enabled."""
        mock_calculate.return_value = [{'model': 'Model_1', 'information_ratio': 0.5}]

        _run_information_ratio_analysis(mock_bootstrap_results, n_models=5, enable_ir=True)

        mock_calculate.assert_called_once()

    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_information_ratio_analysis')
    def test_slices_to_n_models(self, mock_calculate, mock_bootstrap_results):
        """Slices results to n_models."""
        mock_calculate.return_value = [{'model': 'Model_1'}]
        n_models = 2

        _run_information_ratio_analysis(mock_bootstrap_results, n_models=n_models, enable_ir=True)

        # Should pass only first 2 results
        call_args = mock_calculate.call_args[0][0]
        assert len(call_args) == n_models

    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_information_ratio_analysis')
    def test_returns_results(self, mock_calculate, mock_bootstrap_results):
        """Returns results from calculate function."""
        expected = [{'model': 'Model_1', 'information_ratio': 0.5}]
        mock_calculate.return_value = expected

        result = _run_information_ratio_analysis(mock_bootstrap_results, n_models=5, enable_ir=True)

        assert result == expected

    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_information_ratio_analysis')
    def test_prints_completion_message(self, mock_calculate, mock_bootstrap_results, capsys):
        """Prints completion message."""
        mock_calculate.return_value = [{'model': 'M1'}, {'model': 'M2'}, {'model': 'M3'}]

        _run_information_ratio_analysis(mock_bootstrap_results, n_models=5, enable_ir=True)

        captured = capsys.readouterr()
        assert "Information Ratio Analysis complete" in captured.out
        assert "3 models" in captured.out

    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_information_ratio_analysis')
    def test_raises_runtime_error_on_failure(self, mock_calculate, mock_bootstrap_results):
        """Raises RuntimeError on analysis failure."""
        mock_calculate.side_effect = Exception("IR calculation failed")

        with pytest.raises(RuntimeError, match="Information Ratio Analysis failed"):
            _run_information_ratio_analysis(mock_bootstrap_results, n_models=5, enable_ir=True)


# =============================================================================
# Visualizations Runner Tests
# =============================================================================


class TestCreateVisualizationsIfEnabled:
    """Tests for _create_visualizations_if_enabled."""

    def test_returns_empty_when_disabled(self):
        """Returns empty dict when disabled."""
        win_rate_results = [{'model': 'M1'}]
        ir_results = [{'model': 'M1'}]

        result = _create_visualizations_if_enabled(
            win_rate_results, ir_results, create_vis=False, config={}
        )

        assert result == {}

    def test_returns_empty_when_no_results(self):
        """Returns empty dict when no results available."""
        result = _create_visualizations_if_enabled(
            None, None, create_vis=True, config={}
        )

        assert result == {}

    @patch('src.features.selection.stability.bootstrap_stability_analysis.create_advanced_visualizations')
    def test_calls_create_visualizations(self, mock_create, default_config):
        """Calls create_advanced_visualizations when enabled."""
        mock_create.return_value = {'fig': MagicMock()}
        win_rate_results = [{'model': 'M1'}]
        ir_results = [{'model': 'M1'}]

        _create_visualizations_if_enabled(
            win_rate_results, ir_results, create_vis=True, config=default_config
        )

        mock_create.assert_called_once()

    @patch('src.features.selection.stability.bootstrap_stability_analysis.create_advanced_visualizations')
    def test_passes_config(self, mock_create, default_config):
        """Passes config to visualization function."""
        mock_create.return_value = {'fig': MagicMock()}
        win_rate_results = [{'model': 'M1'}]

        _create_visualizations_if_enabled(
            win_rate_results, None, create_vis=True, config=default_config
        )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs['config'] == default_config

    @patch('src.features.selection.stability.bootstrap_stability_analysis.create_advanced_visualizations')
    def test_returns_visualizations(self, mock_create):
        """Returns visualization figures."""
        expected = {'fig1': MagicMock(), 'fig2': MagicMock()}
        mock_create.return_value = expected
        win_rate_results = [{'model': 'M1'}]

        result = _create_visualizations_if_enabled(
            win_rate_results, None, create_vis=True, config={}
        )

        assert result == expected

    @patch('src.features.selection.stability.bootstrap_stability_analysis.create_advanced_visualizations')
    def test_prints_success_message(self, mock_create, capsys):
        """Prints success message with figure count."""
        mock_create.return_value = {'fig1': MagicMock(), 'fig2': MagicMock()}
        win_rate_results = [{'model': 'M1'}]

        _create_visualizations_if_enabled(
            win_rate_results, None, create_vis=True, config={}
        )

        captured = capsys.readouterr()
        assert "Advanced visualizations created" in captured.out
        assert "2" in captured.out

    @patch('src.features.selection.stability.bootstrap_stability_analysis.create_advanced_visualizations')
    def test_handles_visualization_failure(self, mock_create, capsys):
        """Returns empty dict and prints warning on failure."""
        mock_create.side_effect = Exception("Plot failed")
        win_rate_results = [{'model': 'M1'}]

        result = _create_visualizations_if_enabled(
            win_rate_results, None, create_vis=True, config={}
        )

        assert result == {}
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_works_with_only_win_rate(self):
        """Works with only win_rate_results."""
        with patch('src.features.selection.stability.bootstrap_stability_analysis.create_advanced_visualizations') as mock:
            mock.return_value = {'fig': MagicMock()}
            win_rate_results = [{'model': 'M1'}]

            result = _create_visualizations_if_enabled(
                win_rate_results, None, create_vis=True, config={}
            )

            assert result is not None

    def test_works_with_only_ir(self):
        """Works with only ir_results."""
        with patch('src.features.selection.stability.bootstrap_stability_analysis.create_advanced_visualizations') as mock:
            mock.return_value = {'fig': MagicMock()}
            ir_results = [{'model': 'M1'}]

            result = _create_visualizations_if_enabled(
                None, ir_results, create_vis=True, config={}
            )

            assert result is not None


# =============================================================================
# Main Public API Tests
# =============================================================================


class TestRunAdvancedStabilityAnalysis:
    """Tests for run_advanced_stability_analysis."""

    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_bootstrap_win_rates')
    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_information_ratio_analysis')
    def test_returns_dict(self, mock_ir, mock_wr, mock_bootstrap_results):
        """Returns dictionary."""
        mock_wr.return_value = [{'model': 'M1'}]
        mock_ir.return_value = [{'model': 'M1'}]
        config = {'enable_win_rate_analysis': True, 'enable_information_ratio': True,
                  'create_visualizations': False}

        result = run_advanced_stability_analysis(mock_bootstrap_results, config)

        assert isinstance(result, dict)

    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_bootstrap_win_rates')
    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_information_ratio_analysis')
    def test_contains_win_rate_results(self, mock_ir, mock_wr, mock_bootstrap_results):
        """Contains win_rate_results when enabled."""
        mock_wr.return_value = [{'model': 'M1', 'win_rate_pct': 50.0}]
        mock_ir.return_value = [{'model': 'M1'}]
        config = {'enable_win_rate_analysis': True, 'create_visualizations': False}

        result = run_advanced_stability_analysis(mock_bootstrap_results, config)

        assert 'win_rate_results' in result
        assert result['win_rate_results'] == [{'model': 'M1', 'win_rate_pct': 50.0}]

    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_bootstrap_win_rates')
    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_information_ratio_analysis')
    def test_contains_ir_results(self, mock_ir, mock_wr, mock_bootstrap_results):
        """Contains information_ratio_results when enabled."""
        mock_wr.return_value = [{'model': 'M1'}]
        mock_ir.return_value = [{'model': 'M1', 'information_ratio': 0.5}]
        config = {'enable_information_ratio': True, 'create_visualizations': False}

        result = run_advanced_stability_analysis(mock_bootstrap_results, config)

        assert 'information_ratio_results' in result
        assert result['information_ratio_results'] == [{'model': 'M1', 'information_ratio': 0.5}]

    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_bootstrap_win_rates')
    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_information_ratio_analysis')
    def test_prints_success_message(self, mock_ir, mock_wr, mock_bootstrap_results, capsys):
        """Prints SUCCESS message on completion."""
        mock_wr.return_value = [{'model': 'M1'}]
        mock_ir.return_value = [{'model': 'M1'}]
        config = {'create_visualizations': False}

        run_advanced_stability_analysis(mock_bootstrap_results, config)

        captured = capsys.readouterr()
        assert "SUCCESS" in captured.out
        assert "Advanced stability analysis complete" in captured.out

    def test_raises_on_empty_results(self, default_config):
        """Raises ValueError for empty results."""
        with pytest.raises(ValueError, match="No bootstrap results"):
            run_advanced_stability_analysis([], default_config)

    def test_raises_on_invalid_config(self, mock_bootstrap_results):
        """Raises ValueError for invalid config."""
        with pytest.raises(ValueError, match="Configuration must be a dictionary"):
            run_advanced_stability_analysis(mock_bootstrap_results, None)

    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_bootstrap_win_rates')
    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_information_ratio_analysis')
    def test_disables_analyses(self, mock_ir, mock_wr, mock_bootstrap_results):
        """Respects disabled analysis flags."""
        config = {
            'enable_win_rate_analysis': False,
            'enable_information_ratio': False,
            'create_visualizations': False
        }

        result = run_advanced_stability_analysis(mock_bootstrap_results, config)

        mock_wr.assert_not_called()
        mock_ir.assert_not_called()
        assert 'win_rate_results' not in result
        assert 'information_ratio_results' not in result

    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_bootstrap_win_rates')
    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_information_ratio_analysis')
    @patch('src.features.selection.stability.bootstrap_stability_analysis.create_advanced_visualizations')
    def test_creates_visualizations(self, mock_vis, mock_ir, mock_wr, mock_bootstrap_results):
        """Creates visualizations when enabled."""
        mock_wr.return_value = [{'model': 'M1'}]
        mock_ir.return_value = [{'model': 'M1'}]
        mock_vis.return_value = {'fig': MagicMock()}
        config = {'create_visualizations': True}

        result = run_advanced_stability_analysis(mock_bootstrap_results, config)

        mock_vis.assert_called_once()
        assert 'visualizations' in result

    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_bootstrap_win_rates')
    def test_propagates_win_rate_error(self, mock_wr, mock_bootstrap_results):
        """Propagates RuntimeError from win rate analysis."""
        mock_wr.side_effect = Exception("Win rate failed")
        config = {'enable_win_rate_analysis': True, 'enable_information_ratio': False,
                  'create_visualizations': False}

        with pytest.raises(RuntimeError, match="Win Rate Analysis failed"):
            run_advanced_stability_analysis(mock_bootstrap_results, config)

    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_bootstrap_win_rates')
    @patch('src.features.selection.stability.bootstrap_stability_analysis.calculate_information_ratio_analysis')
    def test_propagates_ir_error(self, mock_ir, mock_wr, mock_bootstrap_results):
        """Propagates RuntimeError from IR analysis."""
        mock_wr.return_value = [{'model': 'M1'}]
        mock_ir.side_effect = Exception("IR failed")
        config = {'enable_win_rate_analysis': True, 'enable_information_ratio': True,
                  'create_visualizations': False}

        with pytest.raises(RuntimeError, match="Information Ratio Analysis failed"):
            run_advanced_stability_analysis(mock_bootstrap_results, config)


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_public_api_exported(self):
        """Main API function is exported."""
        from src.features.selection.stability import bootstrap_stability_analysis
        assert 'run_advanced_stability_analysis' in bootstrap_stability_analysis.__all__

    def test_validation_exported(self):
        """Validation function is exported."""
        from src.features.selection.stability import bootstrap_stability_analysis
        assert '_validate_analysis_inputs' in bootstrap_stability_analysis.__all__

    def test_config_extraction_exported(self):
        """Config extraction function is exported."""
        from src.features.selection.stability import bootstrap_stability_analysis
        assert '_extract_analysis_config' in bootstrap_stability_analysis.__all__

    def test_reexports_win_rate(self):
        """Re-exports from stability_win_rate."""
        from src.features.selection.stability import bootstrap_stability_analysis
        assert 'calculate_bootstrap_win_rates' in bootstrap_stability_analysis.__all__

    def test_reexports_ir(self):
        """Re-exports from stability_ir."""
        from src.features.selection.stability import bootstrap_stability_analysis
        assert 'calculate_information_ratio_analysis' in bootstrap_stability_analysis.__all__

    def test_reexports_visualizations(self):
        """Re-exports from stability_visualizations."""
        from src.features.selection.stability import bootstrap_stability_analysis
        assert 'create_advanced_visualizations' in bootstrap_stability_analysis.__all__

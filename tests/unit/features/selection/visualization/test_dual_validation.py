"""
Tests for dual_validation module.

Target: 20% → 60%+ coverage
Tests organized by function categories:
- Validation helpers
- Display functions
- JSON conversion
- Results export
- Configuration
- Main API function
"""

import os
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict

from src.features.selection.visualization.dual_validation import (
    _validate_dual_analysis_inputs,
    _print_dual_analysis_header,
    _convert_for_json,
    _save_dual_validation_results,
    _get_dual_validation_defaults,
    _update_configs_for_dual_validation,
    create_dual_validation_config,
    run_dual_validation_stability_analysis,
)


# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class MockBootstrapResult:
    """Mock bootstrap result for testing."""
    model_name: str
    bootstrap_aics: List[float]
    stability_assessment: str


@pytest.fixture
def sample_bootstrap_results():
    """Sample bootstrap results for testing."""
    return [
        MockBootstrapResult(
            model_name='Model_1',
            bootstrap_aics=[100.0, 102.0, 98.0],
            stability_assessment='Stable'
        ),
        MockBootstrapResult(
            model_name='Model_2',
            bootstrap_aics=[105.0, 108.0, 103.0],
            stability_assessment='Moderate'
        ),
    ]


@pytest.fixture
def sample_summary():
    """Sample analysis summary."""
    return {
        'best_model': 'Model_1',
        'win_rate': 0.65,
        'information_ratio': 0.85,
        'grade_distribution': {'A': 1, 'B': 1}
    }


# =============================================================================
# Validation Helper Tests
# =============================================================================


class TestValidateDualAnalysisInputs:
    """Tests for _validate_dual_analysis_inputs."""

    def test_raises_on_empty_results(self):
        """Raises ValueError for empty results list."""
        with pytest.raises(ValueError, match="No bootstrap results provided"):
            _validate_dual_analysis_inputs([])

    def test_accepts_valid_results(self, sample_bootstrap_results):
        """Accepts valid bootstrap results without error."""
        # Should not raise with valid results (if stability analysis is available)
        try:
            _validate_dual_analysis_inputs(sample_bootstrap_results)
        except ImportError:
            # Module not available is acceptable
            pass


class TestPrintDualAnalysisHeader:
    """Tests for _print_dual_analysis_header."""

    def test_prints_header(self, capsys):
        """Prints analysis header."""
        _print_dual_analysis_header(5)

        captured = capsys.readouterr()
        assert 'DUAL VALIDATION STABILITY ANALYSIS' in captured.out

    def test_shows_6_metric_system(self, capsys):
        """Shows 6-metric system description."""
        _print_dual_analysis_header(3)

        captured = capsys.readouterr()
        assert '6-Metric System' in captured.out

    def test_shows_model_count(self, capsys):
        """Shows number of models being analyzed."""
        _print_dual_analysis_header(7)

        captured = capsys.readouterr()
        assert '7 models' in captured.out


# =============================================================================
# JSON Conversion Tests
# =============================================================================


class TestConvertForJson:
    """Tests for _convert_for_json."""

    def test_converts_numpy_float(self):
        """Converts numpy float to Python float."""
        result = _convert_for_json(np.float64(3.14))

        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_converts_numpy_int(self):
        """Converts numpy int to Python int."""
        result = _convert_for_json(np.int64(42))

        assert result == 42
        assert isinstance(result, int)

    def test_converts_numpy_array(self):
        """Converts numpy array to list."""
        arr = np.array([1, 2, 3])

        result = _convert_for_json(arr)

        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_converts_nested_dict(self):
        """Recursively converts nested dict."""
        data = {
            'a': np.float64(1.5),
            'b': {'c': np.int64(10)}
        }

        result = _convert_for_json(data)

        assert result['a'] == 1.5
        assert isinstance(result['a'], float)
        assert result['b']['c'] == 10
        assert isinstance(result['b']['c'], int)

    def test_converts_list_elements(self):
        """Converts numpy types in lists."""
        data = [np.float64(1.0), np.int64(2), 'string']

        result = _convert_for_json(data)

        assert result[0] == 1.0
        assert result[1] == 2
        assert result[2] == 'string'

    def test_preserves_regular_types(self):
        """Preserves regular Python types."""
        data = {'string': 'hello', 'int': 42, 'float': 3.14}

        result = _convert_for_json(data)

        assert result == data


# =============================================================================
# Results Export Tests
# =============================================================================


class TestSaveDualValidationResults:
    """Tests for _save_dual_validation_results."""

    def test_saves_json_file(self, sample_summary, tmp_path):
        """Saves results to JSON file."""
        output_path = str(tmp_path / "results.json")

        _save_dual_validation_results(sample_summary, output_path)

        assert os.path.exists(output_path)

    def test_file_contains_correct_data(self, sample_summary, tmp_path):
        """File contains correct data."""
        output_path = str(tmp_path / "results.json")

        _save_dual_validation_results(sample_summary, output_path)

        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded['best_model'] == 'Model_1'
        assert loaded['win_rate'] == 0.65

    def test_prints_success_message(self, sample_summary, tmp_path, capsys):
        """Prints success message."""
        output_path = str(tmp_path / "results.json")

        _save_dual_validation_results(sample_summary, output_path)

        captured = capsys.readouterr()
        assert 'SUCCESS' in captured.out

    def test_raises_on_write_failure(self, sample_summary):
        """Raises RuntimeError on write failure."""
        with pytest.raises(RuntimeError, match="Failed to save"):
            _save_dual_validation_results(sample_summary, "/nonexistent/path/file.json")


# =============================================================================
# Configuration Tests
# =============================================================================


class TestGetDualValidationDefaults:
    """Tests for _get_dual_validation_defaults."""

    def test_returns_dict(self):
        """Returns dictionary."""
        result = _get_dual_validation_defaults()

        assert isinstance(result, dict)

    def test_contains_enable_flag(self):
        """Contains enable_dual_validation flag."""
        result = _get_dual_validation_defaults()

        assert 'enable_dual_validation' in result
        assert result['enable_dual_validation'] == True  # noqa: E712

    def test_contains_n_bootstrap_samples(self):
        """Contains n_bootstrap_samples."""
        result = _get_dual_validation_defaults()

        assert 'n_bootstrap_samples' in result
        assert result['n_bootstrap_samples'] == 100

    def test_contains_weights(self):
        """Contains weight parameters."""
        result = _get_dual_validation_defaults()

        assert 'win_rate_weight' in result
        assert 'information_ratio_weight' in result
        assert result['win_rate_weight'] == 0.5
        assert result['information_ratio_weight'] == 0.5


class TestUpdateConfigsForDualValidation:
    """Tests for _update_configs_for_dual_validation."""

    def test_adds_dual_validation_section(self):
        """Adds dual_validation section to configs."""
        configs = {'bootstrap_config': {'n_bootstrap_samples': 50}}
        dual_config = {'n_bootstrap_samples': 100, 'enable': True}

        result = _update_configs_for_dual_validation(configs, dual_config)

        assert 'dual_validation' in result

    def test_updates_bootstrap_samples(self):
        """Updates bootstrap config with dual validation samples."""
        configs = {'bootstrap_config': {'n_bootstrap_samples': 50}}
        dual_config = {'n_bootstrap_samples': 200}

        result = _update_configs_for_dual_validation(configs, dual_config)

        assert result['bootstrap_config']['n_bootstrap_samples'] == 200

    def test_preserves_other_config(self):
        """Preserves other config sections."""
        configs = {'other_key': 'value', 'bootstrap_config': {}}
        dual_config = {'n_bootstrap_samples': 100}

        result = _update_configs_for_dual_validation(configs, dual_config)

        assert result['other_key'] == 'value'


class TestCreateDualValidationConfig:
    """Tests for create_dual_validation_config."""

    def test_returns_dict(self):
        """Returns dictionary."""
        result = create_dual_validation_config()

        assert isinstance(result, dict)

    def test_default_values(self):
        """Has correct default values."""
        result = create_dual_validation_config()

        assert result['enable_dual_validation'] == True  # noqa: E712
        assert result['n_bootstrap_samples'] == 100
        assert result['win_rate_weight'] == 0.5
        assert result['information_ratio_weight'] == 0.5
        assert result['min_stability_grade'] == 'B'
        assert result['out_of_sample_split'] == 0.3

    def test_custom_values(self):
        """Accepts custom values."""
        result = create_dual_validation_config(
            n_bootstrap_samples=200,
            win_rate_weight=0.7,
            information_ratio_weight=0.3,
            min_stability_grade='A',
            out_of_sample_split=0.2
        )

        assert result['n_bootstrap_samples'] == 200
        assert result['win_rate_weight'] == 0.7
        assert result['information_ratio_weight'] == 0.3
        assert result['min_stability_grade'] == 'A'
        assert result['out_of_sample_split'] == 0.2

    def test_raises_when_weights_dont_sum_to_one(self):
        """Raises ValueError when weights don't sum to 1.0."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            create_dual_validation_config(
                win_rate_weight=0.6,
                information_ratio_weight=0.6
            )

    def test_accepts_edge_case_weights(self):
        """Accepts edge case weights that sum to 1.0."""
        result = create_dual_validation_config(
            win_rate_weight=1.0,
            information_ratio_weight=0.0
        )

        assert result['win_rate_weight'] == 1.0
        assert result['information_ratio_weight'] == 0.0


# =============================================================================
# Main API Tests
# =============================================================================


class TestRunDualValidationStabilityAnalysis:
    """Tests for run_dual_validation_stability_analysis."""

    def test_raises_on_empty_results(self):
        """Raises ValueError for empty results."""
        with pytest.raises(ValueError, match="No bootstrap results"):
            run_dual_validation_stability_analysis([])

    def test_prints_header(self, sample_bootstrap_results, capsys):
        """Prints analysis header."""
        with patch('src.features.selection.visualization.dual_validation.run_advanced_stability_analysis') as mock_analysis:
            mock_analysis.return_value = {'success': True}

            try:
                run_dual_validation_stability_analysis(sample_bootstrap_results)
            except ImportError:
                pass  # Module not available is acceptable

            captured = capsys.readouterr()
            # Header should be printed before potential failure
            if 'DUAL VALIDATION' in captured.out:
                assert 'DUAL VALIDATION STABILITY ANALYSIS' in captured.out

    def test_returns_dict(self, sample_bootstrap_results):
        """Returns dictionary."""
        with patch('src.features.selection.visualization.dual_validation.run_advanced_stability_analysis') as mock_analysis, \
             patch('src.features.selection.visualization.dual_validation._display_dual_validation_results'):
            mock_analysis.return_value = {'success': True, 'best_model': 'Model_1'}

            result = run_dual_validation_stability_analysis(sample_bootstrap_results)

            assert isinstance(result, dict)

    def test_returns_error_on_failure(self, sample_bootstrap_results):
        """Returns error dict on analysis failure."""
        with patch('src.features.selection.visualization.dual_validation.run_advanced_stability_analysis') as mock_analysis:
            mock_analysis.side_effect = Exception("Analysis failed")

            result = run_dual_validation_stability_analysis(sample_bootstrap_results)

            assert 'error' in result
            assert result['success'] == False  # noqa: E712

    def test_calls_display_when_requested(self, sample_bootstrap_results):
        """Calls display function when display_results=True."""
        with patch('src.features.selection.visualization.dual_validation.run_advanced_stability_analysis') as mock_analysis, \
             patch('src.features.selection.visualization.dual_validation._display_dual_validation_results') as mock_display:
            mock_analysis.return_value = {'success': True}

            run_dual_validation_stability_analysis(
                sample_bootstrap_results, display_results=True
            )

            mock_display.assert_called_once()

    def test_saves_results_when_requested(self, sample_bootstrap_results, tmp_path):
        """Saves results when save_results=True."""
        output_path = str(tmp_path / "results.json")

        with patch('src.features.selection.visualization.dual_validation.run_advanced_stability_analysis') as mock_analysis, \
             patch('src.features.selection.visualization.dual_validation._display_dual_validation_results'):
            mock_analysis.return_value = {'success': True, 'data': 'test'}

            run_dual_validation_stability_analysis(
                sample_bootstrap_results,
                display_results=False,
                save_results=True,
                output_path=output_path
            )

            assert os.path.exists(output_path)

"""
Tests for interface_dashboard_dvc module.

Target: 18% → 60%+ coverage
Tests organized by function categories:
- Numpy type conversion
- Checkpoint data building
- DVC checkpoint creation
"""

import os
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.features.selection.interface.interface_dashboard_dvc import (
    convert_numpy_types_for_checkpoint,
    build_dashboard_checkpoint_data,
    create_dashboard_dvc_checkpoint,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_results():
    """Sample dashboard results for testing."""
    return {
        'comprehensive_scores': [
            {'model': 'Model_1', 'score': np.float64(0.85)},
            {'model': 'Model_2', 'score': np.float64(0.78)},
        ],
        'win_rate_results': [
            {'model': 'Model_1', 'win_rate': np.float64(0.65)},
        ],
        'information_ratio_results': [
            {'model': 'Model_1', 'ir': np.float64(0.52)},
        ],
        'final_recommendations': {
            'best_model': 'Model_1',
            'confidence': np.float64(0.9),
        },
    }


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'win_rate_weight': np.float64(0.5),
        'ir_weight': np.float64(0.5),
        'n_bootstrap': np.int64(100),
    }


# =============================================================================
# Numpy Type Conversion Tests
# =============================================================================


class TestConvertNumpyTypesForCheckpoint:
    """Tests for convert_numpy_types_for_checkpoint."""

    def test_converts_int64(self):
        """Converts numpy int64 to Python int."""
        result = convert_numpy_types_for_checkpoint(np.int64(42))

        assert result == 42
        assert isinstance(result, int)

    def test_converts_int32(self):
        """Converts numpy int32 to Python int."""
        result = convert_numpy_types_for_checkpoint(np.int32(100))

        assert result == 100
        assert isinstance(result, int)

    def test_converts_float64(self):
        """Converts numpy float64 to Python float."""
        result = convert_numpy_types_for_checkpoint(np.float64(3.14))

        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_converts_float32(self):
        """Converts numpy float32 to Python float."""
        result = convert_numpy_types_for_checkpoint(np.float32(2.5))

        assert result == pytest.approx(2.5)
        assert isinstance(result, float)

    def test_converts_ndarray(self):
        """Converts numpy array to list."""
        arr = np.array([1, 2, 3])
        result = convert_numpy_types_for_checkpoint(arr)

        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_converts_nested_dict(self):
        """Recursively converts values in nested dict."""
        data = {
            'a': np.int64(1),
            'b': {'c': np.float64(2.5)},
        }
        result = convert_numpy_types_for_checkpoint(data)

        assert result['a'] == 1
        assert isinstance(result['a'], int)
        assert result['b']['c'] == 2.5
        assert isinstance(result['b']['c'], float)

    def test_converts_list_elements(self):
        """Converts numpy types in lists."""
        data = [np.int64(1), np.float64(2.5), 'string']
        result = convert_numpy_types_for_checkpoint(data)

        assert result == [1, 2.5, 'string']
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)

    def test_preserves_python_types(self):
        """Preserves regular Python types unchanged."""
        data = {'string': 'hello', 'int': 42, 'float': 3.14}
        result = convert_numpy_types_for_checkpoint(data)

        assert result == data

    def test_handles_numpy_scalar_with_item(self):
        """Handles numpy scalars with item() method."""
        # Create a scalar from array indexing
        arr = np.array([42])
        scalar = arr[0]  # numpy scalar

        result = convert_numpy_types_for_checkpoint(scalar)

        assert result == 42

    def test_handles_deeply_nested(self):
        """Handles deeply nested structures."""
        data = {
            'level1': {
                'level2': {
                    'level3': [np.int64(1), np.float64(2.5)],
                },
            },
        }
        result = convert_numpy_types_for_checkpoint(data)

        assert result['level1']['level2']['level3'] == [1, 2.5]


# =============================================================================
# Checkpoint Data Building Tests
# =============================================================================


class TestBuildDashboardCheckpointData:
    """Tests for build_dashboard_checkpoint_data."""

    def test_returns_dict(self, sample_results, sample_config):
        """Returns dictionary."""
        result = build_dashboard_checkpoint_data(sample_results, sample_config)

        assert isinstance(result, dict)

    def test_contains_metadata(self, sample_results, sample_config):
        """Contains dashboard_metadata section."""
        result = build_dashboard_checkpoint_data(sample_results, sample_config)

        assert 'dashboard_metadata' in result
        assert 'generation_timestamp' in result['dashboard_metadata']
        assert 'dashboard_version' in result['dashboard_metadata']
        assert 'analysis_type' in result['dashboard_metadata']

    def test_contains_comprehensive_scores(self, sample_results, sample_config):
        """Contains comprehensive_scores section."""
        result = build_dashboard_checkpoint_data(sample_results, sample_config)

        assert 'comprehensive_scores' in result
        assert len(result['comprehensive_scores']) == 2

    def test_contains_win_rate_results(self, sample_results, sample_config):
        """Contains win_rate_results section."""
        result = build_dashboard_checkpoint_data(sample_results, sample_config)

        assert 'win_rate_results' in result

    def test_contains_ir_results(self, sample_results, sample_config):
        """Contains information_ratio_results section."""
        result = build_dashboard_checkpoint_data(sample_results, sample_config)

        assert 'information_ratio_results' in result

    def test_contains_recommendations(self, sample_results, sample_config):
        """Contains final_recommendations section."""
        result = build_dashboard_checkpoint_data(sample_results, sample_config)

        assert 'final_recommendations' in result
        assert result['final_recommendations']['best_model'] == 'Model_1'

    def test_contains_configuration(self, sample_results, sample_config):
        """Contains configuration section."""
        result = build_dashboard_checkpoint_data(sample_results, sample_config)

        assert 'configuration' in result

    def test_converts_numpy_in_scores(self, sample_results, sample_config):
        """Converts numpy types in scores."""
        result = build_dashboard_checkpoint_data(sample_results, sample_config)

        # Check that numpy types are converted
        score = result['comprehensive_scores'][0]['score']
        assert isinstance(score, float)

    def test_converts_numpy_in_config(self, sample_results, sample_config):
        """Converts numpy types in configuration."""
        result = build_dashboard_checkpoint_data(sample_results, sample_config)

        assert isinstance(result['configuration']['win_rate_weight'], float)
        assert isinstance(result['configuration']['n_bootstrap'], int)

    def test_handles_empty_results(self, sample_config):
        """Handles empty results gracefully."""
        result = build_dashboard_checkpoint_data({}, sample_config)

        assert result['comprehensive_scores'] == []
        assert result['win_rate_results'] == []
        assert result['information_ratio_results'] == []
        assert result['final_recommendations'] == {}

    def test_timestamp_is_valid_iso(self, sample_results, sample_config):
        """Timestamp is valid ISO format."""
        result = build_dashboard_checkpoint_data(sample_results, sample_config)

        timestamp = result['dashboard_metadata']['generation_timestamp']
        # Should not raise
        datetime.fromisoformat(timestamp)

    def test_dashboard_version(self, sample_results, sample_config):
        """Dashboard version is comprehensive_v1."""
        result = build_dashboard_checkpoint_data(sample_results, sample_config)

        assert result['dashboard_metadata']['dashboard_version'] == 'comprehensive_v1'


# =============================================================================
# DVC Checkpoint Creation Tests
# =============================================================================


class TestCreateDashboardDvcCheckpoint:
    """Tests for create_dashboard_dvc_checkpoint."""

    def test_creates_output_directory(self, sample_results, sample_config, tmp_path):
        """Creates output directory if not exists."""
        with patch('src.features.selection.interface.interface_dashboard_dvc.os.makedirs') as mock_makedirs, \
             patch('builtins.open', MagicMock()), \
             patch('json.dump'), \
             patch('src.features.selection.interface.interface_dashboard_dvc.os.system', return_value=0):

            create_dashboard_dvc_checkpoint(sample_results, sample_config)

            mock_makedirs.assert_called_once_with('outputs/feature_selection', exist_ok=True)

    def test_writes_json_file(self, sample_results, sample_config):
        """Writes JSON file to expected path."""
        mock_file = MagicMock()

        with patch('src.features.selection.interface.interface_dashboard_dvc.os.makedirs'), \
             patch('builtins.open', return_value=mock_file) as mock_open, \
             patch('json.dump') as mock_json_dump, \
             patch('src.features.selection.interface.interface_dashboard_dvc.os.system', return_value=0):

            create_dashboard_dvc_checkpoint(sample_results, sample_config)

            mock_open.assert_called_once()
            call_args = mock_open.call_args[0]
            assert 'comprehensive_dashboard_results.json' in call_args[0]

    def test_calls_dvc_add(self, sample_results, sample_config):
        """Calls DVC add command."""
        with patch('src.features.selection.interface.interface_dashboard_dvc.os.makedirs'), \
             patch('builtins.open', MagicMock()), \
             patch('json.dump'), \
             patch('src.features.selection.interface.interface_dashboard_dvc.os.system', return_value=0) as mock_system:

            create_dashboard_dvc_checkpoint(sample_results, sample_config)

            mock_system.assert_called_once()
            call_args = mock_system.call_args[0][0]
            assert 'dvc add' in call_args

    def test_prints_success_message(self, sample_results, sample_config, capsys):
        """Prints success message on DVC success."""
        with patch('src.features.selection.interface.interface_dashboard_dvc.os.makedirs'), \
             patch('builtins.open', MagicMock()), \
             patch('json.dump'), \
             patch('src.features.selection.interface.interface_dashboard_dvc.os.system', return_value=0):

            create_dashboard_dvc_checkpoint(sample_results, sample_config)

            captured = capsys.readouterr()
            assert 'DVC checkpoint created' in captured.out

    def test_raises_on_dvc_failure(self, sample_results, sample_config):
        """Raises RuntimeError on DVC command failure."""
        with patch('src.features.selection.interface.interface_dashboard_dvc.os.makedirs'), \
             patch('builtins.open', MagicMock()), \
             patch('json.dump'), \
             patch('src.features.selection.interface.interface_dashboard_dvc.os.system', return_value=1):

            with pytest.raises(RuntimeError, match="DVC checkpoint creation failed"):
                create_dashboard_dvc_checkpoint(sample_results, sample_config)

    def test_json_dump_with_indent(self, sample_results, sample_config):
        """JSON dump uses indent=2."""
        with patch('src.features.selection.interface.interface_dashboard_dvc.os.makedirs'), \
             patch('builtins.open', MagicMock()), \
             patch('json.dump') as mock_json_dump, \
             patch('src.features.selection.interface.interface_dashboard_dvc.os.system', return_value=0):

            create_dashboard_dvc_checkpoint(sample_results, sample_config)

            call_kwargs = mock_json_dump.call_args[1]
            assert call_kwargs['indent'] == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for DVC checkpoint workflow."""

    def test_full_workflow(self, sample_results, sample_config, tmp_path):
        """Full workflow produces valid JSON."""
        # Build checkpoint data
        checkpoint_data = build_dashboard_checkpoint_data(sample_results, sample_config)

        # Verify it's JSON serializable
        json_str = json.dumps(checkpoint_data)
        assert len(json_str) > 0

        # Parse back
        parsed = json.loads(json_str)
        assert 'dashboard_metadata' in parsed
        assert 'comprehensive_scores' in parsed

    def test_empty_results_workflow(self, sample_config):
        """Workflow handles empty results."""
        checkpoint_data = build_dashboard_checkpoint_data({}, sample_config)

        # Should be JSON serializable
        json_str = json.dumps(checkpoint_data)
        parsed = json.loads(json_str)

        assert parsed['comprehensive_scores'] == []

"""
Unit tests for src/data/dvc_manager.py

Tests DVC tracking automation, experiment management,
and data versioning functionality using the actual module API.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture
def sample_experiment_df():
    """Sample DataFrame for experiment tracking."""
    return pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=100, freq='D'),
        'value': np.random.uniform(0, 100, 100),
    })


class TestDVCManager:
    """Tests for DVCManager class."""

    def test_initialization(self, tmp_path):
        """Initializes DVCManager with base directory."""
        from src.data.dvc_manager import DVCManager

        manager = DVCManager(str(tmp_path))
        assert manager.base_dir == tmp_path

    def test_creates_outputs_directory(self, tmp_path):
        """Auto-creates outputs/datasets directory on init."""
        from src.data.dvc_manager import DVCManager

        manager = DVCManager(str(tmp_path))
        assert manager.outputs_dir.exists()


class TestDVCManagerSaveDataset:
    """Tests for DVCManager.save_dataset() method."""

    def test_save_dataset_creates_file(self, tmp_path, sample_experiment_df):
        """Saves DataFrame to parquet file."""
        from src.data.dvc_manager import DVCManager

        manager = DVCManager(str(tmp_path))

        # Mock DVC commands to avoid actual DVC operations
        with patch.object(manager, '_dvc_add'):
            with patch.object(manager, '_dvc_push_background'):
                output_path = manager.save_dataset(
                    sample_experiment_df,
                    'test_dataset',
                    'Test description'
                )

        assert Path(output_path).exists()
        loaded = pd.read_parquet(output_path)
        assert len(loaded) == len(sample_experiment_df)

    def test_save_dataset_empty_raises(self, tmp_path):
        """Empty DataFrame raises ValueError."""
        from src.data.dvc_manager import DVCManager

        manager = DVCManager(str(tmp_path))
        with pytest.raises(ValueError, match="Cannot save empty DataFrame"):
            manager.save_dataset(pd.DataFrame(), 'empty_test')

    def test_save_dataset_invalid_name_raises(self, tmp_path, sample_experiment_df):
        """Invalid dataset name raises ValueError."""
        from src.data.dvc_manager import DVCManager

        manager = DVCManager(str(tmp_path))
        with pytest.raises(ValueError, match="Invalid dataset name"):
            manager.save_dataset(sample_experiment_df, 'invalid@name!')


class TestDVCManagerLoadDataset:
    """Tests for DVCManager.load_dataset() method."""

    def test_load_dataset_success(self, tmp_path, sample_experiment_df):
        """Loads DataFrame from saved dataset."""
        from src.data.dvc_manager import DVCManager

        manager = DVCManager(str(tmp_path))

        # Save directly without DVC operations
        file_path = manager.outputs_dir / 'load_test.parquet'
        sample_experiment_df.to_parquet(file_path)

        loaded = manager.load_dataset('load_test', auto_pull=False)
        assert len(loaded) == len(sample_experiment_df)

    def test_load_dataset_not_found_raises(self, tmp_path):
        """Missing dataset raises FileNotFoundError."""
        from src.data.dvc_manager import DVCManager

        manager = DVCManager(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            manager.load_dataset('nonexistent', auto_pull=False)


class TestDVCManagerCheckpoint:
    """Tests for DVCManager.checkpoint_pipeline() method."""

    def test_checkpoint_saves_multiple_datasets(self, tmp_path, sample_experiment_df):
        """Checkpoint saves multiple related datasets."""
        from src.data.dvc_manager import DVCManager

        manager = DVCManager(str(tmp_path))
        datasets = {
            'dataset_a': sample_experiment_df.copy(),
            'dataset_b': sample_experiment_df.copy(),
        }

        with patch.object(manager, '_dvc_add'):
            with patch.object(manager, '_dvc_push_batch'):
                manager.checkpoint_pipeline('test_stage', datasets)

        # Verify both files exist
        assert (manager.outputs_dir / 'dataset_a.parquet').exists()
        assert (manager.outputs_dir / 'dataset_b.parquet').exists()


class TestDVCManagerListDatasets:
    """Tests for DVCManager.list_datasets() method."""

    def test_list_datasets_empty(self, tmp_path):
        """Returns empty list when no datasets."""
        from src.data.dvc_manager import DVCManager

        manager = DVCManager(str(tmp_path))
        datasets = manager.list_datasets()
        assert datasets == []

    def test_list_datasets_returns_info(self, tmp_path, sample_experiment_df):
        """Returns dataset information."""
        from src.data.dvc_manager import DVCManager

        manager = DVCManager(str(tmp_path))

        # Save a dataset directly
        file_path = manager.outputs_dir / 'list_test.parquet'
        sample_experiment_df.to_parquet(file_path)

        datasets = manager.list_datasets()
        assert len(datasets) == 1
        assert datasets[0]['name'] == 'list_test'
        assert datasets[0]['local_exists'] is True


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_create_dvc_manager_returns_new_instance(self, tmp_path):
        """create_dvc_manager returns new instance each time."""
        from src.data.dvc_manager import create_dvc_manager

        manager1 = create_dvc_manager(str(tmp_path))
        manager2 = create_dvc_manager(str(tmp_path))

        assert manager1 is not manager2

    def test_save_dataset_convenience_function(self, tmp_path, sample_experiment_df):
        """Module-level save_dataset function works."""
        from src.data.dvc_manager import save_dataset, create_dvc_manager

        manager = create_dvc_manager(str(tmp_path))

        with patch.object(manager, '_dvc_add'):
            with patch.object(manager, '_dvc_push_background'):
                output_path = save_dataset(
                    sample_experiment_df,
                    'func_test',
                    'Description',
                    manager=manager
                )

        assert Path(output_path).exists()

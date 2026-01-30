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


# =============================================================================
# Additional Coverage Tests (Phase 2)
# =============================================================================


class TestDVCAddMethod:
    """Tests for DVCManager._dvc_add() subprocess handling."""

    def test_dvc_add_success(self, tmp_path, sample_experiment_df):
        """_dvc_add completes successfully with returncode 0."""
        from src.data.dvc_manager import DVCManager

        manager = DVCManager(str(tmp_path))
        file_path = manager.outputs_dir / "test.parquet"
        sample_experiment_df.to_parquet(file_path)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with patch('subprocess.run', return_value=mock_result) as mock_run:
            manager._dvc_add(file_path)

            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args[0] == "dvc"
            assert args[1] == "add"

    def test_dvc_add_warning_on_nonzero_return(self, tmp_path, caplog):
        """_dvc_add logs warning on non-zero return code."""
        from src.data.dvc_manager import DVCManager
        import logging

        manager = DVCManager(str(tmp_path))
        file_path = manager.outputs_dir / "test.parquet"

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "DVC warning message"

        with patch('subprocess.run', return_value=mock_result):
            with caplog.at_level(logging.WARNING):
                manager._dvc_add(file_path)

        assert "DVC add warning" in caplog.text

    def test_dvc_add_timeout_logged(self, tmp_path, caplog):
        """_dvc_add logs error on timeout."""
        from src.data.dvc_manager import DVCManager
        import subprocess
        import logging

        manager = DVCManager(str(tmp_path))
        file_path = manager.outputs_dir / "test.parquet"

        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired("dvc", 60)):
            with caplog.at_level(logging.ERROR):
                manager._dvc_add(file_path)

        assert "timeout" in caplog.text.lower()

    def test_dvc_add_exception_logged(self, tmp_path, caplog):
        """_dvc_add logs error on general exception."""
        from src.data.dvc_manager import DVCManager
        import logging

        manager = DVCManager(str(tmp_path))
        file_path = manager.outputs_dir / "test.parquet"

        with patch('subprocess.run', side_effect=OSError("DVC not found")):
            with caplog.at_level(logging.ERROR):
                manager._dvc_add(file_path)

        assert "DVC add failed" in caplog.text


class TestDVCPushBackgroundMethod:
    """Tests for DVCManager._dvc_push_background() Popen handling."""

    def test_dvc_push_background_starts_process(self, tmp_path):
        """_dvc_push_background starts background Popen process."""
        from src.data.dvc_manager import DVCManager

        manager = DVCManager(str(tmp_path))

        with patch('subprocess.Popen') as mock_popen:
            manager._dvc_push_background("test.parquet")

            mock_popen.assert_called_once()
            args = mock_popen.call_args[0][0]
            assert args[0] == "dvc"
            assert args[1] == "push"

    def test_dvc_push_background_exception_logged(self, tmp_path, caplog):
        """_dvc_push_background logs warning on exception."""
        from src.data.dvc_manager import DVCManager
        import logging

        manager = DVCManager(str(tmp_path))

        with patch('subprocess.Popen', side_effect=OSError("Cannot start process")):
            with caplog.at_level(logging.WARNING):
                manager._dvc_push_background("test.parquet")

        assert "Background DVC push failed" in caplog.text


class TestDVCPushBatchMethod:
    """Tests for DVCManager._dvc_push_batch() subprocess handling."""

    def test_dvc_push_batch_success(self, tmp_path, caplog):
        """_dvc_push_batch logs success on returncode 0."""
        from src.data.dvc_manager import DVCManager
        import logging

        manager = DVCManager(str(tmp_path))

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with patch('subprocess.run', return_value=mock_result):
            with caplog.at_level(logging.INFO):
                manager._dvc_push_batch()

        assert "batch push completed successfully" in caplog.text

    def test_dvc_push_batch_warning_on_nonzero(self, tmp_path, caplog):
        """_dvc_push_batch logs warning on non-zero return."""
        from src.data.dvc_manager import DVCManager
        import logging

        manager = DVCManager(str(tmp_path))

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Some files skipped"

        with patch('subprocess.run', return_value=mock_result):
            with caplog.at_level(logging.WARNING):
                manager._dvc_push_batch()

        assert "warnings" in caplog.text.lower()

    def test_dvc_push_batch_timeout_logged(self, tmp_path, caplog):
        """_dvc_push_batch logs error on timeout."""
        from src.data.dvc_manager import DVCManager
        import subprocess
        import logging

        manager = DVCManager(str(tmp_path))

        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired("dvc", 300)):
            with caplog.at_level(logging.ERROR):
                manager._dvc_push_batch()

        assert "timeout" in caplog.text.lower()

    def test_dvc_push_batch_exception_logged(self, tmp_path, caplog):
        """_dvc_push_batch logs error on general exception."""
        from src.data.dvc_manager import DVCManager
        import logging

        manager = DVCManager(str(tmp_path))

        with patch('subprocess.run', side_effect=OSError("Network error")):
            with caplog.at_level(logging.ERROR):
                manager._dvc_push_batch()

        assert "batch push failed" in caplog.text


class TestDVCPullMethod:
    """Tests for DVCManager._dvc_pull() subprocess handling."""

    def test_dvc_pull_success(self, tmp_path):
        """_dvc_pull completes successfully."""
        from src.data.dvc_manager import DVCManager

        manager = DVCManager(str(tmp_path))

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch('subprocess.run', return_value=mock_result) as mock_run:
            manager._dvc_pull("test.parquet")

            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args[0] == "dvc"
            assert args[1] == "pull"

    def test_dvc_pull_failure_raises(self, tmp_path):
        """_dvc_pull raises RuntimeError on non-zero return."""
        from src.data.dvc_manager import DVCManager

        manager = DVCManager(str(tmp_path))

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "File not found in remote"

        with patch('subprocess.run', return_value=mock_result):
            with pytest.raises(RuntimeError, match="DVC pull failed"):
                manager._dvc_pull("nonexistent.parquet")

    def test_dvc_pull_timeout_raises(self, tmp_path):
        """_dvc_pull raises RuntimeError on timeout."""
        from src.data.dvc_manager import DVCManager
        import subprocess

        manager = DVCManager(str(tmp_path))

        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired("dvc", 120)):
            with pytest.raises(RuntimeError, match="timeout"):
                manager._dvc_pull("slow_file.parquet")


class TestGetDVCManagerSingleton:
    """Tests for get_dvc_manager() singleton behavior."""

    def test_get_dvc_manager_returns_singleton(self):
        """get_dvc_manager returns same instance on repeated calls."""
        from src.data import dvc_manager as dm

        # Reset singleton for test isolation
        dm._dvc_manager = None

        manager1 = dm.get_dvc_manager()
        manager2 = dm.get_dvc_manager()

        assert manager1 is manager2

        # Clean up
        dm._dvc_manager = None

    def test_get_dvc_manager_creates_instance_on_first_call(self):
        """get_dvc_manager creates instance if none exists."""
        from src.data import dvc_manager as dm

        # Reset singleton
        dm._dvc_manager = None

        manager = dm.get_dvc_manager()
        assert manager is not None
        assert isinstance(manager, dm.DVCManager)

        # Clean up
        dm._dvc_manager = None


class TestLoadDatasetWithAutoPull:
    """Tests for load_dataset with auto_pull=True."""

    def test_load_dataset_auto_pull_success(self, tmp_path, sample_experiment_df):
        """load_dataset with auto_pull fetches from DVC remote."""
        from src.data.dvc_manager import DVCManager

        manager = DVCManager(str(tmp_path))
        file_path = manager.outputs_dir / "pulled_dataset.parquet"

        # Simulate: file doesn't exist initially, pull creates it
        def simulate_pull(filename):
            sample_experiment_df.to_parquet(file_path)

        with patch.object(manager, '_dvc_pull', side_effect=simulate_pull):
            df = manager.load_dataset('pulled_dataset', auto_pull=True)

        assert len(df) == len(sample_experiment_df)

    def test_load_dataset_auto_pull_failure_raises(self, tmp_path):
        """load_dataset raises when auto_pull also fails."""
        from src.data.dvc_manager import DVCManager

        manager = DVCManager(str(tmp_path))

        with patch.object(manager, '_dvc_pull', side_effect=RuntimeError("Pull failed")):
            with pytest.raises(FileNotFoundError, match="Dataset not found"):
                manager.load_dataset('missing_dataset', auto_pull=True)


class TestLoadDatasetLocalReadFailure:
    """Tests for load_dataset when local file read fails."""

    def test_load_dataset_corrupted_file_attempts_pull(self, tmp_path, caplog):
        """load_dataset attempts DVC pull when local read fails."""
        from src.data.dvc_manager import DVCManager
        import logging

        manager = DVCManager(str(tmp_path))
        file_path = manager.outputs_dir / "corrupted.parquet"

        # Create corrupted file
        file_path.write_text("not a parquet file")

        with patch.object(manager, '_dvc_pull', side_effect=RuntimeError("Pull failed")):
            with caplog.at_level(logging.WARNING):
                with pytest.raises(FileNotFoundError):
                    manager.load_dataset('corrupted', auto_pull=True)

        assert "Failed to load local file" in caplog.text


class TestListDatasetsParquetReadFailure:
    """Tests for list_datasets when parquet read fails."""

    def test_list_datasets_handles_corrupted_parquet(self, tmp_path, caplog):
        """list_datasets handles corrupted parquet gracefully."""
        from src.data.dvc_manager import DVCManager
        import logging

        manager = DVCManager(str(tmp_path))
        file_path = manager.outputs_dir / "bad_file.parquet"

        # Create corrupted parquet
        file_path.write_text("not a valid parquet")

        with caplog.at_level(logging.DEBUG):
            datasets = manager.list_datasets()

        # Should still return dataset info, just without row/column counts
        assert len(datasets) == 1
        assert datasets[0]['name'] == 'bad_file'
        assert datasets[0]['rows'] is None
        assert datasets[0]['columns'] is None


class TestConvenienceFunctionsWithSingleton:
    """Tests for convenience functions using singleton fallback."""

    def test_load_dataset_convenience_uses_singleton(self, tmp_path, sample_experiment_df):
        """load_dataset convenience function uses singleton when no manager provided."""
        from src.data import dvc_manager as dm

        # Setup: create a manager and save data
        manager = dm.create_dvc_manager(str(tmp_path))
        file_path = manager.outputs_dir / "singleton_test.parquet"
        sample_experiment_df.to_parquet(file_path)

        # Set singleton to our manager
        dm._dvc_manager = manager

        # Call without manager argument
        df = dm.load_dataset('singleton_test')
        assert len(df) == len(sample_experiment_df)

        # Clean up
        dm._dvc_manager = None

    def test_checkpoint_pipeline_convenience_function(self, tmp_path, sample_experiment_df):
        """checkpoint_pipeline convenience function works with DI."""
        from src.data.dvc_manager import checkpoint_pipeline, create_dvc_manager

        manager = create_dvc_manager(str(tmp_path))
        datasets = {'checkpoint_test': sample_experiment_df.copy()}

        with patch.object(manager, '_dvc_add'):
            with patch.object(manager, '_dvc_push_batch'):
                checkpoint_pipeline('test_stage', datasets, manager=manager)

        assert (manager.outputs_dir / 'checkpoint_test.parquet').exists()

    def test_list_datasets_convenience_function(self, tmp_path, sample_experiment_df):
        """list_datasets convenience function works with DI."""
        from src.data.dvc_manager import list_datasets, create_dvc_manager

        manager = create_dvc_manager(str(tmp_path))
        file_path = manager.outputs_dir / "list_convenience.parquet"
        sample_experiment_df.to_parquet(file_path)

        result = list_datasets(manager=manager)
        assert len(result) == 1
        assert result[0]['name'] == 'list_convenience'

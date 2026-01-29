"""
Unit tests for src/data/output_manager.py

Tests output directory management, file writing, and
output destination control using the actual module API.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_output_df():
    """Sample DataFrame for output testing."""
    return pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=50, freq='W'),
        'sales': np.random.uniform(10000, 100000, 50),
        'rate': np.random.uniform(0.01, 0.05, 50),
    })


class TestOutputManager:
    """Tests for OutputManager class."""

    def test_initialization(self, tmp_path):
        """Initializes OutputManager with base directory."""
        from src.data.output_manager import OutputManager

        manager = OutputManager(str(tmp_path))
        assert manager.base_dir == tmp_path

    def test_auto_creates_output_directory(self, tmp_path):
        """Auto-creates outputs/datasets directory on init."""
        from src.data.output_manager import OutputManager

        manager = OutputManager(str(tmp_path))
        assert manager.local_output_dir.exists()

    def test_default_write_to_s3_false(self, tmp_path):
        """Default write_to_s3 is False."""
        from src.data.output_manager import OutputManager

        manager = OutputManager(str(tmp_path))
        assert manager.write_to_s3 is False


class TestOutputManagerConfigure:
    """Tests for OutputManager.configure() method."""

    def test_configure_local_mode(self, tmp_path):
        """Configure for local output mode."""
        from src.data.output_manager import OutputManager

        manager = OutputManager(str(tmp_path))
        manager.configure(write_to_s3=False)
        assert manager.write_to_s3 is False

    def test_configure_s3_mode_requires_config(self, tmp_path):
        """S3 mode requires s3_config parameter."""
        from src.data.output_manager import OutputManager

        manager = OutputManager(str(tmp_path))
        with pytest.raises(ValueError, match="s3_config required"):
            manager.configure(write_to_s3=True)

    def test_configure_s3_mode_validates_keys(self, tmp_path):
        """S3 mode validates required config keys."""
        from src.data.output_manager import OutputManager

        manager = OutputManager(str(tmp_path))
        with pytest.raises(ValueError, match="missing required keys"):
            manager.configure(write_to_s3=True, s3_config={'incomplete': 'config'})


class TestOutputManagerSaveOutput:
    """Tests for OutputManager.save_output() method."""

    def test_save_output_parquet(self, tmp_path, sample_output_df):
        """Saves DataFrame to parquet file."""
        from src.data.output_manager import OutputManager

        manager = OutputManager(str(tmp_path))
        output_path = manager.save_output(sample_output_df, 'test_output')

        assert Path(output_path).exists()
        loaded = pd.read_parquet(output_path)
        assert len(loaded) == len(sample_output_df)

    def test_save_output_empty_raises(self, tmp_path):
        """Empty DataFrame raises ValueError."""
        from src.data.output_manager import OutputManager

        manager = OutputManager(str(tmp_path))
        with pytest.raises(ValueError, match="Cannot save empty DataFrame"):
            manager.save_output(pd.DataFrame(), 'empty_test')

    def test_save_output_invalid_name_raises(self, tmp_path, sample_output_df):
        """Invalid dataset name raises ValueError."""
        from src.data.output_manager import OutputManager

        manager = OutputManager(str(tmp_path))
        with pytest.raises(ValueError, match="Invalid dataset name"):
            manager.save_output(sample_output_df, 'invalid@name!')

    def test_save_output_with_subdirectory(self, tmp_path, sample_output_df):
        """Saves output with subdirectory."""
        from src.data.output_manager import OutputManager

        manager = OutputManager(str(tmp_path))
        output_path = manager.save_output(sample_output_df, 'test_output', subdirectory='subdir')

        assert Path(output_path).exists()
        assert 'subdir' in output_path


class TestOutputManagerLoadOutput:
    """Tests for OutputManager.load_output() method."""

    def test_load_output_success(self, tmp_path, sample_output_df):
        """Loads DataFrame from saved output."""
        from src.data.output_manager import OutputManager

        manager = OutputManager(str(tmp_path))
        manager.save_output(sample_output_df, 'load_test')
        loaded = manager.load_output('load_test')

        assert len(loaded) == len(sample_output_df)

    def test_load_output_not_found_raises(self, tmp_path):
        """Missing file raises FileNotFoundError."""
        from src.data.output_manager import OutputManager

        manager = OutputManager(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            manager.load_output('nonexistent')


class TestOutputManagerInfo:
    """Tests for OutputManager.get_output_location_info() method."""

    def test_info_local_mode(self, tmp_path):
        """Returns correct info for local mode."""
        from src.data.output_manager import OutputManager

        manager = OutputManager(str(tmp_path))
        info = manager.get_output_location_info()

        assert info['mode'] == 'Local'
        assert 'directory' in info


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_save_output_function(self, tmp_path, sample_output_df):
        """Module-level save_output function works."""
        from src.data.output_manager import save_output, create_output_manager

        manager = create_output_manager(str(tmp_path))
        output_path = save_output(sample_output_df, 'func_test', manager=manager)

        assert Path(output_path).exists()

    def test_create_output_manager_returns_new_instance(self, tmp_path):
        """create_output_manager returns new instance each time."""
        from src.data.output_manager import create_output_manager

        manager1 = create_output_manager(str(tmp_path))
        manager2 = create_output_manager(str(tmp_path))

        assert manager1 is not manager2

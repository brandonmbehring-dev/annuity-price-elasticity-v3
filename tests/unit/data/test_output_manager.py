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


# =============================================================================
# Additional Coverage Tests (Phase 2)
# =============================================================================


class TestConfigureS3Mode:
    """Tests for OutputManager.configure() with valid S3 config."""

    def test_configure_s3_mode_success(self, tmp_path):
        """Configure S3 mode with valid config initializes s3_client."""
        from src.data.output_manager import OutputManager
        from unittest.mock import patch, MagicMock

        manager = OutputManager(str(tmp_path))
        s3_config = {
            'output_bucket_name': 'test-bucket',
            'output_base_path': 'test/path'
        }

        with patch('boto3.client') as mock_boto:
            mock_client = MagicMock()
            mock_boto.return_value = mock_client
            manager.configure(write_to_s3=True, s3_config=s3_config)

        assert manager.write_to_s3 is True
        assert manager.s3_config == s3_config
        assert manager.s3_client is not None
        mock_boto.assert_called_once_with('s3')

    def test_configure_s3_reuses_existing_client(self, tmp_path):
        """Configure S3 mode reuses existing s3_client if already set."""
        from src.data.output_manager import OutputManager
        from unittest.mock import patch, MagicMock

        manager = OutputManager(str(tmp_path))
        s3_config = {
            'output_bucket_name': 'test-bucket',
            'output_base_path': 'test/path'
        }

        # Pre-set client
        existing_client = MagicMock()
        manager.s3_client = existing_client

        with patch('boto3.client') as mock_boto:
            manager.configure(write_to_s3=True, s3_config=s3_config)

        # Should not create new client
        mock_boto.assert_not_called()
        assert manager.s3_client is existing_client


class TestSaveToS3:
    """Tests for OutputManager._save_to_s3() method."""

    def test_save_to_s3_success(self, tmp_path, sample_output_df):
        """_save_to_s3 uploads parquet to S3."""
        from src.data.output_manager import OutputManager
        from unittest.mock import MagicMock

        manager = OutputManager(str(tmp_path))
        manager.s3_client = MagicMock()
        manager.s3_config = {
            'output_bucket_name': 'test-bucket',
            'output_base_path': 'outputs'
        }
        manager.write_to_s3 = True

        uri = manager._save_to_s3(sample_output_df, 'test.parquet', '')

        assert uri == 's3://test-bucket/outputs/test.parquet'
        manager.s3_client.put_object.assert_called_once()

    def test_save_to_s3_with_subdirectory(self, tmp_path, sample_output_df):
        """_save_to_s3 includes subdirectory in S3 key."""
        from src.data.output_manager import OutputManager
        from unittest.mock import MagicMock

        manager = OutputManager(str(tmp_path))
        manager.s3_client = MagicMock()
        manager.s3_config = {
            'output_bucket_name': 'test-bucket',
            'output_base_path': 'base'
        }

        uri = manager._save_to_s3(sample_output_df, 'data.parquet', 'subdir')

        assert uri == 's3://test-bucket/base/subdir/data.parquet'

    def test_save_to_s3_not_configured_raises(self, tmp_path, sample_output_df):
        """_save_to_s3 raises RuntimeError if S3 not configured."""
        from src.data.output_manager import OutputManager

        manager = OutputManager(str(tmp_path))
        # s3_client and s3_config are None by default

        with pytest.raises(RuntimeError, match="S3 not configured"):
            manager._save_to_s3(sample_output_df, 'test.parquet', '')


class TestLoadFromS3:
    """Tests for OutputManager._load_from_s3() method."""

    def test_load_from_s3_success(self, tmp_path, sample_output_df):
        """_load_from_s3 downloads and reads parquet from S3."""
        from src.data.output_manager import OutputManager
        from unittest.mock import MagicMock
        import io

        manager = OutputManager(str(tmp_path))
        manager.s3_client = MagicMock()
        manager.s3_config = {
            'output_bucket_name': 'test-bucket',
            'output_base_path': 'outputs'
        }

        # Create mock parquet data
        parquet_buffer = io.BytesIO()
        sample_output_df.to_parquet(parquet_buffer)
        parquet_bytes = parquet_buffer.getvalue()

        def mock_download(bucket, key, fileobj):
            fileobj.write(parquet_bytes)

        manager.s3_client.download_fileobj.side_effect = mock_download

        df = manager._load_from_s3('test.parquet', '')

        assert len(df) == len(sample_output_df)
        manager.s3_client.download_fileobj.assert_called_once()

    def test_load_from_s3_with_subdirectory(self, tmp_path, sample_output_df):
        """_load_from_s3 includes subdirectory in S3 key."""
        from src.data.output_manager import OutputManager
        from unittest.mock import MagicMock
        import io

        manager = OutputManager(str(tmp_path))
        manager.s3_client = MagicMock()
        manager.s3_config = {
            'output_bucket_name': 'test-bucket',
            'output_base_path': 'base'
        }

        # Create mock parquet data
        parquet_buffer = io.BytesIO()
        sample_output_df.to_parquet(parquet_buffer)
        parquet_bytes = parquet_buffer.getvalue()

        def mock_download(bucket, key, fileobj):
            fileobj.write(parquet_bytes)

        manager.s3_client.download_fileobj.side_effect = mock_download

        df = manager._load_from_s3('data.parquet', 'subdir')

        # Verify correct key was used
        call_args = manager.s3_client.download_fileobj.call_args[0]
        assert call_args[0] == 'test-bucket'
        assert call_args[1] == 'base/subdir/data.parquet'

    def test_load_from_s3_not_configured_raises(self, tmp_path):
        """_load_from_s3 raises RuntimeError if S3 not configured."""
        from src.data.output_manager import OutputManager

        manager = OutputManager(str(tmp_path))

        with pytest.raises(RuntimeError, match="S3 not configured"):
            manager._load_from_s3('test.parquet', '')


class TestLoadOutputWithS3Fallback:
    """Tests for load_output with try_s3_if_local_missing=True."""

    def test_load_output_tries_s3_when_local_missing(self, tmp_path, sample_output_df):
        """load_output tries S3 when local file not found and s3_config set."""
        from src.data.output_manager import OutputManager
        from unittest.mock import MagicMock, patch

        manager = OutputManager(str(tmp_path))
        manager.s3_config = {
            'output_bucket_name': 'test-bucket',
            'output_base_path': 'outputs'
        }

        with patch.object(manager, '_load_from_s3', return_value=sample_output_df) as mock_s3:
            df = manager.load_output('missing_local', try_s3_if_local_missing=True)

        mock_s3.assert_called_once_with('missing_local.parquet', '')
        assert len(df) == len(sample_output_df)

    def test_load_output_no_s3_fallback_without_config(self, tmp_path):
        """load_output raises FileNotFoundError if no s3_config set."""
        from src.data.output_manager import OutputManager

        manager = OutputManager(str(tmp_path))
        # s3_config is None

        with pytest.raises(FileNotFoundError):
            manager.load_output('missing', try_s3_if_local_missing=True)


class TestGetOutputManagerSingleton:
    """Tests for get_output_manager() singleton behavior."""

    def test_get_output_manager_returns_singleton(self):
        """get_output_manager returns same instance on repeated calls."""
        from src.data import output_manager as om

        # Reset singleton
        om._output_manager = None

        manager1 = om.get_output_manager()
        manager2 = om.get_output_manager()

        assert manager1 is manager2

        # Clean up
        om._output_manager = None

    def test_get_output_manager_creates_instance_on_first_call(self):
        """get_output_manager creates instance if none exists."""
        from src.data import output_manager as om

        # Reset singleton
        om._output_manager = None

        manager = om.get_output_manager()
        assert manager is not None
        assert isinstance(manager, om.OutputManager)

        # Clean up
        om._output_manager = None


class TestConfigureOutputModeConvenienceFunction:
    """Tests for configure_output_mode() convenience function."""

    def test_configure_output_mode_with_di(self, tmp_path):
        """configure_output_mode works with DI pattern."""
        from src.data.output_manager import configure_output_mode, create_output_manager

        manager = create_output_manager(str(tmp_path))
        configure_output_mode(write_to_s3=False, manager=manager)

        assert manager.write_to_s3 is False

    def test_configure_output_mode_uses_singleton_when_no_manager(self):
        """configure_output_mode uses singleton when no manager provided."""
        from src.data import output_manager as om
        from unittest.mock import MagicMock

        # Set up singleton
        mock_manager = MagicMock()
        om._output_manager = mock_manager

        om.configure_output_mode(write_to_s3=False)

        mock_manager.configure.assert_called_once_with(write_to_s3=False, s3_config=None)

        # Clean up
        om._output_manager = None


class TestGetOutputInfoConvenienceFunction:
    """Tests for get_output_info() convenience function."""

    def test_get_output_info_with_di(self, tmp_path):
        """get_output_info works with DI pattern."""
        from src.data.output_manager import get_output_info, create_output_manager

        manager = create_output_manager(str(tmp_path))
        info = get_output_info(manager=manager)

        assert info['mode'] == 'Local'
        assert 'directory' in info

    def test_get_output_info_uses_singleton(self):
        """get_output_info uses singleton when no manager provided."""
        from src.data import output_manager as om
        from unittest.mock import MagicMock

        mock_manager = MagicMock()
        mock_manager.get_output_location_info.return_value = {'mode': 'Local'}
        om._output_manager = mock_manager

        info = om.get_output_info()

        mock_manager.get_output_location_info.assert_called_once()
        assert info == {'mode': 'Local'}

        # Clean up
        om._output_manager = None


class TestGetOutputLocationInfoS3Mode:
    """Tests for get_output_location_info() in S3 mode."""

    def test_get_output_location_info_s3(self, tmp_path):
        """get_output_location_info returns S3 details when configured."""
        from src.data.output_manager import OutputManager
        from unittest.mock import patch

        manager = OutputManager(str(tmp_path))
        s3_config = {
            'output_bucket_name': 'my-bucket',
            'output_base_path': 'my/path'
        }

        with patch('boto3.client'):
            manager.configure(write_to_s3=True, s3_config=s3_config)

        info = manager.get_output_location_info()

        assert info['mode'] == 'S3'
        assert info['bucket'] == 'my-bucket'
        assert info['base_path'] == 'my/path'
        assert info['full_path'] == 's3://my-bucket/my/path'


class TestLoadOutputConvenienceFunction:
    """Tests for load_output() convenience function."""

    def test_load_output_with_di(self, tmp_path, sample_output_df):
        """load_output works with DI pattern."""
        from src.data.output_manager import load_output, save_output, create_output_manager

        manager = create_output_manager(str(tmp_path))
        save_output(sample_output_df, 'di_test', manager=manager)

        df = load_output('di_test', manager=manager)
        assert len(df) == len(sample_output_df)

    def test_load_output_uses_singleton(self, tmp_path, sample_output_df):
        """load_output uses singleton when no manager provided."""
        from src.data import output_manager as om

        # Reset and setup singleton
        manager = om.create_output_manager(str(tmp_path))
        om._output_manager = manager

        # Save directly
        file_path = manager.local_output_dir / "singleton_load.parquet"
        sample_output_df.to_parquet(file_path)

        # Load via convenience function
        df = om.load_output('singleton_load')
        assert len(df) == len(sample_output_df)

        # Clean up
        om._output_manager = None

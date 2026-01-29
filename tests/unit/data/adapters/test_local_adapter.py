"""
Unit tests for src/data/adapters/local_adapter.py.

Tests validate LocalAdapter loads data from local filesystem correctly.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.adapters.local_adapter import LocalAdapter


@pytest.fixture
def local_data_dir(sample_sales_df, sample_wink_df, sample_market_weights_df, sample_macro_df):
    """Create local data directory structure for testing.

    Yields
    ------
    Path
        Path to temporary data directory with subdirectories
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir)

        # Create subdirectory structure
        (data_path / 'sales').mkdir()
        (data_path / 'rates').mkdir()
        (data_path / 'weights').mkdir()
        (data_path / 'macro').mkdir()

        # Write parquet files
        sample_sales_df.to_parquet(data_path / 'sales' / 'sales_data.parquet')
        sample_wink_df.to_parquet(data_path / 'rates' / 'rates_data.parquet')
        sample_market_weights_df.to_parquet(data_path / 'weights' / 'weights_data.parquet')
        sample_macro_df.to_parquet(data_path / 'macro' / 'macro_data.parquet')

        yield data_path


@pytest.fixture
def empty_local_dir():
    """Create empty local data directory.

    Yields
    ------
    Path
        Path to empty temporary directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestLocalAdapterInit:
    """Tests for LocalAdapter initialization."""

    def test_init_with_valid_directory(self, local_data_dir):
        """LocalAdapter should initialize with valid directory."""
        adapter = LocalAdapter(local_data_dir)

        assert adapter.data_dir == local_data_dir
        assert adapter.source_type == "local"

    def test_init_with_invalid_directory_raises(self):
        """LocalAdapter should raise ValueError for nonexistent directory."""
        with pytest.raises(ValueError, match="Data directory does not exist"):
            LocalAdapter(Path("/nonexistent/path"))

    def test_init_sets_default_output_dir(self, local_data_dir):
        """LocalAdapter should set default output dir."""
        adapter = LocalAdapter(local_data_dir)
        expected_output = local_data_dir / "outputs"
        assert adapter._output_dir == expected_output

    def test_init_with_custom_output_dir(self, local_data_dir):
        """LocalAdapter should accept custom output directory."""
        custom_output = Path("/tmp/custom_output")
        adapter = LocalAdapter(local_data_dir, output_dir=custom_output)
        assert adapter._output_dir == custom_output


class TestLocalAdapterSourceType:
    """Tests for source_type property."""

    def test_source_type_is_local(self, local_data_dir):
        """source_type should return 'local'."""
        adapter = LocalAdapter(local_data_dir)
        assert adapter.source_type == "local"


class TestLocalAdapterLoadSalesData:
    """Tests for load_sales_data method."""

    def test_load_sales_data_returns_dataframe(self, local_data_dir):
        """load_sales_data should return DataFrame."""
        adapter = LocalAdapter(local_data_dir)
        df = adapter.load_sales_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_sales_data_with_product_filter(self, local_data_dir):
        """load_sales_data should filter by product_name."""
        adapter = LocalAdapter(local_data_dir)
        df = adapter.load_sales_data(product_filter="FlexGuard indexed variable annuity")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_sales_data_missing_dir_raises(self, empty_local_dir):
        """load_sales_data should raise ValueError when subdir missing."""
        adapter = LocalAdapter(empty_local_dir)

        with pytest.raises(ValueError, match="Subdirectory not found"):
            adapter.load_sales_data()


class TestLocalAdapterLoadCompetitiveRates:
    """Tests for load_competitive_rates method."""

    def test_load_competitive_rates_returns_dataframe(self, local_data_dir):
        """load_competitive_rates should return DataFrame."""
        adapter = LocalAdapter(local_data_dir)
        df = adapter.load_competitive_rates(start_date="2022-01-01")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestLocalAdapterLoadMarketWeights:
    """Tests for load_market_weights method."""

    def test_load_market_weights_returns_dataframe(self, local_data_dir):
        """load_market_weights should return DataFrame."""
        adapter = LocalAdapter(local_data_dir)
        df = adapter.load_market_weights()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestLocalAdapterLoadMacroData:
    """Tests for load_macro_data method."""

    def test_load_macro_data_returns_dataframe(self, local_data_dir):
        """load_macro_data should return DataFrame."""
        adapter = LocalAdapter(local_data_dir)
        df = adapter.load_macro_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestLocalAdapterSaveOutput:
    """Tests for save_output method."""

    def test_save_output_parquet(self, local_data_dir, sample_sales_df):
        """save_output should save DataFrame as parquet."""
        adapter = LocalAdapter(local_data_dir)
        path = adapter.save_output(sample_sales_df, "test_output", format="parquet")

        assert Path(path).exists()
        assert path.endswith(".parquet")

        # Verify data roundtrip
        loaded = pd.read_parquet(path)
        pd.testing.assert_frame_equal(loaded, sample_sales_df)

    def test_save_output_csv(self, local_data_dir, sample_sales_df):
        """save_output should save DataFrame as CSV."""
        adapter = LocalAdapter(local_data_dir)
        path = adapter.save_output(sample_sales_df, "test_output", format="csv")

        assert Path(path).exists()
        assert path.endswith(".csv")

    def test_save_output_excel(self, local_data_dir, sample_sales_df):
        """save_output should save DataFrame as Excel."""
        pytest.importorskip("openpyxl")
        adapter = LocalAdapter(local_data_dir)
        path = adapter.save_output(sample_sales_df, "test_output", format="excel")

        assert Path(path).exists()
        assert path.endswith(".xlsx")

    def test_save_output_invalid_format_raises(self, local_data_dir, sample_sales_df):
        """save_output should raise ValueError for unsupported format."""
        adapter = LocalAdapter(local_data_dir)

        with pytest.raises(ValueError, match="Unsupported format"):
            adapter.save_output(sample_sales_df, "test", format="invalid")


class TestLocalAdapterProtocol:
    """Tests verifying LocalAdapter implements DataAdapterBase."""

    def test_implements_abstract_methods(self, local_data_dir):
        """LocalAdapter should implement all abstract methods."""
        adapter = LocalAdapter(local_data_dir)

        assert callable(adapter.load_sales_data)
        assert callable(adapter.load_competitive_rates)
        assert callable(adapter.load_market_weights)
        assert callable(adapter.load_macro_data)
        assert callable(adapter.save_output)
        assert hasattr(adapter, 'source_type')

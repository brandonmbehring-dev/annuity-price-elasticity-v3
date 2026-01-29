"""
Unit tests for src/data/adapters/fixture_adapter.py.

Tests validate the FixtureAdapter loads test data correctly
and handles error cases appropriately.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.adapters.fixture_adapter import FixtureAdapter


class TestFixtureAdapterInit:
    """Tests for FixtureAdapter initialization."""

    def test_init_with_valid_directory(self, temp_fixtures_dir):
        """FixtureAdapter should initialize with valid directory."""
        adapter = FixtureAdapter(temp_fixtures_dir)
        assert adapter.fixtures_dir == temp_fixtures_dir
        assert adapter.source_type == "fixture"

    def test_init_with_invalid_directory_raises(self):
        """FixtureAdapter should raise ValueError for nonexistent directory."""
        with pytest.raises(ValueError, match="Fixtures directory not found"):
            FixtureAdapter(Path("/nonexistent/path"))

    def test_init_sets_default_output_dir(self, temp_fixtures_dir):
        """FixtureAdapter should set default output dir."""
        adapter = FixtureAdapter(temp_fixtures_dir)
        expected_output = temp_fixtures_dir / "outputs"
        assert adapter._output_dir == expected_output

    def test_init_with_custom_output_dir(self, temp_fixtures_dir):
        """FixtureAdapter should accept custom output directory."""
        custom_output = Path("/tmp/custom_output")
        adapter = FixtureAdapter(temp_fixtures_dir, output_dir=custom_output)
        assert adapter._output_dir == custom_output


class TestFixtureAdapterSourceType:
    """Tests for source_type property."""

    def test_source_type_is_fixture(self, temp_fixtures_dir):
        """source_type should return 'fixture'."""
        adapter = FixtureAdapter(temp_fixtures_dir)
        assert adapter.source_type == "fixture"


class TestLoadSalesData:
    """Tests for load_sales_data method."""

    def test_load_sales_data_returns_dataframe(self, temp_fixtures_dir):
        """load_sales_data should return DataFrame."""
        adapter = FixtureAdapter(temp_fixtures_dir)
        df = adapter.load_sales_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_sales_data_with_product_filter(self, temp_fixtures_dir):
        """load_sales_data should filter by product_name."""
        adapter = FixtureAdapter(temp_fixtures_dir)
        df = adapter.load_sales_data(product_filter="FlexGuard indexed variable annuity")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_sales_data_missing_fixture_raises(self, empty_fixtures_dir):
        """load_sales_data should raise FileNotFoundError when no fixture."""
        adapter = FixtureAdapter(empty_fixtures_dir)

        with pytest.raises(FileNotFoundError, match="No fixture found"):
            adapter.load_sales_data()


class TestLoadCompetitiveRates:
    """Tests for load_competitive_rates method."""

    def test_load_competitive_rates_returns_dataframe(self, temp_fixtures_dir):
        """load_competitive_rates should return DataFrame."""
        adapter = FixtureAdapter(temp_fixtures_dir)
        df = adapter.load_competitive_rates(start_date="2022-01-01")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestLoadMarketWeights:
    """Tests for load_market_weights method."""

    def test_load_market_weights_returns_dataframe(self, temp_fixtures_dir):
        """load_market_weights should return DataFrame."""
        adapter = FixtureAdapter(temp_fixtures_dir)
        df = adapter.load_market_weights()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestLoadMacroData:
    """Tests for load_macro_data method."""

    def test_load_macro_data_returns_dataframe(self, temp_fixtures_dir):
        """load_macro_data should return DataFrame."""
        adapter = FixtureAdapter(temp_fixtures_dir)
        df = adapter.load_macro_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestSaveOutput:
    """Tests for save_output method."""

    def test_save_output_parquet(self, temp_fixtures_dir, sample_sales_df):
        """save_output should save DataFrame as parquet."""
        adapter = FixtureAdapter(temp_fixtures_dir)
        path = adapter.save_output(sample_sales_df, "test_output", format="parquet")

        assert Path(path).exists()
        assert path.endswith(".parquet")

        # Verify data roundtrip
        loaded = pd.read_parquet(path)
        pd.testing.assert_frame_equal(loaded, sample_sales_df)

    def test_save_output_csv(self, temp_fixtures_dir, sample_sales_df):
        """save_output should save DataFrame as CSV."""
        adapter = FixtureAdapter(temp_fixtures_dir)
        path = adapter.save_output(sample_sales_df, "test_output", format="csv")

        assert Path(path).exists()
        assert path.endswith(".csv")

    def test_save_output_invalid_format_raises(self, temp_fixtures_dir, sample_sales_df):
        """save_output should raise ValueError for unsupported format."""
        adapter = FixtureAdapter(temp_fixtures_dir)

        with pytest.raises(ValueError, match="Unsupported format"):
            adapter.save_output(sample_sales_df, "test", format="invalid")


class TestFindFixtureFile:
    """Tests for _find_fixture_file internal method."""

    def test_find_fixture_tries_multiple_names(self, temp_fixtures_dir):
        """_find_fixture_file should try multiple naming conventions."""
        adapter = FixtureAdapter(temp_fixtures_dir)

        # Should find raw_sales_data.parquet (one of the expected names)
        path = adapter._find_fixture_file("sales")
        assert path.exists()

    def test_find_fixture_not_found_raises(self, temp_fixtures_dir):
        """_find_fixture_file should raise for nonexistent type."""
        adapter = FixtureAdapter(temp_fixtures_dir)

        with pytest.raises(FileNotFoundError, match="No fixture found"):
            adapter._find_fixture_file("nonexistent_type")


class TestDataAdapterBaseProtocol:
    """Tests verifying FixtureAdapter implements DataAdapterBase."""

    def test_implements_abstract_methods(self, temp_fixtures_dir):
        """FixtureAdapter should implement all abstract methods."""
        adapter = FixtureAdapter(temp_fixtures_dir)

        # All these should be callable without NotImplementedError
        assert callable(adapter.load_sales_data)
        assert callable(adapter.load_competitive_rates)
        assert callable(adapter.load_market_weights)
        assert callable(adapter.load_macro_data)
        assert callable(adapter.save_output)
        assert hasattr(adapter, 'source_type')

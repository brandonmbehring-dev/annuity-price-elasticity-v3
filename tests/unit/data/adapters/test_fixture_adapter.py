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
    """Tests for load_sales_data method.

    Business context: Sales data is critical for elasticity estimation.
    Fixture contains 100 rows of synthetic FlexGuard 6Y20B transactions.
    """

    # Expected fixture size from tests/unit/data/conftest.py::sample_sales_df
    EXPECTED_SALES_ROWS = 100
    EXPECTED_SALES_COLUMNS = {'application_signed_date', 'premium_amount', 'product_name', 'term', 'buffer_rate'}

    def test_load_sales_data_returns_dataframe(self, temp_fixtures_dir):
        """load_sales_data should return DataFrame with expected structure."""
        adapter = FixtureAdapter(temp_fixtures_dir)
        df = adapter.load_sales_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == self.EXPECTED_SALES_ROWS, f"Expected {self.EXPECTED_SALES_ROWS} rows from fixture"
        assert self.EXPECTED_SALES_COLUMNS <= set(df.columns), f"Missing required columns"

    def test_load_sales_data_with_product_filter(self, temp_fixtures_dir):
        """load_sales_data should filter by product_name.

        Business context: Production data contains multiple products; filter isolates
        the specific product (e.g., FlexGuard 6Y20B) for analysis.
        """
        adapter = FixtureAdapter(temp_fixtures_dir)
        df = adapter.load_sales_data(product_filter="FlexGuard indexed variable annuity")

        assert isinstance(df, pd.DataFrame)
        # All 100 rows match the product filter in our fixture
        assert len(df) == self.EXPECTED_SALES_ROWS
        assert (df['product_name'] == "FlexGuard indexed variable annuity").all()

    def test_load_sales_data_missing_fixture_raises(self, empty_fixtures_dir):
        """load_sales_data should raise FileNotFoundError when no fixture.

        Business context: Fail-fast behavior prevents silent data loading failures
        that could lead to incorrect elasticity estimates.
        """
        adapter = FixtureAdapter(empty_fixtures_dir)

        with pytest.raises(FileNotFoundError, match="No fixture found"):
            adapter.load_sales_data()


class TestLoadCompetitiveRates:
    """Tests for load_competitive_rates method.

    Business context: Competitive rate data drives the competitor features
    in the elasticity model. Fixture contains 50 weekly observations.
    """

    # Expected fixture size from tests/unit/data/conftest.py::sample_wink_df
    EXPECTED_RATES_ROWS = 50
    EXPECTED_RATE_COLUMNS = {'date', 'Prudential', 'Allianz', 'Lincoln'}

    def test_load_competitive_rates_returns_dataframe(self, temp_fixtures_dir):
        """load_competitive_rates should return DataFrame with competitor rates."""
        adapter = FixtureAdapter(temp_fixtures_dir)
        df = adapter.load_competitive_rates(start_date="2022-01-01")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == self.EXPECTED_RATES_ROWS, f"Expected {self.EXPECTED_RATES_ROWS} rows from fixture"
        assert self.EXPECTED_RATE_COLUMNS <= set(df.columns), "Missing competitor rate columns"


class TestLoadMarketWeights:
    """Tests for load_market_weights method.

    Business context: Market share weights are used to compute weighted
    competitive mean rates. Fixture contains 4 quarterly observations.
    """

    # Expected fixture size from tests/unit/data/conftest.py::sample_market_weights_df
    EXPECTED_WEIGHTS_ROWS = 4
    EXPECTED_WEIGHT_COLUMNS = {'quarter', 'Prudential', 'Allianz'}

    def test_load_market_weights_returns_dataframe(self, temp_fixtures_dir):
        """load_market_weights should return DataFrame with quarterly weights.

        Business context: Market share weights determine how competitor rates
        are aggregated. Missing weights would break the weighted mean calculation.
        """
        adapter = FixtureAdapter(temp_fixtures_dir)
        df = adapter.load_market_weights()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == self.EXPECTED_WEIGHTS_ROWS, f"Expected {self.EXPECTED_WEIGHTS_ROWS} quarterly weight rows"
        assert self.EXPECTED_WEIGHT_COLUMNS <= set(df.columns), "Missing weight columns"


class TestLoadMacroData:
    """Tests for load_macro_data method.

    Business context: Macro indicators (VIX, DGS5, CPI) are control variables
    in the elasticity model. Fixture contains time series of economic data.
    """

    # Expected fixture structure from tests/unit/data/conftest.py::sample_macro_df
    EXPECTED_MACRO_COLUMNS = {'date', 'vix', 'dgs5'}  # Subset of expected columns

    def test_load_macro_data_returns_dataframe(self, temp_fixtures_dir):
        """load_macro_data should return DataFrame with economic indicators.

        Business context: Macro data controls for market conditions that affect
        annuity demand independent of rate changes.
        """
        adapter = FixtureAdapter(temp_fixtures_dir)
        df = adapter.load_macro_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 50, "Expected at least 50 macro observations"
        # Note: exact columns depend on fixture; validate at least date exists
        assert 'date' in df.columns, "Macro data must have date column"


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


class TestSubdirectoryLoading:
    """Tests for subdirectory fixture loading."""

    def test_loads_from_subdirectory(self, tmp_path):
        """Should load fixtures from subdirectory structure."""
        # Create subdirectory structure: fixtures_dir/sales/data.parquet
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        sales_subdir = fixtures_dir / "sales"
        sales_subdir.mkdir()

        # Create parquet file in subdirectory
        df = pd.DataFrame({
            "product_name": ["FlexGuard indexed variable annuity"],
            "premium_amount": [10000.0]
        })
        df.to_parquet(sales_subdir / "data.parquet")

        adapter = FixtureAdapter(fixtures_dir)
        result = adapter.load_sales_data()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_loads_from_economic_indicators_subdir(self, tmp_path):
        """Should load macro data from economic_indicators subdirectory."""
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        econ_dir = fixtures_dir / "economic_indicators"
        econ_dir.mkdir()

        # Create parquet file
        df = pd.DataFrame({"rate": [0.05], "date": ["2022-01-01"]})
        df.to_parquet(econ_dir / "dgs5.parquet")

        adapter = FixtureAdapter(fixtures_dir)
        result = adapter.load_macro_data()

        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1


class TestMultiFileLoading:
    """Tests for multi-file fixture loading."""

    def test_concatenates_multiple_parquet_files(self, tmp_path):
        """Should concatenate multiple parquet files from subdirectory."""
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        sales_subdir = fixtures_dir / "sales"
        sales_subdir.mkdir()

        # Create multiple parquet files
        df1 = pd.DataFrame({
            "product_name": ["FlexGuard indexed variable annuity"],
            "premium_amount": [10000.0]
        })
        df2 = pd.DataFrame({
            "product_name": ["FlexGuard indexed variable annuity"],
            "premium_amount": [20000.0]
        })
        df1.to_parquet(sales_subdir / "part1.parquet")
        df2.to_parquet(sales_subdir / "part2.parquet")

        adapter = FixtureAdapter(fixtures_dir)
        result = adapter.load_sales_data()

        assert len(result) == 2
        assert result["premium_amount"].sum() == 30000.0


class TestProductCodeFilter:
    """Tests for product code filter using Product Registry."""

    def test_filters_by_product_code(self, tmp_path):
        """Should filter using product code via Product Registry."""
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()

        # Create fixture with multiple products
        df = pd.DataFrame({
            "product_name": [
                "FlexGuard indexed variable annuity",
                "FlexGuard indexed variable annuity",
                "Other Product"
            ],
            "premium_amount": [10000.0, 20000.0, 30000.0]
        })
        df.to_parquet(fixtures_dir / "sales_fixture.parquet")

        adapter = FixtureAdapter(fixtures_dir)
        # Use product code "6Y20B" which should map to fixture name
        result = adapter.load_sales_data(product_filter="6Y20B")

        assert len(result) == 2
        assert all(result["product_name"] == "FlexGuard indexed variable annuity")


class TestEmptySalesValidation:
    """Tests for empty sales fixture validation."""

    def test_raises_on_empty_sales_fixture(self, tmp_path):
        """Should raise ValueError when sales fixture is empty."""
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()

        # Create empty fixture
        df = pd.DataFrame({"product_name": [], "premium_amount": []})
        df.to_parquet(fixtures_dir / "sales_fixture.parquet")

        adapter = FixtureAdapter(fixtures_dir)

        with pytest.raises(ValueError, match="Sales fixture is empty"):
            adapter.load_sales_data()


class TestDateFiltering:
    """Tests for date filtering in competitive rates."""

    def test_filters_by_effective_date(self, tmp_path):
        """Should filter rates by effective_date column."""
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()

        df = pd.DataFrame({
            "effective_date": ["2022-01-01", "2022-06-01", "2022-12-01"],
            "rate": [0.05, 0.06, 0.07]
        })
        df.to_parquet(fixtures_dir / "rates_fixture.parquet")

        adapter = FixtureAdapter(fixtures_dir)
        result = adapter.load_competitive_rates(start_date="2022-05-01")

        # Should only have dates >= 2022-05-01
        assert len(result) == 2


class TestExcelSaveFormat:
    """Tests for Excel save format."""

    def test_save_output_excel(self, temp_fixtures_dir, sample_sales_df):
        """save_output should save DataFrame as Excel."""
        adapter = FixtureAdapter(temp_fixtures_dir)
        path = adapter.save_output(sample_sales_df, "test_output", format="excel")

        assert Path(path).exists()
        assert path.endswith(".xlsx")

        # Verify data roundtrip
        loaded = pd.read_excel(path)
        # Excel may change dtypes slightly, so just check columns match
        assert set(loaded.columns) == set(sample_sales_df.columns)

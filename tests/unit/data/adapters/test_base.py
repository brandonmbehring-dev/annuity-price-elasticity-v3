"""
Unit tests for src/data/adapters/base.py.

Tests validate DataAdapterBase abstract class and protocol definition.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.adapters.base import DataAdapterBase


class ConcreteTestAdapter(DataAdapterBase):
    """Concrete implementation of DataAdapterBase for testing."""

    def __init__(self, test_data: pd.DataFrame):
        self._test_data = test_data
        self._source_type = "test"

    @property
    def source_type(self) -> str:
        return self._source_type

    def load_sales_data(self, product_filter=None):
        return self._test_data

    def load_competitive_rates(self, start_date):
        return self._test_data

    def load_market_weights(self):
        return self._test_data

    def load_macro_data(self):
        return self._test_data

    def save_output(self, df, name, format="parquet"):
        return f"/tmp/{name}.{format}"


class TestDataAdapterBaseProtocol:
    """Tests for DataAdapterBase protocol definition."""

    def test_protocol_requires_load_sales_data(self, sample_sales_df):
        """DataAdapterBase requires load_sales_data implementation."""
        adapter = ConcreteTestAdapter(sample_sales_df)
        result = adapter.load_sales_data()
        assert isinstance(result, pd.DataFrame)

    def test_protocol_requires_load_competitive_rates(self, sample_sales_df):
        """DataAdapterBase requires load_competitive_rates implementation."""
        adapter = ConcreteTestAdapter(sample_sales_df)
        result = adapter.load_competitive_rates("2022-01-01")
        assert isinstance(result, pd.DataFrame)

    def test_protocol_requires_load_market_weights(self, sample_sales_df):
        """DataAdapterBase requires load_market_weights implementation."""
        adapter = ConcreteTestAdapter(sample_sales_df)
        result = adapter.load_market_weights()
        assert isinstance(result, pd.DataFrame)

    def test_protocol_requires_load_macro_data(self, sample_sales_df):
        """DataAdapterBase requires load_macro_data implementation."""
        adapter = ConcreteTestAdapter(sample_sales_df)
        result = adapter.load_macro_data()
        assert isinstance(result, pd.DataFrame)

    def test_protocol_requires_save_output(self, sample_sales_df):
        """DataAdapterBase requires save_output implementation."""
        adapter = ConcreteTestAdapter(sample_sales_df)
        result = adapter.save_output(sample_sales_df, "test")
        assert isinstance(result, str)

    def test_protocol_requires_source_type(self, sample_sales_df):
        """DataAdapterBase requires source_type property."""
        adapter = ConcreteTestAdapter(sample_sales_df)
        assert adapter.source_type == "test"


class TestValidateDataframe:
    """Tests for _validate_dataframe method."""

    def test_validate_dataframe_success(self, sample_sales_df):
        """_validate_dataframe should pass for valid DataFrame."""
        adapter = ConcreteTestAdapter(sample_sales_df)

        # Should not raise
        adapter._validate_dataframe(
            sample_sales_df,
            required_columns=['application_signed_date', 'premium_amount'],
            data_name="test_data"
        )

    def test_validate_dataframe_empty_raises(self, sample_sales_df):
        """_validate_dataframe should raise for empty DataFrame."""
        adapter = ConcreteTestAdapter(sample_sales_df)
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="is empty"):
            adapter._validate_dataframe(
                empty_df,
                required_columns=['col1'],
                data_name="test_data"
            )

    def test_validate_dataframe_missing_columns_raises(self, sample_sales_df):
        """_validate_dataframe should raise for missing columns."""
        adapter = ConcreteTestAdapter(sample_sales_df)

        with pytest.raises(ValueError, match="missing required columns"):
            adapter._validate_dataframe(
                sample_sales_df,
                required_columns=['nonexistent_column'],
                data_name="test_data"
            )


class TestSaveHelpers:
    """Tests for save helper methods."""

    def test_save_parquet(self, sample_sales_df):
        """_save_parquet should save DataFrame as parquet."""
        adapter = ConcreteTestAdapter(sample_sales_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            result = adapter._save_parquet(sample_sales_df, path)

            assert Path(result).exists()
            loaded = pd.read_parquet(result)
            assert len(loaded) == len(sample_sales_df)

    def test_save_csv(self, sample_sales_df):
        """_save_csv should save DataFrame as CSV."""
        adapter = ConcreteTestAdapter(sample_sales_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            result = adapter._save_csv(sample_sales_df, path)

            assert Path(result).exists()
            loaded = pd.read_csv(result)
            assert len(loaded) == len(sample_sales_df)

    def test_save_creates_parent_dirs(self, sample_sales_df):
        """Save methods should create parent directories."""
        adapter = ConcreteTestAdapter(sample_sales_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "test.parquet"
            result = adapter._save_parquet(sample_sales_df, path)

            assert Path(result).exists()


class TestAbstractMethodEnforcement:
    """Tests for abstract method enforcement."""

    def test_cannot_instantiate_base_class(self):
        """DataAdapterBase should not be directly instantiable."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            DataAdapterBase()

    def test_incomplete_implementation_raises(self):
        """Incomplete implementation should raise TypeError."""
        class IncompleteAdapter(DataAdapterBase):
            @property
            def source_type(self):
                return "incomplete"

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteAdapter()

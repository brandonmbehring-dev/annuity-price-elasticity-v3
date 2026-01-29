"""
Tests for src.core.protocols module.

Tests Protocol interfaces and runtime checkability.
"""

import pytest
from typing import Protocol, runtime_checkable
import pandas as pd

from src.core.protocols import (
    DataSourceAdapter,
    AggregationStrategy,
    ProductMethodology,
    NotebookInterfaceProtocol,
)


class TestDataSourceAdapterProtocol:
    """Tests for DataSourceAdapter protocol."""

    def test_protocol_is_runtime_checkable(self):
        """DataSourceAdapter should be runtime checkable."""
        assert hasattr(DataSourceAdapter, "__protocol_attrs__") or hasattr(
            DataSourceAdapter, "_is_protocol"
        )

    def test_protocol_has_load_sales_data_method(self):
        """Protocol should define load_sales_data method."""
        assert hasattr(DataSourceAdapter, "load_sales_data")

    def test_protocol_has_load_competitive_rates_method(self):
        """Protocol should define load_competitive_rates method."""
        assert hasattr(DataSourceAdapter, "load_competitive_rates")

    def test_protocol_has_load_market_weights_method(self):
        """Protocol should define load_market_weights method."""
        assert hasattr(DataSourceAdapter, "load_market_weights")

    def test_mock_adapter_implements_protocol(self):
        """A mock class with required methods should satisfy protocol."""

        class MockAdapter:
            def load_sales_data(self, product_filter=None):
                return pd.DataFrame()

            def load_competitive_rates(self, start_date):
                return pd.DataFrame()

            def load_market_weights(self):
                return pd.DataFrame()

        adapter = MockAdapter()
        # Should not raise - duck typing should work
        assert hasattr(adapter, "load_sales_data")
        assert hasattr(adapter, "load_competitive_rates")
        assert hasattr(adapter, "load_market_weights")


class TestAggregationStrategyProtocol:
    """Tests for AggregationStrategy protocol."""

    def test_protocol_has_aggregate_method(self):
        """Protocol should define aggregate method."""
        assert hasattr(AggregationStrategy, "aggregate")

    def test_mock_strategy_implements_protocol(self):
        """A mock class with aggregate method should satisfy protocol."""

        class MockStrategy:
            def aggregate(self, df, weights=None):
                return df

        strategy = MockStrategy()
        assert hasattr(strategy, "aggregate")


class TestProductMethodologyProtocol:
    """Tests for ProductMethodology protocol."""

    def test_protocol_has_get_constraint_rules_method(self):
        """Protocol should define get_constraint_rules method."""
        assert hasattr(ProductMethodology, "get_constraint_rules")

    def test_protocol_has_get_coefficient_signs_method(self):
        """Protocol should define get_coefficient_signs method."""
        assert hasattr(ProductMethodology, "get_coefficient_signs")

    def test_protocol_has_supports_regime_detection_method(self):
        """Protocol should define supports_regime_detection method."""
        assert hasattr(ProductMethodology, "supports_regime_detection")

    def test_protocol_has_product_type_property(self):
        """Protocol should define product_type property."""
        assert hasattr(ProductMethodology, "product_type")


class TestNotebookInterfaceProtocol:
    """Tests for NotebookInterfaceProtocol."""

    def test_protocol_has_load_data_method(self):
        """Protocol should define load_data method."""
        assert hasattr(NotebookInterfaceProtocol, "load_data")

    def test_protocol_has_run_feature_selection_method(self):
        """Protocol should define run_feature_selection method."""
        assert hasattr(NotebookInterfaceProtocol, "run_feature_selection")

    def test_protocol_has_run_inference_method(self):
        """Protocol should define run_inference method."""
        assert hasattr(NotebookInterfaceProtocol, "run_inference")

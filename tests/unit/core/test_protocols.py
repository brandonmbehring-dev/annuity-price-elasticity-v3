"""
Tests for src.core.protocols module.

These tests validate Protocol interfaces through REAL implementations,
not just hasattr checks. Tests verify that concrete implementations:
1. Satisfy protocol contracts
2. Return correct types
3. Handle edge cases appropriately

Following test quality guidelines:
- No hasattr-only tests (those are tautological)
- Real data via fixtures
- Validate actual behavior, not duck-typing
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.core.protocols import (
    DataSourceAdapter,
    AggregationStrategy,
    ProductMethodology,
    NotebookInterfaceProtocol,
)
from src.data.adapters import FixtureAdapter
from src.features.aggregation.strategies import (
    WeightedAggregation,
    TopNAggregation,
    MedianAggregation,
)
from src.products.rila_methodology import RILAMethodology


# =============================================================================
# FIXTURE ADAPTER PROTOCOL COMPLIANCE TESTS
# =============================================================================


class TestDataSourceAdapterWithFixtureAdapter:
    """Tests that FixtureAdapter correctly implements DataSourceAdapter protocol."""

    def test_fixture_adapter_loads_sales_data(self, fixtures_dir: Path):
        """FixtureAdapter.load_sales_data returns DataFrame with required columns."""
        adapter = FixtureAdapter(fixtures_dir)
        df = adapter.load_sales_data()

        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        assert len(df) > 0, "Should not be empty"
        # Validate actual data presence (not just column existence)
        assert df.shape[0] >= 10, f"Expected >10 rows, got {df.shape[0]}"

    def test_fixture_adapter_loads_competitive_rates(self, fixtures_dir: Path):
        """FixtureAdapter.load_competitive_rates returns rate data with date column."""
        adapter = FixtureAdapter(fixtures_dir)
        df = adapter.load_competitive_rates(start_date="2020-01-01")

        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        assert len(df) > 0, "Rates data should not be empty"
        # WINK rates have date column
        assert "date" in df.columns or "effective_date" in df.columns, (
            "Missing date column in competitive rates"
        )

    def test_fixture_adapter_loads_market_weights(self, fixtures_dir: Path):
        """FixtureAdapter.load_market_weights returns weights for aggregation."""
        adapter = FixtureAdapter(fixtures_dir)
        df = adapter.load_market_weights()

        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        assert len(df) > 0, "Weights should not be empty"
        # Weights should have numeric values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 0, "Weights should have numeric columns"

    def test_fixture_adapter_source_type_property(self, fixtures_dir: Path):
        """FixtureAdapter.source_type returns 'fixture'."""
        adapter = FixtureAdapter(fixtures_dir)

        assert adapter.source_type == "fixture", (
            f"Expected 'fixture', got '{adapter.source_type}'"
        )

    def test_fixture_adapter_satisfies_protocol(self, fixtures_dir: Path):
        """FixtureAdapter instance passes isinstance check with protocol."""
        adapter = FixtureAdapter(fixtures_dir)

        # This is the MEANINGFUL test - actual runtime protocol check
        assert isinstance(adapter, DataSourceAdapter), (
            "FixtureAdapter should satisfy DataSourceAdapter protocol"
        )

    def test_fixture_adapter_save_output(self, fixtures_dir: Path, tmp_path: Path):
        """FixtureAdapter.save_output writes parquet file and returns path."""
        adapter = FixtureAdapter(fixtures_dir, output_dir=tmp_path)

        test_df = pd.DataFrame({
            "value": [1, 2, 3],
            "name": ["a", "b", "c"]
        })

        result_path = adapter.save_output(test_df, "test_output", format="parquet")

        assert isinstance(result_path, str), "Should return string path"
        assert Path(result_path).exists(), "Output file should exist"
        # Verify content roundtrips correctly
        loaded = pd.read_parquet(result_path)
        pd.testing.assert_frame_equal(loaded, test_df)

    def test_fixture_adapter_missing_directory_raises(self):
        """FixtureAdapter raises ValueError for non-existent directory."""
        fake_path = Path("/nonexistent/path/to/fixtures")

        with pytest.raises(ValueError) as exc_info:
            FixtureAdapter(fake_path)

        assert "not found" in str(exc_info.value).lower()


# =============================================================================
# AGGREGATION STRATEGY PROTOCOL COMPLIANCE TESTS
# =============================================================================


class TestAggregationStrategyWithRealImplementations:
    """Tests that aggregation strategies correctly implement protocol."""

    @pytest.fixture
    def sample_rates_df(self) -> pd.DataFrame:
        """Create sample rates data for aggregation tests."""
        np.random.seed(42)
        return pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="W"),
            "CompanyA": np.random.uniform(0.08, 0.12, 10),
            "CompanyB": np.random.uniform(0.07, 0.11, 10),
            "CompanyC": np.random.uniform(0.09, 0.13, 10),
            "CompanyD": np.random.uniform(0.06, 0.10, 10),
        })

    @pytest.fixture
    def sample_weights_df(self) -> pd.DataFrame:
        """Create sample market weights for weighted aggregation."""
        return pd.DataFrame({
            "company": ["CompanyA", "CompanyB", "CompanyC", "CompanyD"],
            "market_share": [0.35, 0.25, 0.25, 0.15],
        })

    def test_weighted_aggregation_computes_weighted_mean(
        self, sample_rates_df: pd.DataFrame, sample_weights_df: pd.DataFrame
    ):
        """WeightedAggregation returns weighted mean of competitor rates."""
        strategy = WeightedAggregation()
        company_cols = ["CompanyA", "CompanyB", "CompanyC", "CompanyD"]

        result = strategy.aggregate(sample_rates_df, company_cols, sample_weights_df)

        assert isinstance(result, pd.Series), "Should return Series"
        assert len(result) == len(sample_rates_df), "Should match input length"
        assert result.notna().all(), "Should not have NaN values"
        # Weighted mean should be between min and max of inputs
        row_mins = sample_rates_df[company_cols].min(axis=1)
        row_maxs = sample_rates_df[company_cols].max(axis=1)
        assert (result >= row_mins).all(), "Weighted mean should be >= min"
        assert (result <= row_maxs).all(), "Weighted mean should be <= max"

    def test_weighted_aggregation_requires_weights_property(self):
        """WeightedAggregation.requires_weights returns True."""
        strategy = WeightedAggregation()
        assert strategy.requires_weights is True

    def test_weighted_aggregation_strategy_name(self):
        """WeightedAggregation.strategy_name returns 'weighted'."""
        strategy = WeightedAggregation()
        assert strategy.strategy_name == "weighted"

    def test_top_n_aggregation_computes_top_n_mean(
        self, sample_rates_df: pd.DataFrame
    ):
        """TopNAggregation returns mean of top N competitor rates."""
        strategy = TopNAggregation(n_competitors=2)
        company_cols = ["CompanyA", "CompanyB", "CompanyC", "CompanyD"]

        result = strategy.aggregate(sample_rates_df, company_cols)

        assert isinstance(result, pd.Series), "Should return Series"
        assert len(result) == len(sample_rates_df)
        # Top-2 mean should be higher than simple mean
        simple_mean = sample_rates_df[company_cols].mean(axis=1)
        assert (result >= simple_mean).all(), "Top-N mean should be >= simple mean"

    def test_top_n_aggregation_does_not_require_weights(self):
        """TopNAggregation.requires_weights returns False."""
        strategy = TopNAggregation()
        assert strategy.requires_weights is False

    def test_median_aggregation_computes_median(self, sample_rates_df: pd.DataFrame):
        """MedianAggregation returns median of competitor rates."""
        strategy = MedianAggregation()
        company_cols = ["CompanyA", "CompanyB", "CompanyC", "CompanyD"]

        result = strategy.aggregate(sample_rates_df, company_cols)

        expected = sample_rates_df[company_cols].median(axis=1)
        pd.testing.assert_series_equal(result, expected)

    def test_aggregation_strategies_satisfy_protocol(self):
        """All aggregation strategies pass isinstance check with protocol."""
        weighted = WeightedAggregation()
        top_n = TopNAggregation()
        median = MedianAggregation()

        assert isinstance(weighted, AggregationStrategy)
        assert isinstance(top_n, AggregationStrategy)
        assert isinstance(median, AggregationStrategy)


# =============================================================================
# PRODUCT METHODOLOGY PROTOCOL COMPLIANCE TESTS
# =============================================================================


class TestProductMethodologyWithRILA:
    """Tests that RILAMethodology correctly implements ProductMethodology protocol."""

    def test_rila_get_constraint_rules_returns_list(self):
        """RILAMethodology.get_constraint_rules returns non-empty list."""
        methodology = RILAMethodology()
        rules = methodology.get_constraint_rules()

        assert isinstance(rules, list), "Should return list"
        assert len(rules) >= 3, "Should have multiple constraint rules"

    def test_rila_constraint_rules_have_required_fields(self):
        """Each constraint rule has required fields for validation."""
        methodology = RILAMethodology()
        rules = methodology.get_constraint_rules()

        required_fields = {"feature_pattern", "expected_sign", "constraint_type"}

        for rule in rules:
            for field in required_fields:
                assert hasattr(rule, field), f"Rule missing field: {field}"

    def test_rila_has_own_rate_positive_constraint(self):
        """RILA methodology includes positive own rate constraint."""
        methodology = RILAMethodology()
        rules = methodology.get_constraint_rules()

        own_rate_rules = [
            r for r in rules
            if r.constraint_type == "OWN_RATE_POSITIVE"
        ]

        assert len(own_rate_rules) >= 1, "Should have own rate positive constraint"
        assert own_rate_rules[0].expected_sign == "positive"

    def test_rila_has_competitor_negative_constraint(self):
        """RILA methodology includes negative competitor constraint."""
        methodology = RILAMethodology()
        rules = methodology.get_constraint_rules()

        competitor_rules = [
            r for r in rules
            if r.constraint_type == "COMPETITOR_NEGATIVE"
        ]

        assert len(competitor_rules) >= 1, "Should have competitor negative constraint"
        assert competitor_rules[0].expected_sign == "negative"

    def test_rila_forbids_lag_zero_competitors(self):
        """RILA methodology forbids lag-0 competitor features (leakage prevention)."""
        methodology = RILAMethodology()
        rules = methodology.get_constraint_rules()

        lag_zero_rules = [
            r for r in rules
            if r.constraint_type == "NO_LAG_ZERO_COMPETITOR"
        ]

        assert len(lag_zero_rules) >= 1, "Should have lag-0 forbidden rule"
        assert lag_zero_rules[0].expected_sign == "forbidden"

    def test_rila_get_coefficient_signs_returns_dict(self):
        """RILAMethodology.get_coefficient_signs returns pattern mapping."""
        methodology = RILAMethodology()
        signs = methodology.get_coefficient_signs()

        assert isinstance(signs, dict), "Should return dict"
        assert len(signs) >= 2, "Should have multiple patterns"
        # Verify key patterns exist
        assert any("prudential" in k.lower() for k in signs.keys())
        assert any("competitor" in k.lower() for k in signs.keys())

    def test_rila_product_type_property(self):
        """RILAMethodology.product_type returns 'rila'."""
        methodology = RILAMethodology()
        assert methodology.product_type == "rila"

    def test_rila_supports_regime_detection(self):
        """RILAMethodology.supports_regime_detection returns boolean."""
        methodology = RILAMethodology()
        result = methodology.supports_regime_detection()

        assert isinstance(result, bool), "Should return boolean"
        # RILA doesn't use regime detection (yield-based)
        assert result is False

    def test_rila_satisfies_protocol(self):
        """RILAMethodology instance passes isinstance check with protocol."""
        methodology = RILAMethodology()
        assert isinstance(methodology, ProductMethodology)


# =============================================================================
# NOTEBOOK INTERFACE PROTOCOL TESTS
# =============================================================================


class TestNotebookInterfaceProtocol:
    """Tests for NotebookInterfaceProtocol."""

    def test_protocol_defines_load_data_method(self):
        """Protocol defines load_data method signature."""
        # Create a conforming class to validate protocol
        class ConformingInterface:
            def load_data(self) -> pd.DataFrame:
                return pd.DataFrame()

            def run_feature_selection(self, data, config=None):
                return None

            def run_inference(self, data, config=None):
                return None

            def export_results(self, results, format="excel") -> str:
                return "/path"

        interface = ConformingInterface()
        assert isinstance(interface, NotebookInterfaceProtocol)

    def test_non_conforming_class_fails_protocol(self):
        """Class missing required methods does not satisfy protocol."""
        class NonConformingInterface:
            def load_data(self) -> pd.DataFrame:
                return pd.DataFrame()
            # Missing: run_feature_selection, run_inference, export_results

        interface = NonConformingInterface()
        assert not isinstance(interface, NotebookInterfaceProtocol)


# =============================================================================
# MODULE EXPORT TESTS
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_protocols_exported(self):
        """Test that all protocols are in __all__."""
        from src.core import protocols

        assert "DataSourceAdapter" in protocols.__all__
        assert "AggregationStrategy" in protocols.__all__
        assert "ProductMethodology" in protocols.__all__
        assert "NotebookInterfaceProtocol" in protocols.__all__

    def test_all_has_correct_length(self):
        """Test that __all__ has exactly 4 protocols."""
        from src.core.protocols import __all__

        assert len(__all__) == 4

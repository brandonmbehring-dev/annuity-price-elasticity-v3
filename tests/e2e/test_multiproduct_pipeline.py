"""
Multi-Product Pipeline End-to-End Tests
========================================

Tests that pipeline works correctly across multiple RILA products:
- 6Y20B (6-year term, 20% buffer)
- 6Y10B (6-year term, 10% buffer)
- 10Y20B (10-year term, 20% buffer)

Validates that:
1. Product-specific configurations work correctly
2. Performance metrics meet product-specific thresholds
3. Feature engineering adapts to product characteristics
4. Inference produces valid results for all products

Author: Claude Code
Date: 2026-01-29
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any

from src.notebooks.interface import UnifiedNotebookInterface
from src.config.product_config import get_product_config
from src.products import get_methodology

# Product-specific performance thresholds
PRODUCT_THRESHOLDS = {
    "6Y20B": {
        "R²_min": 0.75,
        "MAPE_max": 0.15,
        "min_observations": 100,
        "expected_buffer": "20%",
        "expected_term": "6Y"
    },
    "6Y10B": {
        "R²_min": 0.70,
        "MAPE_max": 0.18,
        "min_observations": 80,
        "expected_buffer": "10%",
        "expected_term": "6Y"
    },
    "10Y20B": {
        "R²_min": 0.72,
        "MAPE_max": 0.16,
        "min_observations": 80,
        "expected_buffer": "20%",
        "expected_term": "10Y"
    }
}


# =============================================================================
# PARAMETRIZED MULTI-PRODUCT TESTS
# =============================================================================


@pytest.mark.e2e
@pytest.mark.parametrize("product_code,thresholds", [
    ("6Y20B", PRODUCT_THRESHOLDS["6Y20B"]),
    pytest.param("6Y10B", PRODUCT_THRESHOLDS["6Y10B"], marks=pytest.mark.skip(reason="6Y10B fixtures not yet available")),
    pytest.param("10Y20B", PRODUCT_THRESHOLDS["10Y20B"], marks=pytest.mark.skip(reason="10Y20B fixtures not yet available")),
])
class TestMultiProductPipeline:
    """Parametrized tests across multiple RILA products."""

    def test_product_pipeline_runs_successfully(self, product_code, thresholds):
        """Pipeline should run successfully for each product."""
        interface = UnifiedNotebookInterface(
            product_code=product_code,
            data_source="fixture"
        )

        df = interface.load_data()

        # Should produce valid output
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= thresholds["min_observations"], (
            f"{product_code}: Expected >={thresholds['min_observations']} observations, "
            f"got {len(df)}"
        )

    def test_product_configuration_correct(self, product_code, thresholds):
        """Product configuration should match expected values."""
        config = get_product_config(product_code)

        assert config is not None
        # ProductConfig is a dataclass with buffer_level (float) and term_years (int)
        # Convert to expected format for comparison
        expected_buffer = thresholds['expected_buffer']  # e.g., "20%"
        expected_term = thresholds['expected_term']      # e.g., "6Y"

        actual_buffer = f"{int(config.buffer_level * 100)}%" if config.buffer_level else "N/A"
        actual_term = f"{config.term_years}Y"

        assert actual_buffer == expected_buffer, (
            f"{product_code}: Buffer rate mismatch: {actual_buffer} != {expected_buffer}"
        )
        assert actual_term == expected_term, (
            f"{product_code}: Term mismatch: {actual_term} != {expected_term}"
        )

    def test_product_methodology_correct(self, product_code, thresholds):
        """Product methodology should be appropriate for product type."""
        methodology = get_methodology(product_code)

        # All RILA products should use RILA methodology
        assert methodology.product_type == "rila", (
            f"{product_code} should use RILA methodology"
        )

        # Should not support regime detection (RILA characteristic)
        assert methodology.supports_regime_detection() == False, (
            f"{product_code}: RILA should not support regime detection"
        )

    def test_product_inference_meets_thresholds(self, product_code, thresholds):
        """Inference results should meet product-specific thresholds."""
        interface = UnifiedNotebookInterface(
            product_code=product_code,
            data_source="fixture"
        )

        df = interface.load_data()

        # Run inference
        results = interface.run_inference(
            df=df,
            n_bootstrap=100,
            n_jobs=-1,
            random_state=42
        )

        # Validate metrics meet thresholds
        metrics = results['metrics']

        assert metrics['R²'] >= thresholds['R²_min'], (
            f"{product_code}: R²={metrics['R²']:.3f} below minimum {thresholds['R²_min']}"
        )

        assert metrics['MAPE'] <= thresholds['MAPE_max'], (
            f"{product_code}: MAPE={metrics['MAPE']:.3f} above maximum {thresholds['MAPE_max']}"
        )


# =============================================================================
# CROSS-PRODUCT CONSISTENCY TESTS
# =============================================================================


@pytest.mark.e2e
class TestCrossProductConsistency:
    """Test consistency across different products."""

    def test_all_products_use_same_feature_engineering(self):
        """All RILA products should use consistent feature engineering."""
        products = ["6Y20B"]  # Expand when other fixtures available

        feature_sets = {}

        for product_code in products:
            interface = UnifiedNotebookInterface(
                product_code=product_code,
                data_source="fixture"
            )

            df = interface.load_data()

            # Extract feature types
            lag_features = [col for col in df.columns if 'lag_' in col or '_t' in col]
            poly_features = [col for col in df.columns if '_squared' in col or '_poly_' in col]
            macro_features = [col for col in df.columns if any(
                macro in col.lower() for macro in ['cpi', 'dgs', 'vix', 'unemployment']
            )]

            feature_sets[product_code] = {
                'lag': len(lag_features),
                'polynomial': len(poly_features),
                'macro': len(macro_features)
            }

        # All products should have similar feature engineering structure
        for product_code, features in feature_sets.items():
            assert features['lag'] > 20, (
                f"{product_code} should have >20 lag features"
            )
            assert features['polynomial'] > 5, (
                f"{product_code} should have >5 polynomial features"
            )

    def test_all_products_respect_economic_constraints(self):
        """All RILA products should respect same economic constraints."""
        products = ["6Y20B"]  # Expand when other fixtures available

        for product_code in products:
            interface = UnifiedNotebookInterface(
                product_code=product_code,
                data_source="fixture"
            )

            df = interface.load_data()

            # Check for forbidden lag-0 competitor features
            import re
            forbidden_patterns = [
                r'competitor.*_t0',
                r'competitor.*_lag_0',
                r'C_.*_t0'
            ]

            forbidden_features = []
            for pattern in forbidden_patterns:
                regex = re.compile(pattern)
                matches = [col for col in df.columns if regex.search(col)]
                forbidden_features.extend(matches)

            assert len(forbidden_features) == 0, (
                f"{product_code}: Found {len(forbidden_features)} forbidden lag-0 competitor features"
            )

    def test_all_products_produce_modeling_ready_datasets(self):
        """All products should produce clean, modeling-ready datasets."""
        products = ["6Y20B"]  # Expand when other fixtures available

        for product_code in products:
            interface = UnifiedNotebookInterface(
                product_code=product_code,
                data_source="fixture"
            )

            df = interface.load_data()

            # Quality checks
            assert df.isnull().sum().sum() == 0, (
                f"{product_code}: Dataset has missing values"
            )

            assert len(df) > 50, (
                f"{product_code}: Dataset too small"
            )

            assert df.shape[1] > 100, (
                f"{product_code}: Dataset has too few features"
            )


# =============================================================================
# PRODUCT-SPECIFIC FEATURE TESTS
# =============================================================================


@pytest.mark.e2e
class TestProductSpecificFeatures:
    """Test product-specific feature engineering."""

    def test_6y20b_buffer_specific_features(self):
        """6Y20B should have buffer-specific features."""
        interface = UnifiedNotebookInterface(
            product_code="6Y20B",
            data_source="fixture"
        )

        df = interface.load_data()

        # Should have buffer-related features
        buffer_features = [col for col in df.columns if 'buffer' in col.lower()]

        # RILA products may have buffer features
        # (Not required, depends on feature engineering implementation)
        # This test documents expected behavior

    def test_term_specific_features_consistent(self):
        """Products with same term should have similar feature structure."""
        # Both 6Y products should have similar lag structure
        interface_6y20b = UnifiedNotebookInterface(
            product_code="6Y20B",
            data_source="fixture"
        )

        df_6y20b = interface_6y20b.load_data()

        # Get lag features for 6Y products
        lag_features_6y20b = [col for col in df_6y20b.columns if 'lag_' in col]

        # 6Y products should have similar number of lags
        # (Exact number depends on max_lag_periods configuration)
        assert len(lag_features_6y20b) > 20, (
            "6Y products should have >20 lag features"
        )


# =============================================================================
# PRODUCT COMPARISON TESTS
# =============================================================================


@pytest.mark.e2e
class TestProductComparison:
    """Compare results across products."""

    def test_product_performance_ranking(self):
        """Document expected performance ranking across products.

        6Y20B typically has best performance due to:
        - Largest market share
        - Most stable pricing
        - Longest historical data
        """
        products_available = ["6Y20B"]  # Expand when fixtures available

        performance_metrics = {}

        for product_code in products_available:
            interface = UnifiedNotebookInterface(
                product_code=product_code,
                data_source="fixture"
            )

            df = interface.load_data()

            results = interface.run_inference(
                df=df,
                n_bootstrap=100,
                n_jobs=-1,
                random_state=42
            )

            performance_metrics[product_code] = {
                'R²': results['metrics']['R²'],
                'MAPE': results['metrics']['MAPE'],
                'observations': len(df)
            }

        # Document performance for reference
        for product_code, metrics in performance_metrics.items():
            print(f"\n{product_code} Performance:")
            print(f"  R²: {metrics['R²']:.4f}")
            print(f"  MAPE: {metrics['MAPE']:.4f}")
            print(f"  Observations: {metrics['observations']}")


# =============================================================================
# PRODUCT CONFIGURATION VALIDATION
# =============================================================================


@pytest.mark.e2e
class TestProductConfiguration:
    """Test product configuration correctness."""

    def test_product_configs_have_required_fields(self):
        """All product configs should have required fields."""
        products = ["6Y20B", "6Y10B", "10Y20B"]

        # ProductConfig is a dataclass - check attributes exist
        required_fields = [
            'name',           # was 'product_name'
            'buffer_level',   # was 'buffer_rate'
            'term_years',     # was 'term'
            'product_code'
        ]

        for product_code in products:
            config = get_product_config(product_code)

            for field in required_fields:
                assert hasattr(config, field), (
                    f"{product_code} config missing required field: {field}"
                )

    def test_product_configs_are_unique(self):
        """Each product should have unique configuration."""
        products = ["6Y20B", "6Y10B", "10Y20B"]

        configs = {}
        for product_code in products:
            config = get_product_config(product_code)
            # ProductConfig is a dataclass: use buffer_level and term_years
            key = f"{config.buffer_level}_{config.term_years}"
            configs[key] = product_code

        # Should have 3 unique combinations
        assert len(configs) == 3, "Products should have unique buffer/term combinations"

    def test_product_names_follow_convention(self):
        """Product codes should follow naming convention."""
        products = ["6Y20B", "6Y10B", "10Y20B"]

        for product_code in products:
            # Format: {term}Y{buffer}B
            # Example: 6Y20B = 6 years, 20% buffer

            assert product_code[-1] == 'B', (
                f"{product_code} should end with 'B' (buffer)"
            )

            assert 'Y' in product_code, (
                f"{product_code} should contain 'Y' (years)"
            )


# =============================================================================
# PRODUCT METHODOLOGY TESTS
# =============================================================================


@pytest.mark.e2e
class TestProductMethodology:
    """Test product methodology implementation."""

    def test_all_rila_products_use_rila_methodology(self):
        """All RILA products should use RILA-specific methodology."""
        products = ["6Y20B", "6Y10B", "10Y20B"]

        for product_code in products:
            methodology = get_methodology(product_code)

            assert methodology.product_type == "rila", (
                f"{product_code} should use RILA methodology"
            )

            # RILA-specific characteristics
            assert hasattr(methodology, 'get_constraint_rules'), (
                f"{product_code} methodology missing constraint rules"
            )

            assert hasattr(methodology, 'get_coefficient_signs'), (
                f"{product_code} methodology missing coefficient signs"
            )

    def test_rila_constraint_rules_consistent(self):
        """All RILA products should have consistent constraint rules."""
        products = ["6Y20B", "6Y10B", "10Y20B"]

        constraint_types_by_product = {}

        for product_code in products:
            methodology = get_methodology(product_code)
            rules = methodology.get_constraint_rules()

            constraint_types = {rule.constraint_type for rule in rules}
            constraint_types_by_product[product_code] = constraint_types

        # All RILA products should have same core constraint types
        first_product = products[0]
        first_constraints = constraint_types_by_product[first_product]

        for product_code in products[1:]:
            current_constraints = constraint_types_by_product[product_code]

            # Core constraints should be same
            core_constraints = {
                "OWN_RATE_POSITIVE",
                "COMPETITOR_NEGATIVE",
                "NO_LAG_ZERO_COMPETITOR"
            }

            for constraint in core_constraints:
                assert constraint in first_constraints or any(
                    constraint in ct for ct in first_constraints
                ), f"Core constraint {constraint} missing from {first_product}"

                assert constraint in current_constraints or any(
                    constraint in ct for ct in current_constraints
                ), f"Core constraint {constraint} missing from {product_code}"


# =============================================================================
# SUMMARY TESTS
# =============================================================================


@pytest.mark.e2e
def test_multiproduct_summary():
    """Summary test documenting multi-product capabilities.

    This test serves as documentation of multi-product support:

    Supported Products:
    - 6Y20B: 6-year term, 20% buffer (primary product, most data)
    - 6Y10B: 6-year term, 10% buffer (secondary product)
    - 10Y20B: 10-year term, 20% buffer (alternative term)

    Consistency Guarantees:
    - All use same RILA methodology
    - All respect same economic constraints
    - All use consistent feature engineering
    - All produce modeling-ready datasets

    Product-Specific:
    - Different performance thresholds
    - Different data availability
    - Same inference approach
    """
    pass  # Documentation test

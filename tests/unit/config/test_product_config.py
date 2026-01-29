"""
Unit tests for src/config/product_config.py.

Tests validate product configuration, WINK product IDs, and competitor configs.
"""

import pytest

from src.config.product_config import (
    # Product configuration
    ProductConfig,
    PRODUCT_REGISTRY,
    get_product_config,
    get_default_product,
    # WINK product IDs
    WinkProductIds,
    get_wink_product_ids,
    get_pipeline_product_ids_as_lists,
    get_metadata_product_ids_as_lists,
    # Date configuration
    ProductDateConfig,
    get_default_date_config,
    # Feature configuration
    ProductFeatureConfig,
    get_default_feature_config,
    get_feature_config_for_product_type,
    # Competitor configuration
    CompetitorConfig,
    get_competitor_config,
    # Constants
    DEFAULT_MAX_FEATURES,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_CV_FOLDS,
    DiagnosticSeverity,
)


class TestGetProductConfig:
    """Tests for get_product_config function."""

    def test_get_product_config_6y20b(self):
        """get_product_config should return 6Y20B config correctly."""
        config = get_product_config("6Y20B")

        assert config.product_code == "6Y20B"
        assert config.buffer_level == 0.20
        assert config.term_years == 6
        assert config.product_type == "rila"

    def test_get_product_config_6y10b(self):
        """get_product_config should return 6Y10B config correctly."""
        config = get_product_config("6Y10B")

        assert config.product_code == "6Y10B"
        assert config.buffer_level == 0.10
        assert config.term_years == 6
        assert config.product_type == "rila"

    def test_get_product_config_10y20b(self):
        """get_product_config should return 10Y20B config correctly."""
        config = get_product_config("10Y20B")

        assert config.product_code == "10Y20B"
        assert config.buffer_level == 0.20
        assert config.term_years == 10
        assert config.product_type == "rila"

    def test_get_product_config_invalid_raises(self, invalid_product_codes):
        """get_product_config should raise KeyError for invalid codes."""
        for code in invalid_product_codes:
            if code:  # Skip empty string
                with pytest.raises(KeyError, match="Unknown product"):
                    get_product_config(code)

    def test_get_product_config_fia_products(self, valid_fia_product_codes):
        """get_product_config should return FIA configs correctly."""
        for code in valid_fia_product_codes:
            config = get_product_config(code)
            assert config.product_type == "fia"
            assert config.buffer_level is None


class TestGetDefaultProduct:
    """Tests for get_default_product function."""

    def test_get_default_product_returns_6y20b(self):
        """get_default_product should return 6Y20B as default."""
        config = get_default_product()

        assert config.product_code == "6Y20B"
        assert config.buffer_level == 0.20


class TestProductRegistry:
    """Tests for PRODUCT_REGISTRY completeness."""

    def test_product_registry_contains_all_rila_products(self, valid_product_codes):
        """PRODUCT_REGISTRY should contain all expected RILA products."""
        for code in valid_product_codes:
            assert code in PRODUCT_REGISTRY
            assert PRODUCT_REGISTRY[code].product_type == "rila"

    def test_product_registry_contains_fia_products(self, valid_fia_product_codes):
        """PRODUCT_REGISTRY should contain FIA products."""
        for code in valid_fia_product_codes:
            assert code in PRODUCT_REGISTRY
            assert PRODUCT_REGISTRY[code].product_type == "fia"

    def test_product_config_fields_complete(self):
        """ProductConfig should have all required fields."""
        required_fields = [
            'name', 'product_code', 'product_type', 'rate_column',
            'own_rate_prefix', 'competitor_rate_prefix', 'buffer_level',
            'term_years', 'primary_index', 'max_lag', 'competitor_count'
        ]

        for code, config in PRODUCT_REGISTRY.items():
            for field in required_fields:
                assert hasattr(config, field), f"{code} missing {field}"


class TestWinkProductIds:
    """Tests for WINK product ID mappings."""

    def test_get_wink_product_ids_returns_dataclass(self):
        """get_wink_product_ids should return WinkProductIds dataclass."""
        ids = get_wink_product_ids()
        assert isinstance(ids, WinkProductIds)

    def test_wink_pipeline_ids_structure(self):
        """Pipeline IDs should have correct structure."""
        ids = get_wink_product_ids()
        assert isinstance(ids.pipeline_ids, dict)
        assert "Prudential" in ids.pipeline_ids
        assert ids.pipeline_ids["Prudential"] == (2979,)

    def test_wink_metadata_ids_structure(self):
        """Metadata IDs should have correct structure."""
        ids = get_wink_product_ids()
        assert isinstance(ids.metadata_ids, dict)
        assert "Prudential" in ids.metadata_ids

    def test_wink_ids_are_tuples(self):
        """All WINK IDs should be tuples of integers."""
        ids = get_wink_product_ids()

        for company, product_ids in ids.pipeline_ids.items():
            assert isinstance(product_ids, tuple), f"{company} pipeline IDs not tuple"
            for pid in product_ids:
                assert isinstance(pid, int), f"{company} ID {pid} not int"

    def test_get_pipeline_product_ids_as_lists(self):
        """get_pipeline_product_ids_as_lists should return dict with lists."""
        ids = get_pipeline_product_ids_as_lists()
        assert isinstance(ids, dict)
        assert "Prudential" in ids
        assert isinstance(ids["Prudential"], list)

    def test_get_metadata_product_ids_as_lists(self):
        """get_metadata_product_ids_as_lists should return dict with lists."""
        ids = get_metadata_product_ids_as_lists()
        assert isinstance(ids, dict)
        assert "Prudential" in ids
        assert isinstance(ids["Prudential"], list)


class TestProductDateConfig:
    """Tests for ProductDateConfig."""

    def test_get_default_date_config_returns_dataclass(self):
        """get_default_date_config should return ProductDateConfig."""
        config = get_default_date_config()
        assert isinstance(config, ProductDateConfig)

    def test_default_date_config_values(self):
        """Default date config should have expected values."""
        config = get_default_date_config()
        assert config.rate_analysis_start_date == "2018-06-21"
        assert config.analysis_start_date == "2021-01-01"
        assert config.feature_analysis_start_date == "2022-01-01"
        assert config.data_filter_start_date == "2018-01-01"


class TestProductFeatureConfig:
    """Tests for ProductFeatureConfig."""

    def test_get_default_feature_config_returns_dataclass(self):
        """get_default_feature_config should return ProductFeatureConfig."""
        config = get_default_feature_config()
        assert isinstance(config, ProductFeatureConfig)

    def test_default_feature_config_base_features(self, expected_base_features):
        """Base features should include prudential_rate_t0 (unified naming)."""
        config = get_default_feature_config()
        for feature in expected_base_features:
            assert feature in config.base_features

    def test_default_feature_config_candidate_features(self):
        """Candidate features should include competitor and prudential features."""
        config = get_default_feature_config()
        assert len(config.candidate_features) > 0

        # Should have competitor features
        competitor_features = [f for f in config.candidate_features if 'competitor' in f]
        assert len(competitor_features) > 0

    def test_get_feature_config_for_product_type(self):
        """get_feature_config_for_product_type should return config for any type."""
        for product_type in ["rila", "fia", "myga"]:
            config = get_feature_config_for_product_type(product_type)
            assert isinstance(config, ProductFeatureConfig)


class TestCompetitorConfig:
    """Tests for CompetitorConfig."""

    def test_get_competitor_config_returns_dataclass(self):
        """get_competitor_config should return CompetitorConfig."""
        config = get_competitor_config()
        assert isinstance(config, CompetitorConfig)

    def test_rila_competitors_list(self, expected_rila_competitors):
        """RILA competitors should match expected list."""
        config = get_competitor_config()
        for competitor in expected_rila_competitors:
            assert competitor in config.rila_competitors

    def test_core_competitors_list(self, expected_core_competitors):
        """Core competitors should match expected list."""
        config = get_competitor_config()
        for competitor in expected_core_competitors:
            assert competitor in config.core_competitors

    def test_own_company_is_prudential(self):
        """Own company should be Prudential."""
        config = get_competitor_config()
        assert config.own_company == "Prudential"

    def test_core_competitors_subset_of_rila_competitors(self):
        """Core competitors should be subset of RILA competitors."""
        config = get_competitor_config()
        for competitor in config.core_competitors:
            assert competitor in config.rila_competitors


class TestProductConfigValidation:
    """Tests for ProductConfig validation logic."""

    def test_rila_requires_buffer_level(self):
        """RILA product without buffer_level should raise ValueError."""
        with pytest.raises(ValueError, match="buffer_level is required for RILA"):
            ProductConfig(
                name="Test RILA",
                product_code="TEST",
                product_type="rila",
                buffer_level=None,
            )

    def test_rila_buffer_level_must_be_positive(self):
        """RILA buffer_level must be in (0, 1]."""
        with pytest.raises(ValueError, match="buffer_level must be in"):
            ProductConfig(
                name="Test RILA",
                product_code="TEST",
                product_type="rila",
                buffer_level=0.0,  # Invalid: must be > 0
            )

    def test_invalid_product_type_raises(self):
        """Invalid product_type should raise ValueError."""
        with pytest.raises(ValueError, match="product_type must be one of"):
            ProductConfig(
                name="Test",
                product_code="TEST",
                product_type="invalid",
            )

    def test_fia_allows_none_buffer(self):
        """FIA product should allow None buffer_level."""
        config = ProductConfig(
            name="Test FIA",
            product_code="TEST",
            product_type="fia",
            buffer_level=None,
        )
        assert config.buffer_level is None


class TestDiagnosticSeverity:
    """Tests for DiagnosticSeverity constants."""

    def test_severity_levels_defined(self):
        """All severity levels should be defined."""
        assert DiagnosticSeverity.NONE == "NONE"
        assert DiagnosticSeverity.LOW == "LOW"
        assert DiagnosticSeverity.MODERATE == "MODERATE"
        assert DiagnosticSeverity.SEVERE == "SEVERE"
        assert DiagnosticSeverity.POOR == "POOR"


class TestDefaultConstants:
    """Tests for default constants in product_config."""

    def test_default_max_features(self):
        """DEFAULT_MAX_FEATURES should be 3."""
        assert DEFAULT_MAX_FEATURES == 3

    def test_default_bootstrap_samples(self):
        """DEFAULT_BOOTSTRAP_SAMPLES should be 100."""
        assert DEFAULT_BOOTSTRAP_SAMPLES == 100

    def test_default_cv_folds(self):
        """DEFAULT_CV_FOLDS should be 5."""
        assert DEFAULT_CV_FOLDS == 5

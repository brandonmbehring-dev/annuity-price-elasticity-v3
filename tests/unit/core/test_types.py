"""
Tests for src.core.types module.

Tests TypedDict definitions for type correctness and field validation.
"""

import pytest
from typing import get_type_hints

from src.core.types import (
    AWSConfig,
    InferenceConfig,
    FeatureConfig,
    DataPaths,
    AggregationConfig,
    ConstraintConfig,
    InferenceResults,
)


class TestAWSConfig:
    """Tests for AWSConfig TypedDict."""

    def test_aws_config_has_required_fields(self):
        """AWSConfig should have all required fields."""
        config: AWSConfig = {
            "sts_endpoint_url": "https://sts.amazonaws.com",
            "role_arn": "arn:aws:iam::123456789:role/TestRole",
            "xid": "test_user",
            "bucket_name": "test-bucket",
        }
        assert config["sts_endpoint_url"] == "https://sts.amazonaws.com"
        assert config["role_arn"] == "arn:aws:iam::123456789:role/TestRole"
        assert config["xid"] == "test_user"
        assert config["bucket_name"] == "test-bucket"

    def test_aws_config_type_annotations(self):
        """AWSConfig should have correct type annotations."""
        hints = get_type_hints(AWSConfig)
        assert hints["sts_endpoint_url"] == str
        assert hints["role_arn"] == str
        assert hints["xid"] == str
        assert hints["bucket_name"] == str


class TestInferenceConfig:
    """Tests for InferenceConfig TypedDict."""

    def test_inference_config_has_required_fields(self):
        """InferenceConfig should have all required fields."""
        config: InferenceConfig = {
            "product_code": "6Y20B",
            "product_type": "rila",
            "own_rate_column": "prudential_rate_current",
            "competitor_rate_column": "competitor_mid_t2",
            "target_column": "sales_target_current",
            "rate_adjustment_range": (-300, 300),
            "n_bootstrap": 1000,
            "confidence_levels": [0.80, 0.90, 0.95],
        }
        assert config["product_code"] == "6Y20B"
        assert config["product_type"] == "rila"
        assert config["n_bootstrap"] == 1000

    def test_inference_config_tuple_field(self):
        """InferenceConfig rate_adjustment_range should be a tuple."""
        config: InferenceConfig = {
            "product_code": "6Y20B",
            "product_type": "rila",
            "own_rate_column": "prudential_rate_current",
            "competitor_rate_column": "competitor_mid_t2",
            "target_column": "sales_target_current",
            "rate_adjustment_range": (-300, 300),
            "n_bootstrap": 1000,
            "confidence_levels": [0.80, 0.90, 0.95],
        }
        assert len(config["rate_adjustment_range"]) == 2
        assert config["rate_adjustment_range"][0] == -300
        assert config["rate_adjustment_range"][1] == 300


class TestFeatureConfig:
    """Tests for FeatureConfig TypedDict."""

    def test_feature_config_list_fields(self):
        """FeatureConfig should handle list fields correctly."""
        config: FeatureConfig = {
            "base_features": ["feature1", "feature2"],
            "candidate_features": ["candidate1", "candidate2"],
            "exclude_patterns": ["excluded_*"],
        }
        assert len(config["base_features"]) == 2
        assert "feature1" in config["base_features"]


class TestDataPaths:
    """Tests for DataPaths TypedDict."""

    def test_data_paths_string_fields(self):
        """DataPaths should have string path fields."""
        paths: DataPaths = {
            "sales_data": "/data/sales.parquet",
            "competitive_rates": "/data/rates.parquet",
            "market_weights": "/data/weights.parquet",
        }
        assert paths["sales_data"].endswith(".parquet")


class TestConstraintConfig:
    """Tests for ConstraintConfig TypedDict."""

    def test_constraint_config_boolean_fields(self):
        """ConstraintConfig should have boolean control fields."""
        config: ConstraintConfig = {
            "enforce_positive_own_rate": True,
            "enforce_negative_competitor": True,
            "strict_mode": False,
        }
        assert config["enforce_positive_own_rate"] is True
        assert config["strict_mode"] is False

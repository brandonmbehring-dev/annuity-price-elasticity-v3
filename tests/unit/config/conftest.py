"""
Shared fixtures for src/config unit tests.

Provides reusable fixtures for testing configuration modules including
product configs, pipeline configs, and feature selection configs.
"""

import pytest
from typing import Dict, List


@pytest.fixture
def valid_product_codes() -> List[str]:
    """Standard RILA product codes for testing.

    Returns
    -------
    List[str]
        Valid product codes from PRODUCT_REGISTRY
    """
    return ["6Y20B", "6Y10B", "10Y20B"]


@pytest.fixture
def valid_fia_product_codes() -> List[str]:
    """FIA product codes for testing.

    Returns
    -------
    List[str]
        Valid FIA product codes from PRODUCT_REGISTRY
    """
    return ["FIA5YR", "FIA7YR", "FIACA5YR", "FIACA7YR"]


@pytest.fixture
def invalid_product_codes() -> List[str]:
    """Invalid product codes that should raise errors.

    Returns
    -------
    List[str]
        Product codes not in PRODUCT_REGISTRY
    """
    return ["INVALID", "NOSUCH", "12345", ""]


@pytest.fixture
def sample_aws_config() -> Dict[str, str]:
    """Mock AWS configuration for testing (no real credentials).

    Returns
    -------
    Dict[str, str]
        AWS configuration keys
    """
    return {
        'xid': "x259830",
        'role_arn': "arn:aws:iam::159058241883:role/test-role",
        'sts_endpoint_url': "https://sts.us-east-1.amazonaws.com",
        'bucket_name': "test-bucket",
        'source_bucket_name': "test-source-bucket",
        'output_bucket_name': "test-output-bucket",
    }


@pytest.fixture
def expected_rila_competitors() -> List[str]:
    """Expected RILA competitor companies.

    Returns
    -------
    List[str]
        List of competitor company names
    """
    return [
        "Allianz",
        "Athene",
        "Brighthouse",
        "Equitable",
        "Jackson",
        "Lincoln",
        "Symetra",
        "Trans",
    ]


@pytest.fixture
def expected_core_competitors() -> List[str]:
    """Expected core competitor companies.

    Returns
    -------
    List[str]
        List of core competitor company names
    """
    return ["Brighthouse", "Equitable", "Lincoln"]


@pytest.fixture
def expected_base_features() -> List[str]:
    """Expected base features for feature selection.

    Feature Naming Unification (2026-01-26): Uses _t0 naming.

    Returns
    -------
    List[str]
        List of base feature names
    """
    return ["prudential_rate_t0"]


@pytest.fixture
def expected_pipeline_config_keys() -> List[str]:
    """Expected keys in pipeline configuration.

    Returns
    -------
    List[str]
        Keys expected from build_pipeline_configs()
    """
    return [
        'product_filter',
        'sales_cleanup',
        'wink_processing',
        'competitive',
        'lag_features',
        'final_features',
    ]

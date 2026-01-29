"""
Shared fixtures for src/data unit tests.

Provides reusable fixtures for testing data adapters, extraction,
preprocessing, and pipeline modules.
"""

import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_sales_df() -> pd.DataFrame:
    """Small synthetic sales DataFrame for testing.

    Returns
    -------
    pd.DataFrame
        Minimal sales data with required columns
    """
    return pd.DataFrame({
        'application_signed_date': pd.date_range('2022-01-01', periods=100, freq='D'),
        'contract_issue_date': pd.date_range('2022-01-05', periods=100, freq='D'),
        'premium_amount': np.random.uniform(10000, 100000, 100),
        'product_name': ['FlexGuard indexed variable annuity'] * 100,
        'term': ['6Y'] * 100,
        'buffer_rate': ['20%'] * 100,
    })


@pytest.fixture
def sample_wink_df() -> pd.DataFrame:
    """Small synthetic WINK DataFrame for testing.

    Returns
    -------
    pd.DataFrame
        Minimal WINK rate data
    """
    dates = pd.date_range('2022-01-01', periods=50, freq='W')
    return pd.DataFrame({
        'date': dates,
        'effective_date': dates,
        'Prudential': np.random.uniform(3.0, 5.0, 50),
        'Allianz': np.random.uniform(3.0, 5.0, 50),
        'Athene': np.random.uniform(3.0, 5.0, 50),
        'Brighthouse': np.random.uniform(3.0, 5.0, 50),
        'Lincoln': np.random.uniform(3.0, 5.0, 50),
    })


@pytest.fixture
def sample_market_weights_df() -> pd.DataFrame:
    """Synthetic market weights DataFrame for testing.

    Returns
    -------
    pd.DataFrame
        Market share weights by company
    """
    return pd.DataFrame({
        'quarter': ['2022Q1', '2022Q2', '2022Q3', '2022Q4'],
        'Prudential': [0.15, 0.16, 0.14, 0.15],
        'Allianz': [0.20, 0.19, 0.21, 0.20],
        'Athene': [0.12, 0.13, 0.12, 0.13],
        'Brighthouse': [0.18, 0.17, 0.18, 0.17],
        'Lincoln': [0.15, 0.15, 0.15, 0.15],
    })


@pytest.fixture
def sample_macro_df() -> pd.DataFrame:
    """Synthetic macro data DataFrame for testing.

    Returns
    -------
    pd.DataFrame
        Macroeconomic indicators
    """
    dates = pd.date_range('2022-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'date': dates,
        'DGS5': np.random.uniform(1.5, 3.5, 100),
        'DGS10': np.random.uniform(2.0, 4.0, 100),
        'VIX': np.random.uniform(15, 30, 100),
        'SP500': np.random.uniform(3800, 4500, 100),
    })


@pytest.fixture
def temp_fixtures_dir(
    sample_sales_df: pd.DataFrame,
    sample_wink_df: pd.DataFrame,
    sample_market_weights_df: pd.DataFrame,
    sample_macro_df: pd.DataFrame,
) -> Path:
    """Create temporary fixtures directory with test data.

    Yields
    ------
    Path
        Path to temporary fixtures directory with parquet files
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fixtures_path = Path(tmpdir)

        # Write fixture files
        sample_sales_df.to_parquet(fixtures_path / 'raw_sales_data.parquet')
        sample_wink_df.to_parquet(fixtures_path / 'rates_fixture.parquet')
        sample_market_weights_df.to_parquet(fixtures_path / 'weights_fixture.parquet')
        sample_macro_df.to_parquet(fixtures_path / 'macro_fixture.parquet')

        yield fixtures_path


@pytest.fixture
def empty_fixtures_dir() -> Path:
    """Create temporary empty fixtures directory.

    Yields
    ------
    Path
        Path to empty temporary directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def aws_config_dict() -> Dict[str, str]:
    """Mock AWS configuration dictionary.

    Returns
    -------
    Dict[str, str]
        AWS configuration for adapter testing
    """
    return {
        'xid': 'test_user',
        'role_arn': 'arn:aws:iam::123456789012:role/test-role',
        'sts_endpoint_url': 'https://sts.us-east-1.amazonaws.com',
        'bucket_name': 'test-bucket',
    }

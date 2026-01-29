"""
Stage-by-Stage Pipeline Equivalence Tests
==========================================

Tests that validate each of the 10 pipeline stages independently against baselines.
This enables pinpointing exactly where mathematical divergence occurs if any.

Pipeline Stages:
1. Product Filtering
2. Sales Data Cleanup
3. Time Series Conversion (Application + Contract)
4. WINK Rate Processing
5. Market Share Weighting
6. Data Integration (Daily)
7. Competitive Features Engineering
8. Weekly Aggregation
9. Lag and Polynomial Features
10. Final Feature Preparation

Mathematical Equivalence: 1e-12 precision for all stages

Author: Claude Code
Date: 2026-01-29
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.pipelines import (
    apply_product_filters,
    apply_sales_data_cleanup,
    apply_application_time_series,
    apply_contract_time_series,
    apply_wink_rate_processing,
    apply_market_share_weighting,
    apply_data_integration,
    apply_competitive_features,
    apply_weekly_aggregation,
    apply_lag_and_polynomial_features,
    apply_final_feature_preparation
)
from src.config.pipeline_config import (
    ProductFilterConfig,
    SalesCleanupConfig,
    TimeSeriesConfig,
    WinkProcessingConfig,
    CompetitiveConfig,
    LagFeatureConfig,
    FeatureConfig
)

# Mathematical equivalence tolerance
TOLERANCE = 1e-12

# Baseline path
BASELINE_PATH = Path(__file__).parent.parent / "baselines/aws_mode"


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def baseline_stage_01():
    """Baseline output from Stage 1: Product Filtering."""
    return pd.read_parquet(BASELINE_PATH / "01_filtered_products.parquet")


@pytest.fixture(scope="module")
def baseline_stage_02():
    """Baseline output from Stage 2: Sales Cleanup."""
    return pd.read_parquet(BASELINE_PATH / "02_cleaned_sales.parquet")


@pytest.fixture(scope="module")
def baseline_stage_03a():
    """Baseline output from Stage 3a: Application Time Series."""
    return pd.read_parquet(BASELINE_PATH / "03a_application_time_series.parquet")


@pytest.fixture(scope="module")
def baseline_stage_03b():
    """Baseline output from Stage 3b: Contract Time Series."""
    return pd.read_parquet(BASELINE_PATH / "03b_contract_time_series.parquet")


@pytest.fixture(scope="module")
def baseline_stage_04():
    """Baseline output from Stage 4: WINK Processed."""
    return pd.read_parquet(BASELINE_PATH / "04_wink_processed.parquet")


@pytest.fixture(scope="module")
def baseline_stage_05():
    """Baseline output from Stage 5: Market Weighted."""
    return pd.read_parquet(BASELINE_PATH / "05_market_weighted.parquet")


@pytest.fixture(scope="module")
def baseline_stage_06():
    """Baseline output from Stage 6: Integrated Daily."""
    return pd.read_parquet(BASELINE_PATH / "06_integrated_daily.parquet")


@pytest.fixture(scope="module")
def baseline_stage_07():
    """Baseline output from Stage 7: Competitive Features."""
    return pd.read_parquet(BASELINE_PATH / "07_competitive_features.parquet")


@pytest.fixture(scope="module")
def baseline_stage_08():
    """Baseline output from Stage 8: Weekly Aggregated."""
    return pd.read_parquet(BASELINE_PATH / "08_weekly_aggregated.parquet")


@pytest.fixture(scope="module")
def baseline_stage_09():
    """Baseline output from Stage 9: Lag Features."""
    return pd.read_parquet(BASELINE_PATH / "09_lag_features.parquet")


@pytest.fixture(scope="module")
def baseline_stage_10():
    """Baseline output from Stage 10: Final Dataset."""
    return pd.read_parquet(BASELINE_PATH / "10_final_dataset.parquet")


@pytest.fixture(scope="module")
def rila_6y20b_product_config():
    """Product configuration for RILA 6Y20B."""
    return ProductFilterConfig(
        product_name="FlexGuard indexed variable annuity",
        buffer_rate="20%",
        term="6Y"
    )


@pytest.fixture(scope="module")
def sales_cleanup_config():
    """Sales cleanup configuration."""
    return SalesCleanupConfig(
        date_column='application_signed_date',
        premium_column='premium',
        alias_map={'premium': 'sales'},
        outlier_quantile=0.99
    )


@pytest.fixture(scope="module")
def time_series_config():
    """Time series configuration."""
    return TimeSeriesConfig(
        date_column='application_signed_date',
        value_column='sales',
        agg_function='sum'
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def validate_dataframe_equivalence(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    tolerance: float = TOLERANCE,
    stage: str = "unknown"
) -> None:
    """Validate two DataFrames are mathematically equivalent.

    Parameters
    ----------
    actual : pd.DataFrame
        DataFrame produced by current implementation
    expected : pd.DataFrame
        Baseline DataFrame to compare against
    tolerance : float
        Numerical tolerance for floating point comparisons (default 1e-12)
    stage : str
        Stage name for error messages

    Raises
    ------
    AssertionError
        If DataFrames are not equivalent within tolerance
    """
    # Check shape
    assert actual.shape == expected.shape, (
        f"[{stage}] Shape mismatch: actual={actual.shape}, expected={expected.shape}"
    )

    # Check columns
    assert set(actual.columns) == set(expected.columns), (
        f"[{stage}] Column mismatch:\n"
        f"  Missing: {set(expected.columns) - set(actual.columns)}\n"
        f"  Extra: {set(actual.columns) - set(expected.columns)}"
    )

    # Align column order
    actual = actual[expected.columns]

    # Check dtypes (allow some flexibility for int/float)
    for col in expected.columns:
        actual_dtype = actual[col].dtype
        expected_dtype = expected[col].dtype

        # Allow int64 <-> float64 conversions
        if pd.api.types.is_numeric_dtype(actual_dtype) and pd.api.types.is_numeric_dtype(expected_dtype):
            continue

        assert actual_dtype == expected_dtype, (
            f"[{stage}] dtype mismatch for column '{col}': "
            f"actual={actual_dtype}, expected={expected_dtype}"
        )

    # Check values column by column
    for col in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[col]):
            # Numeric columns: use tolerance-based comparison
            np.testing.assert_allclose(
                actual[col].values,
                expected[col].values,
                rtol=tolerance,
                atol=tolerance,
                err_msg=f"[{stage}] Numeric column '{col}' differs beyond tolerance {tolerance}"
            )
        else:
            # Non-numeric columns: exact match
            pd.testing.assert_series_equal(
                actual[col],
                expected[col],
                check_exact=True,
                obj=f"[{stage}] Column '{col}'"
            )


# =============================================================================
# STAGE 1: PRODUCT FILTERING
# =============================================================================


@pytest.mark.integration
def test_stage_01_product_filtering(
    raw_sales_data,
    rila_6y20b_product_config,
    baseline_stage_01
):
    """Stage 1: Product filtering must match baseline at 1e-12 precision."""
    # Apply product filtering
    filtered = apply_product_filters(raw_sales_data, rila_6y20b_product_config)

    # Validate against baseline
    validate_dataframe_equivalence(
        actual=filtered,
        expected=baseline_stage_01,
        tolerance=TOLERANCE,
        stage="01_product_filtering"
    )

    # Additional sanity checks
    assert len(filtered) > 0, "Filtered dataset should not be empty"
    assert 'product_name' in filtered.columns
    assert (filtered['product_name'] == rila_6y20b_product_config['product_name']).all()


# =============================================================================
# STAGE 2: SALES DATA CLEANUP
# =============================================================================


@pytest.mark.integration
def test_stage_02_sales_cleanup(
    baseline_stage_01,
    sales_cleanup_config,
    baseline_stage_02
):
    """Stage 2: Sales cleanup must match baseline at 1e-12 precision."""
    # Apply sales cleanup
    cleaned = apply_sales_data_cleanup(baseline_stage_01, sales_cleanup_config)

    # Validate against baseline
    validate_dataframe_equivalence(
        actual=cleaned,
        expected=baseline_stage_02,
        tolerance=TOLERANCE,
        stage="02_sales_cleanup"
    )

    # Additional sanity checks
    assert 'sales' in cleaned.columns
    assert cleaned['sales'].notna().all()
    assert (cleaned['sales'] > 0).all()


# =============================================================================
# STAGE 3: TIME SERIES CONVERSION
# =============================================================================


@pytest.mark.integration
def test_stage_03a_application_time_series(
    baseline_stage_02,
    time_series_config,
    baseline_stage_03a
):
    """Stage 3a: Application time series must match baseline at 1e-12 precision."""
    # Apply time series conversion (application date)
    ts_application = apply_application_time_series(baseline_stage_02, time_series_config)

    # Validate against baseline
    validate_dataframe_equivalence(
        actual=ts_application,
        expected=baseline_stage_03a,
        tolerance=TOLERANCE,
        stage="03a_application_time_series"
    )


@pytest.mark.integration
def test_stage_03b_contract_time_series(
    baseline_stage_02,
    time_series_config,
    baseline_stage_03b
):
    """Stage 3b: Contract time series must match baseline at 1e-12 precision."""
    # Apply time series conversion (contract date)
    config_contract = {**time_series_config, 'date_column': 'contract_effective_date'}
    ts_contract = apply_contract_time_series(baseline_stage_02, config_contract)

    # Validate against baseline
    validate_dataframe_equivalence(
        actual=ts_contract,
        expected=baseline_stage_03b,
        tolerance=TOLERANCE,
        stage="03b_contract_time_series"
    )


# =============================================================================
# STAGE 4: WINK RATE PROCESSING
# =============================================================================


@pytest.mark.integration
def test_stage_04_wink_processing(
    raw_wink_data,
    baseline_stage_04
):
    """Stage 4: WINK rate processing must match baseline at 1e-12 precision."""
    # Configure WINK processing
    wink_config = WinkProcessingConfig(
        date_column='as_of_date',
        company_column='company_name',
        rate_column='cap_rate'
    )

    # Apply WINK processing
    wink_processed = apply_wink_rate_processing(raw_wink_data, wink_config)

    # Validate against baseline
    validate_dataframe_equivalence(
        actual=wink_processed,
        expected=baseline_stage_04,
        tolerance=TOLERANCE,
        stage="04_wink_processing"
    )


# =============================================================================
# STAGE 5: MARKET SHARE WEIGHTING
# =============================================================================


@pytest.mark.integration
def test_stage_05_market_weighting(
    baseline_stage_04,
    market_share_weights,
    baseline_stage_05
):
    """Stage 5: Market share weighting must match baseline at 1e-12 precision."""
    # Apply market share weighting
    market_weighted = apply_market_share_weighting(baseline_stage_04, market_share_weights)

    # Validate against baseline
    validate_dataframe_equivalence(
        actual=market_weighted,
        expected=baseline_stage_05,
        tolerance=TOLERANCE,
        stage="05_market_weighting"
    )


# =============================================================================
# STAGE 6: DATA INTEGRATION
# =============================================================================


@pytest.mark.integration
def test_stage_06_data_integration(
    baseline_stage_03a,
    baseline_stage_05,
    baseline_stage_06
):
    """Stage 6: Data integration must match baseline at 1e-12 precision."""
    # Apply data integration
    integrated = apply_data_integration(
        sales_ts=baseline_stage_03a,
        competitive_rates=baseline_stage_05,
        macro_indicators={}  # Will load from fixtures if needed
    )

    # Validate against baseline
    validate_dataframe_equivalence(
        actual=integrated,
        expected=baseline_stage_06,
        tolerance=TOLERANCE,
        stage="06_data_integration"
    )


# =============================================================================
# STAGE 7: COMPETITIVE FEATURES
# =============================================================================


@pytest.mark.integration
def test_stage_07_competitive_features(
    baseline_stage_06,
    baseline_stage_07
):
    """Stage 7: Competitive features must match baseline at 1e-12 precision."""
    # Configure competitive features
    competitive_config = CompetitiveConfig(
        competitor_columns=['C_weighted_mean', 'C_median', 'C_top5_mean'],
        aggregation_method='weighted'
    )

    # Apply competitive features
    competitive = apply_competitive_features(baseline_stage_06, competitive_config)

    # Validate against baseline
    validate_dataframe_equivalence(
        actual=competitive,
        expected=baseline_stage_07,
        tolerance=TOLERANCE,
        stage="07_competitive_features"
    )


# =============================================================================
# STAGE 8: WEEKLY AGGREGATION
# =============================================================================


@pytest.mark.integration
def test_stage_08_weekly_aggregation(
    baseline_stage_07,
    baseline_stage_08
):
    """Stage 8: Weekly aggregation must match baseline at 1e-12 precision."""
    # Apply weekly aggregation
    weekly = apply_weekly_aggregation(
        df=baseline_stage_07,
        date_column='date',
        agg_config={'sales': 'sum', 'own_cap_rate': 'mean'}
    )

    # Validate against baseline
    validate_dataframe_equivalence(
        actual=weekly,
        expected=baseline_stage_08,
        tolerance=TOLERANCE,
        stage="08_weekly_aggregation"
    )


# =============================================================================
# STAGE 9: LAG AND POLYNOMIAL FEATURES
# =============================================================================


@pytest.mark.integration
def test_stage_09_lag_features(
    baseline_stage_08,
    baseline_stage_09
):
    """Stage 9: Lag features must match baseline at 1e-12 precision."""
    # Configure lag features
    lag_config = LagFeatureConfig(
        lag_columns=['own_cap_rate', 'C_weighted_mean'],
        max_lag_periods=4,
        polynomial_degree=2
    )

    # Apply lag and polynomial features
    lag_features = apply_lag_and_polynomial_features(baseline_stage_08, lag_config)

    # Validate against baseline
    validate_dataframe_equivalence(
        actual=lag_features,
        expected=baseline_stage_09,
        tolerance=TOLERANCE,
        stage="09_lag_features"
    )


# =============================================================================
# STAGE 10: FINAL PREPARATION
# =============================================================================


@pytest.mark.integration
def test_stage_10_final_preparation(
    baseline_stage_09,
    baseline_stage_10
):
    """Stage 10: Final preparation must match baseline at 1e-12 precision."""
    # Configure final preparation
    feature_config = FeatureConfig(
        drop_columns=['processing_days', 'outlier_flag'],
        handle_missing='drop'
    )

    # Apply final preparation
    final = apply_final_feature_preparation(baseline_stage_09, feature_config)

    # Validate against baseline
    validate_dataframe_equivalence(
        actual=final,
        expected=baseline_stage_10,
        tolerance=TOLERANCE,
        stage="10_final_preparation"
    )

    # Final dataset should be modeling-ready
    assert final.isnull().sum().sum() == 0, "Final dataset should have no missing values"
    assert len(final) > 100, "Final dataset should have sufficient observations"


# =============================================================================
# COMPREHENSIVE PIPELINE TEST
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline_stage_by_stage(
    raw_sales_data,
    raw_wink_data,
    market_share_weights,
    rila_6y20b_product_config,
    baseline_stage_10
):
    """Run complete pipeline stage-by-stage and validate final output.

    This test runs all 10 stages in sequence and validates the final result.
    Individual stage tests above can pinpoint where divergence occurs.
    """
    # Stage 1: Product Filtering
    stage_01 = apply_product_filters(raw_sales_data, rila_6y20b_product_config)

    # Stage 2: Sales Cleanup
    sales_config = SalesCleanupConfig(
        date_column='application_signed_date',
        premium_column='premium',
        alias_map={'premium': 'sales'},
        outlier_quantile=0.99
    )
    stage_02 = apply_sales_data_cleanup(stage_01, sales_config)

    # Stage 3a: Application Time Series
    ts_config = TimeSeriesConfig(
        date_column='application_signed_date',
        value_column='sales',
        agg_function='sum'
    )
    stage_03a = apply_application_time_series(stage_02, ts_config)

    # Stage 4: WINK Processing
    wink_config = WinkProcessingConfig(
        date_column='as_of_date',
        company_column='company_name',
        rate_column='cap_rate'
    )
    stage_04 = apply_wink_rate_processing(raw_wink_data, wink_config)

    # Stage 5: Market Weighting
    stage_05 = apply_market_share_weighting(stage_04, market_share_weights)

    # Stage 6: Data Integration
    stage_06 = apply_data_integration(
        sales_ts=stage_03a,
        competitive_rates=stage_05,
        macro_indicators={}
    )

    # Stage 7: Competitive Features
    competitive_config = CompetitiveConfig(
        competitor_columns=['C_weighted_mean'],
        aggregation_method='weighted'
    )
    stage_07 = apply_competitive_features(stage_06, competitive_config)

    # Stage 8: Weekly Aggregation
    stage_08 = apply_weekly_aggregation(
        df=stage_07,
        date_column='date',
        agg_config={'sales': 'sum', 'own_cap_rate': 'mean'}
    )

    # Stage 9: Lag Features
    lag_config = LagFeatureConfig(
        lag_columns=['own_cap_rate', 'C_weighted_mean'],
        max_lag_periods=4,
        polynomial_degree=2
    )
    stage_09 = apply_lag_and_polynomial_features(stage_08, lag_config)

    # Stage 10: Final Preparation
    feature_config = FeatureConfig(
        drop_columns=['processing_days', 'outlier_flag'],
        handle_missing='drop'
    )
    stage_10 = apply_final_feature_preparation(stage_09, feature_config)

    # Validate final output matches baseline
    validate_dataframe_equivalence(
        actual=stage_10,
        expected=baseline_stage_10,
        tolerance=TOLERANCE,
        stage="full_pipeline_end_to_end"
    )


# =============================================================================
# STAGE ISOLATION TESTS
# =============================================================================


@pytest.mark.integration
class TestStageIsolation:
    """Test that each stage can be run independently using baseline inputs."""

    def test_each_stage_runs_independently(
        self,
        baseline_stage_01,
        baseline_stage_02,
        baseline_stage_03a,
        baseline_stage_04,
        baseline_stage_05,
        baseline_stage_06,
        baseline_stage_07,
        baseline_stage_08,
        baseline_stage_09,
        baseline_stage_10
    ):
        """Verify all baseline stages load successfully."""
        stages = [
            baseline_stage_01, baseline_stage_02, baseline_stage_03a,
            baseline_stage_04, baseline_stage_05, baseline_stage_06,
            baseline_stage_07, baseline_stage_08, baseline_stage_09,
            baseline_stage_10
        ]

        for i, stage in enumerate(stages, 1):
            assert stage is not None, f"Stage {i:02d} baseline should load"
            assert isinstance(stage, pd.DataFrame), f"Stage {i:02d} should be DataFrame"
            assert len(stage) > 0, f"Stage {i:02d} should not be empty"

    def test_stage_dependencies(
        self,
        baseline_stage_01,
        baseline_stage_02,
        baseline_stage_08,
        baseline_stage_09
    ):
        """Verify stage outputs have expected relationships."""
        # Stage 2 should have fewer rows than Stage 1 (outlier removal)
        assert len(baseline_stage_02) <= len(baseline_stage_01)

        # Stage 9 should have more columns than Stage 8 (lag features added)
        assert baseline_stage_09.shape[1] > baseline_stage_08.shape[1]

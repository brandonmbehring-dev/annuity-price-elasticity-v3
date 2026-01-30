"""
Full Pipeline End-to-End Tests (Offline with Fixtures)
========================================================

Tests complete pipeline execution using only fixtures (no AWS required).
Validates that entire data pipeline, feature engineering, and inference
can run successfully in offline mode with mathematical equivalence.

Pipeline Stages Tested:
1. Data loading from fixtures
2. Product filtering (Stage 1)
3. Sales cleanup (Stage 2)
4. Time series conversion (Stage 3)
5. WINK processing (Stage 4)
6. Market weighting (Stage 5)
7. Data integration (Stage 6)
8. Competitive features (Stage 7)
9. Weekly aggregation (Stage 8)
10. Lag/polynomial features (Stage 9)
11. Final preparation (Stage 10)
12. Feature selection
13. Bootstrap inference

Mathematical Equivalence: 1e-12 for baseline comparisons

Author: Claude Code
Date: 2026-01-29
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import time

from src.notebooks.interface import UnifiedNotebookInterface
from src.data.pipelines import (
    apply_product_filters,
    apply_sales_data_cleanup,
    apply_application_time_series,
    apply_wink_rate_processing,
    apply_market_share_weighting,
    apply_data_integration,
    apply_competitive_features,
    apply_weekly_aggregation,
    apply_lag_and_polynomial_features,
    apply_final_feature_preparation
)
from src.features.selection.notebook_interface import run_feature_selection

# Tolerance for baseline comparisons
TOLERANCE = 1e-12
STATISTICAL_TOLERANCE = 1e-6


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def baseline_final_dataset():
    """Load baseline final dataset for comparison."""
    baseline_path = Path(__file__).parent.parent / "fixtures/rila/final_weekly_dataset.parquet"
    return pd.read_parquet(baseline_path)


@pytest.fixture(scope="module")
def baseline_inference_results():
    """Load baseline inference results if available."""
    baseline_path = Path(__file__).parent.parent / "baselines/rila/reference/inference_results.parquet"
    if baseline_path.exists():
        return pd.read_parquet(baseline_path)
    return None


# =============================================================================
# COMPLETE PIPELINE E2E TEST
# =============================================================================


@pytest.mark.e2e
@pytest.mark.slow
class TestFullPipelineOffline:
    """End-to-end pipeline tests using only fixtures (no AWS)."""

    def test_complete_rila_6y20b_pipeline_offline(self, baseline_final_dataset):
        """Run complete RILA 6Y20B pipeline using only fixtures.

        This is the most comprehensive test - validates entire pipeline
        from raw data to final modeling dataset at 1e-12 precision.
        """
        # Create interface in fixture mode (offline)
        interface = UnifiedNotebookInterface(
            product_code="6Y20B",
            data_source="fixture"
        )

        # Load data from fixtures
        df = interface.load_data()

        # Validate data loaded successfully
        assert df is not None, "Failed to load data from fixtures"
        assert isinstance(df, pd.DataFrame), "Loaded data should be DataFrame"
        assert len(df) > 0, "Loaded data should not be empty"

        # Should match baseline dimensions
        assert 150 < len(df) < 250, f"Expected 150-250 rows, got {len(df)}"
        assert df.shape[1] > 500, f"Expected >500 features, got {df.shape[1]}"

        # Validate against baseline
        # Allow column order differences
        common_cols = set(df.columns) & set(baseline_final_dataset.columns)
        assert len(common_cols) > 500, "Should have >500 common columns with baseline"

        # Check numerical equivalence for common columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in baseline_final_dataset.columns:
                np.testing.assert_allclose(
                    df[col].values,
                    baseline_final_dataset[col].values,
                    rtol=TOLERANCE,
                    atol=TOLERANCE,
                    err_msg=f"Column '{col}' differs from baseline"
                )

    def test_pipeline_with_feature_selection(self):
        """Test complete pipeline including feature selection."""
        # Create interface
        interface = UnifiedNotebookInterface(
            product_code="6Y20B",
            data_source="fixture"
        )

        # Load data
        df = interface.load_data()

        # Run feature selection
        selection_results = run_feature_selection(
            df=df,
            target_column='sales',
            max_features=30,
            n_bootstrap=100  # Reduced for speed
        )

        # Validate feature selection results
        assert selection_results is not None
        assert 'selected_features' in selection_results
        assert 'feature_scores' in selection_results
        assert 'selection_metrics' in selection_results

        # Should select reasonable number of features
        selected = selection_results['selected_features']
        assert 10 <= len(selected) <= 30, f"Expected 10-30 features, got {len(selected)}"

        # Key features should be selected
        assert 'own_cap_rate_lag_1' in selected or any('own' in f for f in selected), (
            "Own rate features should be selected"
        )

    def test_pipeline_with_full_inference(self):
        """Test complete pipeline with bootstrap inference."""
        # Create interface
        interface = UnifiedNotebookInterface(
            product_code="6Y20B",
            data_source="fixture"
        )

        # Load data
        df = interface.load_data()

        # Run inference
        inference_results = interface.run_inference(
            df=df,
            n_bootstrap=100,  # Reduced for speed
            n_jobs=-1,
            random_state=42
        )

        # Validate inference results
        assert inference_results is not None
        assert 'coefficients' in inference_results
        assert 'predictions' in inference_results
        assert 'metrics' in inference_results

        # Validate metrics
        metrics = inference_results['metrics']
        assert 'R²' in metrics
        assert 'MAPE' in metrics
        assert 'AIC' in metrics
        assert 'BIC' in metrics

        # Performance should meet minimum thresholds
        assert metrics['R²'] > 0.70, f"R²={metrics['R²']:.3f} below minimum 0.70"
        assert metrics['MAPE'] < 0.20, f"MAPE={metrics['MAPE']:.3f} above maximum 0.20"

    def test_pipeline_reproducibility(self):
        """Pipeline should produce identical results across runs.

        With same seed and fixture data, results should be byte-for-byte identical.
        """
        # Run pipeline twice with same configuration
        interface1 = UnifiedNotebookInterface(
            product_code="6Y20B",
            data_source="fixture"
        )
        df1 = interface1.load_data()

        interface2 = UnifiedNotebookInterface(
            product_code="6Y20B",
            data_source="fixture"
        )
        df2 = interface2.load_data()

        # Should be identical
        pd.testing.assert_frame_equal(
            df1,
            df2,
            check_exact=False,
            rtol=TOLERANCE,
            atol=TOLERANCE,
            obj="Pipeline should be reproducible"
        )

    def test_pipeline_stages_produce_valid_output(
        self,
        raw_sales_data,
        raw_wink_data,
        market_share_weights
    ):
        """Test that all pipeline stages produce valid output."""
        from src.config.pipeline_config import (
            ProductFilterConfig,
            SalesCleanupConfig,
            TimeSeriesConfig,
            WinkProcessingConfig,
            CompetitiveConfig,
            LagFeatureConfig,
            FeatureConfig
        )

        # Stage 1: Product Filtering
        product_config = ProductFilterConfig(
            product_name="FlexGuard indexed variable annuity",
            buffer_rate="20%",
            term="6Y"
        )
        stage_01 = apply_product_filters(raw_sales_data, product_config)
        assert len(stage_01) > 0
        assert 'product_name' in stage_01.columns

        # Stage 2: Sales Cleanup
        sales_config = SalesCleanupConfig(
            date_column='application_signed_date',
            premium_column='premium',
            alias_map={'premium': 'sales'},
            outlier_quantile=0.99
        )
        stage_02 = apply_sales_data_cleanup(stage_01, sales_config)
        assert len(stage_02) > 0
        assert 'sales' in stage_02.columns

        # Stage 3: Time Series
        ts_config = TimeSeriesConfig(
            date_column='application_signed_date',
            value_column='sales',
            agg_function='sum'
        )
        stage_03 = apply_application_time_series(stage_02, ts_config)
        assert len(stage_03) > 0

        # Stage 4: WINK Processing
        wink_config = WinkProcessingConfig(
            date_column='as_of_date',
            company_column='company_name',
            rate_column='cap_rate'
        )
        stage_04 = apply_wink_rate_processing(raw_wink_data, wink_config)
        assert len(stage_04) > 0

        # Stage 5: Market Weighting
        stage_05 = apply_market_share_weighting(stage_04, market_share_weights)
        assert len(stage_05) > 0

        # Stage 6: Data Integration
        stage_06 = apply_data_integration(
            sales_ts=stage_03,
            competitive_rates=stage_05,
            macro_indicators={}
        )
        assert len(stage_06) > 0

        # Stage 7: Competitive Features
        competitive_config = CompetitiveConfig(
            competitor_columns=['C_weighted_mean'],
            aggregation_method='weighted'
        )
        stage_07 = apply_competitive_features(stage_06, competitive_config)
        assert len(stage_07) > 0

        # Stage 8: Weekly Aggregation
        stage_08 = apply_weekly_aggregation(
            df=stage_07,
            date_column='date',
            agg_config={'sales': 'sum', 'own_cap_rate': 'mean'}
        )
        assert len(stage_08) > 0

        # Stage 9: Lag Features
        lag_config = LagFeatureConfig(
            lag_columns=['own_cap_rate', 'C_weighted_mean'],
            max_lag_periods=4,
            polynomial_degree=2
        )
        stage_09 = apply_lag_and_polynomial_features(stage_08, lag_config)
        assert len(stage_09) > 0
        assert stage_09.shape[1] > stage_08.shape[1], "Lag features should add columns"

        # Stage 10: Final Preparation
        feature_config = FeatureConfig(
            drop_columns=[],
            handle_missing='drop'
        )
        stage_10 = apply_final_feature_preparation(stage_09, feature_config)
        assert len(stage_10) > 0

        # Final stage should be modeling-ready
        assert stage_10.isnull().sum().sum() == 0, "Final dataset should have no missing values"


# =============================================================================
# PIPELINE PERFORMANCE TESTS
# =============================================================================


@pytest.mark.e2e
class TestPipelinePerformance:
    """Test pipeline performance meets reasonable benchmarks."""

    def test_full_pipeline_completes_in_reasonable_time(self):
        """Full pipeline should complete within performance baseline.

        With fixtures and parallel processing, should complete in < 60s.
        """
        start = time.time()

        interface = UnifiedNotebookInterface(
            product_code="6Y20B",
            data_source="fixture"
        )

        # Load data (full pipeline)
        df = interface.load_data()

        elapsed = time.time() - start

        # Should complete within 60 seconds
        assert elapsed < 60.0, (
            f"Pipeline took {elapsed:.2f}s (expected < 60s). "
            "This may indicate performance regression."
        )

        # Should produce valid output
        assert df is not None
        assert len(df) > 100

    def test_inference_completes_in_reasonable_time(self):
        """Inference with 100 bootstrap samples should complete quickly."""
        interface = UnifiedNotebookInterface(
            product_code="6Y20B",
            data_source="fixture"
        )

        # Load data
        df = interface.load_data()

        # Time inference
        start = time.time()

        results = interface.run_inference(
            df=df,
            n_bootstrap=100,
            n_jobs=-1,
            random_state=42
        )

        elapsed = time.time() - start

        # With parallel processing, 100 samples should complete quickly
        assert elapsed < 30.0, (
            f"Inference took {elapsed:.2f}s (expected < 30s)"
        )

        assert results is not None


# =============================================================================
# PIPELINE QUALITY VALIDATION
# =============================================================================


@pytest.mark.e2e
class TestPipelineQuality:
    """Test pipeline output quality and correctness."""

    def test_pipeline_produces_modeling_ready_dataset(self):
        """Pipeline output should be ready for modeling."""
        interface = UnifiedNotebookInterface(
            product_code="6Y20B",
            data_source="fixture"
        )

        df = interface.load_data()

        # Check data quality
        assert df.isnull().sum().sum() == 0, "Should have no missing values"
        assert len(df) > 100, "Should have sufficient observations"
        assert df.shape[1] > 500, "Should have sufficient features"

        # Check key columns exist
        assert 'sales' in df.columns or 'sales_target' in df.columns, (
            "Target variable should exist"
        )

        # Check for lag features (critical for RILA)
        lag_features = [col for col in df.columns if 'lag_' in col or '_t' in col]
        assert len(lag_features) > 20, f"Should have >20 lag features, found {len(lag_features)}"

        # Check for polynomial features
        poly_features = [col for col in df.columns if '_squared' in col or '_poly_' in col]
        assert len(poly_features) > 5, f"Should have >5 polynomial features, found {len(poly_features)}"

    def test_pipeline_respects_economic_constraints(self):
        """Pipeline should not create forbidden features (lag-0 competitors)."""
        interface = UnifiedNotebookInterface(
            product_code="6Y20B",
            data_source="fixture"
        )

        df = interface.load_data()

        # Check for forbidden lag-0 competitor features
        forbidden_patterns = [
            r'competitor.*_t0',
            r'competitor.*_lag_0',
            r'competitor.*_current',
            r'C_.*_t0',
            r'C_.*_lag_0'
        ]

        import re
        forbidden_features = []
        for pattern in forbidden_patterns:
            regex = re.compile(pattern)
            matches = [col for col in df.columns if regex.search(col)]
            forbidden_features.extend(matches)

        # Should not have lag-0 competitor features (causal identification)
        assert len(forbidden_features) == 0, (
            f"Found {len(forbidden_features)} forbidden lag-0 competitor features:\n"
            f"{forbidden_features[:10]}"  # Show first 10
        )

    def test_pipeline_feature_naming_consistency(self):
        """All features should follow consistent naming conventions."""
        interface = UnifiedNotebookInterface(
            product_code="6Y20B",
            data_source="fixture"
        )

        df = interface.load_data()

        # Check lag feature naming
        lag_features = [col for col in df.columns if '_lag_' in col or '_t' in col]

        for feature in lag_features:
            # Lag features should have numeric suffix
            if '_lag_' in feature:
                lag_num = feature.split('_lag_')[-1]
                assert lag_num.isdigit(), f"Lag feature '{feature}' should have numeric suffix"
            elif '_t' in feature:
                t_suffix = feature.split('_t')[-1]
                assert t_suffix.isdigit(), f"Temporal feature '{feature}' should have numeric suffix"


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


@pytest.mark.e2e
class TestPipelineErrorHandling:
    """Test pipeline error handling and edge cases."""

    def test_pipeline_handles_missing_fixture_gracefully(self):
        """Pipeline should fail fast with clear error if fixtures missing."""
        # Try to create interface with non-existent product
        with pytest.raises(Exception) as exc_info:
            interface = UnifiedNotebookInterface(
                product_code="NONEXISTENT",
                data_source="fixture"
            )
            df = interface.load_data()

        # Error should be informative
        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ['not found', 'missing', 'invalid', 'unknown']), (
            "Error message should indicate missing/invalid product"
        )

    def test_pipeline_validates_data_quality(self):
        """Pipeline should validate data quality at checkpoints."""
        interface = UnifiedNotebookInterface(
            product_code="6Y20B",
            data_source="fixture"
        )

        df = interface.load_data()

        # Data should pass quality checks
        assert len(df) > 0, "Should not produce empty dataset"
        assert df.shape[1] > 0, "Should have features"
        assert not df.isnull().all().any(), "Should not have all-null columns"


# =============================================================================
# BASELINE COMPARISON TESTS
# =============================================================================


@pytest.mark.e2e
@pytest.mark.slow
class TestBaselineComparison:
    """Compare pipeline output to saved baselines."""

    def test_pipeline_matches_baseline_dataset(self, baseline_final_dataset):
        """Pipeline output should match baseline dataset at 1e-12 precision."""
        interface = UnifiedNotebookInterface(
            product_code="6Y20B",
            data_source="fixture"
        )

        df = interface.load_data()

        # Compare shapes
        assert df.shape == baseline_final_dataset.shape, (
            f"Shape mismatch: actual={df.shape}, baseline={baseline_final_dataset.shape}"
        )

        # Compare values for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in baseline_final_dataset.columns:
                np.testing.assert_allclose(
                    df[col].values,
                    baseline_final_dataset[col].values,
                    rtol=TOLERANCE,
                    atol=TOLERANCE,
                    err_msg=f"Column '{col}' differs from baseline"
                )

    def test_inference_matches_baseline_metrics(self, baseline_inference_results):
        """Inference metrics should match baseline within statistical tolerance."""
        if baseline_inference_results is None:
            pytest.skip("Baseline inference results not available")

        interface = UnifiedNotebookInterface(
            product_code="6Y20B",
            data_source="fixture"
        )

        df = interface.load_data()

        # Run inference with same config as baseline
        results = interface.run_inference(
            df=df,
            n_bootstrap=1000,
            n_jobs=-1,
            random_state=42
        )

        # Compare metrics (relaxed tolerance for stochastic bootstrap)
        baseline_metrics = baseline_inference_results['metrics']
        actual_metrics = results['metrics']

        for metric in ['R²', 'MAPE', 'AIC', 'BIC']:
            if metric in baseline_metrics and metric in actual_metrics:
                assert abs(actual_metrics[metric] - baseline_metrics[metric]) < STATISTICAL_TOLERANCE, (
                    f"Metric '{metric}' differs from baseline: "
                    f"actual={actual_metrics[metric]:.6f}, baseline={baseline_metrics[metric]:.6f}"
                )


# =============================================================================
# INTEGRATION WITH OTHER SYSTEMS
# =============================================================================


@pytest.mark.e2e
class TestSystemIntegration:
    """Test pipeline integrates with other systems correctly."""

    def test_pipeline_output_compatible_with_forecasting(self):
        """Pipeline output should be compatible with forecasting module."""
        interface = UnifiedNotebookInterface(
            product_code="6Y20B",
            data_source="fixture"
        )

        df = interface.load_data()

        # Should have required columns for forecasting
        assert 'date' in df.columns or df.index.name == 'date', (
            "Should have date column/index for time series forecasting"
        )

        # Should have sufficient history
        assert len(df) >= 52, "Should have at least 52 weeks (1 year) of data"

    def test_pipeline_output_compatible_with_visualization(self):
        """Pipeline output should be compatible with visualization tools."""
        interface = UnifiedNotebookInterface(
            product_code="6Y20B",
            data_source="fixture"
        )

        df = interface.load_data()

        # Should be able to extract time series
        assert len(df) > 0
        assert df.select_dtypes(include=[np.number]).shape[1] > 0, (
            "Should have numeric columns for visualization"
        )

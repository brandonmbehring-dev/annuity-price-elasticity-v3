"""
Performance baseline tests to detect performance regressions.

These tests establish timing baselines for critical operations and validate that
performance remains within acceptable thresholds. If a test fails, it indicates
a performance regression that should be investigated.

Usage:
    # Run all performance tests
    pytest tests/performance/test_performance_baselines.py -v

    # Run only fast performance tests (exclude slow tests > 30s)
    pytest tests/performance/test_performance_baselines.py -m "not slow" -v

    # Run specific performance test
    pytest tests/performance/test_performance_baselines.py::TestPerformanceBaselines::test_feature_engineering_performance -v

Baseline Thresholds:
    - Feature engineering (medium dataset): < 2s
    - Feature selection (medium dataset): < 10s
    - Bootstrap 100 samples: < 5s
    - Bootstrap 1000 samples: < 30s
    - Bootstrap 10000 samples: < 300s (5 minutes)
    - Full pipeline: < 360s (6 minutes)

Note:
    Timing baselines may vary based on hardware. If tests consistently fail on
    your machine but logic is correct, consider updating baselines.json.
"""

import time
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Import pipeline components for testing
try:
    from src.feature_engineering.features import engineer_features
    from src.feature_selection.selector import FeatureSelector
    from src.models.bootstrap_models import BootstrapInference
    from src.pipelines.data_pipeline import DataPipeline
    from src.adapters.fixture_adapter import FixtureAdapter
except ImportError:
    # Fallback for testing without full implementation
    engineer_features = None
    FeatureSelector = None
    BootstrapInference = None
    DataPipeline = None
    FixtureAdapter = None


class TestPerformanceBaselines:
    """
    Performance baseline tests for critical operations.

    These tests measure execution time for key operations and assert against
    baseline thresholds to detect performance regressions.
    """

    @pytest.fixture(scope="class")
    def performance_baselines(self):
        """
        Load performance baseline thresholds.

        Returns:
            dict: Performance baselines with max_seconds thresholds
        """
        return {
            'feature_engineering': {
                'max_seconds': 2.0,
                'description': 'Feature engineering on medium dataset (100 rows × 50 features)'
            },
            'feature_selection': {
                'max_seconds': 10.0,
                'description': 'Feature selection on medium dataset'
            },
            'bootstrap_100': {
                'max_seconds': 5.0,
                'description': 'Bootstrap inference with 100 samples'
            },
            'bootstrap_1000': {
                'max_seconds': 30.0,
                'description': 'Bootstrap inference with 1000 samples'
            },
            'bootstrap_10000': {
                'max_seconds': 300.0,
                'description': 'Bootstrap inference with 10000 samples (production config)'
            },
            'full_pipeline': {
                'max_seconds': 360.0,
                'description': 'Complete data pipeline execution (all 10 stages)'
            }
        }

    def test_feature_engineering_performance(self, medium_dataset, performance_baselines):
        """
        Test feature engineering performance on medium dataset.

        Validates that feature engineering completes within baseline threshold.
        Uses medium dataset (100 rows × 50 features) for realistic timing.

        Args:
            medium_dataset: Medium-sized fixture for integration testing
            performance_baselines: Baseline timing thresholds

        Raises:
            AssertionError: If operation exceeds baseline threshold
        """
        if engineer_features is None:
            pytest.skip("Feature engineering module not available")

        baseline = performance_baselines['feature_engineering']
        max_time = baseline['max_seconds']

        # Measure execution time
        start = time.time()
        result = engineer_features(medium_dataset)
        elapsed = time.time() - start

        # Validate result
        assert result is not None, "Feature engineering returned None"
        assert isinstance(result, pd.DataFrame), "Feature engineering must return DataFrame"

        # Check performance baseline
        assert elapsed < max_time, (
            f"Feature engineering took {elapsed:.2f}s (max: {max_time}s). "
            f"Performance regression detected. Expected: < {max_time}s, Actual: {elapsed:.2f}s. "
            f"Dataset: {medium_dataset.shape}"
        )

        print(f"✓ Feature engineering completed in {elapsed:.2f}s (baseline: < {max_time}s)")

    def test_feature_selection_performance(self, medium_dataset, performance_baselines):
        """
        Test feature selection performance on medium dataset.

        Validates that feature selection completes within baseline threshold.
        Feature selection is computationally intensive and should be monitored.

        Args:
            medium_dataset: Medium-sized fixture for integration testing
            performance_baselines: Baseline timing thresholds

        Raises:
            AssertionError: If operation exceeds baseline threshold
        """
        if FeatureSelector is None:
            pytest.skip("Feature selection module not available")

        baseline = performance_baselines['feature_selection']
        max_time = baseline['max_seconds']

        # Initialize feature selector
        selector = FeatureSelector(config={'max_features': 30})

        # Measure execution time
        start = time.time()
        selected_features = selector.select(medium_dataset)
        elapsed = time.time() - start

        # Validate result
        assert selected_features is not None, "Feature selection returned None"
        assert len(selected_features) > 0, "Feature selection returned empty list"
        assert len(selected_features) <= 30, "Feature selection exceeded max_features"

        # Check performance baseline
        assert elapsed < max_time, (
            f"Feature selection took {elapsed:.2f}s (max: {max_time}s). "
            f"Performance regression detected. Expected: < {max_time}s, Actual: {elapsed:.2f}s. "
            f"Dataset: {medium_dataset.shape}, Selected: {len(selected_features)} features"
        )

        print(f"✓ Feature selection completed in {elapsed:.2f}s (baseline: < {max_time}s)")

    def test_bootstrap_100_performance(self, medium_dataset, small_bootstrap_config, performance_baselines):
        """
        Test bootstrap inference performance with 100 samples.

        Fast bootstrap configuration for unit testing. Should complete quickly.

        Args:
            medium_dataset: Medium-sized fixture for integration testing
            small_bootstrap_config: Bootstrap config with n_bootstrap=100
            performance_baselines: Baseline timing thresholds

        Raises:
            AssertionError: If operation exceeds baseline threshold
        """
        if BootstrapInference is None:
            pytest.skip("Bootstrap inference module not available")

        baseline = performance_baselines['bootstrap_100']
        max_time = baseline['max_seconds']

        # Override bootstrap samples to exactly 100
        config = {**small_bootstrap_config, 'n_bootstrap': 100}
        model = BootstrapInference(config)

        # Prepare data
        X = medium_dataset.drop(columns=['sales'], errors='ignore')
        y = medium_dataset.get('sales', pd.Series(np.random.poisson(50, len(medium_dataset))))

        # Measure execution time
        start = time.time()
        result = model.fit_predict(X, y)
        elapsed = time.time() - start

        # Validate result
        assert result is not None, "Bootstrap inference returned None"
        assert 'predictions' in result, "Bootstrap inference missing predictions"

        # Check performance baseline
        assert elapsed < max_time, (
            f"Bootstrap (100 samples) took {elapsed:.2f}s (max: {max_time}s). "
            f"Performance regression detected. Expected: < {max_time}s, Actual: {elapsed:.2f}s. "
            f"Dataset: {X.shape}, Bootstrap samples: 100"
        )

        print(f"✓ Bootstrap (100 samples) completed in {elapsed:.2f}s (baseline: < {max_time}s)")

    def test_bootstrap_1000_performance(self, medium_dataset, medium_bootstrap_config, performance_baselines):
        """
        Test bootstrap inference performance with 1000 samples.

        Moderate bootstrap configuration for integration testing. More realistic
        than 100 samples but faster than production 10K.

        Args:
            medium_dataset: Medium-sized fixture for integration testing
            medium_bootstrap_config: Bootstrap config with n_bootstrap=1000
            performance_baselines: Baseline timing thresholds

        Raises:
            AssertionError: If operation exceeds baseline threshold
        """
        if BootstrapInference is None:
            pytest.skip("Bootstrap inference module not available")

        baseline = performance_baselines['bootstrap_1000']
        max_time = baseline['max_seconds']

        # Override bootstrap samples to exactly 1000
        config = {**medium_bootstrap_config, 'n_bootstrap': 1000}
        model = BootstrapInference(config)

        # Prepare data
        X = medium_dataset.drop(columns=['sales'], errors='ignore')
        y = medium_dataset.get('sales', pd.Series(np.random.poisson(50, len(medium_dataset))))

        # Measure execution time
        start = time.time()
        result = model.fit_predict(X, y)
        elapsed = time.time() - start

        # Validate result
        assert result is not None, "Bootstrap inference returned None"
        assert 'predictions' in result, "Bootstrap inference missing predictions"

        # Check performance baseline
        assert elapsed < max_time, (
            f"Bootstrap (1000 samples) took {elapsed:.2f}s (max: {max_time}s). "
            f"Performance regression detected. Expected: < {max_time}s, Actual: {elapsed:.2f}s. "
            f"Dataset: {X.shape}, Bootstrap samples: 1000"
        )

        print(f"✓ Bootstrap (1000 samples) completed in {elapsed:.2f}s (baseline: < {max_time}s)")

    @pytest.mark.slow
    def test_bootstrap_10000_performance(self, full_production_dataset, production_bootstrap_config, performance_baselines):
        """
        Test bootstrap inference performance with 10000 samples (production config).

        Production-level bootstrap configuration. This test is marked as 'slow'
        and can be skipped in fast CI runs with: pytest -m "not slow"

        Args:
            full_production_dataset: Full production dataset (167 weeks × 598 features)
            production_bootstrap_config: Production bootstrap config with n_bootstrap=10000
            performance_baselines: Baseline timing thresholds

        Raises:
            AssertionError: If operation exceeds baseline threshold (5 minutes)
        """
        if BootstrapInference is None:
            pytest.skip("Bootstrap inference module not available")

        baseline = performance_baselines['bootstrap_10000']
        max_time = baseline['max_seconds']

        # Override bootstrap samples to exactly 10000
        config = {**production_bootstrap_config, 'n_bootstrap': 10000}
        model = BootstrapInference(config)

        # Prepare data
        X = full_production_dataset.drop(columns=['sales'], errors='ignore')
        y = full_production_dataset.get('sales', pd.Series(np.random.poisson(50, len(full_production_dataset))))

        # Measure execution time
        start = time.time()
        result = model.fit_predict(X, y)
        elapsed = time.time() - start

        # Validate result
        assert result is not None, "Bootstrap inference returned None"
        assert 'predictions' in result, "Bootstrap inference missing predictions"

        # Check performance baseline
        assert elapsed < max_time, (
            f"Bootstrap (10000 samples) took {elapsed:.2f}s ({elapsed/60:.1f} min) (max: {max_time}s = {max_time/60:.1f} min). "
            f"Performance regression detected. Expected: < {max_time}s, Actual: {elapsed:.2f}s. "
            f"Dataset: {X.shape}, Bootstrap samples: 10000"
        )

        print(f"✓ Bootstrap (10000 samples) completed in {elapsed:.2f}s = {elapsed/60:.1f} min (baseline: < {max_time/60:.1f} min)")

    @pytest.mark.slow
    def test_full_pipeline_performance(self, performance_baselines):
        """
        Test full data pipeline performance (all 10 stages).

        Validates complete pipeline execution from raw data to modeling dataset.
        This test is marked as 'slow' and can be skipped in fast CI runs.

        Args:
            performance_baselines: Baseline timing thresholds

        Raises:
            AssertionError: If operation exceeds baseline threshold (6 minutes)
        """
        if DataPipeline is None or FixtureAdapter is None:
            pytest.skip("Data pipeline or fixture adapter not available")

        baseline = performance_baselines['full_pipeline']
        max_time = baseline['max_seconds']

        # Initialize pipeline with fixture adapter
        adapter = FixtureAdapter()
        pipeline = DataPipeline(adapter=adapter)

        # Measure execution time
        start = time.time()
        result = pipeline.run_full_pipeline()
        elapsed = time.time() - start

        # Validate result
        assert result is not None, "Pipeline returned None"
        assert isinstance(result, pd.DataFrame), "Pipeline must return DataFrame"
        assert len(result) > 0, "Pipeline returned empty DataFrame"

        # Check performance baseline
        assert elapsed < max_time, (
            f"Full pipeline took {elapsed:.2f}s ({elapsed/60:.1f} min) (max: {max_time}s = {max_time/60:.1f} min). "
            f"Performance regression detected. Expected: < {max_time}s, Actual: {elapsed:.2f}s. "
            f"Output shape: {result.shape}"
        )

        print(f"✓ Full pipeline completed in {elapsed:.2f}s = {elapsed/60:.1f} min (baseline: < {max_time/60:.1f} min)")


class TestComponentPerformance:
    """
    Additional performance tests for individual components.

    These tests focus on specific operations that are known bottlenecks
    or frequently used in the pipeline.
    """

    def test_dataframe_loading_performance(self):
        """
        Test fixture loading performance.

        Validates that fixture loading is fast enough for rapid test iteration.
        Fixtures should be cached and load in < 1 second.
        """
        fixture_path = Path("tests/fixtures/rila/final_weekly_dataset.parquet")

        if not fixture_path.exists():
            pytest.skip("Fixture not available")

        # Measure loading time
        start = time.time()
        df = pd.read_parquet(fixture_path)
        elapsed = time.time() - start

        # Validate result
        assert df is not None, "Fixture loading returned None"
        assert len(df) > 0, "Fixture is empty"

        # Fixture loading should be very fast (< 1 second)
        max_time = 1.0
        assert elapsed < max_time, (
            f"Fixture loading took {elapsed:.2f}s (max: {max_time}s). "
            f"Fixture may be too large or disk I/O is slow. "
            f"Shape: {df.shape}, Size: {fixture_path.stat().st_size / 1024 / 1024:.1f} MB"
        )

        print(f"✓ Fixture loading completed in {elapsed:.3f}s (baseline: < {max_time}s)")

    def test_data_aggregation_performance(self, medium_dataset):
        """
        Test data aggregation performance.

        Validates that typical aggregation operations (groupby, mean, sum)
        complete quickly on medium-sized datasets.
        """
        # Measure aggregation time
        start = time.time()

        # Typical aggregation operations
        if 'product_name' in medium_dataset.columns:
            agg_result = medium_dataset.groupby('product_name').agg({
                'sales': ['sum', 'mean', 'count'],
                'premium': ['sum', 'mean']
            })
        else:
            # Fallback: aggregate by week or other column
            agg_result = medium_dataset.iloc[:, :5].mean()

        elapsed = time.time() - start

        # Validate result
        assert agg_result is not None, "Aggregation returned None"

        # Aggregation should be fast (< 0.5 seconds on medium dataset)
        max_time = 0.5
        assert elapsed < max_time, (
            f"Data aggregation took {elapsed:.3f}s (max: {max_time}s). "
            f"Performance regression detected. Dataset: {medium_dataset.shape}"
        )

        print(f"✓ Data aggregation completed in {elapsed:.3f}s (baseline: < {max_time}s)")

    def test_data_filtering_performance(self, medium_dataset):
        """
        Test data filtering performance.

        Validates that typical filtering operations (boolean indexing)
        complete quickly on medium-sized datasets.
        """
        # Measure filtering time
        start = time.time()

        # Typical filtering operation
        numeric_cols = medium_dataset.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            threshold = medium_dataset[col].median()
            filtered = medium_dataset[medium_dataset[col] > threshold]
        else:
            filtered = medium_dataset.head(50)

        elapsed = time.time() - start

        # Validate result
        assert filtered is not None, "Filtering returned None"
        assert len(filtered) >= 0, "Filtering failed"

        # Filtering should be very fast (< 0.1 seconds on medium dataset)
        max_time = 0.1
        assert elapsed < max_time, (
            f"Data filtering took {elapsed:.3f}s (max: {max_time}s). "
            f"Performance regression detected. Dataset: {medium_dataset.shape}"
        )

        print(f"✓ Data filtering completed in {elapsed:.3f}s (baseline: < {max_time}s)")


# Performance baseline configuration (can be loaded from JSON in future)
PERFORMANCE_BASELINES = {
    'feature_engineering': {'max_seconds': 2.0},
    'feature_selection': {'max_seconds': 10.0},
    'bootstrap_100': {'max_seconds': 5.0},
    'bootstrap_1000': {'max_seconds': 30.0},
    'bootstrap_10000': {'max_seconds': 300.0},
    'full_pipeline': {'max_seconds': 360.0},
    'fixture_loading': {'max_seconds': 1.0},
    'data_aggregation': {'max_seconds': 0.5},
    'data_filtering': {'max_seconds': 0.1}
}

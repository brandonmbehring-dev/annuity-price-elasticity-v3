"""
Memory baseline tests to detect memory leaks and excessive memory usage.

These tests establish memory usage baselines for critical operations and validate
that memory consumption remains within acceptable limits. Memory regressions can
indicate leaks, inefficient algorithms, or unnecessary data duplication.

Usage:
    # Run all memory tests
    pytest tests/performance/test_memory_baselines.py -v

    # Run only fast memory tests
    pytest tests/performance/test_memory_baselines.py -m "not slow" -v

    # Run with memory profiling details
    pytest tests/performance/test_memory_baselines.py -v -s

Memory Baselines:
    - Medium dataset operations: < 500 MB
    - Full dataset operations: < 2 GB
    - Bootstrap 10K samples: < 8 GB
    - Feature engineering: < 1 GB
    - Feature selection: < 2 GB

Dependencies:
    - psutil: Install with `pip install psutil`

Note:
    Memory measurements may vary based on Python version and OS.
    If tests consistently fail but logic is correct, review baselines.
"""

import os
import gc
import pytest
import pandas as pd
import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Import pipeline components for testing
try:
    from src.feature_engineering.features import engineer_features
    from src.feature_selection.selector import FeatureSelector
    from src.models.bootstrap_models import BootstrapInference
    from src.pipelines.data_pipeline import DataPipeline
    from src.adapters.fixture_adapter import FixtureAdapter
except ImportError:
    engineer_features = None
    FeatureSelector = None
    BootstrapInference = None
    DataPipeline = None
    FixtureAdapter = None


def get_memory_usage_mb():
    """
    Get current process memory usage in megabytes.

    Uses psutil to measure Resident Set Size (RSS), which is the
    portion of memory occupied by the process in RAM.

    Returns:
        float: Memory usage in MB

    Raises:
        RuntimeError: If psutil is not available
    """
    if not PSUTIL_AVAILABLE:
        raise RuntimeError("psutil is required for memory testing. Install with: pip install psutil")

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


@pytest.fixture(scope="function", autouse=False)
def memory_cleanup():
    """
    Fixture to clean up memory before and after tests.

    Runs garbage collection to ensure clean baseline for memory measurements.
    Yields control to test, then cleans up again.
    """
    # Clean up before test
    gc.collect()

    yield

    # Clean up after test
    gc.collect()


class TestMemoryBaselines:
    """
    Memory baseline tests for critical operations.

    These tests measure memory consumption for key operations and assert against
    baseline thresholds to detect memory regressions and leaks.
    """

    @pytest.fixture(scope="class")
    def memory_baselines(self):
        """
        Load memory baseline thresholds.

        Returns:
            dict: Memory baselines with max_mb thresholds
        """
        return {
            'medium_dataset_operations': {
                'max_mb': 500,
                'description': 'Operations on medium dataset (100 rows × 50 features)'
            },
            'full_dataset_operations': {
                'max_mb': 2000,
                'description': 'Operations on full production dataset (167 weeks × 598 features)'
            },
            'feature_engineering': {
                'max_mb': 1000,
                'description': 'Feature engineering memory usage'
            },
            'feature_selection': {
                'max_mb': 2000,
                'description': 'Feature selection memory usage'
            },
            'bootstrap_10000': {
                'max_mb': 8000,
                'description': 'Bootstrap with 10000 samples (production config)'
            },
            'full_pipeline': {
                'max_mb': 4000,
                'description': 'Complete data pipeline memory usage'
            }
        }

    def test_psutil_available(self):
        """
        Verify psutil is available for memory testing.

        Memory tests require psutil for accurate memory measurements.
        This test will fail if psutil is not installed.
        """
        assert PSUTIL_AVAILABLE, (
            "psutil is required for memory testing. Install with: pip install psutil"
        )

    def test_medium_dataset_memory(self, medium_dataset, memory_baselines, memory_cleanup):
        """
        Test memory usage for operations on medium dataset.

        Validates that typical operations on medium dataset don't consume
        excessive memory. Medium dataset should be manageable in < 500 MB.

        Args:
            medium_dataset: Medium-sized fixture for integration testing
            memory_baselines: Memory usage thresholds
            memory_cleanup: Fixture to clean up memory

        Raises:
            AssertionError: If memory usage exceeds baseline threshold
        """
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available")

        baseline = memory_baselines['medium_dataset_operations']
        max_mb = baseline['max_mb']

        # Measure memory before operation
        mem_before = get_memory_usage_mb()

        # Perform typical operations
        result = medium_dataset.copy()

        # Find first numeric column for operation
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            result['new_feature'] = result[numeric_cols[0]] * 2
        else:
            result['new_feature'] = 0

        result_agg = result.describe() if len(result) > 0 else None

        # Measure memory after operation
        mem_after = get_memory_usage_mb()
        mem_delta = mem_after - mem_before

        # Validate result
        assert result is not None, "Operation returned None"

        # Check memory baseline
        assert mem_delta < max_mb, (
            f"Medium dataset operations used {mem_delta:.1f} MB (max: {max_mb} MB). "
            f"Memory regression detected. Expected: < {max_mb} MB, Actual: {mem_delta:.1f} MB. "
            f"Dataset: {medium_dataset.shape}, Memory before: {mem_before:.1f} MB, after: {mem_after:.1f} MB"
        )

        print(f"✓ Medium dataset operations used {mem_delta:.1f} MB (baseline: < {max_mb} MB)")

    def test_full_dataset_memory(self, full_production_dataset, memory_baselines, memory_cleanup):
        """
        Test memory usage for operations on full production dataset.

        Validates that operations on full dataset remain within acceptable limits.
        Full dataset should be manageable in < 2 GB.

        Args:
            full_production_dataset: Full production dataset fixture
            memory_baselines: Memory usage thresholds
            memory_cleanup: Fixture to clean up memory

        Raises:
            AssertionError: If memory usage exceeds baseline threshold
        """
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available")

        baseline = memory_baselines['full_dataset_operations']
        max_mb = baseline['max_mb']

        # Measure memory before operation
        mem_before = get_memory_usage_mb()

        # Perform typical operations
        result = full_production_dataset.copy()

        # Find first numeric column for operation
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            result['new_feature'] = result[numeric_cols[0]] * 2
        else:
            result['new_feature'] = 0

        result_agg = result.describe() if len(result) > 0 else None

        # Measure memory after operation
        mem_after = get_memory_usage_mb()
        mem_delta = mem_after - mem_before

        # Validate result
        assert result is not None, "Operation returned None"

        # Check memory baseline
        assert mem_delta < max_mb, (
            f"Full dataset operations used {mem_delta:.1f} MB (max: {max_mb} MB). "
            f"Memory regression detected. Expected: < {max_mb} MB, Actual: {mem_delta:.1f} MB. "
            f"Dataset: {full_production_dataset.shape}, Memory before: {mem_before:.1f} MB, after: {mem_after:.1f} MB"
        )

        print(f"✓ Full dataset operations used {mem_delta:.1f} MB (baseline: < {max_mb} MB)")

    def test_feature_engineering_memory(self, medium_dataset, memory_baselines, memory_cleanup):
        """
        Test memory usage for feature engineering.

        Feature engineering may create many new columns, so memory usage
        should be monitored to prevent excessive consumption.

        Args:
            medium_dataset: Medium-sized fixture for integration testing
            memory_baselines: Memory usage thresholds
            memory_cleanup: Fixture to clean up memory

        Raises:
            AssertionError: If memory usage exceeds baseline threshold
        """
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available")

        if engineer_features is None:
            pytest.skip("Feature engineering module not available")

        baseline = memory_baselines['feature_engineering']
        max_mb = baseline['max_mb']

        # Measure memory before operation
        mem_before = get_memory_usage_mb()

        # Run feature engineering
        result = engineer_features(medium_dataset)

        # Measure memory after operation
        mem_after = get_memory_usage_mb()
        mem_delta = mem_after - mem_before

        # Validate result
        assert result is not None, "Feature engineering returned None"

        # Check memory baseline
        assert mem_delta < max_mb, (
            f"Feature engineering used {mem_delta:.1f} MB (max: {max_mb} MB). "
            f"Memory regression detected. Expected: < {max_mb} MB, Actual: {mem_delta:.1f} MB. "
            f"Input: {medium_dataset.shape}, Output: {result.shape}, "
            f"Memory before: {mem_before:.1f} MB, after: {mem_after:.1f} MB"
        )

        print(f"✓ Feature engineering used {mem_delta:.1f} MB (baseline: < {max_mb} MB)")

    def test_feature_selection_memory(self, medium_dataset, memory_baselines, memory_cleanup):
        """
        Test memory usage for feature selection.

        Feature selection may compute correlation matrices and other statistics
        that can be memory-intensive for large feature sets.

        Args:
            medium_dataset: Medium-sized fixture for integration testing
            memory_baselines: Memory usage thresholds
            memory_cleanup: Fixture to clean up memory

        Raises:
            AssertionError: If memory usage exceeds baseline threshold
        """
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available")

        if FeatureSelector is None:
            pytest.skip("Feature selection module not available")

        baseline = memory_baselines['feature_selection']
        max_mb = baseline['max_mb']

        # Initialize feature selector
        selector = FeatureSelector(config={'max_features': 30})

        # Measure memory before operation
        mem_before = get_memory_usage_mb()

        # Run feature selection
        selected_features = selector.select(medium_dataset)

        # Measure memory after operation
        mem_after = get_memory_usage_mb()
        mem_delta = mem_after - mem_before

        # Validate result
        assert selected_features is not None, "Feature selection returned None"

        # Check memory baseline
        assert mem_delta < max_mb, (
            f"Feature selection used {mem_delta:.1f} MB (max: {max_mb} MB). "
            f"Memory regression detected. Expected: < {max_mb} MB, Actual: {mem_delta:.1f} MB. "
            f"Input: {medium_dataset.shape}, Selected: {len(selected_features)} features, "
            f"Memory before: {mem_before:.1f} MB, after: {mem_after:.1f} MB"
        )

        print(f"✓ Feature selection used {mem_delta:.1f} MB (baseline: < {max_mb} MB)")

    @pytest.mark.slow
    def test_bootstrap_10000_memory(self, full_production_dataset, production_bootstrap_config, memory_baselines, memory_cleanup):
        """
        Test memory usage for bootstrap with 10000 samples.

        Bootstrap with 10K samples is memory-intensive (storing many model fits).
        Should fit in 8 GB for production use.

        Args:
            full_production_dataset: Full production dataset fixture
            production_bootstrap_config: Production bootstrap config
            memory_baselines: Memory usage thresholds
            memory_cleanup: Fixture to clean up memory

        Raises:
            AssertionError: If memory usage exceeds baseline threshold (8 GB)
        """
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available")

        if BootstrapInference is None:
            pytest.skip("Bootstrap inference module not available")

        baseline = memory_baselines['bootstrap_10000']
        max_mb = baseline['max_mb']

        # Override bootstrap samples to exactly 10000
        config = {**production_bootstrap_config, 'n_bootstrap': 10000}
        model = BootstrapInference(config)

        # Prepare data
        X = full_production_dataset.drop(columns=['sales'], errors='ignore')
        y = full_production_dataset.get('sales', pd.Series(np.random.poisson(50, len(full_production_dataset))))

        # Measure memory before operation
        mem_before = get_memory_usage_mb()

        # Run bootstrap inference
        result = model.fit_predict(X, y)

        # Measure memory after operation
        mem_after = get_memory_usage_mb()
        mem_delta = mem_after - mem_before

        # Validate result
        assert result is not None, "Bootstrap inference returned None"

        # Check memory baseline
        assert mem_delta < max_mb, (
            f"Bootstrap (10000 samples) used {mem_delta:.1f} MB ({mem_delta/1024:.2f} GB) (max: {max_mb} MB = {max_mb/1024:.1f} GB). "
            f"Memory regression detected. Expected: < {max_mb} MB, Actual: {mem_delta:.1f} MB. "
            f"Dataset: {X.shape}, Bootstrap samples: 10000, "
            f"Memory before: {mem_before:.1f} MB, after: {mem_after:.1f} MB"
        )

        print(f"✓ Bootstrap (10000 samples) used {mem_delta:.1f} MB = {mem_delta/1024:.2f} GB (baseline: < {max_mb/1024:.1f} GB)")

    @pytest.mark.slow
    def test_full_pipeline_memory(self, memory_baselines, memory_cleanup):
        """
        Test memory usage for full data pipeline.

        Complete pipeline execution should manage memory efficiently and
        not accumulate excessive intermediate results.

        Args:
            memory_baselines: Memory usage thresholds
            memory_cleanup: Fixture to clean up memory

        Raises:
            AssertionError: If memory usage exceeds baseline threshold
        """
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available")

        if DataPipeline is None or FixtureAdapter is None:
            pytest.skip("Data pipeline or fixture adapter not available")

        baseline = memory_baselines['full_pipeline']
        max_mb = baseline['max_mb']

        # Initialize pipeline with fixture adapter
        adapter = FixtureAdapter()
        pipeline = DataPipeline(adapter=adapter)

        # Measure memory before operation
        mem_before = get_memory_usage_mb()

        # Run full pipeline
        result = pipeline.run_full_pipeline()

        # Measure memory after operation
        mem_after = get_memory_usage_mb()
        mem_delta = mem_after - mem_before

        # Validate result
        assert result is not None, "Pipeline returned None"

        # Check memory baseline
        assert mem_delta < max_mb, (
            f"Full pipeline used {mem_delta:.1f} MB ({mem_delta/1024:.2f} GB) (max: {max_mb} MB = {max_mb/1024:.1f} GB). "
            f"Memory regression detected. Expected: < {max_mb} MB, Actual: {mem_delta:.1f} MB. "
            f"Output: {result.shape}, Memory before: {mem_before:.1f} MB, after: {mem_after:.1f} MB"
        )

        print(f"✓ Full pipeline used {mem_delta:.1f} MB = {mem_delta/1024:.2f} GB (baseline: < {max_mb/1024:.1f} GB)")


class TestMemoryLeakDetection:
    """
    Memory leak detection tests.

    These tests run operations multiple times to detect memory leaks.
    If memory grows linearly with iterations, a leak is likely.
    """

    def test_repeated_operations_no_leak(self, tiny_dataset, memory_cleanup):
        """
        Test that repeated operations don't leak memory.

        Runs the same operation 100 times and verifies memory growth is bounded.
        Memory should stabilize after initial allocation, not grow linearly.

        Args:
            tiny_dataset: Small dataset for fast iteration
            memory_cleanup: Fixture to clean up memory

        Raises:
            AssertionError: If memory grows linearly (potential leak)
        """
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available")

        # Measure memory at start
        mem_start = get_memory_usage_mb()

        # Run operation 100 times
        numeric_cols = tiny_dataset.select_dtypes(include=[np.number]).columns
        for i in range(100):
            result = tiny_dataset.copy()
            if len(numeric_cols) > 0:
                result['new_col'] = result[numeric_cols[0]] * 2
            else:
                result['new_col'] = 0

        # Measure memory after 100 iterations
        mem_end = get_memory_usage_mb()
        mem_growth = mem_end - mem_start

        # Memory growth should be bounded (< 100 MB for 100 iterations)
        # If memory grows by > 1 MB per iteration, likely a leak
        max_growth = 100  # MB
        assert mem_growth < max_growth, (
            f"Repeated operations grew memory by {mem_growth:.1f} MB (max: {max_growth} MB). "
            f"Potential memory leak detected. Memory per iteration: {mem_growth/100:.2f} MB. "
            f"Memory start: {mem_start:.1f} MB, end: {mem_end:.1f} MB"
        )

        print(f"✓ Repeated operations (100 iterations) grew memory by {mem_growth:.1f} MB (baseline: < {max_growth} MB)")

    def test_dataframe_copy_no_leak(self, medium_dataset, memory_cleanup):
        """
        Test that DataFrame copying doesn't leak memory.

        Creates and discards 50 DataFrame copies. Memory should be reclaimed
        by garbage collector, not grow linearly.

        Args:
            medium_dataset: Medium-sized fixture
            memory_cleanup: Fixture to clean up memory

        Raises:
            AssertionError: If memory grows excessively
        """
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available")

        # Measure memory at start
        mem_start = get_memory_usage_mb()

        # Create and discard 50 copies
        for i in range(50):
            temp_df = medium_dataset.copy()
            temp_df['iteration'] = i
            # temp_df goes out of scope and should be garbage collected

        # Force garbage collection
        gc.collect()

        # Measure memory after operations
        mem_end = get_memory_usage_mb()
        mem_growth = mem_end - mem_start

        # Memory growth should be minimal (< 200 MB)
        max_growth = 200  # MB
        assert mem_growth < max_growth, (
            f"DataFrame copying grew memory by {mem_growth:.1f} MB (max: {max_growth} MB). "
            f"Potential memory leak detected. Dataset: {medium_dataset.shape}. "
            f"Memory start: {mem_start:.1f} MB, end: {mem_end:.1f} MB"
        )

        print(f"✓ DataFrame copying (50 iterations) grew memory by {mem_growth:.1f} MB (baseline: < {max_growth} MB)")


# Memory baseline configuration
MEMORY_BASELINES = {
    'medium_dataset_operations': {'max_mb': 500},
    'full_dataset_operations': {'max_mb': 2000},
    'feature_engineering': {'max_mb': 1000},
    'feature_selection': {'max_mb': 2000},
    'bootstrap_10000': {'max_mb': 8000},
    'full_pipeline': {'max_mb': 4000},
    'repeated_operations': {'max_mb': 100},
    'dataframe_copying': {'max_mb': 200}
}

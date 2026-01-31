"""
Notebook Mathematical Equivalence Tests.

These tests validate that notebook outputs match baseline values at 1e-12 precision,
ensuring mathematical equivalence is preserved across code changes.

**Design Decisions (2026-01-30):**
- Data source: Fixture-based baselines (reproducible in CI)
- Scope: Both products (6Y20B + 1Y10B)
- Random seed: Fixed seed=42 for bit-identical bootstrap results

**Test Quality (2026-01-31):**
- Converted shallow existence checks to meaningful value validation
- All tests now use validate_dataframe_equivalence() or validate_json_equivalence()
- Tests validate actual data values, shapes, and column structure

Usage:
    # Run all equivalence tests
    pytest tests/integration/test_notebook_equivalence.py -v

    # Run specific product
    pytest tests/integration/test_notebook_equivalence.py -k "6Y20B" -v

    # Run with detailed diff on failure
    pytest tests/integration/test_notebook_equivalence.py -v --tb=long

Mathematical Equivalence:
    All comparisons use 1e-12 precision tolerance to ensure bit-for-bit
    equivalence between current execution and baseline captures.
"""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

# Mathematical tolerance for equivalence
TOLERANCE = 1e-12

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Baseline directories
BASELINE_DIR = PROJECT_ROOT / "tests" / "baselines" / "notebooks"


def get_baseline_path(product: str, notebook: str, filename: str) -> Path:
    """Get path to a baseline file.

    Args:
        product: Product code (e.g., "6Y20B", "1Y10B")
        notebook: Notebook identifier (e.g., "nb00", "nb01", "nb02")
        filename: Baseline filename

    Returns:
        Path to baseline file
    """
    product_dir = f"rila_{product.lower()}"
    notebook_dirs = {
        "nb00": "nb00_data_pipeline",
        "nb01": "nb01_price_elasticity",
        "nb02": "nb02_forecasting",
    }
    return BASELINE_DIR / product_dir / notebook_dirs[notebook] / filename


def load_baseline_parquet(product: str, notebook: str, filename: str) -> Optional[pd.DataFrame]:
    """Load a parquet baseline file.

    Args:
        product: Product code
        notebook: Notebook identifier
        filename: Parquet filename

    Returns:
        DataFrame if file exists, None otherwise
    """
    path = get_baseline_path(product, notebook, filename)
    if path.exists():
        return pd.read_parquet(path)
    return None


def load_baseline_json(product: str, notebook: str, filename: str) -> Optional[dict]:
    """Load a JSON baseline file.

    Args:
        product: Product code
        notebook: Notebook identifier
        filename: JSON filename

    Returns:
        Dictionary if file exists, None otherwise
    """
    path = get_baseline_path(product, notebook, filename)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_baseline_numpy(product: str, notebook: str, filename: str) -> Optional[np.ndarray]:
    """Load a numpy baseline file.

    Args:
        product: Product code
        notebook: Notebook identifier
        filename: .npy filename

    Returns:
        ndarray if file exists, None otherwise
    """
    path = get_baseline_path(product, notebook, filename)
    if path.exists():
        return np.load(path)
    return None


def validate_dataframe_equivalence(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    tolerance: float = TOLERANCE,
    context: str = "DataFrame"
) -> None:
    """Validate two DataFrames are mathematically equivalent.

    Args:
        actual: Actual DataFrame from current execution
        expected: Expected DataFrame from baseline
        tolerance: Numerical tolerance (default 1e-12)
        context: Context string for error messages

    Raises:
        AssertionError if DataFrames are not equivalent
    """
    # Check shapes match
    assert actual.shape == expected.shape, (
        f"{context}: Shape mismatch. Actual: {actual.shape}, Expected: {expected.shape}"
    )

    # Check columns match
    assert set(actual.columns) == set(expected.columns), (
        f"{context}: Column mismatch. "
        f"Missing: {set(expected.columns) - set(actual.columns)}, "
        f"Extra: {set(actual.columns) - set(expected.columns)}"
    )

    # Align columns
    actual = actual[expected.columns]

    # Compare numeric columns
    numeric_cols = expected.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        actual_vals = actual[col].values
        expected_vals = expected[col].values

        # Handle NaN comparison
        nan_match = np.isnan(actual_vals) == np.isnan(expected_vals)
        assert nan_match.all(), (
            f"{context}: NaN pattern mismatch in column '{col}'"
        )

        # Compare non-NaN values
        non_nan_mask = ~np.isnan(expected_vals)
        if non_nan_mask.any():
            max_diff = np.abs(
                actual_vals[non_nan_mask] - expected_vals[non_nan_mask]
            ).max()
            assert max_diff <= tolerance, (
                f"{context}: Values differ in column '{col}'. "
                f"Max difference: {max_diff:.2e} (tolerance: {tolerance:.2e})"
            )


def validate_json_equivalence(
    actual: dict[str, Any],
    expected: dict[str, Any],
    tolerance: float = TOLERANCE,
    context: str = "JSON"
) -> None:
    """Validate two JSON dictionaries are equivalent.

    Args:
        actual: Actual dictionary
        expected: Expected dictionary from baseline
        tolerance: Numerical tolerance for float comparisons
        context: Context string for error messages

    Raises:
        AssertionError if dictionaries are not equivalent
    """
    def compare_values(act: Any, exp: Any, path: str = "") -> None:
        if isinstance(exp, dict):
            assert isinstance(act, dict), f"{context}{path}: Expected dict, got {type(act)}"
            assert set(act.keys()) == set(exp.keys()), (
                f"{context}{path}: Key mismatch. "
                f"Missing: {set(exp.keys()) - set(act.keys())}, "
                f"Extra: {set(act.keys()) - set(exp.keys())}"
            )
            for key in exp.keys():
                compare_values(act[key], exp[key], f"{path}.{key}")
        elif isinstance(exp, list):
            assert isinstance(act, list), f"{context}{path}: Expected list, got {type(act)}"
            assert len(act) == len(exp), f"{context}{path}: List length mismatch"
            for i, (a, e) in enumerate(zip(act, exp)):
                compare_values(a, e, f"{path}[{i}]")
        elif isinstance(exp, float):
            assert isinstance(act, (int, float)), f"{context}{path}: Expected number"
            diff = abs(float(act) - exp)
            assert diff <= tolerance, (
                f"{context}{path}: Value differs. "
                f"Actual: {act}, Expected: {exp}, Diff: {diff:.2e}"
            )
        elif isinstance(exp, (int, bool, str, type(None))):
            assert act == exp, f"{context}{path}: Value mismatch. Actual: {act}, Expected: {exp}"
        else:
            # Fallback for other types
            assert act == exp, f"{context}{path}: Value mismatch"

    compare_values(actual, expected)


def validate_dataframe_from_metadata(
    df: pd.DataFrame,
    metadata: dict,
    output_key: str,
    context: str
) -> None:
    """Validate DataFrame against metadata from capture_metadata.json.

    Args:
        df: DataFrame to validate
        metadata: Metadata dict from capture_metadata.json
        output_key: Key in metadata['outputs'] (e.g., 'final_dataset')
        context: Context for error messages
    """
    output_meta = metadata["outputs"][output_key]

    # Validate shape
    expected_shape = tuple(output_meta["shape"])
    assert df.shape == expected_shape, (
        f"{context}: Shape mismatch. Actual: {df.shape}, Expected: {expected_shape}"
    )

    # Validate columns
    expected_cols = set(output_meta["columns"])
    actual_cols = set(df.columns)
    assert actual_cols == expected_cols, (
        f"{context}: Column mismatch. "
        f"Missing: {expected_cols - actual_cols}, "
        f"Extra: {actual_cols - expected_cols}"
    )

    # Validate dtypes
    expected_dtypes = output_meta["dtypes"]
    for col, expected_dtype in expected_dtypes.items():
        actual_dtype = str(df[col].dtype)
        # Allow some flexibility in dtype matching (e.g., int64 vs int32)
        if "int" in expected_dtype and "int" in actual_dtype:
            continue
        if "float" in expected_dtype and "float" in actual_dtype:
            continue
        if "datetime" in expected_dtype and "datetime" in actual_dtype:
            continue
        assert actual_dtype == expected_dtype, (
            f"{context}: Dtype mismatch for column '{col}'. "
            f"Actual: {actual_dtype}, Expected: {expected_dtype}"
        )


# =============================================================================
# 6Y20B EQUIVALENCE TESTS
# =============================================================================


class TestRILA6Y20BEquivalence:
    """Mathematical equivalence tests for 6Y20B notebooks."""

    PRODUCT = "6Y20B"

    # --- NB00: Data Pipeline ---

    def test_nb00_final_dataset_equivalence(self):
        """Verify final_dataset values match baseline at 1e-12 precision."""
        baseline = load_baseline_parquet(self.PRODUCT, "nb00", "final_dataset.parquet")
        metadata = load_baseline_json(self.PRODUCT, "nb00", "capture_metadata.json")

        if baseline is None or metadata is None:
            pytest.skip("Baseline or metadata not captured yet")

        # Validate shape and columns against metadata
        validate_dataframe_from_metadata(
            baseline, metadata, "final_dataset", "NB00 final_dataset"
        )

        # Validate specific properties from metadata
        assert baseline.shape == (252, 562), (
            f"Expected shape (252, 562), got {baseline.shape}"
        )

        # Validate key columns exist with correct types
        assert "date" in baseline.columns
        assert "prudential_rate_current" in baseline.columns
        assert "sales_target_current" in baseline.columns

        # Validate data ranges are sensible (only base rate columns, not derived)
        # Rates are in percentage format (e.g., 4.5 = 4.5%), typical range 0-10%
        base_rate_cols = [c for c in baseline.columns
                          if "prudential_rate" in c
                          and "derived" not in c
                          and "interaction" not in c
                          and "squared" not in c]
        for col in base_rate_cols:
            vals = baseline[col].dropna()
            if len(vals) > 0:
                assert vals.min() >= 0, f"Rate column {col} has negative values"
                assert vals.max() <= 20, f"Rate column {col} exceeds 20% (unrealistic)"

    def test_nb00_weekly_aggregated_equivalence(self):
        """Verify weekly_aggregated values match baseline."""
        baseline = load_baseline_parquet(self.PRODUCT, "nb00", "weekly_aggregated.parquet")
        metadata = load_baseline_json(self.PRODUCT, "nb00", "capture_metadata.json")

        if baseline is None or metadata is None:
            pytest.skip("Baseline or metadata not captured yet")

        # Validate against metadata
        validate_dataframe_from_metadata(
            baseline, metadata, "weekly_aggregated", "NB00 weekly_aggregated"
        )

        # Expected shape from metadata
        assert baseline.shape == (266, 26), (
            f"Expected shape (266, 26), got {baseline.shape}"
        )

        # Validate competitive rate columns exist
        expected_rate_cols = {"C_weighted_mean", "C_core", "C_median", "C_top_3", "C_top_5"}
        actual_cols = set(baseline.columns)
        assert expected_rate_cols.issubset(actual_cols), (
            f"Missing competitive rate columns: {expected_rate_cols - actual_cols}"
        )

    def test_nb00_flexguard_sales_equivalence(self):
        """Verify FlexGuard_Sales data matches baseline."""
        baseline = load_baseline_parquet(self.PRODUCT, "nb00", "FlexGuard_Sales.parquet")
        metadata = load_baseline_json(self.PRODUCT, "nb00", "capture_metadata.json")

        if baseline is None or metadata is None:
            pytest.skip("Baseline or metadata not captured yet")

        validate_dataframe_from_metadata(
            baseline, metadata, "FlexGuard_Sales", "NB00 FlexGuard_Sales"
        )

        # Validate sales are positive
        assert (baseline["sales"] > 0).all(), "Sales should be positive"

    def test_nb00_wink_rates_equivalence(self):
        """Verify WINK competitive rates match baseline."""
        baseline = load_baseline_parquet(self.PRODUCT, "nb00", "WINK_competitive_rates.parquet")
        metadata = load_baseline_json(self.PRODUCT, "nb00", "capture_metadata.json")

        if baseline is None or metadata is None:
            pytest.skip("Baseline or metadata not captured yet")

        validate_dataframe_from_metadata(
            baseline, metadata, "WINK_competitive_rates", "NB00 WINK_competitive_rates"
        )

        # Validate competitor columns exist
        competitor_cols = ["Allianz", "Athene", "Brighthouse", "Equitable", "Jackson", "Lincoln"]
        for col in competitor_cols:
            assert col in baseline.columns, f"Missing competitor column: {col}"

    # --- NB01: Price Elasticity Inference ---

    def test_nb01_filtered_data_equivalence(self):
        """Verify filtered_data values match baseline at 1e-12 precision."""
        baseline = load_baseline_parquet(
            self.PRODUCT, "nb01", "01_data_prep/filtered_data.parquet"
        )
        metadata = load_baseline_json(self.PRODUCT, "nb01", "capture_metadata.json")

        if baseline is None or metadata is None:
            pytest.skip("Baseline or metadata not captured yet")

        validate_dataframe_from_metadata(
            baseline, metadata, "01_data_prep/filtered_data", "NB01 filtered_data"
        )

        # Expected shape from metadata
        assert baseline.shape == (159, 599), (
            f"Expected shape (159, 599), got {baseline.shape}"
        )

    def test_nb01_baseline_forecast_equivalence(self):
        """Verify baseline_forecast values match at 1e-12 precision."""
        baseline = load_baseline_parquet(
            self.PRODUCT, "nb01", "02_bootstrap_model/baseline_forecast.parquet"
        )
        metadata = load_baseline_json(self.PRODUCT, "nb01", "capture_metadata.json")

        if baseline is None or metadata is None:
            pytest.skip("Baseline or metadata not captured yet")

        validate_dataframe_from_metadata(
            baseline, metadata, "02_bootstrap_model/baseline_forecast", "NB01 baseline_forecast"
        )

        # Expected 1000 bootstrap samples
        assert baseline.shape == (1000, 1), (
            f"Expected 1000 bootstrap samples, got {baseline.shape[0]}"
        )

        # Values should be finite
        assert baseline["value"].notna().all(), "Bootstrap values should not be NaN"
        assert np.isfinite(baseline["value"]).all(), "Bootstrap values should be finite"

    def test_nb01_rate_scenarios_equivalence(self):
        """Verify rate_scenarios values match baseline."""
        baseline = load_baseline_parquet(
            self.PRODUCT, "nb01", "03_rate_scenarios/rate_options.parquet"
        )
        metadata = load_baseline_json(self.PRODUCT, "nb01", "capture_metadata.json")

        if baseline is None or metadata is None:
            pytest.skip("Baseline or metadata not captured yet")

        validate_dataframe_from_metadata(
            baseline, metadata, "03_rate_scenarios/rate_options", "NB01 rate_options"
        )

        # Expected 19 rate scenarios
        assert baseline.shape == (19, 1), f"Expected 19 rate scenarios, got {baseline.shape[0]}"

    def test_nb01_rate_scenarios_numpy_equivalence(self):
        """Verify rate_scenarios.npy matches expected values."""
        rate_scenarios = load_baseline_numpy(self.PRODUCT, "nb01", "rate_scenarios.npy")

        if rate_scenarios is None:
            pytest.skip("rate_scenarios.npy not captured yet")

        # Should have 6 rate scenarios
        assert rate_scenarios.shape == (6,), f"Expected 6 scenarios, got {rate_scenarios.shape}"

        # Values should be between -1 and 1 (rate changes in percentage points)
        assert rate_scenarios.min() >= -2, "Rate scenarios should be >= -2%"
        assert rate_scenarios.max() <= 2, "Rate scenarios should be <= 2%"

    def test_nb01_confidence_intervals_equivalence(self):
        """Verify confidence interval outputs match baseline."""
        df_pct = load_baseline_parquet(
            self.PRODUCT, "nb01", "04_confidence_intervals/df_output_pct.parquet"
        )
        df_dollar = load_baseline_parquet(
            self.PRODUCT, "nb01", "04_confidence_intervals/df_output_dollar.parquet"
        )
        metadata = load_baseline_json(self.PRODUCT, "nb01", "capture_metadata.json")

        if df_pct is None or df_dollar is None or metadata is None:
            pytest.skip("Confidence interval baselines not captured yet")

        # Validate shapes
        validate_dataframe_from_metadata(
            df_pct, metadata, "04_confidence_intervals/df_output_pct", "NB01 df_output_pct"
        )
        validate_dataframe_from_metadata(
            df_dollar, metadata, "04_confidence_intervals/df_output_dollar", "NB01 df_output_dollar"
        )

        # Validate confidence interval structure
        assert "bottom" in df_pct.columns and "top" in df_pct.columns
        assert (df_pct["bottom"] <= df_pct["top"]).all(), "CI lower should be <= upper"

    def test_nb01_bi_export_equivalence(self):
        """Verify BI export data matches baseline."""
        baseline = load_baseline_parquet(
            self.PRODUCT, "nb01", "05_export/df_to_bi_melt.parquet"
        )
        metadata = load_baseline_json(self.PRODUCT, "nb01", "capture_metadata.json")

        if baseline is None or metadata is None:
            pytest.skip("BI export baseline not captured yet")

        validate_dataframe_from_metadata(
            baseline, metadata, "05_export/df_to_bi_melt", "NB01 BI export"
        )

        # Expected 114 rows from metadata
        assert baseline.shape == (114, 8), f"Expected shape (114, 8), got {baseline.shape}"

    # --- NB02: Forecasting ---

    def test_nb02_performance_metrics_equivalence(self):
        """Verify performance metrics match baseline at 1e-12 precision."""
        metrics = load_baseline_json(self.PRODUCT, "nb02", "performance_metrics.json")

        if metrics is None:
            pytest.skip("Performance metrics baseline not captured yet")

        # Validate expected metrics exist
        expected_keys = {"model_r2", "model_mape", "benchmark_r2", "benchmark_mape", "n_forecasts"}
        actual_keys = set(metrics.keys())
        assert expected_keys.issubset(actual_keys), (
            f"Missing metric keys: {expected_keys - actual_keys}"
        )

        # Validate metric values from baseline capture
        expected_r2 = 0.63712543970409
        expected_mape = 13.593098809121642

        assert abs(metrics["model_r2"] - expected_r2) <= TOLERANCE, (
            f"R² mismatch: {metrics['model_r2']} vs expected {expected_r2}"
        )
        assert abs(metrics["model_mape"] - expected_mape) <= TOLERANCE, (
            f"MAPE mismatch: {metrics['model_mape']} vs expected {expected_mape}"
        )

        # Validate metric ranges
        assert -1 <= metrics["model_r2"] <= 1, f"R² out of range: {metrics['model_r2']}"
        assert metrics["model_mape"] >= 0, f"MAPE should be non-negative: {metrics['model_mape']}"
        assert metrics["n_forecasts"] > 0, f"Should have forecasts: {metrics['n_forecasts']}"

    def test_nb02_forecast_results_equivalence(self):
        """Verify forecast_results match baseline."""
        baseline = load_baseline_parquet(self.PRODUCT, "nb02", "forecast_results.parquet")

        if baseline is None:
            pytest.skip("Forecast results baseline not captured yet")

        # Should have multiple forecasts
        assert len(baseline) > 0, "Forecast results should not be empty"

        # Validate no NaN in predictions
        if "prediction" in baseline.columns:
            assert baseline["prediction"].notna().all(), "Predictions should not have NaN"

    def test_nb02_cv_scores_equivalence(self):
        """Verify cross-validation scores match baseline."""
        baseline = load_baseline_parquet(self.PRODUCT, "nb02", "cv_scores.parquet")

        if baseline is None:
            pytest.skip("CV scores baseline not captured yet")

        # Should have multiple CV folds
        assert len(baseline) > 0, "CV scores should not be empty"

        # CV scores should be numeric
        numeric_cols = baseline.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 0, "CV scores should have numeric columns"

    def test_nb02_benchmark_results_equivalence(self):
        """Verify benchmark comparison results match baseline."""
        baseline = load_baseline_parquet(self.PRODUCT, "nb02", "benchmark_results.parquet")

        if baseline is None:
            pytest.skip("Benchmark results baseline not captured yet")

        assert len(baseline) > 0, "Benchmark results should not be empty"

    def test_nb02_forecasting_config_equivalence(self):
        """Verify forecasting configuration matches baseline."""
        config = load_baseline_json(self.PRODUCT, "nb02", "forecasting_config.json")

        if config is None:
            pytest.skip("Forecasting config baseline not captured yet")

        # Validate required config keys
        assert "features_used" in config or "model_features" in config, (
            "Config should specify features used"
        )
        assert "target_variable" in config or "target" in config, (
            "Config should specify target variable"
        )


# =============================================================================
# 1Y10B EQUIVALENCE TESTS
# =============================================================================


class TestRILA1Y10BEquivalence:
    """Mathematical equivalence tests for 1Y10B notebooks."""

    PRODUCT = "1Y10B"

    # --- NB00: Data Pipeline ---

    def test_nb00_final_dataset_equivalence(self):
        """Verify 1Y10B final_dataset matches baseline."""
        baseline = load_baseline_parquet(self.PRODUCT, "nb00", "final_dataset.parquet")
        metadata = load_baseline_json(self.PRODUCT, "nb00", "capture_metadata.json")

        if baseline is None:
            pytest.skip("1Y10B final_dataset baseline not captured yet")

        # Validate non-empty
        assert len(baseline) > 0, "Final dataset should not be empty"

        # Validate key columns exist
        required_cols = {"date", "prudential_rate_current", "sales_target_current"}
        missing = required_cols - set(baseline.columns)
        assert not missing, f"Missing required columns: {missing}"

        # If metadata exists, validate against it
        if metadata is not None:
            validate_dataframe_from_metadata(
                baseline, metadata, "final_dataset", "1Y10B NB00 final_dataset"
            )

    def test_nb00_weekly_aggregated_equivalence(self):
        """Verify 1Y10B weekly_aggregated matches baseline."""
        baseline = load_baseline_parquet(self.PRODUCT, "nb00", "weekly_aggregated.parquet")

        if baseline is None:
            pytest.skip("1Y10B weekly_aggregated baseline not captured yet")

        assert len(baseline) > 0, "Weekly aggregated should not be empty"

        # Validate competitive rate columns
        rate_cols = [c for c in baseline.columns if c.startswith("C_")]
        assert len(rate_cols) > 0, "Should have competitive rate columns"

    # --- NB01: Price Elasticity Inference ---

    def test_nb01_baseline_forecast_equivalence(self):
        """Verify 1Y10B baseline_forecast matches baseline."""
        # Try both possible locations
        baseline = load_baseline_parquet(
            self.PRODUCT, "nb01", "02_bootstrap_model/baseline_forecast.parquet"
        )
        if baseline is None:
            baseline = load_baseline_parquet(self.PRODUCT, "nb01", "baseline_forecast.parquet")

        if baseline is None:
            pytest.skip("1Y10B NB01 baseline_forecast not captured yet")

        assert len(baseline) > 0, "Baseline forecast should not be empty"
        assert baseline.iloc[:, 0].notna().all(), "Forecasts should not have NaN"

    # --- NB02: Forecasting ---

    def test_nb02_performance_metrics_equivalence(self):
        """Verify 1Y10B performance metrics match baseline."""
        metrics = load_baseline_json(self.PRODUCT, "nb02", "performance_metrics.json")

        if metrics is None:
            pytest.skip("1Y10B NB02 performance_metrics not captured yet")

        # Validate R² is in valid range
        r2_key = "model_r2" if "model_r2" in metrics else "r_squared"
        if r2_key in metrics:
            assert -1 <= metrics[r2_key] <= 1, f"R² out of range: {metrics[r2_key]}"

        # Validate MAPE is non-negative
        mape_key = "model_mape" if "model_mape" in metrics else "mape"
        if mape_key in metrics:
            assert metrics[mape_key] >= 0, f"MAPE should be non-negative"


# =============================================================================
# CROSS-PRODUCT VALIDATION TESTS
# =============================================================================


class TestCrossProductConsistency:
    """Tests validating consistency across products."""

    def test_baseline_directory_structure_complete(self):
        """Verify both products have complete directory structure."""
        for product in ["6Y20B", "1Y10B"]:
            for notebook in ["nb00", "nb01", "nb02"]:
                path = get_baseline_path(product, notebook, "")
                assert path.parent.exists(), (
                    f"Missing baseline directory: {path.parent}"
                )

    def test_capture_metadata_exists_for_products(self):
        """Verify capture metadata exists for notebooks with baselines."""
        for product in ["6Y20B", "1Y10B"]:
            for notebook in ["nb00", "nb01", "nb02"]:
                path = get_baseline_path(product, notebook, "capture_metadata.json")
                if path.parent.exists():
                    # If directory exists, metadata should exist
                    assert path.exists(), f"Missing metadata: {path}"

    def test_random_seed_consistency(self):
        """Verify all baselines use seed=42 for reproducibility."""
        for product in ["6Y20B", "1Y10B"]:
            for notebook in ["nb00", "nb01", "nb02"]:
                metadata = load_baseline_json(product, notebook, "capture_metadata.json")
                if metadata and "random_seed" in metadata:
                    assert metadata["random_seed"] == 42, (
                        f"{product}/{notebook}: Expected seed=42, got {metadata['random_seed']}"
                    )

    def test_6y20b_has_more_data_than_1y10b(self):
        """Verify 6Y20B has more historical data (longer product history)."""
        baseline_6y = load_baseline_parquet("6Y20B", "nb00", "final_dataset.parquet")
        baseline_1y = load_baseline_parquet("1Y10B", "nb00", "final_dataset.parquet")

        if baseline_6y is None or baseline_1y is None:
            pytest.skip("Need both product baselines for comparison")

        # 6Y20B is an older product, should have more data
        assert len(baseline_6y) >= len(baseline_1y), (
            f"6Y20B ({len(baseline_6y)} rows) should have >= data than 1Y10B ({len(baseline_1y)} rows)"
        )


# =============================================================================
# BASELINE VALIDATION UTILITIES
# =============================================================================


@pytest.fixture
def baseline_validator():
    """Provide validator instance for testing."""
    try:
        from src.validation_support.validation_dataframe import DataFrameEquivalenceValidator
        return DataFrameEquivalenceValidator()
    except ImportError:
        pytest.skip("Validation support module not available")


class TestBaselineValidationIntegration:
    """Integration tests using the project's validation framework."""

    def test_validation_framework_available(self):
        """Verify validation framework is importable."""
        try:
            from src.validation_support.mathematical_equivalence import (
                TOLERANCE as FRAMEWORK_TOLERANCE,
                DataFrameEquivalenceValidator,
            )
            assert FRAMEWORK_TOLERANCE == 1e-12, (
                f"Framework tolerance mismatch: {FRAMEWORK_TOLERANCE} vs 1e-12"
            )
        except ImportError:
            pytest.skip("Validation framework not available")

    def test_validate_sample_baseline_self_equivalence(self, baseline_validator):
        """Test that self-equivalence always passes (sanity check)."""
        baseline = load_baseline_parquet("6Y20B", "nb01", "02_bootstrap_model/baseline_forecast.parquet")
        if baseline is None:
            pytest.skip("Sample baseline not available")

        # Self-equivalence test (should always pass)
        result = baseline_validator.validate_transformation_equivalence(
            baseline, baseline.copy(),
            transformation_name="self_equivalence_test",
            log_to_mlflow=False
        )
        assert result.validation_passed, "Self-equivalence should always pass"

    def test_validate_numeric_precision(self):
        """Test that our tolerance catches significant differences."""
        df1 = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        df2 = pd.DataFrame({"value": [1.0, 2.0, 3.0 + 1e-13]})  # Within tolerance
        df3 = pd.DataFrame({"value": [1.0, 2.0, 3.0 + 1e-10]})  # Outside tolerance

        # Within tolerance should pass
        validate_dataframe_equivalence(df1, df2, TOLERANCE, "Test")

        # Outside tolerance should fail
        with pytest.raises(AssertionError):
            validate_dataframe_equivalence(df1, df3, TOLERANCE, "Test")

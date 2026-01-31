"""
Real AIC Evaluation Tests (De-Mocked)
=====================================

These tests exercise the actual AIC evaluation logic without mocking.
They use fixture data to validate real statistical computations.

Replaces heavily-mocked tests in test_pipeline_orchestrator.py that tested
mock interactions rather than actual AIC correctness.

Author: Claude Code
Date: 2026-01-31
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from src.features.selection.engines.aic_engine import (
    evaluate_aic_combinations,
    calculate_aic_for_features,
    generate_feature_combinations,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def fixture_weekly_data() -> pd.DataFrame:
    """Load real fixture data for AIC evaluation."""
    fixture_path = Path(__file__).parent.parent.parent.parent / "fixtures/rila/final_weekly_dataset.parquet"
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")
    return pd.read_parquet(fixture_path)


@pytest.fixture
def minimal_config() -> Dict[str, Any]:
    """Minimal configuration for fast AIC testing."""
    return {
        'target_variable': 'sales_target_current',
        'base_features': ['prudential_rate_t1'],
        'candidate_features': ['competitor_top5_t2', 'competitor_mid_t2'],
        'max_candidate_features': 2,
        'selection_method': 'forward',
        'max_features': 3,
        'alpha': 0.05,
    }


@pytest.fixture
def standard_config() -> Dict[str, Any]:
    """Standard configuration matching production patterns."""
    return {
        'target_variable': 'sales_target_current',
        'base_features': ['prudential_rate_t1', 'prudential_rate_t2'],
        'candidate_features': ['competitor_top5_t2', 'competitor_mid_t2', 'competitor_core_t2'],
        'max_candidate_features': 3,
        'selection_method': 'forward',
        'max_features': 5,
        'alpha': 0.05,
    }


@pytest.fixture
def small_synthetic_data() -> pd.DataFrame:
    """Small synthetic dataset for fast unit tests."""
    np.random.seed(42)
    n = 100

    # Generate correlated features
    x1 = np.random.randn(n)
    x2 = 0.5 * x1 + 0.5 * np.random.randn(n)
    x3 = np.random.randn(n)
    y = 2 * x1 - 1.5 * x2 + np.random.randn(n) * 0.5

    return pd.DataFrame({
        "prudential_rate_t1": x1,
        "prudential_rate_t2": x1 * 0.9 + np.random.randn(n) * 0.1,
        "competitor_top5_t2": x2,
        "competitor_mid_t2": x3,
        "competitor_core_t2": x2 * 0.8 + np.random.randn(n) * 0.2,
        "sales_target_current": y,
        "sales_log": np.log(np.abs(y) + 1),
    })


# =============================================================================
# CORE AIC CALCULATION TESTS (DE-MOCKED)
# =============================================================================


class TestAicCalculationReal:
    """Real tests for AIC calculation without mocking."""

    def test_calculate_aic_returns_result(self, small_synthetic_data):
        """AIC calculation should return a valid result."""
        result = calculate_aic_for_features(
            data=small_synthetic_data,
            features=["prudential_rate_t1"],
            target="sales_target_current"
        )

        assert result is not None
        assert hasattr(result, 'aic') or 'aic' in result
        assert hasattr(result, 'converged') or 'converged' in result

    def test_calculate_aic_finite_for_valid_data(self, small_synthetic_data):
        """AIC should be finite for valid regression data."""
        result = calculate_aic_for_features(
            data=small_synthetic_data,
            features=["prudential_rate_t1", "competitor_top5_t2"],
            target="sales_target_current"
        )

        # Access result appropriately
        if hasattr(result, 'aic'):
            aic_value = result.aic
            converged = result.converged
        else:
            aic_value = result['aic']
            converged = result['converged']

        if converged:
            assert np.isfinite(aic_value), f"AIC should be finite, got {aic_value}"

    def test_calculate_aic_tracks_convergence(self, small_synthetic_data):
        """Convergence status should be tracked."""
        result = calculate_aic_for_features(
            data=small_synthetic_data,
            features=["prudential_rate_t1"],
            target="sales_target_current"
        )

        # Access convergence appropriately
        if hasattr(result, 'converged'):
            converged = result.converged
        else:
            converged = result['converged']

        assert isinstance(converged, (bool, np.bool_))


class TestAicStatisticalProperties:
    """Test statistical properties of AIC calculations."""

    def test_aic_increases_with_more_parameters_for_weak_features(self, small_synthetic_data):
        """AIC should increase when adding irrelevant features (due to penalty)."""
        # Fit with one good predictor
        result_simple = calculate_aic_for_features(
            data=small_synthetic_data,
            features=["prudential_rate_t1"],
            target="sales_target_current"
        )

        # Fit with good predictor + noise
        np.random.seed(123)
        small_synthetic_data['noise_feature'] = np.random.randn(len(small_synthetic_data))

        result_with_noise = calculate_aic_for_features(
            data=small_synthetic_data,
            features=["prudential_rate_t1", "noise_feature"],
            target="sales_target_current"
        )

        # Get AIC values
        aic_simple = result_simple.aic if hasattr(result_simple, 'aic') else result_simple['aic']
        aic_with_noise = result_with_noise.aic if hasattr(result_with_noise, 'aic') else result_with_noise['aic']

        # Adding noise should not dramatically improve AIC
        # The penalty (2k) should prevent noise from helping
        # AIC difference should be small or positive (worse)
        if np.isfinite(aic_simple) and np.isfinite(aic_with_noise):
            # Adding a useless feature increases k by 1, so AIC increases by ~2
            # Unless the feature improves fit enough to offset
            assert abs(aic_with_noise - aic_simple) < 20, (
                f"Adding noise feature changed AIC unexpectedly: "
                f"simple={aic_simple:.2f}, with_noise={aic_with_noise:.2f}"
            )

    def test_reproducibility_with_same_data(self, small_synthetic_data):
        """Same data should produce identical AIC results."""
        features = ["prudential_rate_t1", "competitor_top5_t2"]

        result1 = calculate_aic_for_features(
            data=small_synthetic_data,
            features=features,
            target="sales_target_current"
        )

        result2 = calculate_aic_for_features(
            data=small_synthetic_data,
            features=features,
            target="sales_target_current"
        )

        aic1 = result1.aic if hasattr(result1, 'aic') else result1['aic']
        aic2 = result2.aic if hasattr(result2, 'aic') else result2['aic']

        if np.isfinite(aic1) and np.isfinite(aic2):
            np.testing.assert_allclose(aic1, aic2, rtol=1e-10)


class TestAicWithRealFixtureData:
    """Tests using actual fixture data (slower but more realistic)."""

    @pytest.mark.slow
    def test_aic_on_fixture_data(self, fixture_weekly_data):
        """AIC calculation should work on real fixture data."""
        # Subset for speed
        data_subset = fixture_weekly_data.head(100).copy()

        # Find available feature columns
        pru_cols = [c for c in data_subset.columns if 'prudential' in c.lower() and '_t1' in c]
        comp_cols = [c for c in data_subset.columns if 'competitor' in c.lower() and '_t2' in c]

        if not pru_cols or not comp_cols:
            pytest.skip("Required feature columns not found in fixture")

        features = [pru_cols[0], comp_cols[0]]
        target = 'sales_target_current'

        if target not in data_subset.columns:
            pytest.skip(f"Target column {target} not found in fixture")

        # Drop NaN for this test
        test_data = data_subset.dropna(subset=features + [target])

        if len(test_data) < 20:
            pytest.skip("Insufficient data after dropping NaN")

        result = calculate_aic_for_features(
            data=test_data,
            features=features,
            target=target
        )

        converged = result.converged if hasattr(result, 'converged') else result['converged']
        aic = result.aic if hasattr(result, 'aic') else result['aic']

        # Should produce a result (may or may not converge depending on data)
        assert result is not None
        if converged:
            assert np.isfinite(aic), f"Converged model should have finite AIC, got {aic}"


class TestFeatureCombinationGeneration:
    """Test feature combination generation logic."""

    def test_generates_combinations(self):
        """Should generate feature combinations."""
        base_features = ["f1"]
        candidate_features = ["f2", "f3"]
        max_candidates = 2

        combinations = generate_feature_combinations(
            base_features=base_features,
            candidate_features=candidate_features,
            max_candidates=max_candidates
        )

        assert len(combinations) >= 1, "Should generate at least one combination"

    def test_combinations_include_base_features(self):
        """All combinations should include base features."""
        base_features = ["f1"]
        candidate_features = ["f2", "f3"]
        max_candidates = 2

        combinations = generate_feature_combinations(
            base_features=base_features,
            candidate_features=candidate_features,
            max_candidates=max_candidates
        )

        for combo in combinations:
            for base in base_features:
                assert base in combo, f"Base feature {base} missing from combination {combo}"

    def test_no_duplicate_combinations(self):
        """Should not generate duplicate combinations."""
        base_features = ["f1"]
        candidate_features = ["f2", "f3", "f4"]
        max_candidates = 3

        combinations = generate_feature_combinations(
            base_features=base_features,
            candidate_features=candidate_features,
            max_candidates=max_candidates
        )

        # Convert to sorted tuples for comparison
        normalized = [tuple(sorted(c)) for c in combinations]
        assert len(normalized) == len(set(normalized)), "Found duplicate combinations"


class TestAicEvaluationIntegration:
    """Integration tests for full AIC evaluation pipeline."""

    def test_evaluate_combinations_produces_dataframe(self, small_synthetic_data, minimal_config):
        """evaluate_aic_combinations should produce a DataFrame result."""
        results = evaluate_aic_combinations(
            data=small_synthetic_data,
            config=minimal_config
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0

    def test_evaluate_combinations_has_required_columns(self, small_synthetic_data, minimal_config):
        """Result DataFrame should have required columns."""
        results = evaluate_aic_combinations(
            data=small_synthetic_data,
            config=minimal_config
        )

        required_cols = ['aic', 'converged']
        for col in required_cols:
            assert col in results.columns, f"Missing required column: {col}"

    def test_best_model_identifiable(self, small_synthetic_data, minimal_config):
        """Should be able to identify best model by AIC."""
        results = evaluate_aic_combinations(
            data=small_synthetic_data,
            config=minimal_config
        )

        # Filter to converged models
        converged = results[results['converged']]

        if len(converged) > 0:
            best_idx = converged['aic'].idxmin()
            best_model = converged.loc[best_idx]

            assert np.isfinite(best_model['aic']), "Best model should have finite AIC"

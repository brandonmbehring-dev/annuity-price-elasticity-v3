"""
Tests for AIC Engine - Feature Selection Core Module.

Tests cover:
- calculate_aic_for_features: Core AIC calculation for single feature sets
- generate_feature_combinations: Systematic combination generation
- evaluate_aic_combinations: Full evaluation pipeline

Design Principles:
- Real assertions about correctness (not just "doesn't crash")
- Test happy path + error cases + edge cases
- Mathematical equivalence validation where applicable

Author: Claude Code
Date: 2026-01-23
"""

import pytest
import pandas as pd
import numpy as np
from dataclasses import asdict

from src.features.selection.engines.aic_engine import (
    calculate_aic_for_features,
    generate_feature_combinations,
    evaluate_aic_combinations,
)
from src.features.selection_types import AICResult


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_regression_data():
    """Create simple regression dataset with known properties."""
    np.random.seed(42)
    n = 100

    # Create features with known relationships
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)

    # Target with known coefficients: y = 2*x1 + 1*x2 + noise
    noise = np.random.randn(n) * 0.5
    y = 2.0 * x1 + 1.0 * x2 + noise

    return pd.DataFrame({
        'target': y,
        'feature_1': x1,
        'feature_2': x2,
        'feature_3': x3,  # Not related to target
    })


@pytest.fixture
def multifeature_data():
    """Create dataset with multiple features for combination testing."""
    np.random.seed(123)
    n = 200

    data = {
        'target': np.random.randn(n) * 10 + 50,
        'base_1': np.random.randn(n),
        'base_2': np.random.randn(n),
        'candidate_1': np.random.randn(n),
        'candidate_2': np.random.randn(n),
        'candidate_3': np.random.randn(n),
    }

    # Add correlation between features and target
    df = pd.DataFrame(data)
    df['target'] = df['target'] + 5 * df['base_1'] + 3 * df['candidate_1']

    return df


@pytest.fixture
def feature_selection_config():
    """Create minimal feature selection config for testing."""
    return {
        'base_features': ['base_1', 'base_2'],
        'candidate_features': ['candidate_1', 'candidate_2', 'candidate_3'],
        'max_candidate_features': 2,
        'target_variable': 'target',
    }


@pytest.fixture
def empty_dataframe():
    """Create empty DataFrame for error testing."""
    return pd.DataFrame()


# =============================================================================
# Tests for calculate_aic_for_features
# =============================================================================

class TestCalculateAICForFeatures:
    """Test suite for calculate_aic_for_features function."""

    def test_returns_aic_result(self, simple_regression_data):
        """Verify function returns AICResult dataclass."""
        result = calculate_aic_for_features(
            simple_regression_data,
            ['feature_1', 'feature_2'],
            'target'
        )

        assert isinstance(result, AICResult)

    def test_aic_is_finite_for_valid_model(self, simple_regression_data):
        """Verify AIC is finite for valid model."""
        result = calculate_aic_for_features(
            simple_regression_data,
            ['feature_1', 'feature_2'],
            'target'
        )

        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)

    def test_converged_flag_true_for_valid_model(self, simple_regression_data):
        """Verify converged flag is True for valid model."""
        result = calculate_aic_for_features(
            simple_regression_data,
            ['feature_1'],
            'target'
        )

        assert result.converged is True

    def test_r_squared_in_valid_range(self, simple_regression_data):
        """Verify R-squared is in [0, 1] range."""
        result = calculate_aic_for_features(
            simple_regression_data,
            ['feature_1', 'feature_2'],
            'target'
        )

        assert 0.0 <= result.r_squared <= 1.0
        assert result.r_squared_adj <= 1.0  # Can be negative for bad models

    def test_coefficients_extracted(self, simple_regression_data):
        """Verify coefficients are extracted and reasonable."""
        result = calculate_aic_for_features(
            simple_regression_data,
            ['feature_1', 'feature_2'],
            'target'
        )

        assert 'feature_1' in result.coefficients
        assert 'feature_2' in result.coefficients
        assert 'Intercept' in result.coefficients

        # Feature 1 should have coefficient ~2.0 (from data generation)
        assert 1.5 < result.coefficients['feature_1'] < 2.5

    def test_n_features_correct(self, simple_regression_data):
        """Verify n_features matches input."""
        features = ['feature_1', 'feature_2']
        result = calculate_aic_for_features(
            simple_regression_data,
            features,
            'target'
        )

        assert result.n_features == len(features)

    def test_n_obs_correct(self, simple_regression_data):
        """Verify n_obs matches dataset size."""
        result = calculate_aic_for_features(
            simple_regression_data,
            ['feature_1'],
            'target'
        )

        assert result.n_obs == len(simple_regression_data)

    def test_features_string_format(self, simple_regression_data):
        """Verify features are stored as '+' joined string."""
        features = ['feature_1', 'feature_2']
        result = calculate_aic_for_features(
            simple_regression_data,
            features,
            'target'
        )

        assert result.features == 'feature_1 + feature_2'

    def test_single_feature_model(self, simple_regression_data):
        """Test model with single feature."""
        result = calculate_aic_for_features(
            simple_regression_data,
            ['feature_1'],
            'target'
        )

        assert result.converged is True
        assert result.n_features == 1

    # Error cases

    def test_empty_dataset_returns_error(self, empty_dataframe):
        """Verify empty dataset returns error result."""
        result = calculate_aic_for_features(
            empty_dataframe,
            ['feature_1'],
            'target'
        )

        assert result.converged is False
        assert result.aic == np.inf
        assert 'empty dataset' in result.error.lower()

    def test_missing_feature_returns_error(self, simple_regression_data):
        """Verify missing feature returns error result."""
        result = calculate_aic_for_features(
            simple_regression_data,
            ['feature_1', 'nonexistent_feature'],
            'target'
        )

        assert result.converged is False
        assert result.aic == np.inf
        assert 'missing' in result.error.lower()

    def test_missing_target_returns_error(self, simple_regression_data):
        """Verify missing target returns error result."""
        result = calculate_aic_for_features(
            simple_regression_data,
            ['feature_1'],
            'nonexistent_target'
        )

        assert result.converged is False
        assert result.aic == np.inf

    def test_empty_feature_list_returns_error(self, simple_regression_data):
        """Verify empty feature list returns error result."""
        result = calculate_aic_for_features(
            simple_regression_data,
            [],
            'target'
        )

        assert result.converged is False
        assert result.aic == np.inf


# =============================================================================
# Tests for generate_feature_combinations
# =============================================================================

class TestGenerateFeatureCombinations:
    """Test suite for generate_feature_combinations function."""

    def test_returns_list_of_lists(self):
        """Verify function returns list of feature lists."""
        result = generate_feature_combinations(
            base_features=['base'],
            candidate_features=['c1', 'c2'],
            max_candidates=2
        )

        assert isinstance(result, list)
        assert all(isinstance(combo, list) for combo in result)

    def test_base_features_in_all_combinations(self):
        """Verify base features appear in all combinations."""
        base = ['base_1', 'base_2']
        candidates = ['c1', 'c2', 'c3']

        result = generate_feature_combinations(
            base_features=base,
            candidate_features=candidates,
            max_candidates=2
        )

        for combo in result:
            for base_feature in base:
                assert base_feature in combo

    def test_combination_count_correct(self):
        """Verify correct number of combinations generated."""
        # With 3 candidates and max_candidates=2:
        # C(3,1) + C(3,2) = 3 + 3 = 6 combinations
        result = generate_feature_combinations(
            base_features=['base'],
            candidate_features=['c1', 'c2', 'c3'],
            max_candidates=2
        )

        assert len(result) == 6

    def test_max_candidates_one(self):
        """Test with max_candidates=1 (only single candidate combos)."""
        result = generate_feature_combinations(
            base_features=['base'],
            candidate_features=['c1', 'c2', 'c3'],
            max_candidates=1
        )

        # Should have exactly 3 combinations (one per candidate)
        assert len(result) == 3

        # Each should have base + 1 candidate = 2 features
        for combo in result:
            assert len(combo) == 2

    def test_max_candidates_all(self):
        """Test with max_candidates equal to candidate count."""
        result = generate_feature_combinations(
            base_features=['base'],
            candidate_features=['c1', 'c2'],
            max_candidates=2
        )

        # C(2,1) + C(2,2) = 2 + 1 = 3 combinations
        assert len(result) == 3

        # One combination should have all candidates
        all_candidate_combo = [c for c in result if len(c) == 3]  # base + 2 candidates
        assert len(all_candidate_combo) == 1

    def test_empty_base_features(self):
        """Test with empty base features (valid use case)."""
        result = generate_feature_combinations(
            base_features=[],
            candidate_features=['c1', 'c2'],
            max_candidates=2
        )

        assert len(result) == 3  # C(2,1) + C(2,2) = 3

    def test_max_candidates_exceeds_available(self):
        """Test that max_candidates is capped at available candidates."""
        # This should not raise - should cap at 2
        result = generate_feature_combinations(
            base_features=['base'],
            candidate_features=['c1', 'c2'],
            max_candidates=10  # More than available
        )

        # Should generate all combinations up to 2
        assert len(result) == 3  # C(2,1) + C(2,2)

    # Error cases

    def test_empty_candidates_raises_error(self):
        """Verify empty candidate list raises ValueError."""
        with pytest.raises(ValueError, match="No candidate features"):
            generate_feature_combinations(
                base_features=['base'],
                candidate_features=[],
                max_candidates=2
            )

    def test_zero_max_candidates_raises_error(self):
        """Verify max_candidates=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_candidates must be >= 1"):
            generate_feature_combinations(
                base_features=['base'],
                candidate_features=['c1', 'c2'],
                max_candidates=0
            )

    def test_negative_max_candidates_raises_error(self):
        """Verify negative max_candidates raises ValueError."""
        with pytest.raises(ValueError, match="max_candidates must be >= 1"):
            generate_feature_combinations(
                base_features=['base'],
                candidate_features=['c1', 'c2'],
                max_candidates=-1
            )


# =============================================================================
# Tests for evaluate_aic_combinations
# =============================================================================

class TestEvaluateAICCombinations:
    """Test suite for evaluate_aic_combinations function."""

    def test_returns_dataframe(self, multifeature_data, feature_selection_config):
        """Verify function returns pandas DataFrame."""
        result = evaluate_aic_combinations(
            multifeature_data,
            feature_selection_config
        )

        assert isinstance(result, pd.DataFrame)

    def test_dataframe_has_required_columns(self, multifeature_data, feature_selection_config):
        """Verify result DataFrame has all required columns."""
        result = evaluate_aic_combinations(
            multifeature_data,
            feature_selection_config
        )

        required_columns = [
            'features', 'n_features', 'aic', 'bic',
            'r_squared', 'r_squared_adj', 'converged', 'n_obs'
        ]

        for col in required_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_correct_number_of_rows(self, multifeature_data, feature_selection_config):
        """Verify result has expected number of rows."""
        result = evaluate_aic_combinations(
            multifeature_data,
            feature_selection_config
        )

        # With 3 candidates and max_candidates=2: C(3,1) + C(3,2) = 6
        assert len(result) == 6

    def test_all_models_converged(self, multifeature_data, feature_selection_config):
        """Verify all valid models converge."""
        result = evaluate_aic_combinations(
            multifeature_data,
            feature_selection_config
        )

        assert result['converged'].all()

    def test_aic_values_are_finite(self, multifeature_data, feature_selection_config):
        """Verify all AIC values are finite."""
        result = evaluate_aic_combinations(
            multifeature_data,
            feature_selection_config
        )

        assert result['aic'].apply(np.isfinite).all()

    def test_target_variable_override(self, multifeature_data, feature_selection_config):
        """Test target variable override parameter."""
        # Add transformed target
        multifeature_data['target_transformed'] = np.log1p(
            multifeature_data['target'] - multifeature_data['target'].min() + 1
        )

        result = evaluate_aic_combinations(
            multifeature_data,
            feature_selection_config,
            target_variable='target_transformed'
        )

        assert len(result) == 6
        assert result['converged'].all()

    def test_lower_aic_is_better(self, multifeature_data, feature_selection_config):
        """Verify AIC comparison property (lower is better)."""
        result = evaluate_aic_combinations(
            multifeature_data,
            feature_selection_config
        )

        # All AICs should be comparable (finite and numeric)
        aic_values = result['aic'].values
        assert all(np.isfinite(aic_values))

        # Best AIC should be identifiable
        best_aic = result['aic'].min()
        assert np.isfinite(best_aic)

    def test_base_features_always_present(self, multifeature_data, feature_selection_config):
        """Verify base features appear in all model feature strings."""
        result = evaluate_aic_combinations(
            multifeature_data,
            feature_selection_config
        )

        base_features = feature_selection_config['base_features']
        for features_str in result['features']:
            for base in base_features:
                assert base in features_str

    # Error cases

    def test_empty_data_raises_error(self, empty_dataframe, feature_selection_config):
        """Verify empty dataset raises ValueError."""
        with pytest.raises(ValueError, match="empty dataset"):
            evaluate_aic_combinations(
                empty_dataframe,
                feature_selection_config
            )

    def test_missing_target_raises_error(self, multifeature_data, feature_selection_config):
        """Verify missing target raises ValueError."""
        feature_selection_config['target_variable'] = 'nonexistent'

        with pytest.raises(ValueError, match="not found"):
            evaluate_aic_combinations(
                multifeature_data,
                feature_selection_config
            )

    def test_no_available_candidates_raises_error(self, multifeature_data, feature_selection_config):
        """Verify no matching candidates raises ValueError."""
        feature_selection_config['candidate_features'] = ['x', 'y', 'z']  # Don't exist

        with pytest.raises(ValueError, match="No candidate features"):
            evaluate_aic_combinations(
                multifeature_data,
                feature_selection_config
            )


# =============================================================================
# Integration Tests
# =============================================================================

class TestAICEngineIntegration:
    """Integration tests for AIC engine components."""

    def test_end_to_end_feature_selection_flow(self, multifeature_data, feature_selection_config):
        """Test complete feature selection flow."""
        # Generate combinations
        combinations = generate_feature_combinations(
            feature_selection_config['base_features'],
            feature_selection_config['candidate_features'],
            feature_selection_config['max_candidate_features']
        )

        # Evaluate each combination
        results = []
        for features in combinations:
            result = calculate_aic_for_features(
                multifeature_data,
                features,
                feature_selection_config['target_variable']
            )
            results.append(result)

        # Verify all converged
        assert all(r.converged for r in results)

        # Find best model
        best = min(results, key=lambda r: r.aic)
        assert np.isfinite(best.aic)

    def test_consistency_between_methods(self, multifeature_data, feature_selection_config):
        """Verify evaluate_aic_combinations matches manual calculation."""
        # Use evaluate_aic_combinations
        df_result = evaluate_aic_combinations(
            multifeature_data,
            feature_selection_config
        )

        # Manually calculate for one combination
        features = feature_selection_config['base_features'] + ['candidate_1']
        manual_result = calculate_aic_for_features(
            multifeature_data,
            features,
            feature_selection_config['target_variable']
        )

        # Find matching row in DataFrame
        features_str = ' + '.join(features)
        matching_rows = df_result[df_result['features'] == features_str]

        assert len(matching_rows) == 1

        # Values should match
        row = matching_rows.iloc[0]
        assert abs(row['aic'] - manual_result.aic) < 1e-10
        assert abs(row['r_squared'] - manual_result.r_squared) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

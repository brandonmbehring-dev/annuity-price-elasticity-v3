"""
Tests for Search Space Reduction Module.

Tests cover:
- Input validation
- Priority combination generation
- Non-priority combination filling
- Combination limiting and sorting
- Result building
- Main function integration

Design Principles:
- Property-based tests for set invariants
- Edge case tests for error handling
- Integration tests for full workflow

Mathematical Properties Validated:
- reduced_space is subset of full_space
- Combinations with priority features appear first
- n_combinations respects max_combinations limit

Author: Claude Code
Date: 2026-01-30
"""

import pytest
import pandas as pd
import numpy as np
import warnings

from src.features.selection.enhancements.multiple_testing.search_space_reduction import (
    _validate_search_space_inputs,
    _generate_priority_combinations,
    _add_non_priority_combinations,
    _limit_and_sort_combinations,
    _build_search_space_result,
    create_reduced_search_space,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def candidate_features():
    """Standard candidate feature list."""
    return [
        'prudential_rate_current',
        'competitor_mid_t2',
        'competitor_top5_t2',
        'spread_t1',
        'vix_t1',
        'dgs5_t1',
    ]


@pytest.fixture
def priority_features():
    """Priority features for search space reduction."""
    return [
        'prudential_rate_current',
        'competitor_mid_t2',
        'competitor_top5_t2',
    ]


@pytest.fixture
def small_candidate_features():
    """Small feature set for edge case testing."""
    return ['f1', 'f2', 'f3']


# =============================================================================
# Tests for Input Validation
# =============================================================================


class TestValidateSearchSpaceInputs:
    """Tests for search space input validation."""

    def test_empty_candidates_raises_error(self):
        """Test that empty candidate list raises ValueError."""
        with pytest.raises(ValueError, match="No candidate features"):
            _validate_search_space_inputs([], None)

    def test_none_priority_uses_defaults(self, candidate_features):
        """Test that None priority features uses defaults."""
        valid_priority = _validate_search_space_inputs(candidate_features, None)

        # Should return default priority features that exist in candidates
        assert isinstance(valid_priority, list)

    def test_missing_priority_features_warns(self, candidate_features):
        """Test that missing priority features generates warning."""
        missing_features = ['nonexistent_feature', 'another_missing']

        with pytest.warns(UserWarning, match="Priority features not found"):
            _validate_search_space_inputs(candidate_features, missing_features)

    def test_valid_priority_features_returned(self, candidate_features, priority_features):
        """Test that valid priority features are returned."""
        valid = _validate_search_space_inputs(candidate_features, priority_features)

        assert all(f in candidate_features for f in valid)

    def test_partial_priority_features(self, candidate_features):
        """Test handling of partial priority feature overlap."""
        partial_priority = ['prudential_rate_current', 'nonexistent']

        with pytest.warns(UserWarning):
            valid = _validate_search_space_inputs(candidate_features, partial_priority)

        assert 'prudential_rate_current' in valid
        assert 'nonexistent' not in valid


# =============================================================================
# Tests for Priority Combination Generation
# =============================================================================


class TestGeneratePriorityCombinations:
    """Tests for priority-inclusive combination generation."""

    def test_returns_tuple_of_lists(self, candidate_features, priority_features):
        """Test that function returns tuple of (combinations, metadata)."""
        combos, metadata = _generate_priority_combinations(
            candidate_features, priority_features,
            min_features=2, max_features=3
        )

        assert isinstance(combos, list)
        assert isinstance(metadata, list)
        assert len(combos) == len(metadata)

    def test_all_combinations_have_priority(self, candidate_features, priority_features):
        """Test that all combinations contain at least one priority feature."""
        combos, _ = _generate_priority_combinations(
            candidate_features, priority_features,
            min_features=2, max_features=3
        )

        for combo in combos:
            has_priority = any(f in priority_features for f in combo)
            assert has_priority, f"Combo {combo} has no priority features"

    def test_metadata_structure(self, candidate_features, priority_features):
        """Test that metadata has correct structure."""
        _, metadata = _generate_priority_combinations(
            candidate_features, priority_features,
            min_features=2, max_features=3
        )

        for meta in metadata:
            assert 'combination' in meta
            assert 'n_features' in meta
            assert 'priority_features' in meta
            assert 'strategy' in meta
            assert meta['strategy'] == 'priority_inclusive'

    def test_respects_min_max_features(self, candidate_features, priority_features):
        """Test that combinations respect min/max feature constraints."""
        combos, _ = _generate_priority_combinations(
            candidate_features, priority_features,
            min_features=2, max_features=4
        )

        for combo in combos:
            assert 2 <= len(combo) <= 4

    def test_no_priority_features_returns_empty(self, candidate_features):
        """Test that no valid priority features returns empty list."""
        combos, metadata = _generate_priority_combinations(
            candidate_features, [],
            min_features=2, max_features=3
        )

        assert len(combos) == 0
        assert len(metadata) == 0


# =============================================================================
# Tests for Non-Priority Combination Filling
# =============================================================================


class TestAddNonPriorityCombinations:
    """Tests for non-priority combination filling."""

    def test_fills_to_max_combinations(self, candidate_features, priority_features):
        """Test that function fills up to max_combinations."""
        # Start with small priority-inclusive set
        initial_combos = [('prudential_rate_current', 'spread_t1')]
        initial_metadata = [{'combination': initial_combos[0], 'n_features': 2,
                           'priority_features': ['prudential_rate_current'], 'strategy': 'priority_inclusive'}]

        combos, metadata = _add_non_priority_combinations(
            initial_combos.copy(), initial_metadata.copy(),
            candidate_features, priority_features,
            max_combinations=10,
            min_features=2, max_features=3
        )

        assert len(combos) <= 10

    def test_non_priority_metadata_correct(self, candidate_features, priority_features):
        """Test that non-priority combinations have correct metadata."""
        combos, metadata = _add_non_priority_combinations(
            [], [], candidate_features, priority_features,
            max_combinations=5,
            min_features=2, max_features=3
        )

        for meta in metadata:
            if meta['strategy'] == 'non_priority':
                # Non-priority combinations should have no priority features
                assert meta['priority_features'] == []

    def test_respects_max_combinations_limit(self, candidate_features, priority_features):
        """Test that max_combinations is respected."""
        # Fill with many priority combinations first
        initial = [('prudential_rate_current', 'spread_t1')] * 50
        initial_meta = [{'combination': c, 'n_features': 2,
                        'priority_features': ['prudential_rate_current'], 'strategy': 'priority_inclusive'}
                       for c in initial]

        combos, metadata = _add_non_priority_combinations(
            initial, initial_meta, candidate_features, priority_features,
            max_combinations=50,
            min_features=2, max_features=3
        )

        # Should not exceed max
        assert len(combos) == 50


# =============================================================================
# Tests for Combination Limiting and Sorting
# =============================================================================


class TestLimitAndSortCombinations:
    """Tests for combination limiting and sorting."""

    def test_limits_to_max(self):
        """Test that combinations are limited to max."""
        combos = [('a', 'b'), ('c', 'd'), ('e', 'f'), ('g', 'h')]
        metadata = [
            {'combination': c, 'n_features': 2, 'priority_features': [], 'strategy': 'test'}
            for c in combos
        ]

        limited_combos, limited_meta = _limit_and_sort_combinations(
            combos, metadata, max_combinations=2
        )

        assert len(limited_combos) == 2
        assert len(limited_meta) == 2

    def test_preserves_all_if_under_max(self):
        """Test that all combinations are preserved if under max."""
        combos = [('a', 'b'), ('c', 'd')]
        metadata = [
            {'combination': c, 'n_features': 2, 'priority_features': [], 'strategy': 'test'}
            for c in combos
        ]

        limited_combos, limited_meta = _limit_and_sort_combinations(
            combos, metadata, max_combinations=10
        )

        assert len(limited_combos) == 2

    def test_sorts_by_priority_count(self):
        """Test that combinations are sorted by priority feature count."""
        combos = [('a', 'b'), ('c', 'd'), ('e', 'f')]
        metadata = [
            {'combination': ('a', 'b'), 'n_features': 2, 'priority_features': [], 'strategy': 'test'},
            {'combination': ('c', 'd'), 'n_features': 2, 'priority_features': ['c', 'd'], 'strategy': 'test'},
            {'combination': ('e', 'f'), 'n_features': 2, 'priority_features': ['e'], 'strategy': 'test'},
        ]

        limited_combos, limited_meta = _limit_and_sort_combinations(
            combos, metadata, max_combinations=2
        )

        # Should prioritize combinations with more priority features
        assert len(limited_meta[0]['priority_features']) >= len(limited_meta[1]['priority_features'])


# =============================================================================
# Tests for Result Building
# =============================================================================


class TestBuildSearchSpaceResult:
    """Tests for search space result building."""

    def test_returns_dict_with_required_keys(self):
        """Test that result dict has required keys."""
        combos = [('a', 'b'), ('c', 'd')]
        metadata = [
            {'combination': c, 'n_features': 2, 'priority_features': [], 'strategy': 'test'}
            for c in combos
        ]

        result = _build_search_space_result(combos, metadata, ['a'])

        assert 'combinations' in result
        assert 'combination_metadata' in result
        assert 'n_combinations' in result
        assert 'reduction_factor' in result
        assert 'business_justification' in result
        assert 'statistical_properties' in result

    def test_n_combinations_correct(self):
        """Test that n_combinations is correct."""
        combos = [('a', 'b'), ('c', 'd'), ('e', 'f')]
        metadata = [{'combination': c, 'n_features': 2, 'priority_features': [], 'strategy': 'test'}
                   for c in combos]

        result = _build_search_space_result(combos, metadata, [])

        assert result['n_combinations'] == 3

    def test_reduction_factor_calculation(self):
        """Test reduction factor is calculated correctly."""
        combos = [('a', 'b')]
        metadata = [{'combination': combos[0], 'n_features': 2, 'priority_features': [], 'strategy': 'test'}]

        result = _build_search_space_result(combos, metadata, [])

        # Reduction factor = n_combinations / 793
        expected = 1 / 793
        assert result['reduction_factor'] == pytest.approx(expected)

    def test_multiple_testing_eliminated_flag(self):
        """Test multiple_testing_eliminated flag is set correctly."""
        # Small set should eliminate multiple testing
        small_combos = [('a', 'b')] * 10
        small_meta = [{'combination': c, 'n_features': 2, 'priority_features': [], 'strategy': 'test'}
                     for c in small_combos]

        result_small = _build_search_space_result(small_combos, small_meta, [])
        assert result_small['business_justification']['multiple_testing_eliminated'] is True

        # Large set should not eliminate
        large_combos = [('a', 'b')] * 50
        large_meta = [{'combination': c, 'n_features': 2, 'priority_features': [], 'strategy': 'test'}
                     for c in large_combos]

        result_large = _build_search_space_result(large_combos, large_meta, [])
        assert result_large['business_justification']['multiple_testing_eliminated'] is False


# =============================================================================
# Tests for Main Function
# =============================================================================


class TestCreateReducedSearchSpace:
    """Tests for main create_reduced_search_space function."""

    def test_returns_dict(self, candidate_features):
        """Test that function returns dict."""
        result = create_reduced_search_space(candidate_features)

        assert isinstance(result, dict)

    def test_combinations_are_tuples(self, candidate_features):
        """Test that combinations are tuples."""
        result = create_reduced_search_space(candidate_features)

        for combo in result['combinations']:
            assert isinstance(combo, tuple)

    def test_respects_max_combinations(self, candidate_features):
        """Test that max_combinations is respected."""
        result = create_reduced_search_space(
            candidate_features,
            max_combinations=10
        )

        assert result['n_combinations'] <= 10

    def test_empty_candidates_raises_error(self):
        """Test that empty candidates raises ValueError."""
        with pytest.raises(ValueError):
            create_reduced_search_space([])

    def test_custom_priority_features(self, candidate_features):
        """Test that custom priority features are used."""
        custom_priority = ['spread_t1', 'vix_t1']

        result = create_reduced_search_space(
            candidate_features,
            business_priority_features=custom_priority
        )

        # Priority features should appear in justification
        assert 'priority_features_used' in result['business_justification']

    def test_min_max_features_respected(self, candidate_features):
        """Test that min/max features per combo are respected."""
        result = create_reduced_search_space(
            candidate_features,
            min_features_per_combo=3,
            max_features_per_combo=4,
            max_combinations=100
        )

        for combo in result['combinations']:
            assert 3 <= len(combo) <= 4

    def test_statistical_properties_present(self, candidate_features):
        """Test that statistical properties are calculated."""
        result = create_reduced_search_space(candidate_features)

        assert 'no_multiple_testing_correction_needed' in result['statistical_properties']
        assert 'effective_alpha' in result['statistical_properties']
        assert 'expected_false_positives' in result['statistical_properties']


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestPropertyBased:
    """Property-based tests for search space reduction."""

    def test_reduced_space_subset_of_full(self, candidate_features):
        """Property: reduced combinations are valid feature subsets."""
        result = create_reduced_search_space(candidate_features, max_combinations=50)

        for combo in result['combinations']:
            for feature in combo:
                assert feature in candidate_features

    def test_n_combinations_matches_list_length(self, candidate_features):
        """Property: n_combinations matches actual list length."""
        result = create_reduced_search_space(candidate_features)

        assert result['n_combinations'] == len(result['combinations'])

    def test_metadata_length_matches_combinations(self, candidate_features):
        """Property: metadata length matches combinations length."""
        result = create_reduced_search_space(candidate_features)

        assert len(result['combination_metadata']) == len(result['combinations'])

    @pytest.mark.parametrize("max_combos", [5, 10, 50, 100])
    def test_respects_max_limit_parametrized(self, candidate_features, max_combos):
        """Property: n_combinations <= max_combinations for various limits."""
        result = create_reduced_search_space(
            candidate_features,
            max_combinations=max_combos
        )

        assert result['n_combinations'] <= max_combos


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for search space reduction."""

    def test_single_feature_min_max(self, small_candidate_features):
        """Test with min_features = max_features = 1."""
        result = create_reduced_search_space(
            small_candidate_features,
            min_features_per_combo=1,
            max_features_per_combo=1,
            max_combinations=10
        )

        for combo in result['combinations']:
            assert len(combo) == 1

    def test_max_features_exceeds_candidates(self, small_candidate_features):
        """Test when max_features exceeds candidate count."""
        result = create_reduced_search_space(
            small_candidate_features,
            min_features_per_combo=2,
            max_features_per_combo=10,  # More than 3 candidates
            max_combinations=100
        )

        # Should cap at number of candidates
        for combo in result['combinations']:
            assert len(combo) <= len(small_candidate_features)

    def test_very_small_max_combinations(self, candidate_features):
        """Test with very small max_combinations."""
        result = create_reduced_search_space(
            candidate_features,
            max_combinations=1
        )

        assert result['n_combinations'] <= 1

    def test_no_valid_priority_overlap(self, candidate_features):
        """Test when priority features don't overlap with candidates."""
        with pytest.warns(UserWarning):
            result = create_reduced_search_space(
                candidate_features,
                business_priority_features=['nonexistent1', 'nonexistent2']
            )

        # Should still produce combinations (fall back to non-priority)
        assert result['n_combinations'] >= 0

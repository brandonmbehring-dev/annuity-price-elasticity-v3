"""
Search Space Reduction Engine for Multiple Testing Correction.

This module provides domain-guided search space reduction to eliminate
the multiple testing problem by pre-selecting promising combinations.

Key Functions:
- create_reduced_search_space: Domain-guided combination reduction

Instead of testing all possible combinations (793 for 12 choose 2-4),
use business knowledge to pre-select most promising combinations,
eliminating the multiple testing problem entirely.

Part of Phase 6.4 module split.

Module Architecture:
- multiple_testing_types.py: Shared dataclass
- bonferroni_engine.py: Bonferroni correction
- fdr_engine.py: FDR correction
- search_space_reduction.py: Search space reduction (this file)
- multiple_testing_correction.py: Orchestrator + method comparison
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# INPUT VALIDATION
# =============================================================================


def _validate_search_space_inputs(
    candidate_features: List[str],
    business_priority_features: Optional[List[str]]
) -> List[str]:
    """
    Validate inputs and resolve priority features for search space reduction.

    Parameters
    ----------
    candidate_features : List[str]
        Full list of candidate features
    business_priority_features : List[str], optional
        Features that should appear in most combinations

    Returns
    -------
    List[str]
        Validated priority features that exist in candidates

    Raises
    ------
    ValueError
        If no candidate features provided
    """
    if not candidate_features:
        raise ValueError("No candidate features provided for search space reduction")

    if business_priority_features is None:
        business_priority_features = [
            'prudential_rate_current',
            'competitor_mid_t2',
            'competitor_top5_t2'
        ]

    valid_priority = [f for f in business_priority_features if f in candidate_features]
    if len(valid_priority) < len(business_priority_features):
        missing = set(business_priority_features) - set(candidate_features)
        warnings.warn(f"Priority features not found in candidates: {missing}")

    return valid_priority


# =============================================================================
# COMBINATION GENERATION
# =============================================================================


def _generate_priority_combinations(
    candidate_features: List[str],
    valid_priority: List[str],
    min_features: int,
    max_features: int
) -> Tuple[List[tuple], List[Dict[str, Any]]]:
    """
    Generate combinations containing at least one priority feature.

    Parameters
    ----------
    candidate_features : List[str]
        Full list of candidate features
    valid_priority : List[str]
        Priority features that exist in candidates
    min_features : int
        Minimum features per combination
    max_features : int
        Maximum features per combination

    Returns
    -------
    Tuple[List[tuple], List[Dict[str, Any]]]
        Combinations and their metadata
    """
    import itertools

    combinations: List[tuple] = []
    combination_metadata: List[Dict[str, Any]] = []

    for r in range(min_features, max_features + 1):
        for combo in itertools.combinations(candidate_features, r):
            has_priority = any(f in valid_priority for f in combo)
            if has_priority:
                combinations.append(combo)
                combination_metadata.append({
                    'combination': combo,
                    'n_features': len(combo),
                    'priority_features': [f for f in combo if f in valid_priority],
                    'strategy': 'priority_inclusive'
                })

    return combinations, combination_metadata


def _add_non_priority_combinations(
    combinations: List[tuple],
    combination_metadata: List[Dict[str, Any]],
    candidate_features: List[str],
    valid_priority: List[str],
    max_combinations: int,
    min_features: int,
    max_features: int
) -> Tuple[List[tuple], List[Dict[str, Any]]]:
    """Add non-priority combinations to fill remaining slots."""
    import itertools

    if len(combinations) >= max_combinations:
        return combinations, combination_metadata

    remaining_slots = max_combinations - len(combinations)
    non_priority = [f for f in candidate_features if f not in valid_priority]

    for r in range(min_features, max_features + 1):
        if remaining_slots <= 0:
            break
        for combo in itertools.combinations(non_priority, r):
            if remaining_slots <= 0:
                break
            if combo not in combinations:
                combinations.append(combo)
                combination_metadata.append({
                    'combination': combo,
                    'n_features': len(combo),
                    'priority_features': [],
                    'strategy': 'non_priority'
                })
                remaining_slots -= 1

    return combinations, combination_metadata


# =============================================================================
# COMBINATION LIMITING AND SORTING
# =============================================================================


def _limit_and_sort_combinations(
    combinations: List[tuple],
    combination_metadata: List[Dict[str, Any]],
    max_combinations: int
) -> Tuple[List[tuple], List[Dict[str, Any]]]:
    """
    Limit combinations to max and sort by priority.

    Parameters
    ----------
    combinations : List[tuple]
        All generated combinations
    combination_metadata : List[Dict[str, Any]]
        All combination metadata
    max_combinations : int
        Maximum combinations to keep

    Returns
    -------
    Tuple[List[tuple], List[Dict[str, Any]]]
        Limited and sorted combinations with metadata
    """
    if len(combinations) <= max_combinations:
        return combinations, combination_metadata

    combination_metadata.sort(
        key=lambda x: (len(x['priority_features']), x['n_features']),
        reverse=True
    )
    combinations = [meta['combination'] for meta in combination_metadata[:max_combinations]]
    combination_metadata = combination_metadata[:max_combinations]

    return combinations, combination_metadata


# =============================================================================
# RESULT BUILDING
# =============================================================================


def _build_search_space_result(
    combinations: List[tuple],
    combination_metadata: List[Dict[str, Any]],
    valid_priority: List[str]
) -> Dict[str, Any]:
    """
    Build final search space result dictionary.

    Parameters
    ----------
    combinations : List[tuple]
        Final list of feature combinations
    combination_metadata : List[Dict[str, Any]]
        Metadata for each combination
    valid_priority : List[str]
        Priority features used

    Returns
    -------
    Dict[str, Any]
        Complete search space result with business justification
    """
    n_combinations = len(combinations)

    return {
        'combinations': combinations,
        'combination_metadata': combination_metadata,
        'n_combinations': n_combinations,
        'reduction_factor': n_combinations / 793,
        'business_justification': {
            'priority_features_used': valid_priority,
            'strategy_distribution': pd.Series([
                meta['strategy'] for meta in combination_metadata
            ]).value_counts().to_dict(),
            'multiple_testing_eliminated': n_combinations <= 20
        },
        'statistical_properties': {
            'no_multiple_testing_correction_needed': n_combinations <= 20,
            'effective_alpha': 0.05 if n_combinations <= 20 else 0.05 / n_combinations,
            'expected_false_positives': min(n_combinations * 0.05, 1.0)
        }
    }


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def create_reduced_search_space(
    candidate_features: List[str],
    business_priority_features: Optional[List[str]] = None,
    max_combinations: int = 100,
    min_features_per_combo: int = 2,
    max_features_per_combo: int = 4
) -> Dict[str, Any]:
    """Create reduced search space using domain knowledge to eliminate multiple testing."""
    # Step 1: Validate inputs and resolve priority features
    valid_priority = _validate_search_space_inputs(
        candidate_features, business_priority_features
    )

    # Step 2: Generate priority-inclusive combinations
    combinations, combination_metadata = _generate_priority_combinations(
        candidate_features, valid_priority,
        min_features_per_combo, max_features_per_combo
    )

    # Step 3: Fill remaining slots with non-priority combinations
    combinations, combination_metadata = _add_non_priority_combinations(
        combinations, combination_metadata,
        candidate_features, valid_priority, max_combinations,
        min_features_per_combo, max_features_per_combo
    )

    # Step 4: Limit and sort by priority
    combinations, combination_metadata = _limit_and_sort_combinations(
        combinations, combination_metadata, max_combinations
    )

    # Step 5: Build and return result
    return _build_search_space_result(
        combinations, combination_metadata, valid_priority
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Validation
    '_validate_search_space_inputs',
    # Combination generation
    '_generate_priority_combinations',
    '_add_non_priority_combinations',
    # Limiting and sorting
    '_limit_and_sort_combinations',
    # Result building
    '_build_search_space_result',
    # Main function
    'create_reduced_search_space',
]

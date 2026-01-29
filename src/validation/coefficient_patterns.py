"""
Unified Coefficient Validation Patterns

Provides regex-based coefficient sign validation to replace substring matching
that causes false positives with P_ and C_ patterns.

Decision: Unified validation patterns across RILA/FIA/MYGA products.
Naming refactor (own_rate.t1, competitor_weighted.t2) deferred to backlog.

Usage:
    from src.validation.coefficient_patterns import validate_coefficient_sign

    # Unified naming (2026-01-26): _t0 instead of _current
    is_valid, reason = validate_coefficient_sign("prudential_rate_t0", 0.5)
    assert is_valid  # True for positive coefficient

    # competitor_weighted instead of competitor_mid
    is_valid, reason = validate_coefficient_sign("competitor_weighted_t2", 0.3)
    assert not is_valid  # False - should be negative
"""

import re
from typing import Dict, List, Tuple, Optional


# =============================================================================
# PATTERN REGISTRY
# =============================================================================

# Unified patterns across all products
# Format: pattern_name -> (regex, expected_sign)
# expected_sign: "positive", "negative", or "context_dependent"
#
# Feature Naming Unification (2026-01-26):
# - competitor_mid renamed to competitor_weighted (regex already covers both)
# - _current normalized to _t0 (regex uses _t\d+ which covers t0)
COEFFICIENT_PATTERNS: Dict[str, Tuple[str, str]] = {
    # Own-rate features (should be POSITIVE - higher rates attract customers)
    "own_rate": (
        r"(prudential|own|p)_?rate",
        "positive"
    ),

    # Competitor aggregate features (should be NEGATIVE - substitution effect)
    # Matches: competitor_weighted, competitor_mid (legacy), competitor_mean
    "competitor_weighted": (
        r"competitor_(mid|weighted|mean)",
        "negative"
    ),
    "competitor_core": (
        r"competitor_core",
        "negative"
    ),
    "competitor_top": (
        r"competitor_(top\d+|1st|2nd|3rd)",
        "negative"
    ),

    # Competitor lag features (should be NEGATIVE)
    # Matches _t0, _t1, _t2, etc. (unified naming)
    "competitor_lag": (
        r"competitor_.*_t\d+",
        "negative"
    ),
    "c_weighted_mean": (
        r"c_weighted_mean",
        "negative"
    ),

    # Economic indicators (context dependent)
    "treasury": (
        r"(dgs5|treasury|econ\.treasury)",
        "context_dependent"
    ),
    "vix": (
        r"(vix|volatility)",
        "context_dependent"
    ),

    # Spread features (context dependent - sign depends on definition)
    "spread": (
        r"spread",
        "context_dependent"
    ),
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_coefficient_sign(
    feature: str,
    coefficient: float,
    product_type: str = "RILA",
    strict: bool = False,
) -> Tuple[bool, str]:
    """Validate coefficient sign using unified regex patterns.

    Parameters
    ----------
    feature : str
        Feature name to validate
    coefficient : float
        Coefficient value from model
    product_type : str, default="RILA"
        Product type for context (RILA, FIA, MYGA)
    strict : bool, default=False
        If True, raise ValueError on violation instead of returning False

    Returns
    -------
    Tuple[bool, str]
        (is_valid, reason) where reason explains the validation result

    Raises
    ------
    ValueError
        If strict=True and coefficient sign violates constraint

    Examples
    --------
    >>> validate_coefficient_sign("prudential_rate_t0", 0.5)
    (True, 'Valid: own_rate pattern (positive)')

    >>> validate_coefficient_sign("competitor_weighted_t2", 0.3)
    (False, 'competitor_weighted_t2 should be negative (got 0.3000)')

    >>> validate_coefficient_sign("dgs5", -0.1)
    (True, 'Context-dependent: treasury pattern')
    """
    feature_lower = feature.lower()

    for pattern_name, (regex, expected) in COEFFICIENT_PATTERNS.items():
        if re.search(regex, feature_lower, re.IGNORECASE):
            if expected == "context_dependent":
                return True, f"Context-dependent: {pattern_name} pattern"

            if expected == "positive" and coefficient <= 0:
                reason = f"{feature} should be positive (got {coefficient:.4f})"
                if strict:
                    raise ValueError(reason)
                return False, reason

            if expected == "negative" and coefficient >= 0:
                reason = f"{feature} should be negative (got {coefficient:.4f})"
                if strict:
                    raise ValueError(reason)
                return False, reason

            return True, f"Valid: {pattern_name} pattern ({expected})"

    # No constraint defined for this feature
    return True, "No constraint defined"


def validate_all_coefficients(
    coefficients: Dict[str, float],
    product_type: str = "RILA",
) -> Dict[str, List[Dict[str, any]]]:
    """Validate all coefficients against economic constraints.

    Parameters
    ----------
    coefficients : Dict[str, float]
        Feature name to coefficient mapping
    product_type : str, default="RILA"
        Product type for context

    Returns
    -------
    Dict[str, List[Dict]]
        Results with 'passed', 'violated', and 'warnings' lists
    """
    results = {
        "passed": [],
        "violated": [],
        "warnings": [],
    }

    for feature, coef in coefficients.items():
        is_valid, reason = validate_coefficient_sign(
            feature, coef, product_type
        )

        entry = {
            "feature": feature,
            "coefficient": coef,
            "reason": reason,
        }

        if "Context-dependent" in reason:
            results["warnings"].append(entry)
        elif is_valid:
            results["passed"].append(entry)
        else:
            results["violated"].append(entry)

    return results


def get_expected_sign(feature: str) -> Optional[str]:
    """Get expected coefficient sign for a feature.

    Parameters
    ----------
    feature : str
        Feature name

    Returns
    -------
    Optional[str]
        "positive", "negative", "context_dependent", or None if unknown
    """
    feature_lower = feature.lower()

    for pattern_name, (regex, expected) in COEFFICIENT_PATTERNS.items():
        if re.search(regex, feature_lower, re.IGNORECASE):
            return expected

    return None


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "COEFFICIENT_PATTERNS",
    "validate_coefficient_sign",
    "validate_all_coefficients",
    "get_expected_sign",
]

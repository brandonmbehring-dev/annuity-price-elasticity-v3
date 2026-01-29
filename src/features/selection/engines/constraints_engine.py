"""
Economic Constraints Validation Engine for Feature Selection.

This module provides atomic functions for validating economic constraints
on regression coefficients following established pipeline architecture.

Key Functions:
- apply_economic_constraints: Main atomic function for constraint filtering
- validate_constraint_rule: Single constraint rule validation
- generate_constraint_violations: Detailed violation analysis

Design Principles:
- Single responsibility: Economic validation only
- Immutable operations: (results, rules) -> filtered results
- Business-context error messages and rationale
- Power-user configurability with sensible defaults
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

# Import types - Fail fast with clear error if imports fail
from src.features.selection_types import (
    EconomicConstraintConfig,
    ConstraintRule,
    ConstraintViolation,
    ConstraintType,
    AICResult,
    create_default_constraint_rules
)


def _check_sign_violation(
    coefficient: float,
    expected_sign: str,
    standard_error: float = None,
    confidence_level: float = 0.95
) -> bool:
    """
    Check if coefficient violates expected sign constraint using confidence intervals.

    Implements STRICT CI validation (Issue #6 fix): The entire confidence interval
    must have the correct sign for the constraint to pass. This prevents treating
    coefficients with high uncertainty (e.g., -0.0001 ± 0.5) as meaningful violations
    or passing coefficients that are statistically indistinguishable from zero.

    Parameters
    ----------
    coefficient : float
        Point estimate of the coefficient
    expected_sign : str
        Expected sign: "positive" or "negative"
    standard_error : float, optional
        Standard error of the coefficient. If None, falls back to point estimate check.
    confidence_level : float, default 0.95
        Confidence level for the interval (0.95 = 95% CI)

    Returns
    -------
    bool
        True if constraint is VIOLATED, False if satisfied

    Notes
    -----
    Strict CI validation:
    - Positive constraint: ci_lower > 0 (entire CI must be positive)
    - Negative constraint: ci_upper < 0 (entire CI must be negative)
    - If CI includes zero, the coefficient is statistically indistinguishable
      from zero and violates any sign constraint.
    """
    if coefficient is None or np.isnan(coefficient):
        return False

    # If no standard error provided, fall back to point estimate check
    # (backward compatibility, but logs warning)
    if standard_error is None or np.isnan(standard_error) or standard_error <= 0:
        # Point estimate only - legacy behavior
        if expected_sign == "positive" and coefficient <= 0:
            return True
        if expected_sign == "negative" and coefficient >= 0:
            return True
        return False

    # CI-based validation (Strict: entire CI must have correct sign)
    # Using z-critical for large samples; t-critical would require df
    # For 95% CI: z = 1.96
    from scipy import stats
    z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    ci_lower = coefficient - z_critical * standard_error
    ci_upper = coefficient + z_critical * standard_error

    if expected_sign == "positive":
        # Entire CI must be > 0 for constraint to be satisfied
        # If ci_lower <= 0, CI includes zero or negative values -> VIOLATED
        return ci_lower <= 0

    if expected_sign == "negative":
        # Entire CI must be < 0 for constraint to be satisfied
        # If ci_upper >= 0, CI includes zero or positive values -> VIOLATED
        return ci_upper >= 0

    return False


def _create_violation_record(
    feature: str,
    coefficient: float,
    rule: ConstraintRule,
    standard_error: float = None,
    confidence_level: float = 0.95
) -> ConstraintViolation:
    """
    Create a standardized constraint violation record with CI context.

    Parameters
    ----------
    feature : str
        Feature name that violated the constraint
    coefficient : float
        Point estimate of the coefficient
    rule : ConstraintRule
        The constraint rule that was violated
    standard_error : float, optional
        Standard error for CI calculation
    confidence_level : float, default 0.95
        Confidence level used for CI-based validation

    Returns
    -------
    ConstraintViolation
        Detailed violation record with CI context in rationale
    """
    severity = "ERROR" if rule.strict else "WARNING"

    # Build enhanced rationale with CI context if SE available
    rationale = rule.business_rationale
    if standard_error is not None and not np.isnan(standard_error) and standard_error > 0:
        from scipy import stats
        z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        ci_lower = coefficient - z_critical * standard_error
        ci_upper = coefficient + z_critical * standard_error
        rationale = (
            f"{rule.business_rationale} "
            f"[CI-based: coef={coefficient:.4f}, "
            f"{int(confidence_level*100)}% CI=({ci_lower:.4f}, {ci_upper:.4f}), "
            f"SE={standard_error:.4f}]"
        )

    return ConstraintViolation(
        feature_name=feature,
        actual_coefficient=coefficient,
        expected_sign=rule.expected_sign,
        constraint_type=rule.constraint_type,
        business_rationale=rationale,
        violation_severity=severity
    )


def validate_constraint_rule(
    coefficients: Dict[str, float],
    rule: ConstraintRule,
    standard_errors: Optional[Dict[str, float]] = None,
    confidence_level: float = 0.95
) -> List[ConstraintViolation]:
    """
    Validate single economic constraint rule against model coefficients using CI.

    Parameters
    ----------
    coefficients : Dict[str, float]
        Model coefficients from regression (feature -> coefficient)
    rule : ConstraintRule
        Single constraint rule with pattern, expected sign, and rationale
    standard_errors : Dict[str, float], optional
        Standard errors for each coefficient (enables CI-based validation)
    confidence_level : float, default 0.95
        Confidence level for CI-based constraint validation

    Returns
    -------
    List[ConstraintViolation]
        List of violations (empty if rule is satisfied)

    Notes
    -----
    When standard_errors are provided, uses STRICT CI validation:
    the entire confidence interval must have the correct sign.
    Falls back to point estimate check if standard_errors not available.
    """
    violations = []

    try:
        matching_features = [
            f for f in coefficients.keys()
            if rule.feature_pattern in f and f != 'Intercept'
        ]

        if not matching_features:
            return violations

        for feature in matching_features:
            coefficient = coefficients[feature]
            # Get standard error if available
            se = None
            if standard_errors is not None and feature in standard_errors:
                se = standard_errors[feature]

            if _check_sign_violation(coefficient, rule.expected_sign, se, confidence_level):
                violations.append(_create_violation_record(
                    feature, coefficient, rule, se, confidence_level
                ))

        return violations

    except Exception as e:
        return [ConstraintViolation(
            feature_name=f"RULE_ERROR_{rule.feature_pattern}",
            actual_coefficient=np.nan,
            expected_sign=rule.expected_sign,
            constraint_type=rule.constraint_type,
            business_rationale=f"Rule validation failed: {str(e)}",
            violation_severity="ERROR"
        )]


def generate_constraint_violations(
    results_df: pd.DataFrame,
    constraint_rules: List[ConstraintRule],
    confidence_level: float = 0.95
) -> List[ConstraintViolation]:
    """
    Generate comprehensive constraint violation analysis across all models.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with 'coefficients' and optionally 'standard_errors' columns
    constraint_rules : List[ConstraintRule]
        Economic constraint rules to validate
    confidence_level : float, default 0.95
        Confidence level for CI-based validation

    Returns
    -------
    List[ConstraintViolation]
        All violations found across all models

    Notes
    -----
    If 'standard_errors' column exists in results_df, uses CI-based validation.
    Otherwise falls back to point estimate validation (legacy behavior).
    """
    all_violations = []

    if results_df.empty or 'coefficients' not in results_df.columns:
        return all_violations

    # Check if standard errors are available for CI-based validation
    has_standard_errors = 'standard_errors' in results_df.columns

    try:
        # Check each model's coefficients against all rules
        for idx, row in results_df.iterrows():
            if not row.get('converged', False):
                continue  # Skip non-converged models

            coefficients = row['coefficients']
            if not isinstance(coefficients, dict):
                continue  # Skip malformed coefficient data

            # Extract standard errors if available
            standard_errors = None
            if has_standard_errors:
                se_data = row.get('standard_errors')
                if isinstance(se_data, dict):
                    standard_errors = se_data

            # Apply each rule to this model's coefficients
            for rule in constraint_rules:
                model_violations = validate_constraint_rule(
                    coefficients, rule, standard_errors, confidence_level
                )

                # Add model context to violations
                for violation in model_violations:
                    # Add model identification for debugging
                    violation.feature_name = f"Model_{idx}_{violation.feature_name}"
                    all_violations.append(violation)

        return all_violations

    except Exception as e:
        # Create system error violation
        system_violation = ConstraintViolation(
            feature_name="SYSTEM_ERROR",
            actual_coefficient=np.nan,
            expected_sign="unknown",
            constraint_type=ConstraintType.COMPETITOR_NEGATIVE,  # Default type
            business_rationale=f"Constraint validation system error: {str(e)}",
            violation_severity="ERROR"
        )
        return [system_violation]


def _build_constraint_rules(config: EconomicConstraintConfig) -> List[ConstraintRule]:
    """
    Build constraint rules from configuration.

    Parameters
    ----------
    config : EconomicConstraintConfig
        Constraint configuration with optional custom rules

    Returns
    -------
    List[ConstraintRule]
        List of constraint rules to apply
    """
    if 'custom_rules' in config and config['custom_rules']:
        print(f"Using {len(config['custom_rules'])} custom constraint rules")
        return [ConstraintRule(**rule) for rule in config['custom_rules']]

    constraint_rules = create_default_constraint_rules()
    print(f"Using {len(constraint_rules)} default economic constraint rules")
    return constraint_rules


def _filter_violated_models(
    results_df: pd.DataFrame,
    violations: List[ConstraintViolation]
) -> pd.DataFrame:
    """
    Filter out models with constraint violations in strict mode.

    Parameters
    ----------
    results_df : pd.DataFrame
        Original results DataFrame
    violations : List[ConstraintViolation]
        Detected violations with model indices

    Returns
    -------
    pd.DataFrame
        Filtered results with violated models removed
    """
    violation_model_indices = set()
    for violation in violations:
        if violation.violation_severity == "ERROR":
            try:
                model_idx = int(violation.feature_name.split('_')[1])
                violation_model_indices.add(model_idx)
            except (IndexError, ValueError):
                pass

    if not violation_model_indices:
        return results_df.copy()

    all_positions = set(range(len(results_df)))
    valid_positions = all_positions - violation_model_indices
    valid_results = results_df.iloc[list(valid_positions)].reset_index(drop=True)
    print(f"Strict validation: Removed {len(violation_model_indices)} models")
    print(f"Removed models at positions: {sorted(violation_model_indices)}")
    return valid_results


def apply_economic_constraints(
    results_df: pd.DataFrame,
    config: EconomicConstraintConfig
) -> Tuple[pd.DataFrame, List[ConstraintViolation]]:
    """Apply economic constraints to filter AIC results. Returns (filtered_df, violations)."""
    if results_df.empty:
        return results_df.copy(), []

    if not config.get('enabled', False):
        print("Economic constraints disabled - returning all results")
        return results_df.copy(), []

    if 'coefficients' not in results_df.columns:
        print("WARNING: No coefficients column - cannot apply constraints")
        return results_df.copy(), []

    try:
        constraint_rules = _build_constraint_rules(config)
        all_violations = generate_constraint_violations(results_df, constraint_rules)

        strict_mode = config.get('strict_validation', True)
        if strict_mode and all_violations:
            valid_results = _filter_violated_models(results_df, all_violations)
        else:
            valid_results = results_df.copy()

        print(f"Economic constraints applied: {len(valid_results)}/{len(results_df)} valid")
        print(f"Total violations found: {len(all_violations)}")
        return valid_results, all_violations

    except Exception as e:
        print(f"ERROR: Economic constraint validation failed: {e}")
        error_violation = ConstraintViolation(
            feature_name="CONSTRAINT_ENGINE_ERROR",
            actual_coefficient=np.nan,
            expected_sign="unknown",
            constraint_type=ConstraintType.COMPETITOR_NEGATIVE,
            business_rationale=f"Constraint engine failure: {str(e)}",
            violation_severity="ERROR"
        )
        return results_df.copy(), [error_violation]
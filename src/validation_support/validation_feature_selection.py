"""
Feature Selection Validation Module for Mathematical Equivalence.

This module contains the MathematicalEquivalenceValidator class and related
functions for validating AIC calculations, bootstrap stability, economic
constraints, and model selection in feature selection pipelines.

Module Responsibilities:
- AIC calculation equivalence validation
- Bootstrap stability metrics validation
- Economic constraint validation
- Final model selection validation
- Comprehensive validation orchestration

Used by: mathematical_equivalence.py (re-exports for backward compatibility)
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.validation_support.validation_constants import (
    TOLERANCE,
    BOOTSTRAP_STATISTICAL_TOLERANCE,
    ValidationResult,
)

# Configure for high precision mathematical operations
np.set_printoptions(precision=15)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE SELECTION VALIDATION
# =============================================================================


class MathematicalEquivalenceValidator:
    """
    Comprehensive mathematical equivalence validation for feature selection.

    Validates that enhanced feature selection methods maintain mathematical
    equivalence with proven baseline implementations.

    Usage:
        validator = MathematicalEquivalenceValidator()
        validator.capture_baseline_results(aic_df, constraints_df, bootstrap_list, final)
        report = validator.run_comprehensive_validation(test_aic, test_constraints, ...)
    """

    DEFAULT_PRECISION = TOLERANCE
    BOOTSTRAP_STATISTICAL_TOLERANCE = BOOTSTRAP_STATISTICAL_TOLERANCE

    def __init__(self, precision: float = DEFAULT_PRECISION):
        """
        Initialize validator with specified precision tolerance.

        Args:
            precision: Maximum allowed difference for mathematical equivalence
        """
        self.precision = precision
        self.validation_results: List[ValidationResult] = []
        self.baseline_data: Dict[str, Any] = {}

    def capture_baseline_results(
        self,
        aic_results: pd.DataFrame,
        constraint_results: pd.DataFrame,
        bootstrap_results: List[Dict[str, Any]],
        final_model: Dict[str, Any]
    ) -> None:
        """
        Capture baseline results from refactored notebook for validation.

        Args:
            aic_results: DataFrame with AIC calculations for all models
            constraint_results: DataFrame with economically valid models
            bootstrap_results: List of bootstrap analysis results
            final_model: Final selected model dictionary
        """
        self.baseline_data = {
            'aic_results': aic_results.copy(),
            'constraint_results': constraint_results.copy(),
            'bootstrap_results': [result.copy() for result in bootstrap_results],
            'final_model': final_model.copy(),
            'capture_timestamp': datetime.now().isoformat(),
            'total_models_evaluated': len(aic_results),
            'total_valid_models': len(constraint_results),
            'bootstrap_models_analyzed': len(bootstrap_results),
            'final_selected_features': final_model.get('features', ''),
            'final_aic_score': final_model.get('aic', 0.0),
            'final_r_squared': final_model.get('r_squared', 0.0)
        }

    def _create_error_result(
        self, test_name: str, error: str, tolerance: float = None
    ) -> ValidationResult:
        """Create a ValidationResult for error conditions."""
        return ValidationResult(
            test_name=test_name,
            passed=False,
            max_difference=float('inf'),
            tolerance=tolerance or self.precision,
            baseline_value=None,
            test_value=None,
            details={'error': error},
            timestamp=datetime.now().isoformat()
        )

    def validate_aic_calculations(
        self, test_aic_results: pd.DataFrame
    ) -> ValidationResult:
        """
        Validate AIC calculations for mathematical equivalence.

        Args:
            test_aic_results: AIC results from enhanced implementation

        Returns:
            ValidationResult with detailed comparison metrics
        """
        test_name = "aic_calculations_equivalence"

        if 'aic_results' not in self.baseline_data:
            return self._create_error_result(test_name, 'No baseline data captured')

        baseline_aic = self.baseline_data['aic_results']

        if baseline_aic.shape != test_aic_results.shape:
            return self._create_error_result(
                test_name,
                f'Shape mismatch: {baseline_aic.shape} vs {test_aic_results.shape}'
            )

        aic_diffs = np.abs(baseline_aic['aic'].values - test_aic_results['aic'].values)
        r2_diffs = np.abs(baseline_aic['r_squared'].values - test_aic_results['r_squared'].values)
        max_aic_diff = np.max(aic_diffs)
        max_r2_diff = np.max(r2_diffs)
        max_difference = max(max_aic_diff, max_r2_diff)
        passed = max_difference <= self.precision

        details = {
            'max_aic_difference': max_aic_diff,
            'max_r2_difference': max_r2_diff,
            'models_compared': len(baseline_aic),
            'precision_achieved': max_difference,
            'models_with_differences': int(np.sum(aic_diffs > 0)),
            'validation_status': 'PERFECT_EQUIVALENCE' if passed else 'EQUIVALENCE_FAILED'
        }

        if not passed:
            worst_indices = np.argsort(aic_diffs)[-5:]
            details['worst_models'] = [
                {
                    'index': int(idx),
                    'baseline_aic': float(baseline_aic.iloc[idx]['aic']),
                    'test_aic': float(test_aic_results.iloc[idx]['aic']),
                    'difference': float(aic_diffs[idx])
                }
                for idx in worst_indices
            ]

        result = ValidationResult(
            test_name=test_name,
            passed=passed,
            max_difference=max_difference,
            tolerance=self.precision,
            baseline_value=f"{len(baseline_aic)} models, best AIC: {baseline_aic['aic'].min():.6f}",
            test_value=f"{len(test_aic_results)} models, best AIC: {test_aic_results['aic'].min():.6f}",
            details=details,
            timestamp=datetime.now().isoformat()
        )

        self.validation_results.append(result)
        return result

    def validate_bootstrap_stability_metrics(
        self, test_bootstrap_results: List[Dict[str, Any]]
    ) -> ValidationResult:
        """
        Validate bootstrap stability metrics for mathematical equivalence.

        Args:
            test_bootstrap_results: Bootstrap results from enhanced implementation

        Returns:
            ValidationResult with detailed statistical comparison
        """
        test_name = "bootstrap_stability_metrics_equivalence"

        if 'bootstrap_results' not in self.baseline_data:
            return self._create_error_result(
                test_name, 'No baseline bootstrap data captured',
                self.BOOTSTRAP_STATISTICAL_TOLERANCE
            )

        baseline_bootstrap = self.baseline_data['bootstrap_results']

        if len(baseline_bootstrap) != len(test_bootstrap_results):
            return self._create_error_result(
                test_name,
                f'Model count mismatch: {len(baseline_bootstrap)} vs {len(test_bootstrap_results)}',
                self.BOOTSTRAP_STATISTICAL_TOLERANCE
            )

        comparisons = []
        for i, (b, t) in enumerate(zip(baseline_bootstrap, test_bootstrap_results)):
            baseline_cv = b.get('aic_stability_cv', 0.0)
            test_cv = t.get('aic_stability_cv', 0.0)
            baseline_median = b.get('median_aic', 0.0)
            test_median = t.get('median_aic', 0.0)
            comparisons.append({
                'model_index': i,
                'features': b.get('features', ''),
                'cv_difference': abs(baseline_cv - test_cv),
                'median_difference': abs(baseline_median - test_median),
                'baseline_cv': baseline_cv,
                'test_cv': test_cv
            })

        max_cv_diff = max(c['cv_difference'] for c in comparisons)
        max_median_diff = max(c['median_difference'] for c in comparisons)
        max_difference = max(max_cv_diff, max_median_diff)
        passed = max_difference <= self.BOOTSTRAP_STATISTICAL_TOLERANCE

        result = ValidationResult(
            test_name=test_name,
            passed=passed,
            max_difference=max_difference,
            tolerance=self.BOOTSTRAP_STATISTICAL_TOLERANCE,
            baseline_value=f"{len(baseline_bootstrap)} models analyzed",
            test_value=f"{len(test_bootstrap_results)} models analyzed",
            details={
                'max_cv_difference': max_cv_diff,
                'max_median_difference': max_median_diff,
                'models_compared': len(baseline_bootstrap),
                'precision_achieved': max_difference,
                'model_comparisons': comparisons[:3]
            },
            timestamp=datetime.now().isoformat()
        )

        self.validation_results.append(result)
        return result

    def validate_economic_constraints(
        self, test_constraint_results: pd.DataFrame
    ) -> ValidationResult:
        """
        Validate economic constraint validation results.

        Args:
            test_constraint_results: Constraint validation results from enhanced implementation

        Returns:
            ValidationResult with constraint comparison details
        """
        test_name = "economic_constraints_equivalence"

        if 'constraint_results' not in self.baseline_data:
            return self._create_error_result(test_name, 'No baseline constraint data captured')

        baseline_constraints = self.baseline_data['constraint_results']
        baseline_count = len(baseline_constraints)
        test_count = len(test_constraint_results)
        count_difference = abs(baseline_count - test_count)

        exact_match = False
        if baseline_count == test_count:
            baseline_sorted = baseline_constraints.sort_values('aic')['features'].tolist()
            test_sorted = test_constraint_results.sort_values('aic')['features'].tolist()
            exact_match = baseline_sorted == test_sorted

        passed = (count_difference == 0) and exact_match

        details = {
            'baseline_valid_count': baseline_count,
            'test_valid_count': test_count,
            'count_difference': count_difference,
            'exact_model_match': exact_match,
            'validation_status': 'EXACT_MATCH' if exact_match else 'MISMATCH'
        }

        if not exact_match and baseline_count == test_count:
            baseline_features = set(baseline_constraints['features'])
            test_features = set(test_constraint_results['features'])
            details['missing_from_test'] = list(baseline_features - test_features)
            details['extra_in_test'] = list(test_features - baseline_features)

        result = ValidationResult(
            test_name=test_name,
            passed=passed,
            max_difference=count_difference,
            tolerance=0,
            baseline_value=f"{baseline_count} valid models",
            test_value=f"{test_count} valid models",
            details=details,
            timestamp=datetime.now().isoformat()
        )

        self.validation_results.append(result)
        return result

    def validate_final_model_selection(
        self, test_final_model: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate final model selection matches baseline.

        Args:
            test_final_model: Final selected model from enhanced implementation

        Returns:
            ValidationResult with model comparison details
        """
        test_name = "final_model_selection_equivalence"

        if 'final_model' not in self.baseline_data:
            return self._create_error_result(test_name, 'No baseline final model captured')

        baseline_model = self.baseline_data['final_model']

        baseline_features = baseline_model.get('features', '')
        test_features = test_final_model.get('features', '')
        features_match = baseline_features == test_features

        baseline_aic = baseline_model.get('aic', 0.0)
        test_aic = test_final_model.get('aic', 0.0)
        aic_difference = abs(baseline_aic - test_aic)
        aic_match = aic_difference <= self.precision

        baseline_r2 = baseline_model.get('r_squared', 0.0)
        test_r2 = test_final_model.get('r_squared', 0.0)
        r2_difference = abs(baseline_r2 - test_r2)
        r2_match = r2_difference <= self.precision

        passed = features_match and aic_match and r2_match
        max_difference = max(aic_difference, r2_difference)

        result = ValidationResult(
            test_name=test_name,
            passed=passed,
            max_difference=max_difference,
            tolerance=self.precision,
            baseline_value=f"Features: {baseline_features}, AIC: {baseline_aic:.6f}",
            test_value=f"Features: {test_features}, AIC: {test_aic:.6f}",
            details={
                'features_match': features_match,
                'baseline_features': baseline_features,
                'test_features': test_features,
                'aic_match': aic_match,
                'aic_difference': aic_difference,
                'r2_match': r2_match,
                'r2_difference': r2_difference,
                'validation_status': 'EXACT_MATCH' if passed else 'MISMATCH'
            },
            timestamp=datetime.now().isoformat()
        )

        self.validation_results.append(result)
        return result

    def run_comprehensive_validation(
        self,
        test_aic_results: pd.DataFrame,
        test_constraint_results: pd.DataFrame,
        test_bootstrap_results: List[Dict[str, Any]],
        test_final_model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run comprehensive mathematical equivalence validation.

        Args:
            test_aic_results: AIC results from enhanced implementation
            test_constraint_results: Constraint results from enhanced implementation
            test_bootstrap_results: Bootstrap results from enhanced implementation
            test_final_model: Final model from enhanced implementation

        Returns:
            Comprehensive validation report with pass/fail status
        """
        print("Running comprehensive mathematical equivalence validation...")

        results = [
            self.validate_aic_calculations(test_aic_results),
            self.validate_economic_constraints(test_constraint_results),
            self.validate_bootstrap_stability_metrics(test_bootstrap_results),
            self.validate_final_model_selection(test_final_model)
        ]

        all_passed = all(r.passed for r in results)

        return {
            'overall_validation_passed': all_passed,
            'validation_timestamp': datetime.now().isoformat(),
            'precision_used': self.precision,
            'bootstrap_tolerance': self.BOOTSTRAP_STATISTICAL_TOLERANCE,
            'test_results': {
                'aic_calculations': {'passed': results[0].passed, 'max_difference': results[0].max_difference},
                'economic_constraints': {'passed': results[1].passed, 'max_difference': results[1].max_difference},
                'bootstrap_stability': {'passed': results[2].passed, 'max_difference': results[2].max_difference},
                'final_model_selection': {'passed': results[3].passed, 'max_difference': results[3].max_difference}
            },
            'summary': {
                'tests_run': len(results),
                'tests_passed': sum(r.passed for r in results),
                'critical_failures': [r.test_name for r in results if not r.passed],
                'recommendation': (
                    "APPROVED: Mathematical equivalence validated - safe to deploy"
                    if all_passed else
                    "BLOCKED: Mathematical equivalence validation failed - fix issues"
                )
            }
        }

    def save_baseline_data(self, filepath: str) -> None:
        """Save baseline data to file for future validation runs."""
        with open(filepath, 'w') as f:
            baseline_copy = self.baseline_data.copy()
            if 'aic_results' in baseline_copy:
                baseline_copy['aic_results'] = baseline_copy['aic_results'].to_dict('records')
            if 'constraint_results' in baseline_copy:
                baseline_copy['constraint_results'] = baseline_copy['constraint_results'].to_dict('records')
            json.dump(baseline_copy, f, indent=2, default=str)

    def load_baseline_data(self, filepath: str) -> None:
        """Load baseline data from file."""
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
            if 'aic_results' in loaded_data:
                loaded_data['aic_results'] = pd.DataFrame(loaded_data['aic_results'])
            if 'constraint_results' in loaded_data:
                loaded_data['constraint_results'] = pd.DataFrame(loaded_data['constraint_results'])
            self.baseline_data = loaded_data


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def validate_mathematical_equivalence_comprehensive(
    baseline_aic_results: pd.DataFrame,
    baseline_constraint_results: pd.DataFrame,
    baseline_bootstrap_results: List[Dict[str, Any]],
    baseline_final_model: Dict[str, Any],
    test_aic_results: pd.DataFrame,
    test_constraint_results: pd.DataFrame,
    test_bootstrap_results: List[Dict[str, Any]],
    test_final_model: Dict[str, Any],
    precision: float = TOLERANCE
) -> Dict[str, Any]:
    """
    Convenience function for comprehensive feature selection validation.

    Args:
        baseline_*: Results from refactored notebook baseline
        test_*: Results from enhanced implementation
        precision: Tolerance for mathematical equivalence

    Returns:
        Comprehensive validation report
    """
    validator = MathematicalEquivalenceValidator(precision=precision)
    validator.capture_baseline_results(
        baseline_aic_results,
        baseline_constraint_results,
        baseline_bootstrap_results,
        baseline_final_model
    )
    return validator.run_comprehensive_validation(
        test_aic_results,
        test_constraint_results,
        test_bootstrap_results,
        test_final_model
    )


def _compare_models_for_equivalence(
    model1: Dict[str, Any],
    model2: Dict[str, Any],
    tolerance: float
) -> Dict[str, Any]:
    """Compare two model dictionaries for equivalence."""
    result = {'differences': [], 'max_absolute_difference': 0.0, 'max_relative_difference': 0.0}

    # Compare features
    if model1.get('features') != model2.get('features'):
        result['differences'].append({
            'comparison_type': 'selected_model_features',
            'baseline_features': model1.get('features'),
            'original_features': model2.get('features'),
            'exceeds_tolerance': True
        })

    # Compare numerical metrics
    for metric in ['aic', 'r_squared', 'n_features']:
        if metric in model1 and metric in model2:
            diff = abs(model1[metric] - model2[metric])
            result['max_absolute_difference'] = max(result['max_absolute_difference'], diff)
            if diff > tolerance:
                result['differences'].append({
                    'comparison_type': f'model_{metric}',
                    'difference': diff,
                    'exceeds_tolerance': True
                })

    return result

"""
RidgeCV Feature Selection Engine.

Alternative to OLS AIC method for TD-05 comparison. Uses cross-validated
Ridge regression for feature selection, which may be more appropriate
when regularization is used in the final model.

Design Philosophy:
- Uses RidgeCV for integrated hyperparameter selection
- Cross-validation provides honest error estimates
- Avoids AIC approximation issues with regularized models

TD-05 Context:
- Current default uses OLS AIC for feature selection
- Final model uses Ridge regression
- This engine provides an apples-to-apples comparison

Usage:
    from src.features.selection.engines.ridge_cv_engine import (
        RidgeCVConfig, evaluate_ridge_cv_combinations
    )

    config = RidgeCVConfig(alphas=(0.001, 0.01, 0.1, 1.0, 10.0))
    results = evaluate_ridge_cv_combinations(
        data=df,
        target="sales_target_current",
        candidate_features=["feat1", "feat2", "feat3"],
        config=config,
    )
    print(f"Best features: {results.best_features}")
    print(f"Best alpha: {results.best_alpha}")
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_percentage_error


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class RidgeCVConfig:
    """Configuration for RidgeCV feature selection.

    Attributes
    ----------
    alphas : Tuple[float, ...]
        Regularization strengths to evaluate (default: log-spaced 1e-3 to 10)
    cv_folds : int
        Number of cross-validation folds (default: 5)
    scoring : str
        Scoring metric: "r2", "neg_mean_absolute_percentage_error" (default: "r2")
    max_features : int
        Maximum features in any combination (default: 5)
    min_features : int
        Minimum features in any combination (default: 1)
    require_own_rate : bool
        Whether to require own-rate feature in all combinations (default: True)
    own_rate_pattern : str
        Pattern to identify own-rate feature (default: "prudential_rate")

    Examples
    --------
    >>> config = RidgeCVConfig(
    ...     alphas=(0.01, 0.1, 1.0, 10.0),
    ...     cv_folds=10,
    ...     max_features=3,
    ... )
    """
    alphas: Tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    cv_folds: int = 5
    scoring: str = "r2"
    max_features: int = 5
    min_features: int = 1
    require_own_rate: bool = True
    own_rate_pattern: str = "prudential_rate"


# =============================================================================
# RESULTS
# =============================================================================


@dataclass
class FeatureCombinationResult:
    """Result for a single feature combination.

    Attributes
    ----------
    features : Tuple[str, ...]
        Feature names in this combination
    cv_score_mean : float
        Mean cross-validation score
    cv_score_std : float
        Standard deviation of CV scores
    best_alpha : float
        Best alpha selected by RidgeCV
    n_features : int
        Number of features
    coefficients : Dict[str, float]
        Fitted coefficients for each feature
    """
    features: Tuple[str, ...]
    cv_score_mean: float
    cv_score_std: float
    best_alpha: float
    n_features: int
    coefficients: Dict[str, float]

    @property
    def features_str(self) -> str:
        """Features as '+'-joined string."""
        return " + ".join(self.features)


@dataclass
class RidgeCVResults:
    """Complete results from RidgeCV feature selection.

    Attributes
    ----------
    all_results : List[FeatureCombinationResult]
        Results for all evaluated combinations
    best_result : FeatureCombinationResult
        Best performing combination
    config : RidgeCVConfig
        Configuration used
    n_combinations_evaluated : int
        Total combinations tested

    Properties
    ----------
    best_features : List[str]
        Features in the best combination
    best_alpha : float
        Optimal regularization strength
    best_cv_score : float
        Best CV score achieved
    """
    all_results: List[FeatureCombinationResult]
    best_result: FeatureCombinationResult
    config: RidgeCVConfig
    n_combinations_evaluated: int

    @property
    def best_features(self) -> List[str]:
        """Features in the best combination."""
        return list(self.best_result.features)

    @property
    def best_alpha(self) -> float:
        """Optimal regularization strength."""
        return self.best_result.best_alpha

    @property
    def best_cv_score(self) -> float:
        """Best cross-validation score."""
        return self.best_result.cv_score_mean

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "best_features": self.best_features,
            "best_alpha": self.best_alpha,
            "best_cv_score": self.best_cv_score,
            "best_cv_score_std": self.best_result.cv_score_std,
            "best_coefficients": self.best_result.coefficients,
            "n_combinations_evaluated": self.n_combinations_evaluated,
            "config": {
                "alphas": self.config.alphas,
                "cv_folds": self.config.cv_folds,
                "scoring": self.config.scoring,
                "max_features": self.config.max_features,
            },
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "RidgeCV Feature Selection Results",
            "=" * 40,
            f"Best Features: {self.best_features}",
            f"Best Alpha: {self.best_alpha:.4f}",
            f"Best CV Score ({self.config.scoring}): {self.best_cv_score:.4f} (+/- {self.best_result.cv_score_std:.4f})",
            f"Combinations Evaluated: {self.n_combinations_evaluated}",
            "",
            "Coefficients:",
        ]
        for feat, coef in self.best_result.coefficients.items():
            lines.append(f"  {feat}: {coef:.6f}")

        return "\n".join(lines)

    def top_n(self, n: int = 5) -> List[FeatureCombinationResult]:
        """Get top N combinations by CV score."""
        sorted_results = sorted(
            self.all_results,
            key=lambda x: x.cv_score_mean,
            reverse=True,
        )
        return sorted_results[:n]


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================


def _evaluate_single_combination(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Tuple[str, ...],
    config: RidgeCVConfig,
) -> FeatureCombinationResult:
    """Evaluate a single feature combination with RidgeCV.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix for this combination
    y : np.ndarray
        Target variable
    feature_names : Tuple[str, ...]
        Names of features in X
    config : RidgeCVConfig
        Configuration for evaluation

    Returns
    -------
    FeatureCombinationResult
        Evaluation results for this combination
    """
    # Create RidgeCV model - sklearn requires list, not tuple for alphas
    alphas_list = list(config.alphas)
    model = RidgeCV(
        alphas=alphas_list,
        cv=config.cv_folds,
        scoring=config.scoring if config.scoring != "mape" else None,
    )

    # Fit to get best alpha
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    # Get cross-validation scores with the selected alpha
    cv_scores = cross_val_score(
        RidgeCV(alphas=[model.alpha_], cv=config.cv_folds),
        X, y,
        cv=config.cv_folds,
        scoring=config.scoring if config.scoring != "mape" else "neg_mean_absolute_percentage_error",
    )

    # Extract coefficients
    coefficients = dict(zip(feature_names, model.coef_.tolist()))

    return FeatureCombinationResult(
        features=feature_names,
        cv_score_mean=float(np.mean(cv_scores)),
        cv_score_std=float(np.std(cv_scores)),
        best_alpha=float(model.alpha_),
        n_features=len(feature_names),
        coefficients=coefficients,
    )


def _generate_feature_combinations(
    candidate_features: List[str],
    config: RidgeCVConfig,
) -> List[Tuple[str, ...]]:
    """Generate all valid feature combinations.

    Parameters
    ----------
    candidate_features : List[str]
        All candidate features to consider
    config : RidgeCVConfig
        Configuration with constraints

    Returns
    -------
    List[Tuple[str, ...]]
        All valid feature combinations
    """
    # Find own-rate feature if required
    own_rate_feature = None
    if config.require_own_rate:
        for f in candidate_features:
            if config.own_rate_pattern.lower() in f.lower():
                own_rate_feature = f
                break

        if own_rate_feature is None:
            raise ValueError(
                f"require_own_rate=True but no feature matches pattern "
                f"'{config.own_rate_pattern}' in {candidate_features}"
            )

    all_combinations = []

    for n in range(config.min_features, config.max_features + 1):
        if config.require_own_rate and own_rate_feature:
            # Generate combinations that include own-rate
            other_features = [f for f in candidate_features if f != own_rate_feature]

            if n == 1:
                all_combinations.append((own_rate_feature,))
            else:
                for combo in combinations(other_features, n - 1):
                    all_combinations.append((own_rate_feature,) + combo)
        else:
            # All combinations without constraint
            for combo in combinations(candidate_features, n):
                all_combinations.append(combo)

    return all_combinations


def evaluate_ridge_cv_combinations(
    data: pd.DataFrame,
    target: str,
    candidate_features: List[str],
    config: Optional[RidgeCVConfig] = None,
) -> RidgeCVResults:
    """Evaluate all feature combinations using RidgeCV.

    Uses cross-validated Ridge regression to evaluate each feature
    combination and select the best one based on CV score.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with features and target
    target : str
        Target column name
    candidate_features : List[str]
        Features to consider for selection
    config : Optional[RidgeCVConfig]
        Configuration (uses defaults if not provided)

    Returns
    -------
    RidgeCVResults
        Complete evaluation results with best combination

    Raises
    ------
    ValueError
        If data is empty, target missing, or no valid combinations

    Examples
    --------
    >>> config = RidgeCVConfig(max_features=3)
    >>> results = evaluate_ridge_cv_combinations(
    ...     data=df,
    ...     target="sales",
    ...     candidate_features=["rate", "comp_t2", "comp_t3"],
    ...     config=config,
    ... )
    >>> print(results.best_features)
    ['rate', 'comp_t2']
    """
    if config is None:
        config = RidgeCVConfig()

    # Validate inputs
    if data.empty:
        raise ValueError("Data is empty")

    if target not in data.columns:
        raise ValueError(f"Target '{target}' not in data columns")

    missing_features = [f for f in candidate_features if f not in data.columns]
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")

    # Prepare data
    y = data[target].values

    # Generate all combinations
    all_combinations = _generate_feature_combinations(candidate_features, config)

    if not all_combinations:
        raise ValueError("No valid feature combinations generated")

    # Evaluate each combination
    all_results = []
    for combo in all_combinations:
        X = data[list(combo)].values

        try:
            result = _evaluate_single_combination(X, y, combo, config)
            all_results.append(result)
        except Exception as e:
            # Skip combinations that fail (e.g., singular matrix)
            warnings.warn(f"Skipped combination {combo}: {e}")
            continue

    if not all_results:
        raise ValueError("All feature combinations failed evaluation")

    # Find best result
    best_result = max(all_results, key=lambda x: x.cv_score_mean)

    return RidgeCVResults(
        all_results=all_results,
        best_result=best_result,
        config=config,
        n_combinations_evaluated=len(all_results),
    )


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================


def compare_with_aic_selection(
    ridge_results: RidgeCVResults,
    aic_features: List[str],
    aic_score: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare RidgeCV selection with AIC-based selection.

    Utility for TD-05 research comparing the two methods.

    Parameters
    ----------
    ridge_results : RidgeCVResults
        Results from RidgeCV feature selection
    aic_features : List[str]
        Features selected by AIC method
    aic_score : Optional[float]
        AIC score if available

    Returns
    -------
    Dict[str, Any]
        Comparison metrics
    """
    ridge_features = set(ridge_results.best_features)
    aic_features_set = set(aic_features)

    overlap = ridge_features & aic_features_set
    ridge_only = ridge_features - aic_features_set
    aic_only = aic_features_set - ridge_features

    jaccard = len(overlap) / len(ridge_features | aic_features_set) if (ridge_features | aic_features_set) else 0

    return {
        "ridge_features": list(ridge_features),
        "aic_features": aic_features,
        "overlap": list(overlap),
        "ridge_only": list(ridge_only),
        "aic_only": list(aic_only),
        "jaccard_similarity": jaccard,
        "ridge_cv_score": ridge_results.best_cv_score,
        "ridge_alpha": ridge_results.best_alpha,
        "aic_score": aic_score,
        "same_selection": ridge_features == aic_features_set,
    }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "RidgeCVConfig",
    "RidgeCVResults",
    "FeatureCombinationResult",
    "evaluate_ridge_cv_combinations",
    "compare_with_aic_selection",
]

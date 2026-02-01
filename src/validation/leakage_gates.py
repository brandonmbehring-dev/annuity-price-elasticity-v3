"""
Leakage Detection Gates for Annuity Price Elasticity.

Implements automated validation gates to detect data leakage:
1. Lag-0 detection - Forbidden concurrent competitor features
2. R-squared threshold - Suspiciously high R-squared indicates leakage
3. Improvement threshold - Large improvements suggest leakage
4. Temporal boundary check - No future data in features
5. Shuffled target test - Model should fail on shuffled y
6. Coefficient sign validation - Economic constraints (own-rate positive, competitors negative)

Usage:
    from src.validation.leakage_gates import (
        run_all_gates,
        run_shuffled_target_test,
        check_r_squared_threshold,
        check_improvement_threshold,
        detect_lag0_features,
        check_coefficient_signs,
    )

    # Run all gates
    report = run_all_gates(model, X, y, baseline_r2=0.15, coefficients={"own_rate": 0.5})

    # Individual gates
    passed = run_shuffled_target_test(model, X, y)
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from src.validation.coefficient_patterns import validate_all_coefficients

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================


class GateStatus(Enum):
    """Validation gate status."""

    PASS = "PASS"
    WARN = "WARN"
    HALT = "HALT"


@dataclass
class GateResult:
    """Result of a single validation gate."""

    gate_name: str
    status: GateStatus
    message: str
    metric_value: float | None = None
    threshold: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status_symbol = {"PASS": "[PASS]", "WARN": "[WARN]", "HALT": "[HALT]"}[self.status.value]
        return f"{status_symbol} {self.gate_name}: {self.message}"


@dataclass
class LeakageReport:
    """Complete leakage validation report."""

    gates: list[GateResult] = field(default_factory=list)
    timestamp: str = ""
    model_name: str = ""
    dataset_name: str = ""

    @property
    def passed(self) -> bool:
        """All gates passed (no HALT status)."""
        return all(g.status != GateStatus.HALT for g in self.gates)

    @property
    def has_warnings(self) -> bool:
        """Any gates have warnings."""
        return any(g.status == GateStatus.WARN for g in self.gates)

    @property
    def halt_count(self) -> int:
        """Number of gates that halted."""
        return sum(1 for g in self.gates if g.status == GateStatus.HALT)

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "Leakage Validation Report",
            "=" * 60,
            f"Model: {self.model_name or 'Unknown'}",
            f"Dataset: {self.dataset_name or 'Unknown'}",
            f"Timestamp: {self.timestamp or 'N/A'}",
            "-" * 60,
        ]

        for gate in self.gates:
            lines.append(str(gate))

        lines.append("-" * 60)
        status = "PASSED" if self.passed else f"FAILED ({self.halt_count} halts)"
        lines.append(f"Overall Status: {status}")
        lines.append("=" * 60)

        return "\n".join(lines)


# =============================================================================
# THRESHOLDS
# =============================================================================

# Conservative thresholds (err on side of investigation)
R_SQUARED_HALT_THRESHOLD = 0.30  # Unusually high for this domain
R_SQUARED_WARN_THRESHOLD = 0.20  # Worth investigating
IMPROVEMENT_HALT_THRESHOLD = 0.20  # 20% improvement is suspicious
IMPROVEMENT_WARN_THRESHOLD = 0.10  # 10% worth checking
SHUFFLED_TARGET_THRESHOLD = 0.10  # Model should have R2 < this on shuffled y


# =============================================================================
# INDIVIDUAL GATES
# =============================================================================


def run_shuffled_target_test(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_shuffles: int = 5,
    threshold: float = SHUFFLED_TARGET_THRESHOLD,
) -> GateResult:
    """
    Test that model fails on shuffled target.

    If the model performs well on randomly shuffled y, it indicates leakage.

    Parameters
    ----------
    model : sklearn-compatible model
        Model with fit() and score() methods
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    n_shuffles : int
        Number of shuffle iterations to average
    threshold : float
        Max acceptable R2 on shuffled target

    Returns
    -------
    GateResult
        Validation result
    """
    shuffled_scores = []

    for i in range(n_shuffles):
        # Shuffle target
        y_shuffled = y.sample(frac=1.0, random_state=i).reset_index(drop=True)

        # Fit and score on shuffled
        try:
            model_copy = (
                model.__class__(**model.get_params()) if hasattr(model, "get_params") else model
            )
            model_copy.fit(X, y_shuffled)
            score = model_copy.score(X, y_shuffled)
            shuffled_scores.append(score)
        except Exception as e:
            logger.warning(f"Shuffle iteration {i} failed: {e}")

    if not shuffled_scores:
        return GateResult(
            gate_name="Shuffled Target Test",
            status=GateStatus.WARN,
            message="Could not run shuffled target test",
        )

    avg_shuffled_score = np.mean(shuffled_scores)

    if avg_shuffled_score > threshold:
        return GateResult(
            gate_name="Shuffled Target Test",
            status=GateStatus.HALT,
            message=f"Model performs too well on shuffled target (R2={avg_shuffled_score:.3f})",
            metric_value=avg_shuffled_score,
            threshold=threshold,
            details={"shuffled_scores": shuffled_scores},
        )

    return GateResult(
        gate_name="Shuffled Target Test",
        status=GateStatus.PASS,
        message=f"Model appropriately fails on shuffled target (R2={avg_shuffled_score:.3f})",
        metric_value=avg_shuffled_score,
        threshold=threshold,
    )


def check_r_squared_threshold(
    r_squared: float,
    halt_threshold: float = R_SQUARED_HALT_THRESHOLD,
    warn_threshold: float = R_SQUARED_WARN_THRESHOLD,
) -> GateResult:
    """
    Check if R-squared is suspiciously high.

    In annuity price elasticity, R2 > 0.30 is unusual and suggests leakage.

    Parameters
    ----------
    r_squared : float
        Model R-squared value
    halt_threshold : float
        R2 above this triggers HALT
    warn_threshold : float
        R2 above this triggers WARN

    Returns
    -------
    GateResult
        Validation result
    """
    if r_squared > halt_threshold:
        return GateResult(
            gate_name="R-Squared Threshold",
            status=GateStatus.HALT,
            message=f"R2={r_squared:.3f} exceeds {halt_threshold} - investigate leakage",
            metric_value=r_squared,
            threshold=halt_threshold,
        )

    if r_squared > warn_threshold:
        return GateResult(
            gate_name="R-Squared Threshold",
            status=GateStatus.WARN,
            message=f"R2={r_squared:.3f} is higher than typical - worth checking",
            metric_value=r_squared,
            threshold=warn_threshold,
        )

    return GateResult(
        gate_name="R-Squared Threshold",
        status=GateStatus.PASS,
        message=f"R2={r_squared:.3f} is within expected range",
        metric_value=r_squared,
        threshold=halt_threshold,
    )


def check_improvement_threshold(
    baseline_metric: float,
    new_metric: float,
    metric_name: str = "R2",
    higher_is_better: bool = True,
    halt_threshold: float = IMPROVEMENT_HALT_THRESHOLD,
    warn_threshold: float = IMPROVEMENT_WARN_THRESHOLD,
) -> GateResult:
    """
    Check if improvement over baseline is suspiciously large.

    Large improvements (>20%) often indicate leakage rather than genuine improvement.

    Parameters
    ----------
    baseline_metric : float
        Baseline model performance
    new_metric : float
        New model performance
    metric_name : str
        Name of metric for reporting
    higher_is_better : bool
        Whether higher metric is better (True for R2, False for MAE)
    halt_threshold : float
        Improvement above this triggers HALT
    warn_threshold : float
        Improvement above this triggers WARN

    Returns
    -------
    GateResult
        Validation result
    """
    if baseline_metric == 0:
        return GateResult(
            gate_name="Improvement Threshold",
            status=GateStatus.WARN,
            message="Cannot calculate improvement from zero baseline",
        )

    if higher_is_better:
        improvement = (new_metric - baseline_metric) / abs(baseline_metric)
    else:
        improvement = (baseline_metric - new_metric) / abs(baseline_metric)

    if improvement > halt_threshold:
        return GateResult(
            gate_name="Improvement Threshold",
            status=GateStatus.HALT,
            message=f"{metric_name} improved {improvement:.1%} over baseline - investigate leakage",
            metric_value=improvement,
            threshold=halt_threshold,
            details={"baseline": baseline_metric, "new": new_metric},
        )

    if improvement > warn_threshold:
        return GateResult(
            gate_name="Improvement Threshold",
            status=GateStatus.WARN,
            message=f"{metric_name} improved {improvement:.1%} - worth checking",
            metric_value=improvement,
            threshold=warn_threshold,
            details={"baseline": baseline_metric, "new": new_metric},
        )

    return GateResult(
        gate_name="Improvement Threshold",
        status=GateStatus.PASS,
        message=f"{metric_name} improvement {improvement:.1%} is reasonable",
        metric_value=improvement,
        threshold=halt_threshold,
    )


def detect_lag0_features(
    feature_names: list[str],
    lag0_patterns: list[str] | None = None,
) -> GateResult:
    """
    Detect forbidden lag-0 competitor features.

    Lag-0 competitor features violate causal identification due to simultaneity.

    Parameters
    ----------
    feature_names : List[str]
        Names of features in the model
    lag0_patterns : Optional[List[str]]
        Regex patterns for lag-0 features

    Returns
    -------
    GateResult
        Validation result
    """
    if lag0_patterns is None:
        lag0_patterns = [
            r"C_lag_?0\b",
            r"competitor.*lag.*0",
            r"lag_0.*competitor",
            r"C_t\b(?!_)",
        ]

    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in lag0_patterns]
    detected = []

    for feature in feature_names:
        for pattern in compiled_patterns:
            if pattern.search(feature):
                detected.append(feature)
                break

    if detected:
        return GateResult(
            gate_name="Lag-0 Feature Detection",
            status=GateStatus.HALT,
            message=f"Lag-0 competitor features detected: {detected}",
            details={"detected_features": detected},
        )

    return GateResult(
        gate_name="Lag-0 Feature Detection",
        status=GateStatus.PASS,
        message="No lag-0 competitor features detected",
    )


def check_temporal_boundary(
    train_dates: pd.Series,
    test_dates: pd.Series,
) -> GateResult:
    """
    Verify no temporal leakage (training data doesn't include future dates).

    Parameters
    ----------
    train_dates : pd.Series
        Dates in training set
    test_dates : pd.Series
        Dates in test set

    Returns
    -------
    GateResult
        Validation result
    """
    max_train = train_dates.max()
    min_test = test_dates.min()

    if max_train >= min_test:
        return GateResult(
            gate_name="Temporal Boundary Check",
            status=GateStatus.HALT,
            message=f"Training data extends into test period (train max: {max_train}, test min: {min_test})",
            details={"max_train": str(max_train), "min_test": str(min_test)},
        )

    return GateResult(
        gate_name="Temporal Boundary Check",
        status=GateStatus.PASS,
        message=f"Proper temporal split (train ends: {max_train}, test starts: {min_test})",
    )


def check_coefficient_signs(
    coefficients: dict[str, float],
    product_type: str = "RILA",
    halt_on_violation: bool = True,
) -> GateResult:
    """
    Validate coefficient signs match economic constraints.

    Own-rate features should be POSITIVE (higher rates attract customers).
    Competitor features should be NEGATIVE (substitution effect).

    Parameters
    ----------
    coefficients : Dict[str, float]
        Feature name to coefficient value mapping
    product_type : str
        Product type for context (RILA, FIA, MYGA)
    halt_on_violation : bool
        If True, HALT on violations; if False, WARN only

    Returns
    -------
    GateResult
        Validation result with details on violations
    """
    if not coefficients:
        return GateResult(
            gate_name="Coefficient Sign Validation",
            status=GateStatus.WARN,
            message="No coefficients provided for sign validation",
        )

    results = validate_all_coefficients(coefficients, product_type)

    violated = results["violated"]
    warnings = results["warnings"]
    passed = results["passed"]

    if violated:
        violation_details = [
            f"{v['feature']}: {v['coefficient']:.4f} ({v['reason']})" for v in violated
        ]
        status = GateStatus.HALT if halt_on_violation else GateStatus.WARN
        return GateResult(
            gate_name="Coefficient Sign Validation",
            status=status,
            message=f"{len(violated)} coefficient sign violation(s): {', '.join(v['feature'] for v in violated)}",
            details={
                "violations": violated,
                "violation_details": violation_details,
                "warnings": warnings,
                "passed": len(passed),
            },
        )

    if warnings:
        return GateResult(
            gate_name="Coefficient Sign Validation",
            status=GateStatus.PASS,
            message=f"All constrained coefficients valid ({len(warnings)} context-dependent)",
            details={
                "warnings": warnings,
                "passed": len(passed),
            },
        )

    return GateResult(
        gate_name="Coefficient Sign Validation",
        status=GateStatus.PASS,
        message=f"All {len(passed)} constrained coefficients have correct signs",
        details={"passed": len(passed)},
    )


# =============================================================================
# COMBINED GATES
# =============================================================================


def run_all_gates(
    model: Any = None,
    X: pd.DataFrame = None,
    y: pd.Series = None,
    r_squared: float | None = None,
    baseline_r_squared: float | None = None,
    feature_names: list[str] | None = None,
    train_dates: pd.Series | None = None,
    test_dates: pd.Series | None = None,
    coefficients: dict[str, float] | None = None,
    product_type: str = "RILA",
    model_name: str = "",
    dataset_name: str = "",
) -> LeakageReport:
    """
    Run all leakage detection gates.

    Parameters
    ----------
    model : sklearn-compatible model
        Model to test (optional, needed for shuffled target test)
    X : pd.DataFrame
        Feature matrix (optional)
    y : pd.Series
        Target variable (optional)
    r_squared : float
        Current model R-squared (optional)
    baseline_r_squared : float
        Baseline model R-squared (optional)
    feature_names : List[str]
        Feature names (optional, defaults to X.columns)
    train_dates : pd.Series
        Training set dates (optional)
    test_dates : pd.Series
        Test set dates (optional)
    coefficients : Dict[str, float]
        Feature name to coefficient mapping (optional, needed for sign validation)
    product_type : str
        Product type for coefficient sign validation (RILA, FIA, MYGA)
    model_name : str
        Name for reporting
    dataset_name : str
        Dataset name for reporting

    Returns
    -------
    LeakageReport
        Complete validation report
    """
    from datetime import datetime

    report = LeakageReport(
        timestamp=datetime.now().isoformat(),
        model_name=model_name,
        dataset_name=dataset_name,
    )

    # Feature names from X if not provided
    if feature_names is None and X is not None:
        feature_names = list(X.columns)

    # Gate 1: Lag-0 detection
    if feature_names:
        report.gates.append(detect_lag0_features(feature_names))

    # Gate 2: R-squared threshold
    if r_squared is not None:
        report.gates.append(check_r_squared_threshold(r_squared))

    # Gate 3: Improvement threshold
    if r_squared is not None and baseline_r_squared is not None:
        report.gates.append(
            check_improvement_threshold(
                baseline_metric=baseline_r_squared,
                new_metric=r_squared,
            )
        )

    # Gate 4: Temporal boundary
    if train_dates is not None and test_dates is not None:
        report.gates.append(check_temporal_boundary(train_dates, test_dates))

    # Gate 5: Shuffled target test (expensive, run last among model tests)
    if model is not None and X is not None and y is not None:
        report.gates.append(run_shuffled_target_test(model, X, y))

    # Gate 6: Coefficient sign validation (economic constraints)
    if coefficients is not None:
        report.gates.append(check_coefficient_signs(coefficients, product_type))

    return report

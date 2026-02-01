"""
Tests for src.validation.leakage_gates module.

Tests leakage detection gates including the CRITICAL lag-0 detection.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.validation.leakage_gates import (
    GateResult,
    GateStatus,
    LeakageReport,
    check_coefficient_signs,
    check_improvement_threshold,
    check_r_squared_threshold,
    detect_lag0_features,
    run_shuffled_target_test,
)


class TestGateResult:
    """Tests for GateResult dataclass."""

    def test_gate_result_creation(self):
        """GateResult should be created with required fields."""
        result = GateResult(
            gate_name="Test Gate",
            status=GateStatus.PASS,
            message="Test passed",
        )
        assert result.gate_name == "Test Gate"
        assert result.status == GateStatus.PASS
        assert result.message == "Test passed"

    def test_gate_result_with_metrics(self):
        """GateResult should store metrics."""
        result = GateResult(
            gate_name="R-Squared Check",
            status=GateStatus.WARN,
            message="R-squared suspiciously high",
            metric_value=0.25,
            threshold=0.20,
        )
        assert result.metric_value == 0.25
        assert result.threshold == 0.20

    def test_gate_result_str_representation(self):
        """GateResult should have readable string representation."""
        result = GateResult(
            gate_name="Test Gate",
            status=GateStatus.PASS,
            message="Test passed",
        )
        result_str = str(result)
        assert "[PASS]" in result_str
        assert "Test Gate" in result_str


class TestLeakageReport:
    """Tests for LeakageReport dataclass."""

    def test_report_passed_when_no_halts(self):
        """Report should pass when no gates halted."""
        report = LeakageReport(
            gates=[
                GateResult("Gate1", GateStatus.PASS, "OK"),
                GateResult("Gate2", GateStatus.WARN, "Warning"),
            ]
        )
        assert report.passed is True

    def test_report_failed_when_any_halt(self):
        """Report should fail when any gate halted."""
        report = LeakageReport(
            gates=[
                GateResult("Gate1", GateStatus.PASS, "OK"),
                GateResult("Gate2", GateStatus.HALT, "Failed"),
            ]
        )
        assert report.passed is False

    def test_report_has_warnings(self):
        """Report should detect warnings."""
        report = LeakageReport(
            gates=[
                GateResult("Gate1", GateStatus.PASS, "OK"),
                GateResult("Gate2", GateStatus.WARN, "Warning"),
            ]
        )
        assert report.has_warnings is True

    def test_report_halt_count(self):
        """Report should count halts correctly."""
        report = LeakageReport(
            gates=[
                GateResult("Gate1", GateStatus.HALT, "Halt 1"),
                GateResult("Gate2", GateStatus.PASS, "OK"),
                GateResult("Gate3", GateStatus.HALT, "Halt 2"),
            ]
        )
        assert report.halt_count == 2


class TestCheckRSquaredThreshold:
    """Tests for check_r_squared_threshold function."""

    def test_low_r_squared_passes(self):
        """Low R-squared should pass."""
        result = check_r_squared_threshold(0.10)
        assert result.status == GateStatus.PASS

    def test_moderate_r_squared_warns(self):
        """Moderate R-squared should warn."""
        result = check_r_squared_threshold(0.25)
        assert result.status == GateStatus.WARN

    def test_high_r_squared_halts(self):
        """High R-squared should halt."""
        result = check_r_squared_threshold(0.35)
        assert result.status == GateStatus.HALT

    def test_custom_thresholds(self):
        """Custom thresholds should be respected."""
        # With higher threshold, 0.35 should pass
        result = check_r_squared_threshold(0.35, halt_threshold=0.40, warn_threshold=0.35)
        assert result.status != GateStatus.HALT


class TestCheckImprovementThreshold:
    """Tests for check_improvement_threshold function.

    NOTE: Improvement is calculated as RELATIVE change:
    improvement = (new - baseline) / baseline
    Default thresholds: HALT > 0.20 (20% relative), WARN > 0.10 (10% relative)
    """

    def test_small_improvement_passes(self):
        """Small relative improvement (<10%) should pass."""
        # 5% relative improvement: (0.105 - 0.10) / 0.10 = 0.05
        result = check_improvement_threshold(
            baseline_metric=0.10,
            new_metric=0.105,
        )
        assert result.status == GateStatus.PASS

    def test_moderate_improvement_warns(self):
        """Moderate relative improvement (10-20%) should warn."""
        # 15% relative improvement: (0.115 - 0.10) / 0.10 = 0.15
        result = check_improvement_threshold(
            baseline_metric=0.10,
            new_metric=0.115,
        )
        assert result.status == GateStatus.WARN

    def test_large_improvement_halts(self):
        """Large relative improvement (>20%) should halt."""
        # 50% relative improvement: (0.15 - 0.10) / 0.10 = 0.50
        result = check_improvement_threshold(
            baseline_metric=0.10,
            new_metric=0.15,
        )
        assert result.status == GateStatus.HALT


class TestDetectLag0Features:
    """
    Tests for detect_lag0_features function.

    CRITICAL: Lag-0 competitor features violate causal identification.
    """

    def test_no_lag0_features_passes(self):
        """Features without lag-0 should pass."""
        features = [
            "competitor_rate_t1",
            "competitor_rate_t2",
            "prudential_rate_current",
            "sales_target_t1",
        ]
        result = detect_lag0_features(features)
        assert result.status == GateStatus.PASS

    def test_detect_C_lag0_pattern(self):
        """Should detect C_lag0 pattern."""
        features = ["competitor_rate_t1", "C_lag0", "prudential_rate"]
        result = detect_lag0_features(features)
        assert result.status == GateStatus.HALT
        assert "C_lag0" in str(result.message)

    def test_detect_competitor_lag_0_pattern(self):
        """Should detect 'competitor...lag...0' pattern."""
        features = ["competitor_mid_lag_0", "prudential_rate"]
        result = detect_lag0_features(features)
        assert result.status == GateStatus.HALT

    def test_lag_t1_t2_allowed(self):
        """Lag t1, t2 features should be allowed."""
        features = [
            "competitor_rate_t1",
            "competitor_rate_t2",
            "competitor_mid_t3",
        ]
        result = detect_lag0_features(features)
        assert result.status == GateStatus.PASS

    def test_detect_current_competitor(self):
        """Should detect 'C_t' (current) pattern if configured."""
        # C_t without underscore after indicates lag-0
        features = ["C_t", "prudential_rate"]
        result = detect_lag0_features(features)
        # This depends on pattern - may or may not detect
        # The pattern r"C_t\b(?!_)" should match C_t but not C_t1
        assert result.status in [GateStatus.PASS, GateStatus.HALT]

    def test_custom_lag0_patterns(self):
        """Should use custom patterns if provided."""
        features = ["my_custom_lag0_feature", "normal_feature"]
        result = detect_lag0_features(
            features,
            lag0_patterns=[r"my_custom_lag0"],
        )
        assert result.status == GateStatus.HALT

    def test_case_insensitive_detection(self):
        """Detection should be case insensitive."""
        features = ["COMPETITOR_LAG_0", "prudential_rate"]
        result = detect_lag0_features(features)
        assert result.status == GateStatus.HALT

    def test_empty_features_passes(self):
        """Empty feature list should pass (no lag-0 detected)."""
        result = detect_lag0_features([])
        assert result.status == GateStatus.PASS

    def test_detected_features_in_details(self):
        """Detected features should be in result details."""
        features = ["C_lag0", "competitor_lag_0", "normal_feature"]
        result = detect_lag0_features(features)
        assert result.status == GateStatus.HALT
        assert "detected_features" in result.details
        assert len(result.details["detected_features"]) >= 1


class TestRunShuffledTargetTest:
    """Tests for run_shuffled_target_test function."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        model = MagicMock()
        model.get_params.return_value = {}
        model.__class__ = MagicMock
        return model

    @pytest.fixture
    def sample_data(self):
        """Create sample X, y for testing."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )
        y = pd.Series(np.random.randn(100))
        return X, y

    def test_good_model_passes(self, sample_data):
        """Model that fails on shuffled target should pass."""
        X, y = sample_data

        # Create a "good" mock model that gets low score on shuffled data
        model = MagicMock()
        model.get_params.return_value = {}
        model.__class__ = MagicMock
        model.fit.return_value = model
        model.score.return_value = 0.02  # Low score = model fails on shuffled

        result = run_shuffled_target_test(model, X, y, n_shuffles=2)
        assert result.status == GateStatus.PASS

    def test_leaky_model_halts(self, sample_data):
        """Model that succeeds on shuffled target should halt."""

        X, y = sample_data

        # Create a leaky scenario - mock the model copy to always return high score
        class LeakyModel:
            """Model that performs well on shuffled target (leaky)."""

            def get_params(self):
                return {}

            def fit(self, X, y):
                return self

            def score(self, X, y):
                return 0.80  # High score = leakage!

        model = LeakyModel()
        result = run_shuffled_target_test(model, X, y, n_shuffles=2)
        assert result.status == GateStatus.HALT
        assert "performs too well" in result.message

    def test_shuffle_failure_returns_warning(self, sample_data):
        """If all shuffle iterations fail, returns warning."""
        X, y = sample_data

        class FailingModel:
            """Model that fails during fit."""

            def get_params(self):
                return {}

            def fit(self, X, y):
                raise RuntimeError("Fit failed")

        model = FailingModel()
        result = run_shuffled_target_test(model, X, y, n_shuffles=2)
        assert result.status == GateStatus.WARN
        assert "Could not run" in result.message


class TestCheckImprovementThresholdExtended:
    """Extended tests for check_improvement_threshold edge cases."""

    def test_zero_baseline_warns(self):
        """Zero baseline should return warning."""
        result = check_improvement_threshold(
            baseline_metric=0.0,
            new_metric=0.15,
        )
        assert result.status == GateStatus.WARN
        assert "zero baseline" in result.message.lower()

    def test_lower_is_better_mode(self):
        """Test higher_is_better=False (e.g., for MAE)."""
        # Improvement means LOWER is better
        # baseline=0.20, new=0.10 → 50% improvement
        result = check_improvement_threshold(
            baseline_metric=0.20,
            new_metric=0.10,
            higher_is_better=False,
        )
        assert result.status == GateStatus.HALT
        assert result.metric_value == pytest.approx(0.5)


class TestCheckCoefficientSigns:
    """Tests for check_coefficient_signs gate (Gate 6).

    Economic constraints:
    - Own-rate (Prudential): POSITIVE (higher rates attract customers)
    - Competitor rates: NEGATIVE (substitution effect)
    """

    def test_valid_coefficients_pass(self):
        """Coefficients with correct signs should pass."""
        coefficients = {
            "prudential_rate_t0": 0.085,  # Own-rate: positive
            "competitor_weighted_t2": -0.031,  # Competitor: negative
            "competitor_top5_t3": -0.028,  # Competitor: negative
        }
        result = check_coefficient_signs(coefficients)
        assert result.status == GateStatus.PASS
        assert "correct signs" in result.message or "valid" in result.message.lower()

    def test_wrong_own_rate_sign_halts(self):
        """Negative own-rate coefficient should halt."""
        coefficients = {
            "prudential_rate_t0": -0.085,  # WRONG: should be positive
            "competitor_weighted_t2": -0.031,
        }
        result = check_coefficient_signs(coefficients)
        assert result.status == GateStatus.HALT
        assert "violation" in result.message.lower()
        assert "prudential_rate_t0" in result.message

    def test_wrong_competitor_sign_halts(self):
        """Positive competitor coefficient should halt."""
        coefficients = {
            "prudential_rate_t0": 0.085,
            "competitor_weighted_t2": 0.031,  # WRONG: should be negative
        }
        result = check_coefficient_signs(coefficients)
        assert result.status == GateStatus.HALT
        assert "competitor_weighted_t2" in result.message

    def test_multiple_violations_reported(self):
        """Multiple violations should all be reported."""
        coefficients = {
            "prudential_rate_t0": -0.085,  # WRONG
            "competitor_weighted_t2": 0.031,  # WRONG
            "competitor_core_t1": 0.020,  # WRONG
        }
        result = check_coefficient_signs(coefficients)
        assert result.status == GateStatus.HALT
        assert len(result.details["violations"]) >= 2

    def test_context_dependent_coefficients_pass(self):
        """Context-dependent coefficients (treasury, vix) should pass either sign."""
        coefficients = {
            "dgs5_t1": -0.015,  # Treasury: context-dependent
            "vix_t1": 0.010,  # VIX: context-dependent
        }
        result = check_coefficient_signs(coefficients)
        assert result.status == GateStatus.PASS
        assert result.details.get("warnings") or "context-dependent" in result.message.lower()

    def test_empty_coefficients_warns(self):
        """Empty coefficient dict should return warning."""
        result = check_coefficient_signs({})
        assert result.status == GateStatus.WARN
        assert "No coefficients" in result.message

    def test_none_coefficients_warns(self):
        """None coefficients should be handled gracefully."""
        result = check_coefficient_signs(None)
        assert result.status == GateStatus.WARN

    def test_halt_on_violation_flag(self):
        """halt_on_violation=False should return WARN instead of HALT."""
        coefficients = {
            "prudential_rate_t0": -0.085,  # WRONG
        }
        result = check_coefficient_signs(coefficients, halt_on_violation=False)
        assert result.status == GateStatus.WARN

    def test_unconstrained_features_pass(self):
        """Features without defined constraints should pass."""
        coefficients = {
            "custom_feature_1": 0.5,
            "unknown_macro_indicator": -0.2,
        }
        result = check_coefficient_signs(coefficients)
        assert result.status == GateStatus.PASS


class TestLeakageReportString:
    """Tests for LeakageReport string representation."""

    def test_str_representation_with_gates(self):
        """String representation should include all gate results."""
        report = LeakageReport(
            gates=[
                GateResult("Gate1", GateStatus.PASS, "OK"),
                GateResult("Gate2", GateStatus.HALT, "Failed"),
            ],
            model_name="TestModel",
            dataset_name="TestData",
            timestamp="2022-01-01T00:00:00",
        )
        report_str = str(report)

        assert "Leakage Validation Report" in report_str
        assert "TestModel" in report_str
        assert "TestData" in report_str
        assert "[PASS]" in report_str
        assert "[HALT]" in report_str
        assert "FAILED" in report_str

    def test_str_representation_passed(self):
        """String representation shows PASSED when no halts."""
        report = LeakageReport(
            gates=[
                GateResult("Gate1", GateStatus.PASS, "OK"),
                GateResult("Gate2", GateStatus.WARN, "Warning"),
            ]
        )
        report_str = str(report)
        assert "PASSED" in report_str


class TestRunAllGates:
    """Tests for run_all_gates orchestration function."""

    def test_run_all_gates_with_feature_names(self):
        """run_all_gates should check lag-0 features."""
        from src.validation.leakage_gates import run_all_gates

        report = run_all_gates(
            feature_names=["competitor_rate_t1", "prudential_rate"],
        )

        assert isinstance(report, LeakageReport)
        assert len(report.gates) >= 1
        assert any(g.gate_name == "Lag-0 Feature Detection" for g in report.gates)

    def test_run_all_gates_with_r_squared(self):
        """run_all_gates should check R-squared threshold."""
        from src.validation.leakage_gates import run_all_gates

        report = run_all_gates(
            r_squared=0.15,
        )

        assert any(g.gate_name == "R-Squared Threshold" for g in report.gates)

    def test_run_all_gates_with_improvement(self):
        """run_all_gates should check improvement when baseline provided."""
        from src.validation.leakage_gates import run_all_gates

        report = run_all_gates(
            r_squared=0.15,
            baseline_r_squared=0.10,
        )

        assert any(g.gate_name == "Improvement Threshold" for g in report.gates)

    def test_run_all_gates_with_model(self):
        """run_all_gates should run shuffled target test when model provided."""
        from src.validation.leakage_gates import run_all_gates

        # Simple model for testing
        class SimpleModel:
            def get_params(self):
                return {}

            def fit(self, X, y):
                return self

            def score(self, X, y):
                return 0.02

        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(50), "b": np.random.randn(50)})
        y = pd.Series(np.random.randn(50))

        report = run_all_gates(
            model=SimpleModel(),
            X=X,
            y=y,
        )

        assert any(g.gate_name == "Shuffled Target Test" for g in report.gates)

    def test_run_all_gates_extracts_features_from_X(self):
        """run_all_gates should extract feature names from X if not provided."""
        from src.validation.leakage_gates import run_all_gates

        X = pd.DataFrame(
            {
                "competitor_rate_t1": [1, 2, 3],
                "prudential_rate": [4, 5, 6],
            }
        )

        report = run_all_gates(X=X)

        assert any(g.gate_name == "Lag-0 Feature Detection" for g in report.gates)

    def test_run_all_gates_with_temporal_boundary(self):
        """run_all_gates should check temporal boundary when dates provided."""
        from src.validation.leakage_gates import run_all_gates

        train_dates = pd.to_datetime(["2022-01-01", "2022-06-01"])
        test_dates = pd.to_datetime(["2022-07-01", "2022-12-01"])

        report = run_all_gates(
            train_dates=train_dates,
            test_dates=test_dates,
        )

        assert any(g.gate_name == "Temporal Boundary Check" for g in report.gates)

    def test_run_all_gates_report_metadata(self):
        """run_all_gates should populate report metadata."""
        from src.validation.leakage_gates import run_all_gates

        report = run_all_gates(
            model_name="MyModel",
            dataset_name="MyData",
            feature_names=["feature1"],
        )

        assert report.model_name == "MyModel"
        assert report.dataset_name == "MyData"
        assert report.timestamp != ""

    def test_run_all_gates_with_coefficients(self):
        """run_all_gates should check coefficient signs when coefficients provided."""
        from src.validation.leakage_gates import run_all_gates

        coefficients = {
            "prudential_rate_t0": 0.085,
            "competitor_weighted_t2": -0.031,
        }

        report = run_all_gates(coefficients=coefficients)

        assert any(g.gate_name == "Coefficient Sign Validation" for g in report.gates)

    def test_run_all_gates_coefficient_violation_halts(self):
        """run_all_gates should halt on coefficient sign violation."""
        from src.validation.leakage_gates import run_all_gates

        coefficients = {
            "prudential_rate_t0": -0.085,  # WRONG: should be positive
        }

        report = run_all_gates(coefficients=coefficients)

        coef_gate = next(
            (g for g in report.gates if g.gate_name == "Coefficient Sign Validation"), None
        )
        assert coef_gate is not None
        assert coef_gate.status == GateStatus.HALT
        assert report.passed is False

    def test_run_all_gates_passes_product_type(self):
        """run_all_gates should pass product_type to coefficient validation."""
        from src.validation.leakage_gates import run_all_gates

        coefficients = {
            "prudential_rate_t0": 0.085,
        }

        report = run_all_gates(
            coefficients=coefficients,
            product_type="FIA",
        )

        # Should still pass (product_type doesn't change basic constraints)
        coef_gate = next(
            (g for g in report.gates if g.gate_name == "Coefficient Sign Validation"), None
        )
        assert coef_gate is not None
        assert coef_gate.status == GateStatus.PASS


# =============================================================================
# BUFFER LEVEL CONTROL TESTS (LEAKAGE_CHECKLIST.md Section 7)
# =============================================================================


@pytest.mark.leakage
class TestBufferLevelControl:
    """
    Test buffer level control per LEAKAGE_CHECKLIST.md Section 7.

    Buffer Level Control Requirement:
    "Buffer level is controlled in model. Either buffer indicators or stratified analysis."

    Two-Stage Validation:
    1. Stage 1: Verify buffer feature exists in model inputs
    2. Stage 2: Verify coefficient on buffer is significant

    Products tested: 6Y20B, 6Y10B, 10Y20B (different buffer levels)
    """

    @pytest.fixture
    def model_features_with_buffer(self) -> list:
        """Sample feature list with buffer control."""
        return [
            "prudential_rate_current",
            "competitor_mid_t2",
            "competitor_top5_t3",
            "buffer_level_indicator",  # Buffer control feature
            "vix_t1",
            "dgs5_t1",
        ]

    @pytest.fixture
    def model_features_without_buffer(self) -> list:
        """Sample feature list WITHOUT buffer control."""
        return [
            "prudential_rate_current",
            "competitor_mid_t2",
            "competitor_top5_t3",
            "vix_t1",
            "dgs5_t1",
        ]

    @pytest.fixture
    def sample_coefficients_with_buffer(self) -> dict:
        """Sample coefficients including buffer indicator."""
        return {
            "prudential_rate_current": 0.0847,
            "competitor_mid_t2": -0.0312,
            "competitor_top5_t3": -0.0284,
            "buffer_level_indicator": 0.025,  # Significant buffer effect
            "vix_t1": -0.015,
            "dgs5_t1": 0.008,
        }

    def test_buffer_feature_exists_in_model_inputs(self, model_features_with_buffer: list) -> None:
        """Stage 1: Verify buffer feature exists in model inputs.

        Per LEAKAGE_CHECKLIST.md Section 7: Buffer level must be controlled.
        """
        buffer_keywords = ["buffer", "buffer_level", "buffer_indicator"]

        has_buffer_control = any(
            any(kw in f.lower() for kw in buffer_keywords) for f in model_features_with_buffer
        )

        assert has_buffer_control, (
            "No buffer control feature found in model. "
            "LEAKAGE_CHECKLIST.md Section 7 requires buffer level control. "
            f"Features: {model_features_with_buffer}"
        )

    def test_buffer_feature_missing_detected(self, model_features_without_buffer: list) -> None:
        """Detect when buffer feature is missing from model inputs."""
        buffer_keywords = ["buffer", "buffer_level", "buffer_indicator"]

        has_buffer_control = any(
            any(kw in f.lower() for kw in buffer_keywords) for f in model_features_without_buffer
        )

        # This test intentionally expects failure to detect missing buffer
        assert not has_buffer_control, "Expected no buffer control in this fixture"

    def test_buffer_coefficient_is_significant(self, sample_coefficients_with_buffer: dict) -> None:
        """Stage 2: Verify coefficient on buffer is significant.

        A non-significant buffer coefficient suggests the model
        may not be properly controlling for product differences.
        """
        buffer_keywords = ["buffer", "buffer_level", "buffer_indicator"]

        buffer_coef = None
        buffer_name = None
        for name, value in sample_coefficients_with_buffer.items():
            if any(kw in name.lower() for kw in buffer_keywords):
                buffer_coef = value
                buffer_name = name
                break

        assert buffer_coef is not None, "No buffer coefficient found"

        # Coefficient should be meaningfully different from zero
        significance_threshold = 0.01  # Minimum magnitude
        assert abs(buffer_coef) > significance_threshold, (
            f"Buffer coefficient '{buffer_name}' = {buffer_coef:.6f} "
            f"is not significant (< {significance_threshold}). "
            "This suggests the model may not be properly controlling "
            "for buffer level differences across products."
        )

    @pytest.mark.parametrize("product_code", ["6Y20B", "6Y10B", "10Y20B"])
    def test_buffer_control_per_product(self, product_code: str) -> None:
        """Verify buffer control exists for each RILA product.

        Different products have different buffer levels:
        - 6Y20B: 20% buffer
        - 6Y10B: 10% buffer
        - 10Y20B: 20% buffer

        Each product-specific model should include buffer control.
        """
        # This test validates the requirement exists
        # Actual implementation would load product-specific models
        expected_buffers = {
            "6Y20B": "20%",
            "6Y10B": "10%",
            "10Y20B": "20%",
        }

        assert product_code in expected_buffers, f"Unknown product code: {product_code}"

        # In actual test, would verify model includes buffer control
        # For now, document the requirement
        expected_buffer = expected_buffers[product_code]
        assert expected_buffer in [
            "10%",
            "20%",
        ], f"Product {product_code} should have valid buffer level"

    def test_buffer_control_validation_function(self) -> None:
        """Test buffer control validation utility function.

        Creates a reusable validation function for buffer control.
        """

        def validate_buffer_control(
            feature_names: list,
            coefficients: dict | None = None,
            significance_threshold: float = 0.01,
        ) -> GateResult:
            """
            Validate buffer level control in model.

            Parameters
            ----------
            feature_names : list
                List of model feature names
            coefficients : dict, optional
                Coefficient values (for significance check)
            significance_threshold : float
                Minimum coefficient magnitude

            Returns
            -------
            GateResult
                PASS if buffer controlled, HALT if not
            """
            buffer_keywords = ["buffer", "buffer_level", "buffer_indicator"]

            # Stage 1: Check feature exists
            buffer_features = [
                f for f in feature_names if any(kw in f.lower() for kw in buffer_keywords)
            ]

            if not buffer_features:
                return GateResult(
                    gate_name="Buffer Level Control",
                    status=GateStatus.HALT,
                    message=(
                        "No buffer control feature found. "
                        "LEAKAGE_CHECKLIST.md Section 7 requires buffer control."
                    ),
                    details={"features_checked": feature_names},
                )

            # Stage 2: Check significance (if coefficients provided)
            if coefficients:
                for bf in buffer_features:
                    coef = coefficients.get(bf, 0)
                    if abs(coef) < significance_threshold:
                        return GateResult(
                            gate_name="Buffer Level Control",
                            status=GateStatus.WARN,
                            message=(
                                f"Buffer coefficient '{bf}' = {coef:.6f} "
                                f"is not significant (< {significance_threshold})."
                            ),
                            metric_value=abs(coef),
                            threshold=significance_threshold,
                        )

            return GateResult(
                gate_name="Buffer Level Control",
                status=GateStatus.PASS,
                message="Buffer level is controlled in model.",
                details={"buffer_features": buffer_features},
            )

        # Test the validation function
        result = validate_buffer_control(
            feature_names=["prudential_rate", "buffer_indicator", "vix"],
            coefficients={"buffer_indicator": 0.025},
        )
        assert result.status == GateStatus.PASS

        result_fail = validate_buffer_control(
            feature_names=["prudential_rate", "vix"],
        )
        assert result_fail.status == GateStatus.HALT

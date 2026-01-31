"""
Tests for Statistical Constraint Types Module.

Tests cover:
- ConstraintType enum values and behavior
- StatisticalConstraintResult dataclass creation and field validation
- ComprehensiveConstraintAnalysis dataclass creation and structure

Design Principles:
- Test every dataclass field
- Verify enum member access patterns
- Test edge cases with empty/default values

Author: Claude Code
Date: 2026-01-31
"""

import pytest
from typing import Dict, Any, List

from src.features.selection.enhancements.statistical_constraints.constraint_types import (
    ConstraintType,
    StatisticalConstraintResult,
    ComprehensiveConstraintAnalysis,
)


# =============================================================================
# Tests for ConstraintType Enum
# =============================================================================


class TestConstraintTypeEnum:
    """Tests for ConstraintType enumeration."""

    def test_positive_value(self):
        """Test POSITIVE enum member exists and has correct value."""
        assert ConstraintType.POSITIVE.value == "positive"

    def test_negative_value(self):
        """Test NEGATIVE enum member exists and has correct value."""
        assert ConstraintType.NEGATIVE.value == "negative"

    def test_magnitude_value(self):
        """Test MAGNITUDE enum member exists and has correct value."""
        assert ConstraintType.MAGNITUDE.value == "magnitude"

    def test_ratio_value(self):
        """Test RATIO enum member exists and has correct value."""
        assert ConstraintType.RATIO.value == "ratio"

    def test_enum_iteration(self):
        """Test that enum can be iterated over."""
        members = list(ConstraintType)
        assert len(members) == 4
        assert ConstraintType.POSITIVE in members
        assert ConstraintType.NEGATIVE in members
        assert ConstraintType.MAGNITUDE in members
        assert ConstraintType.RATIO in members

    def test_enum_comparison(self):
        """Test enum member comparison."""
        assert ConstraintType.POSITIVE == ConstraintType.POSITIVE
        assert ConstraintType.POSITIVE != ConstraintType.NEGATIVE

    def test_enum_from_value(self):
        """Test constructing enum from string value."""
        assert ConstraintType("positive") == ConstraintType.POSITIVE
        assert ConstraintType("negative") == ConstraintType.NEGATIVE
        assert ConstraintType("magnitude") == ConstraintType.MAGNITUDE
        assert ConstraintType("ratio") == ConstraintType.RATIO

    def test_invalid_value_raises(self):
        """Test that invalid value raises ValueError."""
        with pytest.raises(ValueError):
            ConstraintType("invalid")


# =============================================================================
# Tests for StatisticalConstraintResult Dataclass
# =============================================================================


class TestStatisticalConstraintResult:
    """Tests for StatisticalConstraintResult dataclass."""

    @pytest.fixture
    def sample_result(self) -> StatisticalConstraintResult:
        """Create a sample constraint result for testing."""
        return StatisticalConstraintResult(
            feature_name="own_rate_t1",
            constraint_type=ConstraintType.POSITIVE,
            coefficient_estimate=0.05,
            standard_error=0.012,
            confidence_interval=(0.026, 0.074),
            t_statistic=4.17,
            p_value=0.0001,
            statistically_significant=True,
            constraint_satisfied=True,
            constraint_strength="STRONG",
            business_interpretation="Higher own rates attract more customers",
            hard_threshold_comparison={"pass_hard_threshold": True, "margin": 0.02}
        )

    def test_creation_all_fields(self, sample_result):
        """Test that all fields are correctly assigned."""
        assert sample_result.feature_name == "own_rate_t1"
        assert sample_result.constraint_type == ConstraintType.POSITIVE
        assert sample_result.coefficient_estimate == 0.05
        assert sample_result.standard_error == 0.012
        assert sample_result.confidence_interval == (0.026, 0.074)
        assert sample_result.t_statistic == 4.17
        assert sample_result.p_value == 0.0001
        assert sample_result.statistically_significant is True
        assert sample_result.constraint_satisfied is True
        assert sample_result.constraint_strength == "STRONG"
        assert sample_result.business_interpretation == "Higher own rates attract more customers"
        assert sample_result.hard_threshold_comparison["pass_hard_threshold"] is True

    def test_constraint_type_negative(self):
        """Test result with NEGATIVE constraint type."""
        result = StatisticalConstraintResult(
            feature_name="competitor_weighted_rate_t1",
            constraint_type=ConstraintType.NEGATIVE,
            coefficient_estimate=-0.03,
            standard_error=0.01,
            confidence_interval=(-0.05, -0.01),
            t_statistic=-3.0,
            p_value=0.003,
            statistically_significant=True,
            constraint_satisfied=True,
            constraint_strength="MODERATE",
            business_interpretation="Higher competitor rates reduce own sales",
            hard_threshold_comparison={}
        )

        assert result.constraint_type == ConstraintType.NEGATIVE
        assert result.coefficient_estimate < 0
        assert result.constraint_satisfied is True

    def test_violated_constraint(self):
        """Test result with violated constraint."""
        result = StatisticalConstraintResult(
            feature_name="own_rate_t1",
            constraint_type=ConstraintType.POSITIVE,
            coefficient_estimate=-0.01,  # Wrong sign!
            standard_error=0.02,
            confidence_interval=(-0.05, 0.03),
            t_statistic=-0.5,
            p_value=0.62,
            statistically_significant=False,
            constraint_satisfied=False,
            constraint_strength="VIOLATED",
            business_interpretation="Own rate has unexpected negative effect",
            hard_threshold_comparison={"pass_hard_threshold": False}
        )

        assert result.constraint_satisfied is False
        assert result.constraint_strength == "VIOLATED"
        assert result.statistically_significant is False

    def test_weak_constraint(self):
        """Test result with weak constraint satisfaction."""
        result = StatisticalConstraintResult(
            feature_name="own_rate_t1",
            constraint_type=ConstraintType.POSITIVE,
            coefficient_estimate=0.005,  # Small positive
            standard_error=0.003,
            confidence_interval=(0.001, 0.011),
            t_statistic=1.67,
            p_value=0.10,
            statistically_significant=False,
            constraint_satisfied=True,
            constraint_strength="WEAK",
            business_interpretation="Own rate effect is positive but weak",
            hard_threshold_comparison={"pass_hard_threshold": False, "margin": -0.005}
        )

        assert result.constraint_strength == "WEAK"
        assert result.constraint_satisfied is True

    def test_empty_hard_threshold_comparison(self):
        """Test result with empty hard threshold comparison dict."""
        result = StatisticalConstraintResult(
            feature_name="test_feature",
            constraint_type=ConstraintType.MAGNITUDE,
            coefficient_estimate=0.1,
            standard_error=0.01,
            confidence_interval=(0.08, 0.12),
            t_statistic=10.0,
            p_value=0.0001,
            statistically_significant=True,
            constraint_satisfied=True,
            constraint_strength="STRONG",
            business_interpretation="Test",
            hard_threshold_comparison={}  # Empty dict
        )

        assert result.hard_threshold_comparison == {}

    def test_confidence_interval_tuple(self, sample_result):
        """Test that confidence interval is accessible as tuple."""
        lower, upper = sample_result.confidence_interval
        assert lower < sample_result.coefficient_estimate < upper


# =============================================================================
# Tests for ComprehensiveConstraintAnalysis Dataclass
# =============================================================================


class TestComprehensiveConstraintAnalysis:
    """Tests for ComprehensiveConstraintAnalysis dataclass."""

    @pytest.fixture
    def sample_constraint_result(self) -> StatisticalConstraintResult:
        """Create a sample constraint result."""
        return StatisticalConstraintResult(
            feature_name="own_rate_t1",
            constraint_type=ConstraintType.POSITIVE,
            coefficient_estimate=0.05,
            standard_error=0.012,
            confidence_interval=(0.026, 0.074),
            t_statistic=4.17,
            p_value=0.0001,
            statistically_significant=True,
            constraint_satisfied=True,
            constraint_strength="STRONG",
            business_interpretation="Test",
            hard_threshold_comparison={}
        )

    @pytest.fixture
    def sample_analysis(self, sample_constraint_result) -> ComprehensiveConstraintAnalysis:
        """Create a sample comprehensive analysis."""
        return ComprehensiveConstraintAnalysis(
            model_specification={
                "product_type": "6Y20B",
                "sample_size": 156,
                "date_range": ("2023-01-01", "2025-12-31")
            },
            constraint_results=[sample_constraint_result],
            overall_assessment={
                "total_constraints": 1,
                "satisfied_count": 1,
                "violated_count": 0,
                "overall_pass": True
            },
            methodology_comparison={
                "hard_threshold_satisfied": 1,
                "statistical_satisfied": 1,
                "agreement_rate": 1.0
            },
            business_recommendations={
                "primary": "Model passes all economic constraints",
                "risk_assessment": "Low risk"
            },
            power_analysis={
                "statistical_power": 0.85,
                "minimum_detectable_effect": 0.02
            }
        )

    def test_creation(self, sample_analysis):
        """Test that all fields are correctly assigned."""
        assert sample_analysis.model_specification["product_type"] == "6Y20B"
        assert len(sample_analysis.constraint_results) == 1
        assert sample_analysis.overall_assessment["overall_pass"] is True
        assert sample_analysis.methodology_comparison["agreement_rate"] == 1.0
        assert "primary" in sample_analysis.business_recommendations
        assert sample_analysis.power_analysis["statistical_power"] == 0.85

    def test_with_empty_lists(self):
        """Test creation with empty constraint results list."""
        analysis = ComprehensiveConstraintAnalysis(
            model_specification={"product_type": "6Y20B"},
            constraint_results=[],  # Empty list
            overall_assessment={"total_constraints": 0, "overall_pass": True},
            methodology_comparison={},
            business_recommendations={},
            power_analysis={}
        )

        assert len(analysis.constraint_results) == 0
        assert analysis.overall_assessment["total_constraints"] == 0

    def test_with_multiple_constraint_results(self, sample_constraint_result):
        """Test creation with multiple constraint results."""
        result2 = StatisticalConstraintResult(
            feature_name="competitor_weighted_rate_t1",
            constraint_type=ConstraintType.NEGATIVE,
            coefficient_estimate=-0.03,
            standard_error=0.01,
            confidence_interval=(-0.05, -0.01),
            t_statistic=-3.0,
            p_value=0.003,
            statistically_significant=True,
            constraint_satisfied=True,
            constraint_strength="MODERATE",
            business_interpretation="Test",
            hard_threshold_comparison={}
        )

        analysis = ComprehensiveConstraintAnalysis(
            model_specification={"product_type": "6Y20B"},
            constraint_results=[sample_constraint_result, result2],
            overall_assessment={
                "total_constraints": 2,
                "satisfied_count": 2,
                "violated_count": 0
            },
            methodology_comparison={},
            business_recommendations={},
            power_analysis={}
        )

        assert len(analysis.constraint_results) == 2
        assert analysis.overall_assessment["satisfied_count"] == 2

    def test_access_nested_model_specification(self, sample_analysis):
        """Test accessing nested model specification data."""
        assert sample_analysis.model_specification["sample_size"] == 156
        date_range = sample_analysis.model_specification["date_range"]
        assert date_range[0] == "2023-01-01"
        assert date_range[1] == "2025-12-31"

    def test_iterate_constraint_results(self, sample_analysis):
        """Test iterating over constraint results."""
        for result in sample_analysis.constraint_results:
            assert isinstance(result, StatisticalConstraintResult)
            assert result.feature_name == "own_rate_t1"


# =============================================================================
# Module Export Tests
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_exist(self):
        """Test that all exported symbols are accessible."""
        from src.features.selection.enhancements.statistical_constraints import constraint_types

        assert hasattr(constraint_types, "ConstraintType")
        assert hasattr(constraint_types, "StatisticalConstraintResult")
        assert hasattr(constraint_types, "ComprehensiveConstraintAnalysis")

    def test_all_list_complete(self):
        """Test that __all__ contains expected exports."""
        from src.features.selection.enhancements.statistical_constraints.constraint_types import __all__

        assert "ConstraintType" in __all__
        assert "StatisticalConstraintResult" in __all__
        assert "ComprehensiveConstraintAnalysis" in __all__

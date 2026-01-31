"""
Tests for Constraint Analyzers Module.

Tests cover:
- Interpretation: _interpret_positive_constraint, _interpret_negative_constraint
- Aggregation: _count_constraint_outcomes, _categorize_features_by_strength
- Compliance: _determine_compliance_level, _assess_overall_constraint_compliance
- Methodology: _compute_methodology_summary, _identify_method_advantages,
               _build_disagreement_analysis, _compare_constraint_methodologies
- Recommendations: _build_feature_recommendations, _generate_constraint_recommendations
- Power Analysis: _compute_average_precision, _estimate_power_metrics,
                  _interpret_power, _analyze_constraint_detection_power

Design Principles:
- Pure function tests (no external dependencies)
- Edge case tests for empty lists and boundary conditions
- Property-based tests for mathematical invariants

Author: Claude Code
Date: 2026-01-30
"""

import pytest
import numpy as np
from typing import Dict, List, Any

from src.features.selection.enhancements.statistical_constraints.constraint_analyzers import (
    # Interpretation
    _interpret_positive_constraint,
    _interpret_negative_constraint,
    # Aggregation
    _count_constraint_outcomes,
    _categorize_features_by_strength,
    # Compliance
    _determine_compliance_level,
    _assess_overall_constraint_compliance,
    # Methodology
    _compute_methodology_summary,
    _identify_method_advantages,
    _build_disagreement_analysis,
    _compare_constraint_methodologies,
    # Recommendations
    _build_feature_recommendations,
    _generate_constraint_recommendations,
    # Power Analysis
    _compute_average_precision,
    _estimate_power_metrics,
    _interpret_power,
    _analyze_constraint_detection_power,
)
from src.features.selection.enhancements.statistical_constraints.constraint_types import (
    ConstraintType,
    StatisticalConstraintResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def create_constraint_result(
    feature_name: str,
    coefficient: float,
    standard_error: float,
    ci_lower: float,
    ci_upper: float,
    p_value: float = 0.01,
    constraint_strength: str = "STRONG",
    constraint_satisfied: bool = True,
    hard_threshold_passes: bool = True,
    statistical_approach_passes: bool = True,
    constraint_type: ConstraintType = ConstraintType.POSITIVE,
) -> StatisticalConstraintResult:
    """Helper to create StatisticalConstraintResult for testing."""
    return StatisticalConstraintResult(
        feature_name=feature_name,
        constraint_type=constraint_type,
        coefficient_estimate=coefficient,
        standard_error=standard_error,
        confidence_interval=(ci_lower, ci_upper),
        t_statistic=coefficient / standard_error if standard_error > 0 else 0,
        p_value=p_value,
        statistically_significant=p_value < 0.05,
        constraint_satisfied=constraint_satisfied,
        constraint_strength=constraint_strength,
        business_interpretation=f"Test interpretation for {feature_name}",
        hard_threshold_comparison={
            'hard_threshold_passes': hard_threshold_passes,
            'statistical_approach_passes': statistical_approach_passes,
            'methods_agree': hard_threshold_passes == statistical_approach_passes,
        }
    )


@pytest.fixture
def strong_positive_result() -> StatisticalConstraintResult:
    """Strongly positive constraint result."""
    return create_constraint_result(
        feature_name='positive_feature',
        coefficient=0.5,
        standard_error=0.1,
        ci_lower=0.3,
        ci_upper=0.7,
        constraint_strength="STRONG",
        constraint_satisfied=True,
    )


@pytest.fixture
def violated_negative_result() -> StatisticalConstraintResult:
    """Violated negative constraint result (positive coef expected negative)."""
    return create_constraint_result(
        feature_name='violated_feature',
        coefficient=0.2,
        standard_error=0.1,
        ci_lower=0.05,
        ci_upper=0.35,
        constraint_strength="VIOLATED",
        constraint_satisfied=False,
        constraint_type=ConstraintType.NEGATIVE,
    )


@pytest.fixture
def weak_result() -> StatisticalConstraintResult:
    """Weak constraint result (CI crosses zero)."""
    return create_constraint_result(
        feature_name='weak_feature',
        coefficient=0.1,
        standard_error=0.15,
        ci_lower=-0.2,
        ci_upper=0.4,
        p_value=0.15,
        constraint_strength="WEAK",
        constraint_satisfied=False,
    )


@pytest.fixture
def mixed_results(
    strong_positive_result, violated_negative_result, weak_result
) -> List[StatisticalConstraintResult]:
    """Mixed set of constraint results for testing."""
    moderate_result = create_constraint_result(
        feature_name='moderate_feature',
        coefficient=0.3,
        standard_error=0.12,
        ci_lower=0.06,
        ci_upper=0.54,
        constraint_strength="MODERATE",
        constraint_satisfied=True,
    )
    return [strong_positive_result, violated_negative_result, weak_result, moderate_result]


# =============================================================================
# Tests for _interpret_positive_constraint
# =============================================================================


class TestInterpretPositiveConstraint:
    """Tests for positive constraint interpretation."""

    def test_strong_evidence_ci_positive(self):
        """Strong evidence when CI is entirely positive."""
        result = _interpret_positive_constraint(
            coefficient=0.5,
            ci_lower=0.2,
            ci_upper=0.8,
            business_rationale="Higher rate improves conversion",
            statistically_significant=True
        )

        assert "STRONG evidence" in result
        assert "positive effect" in result
        assert "0.5000" in result
        assert "statistically supported" in result

    def test_moderate_evidence_significant_but_ci_crosses(self):
        """Moderate evidence when significant but CI crosses zero."""
        result = _interpret_positive_constraint(
            coefficient=0.3,
            ci_lower=-0.05,
            ci_upper=0.65,
            business_rationale="Price increases sales",
            statistically_significant=True
        )

        assert "MODERATE evidence" in result
        assert "some support" in result

    def test_weak_evidence_not_significant(self):
        """Weak evidence when not statistically significant."""
        result = _interpret_positive_constraint(
            coefficient=0.1,
            ci_lower=-0.2,
            ci_upper=0.4,
            business_rationale="Feature improves outcome",
            statistically_significant=False
        )

        assert "WEAK evidence" in result
        assert "not statistically significant" in result

    def test_violated_negative_coefficient(self):
        """Violated when coefficient is negative."""
        result = _interpret_positive_constraint(
            coefficient=-0.3,
            ci_lower=-0.5,
            ci_upper=-0.1,
            business_rationale="Feature should increase",
            statistically_significant=True
        )

        assert "VIOLATED" in result
        assert "Negative effect detected" in result
        assert "Contradicts expectation" in result


# =============================================================================
# Tests for _interpret_negative_constraint
# =============================================================================


class TestInterpretNegativeConstraint:
    """Tests for negative constraint interpretation."""

    def test_strong_evidence_ci_negative(self):
        """Strong evidence when CI is entirely negative."""
        result = _interpret_negative_constraint(
            coefficient=-0.5,
            ci_lower=-0.8,
            ci_upper=-0.2,
            business_rationale="Competition reduces share",
            statistically_significant=True
        )

        assert "STRONG evidence" in result
        assert "negative effect" in result
        assert "-0.5000" in result

    def test_moderate_evidence_significant_but_ci_crosses(self):
        """Moderate evidence when significant but CI crosses zero."""
        result = _interpret_negative_constraint(
            coefficient=-0.2,
            ci_lower=-0.5,
            ci_upper=0.1,
            business_rationale="Factor reduces outcome",
            statistically_significant=True
        )

        assert "MODERATE evidence" in result

    def test_weak_evidence_not_significant(self):
        """Weak evidence when not statistically significant."""
        result = _interpret_negative_constraint(
            coefficient=-0.1,
            ci_lower=-0.4,
            ci_upper=0.2,
            business_rationale="Variable decreases sales",
            statistically_significant=False
        )

        assert "WEAK evidence" in result
        assert "not statistically significant" in result

    def test_violated_positive_coefficient(self):
        """Violated when coefficient is positive (expected negative)."""
        result = _interpret_negative_constraint(
            coefficient=0.3,
            ci_lower=0.1,
            ci_upper=0.5,
            business_rationale="Variable should decrease",
            statistically_significant=True
        )

        assert "VIOLATED" in result
        assert "Positive effect detected" in result


# =============================================================================
# Tests for _count_constraint_outcomes
# =============================================================================


class TestCountConstraintOutcomes:
    """Tests for constraint outcome counting."""

    def test_empty_list(self):
        """Empty list returns zero counts."""
        result = _count_constraint_outcomes([])

        assert result['total'] == 0
        assert result['strong'] == 0
        assert result['violated'] == 0

    def test_mixed_results(self, mixed_results):
        """Mixed results are correctly counted."""
        result = _count_constraint_outcomes(mixed_results)

        assert result['total'] == 4
        assert result['strong'] == 1
        assert result['moderate'] == 1
        assert result['weak'] == 1
        assert result['violated'] == 1

    def test_all_strong(self, strong_positive_result):
        """All strong results correctly counted."""
        results = [strong_positive_result] * 5
        counts = _count_constraint_outcomes(results)

        assert counts['total'] == 5
        assert counts['strong'] == 5
        assert counts['moderate'] == 0
        assert counts['satisfied'] == 5

    def test_satisfied_vs_significant(self):
        """Distinguishes between satisfied and significant."""
        # Satisfied but not significant
        result = create_constraint_result(
            feature_name='test',
            coefficient=0.5,
            standard_error=0.1,
            ci_lower=0.3,
            ci_upper=0.7,
            p_value=0.10,  # Not significant
            constraint_strength="MODERATE",
            constraint_satisfied=True,
        )
        counts = _count_constraint_outcomes([result])

        assert counts['satisfied'] == 1
        assert counts['significant'] == 0


# =============================================================================
# Tests for _categorize_features_by_strength
# =============================================================================


class TestCategorizeFeaturesbyStrength:
    """Tests for feature categorization by strength."""

    def test_empty_list(self):
        """Empty list returns empty categories."""
        result = _categorize_features_by_strength([])

        assert result['violated'] == []
        assert result['weak'] == []
        assert result['strong'] == []

    def test_mixed_results(self, mixed_results):
        """Mixed results are correctly categorized."""
        result = _categorize_features_by_strength(mixed_results)

        assert 'positive_feature' in result['strong']
        assert 'violated_feature' in result['violated']
        assert 'weak_feature' in result['weak']

    def test_moderate_not_included(self, mixed_results):
        """Moderate results are not included in any category."""
        result = _categorize_features_by_strength(mixed_results)

        # Only 3 categories: strong, weak, violated
        assert 'moderate_feature' not in result['strong']
        assert 'moderate_feature' not in result['weak']
        assert 'moderate_feature' not in result['violated']


# =============================================================================
# Tests for _determine_compliance_level
# =============================================================================


class TestDetermineComplianceLevel:
    """Tests for compliance level determination."""

    def test_poor_compliance_when_violated(self):
        """Poor compliance when any constraints violated."""
        counts = {'total': 5, 'strong': 3, 'moderate': 1, 'weak': 0, 'violated': 1}
        compliance, confidence, ready, rec = _determine_compliance_level(counts)

        assert compliance == "POOR"
        assert confidence == "LOW"
        assert ready is False
        assert "violated" in rec

    def test_excellent_compliance_mostly_strong(self):
        """Excellent compliance when 70%+ strong."""
        counts = {'total': 10, 'strong': 8, 'moderate': 2, 'weak': 0, 'violated': 0}
        compliance, confidence, ready, rec = _determine_compliance_level(counts)

        assert compliance == "EXCELLENT"
        assert confidence == "HIGH"
        assert ready is True

    def test_good_compliance_moderate_strong_mix(self):
        """Good compliance when 60%+ strong+moderate."""
        counts = {'total': 10, 'strong': 4, 'moderate': 3, 'weak': 3, 'violated': 0}
        compliance, confidence, ready, rec = _determine_compliance_level(counts)

        assert compliance == "GOOD"
        assert confidence == "MODERATE"
        assert ready is True

    def test_acceptable_compliance_mixed(self):
        """Acceptable compliance when 50%+ not violated."""
        counts = {'total': 10, 'strong': 2, 'moderate': 2, 'weak': 6, 'violated': 0}
        compliance, confidence, ready, rec = _determine_compliance_level(counts)

        assert compliance == "ACCEPTABLE"
        assert ready is True

    def test_concerning_compliance_insufficient_evidence(self):
        """Concerning compliance when insufficient evidence."""
        counts = {'total': 10, 'strong': 1, 'moderate': 1, 'weak': 2, 'violated': 0}
        # strong + moderate + weak = 4 < 5 (50%)
        compliance, confidence, ready, rec = _determine_compliance_level(counts)

        assert compliance == "CONCERNING"
        assert ready is False


# =============================================================================
# Tests for _assess_overall_constraint_compliance
# =============================================================================


class TestAssessOverallConstraintCompliance:
    """Tests for overall compliance assessment."""

    def test_empty_list_returns_no_constraints(self):
        """Empty list returns no_constraints_tested flag."""
        result = _assess_overall_constraint_compliance([])

        assert result.get('no_constraints_tested') is True

    def test_complete_assessment_structure(self, mixed_results):
        """Complete assessment has all required fields."""
        result = _assess_overall_constraint_compliance(mixed_results)

        assert 'overall_compliance' in result
        assert 'confidence_level' in result
        assert 'production_readiness' in result
        assert 'primary_recommendation' in result
        assert 'compliance_statistics' in result

    def test_compliance_statistics_structure(self, mixed_results):
        """Compliance statistics has correct structure."""
        result = _assess_overall_constraint_compliance(mixed_results)
        stats = result['compliance_statistics']

        assert stats['total_constraints'] == 4
        assert 'satisfied_constraints' in stats
        assert 'compliance_rate' in stats
        assert 'significance_rate' in stats
        assert 'strength_distribution' in stats


# =============================================================================
# Tests for _compute_methodology_summary
# =============================================================================


class TestComputeMethodologySummary:
    """Tests for methodology comparison summary."""

    def test_all_agree(self, strong_positive_result):
        """When all methods agree, agreement rate is 100%."""
        results = [strong_positive_result] * 3
        summary = _compute_methodology_summary(results)

        assert summary['agreement_rate'] == 1.0
        assert summary['total_disagreements'] == 0

    def test_disagreement_tracked(self):
        """Disagreements are correctly tracked."""
        agreeing = create_constraint_result(
            feature_name='agree',
            coefficient=0.5,
            standard_error=0.1,
            ci_lower=0.3,
            ci_upper=0.7,
            hard_threshold_passes=True,
            statistical_approach_passes=True,
        )
        disagreeing = create_constraint_result(
            feature_name='disagree',
            coefficient=0.1,
            standard_error=0.2,
            ci_lower=-0.3,
            ci_upper=0.5,
            hard_threshold_passes=True,
            statistical_approach_passes=False,
        )
        summary = _compute_methodology_summary([agreeing, disagreeing])

        assert summary['agreement_rate'] == 0.5
        assert summary['total_disagreements'] == 1


# =============================================================================
# Tests for _identify_method_advantages
# =============================================================================


class TestIdentifyMethodAdvantages:
    """Tests for method advantage identification."""

    def test_statistical_rejects_nonsignificant(self):
        """Statistical method advantage when rejecting non-significant."""
        result = create_constraint_result(
            feature_name='borderline',
            coefficient=0.05,
            standard_error=0.1,
            ci_lower=-0.15,
            ci_upper=0.25,
            p_value=0.60,
            hard_threshold_passes=True,
            statistical_approach_passes=False,
        )
        stat_adv, hard_adv = _identify_method_advantages([result])

        assert len(stat_adv) == 1
        assert "borderline" in stat_adv[0]
        assert "Rejected non-significant" in stat_adv[0]

    def test_no_advantages_when_agreeing(self, strong_positive_result):
        """No advantages when methods agree."""
        stat_adv, hard_adv = _identify_method_advantages([strong_positive_result])

        assert stat_adv == []
        assert hard_adv == []


# =============================================================================
# Tests for _build_disagreement_analysis
# =============================================================================


class TestBuildDisagreementAnalysis:
    """Tests for disagreement analysis building."""

    def test_empty_when_all_agree(self, strong_positive_result):
        """Empty analysis when all methods agree."""
        result = _build_disagreement_analysis([strong_positive_result] * 3)

        assert result == []

    def test_disagreement_captured(self):
        """Disagreements are captured with reasons."""
        disagreeing = create_constraint_result(
            feature_name='uncertain',
            coefficient=0.05,
            standard_error=0.1,
            ci_lower=-0.15,
            ci_upper=0.25,
            p_value=0.60,
            constraint_satisfied=False,
            hard_threshold_passes=True,
            statistical_approach_passes=False,
        )
        analysis = _build_disagreement_analysis([disagreeing])

        assert len(analysis) == 1
        assert analysis[0]['feature'] == 'uncertain'
        assert analysis[0]['hard_threshold_result'] is True
        assert analysis[0]['statistical_result'] is False
        assert 'reason_for_disagreement' in analysis[0]


# =============================================================================
# Tests for _compare_constraint_methodologies
# =============================================================================


class TestCompareConstraintMethodologies:
    """Tests for methodology comparison."""

    def test_empty_list_returns_no_results(self):
        """Empty list returns no_results flag."""
        result = _compare_constraint_methodologies([], {})

        assert result.get('no_results_for_comparison') is True

    def test_complete_comparison_structure(self, mixed_results):
        """Complete comparison has all sections."""
        result = _compare_constraint_methodologies(mixed_results, {})

        assert 'methodology_comparison_summary' in result
        assert 'statistical_method_advantages' in result
        assert 'hard_threshold_advantages' in result
        assert 'disagreement_analysis' in result
        assert 'methodology_recommendation' in result


# =============================================================================
# Tests for _build_feature_recommendations
# =============================================================================


class TestBuildFeatureRecommendations:
    """Tests for feature recommendation building."""

    def test_empty_categories(self):
        """Empty categories produce empty recommendations."""
        categories = {'violated': [], 'weak': [], 'strong': []}
        result = _build_feature_recommendations(categories)

        assert result == {}

    def test_violation_recommendation(self):
        """Violation produces investigation recommendation."""
        categories = {'violated': ['bad_feature'], 'weak': [], 'strong': []}
        result = _build_feature_recommendations(categories)

        assert 'violations' in result
        assert 'bad_feature' in result['violations']
        assert 'Investigate' in result['violations']

    def test_weak_recommendation(self):
        """Weak features produce monitoring recommendation."""
        categories = {'violated': [], 'weak': ['uncertain_feature'], 'strong': []}
        result = _build_feature_recommendations(categories)

        assert 'weak_relationships' in result
        assert 'uncertain_feature' in result['weak_relationships']
        assert 'Monitor' in result['weak_relationships']

    def test_strong_recommendation(self):
        """Strong features produce leverage recommendation."""
        categories = {'violated': [], 'weak': [], 'strong': ['good_feature']}
        result = _build_feature_recommendations(categories)

        assert 'strong_relationships' in result
        assert 'good_feature' in result['strong_relationships']
        assert 'Leverage' in result['strong_relationships']


# =============================================================================
# Tests for _generate_constraint_recommendations
# =============================================================================


class TestGenerateConstraintRecommendations:
    """Tests for comprehensive recommendation generation."""

    def test_includes_primary_recommendation(self, mixed_results):
        """Recommendations include primary assessment."""
        assessment = {'primary_recommendation': 'Test primary', 'production_readiness': True}
        result = _generate_constraint_recommendations(mixed_results, assessment)

        assert result['primary'] == 'Test primary'

    def test_production_ready_message(self, strong_positive_result):
        """Production ready assessment produces appropriate message."""
        assessment = {'primary_recommendation': 'Good', 'production_readiness': True}
        result = _generate_constraint_recommendations([strong_positive_result], assessment)

        assert "suitable for production" in result['production']

    def test_not_production_ready_message(self, violated_negative_result):
        """Not production ready produces resolution message."""
        assessment = {'primary_recommendation': 'Bad', 'production_readiness': False}
        result = _generate_constraint_recommendations([violated_negative_result], assessment)

        assert "requires constraint violation resolution" in result['production']

    def test_methodology_recommendation_included(self, mixed_results):
        """Methodology recommendation is always included."""
        assessment = {'primary_recommendation': 'Test', 'production_readiness': True}
        result = _generate_constraint_recommendations(mixed_results, assessment)

        assert 'methodology' in result
        assert 'confidence interval' in result['methodology']


# =============================================================================
# Tests for _compute_average_precision
# =============================================================================


class TestComputeAveragePrecision:
    """Tests for average precision computation."""

    def test_precision_inverse_of_se(self, strong_positive_result):
        """Precision is computed as inverse of standard error."""
        result = _compute_average_precision([strong_positive_result])

        # strong_positive_result has se=0.1, so precision = 1/0.1 = 10
        assert result == 10.0

    def test_zero_se_handled(self):
        """Zero standard error produces zero precision (not infinity)."""
        result = create_constraint_result(
            feature_name='zero_se',
            coefficient=0.5,
            standard_error=0.0,
            ci_lower=0.5,
            ci_upper=0.5,
        )
        precision = _compute_average_precision([result])

        assert precision == 0.0

    def test_average_multiple(self):
        """Average precision computed across multiple results."""
        result1 = create_constraint_result(
            feature_name='test1',
            coefficient=0.5,
            standard_error=0.1,  # precision = 10
            ci_lower=0.3,
            ci_upper=0.7,
        )
        result2 = create_constraint_result(
            feature_name='test2',
            coefficient=0.5,
            standard_error=0.2,  # precision = 5
            ci_lower=0.1,
            ci_upper=0.9,
        )
        avg_precision = _compute_average_precision([result1, result2])

        assert avg_precision == 7.5  # (10 + 5) / 2


# =============================================================================
# Tests for _estimate_power_metrics
# =============================================================================


class TestEstimatePowerMetrics:
    """Tests for power metrics estimation."""

    def test_returns_three_values(self):
        """Returns t_critical, min_detectable, and estimated_power."""
        t_crit, min_det, power = _estimate_power_metrics(100, 5)

        assert isinstance(t_crit, float)
        assert isinstance(min_det, float)
        assert isinstance(power, float)

    def test_min_detectable_decreases_with_n(self):
        """Minimum detectable effect decreases with sample size."""
        _, min_det_small, _ = _estimate_power_metrics(50, 5)
        _, min_det_large, _ = _estimate_power_metrics(500, 5)

        assert min_det_large < min_det_small

    def test_power_increases_with_n(self):
        """Power increases with sample size."""
        _, _, power_small = _estimate_power_metrics(50, 5)
        _, _, power_large = _estimate_power_metrics(500, 5)

        assert power_large >= power_small

    def test_power_bounded_zero_one(self):
        """Power is bounded between 0 and 1."""
        _, _, power = _estimate_power_metrics(100, 5)

        assert 0 <= power <= 1

    def test_handles_zero_observations(self):
        """Handles edge case of zero observations."""
        _, min_det, _ = _estimate_power_metrics(0, 5)

        assert min_det == 0


# =============================================================================
# Tests for _interpret_power
# =============================================================================


class TestInterpretPower:
    """Tests for power interpretation."""

    def test_high_power(self):
        """High power (>0.8) interpretation."""
        interpretation, recommendation = _interpret_power(0.85)

        assert interpretation == "HIGH"
        assert "Adequate power" in recommendation

    def test_moderate_power(self):
        """Moderate power (0.6-0.8) interpretation."""
        interpretation, recommendation = _interpret_power(0.7)

        assert interpretation == "MODERATE"

    def test_low_power(self):
        """Low power (<0.6) interpretation."""
        interpretation, recommendation = _interpret_power(0.4)

        assert interpretation == "LOW"
        assert "increasing sample size" in recommendation

    def test_boundary_values(self):
        """Test boundary values for power categories."""
        # Exactly 0.8 - should be MODERATE (not >0.8)
        interp_08, _ = _interpret_power(0.8)
        assert interp_08 == "MODERATE"

        # Just above 0.8
        interp_081, _ = _interpret_power(0.81)
        assert interp_081 == "HIGH"

        # Exactly 0.6 - should be LOW (not >0.6)
        interp_06, _ = _interpret_power(0.6)
        assert interp_06 == "LOW"


# =============================================================================
# Tests for _analyze_constraint_detection_power
# =============================================================================


class TestAnalyzeConstraintDetectionPower:
    """Tests for comprehensive power analysis."""

    def test_empty_list_returns_empty_dict(self):
        """Empty constraint list returns empty dict."""
        result = _analyze_constraint_detection_power([], 100)

        assert result == {}

    def test_complete_analysis_structure(self, mixed_results):
        """Complete analysis has all required fields."""
        result = _analyze_constraint_detection_power(mixed_results, 200)

        assert 'average_precision' in result
        assert 'minimum_detectable_effect' in result
        assert 'estimated_power_medium_effect' in result
        assert 'sample_size' in result
        assert 'power_interpretation' in result
        assert 'recommendation' in result

    def test_sample_size_included(self, strong_positive_result):
        """Sample size is included in output."""
        result = _analyze_constraint_detection_power([strong_positive_result], 150)

        assert result['sample_size'] == 150

    def test_power_interpretation_correct(self, mixed_results):
        """Power interpretation is one of the valid categories."""
        result = _analyze_constraint_detection_power(mixed_results, 100)

        valid_interpretations = {"HIGH", "MODERATE", "LOW"}
        assert result['power_interpretation'] in valid_interpretations

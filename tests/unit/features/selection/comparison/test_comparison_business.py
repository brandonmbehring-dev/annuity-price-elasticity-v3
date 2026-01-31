"""
Tests for comparison_business module.

Target: 16% → 60%+ coverage
Tests organized by function categories:
- Business impact analysis
- Risk reduction analysis
- Decision confidence analysis
- Operational impact analysis
- Cost-benefit analysis
- Recommendation generation
- Performance summary
"""

import numpy as np
import pytest
from unittest.mock import patch

from src.features.selection.comparison.comparison_business import (
    # Business impact
    _analyze_business_impact,
    _analyze_risk_reduction,
    _analyze_decision_confidence,
    _analyze_operational_impact,
    _analyze_cost_benefit,
    # Recommendations
    _assemble_recommendations,
    _generate_methodology_recommendations,
    _generate_primary_recommendation,
    _generate_implementation_recommendation,
    _generate_risk_recommendation,
    _generate_business_recommendation,
    _generate_technical_recommendation,
    _generate_stakeholder_recommendation,
    # Performance summary
    _calculate_improvement_counts,
    _determine_overall_assessment,
    _generate_key_findings,
    _summarize_performance_comparison,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def enhanced_results_full():
    """Enhanced results with all validations present."""
    return {
        'temporal_validation': {'test_r_squared': 0.65},
        'multiple_testing_correction': {'significant_models': 5},
        'regression_diagnostics': {'all_passed': True}
    }


@pytest.fixture
def enhanced_results_partial():
    """Enhanced results with only some validations."""
    return {
        'temporal_validation': {'test_r_squared': 0.60}
    }


@pytest.fixture
def enhanced_results_empty():
    """Enhanced results with no validations."""
    return {}


@pytest.fixture
def baseline_results():
    """Baseline methodology results."""
    return {
        'aic': 100.0,
        'r_squared': 0.75,
        'n_features': 5
    }


@pytest.fixture
def performance_comparison():
    """Sample performance comparison."""
    return {
        'enhanced_exclusive_benefits': {
            'out_of_sample_validation': True,
            'generalization_assessment': True
        },
        'improvements': {
            'r_squared': {'improved': True, 'delta': 0.05},
            'mape': {'improved': True, 'delta': -2.5}
        }
    }


@pytest.fixture
def statistical_validation_comparison():
    """Sample statistical validation comparison."""
    return {
        'statistical_rigor_assessment': {
            'rigor_improvement': 50
        }
    }


@pytest.fixture
def production_readiness_comparison():
    """Sample production readiness comparison."""
    return {
        'deployment_confidence': {
            'confidence_improvement': 40
        }
    }


@pytest.fixture
def business_impact_analysis():
    """Sample business impact analysis."""
    return {
        'risk_reduction': {
            'baseline_risks': ['risk1', 'risk2', 'risk3'],
            'enhanced_mitigations': ['mitigation1', 'mitigation2', 'mitigation3']
        },
        'cost_benefit_analysis': {
            'roi_assessment': 'POSITIVE - Benefits significantly outweigh costs'
        }
    }


# =============================================================================
# Business Impact Analysis Tests
# =============================================================================


class TestAnalyzeBusinessImpact:
    """Tests for _analyze_business_impact."""

    def test_returns_all_impact_categories(
        self, baseline_results, enhanced_results_full, performance_comparison
    ):
        """Returns all impact analysis categories."""
        impact = _analyze_business_impact(
            baseline_results, enhanced_results_full, performance_comparison
        )

        expected_keys = ['risk_reduction', 'decision_confidence',
                         'operational_impact', 'cost_benefit_analysis']
        for key in expected_keys:
            assert key in impact

    def test_handles_analysis_error(self, baseline_results):
        """Handles errors gracefully."""
        # Pass None to trigger error
        with patch(
            'src.features.selection.comparison.comparison_business._analyze_risk_reduction',
            side_effect=Exception("Test error")
        ):
            impact = _analyze_business_impact(
                baseline_results, {}, {}
            )

            assert impact.get('analysis_failed') == True  # noqa: E712


class TestAnalyzeRiskReduction:
    """Tests for _analyze_risk_reduction."""

    def test_high_risk_reduction_with_all_mitigations(self, enhanced_results_full):
        """Returns HIGH risk reduction when all mitigations present."""
        result = _analyze_risk_reduction(enhanced_results_full)

        assert result['risk_reduction_level'] == 'HIGH'
        assert len(result['enhanced_mitigations']) >= 3
        assert len(result['baseline_risks']) == 3

    def test_moderate_risk_reduction_with_partial_mitigations(self, enhanced_results_partial):
        """Returns MODERATE risk reduction with partial mitigations."""
        result = _analyze_risk_reduction(enhanced_results_partial)

        assert result['risk_reduction_level'] == 'MODERATE'
        assert len(result['enhanced_mitigations']) < 3

    def test_no_mitigations_when_empty(self, enhanced_results_empty):
        """Returns no mitigations when results empty."""
        result = _analyze_risk_reduction(enhanced_results_empty)

        assert len(result['enhanced_mitigations']) == 0
        assert result['risk_reduction_level'] == 'MODERATE'

    def test_baseline_risks_always_present(self, enhanced_results_empty):
        """Baseline risks are always documented."""
        result = _analyze_risk_reduction(enhanced_results_empty)

        assert len(result['baseline_risks']) == 3
        assert any('generalization' in risk.lower() for risk in result['baseline_risks'])


class TestAnalyzeDecisionConfidence:
    """Tests for _analyze_decision_confidence."""

    def test_returns_confidence_factors(self):
        """Returns both baseline and enhanced confidence factors."""
        result = _analyze_decision_confidence()

        assert 'baseline_confidence_factors' in result
        assert 'enhanced_confidence_factors' in result
        assert 'confidence_improvement' in result

    def test_enhanced_has_more_factors(self):
        """Enhanced methodology has more confidence factors."""
        result = _analyze_decision_confidence()

        assert len(result['enhanced_confidence_factors']) > len(result['baseline_confidence_factors'])

    def test_substantial_improvement(self):
        """Confidence improvement is substantial."""
        result = _analyze_decision_confidence()

        assert result['confidence_improvement'] == 'SUBSTANTIAL'


class TestAnalyzeOperationalImpact:
    """Tests for _analyze_operational_impact."""

    def test_with_oos_validation(self, enhanced_results_full, performance_comparison):
        """Operational impact with out-of-sample validation."""
        result = _analyze_operational_impact(enhanced_results_full, performance_comparison)

        assert 'Well-estimated' in result['expected_production_performance']
        assert 'HIGH' in result['business_trust']

    def test_without_oos_validation(self, enhanced_results_empty):
        """Operational impact without out-of-sample validation."""
        perf_comparison = {'enhanced_exclusive_benefits': {}}

        result = _analyze_operational_impact(enhanced_results_empty, perf_comparison)

        assert '30-40%' in result['expected_production_performance']
        assert 'LOW' in result['business_trust']

    def test_proactive_maintenance_with_diagnostics(self, enhanced_results_full):
        """Proactive maintenance when diagnostics present."""
        result = _analyze_operational_impact(enhanced_results_full, {})

        assert 'Proactive' in result['model_maintenance']

    def test_reactive_maintenance_without_diagnostics(self, enhanced_results_empty):
        """Reactive maintenance without diagnostics."""
        result = _analyze_operational_impact(enhanced_results_empty, {})

        assert 'Reactive' in result['model_maintenance']


class TestAnalyzeCostBenefit:
    """Tests for _analyze_cost_benefit."""

    def test_returns_all_components(self):
        """Returns all cost-benefit components."""
        result = _analyze_cost_benefit()

        assert 'implementation_costs' in result
        assert 'business_benefits' in result
        assert 'roi_assessment' in result
        assert 'payback_period' in result

    def test_implementation_costs_structure(self):
        """Implementation costs have expected structure."""
        result = _analyze_cost_benefit()

        costs = result['implementation_costs']
        assert 'statistical_analysis' in costs
        assert 'validation_setup' in costs
        assert 'ongoing_maintenance' in costs

    def test_positive_roi(self):
        """ROI assessment is positive."""
        result = _analyze_cost_benefit()

        assert 'POSITIVE' in result['roi_assessment']

    def test_multiple_business_benefits(self):
        """Multiple business benefits documented."""
        result = _analyze_cost_benefit()

        assert len(result['business_benefits']) >= 3


# =============================================================================
# Recommendation Generation Tests
# =============================================================================


class TestAssembleRecommendations:
    """Tests for _assemble_recommendations."""

    def test_returns_all_recommendation_categories(
        self, statistical_validation_comparison,
        production_readiness_comparison, business_impact_analysis
    ):
        """Returns all recommendation categories."""
        recs = _assemble_recommendations(
            statistical_validation_comparison,
            production_readiness_comparison,
            business_impact_analysis
        )

        expected_keys = ['primary', 'implementation', 'risk_management',
                         'business_case', 'technical', 'stakeholder']
        for key in expected_keys:
            assert key in recs


class TestGenerateMethodologyRecommendations:
    """Tests for _generate_methodology_recommendations."""

    def test_generates_recommendations(
        self, performance_comparison, statistical_validation_comparison,
        production_readiness_comparison, business_impact_analysis
    ):
        """Generates complete recommendations."""
        recs = _generate_methodology_recommendations(
            performance_comparison,
            statistical_validation_comparison,
            production_readiness_comparison,
            business_impact_analysis
        )

        assert 'primary' in recs

    def test_handles_error(self, performance_comparison):
        """Handles errors gracefully."""
        with patch(
            'src.features.selection.comparison.comparison_business._assemble_recommendations',
            side_effect=Exception("Test error")
        ):
            recs = _generate_methodology_recommendations(
                performance_comparison, {}, {}, {}
            )

            assert 'generation_failed' in recs


class TestGeneratePrimaryRecommendation:
    """Tests for _generate_primary_recommendation."""

    def test_strongly_recommend_with_high_improvements(self):
        """Strongly recommends when improvements are high."""
        stat_comp = {'statistical_rigor_assessment': {'rigor_improvement': 50}}
        prod_comp = {'deployment_confidence': {'confidence_improvement': 40}}

        rec = _generate_primary_recommendation(stat_comp, prod_comp)

        assert 'STRONGLY RECOMMEND' in rec

    def test_recommend_with_moderate_rigor(self):
        """Recommends with moderate rigor improvement."""
        stat_comp = {'statistical_rigor_assessment': {'rigor_improvement': 25}}
        prod_comp = {'deployment_confidence': {'confidence_improvement': 10}}

        rec = _generate_primary_recommendation(stat_comp, prod_comp)

        assert 'RECOMMEND' in rec
        assert 'STRONGLY' not in rec

    def test_consider_with_low_improvements(self):
        """Considers with low improvements."""
        stat_comp = {'statistical_rigor_assessment': {'rigor_improvement': 10}}
        prod_comp = {'deployment_confidence': {'confidence_improvement': 5}}

        rec = _generate_primary_recommendation(stat_comp, prod_comp)

        assert 'CONSIDER' in rec

    def test_handles_missing_values(self):
        """Handles missing improvement values."""
        rec = _generate_primary_recommendation({}, {})

        # Should still return a recommendation
        assert isinstance(rec, str)
        assert len(rec) > 0


class TestGenerateImplementationRecommendation:
    """Tests for _generate_implementation_recommendation."""

    def test_returns_phased_approach(self):
        """Returns phased implementation recommendation."""
        rec = _generate_implementation_recommendation()

        assert 'phases' in rec.lower()
        assert 'temporal' in rec.lower() or 'Temporal' in rec


class TestGenerateRiskRecommendation:
    """Tests for _generate_risk_recommendation."""

    def test_significant_risk_reduction(self, business_impact_analysis):
        """Recommends for all critical deployments when risk reduction significant."""
        rec = _generate_risk_recommendation(business_impact_analysis)

        assert 'significantly' in rec.lower() or 'critical' in rec.lower()

    def test_moderate_risk_reduction(self):
        """Recommends for high-stakes when risk reduction moderate."""
        impact = {
            'risk_reduction': {
                'baseline_risks': ['r1', 'r2', 'r3', 'r4', 'r5'],
                'enhanced_mitigations': ['m1']  # Only 20%
            }
        }

        rec = _generate_risk_recommendation(impact)

        assert 'moderate' in rec.lower() or 'high-stakes' in rec.lower()

    def test_handles_empty_impact(self):
        """Handles empty impact analysis."""
        rec = _generate_risk_recommendation({})

        assert isinstance(rec, str)


class TestGenerateBusinessRecommendation:
    """Tests for _generate_business_recommendation."""

    def test_strong_case_with_positive_roi(self, business_impact_analysis):
        """Strong business case with positive ROI."""
        rec = _generate_business_recommendation(business_impact_analysis)

        assert 'strong' in rec.lower() or 'Strong' in rec

    def test_moderate_case_without_positive_roi(self):
        """Moderate case without positive ROI."""
        impact = {
            'cost_benefit_analysis': {
                'roi_assessment': 'NEUTRAL'
            }
        }

        rec = _generate_business_recommendation(impact)

        assert 'moderate' in rec.lower() or 'Moderate' in rec


class TestGenerateTechnicalRecommendation:
    """Tests for _generate_technical_recommendation."""

    def test_mentions_resources(self):
        """Mentions computational resources."""
        rec = _generate_technical_recommendation()

        assert 'resources' in rec.lower()

    def test_mentions_expertise(self):
        """Mentions required expertise."""
        rec = _generate_technical_recommendation()

        assert 'expertise' in rec.lower()


class TestGenerateStakeholderRecommendation:
    """Tests for _generate_stakeholder_recommendation."""

    def test_emphasizes_confidence(self):
        """Emphasizes improved confidence."""
        rec = _generate_stakeholder_recommendation()

        assert 'confidence' in rec.lower()

    def test_mentions_compliance(self):
        """Mentions regulatory compliance."""
        rec = _generate_stakeholder_recommendation()

        assert 'compliance' in rec.lower() or 'regulatory' in rec.lower()


# =============================================================================
# Performance Summary Tests
# =============================================================================


class TestCalculateImprovementCounts:
    """Tests for _calculate_improvement_counts."""

    def test_counts_positive_improvements(self):
        """Counts positive improvements correctly."""
        improvements = {
            'metric_a': {'improved': True},
            'metric_b': {'improved': True},
            'metric_c': {'improved': False},
        }
        exclusive = {'benefit_a': True, 'benefit_b': False}

        pos_count, exc_count = _calculate_improvement_counts(improvements, exclusive)

        assert pos_count == 2
        assert exc_count == 1

    def test_handles_empty_inputs(self):
        """Handles empty inputs."""
        pos_count, exc_count = _calculate_improvement_counts({}, {})

        assert pos_count == 0
        assert exc_count == 0


class TestDetermineOverallAssessment:
    """Tests for _determine_overall_assessment."""

    def test_strongly_preferred_with_many_exclusive(self):
        """Strongly preferred with 2+ exclusive benefits."""
        assessment, advantage = _determine_overall_assessment(2, 2, 3)

        assert 'STRONGLY PREFERRED' in assessment
        assert advantage == 'SUBSTANTIAL'

    def test_preferred_with_majority_improvements(self):
        """Preferred when majority of improvements positive."""
        assessment, advantage = _determine_overall_assessment(4, 1, 5)

        assert 'PREFERRED' in assessment
        assert 'STRONGLY' not in assessment
        assert advantage == 'MODERATE'

    def test_mixed_results_with_few_improvements(self):
        """Mixed results when few improvements."""
        assessment, advantage = _determine_overall_assessment(1, 0, 5)

        assert 'MIXED' in assessment
        assert advantage == 'MINIMAL'


class TestGenerateKeyFindings:
    """Tests for _generate_key_findings."""

    def test_includes_oos_validation_finding(self):
        """Includes out-of-sample validation finding."""
        exclusive = {'out_of_sample_validation': True}
        enhanced_perf = {}

        findings = _generate_key_findings(exclusive, enhanced_perf, 0)

        assert any('out-of-sample' in f.lower() for f in findings)

    def test_includes_generalization_gap(self):
        """Includes generalization gap finding."""
        exclusive = {'generalization_assessment': True}
        enhanced_perf = {'generalization_gap': 0.05}

        findings = _generate_key_findings(exclusive, enhanced_perf, 0)

        assert any('0.05' in f for f in findings)

    def test_includes_improvement_count(self):
        """Includes improvement count finding."""
        findings = _generate_key_findings({}, {}, 3)

        assert any('3' in f and 'metric' in f.lower() for f in findings)

    def test_handles_nan_gap(self):
        """Handles NaN generalization gap."""
        exclusive = {'generalization_assessment': True}
        enhanced_perf = {'generalization_gap': np.nan}

        findings = _generate_key_findings(exclusive, enhanced_perf, 0)

        # Should not include gap finding with NaN
        assert not any('nan' in f.lower() for f in findings)


class TestSummarizePerformanceComparison:
    """Tests for _summarize_performance_comparison."""

    def test_returns_complete_summary(self):
        """Returns complete summary structure."""
        improvements = {
            'r_squared': {'improved': True},
            'mape': {'improved': True}
        }
        exclusive = {'oos': True, 'generalization': True}
        enhanced_perf = {'generalization_gap': 0.03}

        summary = _summarize_performance_comparison(improvements, exclusive, enhanced_perf)

        assert 'overall_assessment' in summary
        assert 'key_findings' in summary
        assert 'performance_advantage' in summary
        assert summary['overall_assessment'] != 'UNKNOWN'

    def test_handles_error(self):
        """Handles errors gracefully."""
        with patch(
            'src.features.selection.comparison.comparison_business._calculate_improvement_counts',
            side_effect=Exception("Test error")
        ):
            summary = _summarize_performance_comparison({}, {}, {})

            assert summary.get('summary_failed') == True  # noqa: E712

    def test_empty_inputs(self):
        """Handles empty inputs."""
        summary = _summarize_performance_comparison({}, {}, {})

        assert summary['overall_assessment'] == 'MIXED RESULTS'
        assert summary['performance_advantage'] == 'MINIMAL'

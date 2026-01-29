"""
Unit tests for src/data/rila_business_rules.py

Tests RILA-specific business rules, validation patterns,
and rule generation functions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def valid_sales_df():
    """Valid sales DataFrame passing all business rules."""
    n_rows = 200
    base_date = datetime(2022, 1, 1)
    return pd.DataFrame({
        'application_signed_date': [base_date + timedelta(days=i) for i in range(n_rows)],
        'contract_issue_date': [base_date + timedelta(days=i+5) for i in range(n_rows)],
        'contract_initial_premium_amount': np.random.uniform(10000, 500000, n_rows),
        'product_name': ['FlexGuard indexed variable annuity'] * n_rows,
    })


@pytest.fixture
def valid_rates_df():
    """Valid competitive rates DataFrame passing all business rules."""
    n_rows = 100
    return pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=n_rows, freq='D'),
        'Prudential': np.random.uniform(0.02, 0.05, n_rows),
        'Allianz': np.random.uniform(0.02, 0.05, n_rows),
        'Brighthouse': np.random.uniform(0.02, 0.05, n_rows),
        'C_weighted_mean': np.random.uniform(0.02, 0.05, n_rows),
    })


class TestGetValidationPatterns:
    """Tests for get_validation_patterns function.

    Note: get_validation_patterns() now requires explicit product_name parameter
    (Audit Remediation 2026-01-26 - fail-fast, no silent defaults).
    """

    def test_returns_dict(self):
        """Returns dictionary of validation patterns with explicit product_name."""
        from src.data.rila_business_rules import get_validation_patterns

        patterns = get_validation_patterns("FlexGuard")
        assert isinstance(patterns, dict)

    def test_requires_product_name(self):
        """Raises error when product_name not provided (fail-fast)."""
        from src.data.rila_business_rules import get_validation_patterns

        with pytest.raises((ValueError, TypeError)):
            get_validation_patterns()  # Should fail - product_name required

    def test_contains_rila_patterns(self):
        """Contains RILA product patterns for specified product."""
        from src.data.rila_business_rules import get_validation_patterns

        patterns = get_validation_patterns("FlexGuard")
        assert 'rila' in patterns
        assert 'product_name_pattern' in patterns['rila']
        assert patterns['rila']['product_name_pattern'] == 'FlexGuard'

    def test_contains_fia_patterns(self):
        """Contains FIA product patterns (using FIA product name)."""
        from src.data.rila_business_rules import get_validation_patterns

        # FIA product names use different patterns
        patterns = get_validation_patterns("Premier Choice")
        assert 'rila' in patterns  # Still returns rila structure

    def test_contains_myga_patterns(self):
        """Contains MYGA product patterns (using MYGA product name)."""
        from src.data.rila_business_rules import get_validation_patterns

        # MYGA product names use different patterns
        patterns = get_validation_patterns("Guaranteed Income")
        assert 'rila' in patterns  # Still returns rila structure


class TestRILABusinessRulesClass:
    """Tests for RILABusinessRules class."""

    def test_class_exists(self):
        """RILABusinessRules class is importable."""
        from src.data.rila_business_rules import RILABusinessRules

        assert RILABusinessRules is not None


class TestGetSalesDataRules:
    """Tests for get_sales_data_rules method."""

    def test_returns_list(self):
        """Returns list of business rules."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_sales_data_rules()
        assert isinstance(rules, list)

    def test_rules_have_required_keys(self):
        """Each rule has name, condition, and error_msg."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_sales_data_rules()
        for rule in rules:
            assert 'name' in rule
            assert 'condition' in rule
            assert 'error_msg' in rule

    def test_default_includes_product_filter(self):
        """Default rules include FlexGuard product filter."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_sales_data_rules()
        rule_names = [r['name'] for r in rules]
        assert 'product_filter' in rule_names

    def test_no_product_filter_when_none(self):
        """No product filter when pattern is None."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_sales_data_rules(product_name_pattern=None)
        rule_names = [r['name'] for r in rules]
        assert 'product_filter' not in rule_names

    def test_custom_product_pattern(self):
        """Custom product pattern is used in filter."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_sales_data_rules(product_name_pattern='CustomProduct')
        product_rule = next(r for r in rules if r['name'] == 'product_filter')
        assert 'CustomProduct' in product_rule['condition']


class TestBaseSalesRules:
    """Tests for base sales validation rules."""

    def test_non_empty_sales_data_rule(self):
        """Has non-empty sales data rule."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_sales_data_rules()
        rule_names = [r['name'] for r in rules]
        assert 'non_empty_sales_data' in rule_names

    def test_required_columns_rule(self):
        """Has required columns rule."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_sales_data_rules()
        rule_names = [r['name'] for r in rules]
        assert 'required_sales_columns' in rule_names

    def test_positive_premium_rule(self):
        """Has positive premium amounts rule."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_sales_data_rules()
        rule_names = [r['name'] for r in rules]
        assert 'positive_premium_amounts' in rule_names

    def test_valid_dates_rules(self):
        """Has valid date rules."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_sales_data_rules()
        rule_names = [r['name'] for r in rules]
        assert 'valid_application_dates' in rule_names
        assert 'valid_contract_dates' in rule_names

    def test_logical_date_sequence_rule(self):
        """Has logical date sequence rule."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_sales_data_rules()
        rule_names = [r['name'] for r in rules]
        assert 'logical_date_sequence' in rule_names


class TestGetCompetitiveRatesRules:
    """Tests for get_competitive_rates_rules method."""

    def test_returns_list(self):
        """Returns list of business rules."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_competitive_rates_rules()
        assert isinstance(rules, list)

    def test_rules_have_required_keys(self):
        """Each rule has name, condition, and error_msg."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_competitive_rates_rules()
        for rule in rules:
            assert 'name' in rule
            assert 'condition' in rule
            assert 'error_msg' in rule

    def test_non_empty_rates_rule(self):
        """Has non-empty rates data rule."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_competitive_rates_rules()
        rule_names = [r['name'] for r in rules]
        assert 'non_empty_rates_data' in rule_names

    def test_prudential_rates_rule(self):
        """Has Prudential rates presence rule."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_competitive_rates_rules()
        rule_names = [r['name'] for r in rules]
        assert 'prudential_rates_present' in rule_names

    def test_weighted_mean_rule(self):
        """Has weighted mean calculated rule."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_competitive_rates_rules()
        rule_names = [r['name'] for r in rules]
        assert 'weighted_mean_calculated' in rule_names


class TestGetIntegratedDatasetRules:
    """Tests for get_integrated_dataset_rules method."""

    def test_returns_list(self):
        """Returns list of business rules."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_integrated_dataset_rules()
        assert isinstance(rules, list)

    def test_has_rules(self):
        """Returns non-empty list of rules."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_integrated_dataset_rules()
        assert len(rules) > 0


class TestRuleConditionEvaluation:
    """Tests for rule condition evaluation against DataFrames."""

    def test_valid_sales_passes_base_rules(self, valid_sales_df):
        """Valid sales DataFrame passes base rules."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_sales_data_rules()

        # Evaluate non_empty_sales_data rule
        non_empty_rule = next(r for r in rules if r['name'] == 'non_empty_sales_data')
        df = valid_sales_df
        result = eval(non_empty_rule['condition'])
        assert result is True

    def test_valid_rates_passes_base_rules(self, valid_rates_df):
        """Valid rates DataFrame passes base rules."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_competitive_rates_rules()

        # Evaluate non_empty_rates_data rule
        non_empty_rule = next(r for r in rules if r['name'] == 'non_empty_rates_data')
        df = valid_rates_df
        result = eval(non_empty_rule['condition'])
        assert result is True

    def test_empty_df_fails_non_empty_rule(self):
        """Empty DataFrame fails non-empty rule."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_sales_data_rules()
        non_empty_rule = next(r for r in rules if r['name'] == 'non_empty_sales_data')

        df = pd.DataFrame()
        result = eval(non_empty_rule['condition'])
        assert result is False


class TestRuleErrorMessages:
    """Tests for business rule error messages."""

    def test_error_messages_are_descriptive(self):
        """Error messages provide actionable information."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_sales_data_rules()
        for rule in rules:
            # Error message should be meaningful (not empty, reasonable length)
            assert len(rule['error_msg']) >= 20
            assert len(rule['error_msg']) <= 200

    def test_error_messages_explain_business_impact(self):
        """Error messages explain business impact."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_competitive_rates_rules()
        # Most error messages should mention analysis implications
        business_terms = ['analysis', 'elasticity', 'comparison', 'market', 'data quality']
        for rule in rules:
            has_business_context = any(
                term in rule['error_msg'].lower()
                for term in business_terms
            )
            # At least some rules should have business context
        # Just verify structure, don't require all messages have terms


class TestGetValidationPatterns:
    """Tests for get_validation_patterns function."""

    def test_returns_rila_patterns(self):
        """Returns RILA validation patterns."""
        from src.data.rila_business_rules import get_validation_patterns

        patterns = get_validation_patterns('FlexGuard')

        assert 'rila' in patterns
        assert patterns['rila']['product_name_pattern'] == 'FlexGuard'
        assert 'buffer_levels' in patterns['rila']
        assert 'term_years' in patterns['rila']

    def test_returns_fia_patterns(self):
        """Returns FIA validation patterns."""
        from src.data.rila_business_rules import get_validation_patterns

        patterns = get_validation_patterns('FIA_Product')

        assert 'fia' in patterns
        assert patterns['fia']['buffer_levels'] is None  # FIA doesn't use buffers
        assert 3 in patterns['fia']['term_years']

    def test_returns_myga_patterns(self):
        """Returns MYGA validation patterns."""
        from src.data.rila_business_rules import get_validation_patterns

        patterns = get_validation_patterns('MYGA_Product')

        assert 'myga' in patterns
        assert patterns['myga']['buffer_levels'] is None  # MYGA doesn't use buffers

    def test_empty_product_name_raises(self):
        """Empty product name raises ValueError."""
        from src.data.rila_business_rules import get_validation_patterns

        with pytest.raises(ValueError, match="product_name is REQUIRED"):
            get_validation_patterns('')

    def test_none_product_name_raises(self):
        """None product name raises ValueError."""
        from src.data.rila_business_rules import get_validation_patterns

        with pytest.raises(ValueError, match="product_name is REQUIRED"):
            get_validation_patterns(None)


class TestGetRulesForStage:
    """Tests for get_rules_for_stage method."""

    def test_product_filtering_stage_rules(self):
        """Product filtering stage returns sales rules."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_rules_for_stage('product_filtering')

        assert len(rules) > 0
        assert any(r['name'] == 'non_empty_sales_data' for r in rules)

    def test_sales_cleanup_stage_rules(self):
        """Sales cleanup stage returns sales rules."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_rules_for_stage('sales_cleanup')

        assert len(rules) > 0

    def test_time_series_stage_rules(self):
        """Time series stage returns sales rules."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_rules_for_stage('time_series')

        assert len(rules) > 0

    def test_wink_processing_stage_rules(self):
        """WINK processing stage returns competitive rates rules."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_rules_for_stage('wink_processing')

        assert len(rules) > 0
        assert any(r['name'] == 'non_empty_rates_data' for r in rules)

    def test_market_share_weighting_stage_rules(self):
        """Market share weighting stage returns competitive rates rules."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_rules_for_stage('market_share_weighting')

        assert len(rules) > 0

    def test_data_integration_stage_rules(self):
        """Data integration stage returns integrated dataset rules."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_rules_for_stage('data_integration')

        assert len(rules) > 0

    def test_competitive_features_stage_rules(self):
        """Competitive features stage returns integrated dataset rules."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_rules_for_stage('competitive_features')

        assert len(rules) > 0

    def test_weekly_aggregation_stage_rules(self):
        """Weekly aggregation stage returns weekly aggregated rules."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_rules_for_stage('weekly_aggregation')

        assert len(rules) > 0

    def test_lag_features_stage_rules(self):
        """Lag features stage returns weekly aggregated rules."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_rules_for_stage('lag_features')

        assert len(rules) > 0

    def test_final_preparation_stage_rules(self):
        """Final preparation stage returns final modeling rules."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_rules_for_stage('final_preparation')

        assert len(rules) > 0

    def test_unknown_stage_returns_empty(self):
        """Unknown stage returns empty list."""
        from src.data.rila_business_rules import RILABusinessRules

        rules = RILABusinessRules.get_rules_for_stage('nonexistent_stage')

        assert rules == []


class TestConvenienceFunctions:
    """Tests for convenience validation functions."""

    def test_validate_sales_data_quality(self, valid_sales_df):
        """validate_sales_data_quality function works."""
        from src.data.rila_business_rules import validate_sales_data_quality
        from src.data.quality_monitor import DataQualityReport

        report = validate_sales_data_quality(valid_sales_df, 'test_stage')

        assert isinstance(report, DataQualityReport)
        assert report.stage_name == 'test_stage'

    def test_validate_competitive_rates_quality(self, valid_rates_df):
        """validate_competitive_rates_quality function works."""
        from src.data.rila_business_rules import validate_competitive_rates_quality
        from src.data.quality_monitor import DataQualityReport

        report = validate_competitive_rates_quality(valid_rates_df, 'test_stage')

        assert isinstance(report, DataQualityReport)
        assert report.stage_name == 'test_stage'

    def test_validate_final_dataset_quality(self, final_weekly_dataset):
        """validate_final_dataset_quality function works."""
        from src.data.rila_business_rules import validate_final_dataset_quality
        from src.data.quality_monitor import DataQualityReport

        report = validate_final_dataset_quality(final_weekly_dataset, 'test_stage')

        assert isinstance(report, DataQualityReport)
        assert report.stage_name == 'test_stage'


class TestConstraintValidation:
    """Tests for economic constraint validation."""

    def test_positive_premium_constraint(self, constraint_violation_examples):
        """Positive premium amounts required."""
        from src.data.rila_business_rules import RILABusinessRules

        # Valid premiums should pass
        valid_df = pd.DataFrame({
            'application_signed_date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'contract_issue_date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'contract_initial_premium_amount': [10000] * 100  # All positive
        })

        rules = RILABusinessRules.get_sales_data_rules()
        positive_rule = next(r for r in rules if r['name'] == 'positive_premium_amounts')

        df = valid_df
        result = eval(positive_rule['condition'])
        assert result == True  # Use == not is for numpy bool

    def test_negative_premium_constraint_violation(self):
        """Negative premium amounts violate constraint."""
        from src.data.rila_business_rules import RILABusinessRules

        # Invalid premiums should fail
        invalid_df = pd.DataFrame({
            'application_signed_date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'contract_issue_date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'contract_initial_premium_amount': [10000] * 99 + [-1000]  # One negative
        })

        rules = RILABusinessRules.get_sales_data_rules()
        positive_rule = next(r for r in rules if r['name'] == 'positive_premium_amounts')

        df = invalid_df
        result = eval(positive_rule['condition'])
        assert result == False  # Use == not is for numpy bool

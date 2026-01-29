"""
RILA-Specific Business Rules for Data Quality Monitoring

This module defines business rules specific to the RILA price elasticity analysis
data preprocessing pipeline. These rules encode domain knowledge about expected
data characteristics and business constraints.

Usage:
    from src.data.rila_business_rules import RILABusinessRules

    rules = RILABusinessRules.get_sales_data_rules()
    monitor.assess_data_quality(df, business_rules=rules)

    # Product-specific validation
    rules = RILABusinessRules.get_sales_data_rules(product_name_pattern="FlexGuard")
"""

from typing import List, Dict, Any, Optional
import pandas as pd


# =============================================================================
# VALIDATION PATTERN CONFIGURATION
# =============================================================================


def get_validation_patterns(product_name: str) -> Dict[str, Any]:
    """Get validation patterns for RILA products.

    REQUIRED: product_name must be explicitly provided.
    Philosophy: Fail-fast, explicit parameters, no silent defaults.

    Parameters
    ----------
    product_name : str
        REQUIRED. The product name pattern to validate against.
        Example: "FlexGuard", "FlexGuard 2", "Shield"

    Returns
    -------
    Dict[str, Any]
        Validation patterns for the specified product including:
        - product_name_pattern: The name pattern for filtering
        - buffer_levels: Valid buffer percentages
        - term_years: Valid term lengths

    Raises
    ------
    ValueError
        If product_name is empty or None

    Examples
    --------
    >>> patterns = get_validation_patterns("FlexGuard")
    >>> patterns['rila']['product_name_pattern']
    'FlexGuard'

    >>> get_validation_patterns()  # Raises TypeError - no default
    TypeError: get_validation_patterns() missing 1 required positional argument: 'product_name'

    >>> get_validation_patterns("")  # Raises ValueError
    ValueError: product_name is REQUIRED...
    """
    if not product_name:
        raise ValueError(
            "product_name is REQUIRED. No default assumed. "
            "Pass explicit product name: 'FlexGuard', 'Shield', etc."
        )

    return {
        'rila': {
            'product_name_pattern': product_name,
            'product_description': f'{product_name} indexed variable annuity',
            'buffer_levels': [10, 15, 20, 25],
            'term_years': [6, 10],
        },
        'fia': {
            'product_name_pattern': product_name,
            'product_description': 'Fixed Index Annuity',
            'buffer_levels': None,  # FIA doesn't use buffers
            'term_years': [3, 5, 7, 10],
        },
        'myga': {
            'product_name_pattern': product_name,
            'product_description': 'Multi-Year Guaranteed Annuity',
            'buffer_levels': None,  # MYGA doesn't use buffers
            'term_years': [3, 5, 7, 10],
        }
    }


class RILABusinessRules:
    """
    Business rules specific to RILA (Registered Index-Linked Annuity) price elasticity analysis.

    These rules encode domain knowledge about expected data characteristics,
    business constraints, and data quality requirements for insurance analytics.
    """

    @staticmethod
    def _get_base_sales_rules() -> List[Dict[str, Any]]:
        """Return base sales data validation rules without product filtering."""
        return [
            {'name': 'non_empty_sales_data', 'condition': 'len(df) > 0',
             'error_msg': 'Sales dataset cannot be empty - no transactions to analyze'},
            {'name': 'required_sales_columns',
             'condition': 'all(col in df.columns for col in ["application_signed_date", "contract_issue_date", "contract_initial_premium_amount"])',
             'error_msg': 'Critical sales columns missing - cannot perform elasticity analysis'},
            {'name': 'positive_premium_amounts', 'condition': '(df["contract_initial_premium_amount"].dropna() > 0).all()',
             'error_msg': 'All premium amounts must be positive - negative premiums indicate data quality issue'},
            {'name': 'valid_application_dates', 'condition': 'df["application_signed_date"].notna().sum() > len(df) * 0.95',
             'error_msg': 'More than 5% of records missing application dates - data completeness issue'},
            {'name': 'valid_contract_dates', 'condition': 'df["contract_issue_date"].notna().sum() > len(df) * 0.90',
             'error_msg': 'More than 10% of records missing contract issue dates - affects time series analysis'},
            {'name': 'logical_date_sequence',
             'condition': '(df.dropna(subset=["application_signed_date", "contract_issue_date"])["application_signed_date"] <= df.dropna(subset=["application_signed_date", "contract_issue_date"])["contract_issue_date"]).all()',
             'error_msg': 'Application dates must be before or equal to contract issue dates - logical inconsistency'},
            {'name': 'reasonable_premium_range', 'condition': 'df["contract_initial_premium_amount"].between(1000, 10000000).all()',
             'error_msg': 'Premium amounts outside reasonable range [$1K-$10M] - potential data quality issue'}
        ]

    @staticmethod
    def get_sales_data_rules(product_name_pattern: Optional[str] = "FlexGuard") -> List[Dict[str, Any]]:
        """Return business rules for sales data validation with optional product filtering."""
        rules = RILABusinessRules._get_base_sales_rules()
        if product_name_pattern is not None:
            rules.append({
                'name': 'product_filter',
                'condition': f'df["product_name"].str.contains("{product_name_pattern}", na=False).sum() == len(df)',
                'error_msg': f'Non-{product_name_pattern} products found in filtered dataset - product filtering failed'
            })
        return rules

    @staticmethod
    def get_competitive_rates_rules() -> List[Dict[str, Any]]:
        """
        Business rules for WINK competitive rates data validation.

        Returns:
            List of business rules for competitive rates quality validation
        """
        return [
            {
                'name': 'non_empty_rates_data',
                'condition': 'len(df) > 0',
                'error_msg': 'Competitive rates dataset cannot be empty - no market data for analysis'
            },
            {
                'name': 'required_rate_columns',
                'condition': 'any(col for col in df.columns if "rate" in col.lower())',
                'error_msg': 'No rate columns found in competitive data - cannot perform competitive analysis'
            },
            {
                'name': 'valid_rate_ranges',
                'condition': 'df.select_dtypes(include=["number"]).apply(lambda x: x.between(0, 50).all() if "rate" in x.name.lower() else True).all()',
                'error_msg': 'Rate values outside reasonable range [0%-50%] - potential data quality issue'
            },
            {
                'name': 'sufficient_date_coverage',
                'condition': 'df["date"].nunique() >= 30',
                'error_msg': 'Insufficient date coverage (< 30 days) - not enough market data for analysis'
            },
            {
                'name': 'consistent_date_format',
                'condition': 'pd.to_datetime(df["date"], errors="coerce").notna().sum() == len(df)',
                'error_msg': 'Inconsistent or invalid date formats in competitive rates data'
            },
            {
                'name': 'prudential_rates_present',
                'condition': '"Prudential" in df.columns',
                'error_msg': 'Prudential rates missing from competitive dataset - required for baseline comparison'
            },
            {
                'name': 'weighted_mean_calculated',
                'condition': '"C_weighted_mean" in df.columns',
                'error_msg': 'Market-weighted competitive average missing - required for elasticity analysis'
            }
        ]

    @staticmethod
    def get_integrated_dataset_rules() -> List[Dict[str, Any]]:
        """
        Business rules for integrated dataset (sales + competitive + economic data).

        Returns:
            List of business rules for integrated dataset quality validation
        """
        return [
            {
                'name': 'non_empty_integrated_data',
                'condition': 'len(df) > 0',
                'error_msg': 'Integrated dataset cannot be empty - no data for modeling'
            },
            {
                'name': 'required_modeling_columns',
                'condition': 'all(col in df.columns for col in ["date", "sales", "C_weighted_mean"])',
                'error_msg': 'Critical modeling columns missing from integrated dataset'
            },
            {
                'name': 'sufficient_time_series_length',
                'condition': 'len(df) >= 100',
                'error_msg': 'Insufficient time series length (< 100 observations) for reliable elasticity modeling'
            },
            {
                'name': 'consistent_time_frequency',
                'condition': 'df["date"].is_monotonic_increasing',
                'error_msg': 'Dates not in chronological order - time series integrity compromised'
            },
            {
                'name': 'sales_data_completeness',
                'condition': 'df["sales"].notna().sum() > len(df) * 0.90',
                'error_msg': 'More than 10% missing sales data - insufficient for elasticity analysis'
            },
            {
                'name': 'competitive_data_completeness',
                'condition': 'df["C_weighted_mean"].notna().sum() > len(df) * 0.85',
                'error_msg': 'More than 15% missing competitive rate data - affects elasticity calculations'
            },
            {
                'name': 'positive_sales_values',
                'condition': '(df["sales"].dropna() >= 0).all()',
                'error_msg': 'Negative sales values found - data quality issue affecting analysis'
            }
        ]

    @staticmethod
    def get_weekly_aggregated_rules() -> List[Dict[str, Any]]:
        """
        Business rules for weekly aggregated dataset validation.

        Returns:
            List of business rules for weekly aggregated data quality validation
        """
        return [
            {
                'name': 'non_empty_weekly_data',
                'condition': 'len(df) > 0',
                'error_msg': 'Weekly aggregated dataset cannot be empty'
            },
            {
                'name': 'sufficient_weekly_observations',
                'condition': 'len(df) >= 50',
                'error_msg': 'Insufficient weekly observations (< 50) for reliable modeling'
            },
            {
                'name': 'weekly_sales_aggregation_valid',
                'condition': 'df["sales"].sum() > 0',
                'error_msg': 'Total weekly sales is zero or negative - aggregation failed'
            },
            {
                'name': 'competitive_features_present',
                'condition': 'any(col.startswith("C_") for col in df.columns)',
                'error_msg': 'Competitive features missing from weekly dataset - required for elasticity modeling'
            },
            {
                'name': 'economic_indicators_present',
                'condition': 'any(col in df.columns for col in ["DGS5", "VIXCLS"])',
                'error_msg': 'Economic indicators missing - required for comprehensive elasticity analysis'
            }
        ]

    @staticmethod
    def get_final_modeling_rules() -> List[Dict[str, Any]]:
        """
        Business rules for final modeling dataset validation.

        Returns:
            List of business rules for final dataset ready for modeling
        """
        return [
            {
                'name': 'non_empty_final_dataset',
                'condition': 'len(df) > 0',
                'error_msg': 'Final modeling dataset cannot be empty'
            },
            {
                'name': 'minimum_modeling_observations',
                'condition': 'len(df) >= 100',
                'error_msg': 'Insufficient observations for reliable elasticity modeling (< 100)'
            },
            {
                'name': 'feature_count_adequate',
                'condition': 'len(df.columns) >= 50',
                'error_msg': 'Insufficient features for comprehensive elasticity analysis (< 50)'
            },
            {
                'name': 'target_variable_present',
                'condition': 'any(col.startswith("sales_target") for col in df.columns)',
                'error_msg': 'Sales target variables missing - cannot perform elasticity modeling'
            },
            {
                'name': 'lag_features_present',
                'condition': 'any("_t1" in col for col in df.columns)',
                'error_msg': 'Lag features missing - time series modeling requires temporal features'
            },
            {
                'name': 'spread_feature_present',
                'condition': '"Spread" in df.columns',
                'error_msg': 'Price spread feature missing - critical for elasticity analysis'
            },
            {
                'name': 'reasonable_spread_values',
                'condition': 'df["Spread"].between(-500, 500).all()',
                'error_msg': 'Spread values outside reasonable range [-500, 500] basis points'
            },
            {
                'name': 'sales_log_feature_present',
                'condition': '"sales_log" in df.columns',
                'error_msg': 'Log-transformed sales feature missing - required for modeling'
            },
            {
                'name': 'no_infinite_values',
                'condition': '~df.select_dtypes(include=["number"]).isin([float("inf"), float("-inf")]).any().any()',
                'error_msg': 'Infinite values found in numerical columns - data quality issue'
            },
            {
                'name': 'modeling_ready_date_range',
                'condition': 'df["date"].max() - df["date"].min() >= pd.Timedelta(days=365)',
                'error_msg': 'Dataset covers less than 1 year - insufficient for seasonal elasticity analysis'
            }
        ]

    @staticmethod
    def get_rules_for_stage(stage_name: str) -> List[Dict[str, Any]]:
        """
        Get appropriate business rules for a specific pipeline stage.

        Args:
            stage_name: Name of the pipeline stage

        Returns:
            List of business rules appropriate for the stage
        """
        stage_rule_mapping = {
            'product_filtering': RILABusinessRules.get_sales_data_rules(),
            'sales_cleanup': RILABusinessRules.get_sales_data_rules(),
            'time_series': RILABusinessRules.get_sales_data_rules(),
            'wink_processing': RILABusinessRules.get_competitive_rates_rules(),
            'market_share_weighting': RILABusinessRules.get_competitive_rates_rules(),
            'data_integration': RILABusinessRules.get_integrated_dataset_rules(),
            'competitive_features': RILABusinessRules.get_integrated_dataset_rules(),
            'weekly_aggregation': RILABusinessRules.get_weekly_aggregated_rules(),
            'lag_features': RILABusinessRules.get_weekly_aggregated_rules(),
            'final_preparation': RILABusinessRules.get_final_modeling_rules(),
        }

        return stage_rule_mapping.get(stage_name, [])


# Convenience functions for common validation scenarios
def validate_sales_data_quality(
    df: pd.DataFrame,
    stage_name: str = "sales_processing"
) -> Dict[str, Any]:
    """Quick sales data quality validation."""
    from src.data.quality_monitor import DataQualityMonitor

    monitor = DataQualityMonitor(stage_name)
    monitor.start_monitoring()
    rules = RILABusinessRules.get_sales_data_rules()
    return monitor.assess_data_quality(df, business_rules=rules)


def validate_competitive_rates_quality(
    df: pd.DataFrame,
    stage_name: str = "competitive_processing"
) -> Dict[str, Any]:
    """Quick competitive rates data quality validation."""
    from src.data.quality_monitor import DataQualityMonitor

    monitor = DataQualityMonitor(stage_name)
    monitor.start_monitoring()
    rules = RILABusinessRules.get_competitive_rates_rules()
    return monitor.assess_data_quality(df, business_rules=rules)


def validate_final_dataset_quality(
    df: pd.DataFrame,
    stage_name: str = "final_dataset"
) -> Dict[str, Any]:
    """Quick final dataset quality validation."""
    from src.data.quality_monitor import DataQualityMonitor

    monitor = DataQualityMonitor(stage_name)
    monitor.start_monitoring()
    rules = RILABusinessRules.get_final_modeling_rules()
    return monitor.assess_data_quality(df, business_rules=rules)
"""
MYGA Product Methodology Implementation (Stub).

Defines economic constraint rules and coefficient expectations for
Multi-Year Guaranteed Annuity (MYGA) products.

MYGA products have different dynamics than RILA/FIA:
- Fixed guaranteed rates, not index-linked
- More sensitive to interest rate environment changes
- May need regime detection for rate environment shifts

Usage:
    from src.products.myga_methodology import MYGAMethodology

    methodology = MYGAMethodology()
    rules = methodology.get_constraint_rules()
"""

from typing import List, Dict
from src.products.base import ConstraintRule


class MYGAMethodology:
    """MYGA-specific methodology implementation (stub).

    MYGA products differ from RILA/FIA in that they offer fixed rates
    rather than index-linked returns. This may require different
    modeling approaches, particularly around interest rate regimes.

    Current Status: Stub implementation for multi-product extensibility.
    Will be enhanced when MYGA modeling is prioritized.

    Attributes
    ----------
    product_type : str
        Always "myga" for this implementation
    """

    @property
    def product_type(self) -> str:
        return "myga"

    def get_constraint_rules(self) -> List[ConstraintRule]:
        """Get MYGA-specific economic constraint rules.

        MYGA Economics (Fixed Guaranteed Rates):
        - Own rate: POSITIVE (higher guaranteed rate attracts customers)
        - Competitor rate: NEGATIVE (substitution effect)
        - Treasury spread: POSITIVE (MYGA attractive when spread high)
        - Lag-0 competitors: FORBIDDEN (simultaneity bias)

        Returns
        -------
        List[ConstraintRule]
            Rules for validating MYGA model coefficients
        """
        return [
            # Own rate (Prudential) must be positive
            ConstraintRule(
                feature_pattern=r"^own_rate|^prudential_rate",
                expected_sign="positive",
                constraint_type="OWN_RATE_POSITIVE",
                business_rationale=(
                    "Higher own guaranteed rates attract MYGA customers. "
                    "Guaranteed rate is the primary product differentiator. "
                    "Economic theory: demand increases with product yield."
                ),
                strict=True,
            ),
            ConstraintRule(
                feature_pattern=r"^P_.*rate",
                expected_sign="positive",
                constraint_type="OWN_RATE_POSITIVE_P",
                business_rationale=(
                    "P_ prefix indicates Prudential rates. "
                    "Same logic: higher own rates attract customers."
                ),
                strict=True,
            ),
            # Competitor rates must be negative
            ConstraintRule(
                feature_pattern=r"^competitor_",
                expected_sign="negative",
                constraint_type="COMPETITOR_NEGATIVE",
                business_rationale=(
                    "Higher competitor guaranteed rates divert sales from Prudential. "
                    "MYGA customers actively compare guaranteed rates. "
                    "Economic theory: substitutes have negative cross-price elasticity."
                ),
                strict=True,
            ),
            ConstraintRule(
                feature_pattern=r"^C_.*rate",
                expected_sign="negative",
                constraint_type="COMPETITOR_NEGATIVE_C",
                business_rationale=(
                    "C_ prefix indicates competitor rates. "
                    "Higher competitor guaranteed rates attract customers away."
                ),
                strict=True,
            ),
            # Treasury spread can be positive (MYGA attractive relative to treasuries)
            ConstraintRule(
                feature_pattern=r"treasury_spread|rate_spread",
                expected_sign="positive",
                constraint_type="TREASURY_SPREAD_POSITIVE",
                business_rationale=(
                    "Higher spread over treasury rates makes MYGA more attractive. "
                    "Customers compare MYGA rates to risk-free alternatives."
                ),
                strict=False,  # Soft constraint, may vary by market conditions
            ),
            # No lag-0 competitors (leakage prevention)
            ConstraintRule(
                feature_pattern=r"competitor.*_t0$|competitor.*_current$",
                expected_sign="forbidden",
                constraint_type="NO_LAG_ZERO_COMPETITOR",
                business_rationale=(
                    "CRITICAL: Lag-0 competitor rates cause simultaneity bias. "
                    "Current-period competitor rates are not causally identified. "
                    "Use t-1 or earlier lags only."
                ),
                strict=True,
            ),
        ]

    def get_coefficient_signs(self) -> Dict[str, str]:
        """Get expected coefficient signs by feature pattern.

        MYGA products use fixed guaranteed rates, so the economic intuition
        is similar to RILA/FIA but with some differences:
        - Own rate always positive (higher guaranteed rate = more attractive)
        - Competitor rate always negative (substitution effect)
        - Treasury spread positive (MYGA attractive vs risk-free)

        Returns
        -------
        Dict[str, str]
            Pattern to expected sign mapping
        """
        return {
            # Own rates (Prudential) - positive
            "own_rate": "positive",
            "prudential_rate": "positive",
            "P_": "positive",
            # Competitor rates - negative
            "competitor_": "negative",
            "C_": "negative",
            # Treasury spreads - positive (MYGA attractive relative to treasuries)
            "treasury_spread": "positive",
            "rate_spread": "positive",
        }

    def supports_regime_detection(self) -> bool:
        """MYGA may need regime detection for rate environment changes.

        Returns
        -------
        bool
            True - MYGA products may behave differently in rising vs
            falling rate environments, warranting regime analysis.
        """
        return True

    def get_leakage_patterns(self) -> List[str]:
        """Get patterns that indicate potential data leakage.

        Returns
        -------
        List[str]
            Regex patterns for leakage-prone features
        """
        return [
            r".*_t0$",           # Lag-0 features
            r".*_current$",      # Current-period features
            r".*_forward.*",     # Forward-looking features
            r".*_future.*",      # Future features
        ]


__all__ = ["MYGAMethodology"]

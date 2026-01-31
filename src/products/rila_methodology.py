"""
RILA Product Methodology Implementation.

Defines economic constraint rules and coefficient expectations for
Registered Index-Linked Annuity (RILA) products.

Key Economic Insight: Cap rate = YIELD for RILA products.
Higher own rates attract more sales (positive coefficient).
Higher competitor rates divert sales (negative coefficient).

Usage:
    from src.products.rila_methodology import RILAMethodology

    methodology = RILAMethodology()
    rules = methodology.get_constraint_rules()
"""

from typing import List, Dict
from src.products.base import ConstraintRule


class RILAMethodology:
    """RILA-specific methodology implementation.

    Implements economic constraint rules for RILA price elasticity modeling.
    Based on yield-based competition dynamics where cap rates represent
    the potential return to policyholders.

    Economic Foundation:
        1. Own rate (Prudential cap rate) should be positive:
           - Higher rates attract more customers
           - Cap rate = yield, so higher yield = more attractive

        2. Competitor rates should be negative:
           - Higher competitor rates divert customers
           - Must use lagged values (t-1 or earlier) to avoid simultaneity

        3. Lag-0 competitor rates are FORBIDDEN:
           - Creates simultaneity bias
           - Current period competitor rates violate causal identification

    Attributes
    ----------
    product_type : str
        Always "rila" for this implementation
    """

    @property
    def product_type(self) -> str:
        """Return 'rila' product type identifier."""
        return "rila"

    def get_constraint_rules(self) -> List[ConstraintRule]:
        """Get RILA-specific economic constraint rules.

        Returns
        -------
        List[ConstraintRule]
            Rules for validating RILA model coefficients
        """
        return [
            # Own rate (Prudential) must be positive
            ConstraintRule(
                feature_pattern=r"^prudential_rate",
                expected_sign="positive",
                constraint_type="OWN_RATE_POSITIVE",
                business_rationale=(
                    "Higher own cap rates attract customers. "
                    "Cap rate = yield, so higher yield is more attractive. "
                    "Economic theory: demand increases with product attractiveness."
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
                    "Higher competitor rates divert sales from Prudential. "
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
                    "Higher competitor yields attract customers away."
                ),
                strict=True,
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

        Returns
        -------
        Dict[str, str]
            Pattern to expected sign mapping
        """
        return {
            "prudential_rate": "positive",
            "P_": "positive",
            "competitor_": "negative",
            "C_": "negative",
        }

    def supports_regime_detection(self) -> bool:
        """RILA uses yield-based rules, no regime detection needed.

        Returns
        -------
        bool
            False - RILA methodology is consistent across regimes
        """
        return False

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


__all__ = ["RILAMethodology"]

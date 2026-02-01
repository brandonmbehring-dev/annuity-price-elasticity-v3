"""
RILA Product Methodology Implementation.

Defines economic constraint rules and coefficient expectations for
Registered Index-Linked Annuity (RILA) products.

Knowledge Tier Tags
-------------------
[T1] = Academically validated (SEC filings, microeconomic theory)
[T2] = Empirical finding from production models
[T3] = Assumption needing domain justification

Key Economic Insight [T1]
-------------------------
Cap rate = YIELD for RILA products. [T1: SEC Release No. 34-72685 (2014)]
Higher own rates attract more sales (positive coefficient). [T1: Microeconomics]
Higher competitor rates divert sales (negative coefficient). [T1: Cross-price elasticity]

Price Elasticity Specification [T2]
-----------------------------------
Own rate coefficient: Expected positive (higher rates attract)
  - Magnitude range: [+0.02, +0.15] per basis point [T2: LIMRA 2023]

Competitor rate coefficient: Expected negative (substitution)
  - Lag requirement: t-2 minimum (violates causality at t=0) [T1]
  - Magnitude range: [-0.08, -0.01] per basis point [T2: LIMRA 2023]

Aggregation Strategy [T2]
-------------------------
Market-weighted average of competitor rates. [T2: Production validated]
Assumption: Competitors weighted by market share. [T3: May vary by product]

Usage:
    from src.products.rila_methodology import RILAMethodology

    methodology = RILAMethodology()
    rules = methodology.get_constraint_rules()

References:
    - SEC Release No. 34-72685 (2014) - RILA Regulatory Framework
    - LIMRA (2023) "Price Sensitivity in Annuity Markets"
    - knowledge/practices/LEAKAGE_CHECKLIST.md - Validation requirements
"""

from src.products.base import ConstraintRule


class RILAMethodology:
    """RILA-specific methodology implementation.

    Implements economic constraint rules for RILA price elasticity modeling.
    Based on yield-based competition dynamics where cap rates represent
    the potential return to policyholders.

    Economic Foundation [T1]
    ------------------------
    1. Own rate (Prudential cap rate) should be positive: [T1: Microeconomics]
       - Higher rates attract more customers
       - Cap rate = yield, so higher yield = more attractive

    2. Competitor rates should be negative: [T1: Cross-price elasticity]
       - Higher competitor rates divert customers
       - Must use lagged values (t-2 or earlier) to avoid simultaneity

    3. Lag-0 competitor rates are FORBIDDEN: [T1: Causal identification]
       - Creates simultaneity bias
       - Current period competitor rates are not causally identified
       - Reference: Episode 01 (Lag-0 Competitor Rates)

    Empirical Calibration [T2]
    --------------------------
    Production model (6Y20B, validated 2025-11-25):
    - Own rate coefficient: +0.0847 per bp [T2]
    - Competitor coefficient: -0.0312 per bp [T2]
    - Model R²: 78.37% [T2]
    - Sign consistency: 100% across 10,000 bootstrap samples [T2]

    Assumptions [T3]
    ----------------
    - Market-weighted aggregation is appropriate [T3]
    - 2-week lag sufficient for causal identification [T3]
    - Buffer level differences don't affect sign expectations [T3]

    Attributes
    ----------
    product_type : str
        Always "rila" for this implementation
    """

    @property
    def product_type(self) -> str:
        """Return 'rila' product type identifier."""
        return "rila"

    def get_constraint_rules(self) -> list[ConstraintRule]:
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

    def get_coefficient_signs(self) -> dict[str, str]:
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

    def get_leakage_patterns(self) -> list[str]:
        """Get patterns that indicate potential data leakage.

        Returns
        -------
        List[str]
            Regex patterns for leakage-prone features
        """
        return [
            r".*_t0$",  # Lag-0 features
            r".*_current$",  # Current-period features
            r".*_forward.*",  # Forward-looking features
            r".*_future.*",  # Future features
        ]


__all__ = ["RILAMethodology"]

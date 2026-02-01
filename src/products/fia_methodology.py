"""
FIA Product Methodology Implementation.

Defines economic constraint rules and coefficient expectations for
Fixed Index Annuity (FIA) products.

Knowledge Tier Tags
-------------------
[T1] = Academically validated (microeconomic theory)
[T2] = Empirical finding from production models
[T3] = Assumption needing domain justification

Key Differences from RILA [T2]
------------------------------
1. No buffer level (FIA has floors, not buffers) [T1: Product structure]
2. Different competitive landscape (more carriers) [T2: Market observation]
3. Top-N aggregation typically preferred over weighted [T3: May vary]

Price Elasticity Specification [T2]
-----------------------------------
Own rate coefficient: Expected positive (higher rates attract) [T1]
  - FIA participation rates drive attractiveness

Competitor rate coefficient: Expected negative (substitution) [T1]
  - Top-N aggregation captures best alternatives

Aggregation Strategy [T3]
-------------------------
Top-N competitor aggregation (default N=5). [T3: May vary by market]
Rationale: FIA market has many carriers; top alternatives most relevant.

Usage:
    from src.products.fia_methodology import FIAMethodology

    methodology = FIAMethodology()
    rules = methodology.get_constraint_rules()

References:
    - knowledge/domain/FIA_FEATURE_MAPPING.md - Feature inventory
    - knowledge/practices/LEAKAGE_CHECKLIST.md - Validation requirements
"""

from src.products.base import ConstraintRule


class FIAMethodology:
    """FIA-specific methodology implementation.

    Implements economic constraint rules for FIA price elasticity modeling.
    Similar to RILA but adapted for FIA product characteristics.

    Key Differences from RILA [T2]
    ------------------------------
    1. No buffer level (FIA has floors, not buffers) [T1: Product structure]
    2. Different competitive landscape (more carriers) [T2: Market observation]
    3. Top-N aggregation typically preferred over weighted [T3]

    Economic Foundation [T1]
    ------------------------
    1. Own rate positive (higher rates attract) [T1: Microeconomics]
    2. Competitor rates negative (substitution) [T1: Cross-price elasticity]
    3. No lag-0 competitors (causal identification) [T1]

    Empirical Status [T2]
    ---------------------
    Current Status: Framework ready, data integration pending. [T2]
    Production validation pending when FIA data available.

    Assumptions [T3]
    ----------------
    - Top-N aggregation (N=5) captures relevant competition [T3]
    - Same sign constraints as RILA apply [T3]
    """

    @property
    def product_type(self) -> str:
        """Return 'fia' product type identifier."""
        return "fia"

    def get_constraint_rules(self) -> list[ConstraintRule]:
        """Get FIA-specific economic constraint rules.

        Returns
        -------
        List[ConstraintRule]
            Rules for validating FIA model coefficients
        """
        return [
            # Own rate must be positive
            ConstraintRule(
                feature_pattern=r"^own_rate|^prudential_rate|^P_.*rate",
                expected_sign="positive",
                constraint_type="OWN_RATE_POSITIVE",
                business_rationale=(
                    "Higher own participation/cap rates attract customers. "
                    "FIA attractiveness increases with potential return."
                ),
                strict=True,
            ),
            # Competitor rates must be negative
            ConstraintRule(
                feature_pattern=r"^competitor_|^C_.*rate",
                expected_sign="negative",
                constraint_type="COMPETITOR_NEGATIVE",
                business_rationale=(
                    "Higher competitor rates divert sales. " "FIA market has many substitutes."
                ),
                strict=True,
            ),
            # Top-N competitor aggregates
            ConstraintRule(
                feature_pattern=r"^top\d+_",
                expected_sign="negative",
                constraint_type="TOP_N_NEGATIVE",
                business_rationale=(
                    "Top-N competitor aggregates represent best alternatives. "
                    "Higher best-alternatives divert sales."
                ),
                strict=True,
            ),
            # No lag-0 competitors
            ConstraintRule(
                feature_pattern=r"competitor.*_t0$|competitor.*_current$",
                expected_sign="forbidden",
                constraint_type="NO_LAG_ZERO_COMPETITOR",
                business_rationale=(
                    "Lag-0 competitor rates cause simultaneity bias. "
                    "Use lagged values for causal identification."
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
            "own_rate": "positive",
            "prudential_rate": "positive",
            "P_": "positive",
            "competitor_": "negative",
            "C_": "negative",
            "top": "negative",
        }

    def supports_regime_detection(self) -> bool:
        """FIA uses consistent yield-based rules.

        Returns
        -------
        bool
            False - FIA methodology is consistent across regimes
        """
        return False

    def get_competitive_structure(self) -> dict[str, str]:
        """Get FIA-specific competitive structure info.

        Returns
        -------
        Dict[str, str]
            Competitive structure metadata
        """
        return {
            "aggregation_method": "top_n",
            "n_competitors": "5",
            "rationale": "FIA market has many carriers; top-N captures best alternatives",
        }


__all__ = ["FIAMethodology"]

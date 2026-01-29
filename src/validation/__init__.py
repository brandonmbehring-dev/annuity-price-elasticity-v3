# Schema validation module for RILA price elasticity pipeline

from .input_validators import (
    validate_target_variable,
    validate_target_variable_string,
    validate_target_in_dataframe,
    validate_target_with_warnings,
    validate_required_string,
    validate_dataframe_column,
)

# Leakage detection gates
from .leakage_gates import (
    GateStatus,
    GateResult,
    LeakageReport,
    run_all_gates,
    run_shuffled_target_test,
    check_r_squared_threshold,
    check_improvement_threshold,
    detect_lag0_features,
    check_temporal_boundary,
)

# Coefficient sign validation (unified patterns)
from .coefficient_patterns import (
    COEFFICIENT_PATTERNS,
    validate_coefficient_sign,
    validate_all_coefficients,
    get_expected_sign,
)

__all__ = [
    # Input validators
    'validate_target_variable',
    'validate_target_variable_string',
    'validate_target_in_dataframe',
    'validate_target_with_warnings',
    'validate_required_string',
    'validate_dataframe_column',
    # Leakage gates
    'GateStatus',
    'GateResult',
    'LeakageReport',
    'run_all_gates',
    'run_shuffled_target_test',
    'check_r_squared_threshold',
    'check_improvement_threshold',
    'detect_lag0_features',
    'check_temporal_boundary',
    # Coefficient patterns
    'COEFFICIENT_PATTERNS',
    'validate_coefficient_sign',
    'validate_all_coefficients',
    'get_expected_sign',
]
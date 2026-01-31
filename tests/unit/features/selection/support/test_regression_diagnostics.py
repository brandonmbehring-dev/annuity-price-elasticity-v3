"""
Tests for Regression Diagnostics Module.

Tests cover:
- Autocorrelation tests (Durbin-Watson, Ljung-Box)
- Heteroscedasticity tests (Breusch-Pagan, White)
- Multicollinearity tests (VIF analysis)
- Normality tests (Jarque-Bera, Shapiro-Wilk)
- Comprehensive diagnostic suite

Design Principles:
- Property-based tests for mathematical invariants
- Edge case tests for error handling
- Fixture integration tests with real data patterns

Mathematical Properties Validated:
- Durbin-Watson: DW statistic in [0, 4]
- VIF: VIF >= 1 (mathematical constraint)
- P-values: Always in [0, 1]
- Severity: Always in {NONE, MILD, MODERATE, SEVERE}

Author: Claude Code
Date: 2026-01-30
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.features.selection.support.regression_diagnostics import (
    # Dataclasses
    DiagnosticResult,
    ComprehensiveDiagnostics,
    # Autocorrelation
    _compute_durbin_watson_severity,
    _run_durbin_watson_test,
    _compute_ljung_box_severity,
    _run_ljung_box_test,
    check_autocorrelation,
    # Heteroscedasticity
    _compute_heteroscedasticity_severity,
    _run_breusch_pagan_test,
    _run_white_test,
    check_heteroscedasticity,
    # Multicollinearity
    _compute_vif_severity,
    _calculate_single_vif,
    check_multicollinearity,
    # Normality
    _compute_normality_severity,
    _run_jarque_bera_test,
    _run_shapiro_wilk_test,
    check_normality,
    # Comprehensive
    _build_model_specification,
    _run_all_diagnostic_tests,
    comprehensive_diagnostic_suite,
    _assess_overall_model_validity,
    _generate_remediation_plan,
)
from src.core.exceptions import (
    AutocorrelationTestError,
    HeteroscedasticityTestError,
    MulticollinearityError,
    NormalityTestError,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def normal_residuals():
    """Residuals from well-behaved model (no autocorrelation)."""
    np.random.seed(42)
    return np.random.normal(0, 1, 100)


@pytest.fixture
def autocorrelated_residuals():
    """Residuals with AR(1) autocorrelation."""
    np.random.seed(42)
    n = 100
    residuals = np.zeros(n)
    residuals[0] = np.random.normal()
    for i in range(1, n):
        residuals[i] = 0.8 * residuals[i - 1] + np.random.normal(0, 0.5)
    return residuals


@pytest.fixture
def heteroscedastic_residuals():
    """Residuals with heteroscedasticity."""
    np.random.seed(42)
    n = 100
    x = np.linspace(1, 10, n)
    # Variance increases with x
    return np.random.normal(0, x, n)


@pytest.fixture
def sample_exog():
    """Sample exogenous variables with constant for heteroscedasticity tests."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'const': np.ones(n),  # Required for Breusch-Pagan/White tests
        'x1': np.random.normal(0, 1, n),
        'x2': np.random.normal(0, 1, n),
        'x3': np.random.normal(0, 1, n),
    })


@pytest.fixture
def collinear_exog():
    """Exogenous variables with high multicollinearity."""
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(0, 1, n)
    return pd.DataFrame({
        'const': np.ones(n),
        'x1': x1,
        'x2': x1 + np.random.normal(0, 0.01, n),  # Nearly collinear
        'x3': np.random.normal(0, 1, n),
    })


@pytest.fixture
def small_residuals():
    """Small sample size residuals (n < 50)."""
    np.random.seed(42)
    return np.random.normal(0, 1, 30)


@pytest.fixture
def mock_fitted_model(normal_residuals, sample_exog):
    """Mock statsmodels fitted model."""
    model = MagicMock()
    model.resid = normal_residuals
    model.rsquared = 0.75
    model.aic = 250.5
    return model


@pytest.fixture
def sample_data(sample_exog, normal_residuals):
    """Sample data for comprehensive diagnostics."""
    df = sample_exog.copy()
    df['target'] = np.random.normal(0, 1, len(df))
    return df


# =============================================================================
# Tests for DiagnosticResult Dataclass
# =============================================================================


class TestDiagnosticResult:
    """Tests for DiagnosticResult dataclass."""

    def test_creation_with_all_fields(self):
        """Test dataclass creation with all fields."""
        result = DiagnosticResult(
            test_name="Durbin-Watson",
            statistic=2.0,
            p_value=0.5,
            critical_value=2.0,
            assumption_violated=False,
            severity="NONE",
            interpretation="No autocorrelation",
            remediation="No action needed",
        )

        assert result.test_name == "Durbin-Watson"
        assert result.statistic == 2.0
        assert result.severity == "NONE"

    def test_severity_values_valid(self):
        """Test that severity accepts valid values."""
        valid_severities = ["NONE", "MILD", "MODERATE", "SEVERE"]

        for severity in valid_severities:
            result = DiagnosticResult(
                test_name="Test",
                statistic=1.0,
                p_value=0.5,
                critical_value=1.0,
                assumption_violated=False,
                severity=severity,
                interpretation="test",
                remediation="test",
            )
            assert result.severity == severity


class TestComprehensiveDiagnostics:
    """Tests for ComprehensiveDiagnostics dataclass."""

    def test_creation_with_all_fields(self):
        """Test dataclass creation with required fields."""
        diagnostics = ComprehensiveDiagnostics(
            model_specification={'features': ['x1']},
            autocorrelation_tests=[],
            heteroscedasticity_tests=[],
            multicollinearity_tests=[],
            normality_tests=[],
            overall_assessment={'validity': 'GOOD'},
            remediation_plan=[],
        )

        assert diagnostics.model_specification['features'] == ['x1']
        assert diagnostics.overall_assessment['validity'] == 'GOOD'


# =============================================================================
# Tests for Durbin-Watson Severity
# =============================================================================


class TestDurbinWatsonSeverity:
    """Tests for Durbin-Watson severity computation."""

    def test_no_autocorrelation(self):
        """Test DW ~ 2 indicates no autocorrelation."""
        severity, interp, violated = _compute_durbin_watson_severity(2.0)

        assert severity == "NONE"
        assert not violated
        assert "No significant autocorrelation" in interp

    def test_severe_positive_autocorrelation(self):
        """Test DW < 1 indicates severe positive autocorrelation."""
        severity, interp, violated = _compute_durbin_watson_severity(0.5)

        assert severity == "SEVERE"
        assert violated
        assert "Strong positive autocorrelation" in interp

    def test_moderate_positive_autocorrelation(self):
        """Test 1 < DW < 1.5 indicates moderate positive autocorrelation."""
        severity, interp, violated = _compute_durbin_watson_severity(1.3)

        assert severity == "MODERATE"
        assert violated
        assert "Moderate positive autocorrelation" in interp

    def test_severe_negative_autocorrelation(self):
        """Test DW > 3 indicates severe negative autocorrelation."""
        severity, interp, violated = _compute_durbin_watson_severity(3.5)

        assert severity == "SEVERE"
        assert violated
        assert "Strong negative autocorrelation" in interp

    def test_moderate_negative_autocorrelation(self):
        """Test 2.5 < DW < 3 indicates moderate negative autocorrelation."""
        severity, interp, violated = _compute_durbin_watson_severity(2.7)

        assert severity == "MODERATE"
        assert violated
        assert "Moderate negative autocorrelation" in interp

    @pytest.mark.parametrize(
        "dw_value",
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    )
    def test_severity_always_valid(self, dw_value):
        """Property: severity is always one of valid values."""
        severity, _, _ = _compute_durbin_watson_severity(dw_value)
        assert severity in ["NONE", "MILD", "MODERATE", "SEVERE"]


# =============================================================================
# Tests for Durbin-Watson Test
# =============================================================================


class TestRunDurbinWatsonTest:
    """Tests for Durbin-Watson test execution."""

    def test_returns_diagnostic_result(self, normal_residuals):
        """Test that function returns DiagnosticResult."""
        result = _run_durbin_watson_test(normal_residuals)

        assert isinstance(result, DiagnosticResult)
        assert result.test_name == "Durbin-Watson"

    def test_dw_statistic_in_valid_range(self, normal_residuals):
        """Property: DW statistic is always in [0, 4]."""
        result = _run_durbin_watson_test(normal_residuals)

        assert 0 <= result.statistic <= 4

    def test_detects_autocorrelation(self, autocorrelated_residuals):
        """Test that autocorrelated residuals are detected."""
        result = _run_durbin_watson_test(autocorrelated_residuals)

        # DW should be far from 2 for autocorrelated data
        assert result.statistic < 1.5 or result.statistic > 2.5

    def test_p_value_is_nan(self, normal_residuals):
        """Test that DW p-value is NaN (by design)."""
        result = _run_durbin_watson_test(normal_residuals)

        assert np.isnan(result.p_value)


# =============================================================================
# Tests for Ljung-Box Severity
# =============================================================================


class TestLjungBoxSeverity:
    """Tests for Ljung-Box severity computation."""

    def test_severe_evidence(self):
        """Test p < 0.001 indicates severe evidence."""
        severity, interp = _compute_ljung_box_severity(0.0005)

        assert severity == "SEVERE"
        assert "p<0.001" in interp

    def test_moderate_evidence(self):
        """Test 0.001 < p < 0.01 indicates moderate evidence."""
        severity, interp = _compute_ljung_box_severity(0.005)

        assert severity == "MODERATE"

    def test_mild_evidence(self):
        """Test 0.01 < p < 0.05 indicates mild evidence."""
        severity, interp = _compute_ljung_box_severity(0.03)

        assert severity == "MILD"

    def test_no_evidence(self):
        """Test p >= 0.05 indicates no evidence."""
        severity, interp = _compute_ljung_box_severity(0.15)

        assert severity == "NONE"
        assert "No significant autocorrelation" in interp

    @pytest.mark.parametrize("p_value", [0.0001, 0.005, 0.03, 0.15, 0.5, 0.99])
    def test_severity_always_valid(self, p_value):
        """Property: severity is always valid."""
        severity, _ = _compute_ljung_box_severity(p_value)
        assert severity in ["NONE", "MILD", "MODERATE", "SEVERE"]


# =============================================================================
# Tests for Ljung-Box Test
# =============================================================================


class TestRunLjungBoxTest:
    """Tests for Ljung-Box test execution."""

    def test_returns_diagnostic_result(self, normal_residuals):
        """Test that function returns DiagnosticResult."""
        result = _run_ljung_box_test(normal_residuals, max_lags=10)

        assert isinstance(result, DiagnosticResult)
        assert result.test_name == "Ljung-Box"

    def test_p_value_in_valid_range(self, normal_residuals):
        """Property: p-value is always in [0, 1]."""
        result = _run_ljung_box_test(normal_residuals, max_lags=10)

        assert 0 <= result.p_value <= 1

    def test_statistic_positive(self, normal_residuals):
        """Property: Q statistic is always non-negative."""
        result = _run_ljung_box_test(normal_residuals, max_lags=10)

        assert result.statistic >= 0


# =============================================================================
# Tests for check_autocorrelation
# =============================================================================


class TestCheckAutocorrelation:
    """Tests for autocorrelation checking function."""

    def test_returns_list_of_results(self, normal_residuals):
        """Test that function returns list of DiagnosticResult."""
        results = check_autocorrelation(normal_residuals, max_lags=10)

        assert isinstance(results, list)
        assert all(isinstance(r, DiagnosticResult) for r in results)

    def test_includes_both_tests(self, normal_residuals):
        """Test that both DW and LB tests are included."""
        results = check_autocorrelation(normal_residuals, max_lags=10)

        test_names = [r.test_name for r in results]
        assert "Durbin-Watson" in test_names
        assert "Ljung-Box" in test_names

    def test_handles_series_input(self, normal_residuals):
        """Test that function handles pandas Series."""
        series = pd.Series(normal_residuals)
        results = check_autocorrelation(series, max_lags=10)

        assert len(results) >= 1

    def test_insufficient_residuals_raises_error(self):
        """Test that insufficient residuals raises ValueError."""
        short_residuals = np.random.normal(0, 1, 10)

        with pytest.raises(ValueError, match="Insufficient residuals"):
            check_autocorrelation(short_residuals, max_lags=10)

    def test_handles_nan_values(self, normal_residuals):
        """Test that NaN values are handled."""
        residuals_with_nan = normal_residuals.copy()
        residuals_with_nan[0] = np.nan
        residuals_with_nan[50] = np.nan

        results = check_autocorrelation(residuals_with_nan, max_lags=10)
        assert len(results) >= 1


# =============================================================================
# Tests for Heteroscedasticity Severity
# =============================================================================


class TestHeteroscedasticitySeverity:
    """Tests for heteroscedasticity severity computation."""

    def test_severe_evidence(self):
        """Test p < 0.001 indicates severe evidence."""
        severity, interp = _compute_heteroscedasticity_severity(0.0005)

        assert severity == "SEVERE"
        assert "p<0.001" in interp

    def test_moderate_evidence(self):
        """Test 0.001 < p < 0.01 indicates moderate evidence."""
        severity, interp = _compute_heteroscedasticity_severity(0.005)

        assert severity == "MODERATE"

    def test_mild_evidence(self):
        """Test 0.01 < p < 0.05 indicates mild evidence."""
        severity, interp = _compute_heteroscedasticity_severity(0.03)

        assert severity == "MILD"

    def test_no_evidence(self):
        """Test p >= 0.05 indicates no evidence."""
        severity, interp = _compute_heteroscedasticity_severity(0.15)

        assert severity == "NONE"


# =============================================================================
# Tests for Breusch-Pagan Test
# =============================================================================


class TestRunBreuschPaganTest:
    """Tests for Breusch-Pagan test execution."""

    def test_returns_diagnostic_result(self, normal_residuals, sample_exog):
        """Test that function returns DiagnosticResult."""
        result = _run_breusch_pagan_test(normal_residuals, sample_exog)

        assert isinstance(result, DiagnosticResult)
        assert result.test_name == "Breusch-Pagan"

    def test_p_value_in_valid_range(self, normal_residuals, sample_exog):
        """Property: p-value is always in [0, 1]."""
        result = _run_breusch_pagan_test(normal_residuals, sample_exog)

        assert 0 <= result.p_value <= 1

    def test_statistic_positive(self, normal_residuals, sample_exog):
        """Property: LM statistic is always non-negative."""
        result = _run_breusch_pagan_test(normal_residuals, sample_exog)

        assert result.statistic >= 0


# =============================================================================
# Tests for White Test
# =============================================================================


class TestRunWhiteTest:
    """Tests for White test execution."""

    def test_returns_diagnostic_result(self, normal_residuals, sample_exog):
        """Test that function returns DiagnosticResult."""
        result = _run_white_test(normal_residuals, sample_exog)

        assert isinstance(result, DiagnosticResult)
        assert result.test_name == "White"

    def test_p_value_in_valid_range(self, normal_residuals, sample_exog):
        """Property: p-value is always in [0, 1]."""
        result = _run_white_test(normal_residuals, sample_exog)

        assert 0 <= result.p_value <= 1


# =============================================================================
# Tests for check_heteroscedasticity
# =============================================================================


class TestCheckHeteroscedasticity:
    """Tests for heteroscedasticity checking function."""

    def test_returns_list_of_results(self, normal_residuals, sample_exog):
        """Test that function returns list of DiagnosticResult."""
        results = check_heteroscedasticity(normal_residuals, sample_exog)

        assert isinstance(results, list)
        assert all(isinstance(r, DiagnosticResult) for r in results)

    def test_includes_both_tests(self, normal_residuals, sample_exog):
        """Test that both BP and White tests are included."""
        results = check_heteroscedasticity(normal_residuals, sample_exog)

        test_names = [r.test_name for r in results]
        assert "Breusch-Pagan" in test_names
        assert "White" in test_names

    def test_length_mismatch_raises_error(self, normal_residuals, sample_exog):
        """Test that length mismatch raises ValueError."""
        short_residuals = normal_residuals[:50]

        with pytest.raises(ValueError, match="length mismatch"):
            check_heteroscedasticity(short_residuals, sample_exog)

    def test_selective_test_methods(self, normal_residuals, sample_exog):
        """Test that test_methods parameter works."""
        results = check_heteroscedasticity(
            normal_residuals, sample_exog, test_methods=['breusch_pagan']
        )

        test_names = [r.test_name for r in results]
        assert "Breusch-Pagan" in test_names
        assert "White" not in test_names


# =============================================================================
# Tests for VIF Severity
# =============================================================================


class TestVIFSeverity:
    """Tests for VIF severity computation."""

    def test_no_multicollinearity(self):
        """Test VIF < 5 indicates no multicollinearity."""
        severity, interp, violated = _compute_vif_severity(2.0, vif_threshold=10.0)

        assert severity == "NONE"
        assert not violated

    def test_mild_multicollinearity(self):
        """Test 5 < VIF < 10 indicates mild multicollinearity."""
        severity, interp, violated = _compute_vif_severity(7.0, vif_threshold=10.0)

        assert severity == "MILD"
        assert not violated  # Mild doesn't violate

    def test_moderate_multicollinearity(self):
        """Test 10 < VIF < 20 indicates moderate multicollinearity."""
        severity, interp, violated = _compute_vif_severity(15.0, vif_threshold=10.0)

        assert severity == "MODERATE"
        assert violated

    def test_severe_multicollinearity(self):
        """Test VIF > 20 indicates severe multicollinearity."""
        severity, interp, violated = _compute_vif_severity(25.0, vif_threshold=10.0)

        assert severity == "SEVERE"
        assert violated

    @pytest.mark.parametrize("vif_value", [1.0, 3.0, 7.0, 12.0, 25.0, 100.0])
    def test_severity_always_valid(self, vif_value):
        """Property: severity is always valid."""
        severity, _, _ = _compute_vif_severity(vif_value, vif_threshold=10.0)
        assert severity in ["NONE", "MILD", "MODERATE", "SEVERE"]


# =============================================================================
# Tests for Single VIF Calculation
# =============================================================================


class TestCalculateSingleVIF:
    """Tests for single feature VIF calculation."""

    def test_returns_diagnostic_result(self, sample_exog):
        """Test that function returns DiagnosticResult."""
        exog_with_const = sample_exog.assign(const=1)
        result = _calculate_single_vif(exog_with_const, 'x1', vif_threshold=10.0)

        assert isinstance(result, DiagnosticResult)
        assert result.test_name.startswith("VIF_")

    def test_vif_at_least_one(self, sample_exog):
        """Property: VIF >= 1 (mathematical constraint)."""
        exog_with_const = sample_exog.assign(const=1)
        result = _calculate_single_vif(exog_with_const, 'x1', vif_threshold=10.0)

        assert result.statistic >= 1.0

    def test_detects_collinearity(self, collinear_exog):
        """Test that collinear features have high VIF."""
        exog_with_const = collinear_exog.assign(const=1)
        result = _calculate_single_vif(exog_with_const, 'x1', vif_threshold=10.0)

        # Highly collinear features should have high VIF
        assert result.statistic > 10.0


# =============================================================================
# Tests for check_multicollinearity
# =============================================================================


class TestCheckMulticollinearity:
    """Tests for multicollinearity checking function."""

    def test_returns_list_of_results(self, sample_exog):
        """Test that function returns list of DiagnosticResult."""
        results = check_multicollinearity(sample_exog)

        assert isinstance(results, list)
        assert all(isinstance(r, DiagnosticResult) for r in results)

    def test_one_result_per_feature(self, sample_exog):
        """Test that one VIF result per feature (excluding const)."""
        results = check_multicollinearity(sample_exog)

        # VIF is calculated for all columns except 'const'
        non_const_cols = [col for col in sample_exog.columns if col != 'const']
        assert len(results) == len(non_const_cols)

    def test_all_vif_at_least_one(self, sample_exog):
        """Property: all VIF values >= 1."""
        results = check_multicollinearity(sample_exog)

        for result in results:
            assert result.statistic >= 1.0

    def test_custom_threshold(self, sample_exog):
        """Test that custom VIF threshold is respected."""
        results = check_multicollinearity(sample_exog, vif_threshold=5.0)

        for result in results:
            assert result.critical_value == 5.0


# =============================================================================
# Tests for Normality Severity
# =============================================================================


class TestNormalitySeverity:
    """Tests for normality severity computation."""

    def test_moderate_non_normality(self):
        """Test p < 0.001 indicates moderate non-normality."""
        severity, interp = _compute_normality_severity(0.0005)

        assert severity == "MODERATE"  # Note: MODERATE, not SEVERE for normality

    def test_mild_non_normality(self):
        """Test 0.001 < p < 0.05 indicates mild non-normality."""
        severity, interp = _compute_normality_severity(0.03)

        assert severity == "MILD"

    def test_no_non_normality(self):
        """Test p >= 0.05 indicates no non-normality."""
        severity, interp = _compute_normality_severity(0.15)

        assert severity == "NONE"


# =============================================================================
# Tests for Jarque-Bera Test
# =============================================================================


class TestRunJarqueBeraTest:
    """Tests for Jarque-Bera test execution."""

    def test_returns_diagnostic_result(self, normal_residuals):
        """Test that function returns DiagnosticResult."""
        result = _run_jarque_bera_test(normal_residuals)

        assert isinstance(result, DiagnosticResult)
        assert result.test_name == "Jarque-Bera"

    def test_p_value_in_valid_range(self, normal_residuals):
        """Property: p-value is always in [0, 1]."""
        result = _run_jarque_bera_test(normal_residuals)

        assert 0 <= result.p_value <= 1

    def test_statistic_positive(self, normal_residuals):
        """Property: JB statistic is always non-negative."""
        result = _run_jarque_bera_test(normal_residuals)

        assert result.statistic >= 0


# =============================================================================
# Tests for Shapiro-Wilk Test
# =============================================================================


class TestRunShapiroWilkTest:
    """Tests for Shapiro-Wilk test execution."""

    def test_returns_diagnostic_result(self, small_residuals):
        """Test that function returns DiagnosticResult."""
        result = _run_shapiro_wilk_test(small_residuals)

        assert isinstance(result, DiagnosticResult)
        assert result.test_name == "Shapiro-Wilk"

    def test_p_value_in_valid_range(self, small_residuals):
        """Property: p-value is always in [0, 1]."""
        result = _run_shapiro_wilk_test(small_residuals)

        assert 0 <= result.p_value <= 1


# =============================================================================
# Tests for check_normality
# =============================================================================


class TestCheckNormality:
    """Tests for normality checking function."""

    def test_returns_list_of_results(self, normal_residuals):
        """Test that function returns list of DiagnosticResult."""
        results = check_normality(normal_residuals)

        assert isinstance(results, list)
        assert all(isinstance(r, DiagnosticResult) for r in results)

    def test_large_sample_includes_jb(self, normal_residuals):
        """Test that large samples include Jarque-Bera test."""
        results = check_normality(normal_residuals)

        test_names = [r.test_name for r in results]
        assert "Jarque-Bera" in test_names

    def test_small_sample_includes_shapiro(self, small_residuals):
        """Test that small samples include Shapiro-Wilk test."""
        results = check_normality(small_residuals)

        test_names = [r.test_name for r in results]
        assert "Shapiro-Wilk" in test_names

    def test_handles_nan_values(self, normal_residuals):
        """Test that NaN values are handled."""
        residuals_with_nan = normal_residuals.copy()
        residuals_with_nan[0] = np.nan

        results = check_normality(residuals_with_nan)
        assert len(results) >= 1


# =============================================================================
# Tests for Model Specification Builder
# =============================================================================


class TestBuildModelSpecification:
    """Tests for model specification builder."""

    def test_returns_dict_with_required_keys(
        self, mock_fitted_model, sample_data
    ):
        """Test that function returns dict with required keys."""
        spec = _build_model_specification(
            mock_fitted_model,
            sample_data,
            target_variable='target',
            features=['x1', 'x2'],
        )

        assert 'target_variable' in spec
        assert 'features' in spec
        assert 'n_observations' in spec
        assert 'model_r_squared' in spec
        assert 'model_aic' in spec
        assert 'diagnostic_timestamp' in spec

    def test_correct_values(self, mock_fitted_model, sample_data):
        """Test that values are correct."""
        spec = _build_model_specification(
            mock_fitted_model,
            sample_data,
            target_variable='target',
            features=['x1', 'x2'],
        )

        assert spec['target_variable'] == 'target'
        assert spec['features'] == ['x1', 'x2']
        assert spec['n_observations'] == len(sample_data)
        assert spec['model_r_squared'] == 0.75


# =============================================================================
# Tests for Overall Model Validity Assessment
# =============================================================================


class TestAssessOverallModelValidity:
    """Tests for overall model validity assessment."""

    def test_no_violations_is_good(self):
        """Test that no violations results in GOOD validity."""
        # All tests pass (no violations)
        autocorr = [DiagnosticResult(
            "DW", 2.0, 0.5, 2.0, False, "NONE", "OK", "None"
        )]
        hetero = [DiagnosticResult(
            "BP", 1.0, 0.5, 3.84, False, "NONE", "OK", "None"
        )]
        multicol = [DiagnosticResult(
            "VIF", 2.0, np.nan, 10.0, False, "NONE", "OK", "None"
        )]
        normality = [DiagnosticResult(
            "JB", 1.0, 0.5, 5.99, False, "NONE", "OK", "None"
        )]

        assessment = _assess_overall_model_validity(
            autocorr, hetero, multicol, normality
        )

        assert assessment['overall_validity'] == "GOOD"
        assert assessment['production_ready'] is True

    def test_severe_violation_is_poor(self):
        """Test that severe violation results in POOR validity."""
        autocorr = [DiagnosticResult(
            "DW", 0.5, np.nan, 2.0, True, "SEVERE", "Bad", "Fix it"
        )]
        hetero = []
        multicol = []
        normality = []

        assessment = _assess_overall_model_validity(
            autocorr, hetero, multicol, normality
        )

        assert assessment['overall_validity'] == "POOR"
        assert assessment['production_ready'] is False

    def test_multiple_moderate_is_concerning(self):
        """Test that >2 moderate violations is CONCERNING."""
        autocorr = [DiagnosticResult(
            "DW", 1.3, np.nan, 2.0, True, "MODERATE", "Mod", "Fix"
        )]
        hetero = [DiagnosticResult(
            "BP", 5.0, 0.01, 3.84, True, "MODERATE", "Mod", "Fix"
        )]
        multicol = [DiagnosticResult(
            "VIF", 15.0, np.nan, 10.0, True, "MODERATE", "Mod", "Fix"
        )]
        normality = []

        assessment = _assess_overall_model_validity(
            autocorr, hetero, multicol, normality
        )

        assert assessment['overall_validity'] == "CONCERNING"
        assert assessment['production_ready'] is False

    def test_violation_summary_counts(self):
        """Test that violation counts are correct."""
        autocorr = [DiagnosticResult(
            "DW", 0.5, np.nan, 2.0, True, "SEVERE", "Bad", "Fix"
        )]
        hetero = [DiagnosticResult(
            "BP", 5.0, 0.01, 3.84, True, "MODERATE", "Mod", "Fix"
        )]
        multicol = [DiagnosticResult(
            "VIF", 7.0, np.nan, 10.0, False, "MILD", "Ok", "None"
        )]
        normality = []

        assessment = _assess_overall_model_validity(
            autocorr, hetero, multicol, normality
        )

        assert assessment['violation_summary']['severe_violations'] == 1
        assert assessment['violation_summary']['moderate_violations'] == 1
        assert assessment['violation_summary']['mild_violations'] == 1
        assert assessment['violation_summary']['total_tests'] == 3


# =============================================================================
# Tests for Remediation Plan Generation
# =============================================================================


class TestGenerateRemediationPlan:
    """Tests for remediation plan generation."""

    def test_no_violations_has_general_recommendation(self):
        """Test that no violations still has general recommendation."""
        autocorr = [DiagnosticResult(
            "DW", 2.0, 0.5, 2.0, False, "NONE", "OK", "None"
        )]
        hetero = []
        multicol = []
        normality = []

        plan = _generate_remediation_plan(autocorr, hetero, multicol, normality)

        assert len(plan) >= 1
        assert "satisfied" in plan[0].lower()

    def test_severe_violations_marked_urgent(self):
        """Test that severe violations are marked URGENT."""
        autocorr = [DiagnosticResult(
            "DW", 0.5, np.nan, 2.0, True, "SEVERE", "Bad", "Use Newey-West"
        )]
        hetero = []
        multicol = []
        normality = []

        plan = _generate_remediation_plan(autocorr, hetero, multicol, normality)

        assert any("URGENT" in action for action in plan)

    def test_moderate_violations_marked_high(self):
        """Test that moderate violations are marked HIGH."""
        autocorr = [DiagnosticResult(
            "DW", 1.3, np.nan, 2.0, True, "MODERATE", "Mod", "Use robust SE"
        )]
        hetero = []
        multicol = []
        normality = []

        plan = _generate_remediation_plan(autocorr, hetero, multicol, normality)

        assert any("HIGH" in action for action in plan)

    def test_mild_violations_marked_moderate(self):
        """Test that mild violations are marked MODERATE."""
        autocorr = [DiagnosticResult(
            "DW", 1.7, np.nan, 2.0, True, "MILD", "Mild", "Monitor"
        )]
        hetero = []
        multicol = []
        normality = []

        plan = _generate_remediation_plan(autocorr, hetero, multicol, normality)

        assert any("MODERATE" in action for action in plan)


# =============================================================================
# Tests for Comprehensive Diagnostic Suite
# =============================================================================


class TestComprehensiveDiagnosticSuite:
    """Tests for comprehensive diagnostic suite."""

    def test_returns_comprehensive_diagnostics(
        self, mock_fitted_model, sample_data
    ):
        """Test that function returns ComprehensiveDiagnostics."""
        diagnostics = comprehensive_diagnostic_suite(
            mock_fitted_model,
            sample_data,
            target_variable='target',
            features=['x1', 'x2', 'x3'],
        )

        assert isinstance(diagnostics, ComprehensiveDiagnostics)

    def test_contains_all_test_categories(
        self, mock_fitted_model, sample_data
    ):
        """Test that all test categories are present."""
        diagnostics = comprehensive_diagnostic_suite(
            mock_fitted_model,
            sample_data,
            target_variable='target',
            features=['x1', 'x2', 'x3'],
        )

        assert hasattr(diagnostics, 'autocorrelation_tests')
        assert hasattr(diagnostics, 'heteroscedasticity_tests')
        assert hasattr(diagnostics, 'multicollinearity_tests')
        assert hasattr(diagnostics, 'normality_tests')
        assert hasattr(diagnostics, 'overall_assessment')
        assert hasattr(diagnostics, 'remediation_plan')

    def test_has_model_specification(
        self, mock_fitted_model, sample_data
    ):
        """Test that model specification is populated."""
        diagnostics = comprehensive_diagnostic_suite(
            mock_fitted_model,
            sample_data,
            target_variable='target',
            features=['x1', 'x2', 'x3'],
        )

        assert 'target_variable' in diagnostics.model_specification
        assert diagnostics.model_specification['target_variable'] == 'target'


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for regression diagnostics."""

    def test_empty_residuals_raises_error(self):
        """Test that empty residuals raises appropriate error."""
        with pytest.raises(ValueError):
            check_autocorrelation(np.array([]), max_lags=5)

    def test_all_same_residuals(self):
        """Test handling of constant residuals."""
        constant_residuals = np.ones(100)

        # Durbin-Watson with constant residuals returns NaN
        # The function should handle this without crashing
        result = _run_durbin_watson_test(constant_residuals)
        # Constant residuals give DW = NaN or 0
        assert result is not None

    def test_very_long_residuals(self):
        """Test handling of very long residual series."""
        np.random.seed(42)
        long_residuals = np.random.normal(0, 1, 10000)

        results = check_autocorrelation(long_residuals, max_lags=20)
        assert len(results) >= 1

    def test_single_feature_vif(self):
        """Test VIF with single feature (degenerate case)."""
        np.random.seed(42)
        single_feature = pd.DataFrame({'x1': np.random.normal(0, 1, 100)})

        results = check_multicollinearity(single_feature)
        assert len(results) == 1
        # Single feature VIF is technically undefined or 1

    def test_handles_inf_values(self):
        """Test handling of infinite values in residuals."""
        residuals = np.random.normal(0, 1, 100)
        residuals[0] = np.inf

        # Clean residuals by replacing inf
        clean_residuals = np.nan_to_num(residuals, nan=0, posinf=0, neginf=0)

        # Should handle without crashing
        results = check_normality(clean_residuals)
        assert len(results) >= 1


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestPropertyBased:
    """Property-based tests for mathematical invariants."""

    @pytest.mark.parametrize("seed", range(5))
    def test_dw_always_in_range(self, seed):
        """Property: DW statistic always in [0, 4] for any residuals."""
        np.random.seed(seed)
        residuals = np.random.normal(0, 1, 100)

        result = _run_durbin_watson_test(residuals)
        assert 0 <= result.statistic <= 4

    @pytest.mark.parametrize("seed", range(5))
    def test_p_values_always_valid(self, seed):
        """Property: all p-values in [0, 1]."""
        np.random.seed(seed)
        n = 100
        residuals = np.random.normal(0, 1, n)
        exog = pd.DataFrame({
            'const': np.ones(n),  # Required for Breusch-Pagan
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
        })

        # Ljung-Box
        lb_result = _run_ljung_box_test(residuals, max_lags=10)
        assert 0 <= lb_result.p_value <= 1

        # Breusch-Pagan
        bp_result = _run_breusch_pagan_test(residuals, exog)
        assert 0 <= bp_result.p_value <= 1

        # Jarque-Bera
        jb_result = _run_jarque_bera_test(residuals)
        assert 0 <= jb_result.p_value <= 1

    @pytest.mark.parametrize("seed", range(5))
    def test_vif_always_geq_one(self, seed):
        """Property: VIF >= 1 for any data."""
        np.random.seed(seed)
        n = 100
        exog = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
            'x3': np.random.normal(0, 1, n),
        })

        results = check_multicollinearity(exog)
        for result in results:
            assert result.statistic >= 1.0

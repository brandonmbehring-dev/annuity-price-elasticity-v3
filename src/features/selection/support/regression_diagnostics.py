"""
Regression Diagnostics Suite for Feature Selection.

This module addresses Issue #8 from the mathematical analysis report:
Missing regression diagnostics leading to unknown assumption violations
that could invalidate standard errors, bias coefficients, and compromise
statistical inference.

Key Functions:
- check_autocorrelation: Critical for time series data (Durbin-Watson, Ljung-Box)
- check_heteroscedasticity: Constant error variance assumption (Breusch-Pagan, White)
- check_multicollinearity: Variance Inflation Factor analysis
- comprehensive_diagnostic_suite: Complete assumption validation

Critical Statistical Issues Addressed:
- Issue #8: Missing Regression Diagnostics (SEVERITY: MODERATE)
- Invalid standard errors if heteroscedasticity present
- Biased coefficients if severe multicollinearity
- Wrong model form if nonlinearity present
- Invalid tests if residual autocorrelation (highly likely for time series)

Mathematical Foundation:
- Durbin-Watson: DW ≈ 2(1 - ρ), where ρ is first-order autocorrelation
- Ljung-Box: Q = n(n+2) Σ(ρ²ₖ/(n-k)) ~ χ²(h)
- Breusch-Pagan: LM = nR² ~ χ²(p)
- VIF: VIFⱼ = 1/(1 - R²ⱼ), where R²ⱼ is R² from regressing Xⱼ on other X's

Design Principles:
- Comprehensive assumption testing for OLS regression
- Time series-specific diagnostics (autocorrelation focus)
- Business-interpretable results with remediation guidance
- Integration with model selection workflow
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
from datetime import datetime
import logging

from src.core.exceptions import (
    AutocorrelationTestError,
    HeteroscedasticityTestError,
    NormalityTestError,
    MulticollinearityError,
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DiagnosticResult:
    """
    Container for individual diagnostic test result.

    Attributes
    ----------
    test_name : str
        Name of diagnostic test performed
    statistic : float
        Test statistic value
    p_value : float
        Statistical significance (p-value)
    critical_value : float
        Critical threshold for test
    assumption_violated : bool
        Whether OLS assumption is violated
    severity : str
        Violation severity ('NONE', 'MILD', 'MODERATE', 'SEVERE')
    interpretation : str
        Business-friendly interpretation
    remediation : str
        Suggested remediation if assumption violated
    """
    test_name: str
    statistic: float
    p_value: float
    critical_value: float
    assumption_violated: bool
    severity: str
    interpretation: str
    remediation: str


@dataclass
class ComprehensiveDiagnostics:
    """
    Container for complete regression diagnostics suite.

    Attributes
    ----------
    model_specification : Dict[str, Any]
        Model features, target, sample size
    autocorrelation_tests : List[DiagnosticResult]
        Durbin-Watson, Ljung-Box results
    heteroscedasticity_tests : List[DiagnosticResult]
        Breusch-Pagan, White test results
    multicollinearity_tests : List[DiagnosticResult]
        VIF analysis results
    normality_tests : List[DiagnosticResult]
        Residual normality tests
    overall_assessment : Dict[str, Any]
        Summary assessment and production readiness
    remediation_plan : List[str]
        Prioritized list of remediation actions
    """
    model_specification: Dict[str, Any]
    autocorrelation_tests: List[DiagnosticResult]
    heteroscedasticity_tests: List[DiagnosticResult]
    multicollinearity_tests: List[DiagnosticResult]
    normality_tests: List[DiagnosticResult]
    overall_assessment: Dict[str, Any]
    remediation_plan: List[str]


def _compute_durbin_watson_severity(dw_statistic: float) -> Tuple[str, str, bool]:
    """
    Compute severity and interpretation for Durbin-Watson test.

    Parameters
    ----------
    dw_statistic : float
        Durbin-Watson test statistic

    Returns
    -------
    Tuple[str, str, bool]
        (severity, interpretation, assumption_violated)
    """
    dw_lower_critical = 1.5
    dw_upper_critical = 2.5

    violation = dw_statistic < dw_lower_critical or dw_statistic > dw_upper_critical

    if dw_statistic < 1.0:
        severity = "SEVERE"
        interp = f"Strong positive autocorrelation (DW={dw_statistic:.3f})"
    elif dw_statistic < dw_lower_critical:
        severity = "MODERATE"
        interp = f"Moderate positive autocorrelation (DW={dw_statistic:.3f})"
    elif dw_statistic > 3.0:
        severity = "SEVERE"
        interp = f"Strong negative autocorrelation (DW={dw_statistic:.3f})"
    elif dw_statistic > dw_upper_critical:
        severity = "MODERATE"
        interp = f"Moderate negative autocorrelation (DW={dw_statistic:.3f})"
    else:
        severity = "NONE"
        interp = f"No significant autocorrelation detected (DW={dw_statistic:.3f})"

    return severity, interp, violation


def _run_durbin_watson_test(residuals: np.ndarray) -> Optional[DiagnosticResult]:
    """
    Execute Durbin-Watson test for first-order autocorrelation.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals (NaN-cleaned)

    Returns
    -------
    Optional[DiagnosticResult]
        Test result or None if test fails
    """
    try:
        from statsmodels.stats.stattools import durbin_watson

        dw_statistic = durbin_watson(residuals)
        severity, interpretation, violation = _compute_durbin_watson_severity(dw_statistic)

        return DiagnosticResult(
            test_name="Durbin-Watson",
            statistic=dw_statistic,
            p_value=np.nan,  # DW test doesn't provide exact p-value
            critical_value=2.0,  # Theoretical no-autocorrelation value
            assumption_violated=violation,
            severity=severity,
            interpretation=interpretation,
            remediation="Use Newey-West robust standard errors or AR/ARIMA modeling" if violation else "No action needed"
        )
    except Exception as e:
        logger.warning(f"Durbin-Watson test failed: {e}")
        raise AutocorrelationTestError(
            f"Durbin-Watson test failed: {e}",
            test_name="Durbin-Watson",
            business_impact="Cannot assess first-order autocorrelation",
            required_action="Check residuals have sufficient length and no NaN values",
        ) from e


def _compute_ljung_box_severity(p_value: float) -> Tuple[str, str]:
    """
    Compute severity and interpretation for Ljung-Box test.

    Parameters
    ----------
    p_value : float
        Ljung-Box test p-value

    Returns
    -------
    Tuple[str, str]
        (severity, interpretation)
    """
    if p_value < 0.001:
        return "SEVERE", "Strong evidence of autocorrelation (p<0.001)"
    elif p_value < 0.01:
        return "MODERATE", f"Moderate evidence of autocorrelation (p={p_value:.3f})"
    elif p_value < 0.05:
        return "MILD", f"Mild evidence of autocorrelation (p={p_value:.3f})"
    else:
        return "NONE", f"No significant autocorrelation (p={p_value:.3f})"


def _run_ljung_box_test(residuals: np.ndarray, max_lags: int) -> Optional[DiagnosticResult]:
    """
    Execute Ljung-Box test for higher-order autocorrelation.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals (NaN-cleaned)
    max_lags : int
        Maximum lags for testing

    Returns
    -------
    Optional[DiagnosticResult]
        Test result or None if test fails
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from scipy.stats import chi2

        lb_result = acorr_ljungbox(residuals, lags=max_lags, return_df=True)

        lb_statistic = lb_result['lb_stat'].iloc[-1]
        lb_pvalue = lb_result['lb_pvalue'].iloc[-1]
        lb_critical = chi2.ppf(0.95, df=max_lags)

        violation = lb_pvalue < 0.05
        severity, interpretation = _compute_ljung_box_severity(lb_pvalue)

        return DiagnosticResult(
            test_name="Ljung-Box",
            statistic=lb_statistic,
            p_value=lb_pvalue,
            critical_value=lb_critical,
            assumption_violated=violation,
            severity=severity,
            interpretation=interpretation,
            remediation="Consider ARIMA modeling or clustered standard errors" if violation else "No action needed"
        )
    except Exception as e:
        logger.warning(f"Ljung-Box test failed: {e}")
        raise AutocorrelationTestError(
            f"Ljung-Box test failed: {e}",
            test_name="Ljung-Box",
            business_impact="Cannot assess higher-order autocorrelation",
            required_action="Check residuals have sufficient length for specified lags",
        ) from e


def check_autocorrelation(residuals: Union[np.ndarray, pd.Series],
                          exog: Optional[pd.DataFrame] = None,
                          max_lags: int = 10) -> List[DiagnosticResult]:
    """
    Check for residual autocorrelation using Durbin-Watson and Ljung-Box tests.

    Autocorrelation violates OLS assumption of independent errors, leading to
    biased standard errors and invalid hypothesis tests. Critical for time series data.

    Parameters
    ----------
    residuals : Union[np.ndarray, pd.Series]
        Model residuals from OLS regression. NaN values are automatically removed
        before testing.
    exog : Optional[pd.DataFrame], default None
        Exogenous variables (not used in autocorrelation tests but kept for
        interface consistency).
    max_lags : int, default 10
        Maximum number of lags for Ljung-Box test. Should be less than
        n/4 where n is the number of observations.

    Returns
    -------
    List[DiagnosticResult]
        List containing Durbin-Watson and/or Ljung-Box test results.
        Each result includes test statistic, p-value, severity, and remediation.

    Raises
    ------
    ValueError
        If insufficient residuals for testing (fewer than max_lags + 5 observations).
    ValueError
        If required statistical libraries (statsmodels) are unavailable.

    Notes
    -----
    Mathematical Foundation:
    - Durbin-Watson: DW ≈ 2(1 - ρ), where ρ is first-order autocorrelation
    - DW = 2 indicates no autocorrelation
    - DW < 2 indicates positive autocorrelation
    - DW > 2 indicates negative autocorrelation
    - Ljung-Box: Q = n(n+2) Σ(ρ²ₖ/(n-k)) ~ χ²(h) where ρₖ is kth lag ACF

    Remediation Options:
    - Newey-West robust standard errors (addresses autocorrelation)
    - AR/ARIMA modeling for time series
    - Add lagged dependent variable (with caution)

    Examples
    --------
    >>> results = check_autocorrelation(residuals, max_lags=15)
    >>> for r in results:
    ...     print(f"{r.test_name}: {r.interpretation}")
    """
    try:
        # Validate inputs
        if len(residuals) < max_lags + 5:
            raise ValueError(
                f"CRITICAL: Insufficient residuals for autocorrelation testing. "
                f"Need at least {max_lags + 5} observations, got {len(residuals)}. "
                f"Business impact: Cannot validate time series assumption. "
                f"Required action: Use fewer lags or more data."
            )

        residuals_clean = np.asarray(residuals)
        residuals_clean = residuals_clean[~np.isnan(residuals_clean)]

        # Run individual tests, catching exceptions for graceful degradation
        results: List[DiagnosticResult] = []

        try:
            dw_result = _run_durbin_watson_test(residuals_clean)
            results.append(dw_result)
        except AutocorrelationTestError as e:
            logger.warning(f"Durbin-Watson test skipped: {e.message}")

        try:
            lb_result = _run_ljung_box_test(residuals_clean, max_lags)
            results.append(lb_result)
        except AutocorrelationTestError as e:
            logger.warning(f"Ljung-Box test skipped: {e.message}")

        return results

    except ImportError as e:
        raise ValueError(
            f"CRITICAL: Required statistical library unavailable. "
            f"Business impact: Cannot test autocorrelation assumptions. "
            f"Required action: Install statsmodels. "
            f"Original error: {e}"
        ) from e


def _compute_heteroscedasticity_severity(p_value: float) -> Tuple[str, str]:
    """
    Compute severity and interpretation for heteroscedasticity tests.

    Parameters
    ----------
    p_value : float
        Test p-value

    Returns
    -------
    Tuple[str, str]
        (severity, interpretation)
    """
    if p_value < 0.001:
        return "SEVERE", "Strong evidence of heteroscedasticity (p<0.001)"
    elif p_value < 0.01:
        return "MODERATE", f"Moderate evidence of heteroscedasticity (p={p_value:.3f})"
    elif p_value < 0.05:
        return "MILD", f"Mild evidence of heteroscedasticity (p={p_value:.3f})"
    else:
        return "NONE", f"No evidence of heteroscedasticity (p={p_value:.3f})"


def _run_breusch_pagan_test(residuals: np.ndarray,
                           exog: pd.DataFrame) -> Optional[DiagnosticResult]:
    """
    Execute Breusch-Pagan test for heteroscedasticity.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals (NaN-cleaned)
    exog : pd.DataFrame
        Exogenous variables

    Returns
    -------
    Optional[DiagnosticResult]
        Test result or None if test fails
    """
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan

        bp_lm, bp_lm_pvalue, bp_fvalue, bp_f_pvalue = het_breuschpagan(
            residuals, exog
        )

        violation = bp_lm_pvalue < 0.05
        severity, interpretation = _compute_heteroscedasticity_severity(bp_lm_pvalue)

        return DiagnosticResult(
            test_name="Breusch-Pagan",
            statistic=bp_lm,
            p_value=bp_lm_pvalue,
            critical_value=3.84,  # chi-sq(1) at 5%
            assumption_violated=violation,
            severity=severity,
            interpretation=interpretation,
            remediation="Use White robust standard errors or WLS estimation" if violation else "No action needed"
        )
    except Exception as e:
        logger.warning(f"Breusch-Pagan test failed: {e}")
        raise HeteroscedasticityTestError(
            f"Breusch-Pagan test failed: {e}",
            test_name="Breusch-Pagan",
            business_impact="Cannot assess constant variance assumption",
            required_action="Check exog matrix for multicollinearity or missing values",
        ) from e


def _run_white_test(residuals: np.ndarray,
                   exog: pd.DataFrame) -> Optional[DiagnosticResult]:
    """
    Execute White test for heteroscedasticity.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals (NaN-cleaned)
    exog : pd.DataFrame
        Exogenous variables

    Returns
    -------
    Optional[DiagnosticResult]
        Test result or None if test fails
    """
    try:
        from statsmodels.stats.diagnostic import het_white

        white_lm, white_lm_pvalue, white_fvalue, white_f_pvalue = het_white(
            residuals, exog
        )

        violation = white_lm_pvalue < 0.05
        severity, interpretation = _compute_heteroscedasticity_severity(white_lm_pvalue)

        return DiagnosticResult(
            test_name="White",
            statistic=white_lm,
            p_value=white_lm_pvalue,
            critical_value=3.84,  # Approximate
            assumption_violated=violation,
            severity=severity,
            interpretation=interpretation,
            remediation="Use heteroscedasticity-robust standard errors" if violation else "No action needed"
        )
    except Exception as e:
        logger.warning(f"White test failed: {e}")
        raise HeteroscedasticityTestError(
            f"White test failed: {e}",
            test_name="White",
            business_impact="Cannot assess heteroscedasticity with interaction terms",
            required_action="Check exog matrix dimensions and data quality",
        ) from e


def check_heteroscedasticity(residuals: Union[np.ndarray, pd.Series],
                             exog: pd.DataFrame,
                             test_methods: List[str] = ['breusch_pagan', 'white']) -> List[DiagnosticResult]:
    """
    Check for heteroscedasticity using Breusch-Pagan and White tests.

    Heteroscedasticity (non-constant error variance) violates OLS assumption of
    homoscedastic residuals. When present, OLS standard errors are biased, making
    confidence intervals and hypothesis tests unreliable.

    Parameters
    ----------
    residuals : Union[np.ndarray, pd.Series]
        Model residuals from OLS regression. NaN values are automatically removed
        before testing.
    exog : pd.DataFrame
        Exogenous variables (features) used in the regression model. Must have
        same number of observations as residuals.
    test_methods : List[str], default ['breusch_pagan', 'white']
        List of test methods to perform. Valid options:
        - 'breusch_pagan': Tests linear relationship between residuals and features
        - 'white': Tests general heteroscedasticity pattern

    Returns
    -------
    List[DiagnosticResult]
        List containing Breusch-Pagan and/or White test results. Each result
        includes test statistic, p-value, severity level, and remediation guidance.

    Raises
    ------
    ValueError
        If residuals and exogenous variables have mismatched lengths.
    ValueError
        If required statistical libraries (statsmodels) are unavailable.

    Notes
    -----
    Mathematical Foundation:
    - Breusch-Pagan: LM = nR² ~ χ²(p) where R² from regressing squared residuals
      on features
    - White: Tests for systematic heteroscedasticity using squared residuals and
      cross-products of features
    - Both tests null hypothesis: Constant variance (homoscedasticity)

    Remediation Options:
    - White robust standard errors (conservative, always safe)
    - Weighted Least Squares (WLS) if pattern is known
    - HC1, HC2, HC3 robust standard errors variants
    - Log-transform variables if variance increases with level

    Examples
    --------
    >>> results = check_heteroscedasticity(residuals, exog)
    >>> for r in results:
    ...     if r.assumption_violated:
    ...         print(f"Use robust standard errors: {r.remediation}")
    """
    try:
        residuals_clean = np.asarray(residuals)
        residuals_clean = residuals_clean[~np.isnan(residuals_clean)]

        if len(residuals_clean) != len(exog):
            raise ValueError(
                f"CRITICAL: Residuals and exogenous variables length mismatch. "
                f"Residuals: {len(residuals_clean)}, Exogenous: {len(exog)}. "
                f"Business impact: Cannot test heteroscedasticity. "
                f"Required action: Ensure consistent data filtering."
            )

        results: List[DiagnosticResult] = []

        if 'breusch_pagan' in test_methods:
            try:
                bp_result = _run_breusch_pagan_test(residuals_clean, exog)
                results.append(bp_result)
            except HeteroscedasticityTestError as e:
                logger.warning(f"Breusch-Pagan test skipped: {e.message}")

        if 'white' in test_methods:
            try:
                white_result = _run_white_test(residuals_clean, exog)
                results.append(white_result)
            except HeteroscedasticityTestError as e:
                logger.warning(f"White test skipped: {e.message}")

        return results

    except ImportError as e:
        raise ValueError(
            f"CRITICAL: Required statistical library unavailable. "
            f"Business impact: Cannot test heteroscedasticity assumptions. "
            f"Required action: Install statsmodels. "
            f"Original error: {e}"
        ) from e


def _compute_vif_severity(vif_value: float,
                         vif_threshold: float) -> Tuple[str, str, bool]:
    """
    Compute severity, interpretation, and violation flag for VIF value.

    Parameters
    ----------
    vif_value : float
        Computed Variance Inflation Factor
    vif_threshold : float
        Threshold for moderate multicollinearity

    Returns
    -------
    Tuple[str, str, bool]
        (severity, interpretation, assumption_violated)
    """
    if vif_value > 20:
        return "SEVERE", f"Severe multicollinearity (VIF={vif_value:.2f})", True
    elif vif_value > vif_threshold:
        return "MODERATE", f"Moderate multicollinearity (VIF={vif_value:.2f})", True
    elif vif_value > 5:
        return "MILD", f"Mild multicollinearity (VIF={vif_value:.2f})", False
    else:
        return "NONE", f"No multicollinearity concern (VIF={vif_value:.2f})", False


def _calculate_single_vif(exog_with_const: pd.DataFrame,
                         feature: str,
                         vif_threshold: float) -> Optional[DiagnosticResult]:
    """
    Calculate VIF for a single feature and return diagnostic result.

    Parameters
    ----------
    exog_with_const : pd.DataFrame
        Exogenous variables with constant term
    feature : str
        Feature name to calculate VIF for
    vif_threshold : float
        VIF threshold for concerning multicollinearity

    Returns
    -------
    Optional[DiagnosticResult]
        VIF result or None if calculation fails
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        feature_idx = exog_with_const.columns.get_loc(feature)
        vif_value = variance_inflation_factor(exog_with_const.values, feature_idx)

        severity, interpretation, violation = _compute_vif_severity(vif_value, vif_threshold)

        remediation = (f"Consider removing {feature} or using regularization"
                      if violation else "No action needed")

        return DiagnosticResult(
            test_name=f"VIF_{feature}",
            statistic=vif_value,
            p_value=np.nan,  # VIF doesn't have p-value
            critical_value=vif_threshold,
            assumption_violated=violation,
            severity=severity,
            interpretation=interpretation,
            remediation=remediation
        )
    except Exception as e:
        logger.warning(f"VIF calculation failed for {feature}: {e}")
        raise MulticollinearityError(
            f"VIF calculation failed for {feature}: {e}",
            vif_values={feature: float("nan")},
            business_impact=f"Cannot assess multicollinearity for feature '{feature}'",
            required_action="Check feature has sufficient variance and no missing values",
        ) from e


def check_multicollinearity(exog: pd.DataFrame,
                            vif_threshold: float = 10.0) -> List[DiagnosticResult]:
    """
    Test for multicollinearity using Variance Inflation Factor (VIF).

    High multicollinearity leads to unstable coefficient estimates,
    inflated standard errors, and difficulty interpreting individual
    feature effects.

    Parameters
    ----------
    exog : pd.DataFrame
        Exogenous variables (features)
    vif_threshold : float, default 10.0
        VIF threshold for concerning multicollinearity

    Returns
    -------
    List[DiagnosticResult]
        VIF results for each feature

    Mathematical Foundation:
    - VIFⱼ = 1/(1 - R²ⱼ) where R²ⱼ is from regressing Xⱼ on all other X's
    - VIF = 1: No multicollinearity
    - VIF > 5: Moderate multicollinearity
    - VIF > 10: Severe multicollinearity
    """
    try:
        # Add constant term if not present
        exog_with_const = exog.assign(const=1) if 'const' not in exog.columns else exog.copy()

        # Calculate VIF for each feature (excluding constant)
        feature_cols = [col for col in exog_with_const.columns if col != 'const']

        results: List[DiagnosticResult] = []
        for feature in feature_cols:
            try:
                vif_result = _calculate_single_vif(exog_with_const, feature, vif_threshold)
                results.append(vif_result)
            except MulticollinearityError as e:
                logger.warning(f"VIF calculation for '{feature}' skipped: {e.message}")

        return results

    except ImportError as e:
        raise ValueError(
            f"CRITICAL: Required statistical library unavailable. "
            f"Business impact: Cannot test multicollinearity. "
            f"Required action: Install statsmodels. "
            f"Original error: {e}"
        ) from e


def _compute_normality_severity(p_value: float) -> Tuple[str, str]:
    """
    Compute severity and interpretation for normality tests.

    Parameters
    ----------
    p_value : float
        Test p-value

    Returns
    -------
    Tuple[str, str]
        (severity, interpretation)
    """
    if p_value < 0.001:
        return "MODERATE", "Strong evidence of non-normality (p<0.001)"
    elif p_value < 0.05:
        return "MILD", f"Evidence of non-normality (p={p_value:.3f})"
    else:
        return "NONE", f"No evidence against normality (p={p_value:.3f})"


def _run_jarque_bera_test(residuals: np.ndarray) -> Optional[DiagnosticResult]:
    """
    Execute Jarque-Bera test for normality (good for large samples).

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals (NaN-cleaned)

    Returns
    -------
    Optional[DiagnosticResult]
        Test result or None if test fails
    """
    try:
        from scipy.stats import jarque_bera

        jb_statistic, jb_pvalue = jarque_bera(residuals)
        violation = jb_pvalue < 0.05
        severity, interpretation = _compute_normality_severity(jb_pvalue)

        return DiagnosticResult(
            test_name="Jarque-Bera",
            statistic=jb_statistic,
            p_value=jb_pvalue,
            critical_value=5.99,  # chi-sq(2) at 5%
            assumption_violated=violation,
            severity=severity,
            interpretation=interpretation,
            remediation="Use bootstrap inference or larger sample" if violation else "No action needed"
        )
    except Exception as e:
        logger.warning(f"Jarque-Bera test failed: {e}")
        raise NormalityTestError(
            f"Jarque-Bera test failed: {e}",
            test_name="Jarque-Bera",
            business_impact="Cannot assess residual normality for large samples",
            required_action="Check residuals have sufficient length and no infinite values",
        ) from e


def _run_shapiro_wilk_test(residuals: np.ndarray) -> Optional[DiagnosticResult]:
    """
    Execute Shapiro-Wilk test for normality (good for small samples, n <= 50).

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals (NaN-cleaned)

    Returns
    -------
    Optional[DiagnosticResult]
        Test result or None if test fails
    """
    try:
        from scipy.stats import shapiro

        sw_statistic, sw_pvalue = shapiro(residuals)
        violation = sw_pvalue < 0.05
        severity, interpretation = _compute_normality_severity(sw_pvalue)

        return DiagnosticResult(
            test_name="Shapiro-Wilk",
            statistic=sw_statistic,
            p_value=sw_pvalue,
            critical_value=0.95,  # Approximate
            assumption_violated=violation,
            severity=severity,
            interpretation=interpretation,
            remediation="Consider robust inference methods" if violation else "No action needed"
        )
    except Exception as e:
        logger.warning(f"Shapiro-Wilk test failed: {e}")
        raise NormalityTestError(
            f"Shapiro-Wilk test failed: {e}",
            test_name="Shapiro-Wilk",
            business_impact="Cannot assess residual normality for small samples",
            required_action="Check residuals have 3 to 5000 observations with no NaN values",
        ) from e


def check_normality(residuals: Union[np.ndarray, pd.Series]) -> List[DiagnosticResult]:
    """
    Test residual normality assumption using Jarque-Bera and Shapiro-Wilk tests.

    Non-normal residuals affect inference (confidence intervals, hypothesis tests)
    but don't bias coefficient estimates in large samples. Less critical for
    inference than autocorrelation or heteroscedasticity in finite samples.

    Parameters
    ----------
    residuals : Union[np.ndarray, pd.Series]
        Model residuals from OLS regression. NaN values are automatically removed
        before testing.

    Returns
    -------
    List[DiagnosticResult]
        List of normality test results. Includes:
        - Jarque-Bera test (asymptotic, good for n > 50)
        - Shapiro-Wilk test (exact, good for n <= 50)

    Raises
    ------
    ValueError
        If required statistical libraries (scipy) are unavailable.

    Notes
    -----
    Mathematical Foundation:
    - Jarque-Bera: JB = (n/6)(S² + (K-3)²/4) ~ χ²(2)
      where S is skewness and K is kurtosis
    - Shapiro-Wilk: W-statistic for goodness-of-fit to normal distribution
    - Null hypothesis for both: Residuals are normally distributed

    Test Selection:
    - Jarque-Bera: Preferred for large samples (n > 50)
    - Shapiro-Wilk: Better power for small samples (n <= 50)
    - Function automatically selects Shapiro-Wilk when n <= 50

    Remediation Options:
    - Bootstrap inference (robust to non-normality)
    - Larger sample size (normality less critical in large samples)
    - Transform variables if skewness/kurtosis extreme
    - Robust regression methods

    Examples
    --------
    >>> results = check_normality(residuals)
    >>> for r in results:
    ...     print(f"{r.test_name}: {r.severity}")
    """
    try:
        residuals_clean = np.asarray(residuals)
        residuals_clean = residuals_clean[~np.isnan(residuals_clean)]

        results: List[DiagnosticResult] = []

        # Jarque-Bera Test (good for large samples)
        try:
            jb_result = _run_jarque_bera_test(residuals_clean)
            results.append(jb_result)
        except NormalityTestError as e:
            logger.warning(f"Jarque-Bera test skipped: {e.message}")

        # Shapiro-Wilk Test (good for small samples, n <= 50)
        if len(residuals_clean) <= 50:
            try:
                sw_result = _run_shapiro_wilk_test(residuals_clean)
                results.append(sw_result)
            except NormalityTestError as e:
                logger.warning(f"Shapiro-Wilk test skipped: {e.message}")

        return results

    except ImportError as e:
        raise ValueError(
            f"CRITICAL: Required statistical library unavailable. "
            f"Business impact: Cannot test normality assumptions. "
            f"Required action: Install scipy. "
            f"Original error: {e}"
        ) from e


def _build_model_specification(model: Any,
                              data: pd.DataFrame,
                              target_variable: str,
                              features: List[str]) -> Dict[str, Any]:
    """
    Build model specification dictionary for diagnostic reporting.

    Parameters
    ----------
    model : statsmodels regression model
        Fitted OLS model
    data : pd.DataFrame
        Dataset used for modeling
    target_variable : str
        Target variable name
    features : List[str]
        Feature names in model

    Returns
    -------
    Dict[str, Any]
        Model specification metadata
    """
    return {
        'target_variable': target_variable,
        'features': features,
        'n_observations': len(data),
        'n_features': len(features),
        'model_r_squared': model.rsquared,
        'model_aic': model.aic,
        'diagnostic_timestamp': datetime.now().isoformat()
    }


def _run_all_diagnostic_tests(residuals: np.ndarray,
                             exog_data: pd.DataFrame) -> Tuple[
                                 List[DiagnosticResult],
                                 List[DiagnosticResult],
                                 List[DiagnosticResult],
                                 List[DiagnosticResult]]:
    """
    Run all diagnostic test categories and return results.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals
    exog_data : pd.DataFrame
        Exogenous variables (features)

    Returns
    -------
    Tuple of Lists
        (autocorr_tests, hetero_tests, multicol_tests, normality_tests)
    """
    autocorr_tests = check_autocorrelation(residuals, exog_data)
    hetero_tests = check_heteroscedasticity(residuals, exog_data)
    multicol_tests = check_multicollinearity(exog_data)
    normality_tests = check_normality(residuals)

    return autocorr_tests, hetero_tests, multicol_tests, normality_tests


def comprehensive_diagnostic_suite(model: Any,
                                 data: pd.DataFrame,
                                 target_variable: str,
                                 features: List[str]) -> ComprehensiveDiagnostics:
    """
    Run complete regression diagnostics suite for all OLS assumptions.

    Addresses Issue #8 from mathematical analysis: Missing regression diagnostics.
    Validates four critical OLS assumptions: autocorrelation, heteroscedasticity,
    multicollinearity, and normality. Returns comprehensive results with business
    interpretations and prioritized remediation plan.

    Parameters
    ----------
    model : statsmodels regression model
        Fitted OLS regression model with .resid and .rsquared attributes.
    data : pd.DataFrame
        Full dataset used to fit the model. Used for sample size tracking.
    target_variable : str
        Name of the target/dependent variable.
    features : List[str]
        List of feature names (exogenous variables) in the model.

    Returns
    -------
    ComprehensiveDiagnostics
        Dataclass container including:
        - model_specification: Metadata (n_observations, n_features, R², AIC)
        - autocorrelation_tests: Durbin-Watson and Ljung-Box results
        - heteroscedasticity_tests: Breusch-Pagan and White test results
        - multicollinearity_tests: VIF analysis for each feature
        - normality_tests: Jarque-Bera and Shapiro-Wilk results
        - overall_assessment: Validity rating and production readiness
        - remediation_plan: Prioritized action items (URGENT/HIGH/MODERATE)

    Raises
    ------
    ValueError
        If any diagnostic test suite fails or model lacks required attributes.

    Notes
    -----
    Diagnostic Priority (for production deployment):
    1. Autocorrelation: High impact on inference (invalid standard errors)
    2. Heteroscedasticity: High impact on inference (biased standard errors)
    3. Multicollinearity: Moderate impact (unstable coefficients, hard interpretation)
    4. Normality: Low impact in large samples (affects inference, not estimates)

    Production Readiness:
    - POOR validity: Severe violations present
    - CONCERNING: Multiple moderate violations
    - ACCEPTABLE WITH CAUTION: Some mild violations, recommend robust SE
    - GOOD: Assumptions largely satisfied

    Integration Workflow:
    >>> from src.features.selection.support.regression_diagnostics import (
    ...     comprehensive_diagnostic_suite
    ... )
    >>> diagnostics = comprehensive_diagnostic_suite(model, data, 'price', features)
    >>> if diagnostics.overall_assessment['production_ready']:
    ...     deploy_model()
    >>> else:
    ...     print("\\n".join(diagnostics.remediation_plan))

    See Also
    --------
    check_autocorrelation : Individual autocorrelation testing
    check_heteroscedasticity : Individual heteroscedasticity testing
    check_multicollinearity : Individual multicollinearity testing
    check_normality : Individual normality testing
    """
    try:
        residuals = model.resid
        exog_data = data[features]

        model_spec = _build_model_specification(model, data, target_variable, features)
        autocorr, hetero, multicol, normality = _run_all_diagnostic_tests(residuals, exog_data)

        overall = _assess_overall_model_validity(autocorr, hetero, multicol, normality)
        remediation = _generate_remediation_plan(autocorr, hetero, multicol, normality)

        return ComprehensiveDiagnostics(
            model_specification=model_spec,
            autocorrelation_tests=autocorr,
            heteroscedasticity_tests=hetero,
            multicollinearity_tests=multicol,
            normality_tests=normality,
            overall_assessment=overall,
            remediation_plan=remediation
        )

    except Exception as e:
        raise ValueError(
            f"CRITICAL: Comprehensive diagnostics failed. "
            f"Business impact: Cannot validate model assumptions for production. "
            f"Required action: Check model fitting and data consistency. "
            f"Original error: {e}"
        ) from e


def _assess_overall_model_validity(autocorr_tests: List[DiagnosticResult],
                                 hetero_tests: List[DiagnosticResult],
                                 multicol_tests: List[DiagnosticResult],
                                 normality_tests: List[DiagnosticResult]) -> Dict[str, Any]:
    """
    Assess overall model validity from individual diagnostic test results.

    Aggregates all diagnostic tests and produces validity rating, confidence level,
    production readiness determination, and summary statistics.

    Parameters
    ----------
    autocorr_tests : List[DiagnosticResult]
        Results from autocorrelation tests (Durbin-Watson, Ljung-Box)
    hetero_tests : List[DiagnosticResult]
        Results from heteroscedasticity tests (Breusch-Pagan, White)
    multicol_tests : List[DiagnosticResult]
        Results from multicollinearity tests (VIF analysis)
    normality_tests : List[DiagnosticResult]
        Results from normality tests (Jarque-Bera, Shapiro-Wilk)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - overall_validity: Rating (GOOD/ACCEPTABLE WITH CAUTION/CONCERNING/POOR)
        - statistical_confidence: Level (HIGH/MODERATE/LOW)
        - production_ready: Boolean flag for deployment readiness
        - primary_recommendation: Action string
        - violation_summary: Counts of violations by severity

    Notes
    -----
    Severity Counting Logic:
    - Severe violation: Blocks production (e.g., strong autocorrelation)
    - Moderate violation: Requires remediation (e.g., heteroscedasticity)
    - Mild violation: Monitor but may be acceptable (e.g., slight skewness)

    Decision Rules:
    - POOR: Any severe violation present → production_ready=False
    - CONCERNING: >2 moderate violations → production_ready=False
    - ACCEPTABLE WITH CAUTION: Some moderate/mild → production_ready=True
    - GOOD: No severe violations, <=1 moderate → production_ready=True

    Implementation Detail:
    Treats autocorrelation and heteroscedasticity more severely than
    multicollinearity (which affects interpretation more than inference)
    and normality (which has less impact in large samples).
    """
    all_tests = autocorr_tests + hetero_tests + multicol_tests + normality_tests

    # Count violations by severity
    severe_violations = sum(1 for test in all_tests if test.severity == "SEVERE")
    moderate_violations = sum(1 for test in all_tests if test.severity == "MODERATE")
    mild_violations = sum(1 for test in all_tests if test.severity == "MILD")
    total_violations = sum(1 for test in all_tests if test.assumption_violated)

    # Determine overall validity
    if severe_violations > 0:
        validity = "POOR"
        confidence = "LOW"
        production_ready = False
        recommendation = "Major assumption violations - model requires significant revision"
    elif moderate_violations > 2:
        validity = "CONCERNING"
        confidence = "MODERATE"
        production_ready = False
        recommendation = "Multiple assumption violations - consider remediation before production"
    elif moderate_violations > 0 or mild_violations > 3:
        validity = "ACCEPTABLE WITH CAUTION"
        confidence = "MODERATE"
        production_ready = True
        recommendation = "Some assumption violations - use robust standard errors"
    else:
        validity = "GOOD"
        confidence = "HIGH"
        production_ready = True
        recommendation = "Model assumptions largely satisfied"

    return {
        'overall_validity': validity,
        'statistical_confidence': confidence,
        'production_ready': production_ready,
        'primary_recommendation': recommendation,
        'violation_summary': {
            'severe_violations': severe_violations,
            'moderate_violations': moderate_violations,
            'mild_violations': mild_violations,
            'total_violations': total_violations,
            'total_tests': len(all_tests)
        }
    }


def _generate_remediation_plan(autocorr_tests: List[DiagnosticResult],
                             hetero_tests: List[DiagnosticResult],
                             multicol_tests: List[DiagnosticResult],
                             normality_tests: List[DiagnosticResult]) -> List[str]:
    """
    Generate prioritized remediation plan for assumption violations.

    Single responsibility: Remediation planning only.

    Parameters
    ----------
    autocorr_tests, hetero_tests, multicol_tests, normality_tests : List[DiagnosticResult]
        Diagnostic test results

    Returns
    -------
    List[str]
        Prioritized remediation actions
    """
    remediation_actions = []

    # Priority 1: Severe violations (production blockers)
    for test in autocorr_tests + hetero_tests + multicol_tests:
        if test.severity == "SEVERE":
            remediation_actions.append(f"URGENT: {test.remediation} (due to {test.test_name})")

    # Priority 2: Moderate violations (production concerns)
    for test in autocorr_tests + hetero_tests + multicol_tests:
        if test.severity == "MODERATE":
            remediation_actions.append(f"HIGH: {test.remediation} (due to {test.test_name})")

    # Priority 3: Mild violations (monitoring recommended)
    for test in autocorr_tests + hetero_tests + multicol_tests + normality_tests:
        if test.severity == "MILD":
            remediation_actions.append(f"MODERATE: {test.remediation} (due to {test.test_name})")

    # Add general recommendations if no specific violations
    if not remediation_actions:
        remediation_actions.append("All major assumptions satisfied - consider robust standard errors as precaution")

    return remediation_actions
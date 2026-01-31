"""
Tests for src/features/selection/pipeline_orchestrator.py

Comprehensive test coverage for pipeline orchestration functions:
- Input validation functions
- Pipeline summary creation
- Error result generation
- Model selection logic

Target: 12% → 60%+ coverage
"""

import pytest
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock

from src.features.selection.pipeline_orchestrator import (
    # Validation functions
    _validate_target_variable,
    _validate_candidate_features,
    _validate_base_features,
    _validate_config_settings,
    _validate_data_quality,
    validate_pipeline_inputs,
    # Summary functions
    _build_execution_metrics,
    _build_best_model_metrics,
    _build_constraint_metrics,
    _build_bootstrap_metrics,
    _build_business_interpretation,
    create_pipeline_summary,
    # Error handling
    _create_error_result,
    # Model selection
    _select_best_model,
)
from src.features.selection_types import (
    FeatureSelectionConfig,
    EconomicConstraintConfig,
    AICResult,
    FeatureSelectionResults,
    BootstrapResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Standard test DataFrame with target and features."""
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100, freq='W'),
        'sales_target_current': np.random.uniform(50000, 150000, 100),
        'competitor_mid_t2': np.random.uniform(2.0, 4.0, 100),
        'competitor_top5_t3': np.random.uniform(3.0, 5.0, 100),
        'prudential_rate_current': np.random.uniform(1.5, 3.5, 100),
    })


@pytest.fixture
def small_data() -> pd.DataFrame:
    """Small DataFrame (< 50 rows)."""
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=30, freq='W'),
        'sales_target_current': np.random.uniform(50000, 150000, 30),
        'competitor_mid_t2': np.random.uniform(2.0, 4.0, 30),
    })


@pytest.fixture
def feature_config() -> FeatureSelectionConfig:
    """Standard feature selection config."""
    return FeatureSelectionConfig(
        base_features=['prudential_rate_current'],
        candidate_features=['competitor_mid_t2', 'competitor_top5_t3'],
        max_candidate_features=2,
        target_variable='sales_target_current',
    )


@pytest.fixture
def constraint_config() -> EconomicConstraintConfig:
    """Standard constraint config."""
    return EconomicConstraintConfig(
        enabled=True,
        strict_validation=True,
    )


@pytest.fixture
def aic_result() -> AICResult:
    """Standard AIC result."""
    return AICResult(
        features='competitor_mid_t2 + prudential_rate_current',
        n_features=2,
        aic=500.0,
        bic=510.0,
        r_squared=0.75,
        r_squared_adj=0.73,
        coefficients={
            'competitor_mid_t2': -0.5,
            'prudential_rate_current': 0.8,
        },
        converged=True,
        n_obs=100,
    )


@pytest.fixture
def feature_selection_results(
    aic_result: AICResult,
    feature_config: FeatureSelectionConfig,
    constraint_config: EconomicConstraintConfig,
) -> FeatureSelectionResults:
    """Standard feature selection results."""
    return FeatureSelectionResults(
        best_model=aic_result,
        all_results=pd.DataFrame({
            'features': ['f1', 'f2', 'f3'],
            'aic': [500.0, 520.0, 530.0],
            'converged': [True, True, False],
        }),
        valid_results=pd.DataFrame({
            'features': ['f1', 'f2'],
            'aic': [500.0, 520.0],
        }),
        total_combinations=3,
        converged_models=2,
        economically_valid_models=2,
        constraint_violations=[],
        feature_config=feature_config,
        constraint_config=constraint_config,
        execution_time_seconds=5.5,
    )


@pytest.fixture
def results_with_bootstrap(
    feature_selection_results: FeatureSelectionResults,
) -> FeatureSelectionResults:
    """Results with bootstrap analysis."""
    bootstrap_results = [
        BootstrapResult(
            model_name='model_1',
            model_features='f1 + f2',
            bootstrap_aics=[498.0, 502.0, 500.0] * 10,
            bootstrap_r2_values=[0.74, 0.76, 0.75] * 10,
            original_aic=500.0,
            original_r2=0.75,
            aic_stability_coefficient=0.05,
            r2_stability_coefficient=0.02,
            confidence_intervals={
                'aic': {'5%': 498.0, '95%': 502.0},
                'r2': {'5%': 0.74, '95%': 0.76},
            },
            successful_fits=30,
            total_attempts=30,
            stability_assessment='STABLE',
        ),
        BootstrapResult(
            model_name='model_2',
            model_features='f1 + f3',
            bootstrap_aics=[520.0, 525.0, 522.0] * 10,
            bootstrap_r2_values=[0.70, 0.72, 0.71] * 10,
            original_aic=520.0,
            original_r2=0.71,
            aic_stability_coefficient=0.08,
            r2_stability_coefficient=0.03,
            confidence_intervals={
                'aic': {'5%': 518.0, '95%': 525.0},
                'r2': {'5%': 0.69, '95%': 0.73},
            },
            successful_fits=30,
            total_attempts=30,
            stability_assessment='MODERATE',
        ),
    ]
    feature_selection_results.bootstrap_results = bootstrap_results
    return feature_selection_results


# =============================================================================
# Tests for _validate_target_variable
# =============================================================================


class TestValidateTargetVariable:
    """Tests for _validate_target_variable."""

    @patch('src.validation.input_validators.validate_target_with_warnings')
    def test_delegates_to_input_validators(self, mock_validate, sample_data):
        """1.1: Delegates to canonical validator."""
        mock_validate.return_value = []
        result = _validate_target_variable(sample_data, 'sales_target_current')

        mock_validate.assert_called_once_with(
            sample_data, 'sales_target_current', require_numeric=True
        )
        assert result == []

    @patch('src.validation.input_validators.validate_target_with_warnings')
    def test_returns_warnings_from_validator(self, mock_validate, sample_data):
        """1.2: Returns warnings from validator."""
        mock_validate.return_value = ['WARNING: Target has NaN values']
        result = _validate_target_variable(sample_data, 'sales_target_current')

        assert len(result) == 1
        assert 'NaN' in result[0]


# =============================================================================
# Tests for _validate_candidate_features
# =============================================================================


class TestValidateCandidateFeatures:
    """Tests for _validate_candidate_features."""

    def test_all_features_present(self, sample_data):
        """2.1: Returns no warnings when all candidates present."""
        candidates = ['competitor_mid_t2', 'competitor_top5_t3']
        warnings, available = _validate_candidate_features(sample_data, candidates)

        assert warnings == []
        assert set(available) == set(candidates)

    def test_missing_features_warning(self, sample_data):
        """2.2: Returns warning for missing features."""
        candidates = ['competitor_mid_t2', 'nonexistent_feature']
        warnings, available = _validate_candidate_features(sample_data, candidates)

        assert len(warnings) == 1
        assert 'missing' in warnings[0].lower()
        assert 'competitor_mid_t2' in available
        assert 'nonexistent_feature' not in available

    def test_all_features_missing_critical(self, sample_data):
        """2.3: Returns CRITICAL warning when no candidates found."""
        candidates = ['nonexistent1', 'nonexistent2']
        warnings, available = _validate_candidate_features(sample_data, candidates)

        assert len(warnings) == 1
        assert 'CRITICAL' in warnings[0]
        assert available == []


# =============================================================================
# Tests for _validate_base_features
# =============================================================================


class TestValidateBaseFeatures:
    """Tests for _validate_base_features."""

    def test_all_base_features_present(self, sample_data):
        """3.1: Returns no warnings when all base features present."""
        base = ['prudential_rate_current', 'competitor_mid_t2']
        warnings = _validate_base_features(sample_data, base)

        assert warnings == []

    def test_missing_base_features_critical(self, sample_data):
        """3.2: Returns CRITICAL warning for missing base features."""
        base = ['prudential_rate_current', 'missing_base']
        warnings = _validate_base_features(sample_data, base)

        assert len(warnings) == 1
        assert 'CRITICAL' in warnings[0]
        assert 'missing_base' in warnings[0]


# =============================================================================
# Tests for _validate_config_settings
# =============================================================================


class TestValidateConfigSettings:
    """Tests for _validate_config_settings."""

    def test_valid_settings(self):
        """4.1: Returns no warnings for valid settings."""
        warnings = _validate_config_settings(max_candidates=2, available_candidates_count=5)
        assert warnings == []

    def test_max_candidates_less_than_one(self):
        """4.2: Warning when max_candidates < 1."""
        warnings = _validate_config_settings(max_candidates=0, available_candidates_count=5)

        assert len(warnings) == 1
        assert 'should be >= 1' in warnings[0]

    def test_max_candidates_exceeds_available(self):
        """4.3: Warning when max_candidates exceeds available."""
        warnings = _validate_config_settings(max_candidates=10, available_candidates_count=5)

        assert len(warnings) == 1
        assert 'exceeds available' in warnings[0]


# =============================================================================
# Tests for _validate_data_quality
# =============================================================================


class TestValidateDataQuality:
    """Tests for _validate_data_quality."""

    def test_sufficient_rows(self, sample_data):
        """5.1: No warning for >= 50 rows."""
        warnings = _validate_data_quality(sample_data)
        assert warnings == []

    def test_small_dataset_warning(self, small_data):
        """5.2: Warning for < 50 rows."""
        warnings = _validate_data_quality(small_data)

        assert len(warnings) == 1
        assert 'Small dataset' in warnings[0]


# =============================================================================
# Tests for validate_pipeline_inputs
# =============================================================================


class TestValidatePipelineInputs:
    """Tests for validate_pipeline_inputs."""

    def test_empty_dataset_critical(self, feature_config, constraint_config):
        """6.1: Returns CRITICAL for empty dataset."""
        empty_df = pd.DataFrame()
        warnings = validate_pipeline_inputs(empty_df, feature_config, constraint_config)

        assert len(warnings) == 1
        assert 'CRITICAL' in warnings[0]
        assert 'Empty dataset' in warnings[0]

    def test_valid_inputs_no_warnings(self, sample_data, feature_config, constraint_config):
        """6.2: Returns empty list for valid inputs."""
        # Patch the target validator since it calls external module
        with patch('src.validation.input_validators.validate_target_with_warnings') as mock:
            mock.return_value = []
            warnings = validate_pipeline_inputs(sample_data, feature_config, constraint_config)

        assert warnings == []

    def test_aggregates_all_warnings(self, small_data, constraint_config):
        """6.3: Aggregates warnings from all validators."""
        config = FeatureSelectionConfig(
            base_features=['missing_base'],
            candidate_features=['nonexistent1', 'nonexistent2'],
            max_candidate_features=10,
            target_variable='sales_target_current',
        )

        with patch('src.validation.input_validators.validate_target_with_warnings') as mock:
            mock.return_value = []
            warnings = validate_pipeline_inputs(small_data, config, constraint_config)

        # Should have warnings from multiple validators
        assert len(warnings) >= 2  # At minimum: missing base, no candidates found


# =============================================================================
# Tests for _build_execution_metrics
# =============================================================================


class TestBuildExecutionMetrics:
    """Tests for _build_execution_metrics."""

    def test_builds_correct_structure(self, feature_selection_results):
        """7.1: Returns correct keys."""
        metrics = _build_execution_metrics(feature_selection_results)

        expected_keys = {
            'total_combinations_evaluated',
            'models_converged',
            'economically_valid_models',
            'execution_time_seconds',
            'success_rate',
        }
        assert set(metrics.keys()) == expected_keys

    def test_calculates_success_rate(self, feature_selection_results):
        """7.2: Calculates success rate correctly."""
        metrics = _build_execution_metrics(feature_selection_results)

        # 2 converged / 3 total = 66.7%
        assert '66.7%' in metrics['success_rate']

    def test_zero_combinations_handles_division(self, feature_config, constraint_config, aic_result):
        """7.3: Handles zero combinations without division error."""
        results = FeatureSelectionResults(
            best_model=aic_result,
            all_results=pd.DataFrame(),
            valid_results=pd.DataFrame(),
            total_combinations=0,
            converged_models=0,
            economically_valid_models=0,
            constraint_violations=[],
            feature_config=feature_config,
            constraint_config=constraint_config,
        )
        metrics = _build_execution_metrics(results)

        assert metrics['success_rate'] == '0%'


# =============================================================================
# Tests for _build_best_model_metrics
# =============================================================================


class TestBuildBestModelMetrics:
    """Tests for _build_best_model_metrics."""

    def test_builds_correct_structure(self, feature_selection_results):
        """8.1: Returns correct keys."""
        metrics = _build_best_model_metrics(feature_selection_results)

        expected_keys = {
            'features', 'n_features', 'aic_score', 'r_squared', 'model_fit_quality'
        }
        assert set(metrics.keys()) == expected_keys

    def test_excellent_fit_quality(self, feature_selection_results):
        """8.2: R² > 0.7 → Excellent."""
        metrics = _build_best_model_metrics(feature_selection_results)
        assert metrics['model_fit_quality'] == 'Excellent'

    def test_good_fit_quality(self, feature_config, constraint_config):
        """8.3: 0.5 < R² <= 0.7 → Good."""
        result = AICResult(
            features='f1', n_features=1, aic=500.0, bic=510.0,
            r_squared=0.6, r_squared_adj=0.58,
            coefficients={}, converged=True, n_obs=100
        )
        results = FeatureSelectionResults(
            best_model=result,
            all_results=pd.DataFrame(),
            valid_results=pd.DataFrame(),
            total_combinations=1,
            converged_models=1,
            economically_valid_models=1,
            constraint_violations=[],
            feature_config=feature_config,
            constraint_config=constraint_config,
        )
        metrics = _build_best_model_metrics(results)
        assert metrics['model_fit_quality'] == 'Good'

    def test_moderate_fit_quality(self, feature_config, constraint_config):
        """8.4: R² <= 0.5 → Moderate."""
        result = AICResult(
            features='f1', n_features=1, aic=500.0, bic=510.0,
            r_squared=0.4, r_squared_adj=0.38,
            coefficients={}, converged=True, n_obs=100
        )
        results = FeatureSelectionResults(
            best_model=result,
            all_results=pd.DataFrame(),
            valid_results=pd.DataFrame(),
            total_combinations=1,
            converged_models=1,
            economically_valid_models=1,
            constraint_violations=[],
            feature_config=feature_config,
            constraint_config=constraint_config,
        )
        metrics = _build_best_model_metrics(results)
        assert metrics['model_fit_quality'] == 'Moderate'


# =============================================================================
# Tests for _build_constraint_metrics
# =============================================================================


class TestBuildConstraintMetrics:
    """Tests for _build_constraint_metrics."""

    def test_builds_correct_structure(self, feature_selection_results):
        """9.1: Returns correct keys."""
        metrics = _build_constraint_metrics(feature_selection_results)

        expected_keys = {
            'constraints_enabled', 'total_violations', 'constraint_compliance_rate'
        }
        assert set(metrics.keys()) == expected_keys

    def test_calculates_compliance_rate(self, feature_selection_results):
        """9.2: Calculates compliance rate correctly."""
        metrics = _build_constraint_metrics(feature_selection_results)

        # 2 valid / 2 converged = 100%
        assert '100.0%' in metrics['constraint_compliance_rate']

    def test_zero_converged_handles_division(self, feature_config, constraint_config, aic_result):
        """9.3: Handles zero converged without division error."""
        results = FeatureSelectionResults(
            best_model=aic_result,
            all_results=pd.DataFrame(),
            valid_results=pd.DataFrame(),
            total_combinations=3,
            converged_models=0,
            economically_valid_models=0,
            constraint_violations=[],
            feature_config=feature_config,
            constraint_config=constraint_config,
        )
        metrics = _build_constraint_metrics(results)

        assert '0' in metrics['constraint_compliance_rate']


# =============================================================================
# Tests for _build_bootstrap_metrics
# =============================================================================


class TestBuildBootstrapMetrics:
    """Tests for _build_bootstrap_metrics."""

    def test_returns_none_without_bootstrap(self, feature_selection_results):
        """10.1: Returns None when no bootstrap results."""
        metrics = _build_bootstrap_metrics(feature_selection_results)
        assert metrics is None

    def test_builds_correct_structure_with_bootstrap(self, results_with_bootstrap):
        """10.2: Returns correct keys with bootstrap."""
        metrics = _build_bootstrap_metrics(results_with_bootstrap)

        expected_keys = {
            'models_analyzed', 'stable_models', 'stability_rate', 'top_model_stability'
        }
        assert set(metrics.keys()) == expected_keys

    def test_counts_stable_models(self, results_with_bootstrap):
        """10.3: Correctly counts stable models."""
        metrics = _build_bootstrap_metrics(results_with_bootstrap)

        # 1 STABLE, 1 MODERATE
        assert metrics['stable_models'] == 1
        assert metrics['models_analyzed'] == 2


# =============================================================================
# Tests for _build_business_interpretation
# =============================================================================


class TestBuildBusinessInterpretation:
    """Tests for _build_business_interpretation."""

    def test_excellent_fit_interpretation(self, feature_selection_results):
        """11.1: Excellent fit generates strong model message."""
        interpretation = _build_business_interpretation(feature_selection_results, 0)

        assert any('Strong predictive' in msg or 'excellent' in msg.lower()
                   for msg in interpretation)

    def test_high_compliance_interpretation(self, feature_selection_results):
        """11.2: High compliance generates positive constraint message."""
        interpretation = _build_business_interpretation(feature_selection_results, 0)

        assert any('High economic' in msg or 'constraint compliance' in msg.lower()
                   for msg in interpretation)

    def test_bootstrap_stable_interpretation(self, results_with_bootstrap):
        """11.3: Stable bootstrap generates stability message."""
        interpretation = _build_business_interpretation(results_with_bootstrap, 1)

        assert any('stability' in msg.lower() for msg in interpretation)


# =============================================================================
# Tests for create_pipeline_summary
# =============================================================================


class TestCreatePipelineSummary:
    """Tests for create_pipeline_summary."""

    def test_summary_structure(self, feature_selection_results):
        """12.1: Returns expected top-level keys."""
        summary = create_pipeline_summary(feature_selection_results)

        expected_keys = {
            'pipeline_execution', 'best_model', 'economic_constraints',
            'business_interpretation'
        }
        assert expected_keys.issubset(set(summary.keys()))

    def test_includes_bootstrap_when_available(self, results_with_bootstrap):
        """12.2: Includes bootstrap_analysis when bootstrap results present."""
        summary = create_pipeline_summary(results_with_bootstrap)

        assert 'bootstrap_analysis' in summary

    def test_handles_exception_gracefully(self, feature_config, constraint_config):
        """12.3: Returns error summary on exception."""
        # Create a result that will cause an error
        bad_result = FeatureSelectionResults(
            best_model=None,  # This should cause AttributeError
            all_results=pd.DataFrame(),
            valid_results=pd.DataFrame(),
            total_combinations=0,
            converged_models=0,
            economically_valid_models=0,
            constraint_violations=[],
            feature_config=feature_config,
            constraint_config=constraint_config,
        )
        summary = create_pipeline_summary(bad_result)

        assert 'error' in summary


# =============================================================================
# Tests for _create_error_result
# =============================================================================


class TestCreateErrorResult:
    """Tests for _create_error_result."""

    def test_creates_valid_result_structure(self, feature_config, constraint_config):
        """13.1: Returns valid FeatureSelectionResults."""
        result = _create_error_result(
            error_type='TEST_ERROR',
            error_msg='Test error message',
            data_len=100,
            feature_config=feature_config,
            constraint_config=constraint_config,
            execution_time=1.5,
        )

        assert isinstance(result, FeatureSelectionResults)
        assert result.best_model.features == 'TEST_ERROR'
        assert result.best_model.error == 'Test error message'
        assert result.best_model.converged is False

    def test_includes_all_results_when_provided(self, feature_config, constraint_config):
        """13.2: Includes all_results DataFrame when provided."""
        all_results = pd.DataFrame({'aic': [500, 520], 'converged': [True, False]})

        result = _create_error_result(
            error_type='TEST_ERROR',
            error_msg='Test error',
            data_len=100,
            feature_config=feature_config,
            constraint_config=constraint_config,
            execution_time=1.5,
            all_results=all_results,
        )

        assert len(result.all_results) == 2
        assert result.total_combinations == 2


# =============================================================================
# Tests for _select_best_model
# =============================================================================


class TestSelectBestModel:
    """Tests for _select_best_model."""

    def test_selects_lowest_aic_from_valid(self):
        """14.1: Selects model with lowest AIC from valid results."""
        valid_results = pd.DataFrame({
            'features': ['model_1', 'model_2', 'model_3'],
            'n_features': [2, 3, 2],
            'aic': [520.0, 500.0, 510.0],  # model_2 has lowest
            'bic': [530.0, 510.0, 520.0],
            'r_squared': [0.7, 0.75, 0.72],
            'r_squared_adj': [0.68, 0.73, 0.70],
            'coefficients': [{'a': 1}, {'b': 2}, {'c': 3}],
            'converged': [True, True, True],
            'n_obs': [100, 100, 100],
        })

        converged_results = valid_results.copy()

        best = _select_best_model(valid_results, converged_results)

        assert best.features == 'model_2'
        assert best.aic == 500.0
        assert best.error is None

    def test_falls_back_to_converged_when_no_valid(self):
        """14.2: Uses converged results when valid is empty."""
        valid_results = pd.DataFrame(columns=[
            'features', 'n_features', 'aic', 'bic', 'r_squared',
            'r_squared_adj', 'coefficients', 'converged', 'n_obs'
        ])

        converged_results = pd.DataFrame({
            'features': ['model_1', 'model_2'],
            'n_features': [2, 3],
            'aic': [500.0, 520.0],
            'bic': [510.0, 530.0],
            'r_squared': [0.7, 0.65],
            'r_squared_adj': [0.68, 0.63],
            'coefficients': [{'a': 1}, {'b': 2}],
            'converged': [True, True],
            'n_obs': [100, 100],
        })

        # Capture print output
        import io
        import sys
        captured = io.StringIO()
        sys.stdout = captured

        best = _select_best_model(valid_results, converged_results)

        sys.stdout = sys.__stdout__

        assert best.features == 'model_1'
        assert best.error == "Best model violates economic constraints"
        assert 'WARNING' in captured.getvalue()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for pipeline orchestrator."""

    def test_full_summary_workflow(self, results_with_bootstrap):
        """15.1: Full workflow from results to summary."""
        summary = create_pipeline_summary(results_with_bootstrap)

        # Verify all sections present
        assert 'pipeline_execution' in summary
        assert 'best_model' in summary
        assert 'economic_constraints' in summary
        assert 'bootstrap_analysis' in summary
        assert 'business_interpretation' in summary

        # Verify business interpretation is non-empty
        assert len(summary['business_interpretation']) > 0

    def test_error_result_generates_summary(self, feature_config, constraint_config):
        """15.2: Error result can generate summary."""
        error_result = _create_error_result(
            error_type='VALIDATION_FAILED',
            error_msg='Missing target variable',
            data_len=0,
            feature_config=feature_config,
            constraint_config=constraint_config,
            execution_time=0.1,
        )

        # Should not raise
        summary = create_pipeline_summary(error_result)

        # Summary should indicate failure
        assert summary['pipeline_execution']['total_combinations_evaluated'] == 0


# =============================================================================
# Tests for Print and Enhancement Functions
# =============================================================================


class TestPrintPipelineSummary:
    """Tests for _print_pipeline_summary."""

    def test_prints_summary_without_bootstrap(self, feature_selection_results, capsys):
        """16.1: Prints summary without bootstrap section."""
        from src.features.selection.pipeline_orchestrator import _print_pipeline_summary

        summary = create_pipeline_summary(feature_selection_results)
        _print_pipeline_summary(summary, execution_time=5.5, has_bootstrap=False)

        captured = capsys.readouterr()
        assert 'Pipeline Execution Summary' in captured.out
        assert 'Best Model' in captured.out
        assert 'Business Insights' in captured.out

    def test_prints_summary_with_bootstrap(self, results_with_bootstrap, capsys):
        """16.2: Prints stability info when bootstrap available."""
        from src.features.selection.pipeline_orchestrator import _print_pipeline_summary

        summary = create_pipeline_summary(results_with_bootstrap)
        _print_pipeline_summary(summary, execution_time=5.5, has_bootstrap=True)

        captured = capsys.readouterr()
        assert 'Stability' in captured.out


class TestEnhancementFunctions:
    """Tests for enhancement phase functions (feature flag controlled)."""

    def test_search_space_reduction_disabled(self, feature_config, sample_data):
        """17.1: Returns original config when flag disabled."""
        from src.features.selection.pipeline_orchestrator import (
            _run_search_space_reduction
        )
        from src.features.selection.interface.interface_config import FEATURE_FLAGS

        # Ensure flag is disabled
        original_flag = FEATURE_FLAGS.get("ENABLE_SEARCH_SPACE_REDUCTION", False)
        FEATURE_FLAGS["ENABLE_SEARCH_SPACE_REDUCTION"] = False

        try:
            result = _run_search_space_reduction(feature_config, sample_data)
            # Should return same config object when disabled
            assert result == feature_config
        finally:
            FEATURE_FLAGS["ENABLE_SEARCH_SPACE_REDUCTION"] = original_flag

    def test_multiple_testing_disabled(self):
        """17.2: Returns original results when flag disabled."""
        from src.features.selection.pipeline_orchestrator import (
            _run_multiple_testing_correction
        )
        from src.features.selection.interface.interface_config import FEATURE_FLAGS

        original_flag = FEATURE_FLAGS.get("ENABLE_MULTIPLE_TESTING", False)
        FEATURE_FLAGS["ENABLE_MULTIPLE_TESTING"] = False

        converged = pd.DataFrame({'aic': [500, 520], 'pvalue': [0.01, 0.05]})

        try:
            result = _run_multiple_testing_correction(converged)
            pd.testing.assert_frame_equal(result, converged)
        finally:
            FEATURE_FLAGS["ENABLE_MULTIPLE_TESTING"] = original_flag

    def test_oos_validation_disabled(self, sample_data):
        """17.3: Returns None when OOS validation disabled."""
        from src.features.selection.pipeline_orchestrator import (
            _run_oos_validation
        )
        from src.features.selection.interface.interface_config import FEATURE_FLAGS

        original_flag = FEATURE_FLAGS.get("ENABLE_OOS_VALIDATION", False)
        FEATURE_FLAGS["ENABLE_OOS_VALIDATION"] = False

        valid_results = pd.DataFrame({'features': ['f1'], 'aic': [500]})

        try:
            result = _run_oos_validation(sample_data, valid_results, 'sales_target_current')
            assert result is None
        finally:
            FEATURE_FLAGS["ENABLE_OOS_VALIDATION"] = original_flag

    def test_block_bootstrap_disabled(self, sample_data):
        """17.4: Returns None when block bootstrap disabled."""
        from src.features.selection.pipeline_orchestrator import (
            _run_block_bootstrap
        )
        from src.features.selection.interface.interface_config import FEATURE_FLAGS

        original_flag = FEATURE_FLAGS.get("ENABLE_BLOCK_BOOTSTRAP", False)
        FEATURE_FLAGS["ENABLE_BLOCK_BOOTSTRAP"] = False

        valid_results = pd.DataFrame({'features': ['f1'], 'aic': [500]})
        bootstrap_config = {'enabled': True, 'n_samples': 100}

        try:
            result = _run_block_bootstrap(
                sample_data, valid_results, bootstrap_config, 'sales_target_current'
            )
            # Returns None when block bootstrap not enabled
            assert result is None
        finally:
            FEATURE_FLAGS["ENABLE_BLOCK_BOOTSTRAP"] = original_flag

    def test_block_bootstrap_no_config(self, sample_data):
        """17.5: Returns None when bootstrap config is None."""
        from src.features.selection.pipeline_orchestrator import (
            _run_block_bootstrap
        )

        valid_results = pd.DataFrame({'features': ['f1'], 'aic': [500]})

        result = _run_block_bootstrap(
            sample_data, valid_results, None, 'sales_target_current'
        )
        assert result is None

    def test_regression_diagnostics_disabled(self, sample_data, aic_result):
        """17.6: Returns None when regression diagnostics disabled."""
        from src.features.selection.pipeline_orchestrator import (
            _run_regression_diagnostics
        )
        from src.features.selection.interface.interface_config import FEATURE_FLAGS

        original_flag = FEATURE_FLAGS.get("ENABLE_REGRESSION_DIAGNOSTICS", False)
        FEATURE_FLAGS["ENABLE_REGRESSION_DIAGNOSTICS"] = False

        try:
            result = _run_regression_diagnostics(
                sample_data, aic_result, 'sales_target_current'
            )
            assert result is None
        finally:
            FEATURE_FLAGS["ENABLE_REGRESSION_DIAGNOSTICS"] = original_flag

    def test_statistical_constraints_disabled(self):
        """17.7: Returns None when statistical constraints disabled."""
        from src.features.selection.pipeline_orchestrator import (
            _run_statistical_constraints
        )
        from src.features.selection.interface.interface_config import FEATURE_FLAGS

        original_flag = FEATURE_FLAGS.get("ENABLE_STATISTICAL_CONSTRAINTS", False)
        FEATURE_FLAGS["ENABLE_STATISTICAL_CONSTRAINTS"] = False

        valid_results = pd.DataFrame({'features': ['f1'], 'aic': [500]})

        try:
            result = _run_statistical_constraints(valid_results)
            assert result is None
        finally:
            FEATURE_FLAGS["ENABLE_STATISTICAL_CONSTRAINTS"] = original_flag


class TestCorePipelineFunctions:
    """Tests for core pipeline phase functions."""

    def test_bootstrap_analysis_none_config(self, sample_data):
        """18.1: Returns None when bootstrap config is None."""
        from src.features.selection.pipeline_orchestrator import (
            _run_bootstrap_analysis
        )

        valid_results = pd.DataFrame({'features': ['f1'], 'aic': [500]})

        result = _run_bootstrap_analysis(
            sample_data, valid_results, None, 'sales_target_current'
        )
        assert result is None

    def test_bootstrap_analysis_disabled(self, sample_data):
        """18.2: Returns None when bootstrap disabled in config."""
        from src.features.selection.pipeline_orchestrator import (
            _run_bootstrap_analysis
        )

        valid_results = pd.DataFrame({'features': ['f1'], 'aic': [500]})
        config = {'enabled': False, 'n_samples': 100, 'models_to_analyze': 5}

        result = _run_bootstrap_analysis(
            sample_data, valid_results, config, 'sales_target_current'
        )
        assert result is None


class TestBusinessInterpretationEdgeCases:
    """Additional tests for business interpretation edge cases."""

    def test_low_compliance_interpretation(self, feature_config, constraint_config):
        """19.1: Low compliance generates warning message."""
        from src.features.selection.pipeline_orchestrator import (
            _build_business_interpretation
        )

        result = AICResult(
            features='f1', n_features=1, aic=500.0, bic=510.0,
            r_squared=0.6, r_squared_adj=0.58,
            coefficients={}, converged=True, n_obs=100
        )
        results = FeatureSelectionResults(
            best_model=result,
            all_results=pd.DataFrame(),
            valid_results=pd.DataFrame(),
            total_combinations=10,
            converged_models=10,
            economically_valid_models=2,  # Low compliance
            constraint_violations=[],
            feature_config=feature_config,
            constraint_config=constraint_config,
        )

        interpretation = _build_business_interpretation(results, 0)

        assert any('violate' in msg.lower() or 'review' in msg.lower()
                   for msg in interpretation)

    def test_moderate_fit_interpretation(self, feature_config, constraint_config):
        """19.2: Moderate fit suggests feature engineering."""
        from src.features.selection.pipeline_orchestrator import (
            _build_business_interpretation
        )

        result = AICResult(
            features='f1', n_features=1, aic=500.0, bic=510.0,
            r_squared=0.4, r_squared_adj=0.38,
            coefficients={}, converged=True, n_obs=100
        )
        results = FeatureSelectionResults(
            best_model=result,
            all_results=pd.DataFrame(),
            valid_results=pd.DataFrame(),
            total_combinations=1,
            converged_models=1,
            economically_valid_models=1,
            constraint_violations=[],
            feature_config=feature_config,
            constraint_config=constraint_config,
        )

        interpretation = _build_business_interpretation(results, 0)

        assert any('moderate' in msg.lower() or 'feature engineering' in msg.lower()
                   for msg in interpretation)

    def test_unstable_bootstrap_interpretation(self, feature_config, constraint_config):
        """19.3: Unstable bootstrap warns about instability."""
        from src.features.selection.pipeline_orchestrator import (
            _build_business_interpretation
        )

        result = AICResult(
            features='f1', n_features=1, aic=500.0, bic=510.0,
            r_squared=0.75, r_squared_adj=0.73,
            coefficients={}, converged=True, n_obs=100
        )
        results = FeatureSelectionResults(
            best_model=result,
            all_results=pd.DataFrame(),
            valid_results=pd.DataFrame(),
            total_combinations=1,
            converged_models=1,
            economically_valid_models=1,
            constraint_violations=[],
            feature_config=feature_config,
            constraint_config=constraint_config,
            bootstrap_results=[
                BootstrapResult(
                    model_name='m1', model_features='f1',
                    bootstrap_aics=[500, 600, 700],
                    bootstrap_r2_values=[0.5, 0.7, 0.9],
                    original_aic=500, original_r2=0.7,
                    aic_stability_coefficient=0.3,
                    r2_stability_coefficient=0.3,
                    confidence_intervals={},
                    successful_fits=3, total_attempts=3,
                    stability_assessment='UNSTABLE'
                )
            ],
        )

        # 0 stable models with bootstrap results present
        interpretation = _build_business_interpretation(results, 0)

        assert any('instability' in msg.lower() or 'larger dataset' in msg.lower()
                   for msg in interpretation)


# =============================================================================
# Tests for Core Pipeline Execution Functions
# =============================================================================


class TestRunAicEvaluation:
    """Tests for _run_aic_evaluation."""

    @patch('src.features.selection.pipeline_orchestrator.evaluate_aic_combinations')
    def test_returns_all_and_converged_results(self, mock_evaluate, feature_config, sample_data):
        """20.1: Returns tuple of (all_results, converged_results)."""
        from src.features.selection.pipeline_orchestrator import _run_aic_evaluation

        mock_results = pd.DataFrame({
            'features': ['f1', 'f2', 'f3'],
            'aic': [500.0, 520.0, np.inf],
            'converged': [True, True, False],
        })
        mock_evaluate.return_value = mock_results

        all_results, converged = _run_aic_evaluation(sample_data, feature_config)

        assert len(all_results) == 3
        assert len(converged) == 2
        assert all(converged['converged'])

    @patch('src.features.selection.pipeline_orchestrator.evaluate_aic_combinations')
    def test_prints_summary(self, mock_evaluate, feature_config, sample_data, capsys):
        """20.2: Prints AIC evaluation summary."""
        from src.features.selection.pipeline_orchestrator import _run_aic_evaluation

        mock_results = pd.DataFrame({
            'features': ['f1'],
            'aic': [500.0],
            'converged': [True],
        })
        mock_evaluate.return_value = mock_results

        _run_aic_evaluation(sample_data, feature_config)

        captured = capsys.readouterr()
        assert 'Phase 2' in captured.out
        assert 'AIC Evaluation' in captured.out


class TestRunConstraintValidation:
    """Tests for _run_constraint_validation."""

    @patch('src.features.selection.pipeline_orchestrator.apply_economic_constraints')
    def test_returns_valid_results_and_violations(self, mock_apply, constraint_config):
        """21.1: Returns tuple of (valid_results, violations)."""
        from src.features.selection.pipeline_orchestrator import _run_constraint_validation

        converged = pd.DataFrame({
            'features': ['f1', 'f2'],
            'aic': [500.0, 520.0],
        })
        mock_valid = pd.DataFrame({'features': ['f1'], 'aic': [500.0]})
        mock_violations = []
        mock_apply.return_value = (mock_valid, mock_violations)

        valid, violations = _run_constraint_validation(converged, constraint_config)

        assert len(valid) == 1
        assert violations == []

    @patch('src.features.selection.pipeline_orchestrator.apply_economic_constraints')
    def test_prints_phase_header(self, mock_apply, constraint_config, capsys):
        """21.2: Prints Phase 3 header."""
        from src.features.selection.pipeline_orchestrator import _run_constraint_validation

        mock_apply.return_value = (pd.DataFrame(), [])
        converged = pd.DataFrame({'features': []})

        _run_constraint_validation(converged, constraint_config)

        captured = capsys.readouterr()
        assert 'Phase 3' in captured.out


class TestCheckCriticalValidationFailures:
    """Tests for _check_critical_validation_failures."""

    def test_returns_none_when_no_critical_failures(
        self, sample_data, feature_config, constraint_config
    ):
        """22.1: Returns None when no critical failures."""
        from src.features.selection.pipeline_orchestrator import (
            _check_critical_validation_failures
        )
        import time

        with patch('src.validation.input_validators.validate_target_with_warnings') as mock:
            mock.return_value = []
            result = _check_critical_validation_failures(
                sample_data, feature_config, constraint_config, time.time()
            )

        assert result is None

    def test_returns_error_result_on_critical_failure(
        self, feature_config, constraint_config
    ):
        """22.2: Returns error result on critical failure."""
        from src.features.selection.pipeline_orchestrator import (
            _check_critical_validation_failures
        )
        import time

        empty_df = pd.DataFrame()
        result = _check_critical_validation_failures(
            empty_df, feature_config, constraint_config, time.time()
        )

        assert result is not None
        assert isinstance(result, FeatureSelectionResults)
        assert 'VALIDATION_FAILED' in result.best_model.features


class TestFinalizeAndReport:
    """Tests for _finalize_and_report."""

    def test_prints_success_message(self, feature_selection_results, capsys):
        """23.1: Prints success message."""
        from src.features.selection.pipeline_orchestrator import _finalize_and_report

        _finalize_and_report(feature_selection_results, 5.0, False)

        captured = capsys.readouterr()
        assert 'SUCCESS' in captured.out
        assert 'Complete' in captured.out

    def test_returns_results(self, feature_selection_results):
        """23.2: Returns the results unchanged."""
        from src.features.selection.pipeline_orchestrator import _finalize_and_report

        result = _finalize_and_report(feature_selection_results, 5.0, False)

        assert result is feature_selection_results


class TestCompileFinalResults:
    """Tests for _compile_final_results."""

    def test_compiles_all_fields(self, aic_result, feature_config, constraint_config):
        """24.1: Compiles all fields into FeatureSelectionResults."""
        from src.features.selection.pipeline_orchestrator import _compile_final_results

        all_results = pd.DataFrame({'aic': [500, 520]})
        valid_results = pd.DataFrame({'aic': [500]})
        converged_results = pd.DataFrame({'aic': [500, 520]})

        result = _compile_final_results(
            best_model=aic_result,
            all_results=all_results,
            valid_results=valid_results,
            converged_results=converged_results,
            constraint_violations=[],
            feature_config=feature_config,
            constraint_config=constraint_config,
            bootstrap_results=None,
            bootstrap_config=None,
            experiment_config=None,
            execution_time=5.0,
        )

        assert isinstance(result, FeatureSelectionResults)
        assert result.best_model == aic_result
        assert result.total_combinations == 2
        assert result.converged_models == 2
        assert result.economically_valid_models == 1
        assert result.execution_time_seconds == 5.0


class TestRunFeatureSelectionPipeline:
    """Tests for run_feature_selection_pipeline main function."""

    @patch('src.features.selection.pipeline_orchestrator._execute_pipeline_phases')
    @patch('src.features.selection.pipeline_orchestrator._check_critical_validation_failures')
    def test_returns_validation_error_on_critical_failure(
        self, mock_check, mock_execute, sample_data, feature_config, constraint_config
    ):
        """25.1: Returns error result on critical validation failure."""
        from src.features.selection.pipeline_orchestrator import run_feature_selection_pipeline

        # Mock critical failure
        mock_error = FeatureSelectionResults(
            best_model=AICResult(
                features='VALIDATION_FAILED', n_features=0, aic=np.inf, bic=np.inf,
                r_squared=0.0, r_squared_adj=0.0, coefficients={},
                converged=False, n_obs=0, error='Critical failure'
            ),
            all_results=pd.DataFrame(),
            valid_results=pd.DataFrame(),
            total_combinations=0,
            converged_models=0,
            economically_valid_models=0,
            constraint_violations=[],
            feature_config=feature_config,
            constraint_config=constraint_config,
        )
        mock_check.return_value = mock_error

        result = run_feature_selection_pipeline(
            sample_data, feature_config, constraint_config
        )

        assert 'VALIDATION_FAILED' in result.best_model.features
        mock_execute.assert_not_called()

    @patch('src.features.selection.pipeline_orchestrator._execute_pipeline_phases')
    @patch('src.features.selection.pipeline_orchestrator._check_critical_validation_failures')
    def test_executes_pipeline_when_validation_passes(
        self, mock_check, mock_execute, sample_data, feature_config, constraint_config
    ):
        """25.2: Executes pipeline when validation passes."""
        from src.features.selection.pipeline_orchestrator import run_feature_selection_pipeline

        mock_check.return_value = None  # No critical failures
        mock_result = MagicMock()
        mock_execute.return_value = mock_result

        result = run_feature_selection_pipeline(
            sample_data, feature_config, constraint_config
        )

        mock_execute.assert_called_once()
        assert result == mock_result

    def test_handles_exception_gracefully(self, feature_config, constraint_config, capsys):
        """25.3: Returns error result on exception."""
        from src.features.selection.pipeline_orchestrator import run_feature_selection_pipeline

        # Empty DataFrame will cause issues
        with patch('src.features.selection.pipeline_orchestrator.validate_pipeline_inputs') as mock:
            mock.side_effect = Exception("Test exception")

            result = run_feature_selection_pipeline(
                pd.DataFrame(), feature_config, constraint_config
            )

        assert 'PIPELINE_ERROR' in result.best_model.features
        assert result.best_model.converged is False

        captured = capsys.readouterr()
        assert 'ERROR' in captured.out

    def test_prints_dataset_info(self, sample_data, feature_config, constraint_config, capsys):
        """25.4: Prints dataset info at start."""
        from src.features.selection.pipeline_orchestrator import run_feature_selection_pipeline

        # Will fail during validation but should print dataset info first
        with patch('src.features.selection.pipeline_orchestrator.validate_pipeline_inputs') as mock:
            mock.return_value = ['CRITICAL: Test failure']

            run_feature_selection_pipeline(sample_data, feature_config, constraint_config)

        captured = capsys.readouterr()
        assert 'Dataset' in captured.out
        assert '100 rows' in captured.out


class TestExecutePipelinePhases:
    """Tests for _execute_pipeline_phases."""

    @patch('src.features.selection.pipeline_orchestrator._finalize_and_report')
    @patch('src.features.selection.pipeline_orchestrator._compile_final_results')
    @patch('src.features.selection.pipeline_orchestrator._select_best_model')
    @patch('src.features.selection.pipeline_orchestrator._run_bootstrap_analysis')
    @patch('src.features.selection.pipeline_orchestrator._run_block_bootstrap')
    @patch('src.features.selection.pipeline_orchestrator._run_statistical_constraints')
    @patch('src.features.selection.pipeline_orchestrator._run_oos_validation')
    @patch('src.features.selection.pipeline_orchestrator._run_constraint_validation')
    @patch('src.features.selection.pipeline_orchestrator._run_multiple_testing_correction')
    @patch('src.features.selection.pipeline_orchestrator._run_aic_evaluation')
    @patch('src.features.selection.pipeline_orchestrator._run_search_space_reduction')
    @patch('src.features.selection.pipeline_orchestrator._run_regression_diagnostics')
    def test_executes_all_phases(
        self,
        mock_diagnostics, mock_search, mock_aic, mock_mtc, mock_constraint,
        mock_oos, mock_stat, mock_block, mock_bootstrap, mock_select,
        mock_compile, mock_finalize,
        sample_data, feature_config, constraint_config
    ):
        """26.1: Executes all phases in sequence."""
        from src.features.selection.pipeline_orchestrator import _execute_pipeline_phases
        import time

        # Setup mocks
        mock_search.return_value = feature_config
        converged = pd.DataFrame({
            'features': ['f1'], 'aic': [500.0], 'converged': [True],
            'n_features': [2], 'r_squared': [0.7], 'r_squared_adj': [0.68],
            'coefficients': [{}], 'n_obs': [100]
        })
        mock_aic.return_value = (converged, converged)
        mock_mtc.return_value = converged
        mock_constraint.return_value = (converged, [])
        mock_oos.return_value = None
        mock_stat.return_value = None
        mock_block.return_value = None
        mock_bootstrap.return_value = None
        mock_select.return_value = MagicMock()
        mock_diagnostics.return_value = None
        mock_compile.return_value = MagicMock()
        mock_finalize.return_value = MagicMock()

        _execute_pipeline_phases(
            sample_data, feature_config, constraint_config,
            None, None, time.time()
        )

        mock_search.assert_called_once()
        mock_aic.assert_called_once()
        mock_mtc.assert_called_once()
        mock_constraint.assert_called_once()
        mock_select.assert_called_once()
        mock_compile.assert_called_once()
        mock_finalize.assert_called_once()

    @patch('src.features.selection.pipeline_orchestrator._run_search_space_reduction')
    @patch('src.features.selection.pipeline_orchestrator._run_aic_evaluation')
    def test_returns_error_when_no_models_converge(
        self, mock_aic, mock_search,
        sample_data, feature_config, constraint_config
    ):
        """26.2: Returns error result when no models converge."""
        from src.features.selection.pipeline_orchestrator import _execute_pipeline_phases
        import time

        mock_search.return_value = feature_config
        all_results = pd.DataFrame({
            'features': ['f1'], 'aic': [np.inf], 'converged': [False]
        })
        converged = pd.DataFrame()  # Empty - no converged models
        mock_aic.return_value = (all_results, converged)

        result = _execute_pipeline_phases(
            sample_data, feature_config, constraint_config,
            None, None, time.time()
        )

        assert 'NO_CONVERGED_MODELS' in result.best_model.features


class TestEnhancementEnabledPaths:
    """Tests for enhancement functions when flags are enabled."""

    @patch('src.features.selection.pipeline_orchestrator.reduce_search_space')
    def test_search_space_reduction_enabled(self, mock_reduce, feature_config, sample_data):
        """27.1: Calls reduce_search_space when enabled."""
        from src.features.selection.pipeline_orchestrator import _run_search_space_reduction
        from src.features.selection.interface.interface_config import FEATURE_FLAGS

        original = FEATURE_FLAGS.get("ENABLE_SEARCH_SPACE_REDUCTION", False)
        FEATURE_FLAGS["ENABLE_SEARCH_SPACE_REDUCTION"] = True

        try:
            mock_reduce.return_value = ['competitor_mid_t2']

            result = _run_search_space_reduction(feature_config, sample_data)

            mock_reduce.assert_called_once()
            assert 'competitor_mid_t2' in result['candidate_features']
        finally:
            FEATURE_FLAGS["ENABLE_SEARCH_SPACE_REDUCTION"] = original

    @patch('src.features.selection.pipeline_orchestrator.reduce_search_space')
    def test_search_space_reduction_handles_failure(self, mock_reduce, feature_config, sample_data, capsys):
        """27.2: Falls back to original on failure."""
        from src.features.selection.pipeline_orchestrator import _run_search_space_reduction
        from src.features.selection.interface.interface_config import FEATURE_FLAGS

        original = FEATURE_FLAGS.get("ENABLE_SEARCH_SPACE_REDUCTION", False)
        FEATURE_FLAGS["ENABLE_SEARCH_SPACE_REDUCTION"] = True

        try:
            mock_reduce.side_effect = Exception("Test error")

            result = _run_search_space_reduction(feature_config, sample_data)

            assert result == feature_config  # Returns original

            captured = capsys.readouterr()
            assert 'WARNING' in captured.out
        finally:
            FEATURE_FLAGS["ENABLE_SEARCH_SPACE_REDUCTION"] = original

    @patch('src.features.selection.pipeline_orchestrator.apply_multiple_testing_correction')
    def test_multiple_testing_enabled(self, mock_mtc):
        """27.3: Calls apply_multiple_testing_correction when enabled."""
        from src.features.selection.pipeline_orchestrator import _run_multiple_testing_correction
        from src.features.selection.interface.interface_config import FEATURE_FLAGS

        original = FEATURE_FLAGS.get("ENABLE_MULTIPLE_TESTING", False)
        FEATURE_FLAGS["ENABLE_MULTIPLE_TESTING"] = True

        try:
            converged = pd.DataFrame({'aic': [500], 'pvalue': [0.01]})
            corrected = pd.DataFrame({'aic': [500], 'pvalue': [0.05], 'pvalue_corrected': [0.05]})
            mock_mtc.return_value = corrected

            result = _run_multiple_testing_correction(converged)

            mock_mtc.assert_called_once()
            assert 'pvalue_corrected' in result.columns
        finally:
            FEATURE_FLAGS["ENABLE_MULTIPLE_TESTING"] = original

    @patch('src.features.selection.pipeline_orchestrator.apply_multiple_testing_correction')
    def test_multiple_testing_handles_failure(self, mock_mtc, capsys):
        """27.4: Falls back to original on failure."""
        from src.features.selection.pipeline_orchestrator import _run_multiple_testing_correction
        from src.features.selection.interface.interface_config import FEATURE_FLAGS

        original = FEATURE_FLAGS.get("ENABLE_MULTIPLE_TESTING", False)
        FEATURE_FLAGS["ENABLE_MULTIPLE_TESTING"] = True

        try:
            converged = pd.DataFrame({'aic': [500], 'pvalue': [0.01]})
            mock_mtc.side_effect = Exception("Test error")

            result = _run_multiple_testing_correction(converged)

            pd.testing.assert_frame_equal(result, converged)

            captured = capsys.readouterr()
            assert 'WARNING' in captured.out
        finally:
            FEATURE_FLAGS["ENABLE_MULTIPLE_TESTING"] = original

    @patch('src.features.selection.pipeline_orchestrator.evaluate_temporal_generalization')
    def test_oos_validation_enabled(self, mock_oos, sample_data):
        """27.5: Calls evaluate_temporal_generalization when enabled."""
        from src.features.selection.pipeline_orchestrator import _run_oos_validation
        from src.features.selection.interface.interface_config import FEATURE_FLAGS

        original = FEATURE_FLAGS.get("ENABLE_OOS_VALIDATION", False)
        FEATURE_FLAGS["ENABLE_OOS_VALIDATION"] = True

        try:
            valid_results = pd.DataFrame({'features': ['f1']})
            mock_oos.return_value = {'oos_r2': 0.7}

            result = _run_oos_validation(sample_data, valid_results, 'sales_target_current')

            mock_oos.assert_called_once()
            assert result == {'oos_r2': 0.7}
        finally:
            FEATURE_FLAGS["ENABLE_OOS_VALIDATION"] = original

    @patch('src.features.selection.pipeline_orchestrator.run_block_bootstrap_stability')
    def test_block_bootstrap_enabled(self, mock_block, sample_data):
        """27.6: Calls run_block_bootstrap_stability when enabled."""
        from src.features.selection.pipeline_orchestrator import _run_block_bootstrap
        from src.features.selection.interface.interface_config import FEATURE_FLAGS

        original = FEATURE_FLAGS.get("ENABLE_BLOCK_BOOTSTRAP", False)
        FEATURE_FLAGS["ENABLE_BLOCK_BOOTSTRAP"] = True

        try:
            valid_results = pd.DataFrame({'features': ['f1']})
            bootstrap_config = {'enabled': True, 'n_samples': 100, 'block_size': 4}
            mock_block.return_value = [{'stable': True}]

            result = _run_block_bootstrap(
                sample_data, valid_results, bootstrap_config, 'sales_target_current'
            )

            mock_block.assert_called_once()
            assert result == [{'stable': True}]
        finally:
            FEATURE_FLAGS["ENABLE_BLOCK_BOOTSTRAP"] = original

    @patch('src.features.selection.pipeline_orchestrator.apply_statistical_constraints')
    def test_statistical_constraints_enabled(self, mock_stat):
        """27.7: Calls apply_statistical_constraints when enabled."""
        from src.features.selection.pipeline_orchestrator import _run_statistical_constraints
        from src.features.selection.interface.interface_config import FEATURE_FLAGS

        original = FEATURE_FLAGS.get("ENABLE_STATISTICAL_CONSTRAINTS", False)
        FEATURE_FLAGS["ENABLE_STATISTICAL_CONSTRAINTS"] = True

        try:
            valid_results = pd.DataFrame({'features': ['f1']})
            mock_stat.return_value = {'constrained': 1}

            result = _run_statistical_constraints(valid_results)

            mock_stat.assert_called_once()
            assert result == {'constrained': 1}
        finally:
            FEATURE_FLAGS["ENABLE_STATISTICAL_CONSTRAINTS"] = original

    @patch('src.features.selection.pipeline_orchestrator.comprehensive_diagnostic_suite')
    def test_regression_diagnostics_enabled(self, mock_diag, sample_data, aic_result):
        """27.8: Calls comprehensive_diagnostic_suite when enabled."""
        from src.features.selection.pipeline_orchestrator import _run_regression_diagnostics
        from src.features.selection.interface.interface_config import FEATURE_FLAGS

        original = FEATURE_FLAGS.get("ENABLE_REGRESSION_DIAGNOSTICS", False)
        FEATURE_FLAGS["ENABLE_REGRESSION_DIAGNOSTICS"] = True

        # Add required columns to sample_data
        sample_data['competitor_mid_t2'] = np.random.uniform(2.0, 4.0, len(sample_data))
        sample_data['prudential_rate_current'] = np.random.uniform(1.5, 3.5, len(sample_data))

        try:
            mock_diag.return_value = {'passed': True}

            result = _run_regression_diagnostics(
                sample_data, aic_result, 'sales_target_current'
            )

            mock_diag.assert_called_once()
            assert result == {'passed': True}
        finally:
            FEATURE_FLAGS["ENABLE_REGRESSION_DIAGNOSTICS"] = original

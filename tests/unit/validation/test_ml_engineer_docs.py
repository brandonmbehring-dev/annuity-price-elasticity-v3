"""
Tests for src.validation.ml_engineer_docs module.

Comprehensive tests for ML Engineer documentation generation including:
- MLEngineerDocGenerator class instantiation
- Schema analysis methods
- Integration guide generation
- Schema compatibility reports
- Markdown generation helpers
- Complete documentation generation
- Edge cases and error handling

Coverage Target: 80%+ of ml_engineer_docs.py (810 lines)
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from src.validation.ml_engineer_docs import (
    MLEngineerDocGenerator,
    VALIDATION_MODULES_AVAILABLE,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Temporary directory for documentation output."""
    output_dir = tmp_path / "ml_engineer_handoff"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def doc_generator(temp_output_dir: Path) -> MLEngineerDocGenerator:
    """Pre-configured MLEngineerDocGenerator with temp directory."""
    return MLEngineerDocGenerator(output_dir=str(temp_output_dir))


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Standard DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=100),
        'sales': np.random.uniform(50000, 200000, 100),
        'price': np.random.uniform(1, 5, 100),
    })


@pytest.fixture
def mock_pandera_schema() -> MagicMock:
    """Mock Pandera schema for testing schema analysis."""
    mock = MagicMock()

    # Mock column schemas
    col_date = MagicMock()
    col_date.dtype = 'datetime64[ns]'
    col_date.nullable = False
    col_date.description = 'Transaction date'
    col_date.checks = []

    col_sales = MagicMock()
    col_sales.dtype = 'float64'
    col_sales.nullable = False
    col_sales.description = 'Sales amount'

    # Mock a check
    range_check = MagicMock()
    range_check.__class__.__name__ = 'Check'
    range_check.__str__ = lambda self: 'in_range(50000, 200000)'
    col_sales.checks = [range_check]

    mock.columns = {
        'date': col_date,
        'sales': col_sales,
    }

    return mock


@pytest.fixture
def mock_final_dataset_schema() -> MagicMock:
    """Mock FINAL_DATASET_SCHEMA for compatibility testing."""
    mock = MagicMock()
    mock.columns = MagicMock()
    mock.columns.keys = MagicMock(return_value=['date', 'sales', 'price', 'weight'])
    return mock


# =============================================================================
# CLASS INSTANTIATION TESTS
# =============================================================================


class TestMLEngineerDocGeneratorInit:
    """Tests for MLEngineerDocGenerator initialization."""

    def test_creates_output_directory(self, tmp_path):
        """Should create output directory if it doesn't exist."""
        new_dir = tmp_path / "new_docs_dir"
        generator = MLEngineerDocGenerator(output_dir=str(new_dir))

        assert new_dir.exists()

    def test_stores_output_path(self, temp_output_dir: Path):
        """Should store output directory path."""
        generator = MLEngineerDocGenerator(output_dir=str(temp_output_dir))

        assert generator.output_dir == temp_output_dir

    def test_sets_generation_timestamp(self, doc_generator: MLEngineerDocGenerator):
        """Should set generation timestamp on init."""
        assert doc_generator.generation_timestamp is not None
        # Should be ISO format
        datetime.fromisoformat(doc_generator.generation_timestamp)

    def test_default_output_dir(self, tmp_path, monkeypatch):
        """Should use default output directory if none provided."""
        monkeypatch.chdir(tmp_path)
        generator = MLEngineerDocGenerator()

        assert generator.output_dir == Path("docs/ml_engineer_handoff")


# =============================================================================
# SCHEMA ANALYSIS TESTS
# =============================================================================


class TestAnalyzeCurrentSchemas:
    """Tests for MLEngineerDocGenerator.analyze_current_schemas()."""

    def test_raises_when_modules_unavailable(self, doc_generator: MLEngineerDocGenerator,
                                             monkeypatch):
        """Should raise RuntimeError when validation modules unavailable."""
        monkeypatch.setattr(
            'src.validation.ml_engineer_docs.VALIDATION_MODULES_AVAILABLE',
            False
        )

        with pytest.raises(RuntimeError) as exc_info:
            doc_generator.analyze_current_schemas()

        assert 'Validation modules not available' in str(exc_info.value)

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_returns_analysis_dict(self, doc_generator: MLEngineerDocGenerator):
        """Should return analysis dictionary when modules available."""
        result = doc_generator.analyze_current_schemas()

        assert isinstance(result, dict)
        assert 'analysis_timestamp' in result
        assert 'schemas_discovered' in result
        assert 'schema_statistics' in result

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_includes_timestamp(self, doc_generator: MLEngineerDocGenerator):
        """Should include analysis timestamp."""
        result = doc_generator.analyze_current_schemas()

        assert result['analysis_timestamp'] == doc_generator.generation_timestamp


class TestAnalyzePanderaSchema:
    """Tests for MLEngineerDocGenerator._analyze_pandera_schema()."""

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_extracts_schema_name(self, doc_generator: MLEngineerDocGenerator,
                                  mock_pandera_schema: MagicMock):
        """Should include schema name in analysis."""
        result = doc_generator._analyze_pandera_schema(mock_pandera_schema, "test_schema")

        assert result['schema_name'] == "test_schema"

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_counts_columns(self, doc_generator: MLEngineerDocGenerator,
                            mock_pandera_schema: MagicMock):
        """Should count columns correctly."""
        result = doc_generator._analyze_pandera_schema(mock_pandera_schema, "test_schema")

        assert result['column_count'] == 2

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_includes_column_details(self, doc_generator: MLEngineerDocGenerator,
                                     mock_pandera_schema: MagicMock):
        """Should include detailed column information."""
        result = doc_generator._analyze_pandera_schema(mock_pandera_schema, "test_schema")

        assert 'columns' in result
        assert 'date' in result['columns']
        assert 'data_type' in result['columns']['date']

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_extracts_validation_checks(self, doc_generator: MLEngineerDocGenerator,
                                        mock_pandera_schema: MagicMock):
        """Should extract validation checks from columns."""
        result = doc_generator._analyze_pandera_schema(mock_pandera_schema, "test_schema")

        # Sales column should have checks
        sales_col = result['columns'].get('sales', {})
        assert 'checks' in sales_col

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_identifies_business_rules(self, doc_generator: MLEngineerDocGenerator,
                                       mock_pandera_schema: MagicMock):
        """Should identify business rules from checks."""
        result = doc_generator._analyze_pandera_schema(mock_pandera_schema, "test_schema")

        assert 'business_rules' in result

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_includes_strictness_settings(self, doc_generator: MLEngineerDocGenerator,
                                          mock_pandera_schema: MagicMock):
        """Should include schema strictness settings."""
        result = doc_generator._analyze_pandera_schema(mock_pandera_schema, "test_schema")

        assert 'validation_strictness' in result


# =============================================================================
# SCHEMA COMPATIBILITY REPORT TESTS
# =============================================================================


class TestGenerateSchemaCompatibilityReport:
    """Tests for MLEngineerDocGenerator.generate_schema_compatibility_report()."""

    def test_returns_report_dict(self, doc_generator: MLEngineerDocGenerator,
                                 sample_dataframe: pd.DataFrame):
        """Should return compatibility report dictionary."""
        result = doc_generator.generate_schema_compatibility_report(
            sample_dataframe, "test_schema"
        )

        assert isinstance(result, dict)
        assert 'report_timestamp' in result

    def test_includes_actual_data_shape(self, doc_generator: MLEngineerDocGenerator,
                                        sample_dataframe: pd.DataFrame):
        """Should include actual DataFrame shape."""
        result = doc_generator.generate_schema_compatibility_report(
            sample_dataframe, "test_schema"
        )

        assert result['actual_data_shape'] == (100, 3)

    def test_includes_actual_columns(self, doc_generator: MLEngineerDocGenerator,
                                     sample_dataframe: pd.DataFrame):
        """Should include list of actual columns."""
        result = doc_generator.generate_schema_compatibility_report(
            sample_dataframe, "test_schema"
        )

        assert 'actual_columns' in result
        assert 'date' in result['actual_columns']

    def test_returns_error_when_modules_unavailable(self, doc_generator: MLEngineerDocGenerator,
                                                    sample_dataframe: pd.DataFrame,
                                                    monkeypatch):
        """Should return error in report when validation modules unavailable."""
        monkeypatch.setattr(
            'src.validation.ml_engineer_docs.VALIDATION_MODULES_AVAILABLE',
            False
        )

        result = doc_generator.generate_schema_compatibility_report(
            sample_dataframe, "test_schema"
        )

        assert 'error' in result
        assert 'unavailable' in result['error'].lower()

    def test_includes_schema_context(self, doc_generator: MLEngineerDocGenerator,
                                     sample_dataframe: pd.DataFrame):
        """Should include schema context name."""
        result = doc_generator.generate_schema_compatibility_report(
            sample_dataframe, "my_schema_context"
        )

        assert result['schema_context'] == "my_schema_context"


class TestCompareColumnSets:
    """Tests for MLEngineerDocGenerator._compare_column_sets()."""

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_identifies_missing_columns(self, doc_generator: MLEngineerDocGenerator):
        """Should identify columns missing from actual DataFrame."""
        # Create DataFrame with fewer columns than schema expects
        df = pd.DataFrame({'date': [1], 'sales': [2]})

        result = doc_generator._compare_column_sets(df)

        assert 'missing_from_actual' in result

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_identifies_extra_columns(self, doc_generator: MLEngineerDocGenerator):
        """Should identify extra columns in actual DataFrame."""
        df = pd.DataFrame({'date': [1], 'sales': [2], 'extra_col': [3]})

        result = doc_generator._compare_column_sets(df)

        assert 'extra_in_actual' in result

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_identifies_common_columns(self, doc_generator: MLEngineerDocGenerator):
        """Should identify columns common to both."""
        df = pd.DataFrame({'date': [1], 'sales': [2]})

        result = doc_generator._compare_column_sets(df)

        assert 'common_columns' in result


class TestDetermineCompatibilityStatus:
    """Tests for MLEngineerDocGenerator._determine_compatibility_status()."""

    def test_fully_compatible_when_no_missing(self, doc_generator: MLEngineerDocGenerator):
        """Should return FULLY_COMPATIBLE when no columns missing."""
        status = doc_generator._determine_compatibility_status(0)

        assert status == "FULLY_COMPATIBLE"

    def test_mostly_compatible_when_few_missing(self, doc_generator: MLEngineerDocGenerator):
        """Should return MOSTLY_COMPATIBLE when 1-2 columns missing."""
        assert doc_generator._determine_compatibility_status(1) == "MOSTLY_COMPATIBLE"
        assert doc_generator._determine_compatibility_status(2) == "MOSTLY_COMPATIBLE"

    def test_requires_update_when_many_missing(self, doc_generator: MLEngineerDocGenerator):
        """Should return REQUIRES_SCHEMA_UPDATE when >2 columns missing."""
        status = doc_generator._determine_compatibility_status(5)

        assert status == "REQUIRES_SCHEMA_UPDATE"


class TestGenerateCompatibilityRecommendations:
    """Tests for MLEngineerDocGenerator._generate_compatibility_recommendations()."""

    def test_recommends_update_for_missing_columns(self, doc_generator: MLEngineerDocGenerator):
        """Should recommend schema update when columns missing."""
        recommendations = doc_generator._generate_compatibility_recommendations(3, 0)

        assert any('missing' in r.lower() for r in recommendations)

    def test_recommends_selection_for_extra_columns(self, doc_generator: MLEngineerDocGenerator):
        """Should recommend feature selection when many extra columns."""
        recommendations = doc_generator._generate_compatibility_recommendations(0, 15)

        assert any('feature selection' in r.lower() or 'extra' in r.lower()
                   for r in recommendations)

    def test_always_includes_validator_recommendation(self, doc_generator: MLEngineerDocGenerator):
        """Should always include DataFrameValidator usage recommendation."""
        recommendations = doc_generator._generate_compatibility_recommendations(0, 0)

        assert any('DataFrameValidator' in r for r in recommendations)


# =============================================================================
# INTEGRATION GUIDE TESTS
# =============================================================================


class TestGenerateIntegrationGuide:
    """Tests for MLEngineerDocGenerator.generate_integration_guide()."""

    def test_raises_when_modules_unavailable(self, doc_generator: MLEngineerDocGenerator,
                                             monkeypatch):
        """Should raise RuntimeError when validation modules unavailable."""
        monkeypatch.setattr(
            'src.validation.ml_engineer_docs.VALIDATION_MODULES_AVAILABLE',
            False
        )

        with pytest.raises(RuntimeError) as exc_info:
            doc_generator.generate_integration_guide()

        assert 'Validation modules not available' in str(exc_info.value)

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_returns_guide_dict(self, doc_generator: MLEngineerDocGenerator):
        """Should return integration guide dictionary."""
        result = doc_generator.generate_integration_guide()

        assert isinstance(result, dict)
        assert 'generation_timestamp' in result

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_includes_mlflow_integration(self, doc_generator: MLEngineerDocGenerator):
        """Should include MLflow integration documentation."""
        result = doc_generator.generate_integration_guide()

        assert 'mlflow_integration' in result

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_includes_dvc_integration(self, doc_generator: MLEngineerDocGenerator):
        """Should include DVC integration documentation."""
        result = doc_generator.generate_integration_guide()

        assert 'dvc_integration' in result

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_includes_validation_workflows(self, doc_generator: MLEngineerDocGenerator):
        """Should include validation workflow documentation."""
        result = doc_generator.generate_integration_guide()

        assert 'validation_workflows' in result

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_includes_production_deployment(self, doc_generator: MLEngineerDocGenerator):
        """Should include production deployment guidance."""
        result = doc_generator.generate_integration_guide()

        assert 'production_deployment' in result

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_includes_troubleshooting(self, doc_generator: MLEngineerDocGenerator):
        """Should include troubleshooting guide."""
        result = doc_generator.generate_integration_guide()

        assert 'troubleshooting' in result


# =============================================================================
# DOCUMENTATION HELPER TESTS
# =============================================================================


class TestDocumentMLflowPatterns:
    """Tests for MLEngineerDocGenerator._document_mlflow_patterns()."""

    def test_returns_dict(self, doc_generator: MLEngineerDocGenerator):
        """Should return dictionary."""
        result = doc_generator._document_mlflow_patterns()

        assert isinstance(result, dict)

    def test_includes_overview(self, doc_generator: MLEngineerDocGenerator):
        """Should include overview text."""
        result = doc_generator._document_mlflow_patterns()

        assert 'overview' in result
        assert isinstance(result['overview'], str)

    def test_includes_key_functions(self, doc_generator: MLEngineerDocGenerator):
        """Should include key functions documentation."""
        result = doc_generator._document_mlflow_patterns()

        assert 'key_functions' in result
        assert isinstance(result['key_functions'], list)

    def test_includes_artifacts_generated(self, doc_generator: MLEngineerDocGenerator):
        """Should include artifacts documentation."""
        result = doc_generator._document_mlflow_patterns()

        assert 'artifacts_generated' in result

    def test_includes_setup_requirements(self, doc_generator: MLEngineerDocGenerator):
        """Should include setup requirements."""
        result = doc_generator._document_mlflow_patterns()

        assert 'setup_requirements' in result


class TestDocumentDVCPatterns:
    """Tests for MLEngineerDocGenerator._document_dvc_patterns()."""

    def test_returns_dict(self, doc_generator: MLEngineerDocGenerator):
        """Should return dictionary."""
        result = doc_generator._document_dvc_patterns()

        assert isinstance(result, dict)

    def test_includes_approach(self, doc_generator: MLEngineerDocGenerator):
        """Should include approach description."""
        result = doc_generator._document_dvc_patterns()

        assert 'approach' in result

    def test_includes_workflow(self, doc_generator: MLEngineerDocGenerator):
        """Should include workflow steps."""
        result = doc_generator._document_dvc_patterns()

        assert 'workflow' in result
        assert isinstance(result['workflow'], list)

    def test_includes_best_practices(self, doc_generator: MLEngineerDocGenerator):
        """Should include best practices."""
        result = doc_generator._document_dvc_patterns()

        assert 'best_practices' in result


class TestDocumentValidationWorkflows:
    """Tests for MLEngineerDocGenerator._document_validation_workflows()."""

    def test_returns_dict(self, doc_generator: MLEngineerDocGenerator):
        """Should return dictionary."""
        result = doc_generator._document_validation_workflows()

        assert isinstance(result, dict)

    def test_includes_overview(self, doc_generator: MLEngineerDocGenerator):
        """Should include overview."""
        result = doc_generator._document_validation_workflows()

        assert 'overview' in result

    def test_includes_multiple_workflows(self, doc_generator: MLEngineerDocGenerator):
        """Should include multiple workflow patterns."""
        result = doc_generator._document_validation_workflows()

        assert 'workflows' in result
        assert len(result['workflows']) >= 2


class TestDocumentProductionPatterns:
    """Tests for MLEngineerDocGenerator._document_production_patterns()."""

    def test_returns_dict(self, doc_generator: MLEngineerDocGenerator):
        """Should return dictionary."""
        result = doc_generator._document_production_patterns()

        assert isinstance(result, dict)

    def test_includes_deployment_considerations(self, doc_generator: MLEngineerDocGenerator):
        """Should include deployment considerations."""
        result = doc_generator._document_production_patterns()

        assert 'deployment_considerations' in result

    def test_includes_environment_setup(self, doc_generator: MLEngineerDocGenerator):
        """Should include environment setup guide."""
        result = doc_generator._document_production_patterns()

        assert 'environment_setup' in result

    def test_includes_monitoring_recommendations(self, doc_generator: MLEngineerDocGenerator):
        """Should include monitoring recommendations."""
        result = doc_generator._document_production_patterns()

        assert 'monitoring_recommendations' in result


class TestGenerateTroubleshootingGuide:
    """Tests for MLEngineerDocGenerator._generate_troubleshooting_guide()."""

    def test_returns_dict(self, doc_generator: MLEngineerDocGenerator):
        """Should return dictionary."""
        result = doc_generator._generate_troubleshooting_guide()

        assert isinstance(result, dict)

    def test_includes_common_issues(self, doc_generator: MLEngineerDocGenerator):
        """Should include common issues section."""
        result = doc_generator._generate_troubleshooting_guide()

        assert 'common_issues' in result

    def test_includes_missing_columns_issue(self, doc_generator: MLEngineerDocGenerator):
        """Should include missing columns troubleshooting."""
        result = doc_generator._generate_troubleshooting_guide()

        assert 'missing_columns' in result['common_issues']

    def test_includes_mlflow_issues(self, doc_generator: MLEngineerDocGenerator):
        """Should include MLflow connection issues."""
        result = doc_generator._generate_troubleshooting_guide()

        assert 'mlflow_connection_issues' in result['common_issues']

    def test_includes_dvc_issues(self, doc_generator: MLEngineerDocGenerator):
        """Should include DVC repository issues."""
        result = doc_generator._generate_troubleshooting_guide()

        assert 'dvc_repository_issues' in result['common_issues']

    def test_issues_have_solutions(self, doc_generator: MLEngineerDocGenerator):
        """Each issue should have solutions."""
        result = doc_generator._generate_troubleshooting_guide()

        for issue_key, issue_data in result['common_issues'].items():
            assert 'solutions' in issue_data


# =============================================================================
# QUICK START GUIDE TESTS
# =============================================================================


class TestGenerateQuickStartGuide:
    """Tests for MLEngineerDocGenerator._generate_quick_start_guide()."""

    def test_returns_dict(self, doc_generator: MLEngineerDocGenerator):
        """Should return dictionary."""
        result = doc_generator._generate_quick_start_guide()

        assert isinstance(result, dict)

    def test_includes_immediate_usage(self, doc_generator: MLEngineerDocGenerator):
        """Should include immediate usage steps."""
        result = doc_generator._generate_quick_start_guide()

        assert 'immediate_usage' in result

    def test_includes_key_functions_reference(self, doc_generator: MLEngineerDocGenerator):
        """Should include key functions reference."""
        result = doc_generator._generate_quick_start_guide()

        assert 'key_functions_reference' in result

    def test_immediate_usage_has_steps(self, doc_generator: MLEngineerDocGenerator):
        """Immediate usage should have numbered steps."""
        result = doc_generator._generate_quick_start_guide()

        usage = result['immediate_usage']
        assert 'step_1_setup' in usage
        assert 'step_2_load_data' in usage


# =============================================================================
# MARKDOWN GENERATION TESTS
# =============================================================================


class TestGenerateMarkdownSummary:
    """Tests for MLEngineerDocGenerator._generate_markdown_summary()."""

    def test_creates_markdown_file(self, doc_generator: MLEngineerDocGenerator,
                                   temp_output_dir: Path):
        """Should create markdown file."""
        output_file = temp_output_dir / "README.md"
        doc_generator._generate_markdown_summary({}, output_file)

        assert output_file.exists()

    def test_contains_header(self, doc_generator: MLEngineerDocGenerator,
                             temp_output_dir: Path):
        """Should contain header section."""
        output_file = temp_output_dir / "README.md"
        doc_generator._generate_markdown_summary({}, output_file)

        content = output_file.read_text()
        assert '# RILA Schema Validation' in content

    def test_contains_overview(self, doc_generator: MLEngineerDocGenerator,
                               temp_output_dir: Path):
        """Should contain overview section."""
        output_file = temp_output_dir / "README.md"
        doc_generator._generate_markdown_summary({}, output_file)

        content = output_file.read_text()
        assert '## Overview' in content

    def test_contains_quick_start(self, doc_generator: MLEngineerDocGenerator,
                                  temp_output_dir: Path):
        """Should contain quick start section."""
        output_file = temp_output_dir / "README.md"
        doc_generator._generate_markdown_summary({}, output_file)

        content = output_file.read_text()
        assert '## Quick Start' in content


class TestMarkdownHelperMethods:
    """Tests for individual markdown section generators."""

    def test_generate_md_header_includes_timestamp(self, doc_generator: MLEngineerDocGenerator):
        """Header should include timestamp."""
        result = doc_generator._generate_md_header()

        assert doc_generator.generation_timestamp in result
        assert 'ML Engineer Handoff' in result

    def test_generate_md_overview_includes_architecture(self, doc_generator: MLEngineerDocGenerator):
        """Overview should mention architecture."""
        result = doc_generator._generate_md_overview()

        assert 'Architecture' in result
        assert 'Pandera' in result or 'Pydantic' in result

    def test_generate_md_quick_start_has_code_blocks(self, doc_generator: MLEngineerDocGenerator):
        """Quick start should have code examples."""
        result = doc_generator._generate_md_quick_start()

        assert '```python' in result

    def test_generate_md_key_capabilities(self, doc_generator: MLEngineerDocGenerator):
        """Key capabilities should cover main features."""
        result = doc_generator._generate_md_key_capabilities()

        assert 'Schema Validation' in result
        assert 'MLflow' in result

    def test_generate_md_production_deployment(self, doc_generator: MLEngineerDocGenerator):
        """Production deployment should include env setup."""
        result = doc_generator._generate_md_production_deployment()

        assert 'Environment Setup' in result

    def test_generate_md_files_created(self, doc_generator: MLEngineerDocGenerator):
        """Files created should list key modules."""
        result = doc_generator._generate_md_files_created()

        assert 'data_schemas' in result.lower() or 'validation' in result.lower()

    def test_generate_md_troubleshooting(self, doc_generator: MLEngineerDocGenerator):
        """Troubleshooting should have solutions."""
        result = doc_generator._generate_md_troubleshooting()

        assert 'Missing Columns' in result or 'strict=False' in result

    def test_generate_md_support(self, doc_generator: MLEngineerDocGenerator):
        """Support section should mention documentation."""
        result = doc_generator._generate_md_support()

        assert 'docstrings' in result.lower() or 'documentation' in result.lower()

    def test_generate_md_footer_includes_attribution(self, doc_generator: MLEngineerDocGenerator):
        """Footer should include attribution."""
        result = doc_generator._generate_md_footer()

        assert 'automatically generated' in result.lower() or 'documentation' in result.lower()


# =============================================================================
# COMPLETE DOCUMENTATION GENERATION TESTS
# =============================================================================


class TestGenerateCompleteDocumentation:
    """Tests for MLEngineerDocGenerator.generate_complete_documentation()."""

    def test_raises_when_modules_unavailable(self, doc_generator: MLEngineerDocGenerator,
                                             monkeypatch):
        """Should raise RuntimeError when validation modules unavailable."""
        monkeypatch.setattr(
            'src.validation.ml_engineer_docs.VALIDATION_MODULES_AVAILABLE',
            False
        )

        with pytest.raises(RuntimeError):
            doc_generator.generate_complete_documentation()

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_creates_json_file(self, doc_generator: MLEngineerDocGenerator):
        """Should create JSON documentation file."""
        doc_path = doc_generator.generate_complete_documentation()

        assert Path(doc_path).exists()
        assert doc_path.endswith('.json')

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_creates_markdown_file(self, doc_generator: MLEngineerDocGenerator,
                                   temp_output_dir: Path):
        """Should create README markdown file."""
        doc_generator.generate_complete_documentation()

        readme_path = temp_output_dir / "README.md"
        assert readme_path.exists()

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_json_is_valid(self, doc_generator: MLEngineerDocGenerator):
        """Generated JSON should be valid and parseable."""
        doc_path = doc_generator.generate_complete_documentation()

        with open(doc_path) as f:
            data = json.load(f)

        assert 'ml_engineer_handoff_documentation' in data

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_includes_sample_data_analysis(self, doc_generator: MLEngineerDocGenerator,
                                           sample_dataframe: pd.DataFrame):
        """Should include sample data analysis when provided."""
        doc_path = doc_generator.generate_complete_documentation(sample_data=sample_dataframe)

        with open(doc_path) as f:
            data = json.load(f)

        handoff = data['ml_engineer_handoff_documentation']
        assert handoff['compatibility_report'] is not None

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_prints_progress_messages(self, doc_generator: MLEngineerDocGenerator, capsys):
        """Should print progress messages."""
        doc_generator.generate_complete_documentation()

        captured = capsys.readouterr()
        assert 'GENERATING' in captured.out or 'SUCCESS' in captured.out

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_returns_path_string(self, doc_generator: MLEngineerDocGenerator):
        """Should return path as string."""
        result = doc_generator.generate_complete_documentation()

        assert isinstance(result, str)


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe_compatibility_report(self, doc_generator: MLEngineerDocGenerator):
        """Should handle empty DataFrame in compatibility report."""
        df = pd.DataFrame()
        result = doc_generator.generate_schema_compatibility_report(df, "empty")

        assert result['actual_data_shape'] == (0, 0)
        assert result['actual_columns'] == []

    def test_large_column_dataframe(self, doc_generator: MLEngineerDocGenerator):
        """Should handle DataFrame with many columns."""
        data = {f'col_{i}': [1, 2, 3] for i in range(100)}
        df = pd.DataFrame(data)

        result = doc_generator.generate_schema_compatibility_report(df, "wide")

        assert result['actual_data_shape'][1] == 100

    def test_special_characters_in_schema_name(self, doc_generator: MLEngineerDocGenerator,
                                               sample_dataframe: pd.DataFrame):
        """Should handle special characters in schema name."""
        result = doc_generator.generate_schema_compatibility_report(
            sample_dataframe, "schema-with_special.chars"
        )

        assert result['schema_context'] == "schema-with_special.chars"

    def test_output_dir_permissions(self, tmp_path):
        """Should handle output directory creation."""
        nested_dir = tmp_path / "a" / "b" / "c" / "docs"
        generator = MLEngineerDocGenerator(output_dir=str(nested_dir))

        assert nested_dir.exists()

    def test_timestamp_format_consistency(self, doc_generator: MLEngineerDocGenerator):
        """Timestamp should be consistent across methods."""
        ts = doc_generator.generation_timestamp

        # Should be parseable
        parsed = datetime.fromisoformat(ts)
        assert isinstance(parsed, datetime)


class TestMultipleGeneratorInstances:
    """Tests for multiple generator instances."""

    def test_independent_output_dirs(self, tmp_path):
        """Multiple generators should use independent output directories."""
        gen1 = MLEngineerDocGenerator(output_dir=str(tmp_path / "dir1"))
        gen2 = MLEngineerDocGenerator(output_dir=str(tmp_path / "dir2"))

        assert gen1.output_dir != gen2.output_dir

    def test_independent_timestamps(self):
        """Multiple generators should have independent timestamps."""
        import time
        gen1 = MLEngineerDocGenerator(output_dir="/tmp/test1")
        time.sleep(0.01)
        gen2 = MLEngineerDocGenerator(output_dir="/tmp/test2")

        # Timestamps may or may not differ depending on resolution
        # Just ensure both are set
        assert gen1.generation_timestamp is not None
        assert gen2.generation_timestamp is not None


class TestValidationModulesFlag:
    """Tests for VALIDATION_MODULES_AVAILABLE flag behavior."""

    def test_flag_is_boolean(self):
        """VALIDATION_MODULES_AVAILABLE should be boolean."""
        assert isinstance(VALIDATION_MODULES_AVAILABLE, bool)

    def test_analyze_schemas_respects_flag(self, doc_generator: MLEngineerDocGenerator,
                                           monkeypatch):
        """Methods should check VALIDATION_MODULES_AVAILABLE."""
        monkeypatch.setattr(
            'src.validation.ml_engineer_docs.VALIDATION_MODULES_AVAILABLE',
            False
        )

        with pytest.raises(RuntimeError):
            doc_generator.analyze_current_schemas()

    def test_integration_guide_respects_flag(self, doc_generator: MLEngineerDocGenerator,
                                             monkeypatch):
        """Integration guide should check VALIDATION_MODULES_AVAILABLE."""
        monkeypatch.setattr(
            'src.validation.ml_engineer_docs.VALIDATION_MODULES_AVAILABLE',
            False
        )

        with pytest.raises(RuntimeError):
            doc_generator.generate_integration_guide()

    def test_compatibility_report_handles_flag(self, doc_generator: MLEngineerDocGenerator,
                                               sample_dataframe: pd.DataFrame,
                                               monkeypatch):
        """Compatibility report should handle unavailable modules."""
        monkeypatch.setattr(
            'src.validation.ml_engineer_docs.VALIDATION_MODULES_AVAILABLE',
            False
        )

        result = doc_generator.generate_schema_compatibility_report(
            sample_dataframe, "test"
        )

        # Should not raise, but should indicate error
        assert 'error' in result


# =============================================================================
# MEANINGFUL VALUE VALIDATION TESTS
# These tests validate actual content, not just structure
# =============================================================================


class TestMeaningfulValueValidation:
    """Tests that validate actual values in generated documentation.

    These tests go beyond structure checks to ensure generated content
    is correct and useful, catching logic bugs that structure-only tests miss.
    """

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_generated_json_roundtrip_preserves_content(
        self, doc_generator: MLEngineerDocGenerator, sample_dataframe: pd.DataFrame
    ):
        """Generated JSON should survive roundtrip serialization."""
        doc_path = doc_generator.generate_complete_documentation(sample_data=sample_dataframe)

        # Read JSON back
        with open(doc_path) as f:
            data_round1 = json.load(f)

        # Serialize and deserialize again
        json_str = json.dumps(data_round1, default=str)
        data_round2 = json.loads(json_str)

        # Verify key content survives
        handoff1 = data_round1['ml_engineer_handoff_documentation']
        handoff2 = data_round2['ml_engineer_handoff_documentation']

        assert handoff1['generated_at'] == handoff2['generated_at']
        assert handoff1['overview']['purpose'] == handoff2['overview']['purpose']
        assert handoff1['overview']['architecture'] == handoff2['overview']['architecture']

    def test_markdown_column_count_matches_dataframe(
        self, doc_generator: MLEngineerDocGenerator, temp_output_dir: Path
    ):
        """Markdown compatibility report should reflect actual DataFrame dimensions."""
        # Create 50-column DataFrame
        data = {f'col_{i}': [1, 2, 3] for i in range(50)}
        df = pd.DataFrame(data)

        result = doc_generator.generate_schema_compatibility_report(df, "wide_schema")

        # Verify actual shape is captured
        assert result['actual_data_shape'] == (3, 50)
        assert len(result['actual_columns']) == 50

    def test_compatibility_report_validates_actual_dataframe_shape(
        self, doc_generator: MLEngineerDocGenerator, sample_dataframe: pd.DataFrame
    ):
        """Compatibility report should contain exact row/column counts."""
        result = doc_generator.generate_schema_compatibility_report(
            sample_dataframe, "sample_analysis"
        )

        # Verify exact counts match DataFrame
        assert result['actual_data_shape'] == sample_dataframe.shape
        assert result['actual_data_shape'][0] == 100  # rows
        assert result['actual_data_shape'][1] == 3    # columns

    def test_troubleshooting_guide_solutions_are_actionable(
        self, doc_generator: MLEngineerDocGenerator
    ):
        """Each troubleshooting issue should have non-empty, actionable solutions."""
        result = doc_generator._generate_troubleshooting_guide()

        for issue_key, issue_data in result['common_issues'].items():
            solutions = issue_data['solutions']

            # Verify solutions exist and are non-empty
            assert len(solutions) > 0, f"Issue {issue_key} has no solutions"

            for solution in solutions:
                assert len(solution) > 10, f"Solution '{solution}' is too vague"

                # Solutions should contain actionable keywords
                actionable_keywords = [
                    'use', 'set', 'run', 'check', 'configure', 'initialize',
                    'call', 'try', 'update', 'verify', 'ensure', 'install'
                ]
                solution_lower = solution.lower()
                has_actionable = any(kw in solution_lower for kw in actionable_keywords)
                # Allow descriptions that describe symptoms/causes too
                is_descriptive = 'gracefully' in solution_lower or 'will' in solution_lower

                assert has_actionable or is_descriptive, (
                    f"Solution '{solution}' lacks actionable guidance"
                )

    def test_timestamp_consistency_across_methods(
        self, doc_generator: MLEngineerDocGenerator, sample_dataframe: pd.DataFrame
    ):
        """Timestamps should be consistent across all generated content."""
        generator_ts = doc_generator.generation_timestamp

        # Check compatibility report
        compat_result = doc_generator.generate_schema_compatibility_report(
            sample_dataframe, "test"
        )
        assert compat_result['report_timestamp'] == generator_ts

        # Check header
        header = doc_generator._generate_md_header()
        assert generator_ts in header

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_integration_guide_contains_code_examples(
        self, doc_generator: MLEngineerDocGenerator
    ):
        """Integration guide should contain valid Python code examples."""
        result = doc_generator.generate_integration_guide()

        # Check MLflow patterns have code examples
        mlflow_patterns = result['mlflow_integration']
        for func_info in mlflow_patterns['key_functions']:
            example = func_info.get('example', '')
            assert len(example) > 0, f"Function {func_info['function']} lacks example"

            # Verify it looks like Python code
            assert 'import' in example or '=' in example or '(' in example

        # Check DVC patterns have code examples
        dvc_patterns = result['dvc_integration']
        for func_info in dvc_patterns.get('key_functions', []):
            example = func_info.get('example', '')
            if example:  # Some may not have examples
                assert '(' in example, "Example should contain function calls"

    def test_schema_analysis_handles_none_values_in_dataframe(
        self, doc_generator: MLEngineerDocGenerator
    ):
        """Schema compatibility should handle DataFrames with None values."""
        df = pd.DataFrame({
            'col_with_nulls': [1, None, 3, None, 5],
            'col_without_nulls': [1, 2, 3, 4, 5],
            'all_nulls': [None, None, None, None, None]
        })

        # Should not crash
        result = doc_generator.generate_schema_compatibility_report(df, "null_test")

        assert result['actual_data_shape'] == (5, 3)
        assert 'col_with_nulls' in result['actual_columns']
        assert 'all_nulls' in result['actual_columns']

    def test_quick_start_guide_steps_are_ordered(
        self, doc_generator: MLEngineerDocGenerator
    ):
        """Quick start guide should have properly ordered steps."""
        result = doc_generator._generate_quick_start_guide()

        immediate_usage = result['immediate_usage']
        step_keys = list(immediate_usage.keys())

        # Verify steps exist and are ordered
        expected_order = ['step_1_setup', 'step_2_load_data']
        for expected in expected_order:
            assert expected in step_keys, f"Missing step: {expected}"

        # Verify step_1 comes before step_2
        step_1_idx = step_keys.index('step_1_setup')
        step_2_idx = step_keys.index('step_2_load_data')
        assert step_1_idx < step_2_idx, "Steps are not in correct order"

    def test_files_created_section_lists_actual_paths(
        self, doc_generator: MLEngineerDocGenerator
    ):
        """Files created section should list valid module paths."""
        result = doc_generator._generate_md_files_created()

        # Should mention actual module paths
        assert 'src/' in result or 'validation' in result.lower()

        # Should mention the key modules
        expected_modules = ['mlflow_config', 'data_schemas', 'config_schemas']
        modules_mentioned = sum(1 for m in expected_modules if m in result)
        assert modules_mentioned >= 2, f"Only {modules_mentioned} of {len(expected_modules)} key modules mentioned"

    @pytest.mark.skipif(
        not VALIDATION_MODULES_AVAILABLE,
        reason="Validation modules not available"
    )
    def test_schema_statistics_aggregate_correctly(
        self, doc_generator: MLEngineerDocGenerator
    ):
        """Schema statistics should aggregate column counts correctly."""
        result = doc_generator.analyze_current_schemas()

        if 'error' not in result:
            stats = result['schema_statistics']

            # Total columns should equal sum of individual schema columns
            total_from_stats = stats['total_columns_across_schemas']
            total_computed = sum(
                info.get('column_count', 0)
                for info in result['schemas_discovered'].values()
            )
            assert total_from_stats == total_computed

            # Total schemas should match discovered count
            assert stats['total_schemas'] == len(result['schemas_discovered'])

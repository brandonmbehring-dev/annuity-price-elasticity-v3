"""
DataFrame schema validation using Pandera for RILA pipeline.

This module provides schema validation for pandas DataFrames at various
pipeline stages to ensure data quality and catch issues early.

Integration with existing DVC and data processing workflows.
"""

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check, Index
from typing import Optional, List, Dict, Any
import numpy as np
from pathlib import Path


# Core dataset schemas
# Feature Naming Unification (2026-01-26):
# - _current → _t0 (sales_target_t0, prudential_rate_t0)
# - competitor_mid → competitor_weighted
FINAL_DATASET_SCHEMA = DataFrameSchema(
    {
        # Time series index
        "date": Column(
            pa.DateTime,
            nullable=False,
            description="Business date for observations"
        ),

        # Target variables
        "sales": Column(
            pa.Float64,
            Check.ge(0),
            nullable=False,
            description="FlexGuard sales volume"
        ),
        "sales_target_t0": Column(
            pa.Float64,
            Check.ge(0),
            nullable=False,
            description="Current target sales for forecasting (t0 = current period)"
        ),

        # Core features (from feature selection)
        "prudential_rate_t0": Column(
            pa.Float64,
            Check.between(0, 20),
            nullable=False,
            description="Prudential rate at t0 (current period)"
        ),
        "competitor_weighted_t2": Column(
            pa.Float64,
            nullable=False,
            description="Weighted mean competitor rate, t-2 lag"
        ),
        "competitor_top5_t3": Column(
            pa.Float64,
            nullable=False,
            description="Top-5 competitor rate, t-3 lag"
        ),
        "sales_target_contract_t5": Column(
            pa.Float64,
            Check.ge(0),
            nullable=True,
            description="Historical sales pattern, t-5 lag"
        ),

        # Weight for temporal importance
        "weight": Column(
            pa.Float64,
            Check.between(0, 1),
            nullable=True,
            description="Exponential weights for temporal importance"
        ),
    },
    strict="filter",  # Remove unexpected columns
    coerce=True,      # Auto-convert compatible types
    description="Final dataset schema for RILA forecasting pipeline"
)

# Model prediction output schema
FORECAST_RESULTS_SCHEMA = DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False),
        "y_true": Column(pa.Float64, Check.ge(0), nullable=False),
        "y_predict": Column(pa.Float64, Check.ge(0), nullable=False),
        "abs_pct_error": Column(pa.Float64, Check.between(0, 5), nullable=False),  # MAPE should be reasonable
    },
    strict=False,  # Allow additional confidence interval columns
    description="Bootstrap forecasting results schema"
)

# Feature selection results schema
FEATURE_SELECTION_RESULTS_SCHEMA = DataFrameSchema(
    {
        "feature_name": Column(pa.String, nullable=False),
        "coefficient": Column(pa.Float64, nullable=False),
        "p_value": Column(pa.Float64, Check.between(0, 1), nullable=True),
        "selected": Column(pa.Bool, nullable=False),
    },
    description="Feature selection results with AIC scoring"
)


class DataFrameValidator:
    """Schema validation utilities for RILA pipeline DataFrames with MLflow integration."""

    @staticmethod
    def validate_final_dataset(df: pd.DataFrame,
                             strict: bool = True,
                             sample_validation: bool = False) -> pd.DataFrame:
        """
        Validate final dataset with configurable strictness.

        Args:
            df: DataFrame to validate
            strict: If True, raise exceptions on validation errors
            sample_validation: If True, validate only a sample for performance

        Returns:
            Validated DataFrame (potentially filtered/coerced)

        Raises:
            ValueError: If validation fails and strict=True
        """
        try:
            if sample_validation and len(df) > 1000:
                # Sample validation for large datasets
                sample_df = df.sample(n=1000, random_state=42)
                FINAL_DATASET_SCHEMA.validate(sample_df, lazy=True)
                print(f"✓ Sample validation passed ({len(sample_df)} rows sampled)")

            validated_df = FINAL_DATASET_SCHEMA.validate(df, lazy=not strict)

            # Additional business logic validation
            if len(validated_df) < 50:
                raise ValueError(f"Dataset too small: {len(validated_df)} rows < 50 minimum")

            # Check for reasonable date range
            date_range = validated_df['date'].max() - validated_df['date'].min()
            if date_range.days < 365:
                import warnings
                warnings.warn(f"Dataset covers only {date_range.days} days - may be insufficient for modeling")

            return validated_df

        except pa.errors.SchemaErrors as e:
            error_msg = f"DataFrame schema validation failed:\n{e.failure_cases}"
            if strict:
                raise ValueError(error_msg)
            else:
                print(f"WARNING: {error_msg}")
                return df

    @staticmethod
    def validate_forecast_results(df: pd.DataFrame) -> pd.DataFrame:
        """Validate bootstrap forecasting results."""
        try:
            validated_df = FORECAST_RESULTS_SCHEMA.validate(df, lazy=True)

            # Business logic validation
            avg_mape = validated_df['abs_pct_error'].mean()
            if avg_mape > 0.5:  # 50% MAPE threshold
                import warnings
                warnings.warn(f"High average MAPE: {avg_mape:.2%} - model performance may be poor")

            return validated_df

        except pa.errors.SchemaError as e:
            raise ValueError(f"Forecast results validation failed: {e}")

    @staticmethod
    def validate_feature_selection_results(df: pd.DataFrame) -> pd.DataFrame:
        """Validate feature selection results from AIC process."""
        try:
            validated_df = FEATURE_SELECTION_RESULTS_SCHEMA.validate(df)

            # Check that at least some features were selected
            selected_count = validated_df['selected'].sum()
            if selected_count == 0:
                raise ValueError("No features were selected - check AIC selection process")
            elif selected_count > 10:
                import warnings
                warnings.warn(f"Many features selected ({selected_count}) - may lead to overfitting")

            return validated_df

        except pa.errors.SchemaError as e:
            raise ValueError(f"Feature selection results validation failed: {e}")

    @staticmethod
    def validate_and_log_to_mlflow(df: pd.DataFrame,
                                  schema: DataFrameSchema,
                                  schema_name: str,
                                  validation_strict: bool = True,
                                  log_to_mlflow: bool = True) -> pd.DataFrame:
        """
        Validate DataFrame and optionally log validation results to MLflow.

        Parameters:
            df: DataFrame to validate
            schema: Pandera schema to validate against
            schema_name: Name of schema for MLflow logging
            validation_strict: Whether to use strict validation
            log_to_mlflow: Whether to log validation results to MLflow

        Returns:
            Validated DataFrame

        Raises:
            ValueError: If validation fails and strict=True
            RuntimeError: If MLflow logging fails
        """
        # Import MLflow functions with defensive imports
        try:
            from src.config.mlflow_config import safe_mlflow_log_schema_validation
            MLFLOW_INTEGRATION_AVAILABLE = True
        except ImportError:
            MLFLOW_INTEGRATION_AVAILABLE = False

        # Perform schema validation
        try:
            validated_df = schema.validate(df, lazy=not validation_strict)
            print(f"SUCCESS: Schema validation passed for '{schema_name}': {validated_df.shape}")

            # Log to MLflow if requested
            if log_to_mlflow:
                if not MLFLOW_INTEGRATION_AVAILABLE:
                    raise RuntimeError(
                        f"MLflow integration not available for logging '{schema_name}'. "
                        "Required module 'src.config.mlflow_config' cannot be imported. "
                        "Install MLflow or check module path."
                    )
                try:
                    validation_results = safe_mlflow_log_schema_validation(
                        df=validated_df,
                        schema_name=schema_name,
                        validation_strict=validation_strict
                    )
                    print(f"SUCCESS: MLflow logging completed for '{schema_name}'")
                except Exception as e:
                    if validation_strict:
                        raise RuntimeError(f"MLflow logging failed for {schema_name}: {e}")
                    else:
                        print(f"⚠ MLflow logging warning for {schema_name}: {e}")

            elif log_to_mlflow and not MLFLOW_INTEGRATION_AVAILABLE:
                print(f"⚠ MLflow integration not available - skipping logging for {schema_name}")

            return validated_df

        except Exception as e:
            error_msg = f"Schema validation failed for {schema_name}: {e}"
            if validation_strict:
                raise ValueError(error_msg)
            else:
                print(f"⚠ {error_msg} (non-strict mode)")
                return df

    @staticmethod
    def validate_final_dataset_with_mlflow(df: pd.DataFrame,
                                         strict: bool = True,
                                         sample_validation: bool = False,
                                         log_to_mlflow: bool = True) -> pd.DataFrame:
        """
        Validate final dataset with MLflow integration.

        Combines existing validate_final_dataset functionality with MLflow logging.

        Parameters:
            df: DataFrame to validate
            strict: Whether to use strict validation
            sample_validation: Whether to validate only a sample for performance
            log_to_mlflow: Whether to log validation results to MLflow

        Returns:
            Validated DataFrame
        """
        # Use existing validation logic first
        validated_df = DataFrameValidator.validate_final_dataset(
            df=df,
            strict=strict,
            sample_validation=sample_validation
        )

        # Add MLflow logging if requested
        if log_to_mlflow:
            return DataFrameValidator.validate_and_log_to_mlflow(
                df=validated_df,
                schema=FINAL_DATASET_SCHEMA,
                schema_name="final_dataset",
                validation_strict=strict,
                log_to_mlflow=True
            )

        return validated_df

    @staticmethod
    def validate_forecast_results_with_mlflow(df: pd.DataFrame,
                                            log_to_mlflow: bool = True) -> pd.DataFrame:
        """
        Validate forecast results with MLflow integration.

        Parameters:
            df: Forecast results DataFrame to validate
            log_to_mlflow: Whether to log validation results to MLflow

        Returns:
            Validated DataFrame
        """
        # Use existing validation logic first
        validated_df = DataFrameValidator.validate_forecast_results(df)

        # Add MLflow logging if requested
        if log_to_mlflow:
            return DataFrameValidator.validate_and_log_to_mlflow(
                df=validated_df,
                schema=FORECAST_RESULTS_SCHEMA,
                schema_name="forecast_results",
                validation_strict=True,
                log_to_mlflow=True
            )

        return validated_df

    @staticmethod
    def _process_single_validation(
        validation_config: Dict[str, Any], index: int, total: int
    ) -> Dict[str, Any]:
        """Process a single validation configuration."""
        schema_name = validation_config.get('schema_name', f'validation_{index}')
        print(f"\nREPORT: Processing validation {index}/{total}: {schema_name}")

        df = validation_config['df']
        schema = validation_config['schema']
        validation_strict = validation_config.get('validation_strict', True)
        log_to_mlflow = validation_config.get('log_to_mlflow', True)

        validated_df = DataFrameValidator.validate_and_log_to_mlflow(
            df=df,
            schema=schema,
            schema_name=schema_name,
            validation_strict=validation_strict,
            log_to_mlflow=log_to_mlflow
        )

        return {
            "status": "SUCCESS",
            "shape": validated_df.shape,
            "validation_strict": validation_strict,
            "mlflow_logged": log_to_mlflow
        }

    @staticmethod
    def _print_batch_summary(batch_results: Dict[str, Any]) -> None:
        """Print batch validation summary."""
        print(f"\nTARGET: BATCH VALIDATION SUMMARY:")
        print(f"   Total Validations: {batch_results['total_validations']}")
        print(f"   Successful Validations: {batch_results['successful_validations']}/{batch_results['total_validations']}")
        print(f"   MLflow Logs: {batch_results['successful_mlflow_logs']}/{batch_results['total_validations']}")
        print(f"   Success Rate: {batch_results['summary']['validation_success_rate']:.1%}")

    @staticmethod
    def batch_validate_with_mlflow(validations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform batch validation with MLflow logging for multiple DataFrames."""
        from datetime import datetime

        batch_results = {
            "timestamp": datetime.now().isoformat(),
            "total_validations": len(validations),
            "successful_validations": 0,
            "successful_mlflow_logs": 0,
            "validation_results": {},
            "summary": {}
        }

        print(f"Starting batch validation with MLflow for {len(validations)} DataFrames...")

        for i, validation_config in enumerate(validations, 1):
            schema_name = validation_config.get('schema_name', f'validation_{i}')
            try:
                result = DataFrameValidator._process_single_validation(
                    validation_config, i, len(validations)
                )
                batch_results["validation_results"][schema_name] = result
                batch_results["successful_validations"] += 1
                if result.get("mlflow_logged"):
                    batch_results["successful_mlflow_logs"] += 1
            except Exception as e:
                batch_results["validation_results"][schema_name] = {
                    "status": "FAILED",
                    "error": str(e)
                }
                print(f"FAILED: Validation failed for {schema_name}: {e}")

        batch_results["summary"] = {
            "validation_success_rate": batch_results["successful_validations"] / batch_results["total_validations"],
            "mlflow_logging_success_rate": batch_results["successful_mlflow_logs"] / batch_results["total_validations"],
            "all_validations_passed": batch_results["successful_validations"] == batch_results["total_validations"]
        }

        DataFrameValidator._print_batch_summary(batch_results)
        return batch_results


# Integration with existing pipeline functions
def load_validated_dataset(file_path: str,
                          schema: DataFrameSchema = FINAL_DATASET_SCHEMA,
                          **pandas_kwargs) -> pd.DataFrame:
    """
    Load dataset with automatic schema validation.

    Args:
        file_path: Path to parquet file
        schema: Pandera schema to validate against
        **pandas_kwargs: Additional arguments passed to pd.read_parquet

    Returns:
        Validated DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If schema validation fails
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    try:
        df = pd.read_parquet(file_path, **pandas_kwargs)
        validated_df = schema.validate(df, lazy=True)

        print(f"✓ Loaded and validated dataset: {file_path}")
        print(f"   Shape: {validated_df.shape}")
        print(f"   Date range: {validated_df['date'].min()} to {validated_df['date'].max()}")

        return validated_df

    except Exception as e:
        raise ValueError(f"Failed to load/validate dataset {file_path}: {e}")


# DVC integration helper
def validate_and_save_dataset(df: pd.DataFrame,
                            output_path: str,
                            schema: DataFrameSchema,
                            dvc_add: bool = True) -> pd.DataFrame:
    """
    Validate DataFrame and save with optional DVC tracking.

    Args:
        df: DataFrame to validate and save
        output_path: Path to save validated dataset
        schema: Schema to validate against
        dvc_add: Whether to add to DVC tracking

    Returns:
        Validated DataFrame
    """
    import os

    # Validate before saving
    validated_df = schema.validate(df, lazy=True)

    # Create output directory if needed
    os.makedirs(Path(output_path).parent, exist_ok=True)

    # Save validated dataset
    validated_df.to_parquet(output_path)
    print(f"✓ Saved validated dataset: {output_path}")

    # Optional DVC tracking
    if dvc_add:
        try:
            result = os.system(f'dvc add {output_path}')
            if result == 0:
                print(f"✓ Added to DVC tracking: {output_path}")
            else:
                print(f"⚠ DVC add failed for: {output_path}")
        except Exception as e:
            print(f"⚠ DVC add error: {e}")

    return validated_df


class SchemaAwareDVCTracker:
    """
    Schema-aware DVC tracking that validates data before version control.

    Integrates schema validation with DVC tracking following existing
    individual file patterns from UNIFIED_CODING_STANDARDS.md.
    """

    @staticmethod
    def _collect_validation_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """Collect statistics from validated DataFrame."""
        stats = {
            "shape": df.shape,
            "columns_count": len(df.columns),
            "null_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        if 'date' in df.columns:
            date_range = df['date'].max() - df['date'].min()
            stats["date_range_days"] = date_range.days
        return stats

    @staticmethod
    def _run_dvc_tracking(output_path: str, dvc_commit_message: Optional[str]) -> Dict[str, Any]:
        """Run DVC add command and optionally stage to git."""
        import subprocess
        import os

        result = {"dvc_tracking_status": "PENDING"}
        try:
            dvc_add = subprocess.run(
                ['dvc', 'add', output_path],
                capture_output=True, text=True,
                cwd=os.path.dirname(output_path) or '.'
            )
            if dvc_add.returncode == 0:
                result["dvc_tracking_status"] = "SUCCESS"
                result["dvc_file"] = f"{output_path}.dvc"
                if dvc_commit_message:
                    subprocess.run(['git', 'add', f"{output_path}.dvc"], capture_output=True)
            else:
                result["dvc_tracking_status"] = "FAILED"
                result["dvc_error"] = dvc_add.stderr
        except FileNotFoundError:
            result["dvc_tracking_status"] = "DVC_NOT_AVAILABLE"
        except Exception as e:
            result["dvc_tracking_status"] = "ERROR"
            result["dvc_error"] = str(e)
        return result

    @staticmethod
    def validate_and_track_dataset(df: pd.DataFrame,
                                 output_path: str,
                                 schema: DataFrameSchema,
                                 schema_name: str = "dataset",
                                 dvc_commit_message: Optional[str] = None,
                                 validation_strict: bool = True) -> Dict[str, Any]:
        """
        Validate DataFrame schema and track with DVC.

        Returns:
            Dict containing validation results and DVC tracking status
        """
        import os
        from datetime import datetime

        results = {
            "schema_name": schema_name,
            "output_path": output_path,
            "timestamp": datetime.now().isoformat(),
            "validation_status": "PENDING",
            "dvc_tracking_status": "PENDING",
            "statistics": {}
        }

        # Validate schema
        try:
            validated_df = schema.validate(df, lazy=not validation_strict)
            results["statistics"] = SchemaAwareDVCTracker._collect_validation_statistics(validated_df)
            results["validation_status"] = "PASSED"
        except Exception as e:
            results["validation_status"] = "FAILED"
            results["validation_error"] = str(e)
            if validation_strict:
                raise ValueError(f"Schema validation failed for {schema_name}: {e}")
            validated_df = df

        # Save and track with DVC
        os.makedirs(Path(output_path).parent, exist_ok=True)
        validated_df.to_parquet(output_path)
        results.update(SchemaAwareDVCTracker._run_dvc_tracking(output_path, dvc_commit_message))

        print(f"Validation: {results['validation_status']}, DVC: {results['dvc_tracking_status']}")
        return results

    @staticmethod
    def _process_single_dataset(
        dataset_config: Dict[str, Any], index: int, total: int
    ) -> Dict[str, Any]:
        """Process a single dataset for bulk validation and tracking."""
        schema_name = dataset_config.get('schema_name', f'dataset_{index}')
        print(f"\nProcessing dataset {index}/{total}: {schema_name}")

        return SchemaAwareDVCTracker.validate_and_track_dataset(
            df=dataset_config['df'],
            output_path=dataset_config['output_path'],
            schema=dataset_config['schema'],
            schema_name=schema_name,
            dvc_commit_message=dataset_config.get('dvc_commit_message'),
            validation_strict=dataset_config.get('validation_strict', True)
        )

    @staticmethod
    def _generate_bulk_summary(bulk_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for bulk processing."""
        return {
            "validation_success_rate": bulk_results["successful_validations"] / bulk_results["total_datasets"],
            "dvc_tracking_success_rate": bulk_results["successful_dvc_tracking"] / bulk_results["total_datasets"],
            "all_validations_passed": bulk_results["successful_validations"] == bulk_results["total_datasets"],
            "all_dvc_tracking_passed": bulk_results["successful_dvc_tracking"] == bulk_results["total_datasets"]
        }

    @staticmethod
    def _print_bulk_summary(bulk_results: Dict[str, Any]) -> None:
        """Print bulk processing summary."""
        print(f"\nTARGET: BULK PROCESSING SUMMARY:")
        print(f"   Total Datasets: {bulk_results['total_datasets']}")
        print(f"   Validation Success: {bulk_results['successful_validations']}/{bulk_results['total_datasets']}")
        print(f"   DVC Tracking Success: {bulk_results['successful_dvc_tracking']}/{bulk_results['total_datasets']}")
        print(f"   Overall Success Rate: {bulk_results['summary']['validation_success_rate']:.1%}")

    @staticmethod
    def bulk_validate_and_track(datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate and track multiple datasets following DVC individual file patterns."""
        from datetime import datetime

        bulk_results = {
            "timestamp": datetime.now().isoformat(),
            "total_datasets": len(datasets),
            "successful_validations": 0,
            "successful_dvc_tracking": 0,
            "dataset_results": {},
            "summary": {}
        }

        print(f"START: Starting bulk schema validation and DVC tracking for {len(datasets)} datasets...")

        for i, dataset_config in enumerate(datasets, 1):
            schema_name = dataset_config.get('schema_name', f'dataset_{i}')
            try:
                result = SchemaAwareDVCTracker._process_single_dataset(
                    dataset_config, i, len(datasets)
                )
                bulk_results["dataset_results"][schema_name] = result

                if result["validation_status"] == "PASSED":
                    bulk_results["successful_validations"] += 1
                if result["dvc_tracking_status"] == "SUCCESS":
                    bulk_results["successful_dvc_tracking"] += 1

            except Exception as e:
                bulk_results["dataset_results"][schema_name] = {
                    "validation_status": "ERROR",
                    "dvc_tracking_status": "ERROR",
                    "error": str(e)
                }
                print(f"FAILED: Error processing {schema_name}: {e}")

        bulk_results["summary"] = SchemaAwareDVCTracker._generate_bulk_summary(bulk_results)
        SchemaAwareDVCTracker._print_bulk_summary(bulk_results)

        return bulk_results


# Integration helper functions following existing patterns
def load_and_validate_with_dvc_context(file_path: str,
                                      schema: DataFrameSchema,
                                      schema_name: str = "dataset") -> pd.DataFrame:
    """
    Load dataset with DVC context awareness and schema validation.

    Parameters:
        file_path: Path to parquet file (can be DVC-tracked)
        schema: Pandera schema for validation
        schema_name: Name for logging

    Returns:
        Validated DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If schema validation fails
    """
    import subprocess

    # Check if file is DVC-tracked
    dvc_file = f"{file_path}.dvc"
    if Path(dvc_file).exists():
        print(f"ANALYSIS: DVC-tracked file detected: {dvc_file}")

        # Check DVC status
        try:
            dvc_status_result = subprocess.run(
                ['dvc', 'status', file_path],
                capture_output=True, text=True
            )
            if dvc_status_result.returncode == 0 and dvc_status_result.stdout.strip():
                print(f"⚠ DVC file may need updating: {file_path}")
            else:
                print(f"SUCCESS: DVC file up to date: {file_path}")
        except FileNotFoundError:
            print("⚠ DVC not available - skipping status check")

    # Load and validate using existing function
    return load_validated_dataset(file_path, schema)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Pandera DataFrame validation...")

    # Test with sample data (using unified naming: _t0 instead of _current)
    test_data = {
        'date': pd.date_range('2022-01-01', periods=100, freq='D'),
        'sales': np.random.uniform(50000, 200000, 100),
        'sales_target_t0': np.random.uniform(50000, 200000, 100),
        'prudential_rate_t0': np.random.uniform(1, 5, 100),
        'competitor_weighted_t2': np.random.uniform(-2, -1, 100),
        'competitor_top5_t3': np.random.uniform(-4, -2, 100),
        'sales_target_contract_t5': np.random.uniform(40000, 180000, 100),
        'weight': np.random.uniform(0.1, 1.0, 100)
    }

    test_df = pd.DataFrame(test_data)

    try:
        validated_df = DataFrameValidator.validate_final_dataset(test_df)
        print(f"✓ Test dataset validation passed: {validated_df.shape}")
    except Exception as e:
        print(f"✗ Test dataset validation failed: {e}")

    # Test forecast results validation
    forecast_test_data = {
        'date': pd.date_range('2023-01-01', periods=50, freq='D'),
        'y_true': np.random.uniform(50000, 200000, 50),
        'y_predict': np.random.uniform(45000, 210000, 50),
        'abs_pct_error': np.random.uniform(0.05, 0.25, 50)  # 5-25% error
    }

    forecast_df = pd.DataFrame(forecast_test_data)

    try:
        validated_forecast = DataFrameValidator.validate_forecast_results(forecast_df)
        print(f"✓ Forecast results validation passed: {validated_forecast.shape}")
    except Exception as e:
        print(f"✗ Forecast results validation failed: {e}")

    # Test schema-aware DVC tracking functionality
    print("\nTesting Schema-Aware DVC Tracking...")

    # Test individual dataset tracking
    try:
        import tempfile
        import os

        # Create temporary output directory
        temp_dir = tempfile.mkdtemp(prefix="schema_dvc_test_")
        test_output_path = os.path.join(temp_dir, "test_dataset.parquet")

        tracking_result = SchemaAwareDVCTracker.validate_and_track_dataset(
            df=test_df,
            output_path=test_output_path,
            schema=FINAL_DATASET_SCHEMA,
            schema_name="test_final_dataset",
            validation_strict=False  # Non-strict for testing
        )

        print(f"✓ Schema-aware DVC tracking test: {tracking_result['validation_status']}")

        # Test bulk tracking
        datasets_config = [
            {
                'df': test_df,
                'output_path': os.path.join(temp_dir, "bulk_test_1.parquet"),
                'schema': FINAL_DATASET_SCHEMA,
                'schema_name': 'bulk_test_1'
            },
            {
                'df': forecast_df,
                'output_path': os.path.join(temp_dir, "bulk_test_2.parquet"),
                'schema': FORECAST_RESULTS_SCHEMA,
                'schema_name': 'bulk_test_2'
            }
        ]

        bulk_results = SchemaAwareDVCTracker.bulk_validate_and_track(datasets_config)
        print(f"✓ Bulk tracking test: {bulk_results['summary']['validation_success_rate']:.1%} success rate")

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"✗ Schema-aware DVC tracking test failed: {e}")

    # Test MLflow integration functionality
    print("\nTesting DataFrameValidator MLflow Integration...")

    try:
        # Test individual MLflow validation (should work without active MLflow run)
        mlflow_validated_df = DataFrameValidator.validate_and_log_to_mlflow(
            df=test_df,
            schema=FINAL_DATASET_SCHEMA,
            schema_name="test_mlflow_validation",
            validation_strict=False,
            log_to_mlflow=False  # Disable MLflow logging for standalone test
        )

        print(f"✓ MLflow integration validation test: {mlflow_validated_df.shape}")

        # Test batch validation with MLflow
        batch_validations = [
            {
                'df': test_df,
                'schema': FINAL_DATASET_SCHEMA,
                'schema_name': 'batch_final_dataset',
                'log_to_mlflow': False  # Disable for testing
            },
            {
                'df': forecast_df,
                'schema': FORECAST_RESULTS_SCHEMA,
                'schema_name': 'batch_forecast_results',
                'log_to_mlflow': False  # Disable for testing
            }
        ]

        batch_results = DataFrameValidator.batch_validate_with_mlflow(batch_validations)
        print(f"✓ Batch MLflow validation test: {batch_results['summary']['validation_success_rate']:.1%} success rate")

    except Exception as e:
        print(f"✗ MLflow integration test failed: {e}")

    print("Pandera DataFrame validation with MLflow and DVC integration test completed!")
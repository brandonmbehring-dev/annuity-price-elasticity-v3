"""
MLflow configuration and safe wrapper functions.

Provides graceful handling of MLflow operations when the library
may or may not be available in the environment.
"""

from typing import Any, Dict, Optional
import warnings

# Defensive MLflow import
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False


def setup_environment_for_notebooks() -> Dict[str, Any]:
    """
    Set up environment configuration for notebook execution.

    Returns:
        Dictionary with environment configuration including MLflow availability.
    """
    return {
        "mlflow_available": MLFLOW_AVAILABLE,
        "tracking_uri": mlflow.get_tracking_uri() if MLFLOW_AVAILABLE else None,
    }


def setup_mlflow_experiment(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """
    Set up an MLflow experiment with the given name.

    Parameters:
        experiment_name: Name of the experiment to create or use
        tracking_uri: Optional tracking URI (uses default if not provided)
        tags: Optional tags to set on the experiment

    Returns:
        Experiment ID if successful, None if MLflow not available
    """
    if not MLFLOW_AVAILABLE:
        warnings.warn(
            f"MLflow not available - experiment '{experiment_name}' not created",
            UserWarning
        )
        return None

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if tags and experiment:
        # Note: MLflow doesn't support setting experiment tags directly after creation
        # Tags would be set on runs instead
        pass

    return experiment.experiment_id if experiment else None


def safe_mlflow_log_param(key: str, value: Any) -> bool:
    """
    Safely log a parameter to MLflow.

    Parameters:
        key: Parameter name
        value: Parameter value

    Returns:
        True if logged successfully, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        return False

    try:
        mlflow.log_param(key, value)
        return True
    except Exception as e:
        warnings.warn(f"Failed to log param '{key}': {e}", UserWarning)
        return False


def safe_mlflow_log_metric(key: str, value: float, step: Optional[int] = None) -> bool:
    """
    Safely log a metric to MLflow.

    Parameters:
        key: Metric name
        value: Metric value
        step: Optional step number for the metric

    Returns:
        True if logged successfully, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        return False

    try:
        mlflow.log_metric(key, value, step=step)
        return True
    except Exception as e:
        warnings.warn(f"Failed to log metric '{key}': {e}", UserWarning)
        return False


def end_mlflow_experiment() -> bool:
    """
    End the current MLflow run.

    Returns:
        True if ended successfully, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        return False

    try:
        mlflow.end_run()
        return True
    except Exception as e:
        warnings.warn(f"Failed to end MLflow run: {e}", UserWarning)
        return False


def safe_mlflow_log_schema_validation(
    df: Any,
    schema_name: str,
    validation_strict: bool = True
) -> Dict[str, Any]:
    """
    Log schema validation results to MLflow.

    Parameters:
        df: DataFrame that was validated
        schema_name: Name of the schema used for validation
        validation_strict: Whether strict validation was used

    Returns:
        Dictionary with validation results and logging status
    """
    results = {
        "schema_name": schema_name,
        "validation_strict": validation_strict,
        "row_count": len(df) if hasattr(df, '__len__') else 0,
        "column_count": len(df.columns) if hasattr(df, 'columns') else 0,
        "logged_to_mlflow": False,
    }

    if not MLFLOW_AVAILABLE:
        results["mlflow_status"] = "mlflow_not_available"
        return results

    try:
        safe_mlflow_log_param(f"schema_{schema_name}_validated", True)
        safe_mlflow_log_param(f"schema_{schema_name}_strict", validation_strict)
        safe_mlflow_log_metric(f"schema_{schema_name}_rows", results["row_count"])
        safe_mlflow_log_metric(f"schema_{schema_name}_cols", results["column_count"])
        results["logged_to_mlflow"] = True
        results["mlflow_status"] = "success"
    except Exception as e:
        results["mlflow_status"] = f"error: {e}"

    return results


def safe_mlflow_log_config_validation(
    config: Dict[str, Any],
    config_name: str,
    validation_passed: bool = True
) -> Dict[str, Any]:
    """
    Log configuration validation results to MLflow.

    Parameters:
        config: Configuration dictionary that was validated
        config_name: Name of the configuration
        validation_passed: Whether validation passed

    Returns:
        Dictionary with validation results and logging status
    """
    results = {
        "config_name": config_name,
        "validation_passed": validation_passed,
        "config_keys": list(config.keys()) if isinstance(config, dict) else [],
        "logged_to_mlflow": False,
    }

    if not MLFLOW_AVAILABLE:
        results["mlflow_status"] = "mlflow_not_available"
        return results

    try:
        safe_mlflow_log_param(f"config_{config_name}_validated", validation_passed)
        safe_mlflow_log_metric(f"config_{config_name}_keys", len(results["config_keys"]))
        results["logged_to_mlflow"] = True
        results["mlflow_status"] = "success"
    except Exception as e:
        results["mlflow_status"] = f"error: {e}"

    return results

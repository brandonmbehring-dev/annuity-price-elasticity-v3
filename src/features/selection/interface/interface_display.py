"""
Display and Formatting Functions for Feature Selection.

This module contains all display formatting, HTML report generation, and
result presentation functions extracted from notebook_interface.py for maintainability.

Module Responsibilities:
- Results summary display (best model, pipeline, constraints, bootstrap)
- Dual validation results display
- HTML report generation
- Comparison results display

Used by: notebook_interface.py (re-exports public functions)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from IPython.display import display, HTML, Markdown

# Import type definitions
from src.features.selection_types import FeatureSelectionResults


# =============================================================================
# RESULTS DISPLAY FUNCTIONS
# =============================================================================


def _display_best_model_summary(best_model: Any, summary: Dict[str, Any]) -> None:
    """
    Display the best model summary section.

    Parameters
    ----------
    best_model : AICResult
        Best model from feature selection
    summary : Dict[str, Any]
        Pipeline summary dictionary
    """
    display(Markdown("## Feature Selection Results"))
    display(
        HTML(
            f"""
    <div style="background: #f0f8ff; padding: 15px; border-radius: 10px; margin: 10px 0;">
        <h3>Best Model</h3>
        <p><strong>Features:</strong> {best_model.features}</p>
        <p><strong>AIC Score:</strong> {best_model.aic:.1f}</p>
        <p><strong>R-squared Score:</strong> {best_model.r_squared:.3f} ({summary['best_model']['model_fit_quality']})</p>
        <p><strong>Model Status:</strong> {'Converged' if best_model.converged else 'Failed to converge'}</p>
    </div>
    """
        )
    )


def _display_pipeline_execution_summary(summary: Dict[str, Any]) -> None:
    """
    Display the pipeline execution summary section.

    Parameters
    ----------
    summary : Dict[str, Any]
        Pipeline summary dictionary
    """
    display(Markdown("### REPORT Pipeline Execution Summary"))
    exec_summary = summary["pipeline_execution"]
    display(
        HTML(
            f"""
    <ul>
        <li><strong>Total Combinations:</strong> {exec_summary['total_combinations_evaluated']}</li>
        <li><strong>Converged Models:</strong> {exec_summary['models_converged']} ({exec_summary['success_rate']})</li>
        <li><strong>Economically Valid:</strong> {summary['pipeline_execution']['economically_valid_models']}</li>
        <li><strong>Execution Time:</strong> {exec_summary.get('execution_time_seconds', 'N/A')}s</li>
    </ul>
    """
        )
    )


def _display_constraint_violations(results: Any) -> None:
    """
    Display economic constraint violations section.

    Parameters
    ----------
    results : FeatureSelectionResults
        Complete pipeline results
    """
    display(Markdown("### WARNING Economic Constraint Violations"))
    display(
        HTML(
            f"<p>Found {len(results.constraint_violations)} constraint violations across all models.</p>"
        )
    )

    violation_types: Dict[str, List[Any]] = {}
    for violation in results.constraint_violations[:5]:
        constraint_type = violation.constraint_type.value
        if constraint_type not in violation_types:
            violation_types[constraint_type] = []
        violation_types[constraint_type].append(violation)

    for constraint_type, violations in violation_types.items():
        display(
            HTML(f"<p><strong>{constraint_type}:</strong> {len(violations)} violations</p>")
        )


def _display_bootstrap_summary(summary: Dict[str, Any]) -> None:
    """
    Display bootstrap stability analysis summary.

    Parameters
    ----------
    summary : Dict[str, Any]
        Pipeline summary dictionary
    """
    display(Markdown("### Bootstrap Stability Analysis"))
    bootstrap_summary = summary.get("bootstrap_analysis", {})
    display(
        HTML(
            f"""
    <ul>
        <li><strong>Models Analyzed:</strong> {bootstrap_summary.get('models_analyzed', 0)}</li>
        <li><strong>Stable Models:</strong> {bootstrap_summary.get('stable_models', 0)}</li>
        <li><strong>Stability Rate:</strong> {bootstrap_summary.get('stability_rate', 'N/A')}</li>
        <li><strong>Top Model Stability:</strong> {bootstrap_summary.get('top_model_stability', 'N/A')}</li>
    </ul>
    """
        )
    )


def _display_business_insights(summary: Dict[str, Any]) -> None:
    """
    Display business insights section.

    Parameters
    ----------
    summary : Dict[str, Any]
        Pipeline summary dictionary
    """
    display(Markdown("### Business Insights"))
    for insight in summary["business_interpretation"]:
        display(HTML(f"<p>{insight}</p>"))


def _display_model_coefficients(best_model: Any) -> None:
    """
    Display model coefficients table.

    Parameters
    ----------
    best_model : AICResult
        Best model from feature selection
    """
    display(Markdown("### Model Coefficients"))
    coeff_data = []
    for feature, coeff in best_model.coefficients.items():
        coeff_data.append(
            {
                "Feature": feature,
                "Coefficient": f"{coeff:.2f}",
                "Impact": "Positive" if coeff > 0 else "Negative" if coeff < 0 else "None",
            }
        )
    coeff_df = pd.DataFrame(coeff_data)
    display(coeff_df)


def display_results_summary(
    results: Any,
    show_detailed: Optional[bool] = None,
    show_violations: bool = True,
    show_bootstrap: bool = True,
    feature_flags: Optional[Dict[str, bool]] = None,
) -> None:
    """Display formatted results summary in notebook-friendly format."""
    # Import here to avoid circular dependencies
    try:
        from src.features.selection.pipeline_orchestrator import create_pipeline_summary
    except ImportError:
        print("WARNING: Could not import pipeline_orchestrator for summary creation")
        return

    # Use default feature flags if not provided
    if feature_flags is None:
        feature_flags = {"SHOW_DETAILED_OUTPUT": True}

    show_detailed = (
        show_detailed
        if show_detailed is not None
        else feature_flags.get("SHOW_DETAILED_OUTPUT", True)
    )

    summary = create_pipeline_summary(results)
    best_model = results.best_model

    _display_best_model_summary(best_model, summary)

    if show_detailed:
        _display_pipeline_execution_summary(summary)

    if show_violations and len(results.constraint_violations) > 0:
        _display_constraint_violations(results)

    if show_bootstrap and results.bootstrap_results:
        _display_bootstrap_summary(summary)

    _display_business_insights(summary)

    if show_detailed and best_model.converged:
        _display_model_coefficients(best_model)


# =============================================================================
# DUAL VALIDATION DISPLAY FUNCTIONS
# =============================================================================


def _display_dual_validation_header() -> None:
    """Display the header section for dual validation results."""
    display(
        HTML(
            """
    <h2 style='color: #2E8B57; font-family: Arial;'>
    Dual Validation Bootstrap Stability Analysis Results
    </h2>
    <h3 style='color: #4169E1;'>
    6-Metric System: AIC + Adjusted R-squared + MAPE (In-Sample and Out-of-Sample)
    </h3>
    """
        )
    )


def _display_dual_validation_metadata(metadata: Dict[str, Any]) -> None:
    """
    Display analysis metadata section.

    Parameters
    ----------
    metadata : Dict[str, Any]
        Analysis metadata dictionary
    """
    wr_weight = metadata.get("integration_weights", {}).get("win_rate_weight", 0.5) * 100
    ir_weight = (
        metadata.get("integration_weights", {}).get("information_ratio_weight", 0.5) * 100
    )
    display(
        HTML(
            f"""
    <div style='background-color: #F0F8FF; padding: 10px; border-radius: 5px; margin: 10px 0;'>
    <strong>Analysis Overview:</strong><br>
    - Models Analyzed: {metadata.get('n_models_analyzed', 'N/A')}<br>
    - Bootstrap Samples: {metadata.get('n_bootstrap_samples', 'N/A')} per model<br>
    - Metrics: {', '.join(metadata.get('metrics_analyzed', []))}<br>
    - Integration: Win Rate ({wr_weight:.0f}%) + Information Ratio ({ir_weight:.0f}%)
    </div>
    """
        )
    )


def _display_dual_validation_best_model(best_model: Dict[str, Any]) -> None:
    """
    Display the best overall model section.

    Parameters
    ----------
    best_model : Dict[str, Any]
        Best model data dictionary
    """
    features_truncated = best_model["model_features"][:80]
    ellipsis = "..." if len(best_model["model_features"]) > 80 else ""
    display(
        HTML(
            f"""
    <div style='background-color: #E6FFE6; padding: 10px; border-radius: 5px; margin: 10px 0;'>
    <strong>Best Overall Model:</strong><br>
    - Model: {best_model['model_name']}<br>
    - Composite Score: {best_model['composite_score']:.1f}/100 ({best_model['stability_grade']})<br>
    - Win Rate: {best_model['win_rate_score']:.1f}% | IR Score: {best_model['ir_score']:.1f}<br>
    - Features: {features_truncated}{ellipsis}
    </div>
    """
        )
    )


def _display_dual_validation_grade_distribution(perf_summary: Dict[str, Any]) -> None:
    """
    Display grade distribution section.

    Parameters
    ----------
    perf_summary : Dict[str, Any]
        Performance summary dictionary
    """
    grade_dist = perf_summary.get("grade_distribution", {})
    display(
        HTML(
            f"""
    <div style='background-color: #FFF8DC; padding: 10px; border-radius: 5px; margin: 10px 0;'>
    <strong>Performance Distribution:</strong><br>
    - Exceptional (A+): {grade_dist.get('A+', 0)} models<br>
    - Excellent (A): {grade_dist.get('A', 0)} models<br>
    - Very Good (B+): {grade_dist.get('B+', 0)} models<br>
    - Good (B): {grade_dist.get('B', 0)} models<br>
    - Total High Quality (A+/A): {perf_summary.get('total_high_quality', 0)} models
    </div>
    """
        )
    )


def _display_dual_validation_top_models_table(top_models: List[Dict[str, Any]]) -> None:
    """
    Display the top 5 models table.

    Parameters
    ----------
    top_models : List[Dict[str, Any]]
        List of top model data dictionaries
    """
    table_rows = ""
    for i, model in enumerate(top_models, 1):
        table_rows += f"""
        <tr>
            <td>{i}</td>
            <td>{model['model_name']}</td>
            <td>{model['composite_score']:.1f}</td>
            <td>{model['stability_grade'].split(' ')[0]}</td>
            <td>{model['win_rate_score']:.1f}%</td>
            <td>{model['ir_score']:.1f}</td>
            <td>{model['in_sample_win_rate']:.1f}%</td>
            <td>{model['out_sample_win_rate']:.1f}%</td>
        </tr>
        """

    display(
        HTML(
            f"""
    <h4 style='color: #4169E1;'>Top 5 Models - Dual Validation Performance</h4>
    <table style='border-collapse: collapse; width: 100%; font-size: 12px;'>
    <thead style='background-color: #4169E1; color: white;'>
        <tr>
            <th style='border: 1px solid #ddd; padding: 8px;'>Rank</th>
            <th style='border: 1px solid #ddd; padding: 8px;'>Model</th>
            <th style='border: 1px solid #ddd; padding: 8px;'>Composite</th>
            <th style='border: 1px solid #ddd; padding: 8px;'>Grade</th>
            <th style='border: 1px solid #ddd; padding: 8px;'>Win Rate</th>
            <th style='border: 1px solid #ddd; padding: 8px;'>IR Score</th>
            <th style='border: 1px solid #ddd; padding: 8px;'>In-Sample</th>
            <th style='border: 1px solid #ddd; padding: 8px;'>Out-Sample</th>
        </tr>
    </thead>
    <tbody>
        {table_rows}
    </tbody>
    </table>
    """
        )
    )


def _display_dual_validation_recommendations(recommendations: Dict[str, Any]) -> None:
    """
    Display recommendations section.

    Parameters
    ----------
    recommendations : Dict[str, Any]
        Recommendations dictionary
    """
    display(
        HTML(
            f"""
    <div style='background-color: #E6F3FF; padding: 10px; border-radius: 5px; margin: 10px 0;'>
    <strong>Recommendations:</strong><br>
    - Primary Recommendation: {recommendations.get('primary_recommendation', 'None')}<br>
    - Confidence Level: {recommendations.get('confidence_level', 'Unknown')}<br>
    - Alternative Models: {', '.join(recommendations.get('alternative_models', []))}
    </div>
    """
        )
    )


def display_dual_validation_results(summary: Dict[str, Any]) -> None:
    """
    Display formatted dual validation results in notebook.

    Orchestrator function: delegates to focused display helpers following CODING_STANDARDS.md.

    Parameters
    ----------
    summary : Dict[str, Any]
        Dual validation summary dictionary
    """
    _display_dual_validation_header()

    metadata = summary.get("analysis_metadata", {})
    _display_dual_validation_metadata(metadata)

    perf_summary = summary.get("performance_summary", {})
    best_model = perf_summary.get("best_overall_model")

    if best_model:
        _display_dual_validation_best_model(best_model)

    _display_dual_validation_grade_distribution(perf_summary)

    top_models = perf_summary.get("top_5_models", [])
    if top_models:
        _display_dual_validation_top_models_table(top_models)

    recommendations = summary.get("recommendations", {})
    _display_dual_validation_recommendations(recommendations)


# =============================================================================
# COMPARISON DISPLAY FUNCTIONS
# =============================================================================


def _display_comparison_results(comparison: Dict[str, Any]) -> None:
    """
    Display validation comparison results.

    Parameters
    ----------
    comparison : Dict[str, Any]
        Comparison results dictionary
    """
    if comparison["validation_passed"]:
        print("SUCCESS: Validation PASSED: Results match original implementation")
    else:
        print("WARNING: Validation Issues Found:")
        for diff in comparison["differences"]:
            print(f"   {diff}")
        if comparison.get("max_aic_difference", 0) > 1e-6:
            print(f"   Maximum AIC difference: {comparison['max_aic_difference']:.2e}")


# =============================================================================
# HTML REPORT FUNCTIONS
# =============================================================================


def _get_report_html_styles() -> str:
    """
    Return CSS styles for the feature selection report.

    Returns
    -------
    str
        CSS style block for HTML report
    """
    return """
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 10px; }
            .section { margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; background: #f8f9fa; }
            .metric { display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 5px; border: 1px solid #ddd; }
            .success { color: #27ae60; }
            .warning { color: #f39c12; }
            .error { color: #e74c3c; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>"""


def _build_report_executive_summary(results: Any, summary: Dict[str, Any]) -> str:
    """
    Build the executive summary section of the HTML report.

    Parameters
    ----------
    results : FeatureSelectionResults
        Complete pipeline results
    summary : Dict[str, Any]
        Pipeline summary dictionary

    Returns
    -------
    str
        HTML string for executive summary section
    """
    return f"""
        <div class="section">
            <h2>REPORT Executive Summary</h2>
            <div class="metric">
                <strong>Best Model:</strong><br>{results.best_model.features}
            </div>
            <div class="metric">
                <strong>AIC Score:</strong><br>{results.best_model.aic:.1f}
            </div>
            <div class="metric">
                <strong>R-squared Score:</strong><br>{results.best_model.r_squared:.3f}
            </div>
            <div class="metric">
                <strong>Model Quality:</strong><br>{summary['best_model']['model_fit_quality']}
            </div>
        </div>"""


def _build_report_analysis_details(summary: Dict[str, Any]) -> str:
    """
    Build the analysis details section of the HTML report.

    Parameters
    ----------
    summary : Dict[str, Any]
        Pipeline summary dictionary

    Returns
    -------
    str
        HTML string for analysis details section
    """
    exec_summary = summary["pipeline_execution"]
    return f"""
        <div class="section">
            <h2>ANALYSIS Analysis Details</h2>
            <ul>
                <li><strong>Total Combinations Evaluated:</strong> {exec_summary['total_combinations_evaluated']}</li>
                <li><strong>Converged Models:</strong> {exec_summary['models_converged']} ({exec_summary['success_rate']})</li>
                <li><strong>Economically Valid Models:</strong> {exec_summary['economically_valid_models']}</li>
                <li><strong>Execution Time:</strong> {exec_summary.get('execution_time_seconds', 'N/A')}s</li>
            </ul>
        </div>

        <div class="section">
            <h2>Economic Constraints</h2>
            <p><strong>Constraints Enabled:</strong> {summary['economic_constraints']['constraints_enabled']}</p>
            <p><strong>Total Violations:</strong> {summary['economic_constraints']['total_violations']}</p>
            <p><strong>Compliance Rate:</strong> {summary['economic_constraints']['constraint_compliance_rate']}</p>
        </div>"""


def _build_report_recommendations_section(summary: Dict[str, Any]) -> str:
    """
    Build the business recommendations section of the HTML report.

    Parameters
    ----------
    summary : Dict[str, Any]
        Pipeline summary dictionary

    Returns
    -------
    str
        HTML string for recommendations section
    """
    insights_html = "".join(
        f"<li>{insight}</li>" for insight in summary["business_interpretation"]
    )
    return f"""
        <div class="section">
            <h2>Business Recommendations</h2>
            <ul>{insights_html}</ul>
        </div>"""


def _build_report_coefficients_table(results: Any) -> str:
    """
    Build the model coefficients table section of the HTML report.

    Parameters
    ----------
    results : FeatureSelectionResults
        Complete pipeline results

    Returns
    -------
    str
        HTML string for coefficients table section
    """
    rows = []
    for feature, coeff in results.best_model.coefficients.items():
        impact = "Positive" if coeff > 0 else "Negative" if coeff < 0 else "None"
        impact_class = "success" if coeff > 0 else "error" if coeff < 0 else ""
        rows.append(
            f'<tr><td>{feature}</td><td>{coeff:.4f}</td><td class="{impact_class}">{impact}</td></tr>'
        )

    return f"""
        <div class="section">
            <h2>Model Coefficients</h2>
            <table>
                <tr><th>Feature</th><th>Coefficient</th><th>Impact</th></tr>
                {"".join(rows)}
            </table>
        </div>"""


def create_feature_selection_report(
    results: Any, output_path: Optional[str] = None
) -> str:
    """Create comprehensive feature selection report for stakeholders."""
    # Import here to avoid circular dependencies
    try:
        from src.features.selection.pipeline_orchestrator import create_pipeline_summary
    except ImportError:
        return "<html><body><h1>Error: Could not generate report</h1></body></html>"

    summary = create_pipeline_summary(results)

    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Feature Selection Analysis Report</title>
        {_get_report_html_styles()}
    </head>
    <body>
        <div class="header">
            <h1>SUCCESS Feature Selection Analysis Report</h1>
            <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        {_build_report_executive_summary(results, summary)}
        {_build_report_analysis_details(summary)}
        {_build_report_recommendations_section(summary)}
        {_build_report_coefficients_table(results)}
    </body>
    </html>
    """

    if output_path:
        with open(output_path, "w") as f:
            f.write(html_report)
        print(f"SUCCESS: Report saved to {output_path}")

    return html_report

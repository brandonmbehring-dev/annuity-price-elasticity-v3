"""
Visualization modules for Annuity Price Elasticity v2.

Exports:
- visualization: Core charting and plotting functions
- inference_plots: Inference result visualizations
- model_comparison: Model comparison dashboards
- business_communication: Business-facing output generation

Note: To avoid circular imports, use submodule imports directly:
    from src.visualization.visualization import plot_aic_model_comparison
    from src.visualization.inference_plots import plot_inference_results
"""

__all__ = [
    "visualization",
    "inference_plots",
    "model_comparison",
]

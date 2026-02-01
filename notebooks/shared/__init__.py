"""
Shared Notebook Utilities for RILA Price Elasticity Analysis.

Provides common setup, validation, and export utilities that are shared
across product-specific notebooks (6Y20B, 1Y10B, etc.).

Design Philosophy (Hybrid Approach):
    - Shared utilities for boilerplate (setup, paths, reproducibility)
    - Explicit educational steps remain in notebooks for clarity
    - Notebooks remain thin orchestrators, not hidden logic

Usage:
    from notebooks.shared import (
        setup_notebook,
        get_output_paths,
        display_section,
        validate_sales_data,
        validate_rates_data,
        ProductNameDict,
    )

    # In notebook cell 1:
    env = setup_notebook(seed=42)

    # Section headers:
    display_section("1. Load Data")
"""

from notebooks.shared.notebook_common import (
    setup_notebook,
    get_output_paths,
    display_section,
    get_project_root,
    setup_sys_path,
    initialize_reproducibility,
    RANDOM_SEED,
)

from notebooks.shared.validation import (
    validate_sales_data,
    validate_rates_data,
    ValidationResult,
)

from notebooks.shared.exports import (
    prepare_bi_export,
    add_product_metadata,
    format_competitor_rates,
    ProductNameDict,
)

__all__ = [
    # Setup
    "setup_notebook",
    "get_output_paths",
    "display_section",
    "get_project_root",
    "setup_sys_path",
    "initialize_reproducibility",
    "RANDOM_SEED",
    # Validation
    "validate_sales_data",
    "validate_rates_data",
    "ValidationResult",
    # Exports
    "prepare_bi_export",
    "add_product_metadata",
    "format_competitor_rates",
    "ProductNameDict",
]

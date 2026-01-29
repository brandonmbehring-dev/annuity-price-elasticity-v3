# Tolerance Reference

## Standard Tolerances

| Tolerance | Value | Use Case |
|-----------|-------|----------|
| STRICT | 1e-12 | Deterministic transforms, data cleaning |
| RATIO | 1e-6 | Percentage calculations, coefficients |
| BOOTSTRAP | 1e-6 to 1e-10 | Stochastic processes, CV results |
| COUNT | 0 (exact) | Row counts, column counts |
| TARGET | 0.00e+00 | Perfect equivalence goal |

## Decision Tree

```
Is the operation deterministic?
├── YES → Use 1e-12 (STRICT)
└── NO (involves randomness)
    └── Can you control the random seed?
        ├── YES → Use 1e-12 with fixed seed
        └── NO → Use 1e-6 to 1e-10 (BOOTSTRAP)

Is it a count or categorical?
├── YES → Use exact (0)
└── NO → Continue to numerical tolerance

Is it a statistical aggregate?
├── YES → Use 1e-6 (RATIO)
└── NO → Use 1e-12 (default)
```

## Why 1e-12?

- Captures floating-point precision limits (~15-16 decimal digits for float64)
- Strict enough to catch real algorithmic differences
- Loose enough to allow legitimate platform/library variations

## When Tests Fail

| Difference | Likely Cause | Action |
|------------|--------------|--------|
| < 1e-14 | Floating-point noise | Usually safe to ignore |
| 1e-14 to 1e-12 | Platform/library version diff | Investigate, may be acceptable |
| 1e-12 to 1e-6 | Algorithmic change | Review code changes carefully |
| 1e-6 to 1e-3 | Significant logic change | Must investigate before proceeding |
| > 1e-3 | Major behavioral change | Stop, root cause required |

## Tolerance Selection by Data Type

| Data Type | Recommended Tolerance |
|-----------|----------------------|
| DataFrame row counts | Exact (0) |
| Column counts | Exact (0) |
| Financial values (premiums, rates) | 1e-12 |
| Statistical metrics (R2, MAPE) | 1e-6 |
| Bootstrap confidence intervals | 1e-6 |
| Coefficients | 1e-10 |
| Encoded categoricals | Exact (0) |

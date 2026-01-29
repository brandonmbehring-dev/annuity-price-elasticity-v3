# Methodology

## Equivalence Testing (Refactoring)

| Document | Purpose |
|----------|---------|
| [EQUIVALENCE_WORKFLOW.md](EQUIVALENCE_WORKFLOW.md) | Step-by-step refactoring process |
| [TOLERANCE_REFERENCE.md](TOLERANCE_REFERENCE.md) | When to use which tolerance |
| [FAILURE_INVESTIGATION.md](FAILURE_INVESTIGATION.md) | Debugging when tests fail |
| [VALIDATOR_SELECTION.md](VALIDATOR_SELECTION.md) | Which validator for which scenario |

## Hub Patterns

Hub patterns have been copied to `knowledge/practices/` for portability:

| Document | Location |
|----------|----------|
| Testing standards | [practices/testing.md](../practices/testing.md) |
| Data leakage prevention | [practices/data_leakage_prevention.md](../practices/data_leakage_prevention.md) |
| Session workflow | [practices/sessions.md](../practices/sessions.md) |
| Burst methodology | [practices/burst.md](../practices/burst.md) |

**Note**: Previously symlinked from `lever_of_archimedes/patterns/`. Now self-contained for portability.

## Quick Reference

### Standard Tolerances

| Type | Value | Use Case |
|------|-------|----------|
| STRICT | 1e-12 | Deterministic transforms |
| RATIO | 1e-6 | Statistical metrics |
| COUNT | 0 | Row/column counts |

### Workflow Summary

1. Capture baseline before changes
2. Write equivalence test (TDD)
3. Implement changes
4. Validate at 1e-12 precision
5. Keep or archive tests

See [EQUIVALENCE_WORKFLOW.md](EQUIVALENCE_WORKFLOW.md) for details.

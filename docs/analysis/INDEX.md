# Analysis

## Overview

Causal framework, feature rationale, and model interpretation for RILA price elasticity.

## Documents

| Document | Purpose |
|----------|---------|
| [BASELINE_MODEL.md](BASELINE_MODEL.md) | Baseline model definition, production model comparison, validation evidence |
| [CAUSAL_FRAMEWORK.md](CAUSAL_FRAMEWORK.md) | Identification strategy, DAG, exclusion restrictions |
| [FEATURE_RATIONALE.md](FEATURE_RATIONALE.md) | Expert-validated feature selection, lag structure |
| [MODEL_INTERPRETATION.md](MODEL_INTERPRETATION.md) | Coefficient interpretation, sign constraints, buffer effects |

## Key Concepts

### Identification Strategy
- Lagged competitor rates as instruments
- Buffer level as confounder control
- Walk-forward CV for temporal validation
- See [CAUSAL_FRAMEWORK.md](CAUSAL_FRAMEWORK.md) for details

### Feature Engineering
- Market-share weighted competitor aggregation
- Multiple lag structure (t-2, t-3, t-4)
- Treasury and VIX as economic controls
- See [FEATURE_RATIONALE.md](FEATURE_RATIONALE.md) for rationale

### Coefficient Interpretation
- Own rate: POSITIVE (yield economics)
- Competitor rates: NEGATIVE (substitution)
- Buffer level modifies sensitivity
- See [MODEL_INTERPRETATION.md](MODEL_INTERPRETATION.md) for guidance

## Related Documentation

- [domain/RILA_ECONOMICS.md](../domain/RILA_ECONOMICS.md) - Product economics context
- [integration/LESSONS_LEARNED.md](../integration/LESSONS_LEARNED.md) - Critical traps
- [practices/LEAKAGE_CHECKLIST.md](../practices/LEAKAGE_CHECKLIST.md) - Pre-deployment validation
- `CODING_STANDARDS.md` Section 7 - Testing framework

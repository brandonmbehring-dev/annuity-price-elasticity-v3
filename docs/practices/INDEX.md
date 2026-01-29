# Practices

## Overview

Coding practices, testing standards, and workflow patterns for RILA price elasticity development.

**Note**: These files are copied from `lever_of_archimedes/patterns/` for portability. This repo is self-contained and can be transferred to another AWS environment without dependencies.

## Documents

### Validation & Quality

| Document | Purpose |
|----------|---------|
| [LEAKAGE_CHECKLIST.md](LEAKAGE_CHECKLIST.md) | **MANDATORY** pre-deployment validation (9 checks) |
| [ANTI_PATTERNS.md](ANTI_PATTERNS.md) | Common mistakes to avoid (4 key + RILA-specific) |
| [data_leakage_prevention.md](data_leakage_prevention.md) | 10 bug categories, shuffled target test |
| [testing.md](testing.md) | 6-layer validation architecture |

### Workflow Patterns

| Document | Purpose |
|----------|---------|
| [burst.md](burst.md) | Brandon Burst 25-minute focus sessions |
| [sessions.md](sessions.md) | CURRENT_WORK.md and SESSION_*.md tracking |
| [git.md](git.md) | Commit format and git safety protocols |

### Automated Tooling

| Tool | Purpose | Command |
|------|---------|---------|
| Pattern Validator | Import hygiene, lag-0 detection | `make pattern-check` |
| Leakage Gates | Automated leakage detection | `make leakage-audit` |
| Baseline Capture | Pre-refactoring outputs | `python scripts/capture_baselines.py` |
| Emergency Rollback | Safe git rollback | `./scripts/emergency-rollback.sh` |

## Quick Reference

### Pre-Deployment Gate

**Run ALL checks before deployment:**

1. Shuffled target test (model fails on shuffled data)
2. Temporal boundary check (no train/test overlap)
3. Competitor lag check (no lag-0)
4. Suspicious results check (improvement <20%)
5. Coefficient sign check (own rate >0, competitor <0)

See [LEAKAGE_CHECKLIST.md](LEAKAGE_CHECKLIST.md) for complete checklist.

### Anti-Patterns to Avoid

1. **Overengineering** - factories for simple problems
2. **Bloated context** - CLAUDE.md should stay <13KB
3. **Silent failures** - NEVER return None on errors
4. **"Too good" results** - >20% improvement = investigate

See [ANTI_PATTERNS.md](ANTI_PATTERNS.md) for details.

### 6-Layer Testing Architecture

| Layer | What | Frequency |
|-------|------|-----------|
| 1 | Unit (pure functions) | Always |
| 2 | Integration (module interfaces) | Always |
| 3 | Contract (interface agreements) | Always |
| 4 | Regression (critical bugs) | After fixes |
| 5 | Property-based (invariants) | Complex logic |
| 6 | Benchmarking (performance) | Optimization |

See [testing.md](testing.md) for implementation details.

## Related Documentation

- [methodology/EQUIVALENCE_WORKFLOW.md](../methodology/EQUIVALENCE_WORKFLOW.md) - Refactoring validation
- [integration/LESSONS_LEARNED.md](../integration/LESSONS_LEARNED.md) - Critical traps
- [analysis/MODEL_INTERPRETATION.md](../analysis/MODEL_INTERPRETATION.md) - Coefficient interpretation
- `CODING_STANDARDS.md` - Project code quality standards
- `docs/EMERGENCY_PROCEDURES.md` - Crisis response procedures
- `docs/validation/leakage_audit_TEMPLATE.md` - Leakage audit template

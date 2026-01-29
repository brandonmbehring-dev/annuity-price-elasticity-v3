# Hub Pattern References

**Purpose**: Document which patterns from `lever_of_archimedes` apply to RILA and any RILA-specific adaptations.

**Hub Location**: `~/Claude/lever_of_archimedes/patterns/`

---

## Applicable Patterns

### Tier 1: High-Impact (Mandatory Reference)

| Pattern | Hub Path | RILA Application | Status |
|---------|----------|------------------|--------|
| **DS/ML Lifecycle** | `ds_ml_lifecycle.md` | Phase-appropriate standards | INTEGRATED |
| **Testing** | `testing.md` | 6-layer validation | PARTIAL |
| **Data Leakage Prevention** | `data_leakage_prevention.md` | 10 bug categories | INTEGRATED |
| **Sessions** | `sessions.md` | CURRENT_WORK.md, session tracking | INTEGRATED |
| **Git** | `git.md` | Commit format, attribution | INTEGRATED |

### Tier 2: Supporting Patterns

| Pattern | Hub Path | RILA Application | Status |
|---------|----------|------------------|--------|
| **Python Style** | `style/python_style.yaml` | Black 100-char, type hints | INTEGRATED |
| **Burst** | `burst.md` | 25-min focused sessions | AVAILABLE |

---

## Pattern Adaptations for RILA

### DS/ML Lifecycle Adaptation

**Current Phase**: Development (not Exploration, not Deployment)

**Phase-specific standards applied**:
- Full testing required (80%+ coverage target)
- 30-50 line functions
- Type hints mandatory
- Immutability by default

**RILA-specific addition**:
- Mathematical equivalence validation at 1e-12 precision
- Notebook output baselines for regression protection

### Data Leakage Prevention Adaptation

**10 Bug Categories Applied to RILA**:

| # | Original Category | RILA Application |
|---|-------------------|------------------|
| 1 | Target Alignment | Train/test date boundaries in walk-forward CV |
| 2 | Future Data in Lags | Competitor features must be lag-1+ |
| 3 | Persistence Implementation | N/A (not using MIDAS) |
| 4 | Feature Selection Target | Selection on first 80% only |
| 5 | Regime Computation | N/A (not using regime switching) |
| 6 | Weights Computation | Market share weights from training period |
| 7 | Walk-Forward Splits | Temporal splits documented in config |
| 8 | Multiple Sources of Truth | Single CLAUDE.md, single CODING_STANDARDS.md |
| 9 | Internal-Only Validation | Synthetic tests planned |
| 10 | Architecture Mismatch | Weekly aggregation documented |

**RILA-specific additions**:
- MC-01: No lag-0 competitors (simultaneity bias)
- MC-02: Positive own-rate coefficient (yield economics)
- MC-13: Feature selection look-ahead prevention

### Testing Pattern Adaptation

**6-Layer Architecture Applied**:

| Layer | Hub Definition | RILA Implementation |
|-------|----------------|---------------------|
| 1. Type Safety | Type hints | mypy checking |
| 2. Input Validation | Preconditions | config_builder validation |
| 3. Unit Tests | Function tests | tests/unit/ |
| 4. Integration Tests | Multi-function | tests/test_*.py |
| 5. End-to-End | Full workflow | Notebook execution |
| 6. Property-Based | Invariants | PLANNED: tests/property_based/ |

**RILA additions**:
- `tests/anti_patterns/` - Domain-specific guards
- Mathematical equivalence tests at 1e-12 precision

---

## Hub Commands Available

From `lever_of_archimedes` slash commands:

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `/prototype` | Exploration mode | Quick iteration |
| `/refactor` | Production standards | Code cleanup |
| `/review` | Deep analysis | Architecture review |
| `/iterate` | Clarification questions | Planning |
| `/letsgo` | Proceed with plan | After iteration |

---

## Cross-Project Lessons

### From FIA Project

See `knowledge/integration/LESSONS_LEARNED.md` for:
- Three critical traps avoided
- Lag structure lessons
- Coefficient sign insights

### From myga-forecasting-v2

See `data_leakage_prevention.md` for:
- 10 leakage bugs discovered
- Verification protocol
- Suspicious results checklist

---

## Quick Reference Commands

```bash
# View hub patterns
cat ~/Claude/lever_of_archimedes/patterns/ds_ml_lifecycle.md
cat ~/Claude/lever_of_archimedes/patterns/data_leakage_prevention.md
cat ~/Claude/lever_of_archimedes/patterns/testing.md

# RILA validation
make quick-check   # 30-second smoke test
make validate      # Full equivalence tests
make test-all      # Full test suite
```

---

## Pattern Version Tracking

| Pattern | Hub Version | Last Sync | Notes |
|---------|-------------|-----------|-------|
| ds_ml_lifecycle.md | 1.0.0 | 2026-01-16 | Initial integration |
| data_leakage_prevention.md | 2025-11-26 | 2026-01-16 | Full 10 bugs |
| testing.md | - | 2026-01-16 | 6-layer architecture |
| sessions.md | - | 2026-01-16 | CURRENT_WORK.md pattern |

---

**Last Updated**: 2026-01-16
**Hub Commit Reference**: Check lever_of_archimedes for latest

# AI Collaboration Methodology

**Last Updated:** 2026-01-30
**Status:** Production documentation for AI-assisted development

---

## Overview

This document describes the principles and practices for effective AI collaboration in the annuity price elasticity codebase. The goal: leverage AI capabilities while maintaining rigorous quality standards.

---

## Core Principles

### 1. Context is King

**1.1** AI performance scales with context quality, not prompt cleverness.

**1.2** Effective context requires:
- Domain knowledge embedded in documentation
- Decision rationale preserved in `.tracking/`
- Patterns documented in `CLAUDE.md`
- Anti-patterns documented in `LESSONS_LEARNED.md`

**1.3** The `knowledge/` directory exists specifically for AI context retrieval.

### 2. Fail-Fast Over Graceful Degradation

**2.1** AI should never silently work around problems.

**2.2** When encountering issues:
- Raise explicit errors with business context
- Never generate synthetic fallbacks
- Surface uncertainty immediately

**2.3** Example - **WRONG**:
```python
try:
    data = load_from_aws()
except Exception:
    logger.warning("Using synthetic data")
    data = generate_synthetic()  # Silent corruption risk
```

**2.4** Example - **CORRECT**:
```python
try:
    data = load_from_aws()
except AWSConnectionError as e:
    raise DataLoadError(
        "Cannot load production data. "
        "Business impact: Analysis will use stale data. "
        "Required action: Check AWS credentials."
    ) from e
```

### 3. Document Uncertainty

**3.1** Use provenance tags for knowledge confidence:
- `[T1]` - Academic/validated
- `[T2]` - Empirical from data
- `[T3]` - Assumption
- `[UNCONFIRMED]` - Requires validation

**3.2** Uncertain areas should be surfaced, not hidden.

---

## Collaboration Patterns

### Pattern 1: Decision Tracking

Every significant decision should be documented with rationale:

```markdown
## Decision: Use TypedDict over dataclass for configurations

**Date**: 2026-01-15
**Context**: Need typed configuration structures
**Decision**: TypedDict for static configs, dataclass for runtime objects
**Rationale**: TypedDict works better with JSON serialization
**Alternatives**: Pydantic (overhead), plain dict (no typing)
```

Location: `.tracking/decisions.md`

### Pattern 2: Constraint Documentation

Economic constraints must be documented with sources:

| Constraint | Sign | Source | Confidence |
|------------|------|--------|------------|
| Own rate coefficient | Positive | Yield economics [T1] | High |
| Competitor coefficient | Negative | Substitution [T1] | High |
| Lag-0 competitors | Forbidden | Causal ID [T1] | Absolute |

Location: `knowledge/domain/` and `CLAUDE.md`

### Pattern 3: Anti-Pattern Prevention

Document what NOT to do, with examples:

```python
# ANTI-PATTERN: Lag-0 competitor features
competitor_rate_t0 = df["competitor_rate"]  # FORBIDDEN

# CORRECT: Use lagged values only
competitor_rate_t1 = df["competitor_rate"].shift(1)
```

Location: `knowledge/practices/ANTI_PATTERNS.md`

---

## AI-Assisted Workflows

### Adding New Features

1. **AI reads context**: CLAUDE.md, MODULE_HIERARCHY.md, relevant knowledge/
2. **AI proposes approach**: Following existing patterns
3. **Human validates**: Checks against constraints
4. **AI implements**: With tests
5. **Human reviews**: Final quality gate

### Debugging Issues

1. **AI traces flow**: Through DI architecture
2. **AI identifies root cause**: With business context
3. **AI proposes fix**: Using canonical patterns
4. **Human validates**: Ensures no new anti-patterns

### Documentation Updates

1. **AI analyzes current state**: Existing docs + code
2. **AI identifies gaps**: Missing context, outdated info
3. **AI proposes updates**: With [UNCONFIRMED] tags where uncertain
4. **Human validates**: Domain knowledge review

---

## Quality Gates

### Before AI-Generated Code Merges

- [ ] Pattern validator passes (`make pattern-check`)
- [ ] No lag-0 competitor features
- [ ] Mathematical equivalence maintained (1e-12)
- [ ] Unit tests written for new code
- [ ] Error handling follows fail-fast principle

### Before AI-Generated Docs Merge

- [ ] Provenance tags present for claims
- [ ] [UNCONFIRMED] tags on uncertain areas
- [ ] Cross-references valid
- [ ] Timestamps updated

---

## Worked Examples

For detailed decision examples, see:
- [ai_examples/ARCHITECTURE_DECISIONS.md](ai_examples/ARCHITECTURE_DECISIONS.md) - DI patterns, registries, exceptions
- [ai_examples/DOMAIN_DECISIONS.md](ai_examples/DOMAIN_DECISIONS.md) - Lag rules, coefficient signs, naming

---

## Context Engineering

### Effective CLAUDE.md Structure

```markdown
# CLAUDE.md

## Project Phase (current status)
## Core Principles (non-negotiable rules)
## Entry Points (where to start)
## Architecture (how pieces fit)
## Economic Constraints (domain rules)
## Quick Commands (common operations)
```

### Knowledge Base Organization

```
knowledge/
├── domain/           # Product economics, schemas
├── analysis/         # Causal framework, feature rationale
├── practices/        # Testing patterns, anti-patterns
└── integration/      # Cross-module concerns, lessons learned
```

### Session Continuity

Use `CURRENT_WORK.md` to maintain context across sessions:

```markdown
## Right Now
[Current task]

## Why
[Business justification]

## Next Step
[Immediate action]

## Context When I Return
[State to resume from]
```

---

## Anti-Patterns in AI Collaboration

### 1. Over-Prompting

**Wrong**: Complex multi-paragraph prompts with edge cases
**Right**: Good context + simple instructions

### 2. Silent Fallbacks

**Wrong**: AI generates synthetic data when real data unavailable
**Right**: AI raises clear errors with business impact

### 3. Orphaned Decisions

**Wrong**: Decisions made without documented rationale
**Right**: All decisions in `.tracking/decisions.md`

### 4. Stale Context

**Wrong**: CLAUDE.md not updated as code evolves
**Right**: Regular context maintenance in sessions

---

## Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Test coverage | >60% | 44% |
| Pattern validator pass | 100% | 100% |
| Leakage gate pass | 100% | 100% |
| Decision doc coverage | >90% | ~70% |

---

## Related Documentation

- [CLAUDE.md](../../CLAUDE.md) - Primary AI guidance
- [docs/development/CLAUDE.md](../development/CLAUDE.md) - Development-specific guidance
- [knowledge/practices/LEAKAGE_CHECKLIST.md](../../knowledge/practices/LEAKAGE_CHECKLIST.md) - Pre-deployment gate
- [knowledge/integration/LESSONS_LEARNED.md](../../knowledge/integration/LESSONS_LEARNED.md) - Critical traps

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-30 | Initial creation | Claude |

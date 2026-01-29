# RILA Knowledge Base

Navigate to focused documents by topic.

---

## New to the Team?

**Start with the onboarding documentation:**

| Step | Document | Time |
|------|----------|------|
| 1 | [Getting Started](../docs/onboarding/GETTING_STARTED.md) | 2 hours |
| 2 | [Mental Model](../docs/onboarding/MENTAL_MODEL.md) | 20 min |
| 3 | [Architecture Walkthrough](../notebooks/onboarding/architecture_walkthrough.ipynb) | 25 min |
| 4 | [Using Claude Code](../docs/onboarding/USING_CLAUDE_CODE.md) | 15 min |

**Then reference as needed:**
- [Common Tasks](../docs/onboarding/COMMON_TASKS.md) - Copy-paste recipes
- [Troubleshooting](../docs/onboarding/TROUBLESHOOTING.md) - Error solutions
- [AWS Setup](../docs/onboarding/AWS_SETUP_GUIDE.md) - Credentials config

**Fundamentals primers** (if needed):
- [Time Series Primer](../docs/fundamentals/TIME_SERIES_PRIMER.md)
- [Econometrics Primer](../docs/fundamentals/ECONOMETRICS_PRIMER.md)
- [Python Best Practices](../docs/fundamentals/PYTHON_BEST_PRACTICES.md)

---

## Quick Links

| Need | Document |
|------|----------|
| Product mechanics | [domain/RILA_ECONOMICS.md](domain/RILA_ECONOMICS.md) |
| Deep RILA mechanics | [domain/RILA_MECHANICS_DEEP.md](domain/RILA_MECHANICS_DEEP.md) |
| Product comparison | [domain/FIXED_DEFERRED_ANNUITY_TAXONOMY.md](domain/FIXED_DEFERRED_ANNUITY_TAXONOMY.md) |
| Terminology | [domain/GLOSSARY.md](domain/GLOSSARY.md) |
| Data schemas | [domain/WINK_SCHEMA.md](domain/WINK_SCHEMA.md), [domain/TDE_SCHEMA.md](domain/TDE_SCHEMA.md) |
| Causal framework | [analysis/CAUSAL_FRAMEWORK.md](analysis/CAUSAL_FRAMEWORK.md) |
| Coefficient interpretation | [analysis/MODEL_INTERPRETATION.md](analysis/MODEL_INTERPRETATION.md) |
| Refactoring workflow | [methodology/EQUIVALENCE_WORKFLOW.md](methodology/EQUIVALENCE_WORKFLOW.md) |
| Critical traps | [integration/LESSONS_LEARNED.md](integration/LESSONS_LEARNED.md) |
| Pre-deployment checklist | [practices/LEAKAGE_CHECKLIST.md](practices/LEAKAGE_CHECKLIST.md) |

## Sections

### [Domain Knowledge](domain/INDEX.md)
Product economics, schemas, crediting methods, competitive analysis, glossary

### [Analysis](analysis/INDEX.md)
Causal framework, feature rationale, model interpretation

### [Methodology](methodology/INDEX.md)
Equivalence testing, validation

### [Integration](integration/INDEX.md)
Cross-product comparison, lessons learned

### [Practices](practices/INDEX.md)
Data leakage prevention, testing, workflow patterns

## Related Documentation

| Document | Purpose |
|----------|---------|
| `CLAUDE.md` | Primary guidance for Claude Code |
| `CODING_STANDARDS.md` | Code quality requirements |
| `QUICK_REFERENCE.md` | Task checklists and workflows |

## Tooling & Scripts

| Tool | Purpose | Usage |
|------|---------|-------|
| `scripts/pattern_validator.py` | Import hygiene, lag-0 detection, constraint compliance | `make pattern-check` |
| `src/validation/leakage_gates.py` | Automated leakage detection gates | `make leakage-audit` |
| `scripts/capture_baselines.py` | Capture outputs for equivalence testing | `python scripts/capture_baselines.py` |
| `scripts/emergency-rollback.sh` | Safe git rollback with backup | `./scripts/emergency-rollback.sh HEAD~1` |
| `docs/EMERGENCY_PROCEDURES.md` | Crisis response guide | Read when issues occur |

## Session Tracking

| File | Purpose |
|------|---------|
| `CURRENT_WORK.md` | Active work context for quick resume |
| `sessions/` | Historical session logs |
| `.tracking/decisions.md` | WHY decisions were made |
| `.tracking/phase_transitions.log` | Project phase changes |

# Documentation Index

**Master navigation for the V3 Annuity Price Elasticity repository.**

---

## Quick Lookup ("I want to...")

| I want to... | Go to |
|--------------|-------|
| Run my first model | [QUICK_START.md](../QUICK_START.md) |
| Understand RILA products | [domain-knowledge/RILA_ECONOMICS.md](domain-knowledge/RILA_ECONOMICS.md) |
| Check for data leakage | [practices/LEAKAGE_CHECKLIST.md](practices/LEAKAGE_CHECKLIST.md) |
| Debug constraint violations | [integration/LESSONS_LEARNED.md](integration/LESSONS_LEARNED.md) |
| Deploy to production | [operations/DEPLOYMENT_CHECKLIST.md](operations/DEPLOYMENT_CHECKLIST.md) |
| Add a new product | [architecture/PRODUCT_EXTENSION_GUIDE.md](architecture/PRODUCT_EXTENSION_GUIDE.md) |
| Look up a term | [domain-knowledge/GLOSSARY.md](domain-knowledge/GLOSSARY.md) |
| Understand the codebase | [onboarding/MENTAL_MODEL.md](onboarding/MENTAL_MODEL.md) |
| Debug a failing test | [onboarding/TROUBLESHOOTING.md](onboarding/TROUBLESHOOTING.md) |
| Write tests | [development/TESTING_GUIDE.md](development/TESTING_GUIDE.md) |
| Understand feature selection | [architecture/FEATURE_SELECTION_DESIGN.md](architecture/FEATURE_SELECTION_DESIGN.md) |
| Validate mathematical equivalence | [methodology/EQUIVALENCE_WORKFLOW.md](methodology/EQUIVALENCE_WORKFLOW.md) |

---

## Reading Paths by Role

### Data Scientist (New to Project)
1. [QUICK_START.md](../QUICK_START.md) - Run your first model
2. [RILA_ECONOMICS.md](domain-knowledge/RILA_ECONOMICS.md) - Understand the product
3. [CAUSAL_FRAMEWORK.md](analysis/CAUSAL_FRAMEWORK.md) - Theoretical foundation
4. [FIRST_MODEL_GUIDE.md](onboarding/FIRST_MODEL_GUIDE.md) - Deeper walkthrough
5. [MENTAL_MODEL.md](onboarding/MENTAL_MODEL.md) - System architecture

### Model Validator
1. [LEAKAGE_CHECKLIST.md](practices/LEAKAGE_CHECKLIST.md) - Pre-deployment gate
2. [EQUIVALENCE_WORKFLOW.md](methodology/EQUIVALENCE_WORKFLOW.md) - Validation process
3. [ANTI_PATTERNS.md](practices/ANTI_PATTERNS.md) - Common mistakes
4. [TOLERANCE_REFERENCE.md](methodology/TOLERANCE_REFERENCE.md) - Precision standards

### Operations / DevOps
1. [DEPLOYMENT_CHECKLIST.md](operations/DEPLOYMENT_CHECKLIST.md) - Production deployment
2. [MONITORING_GUIDE.md](operations/MONITORING_GUIDE.md) - Observability
3. [EMERGENCY_PROCEDURES.md](operations/EMERGENCY_PROCEDURES.md) - Incident response
4. [DATA_QUALITY_MONITORING.md](operations/DATA_QUALITY_MONITORING.md) - Data health

### Software Engineer
1. [CODING_STANDARDS.md](development/CODING_STANDARDS.md) - Code style
2. [MODULE_HIERARCHY.md](development/MODULE_HIERARCHY.md) - Architecture
3. [TESTING_GUIDE.md](development/TESTING_GUIDE.md) - Test patterns
4. [MULTI_PRODUCT_DESIGN.md](architecture/MULTI_PRODUCT_DESIGN.md) - DI architecture

### Business Stakeholder
1. [executive_summary.md](business/executive_summary.md) - High-level overview
2. [methodology_report.md](business/methodology_report.md) - Technical methodology
3. [rai_governance.md](business/rai_governance.md) - Responsible AI

---

## Directory Reference

### analysis/
Causal framework and feature analysis.

| File | Purpose |
|------|---------|
| [BASELINE_MODEL.md](analysis/BASELINE_MODEL.md) | Baseline model specification |
| [CAUSAL_FRAMEWORK.md](analysis/CAUSAL_FRAMEWORK.md) | Econometric theory and identification |
| [FEATURE_RATIONALE.md](analysis/FEATURE_RATIONALE.md) | Why each feature is included |
| [MODEL_INTERPRETATION.md](analysis/MODEL_INTERPRETATION.md) | Coefficient interpretation |

### api/
API and interface documentation.

| File | Purpose |
|------|---------|
| [API_REFERENCE.md](api/API_REFERENCE.md) | Public API documentation |

### architecture/
System design and extension guides.

| File | Purpose |
|------|---------|
| [FEATURE_SELECTION_DESIGN.md](architecture/FEATURE_SELECTION_DESIGN.md) | Feature selection architecture |
| [MULTI_PRODUCT_DESIGN.md](architecture/MULTI_PRODUCT_DESIGN.md) | DI pattern design |
| [PRODUCT_EXTENSION_GUIDE.md](architecture/PRODUCT_EXTENSION_GUIDE.md) | Adding new products |

### business/
Executive-facing documentation.

| File | Purpose |
|------|---------|
| [executive_summary.md](business/executive_summary.md) | Business overview |
| [methodology_report.md](business/methodology_report.md) | Technical methodology |
| [rai_governance.md](business/rai_governance.md) | Responsible AI governance |

### development/
Developer guides and standards.

| File | Purpose |
|------|---------|
| [CODING_STANDARDS.md](development/CODING_STANDARDS.md) | Code style guidelines |
| [MODULE_HIERARCHY.md](development/MODULE_HIERARCHY.md) | Module organization |
| [TESTING_GUIDE.md](development/TESTING_GUIDE.md) | Testing patterns |
| [TECHNICAL_DEBT.md](development/TECHNICAL_DEBT.md) | Known debt items |
| [TEST_COVERAGE_REPORT.md](development/TEST_COVERAGE_REPORT.md) | Coverage tracking |

### domain-knowledge/
Product economics and schemas.

| File | Purpose |
|------|---------|
| [RILA_ECONOMICS.md](domain-knowledge/RILA_ECONOMICS.md) | RILA product economics |
| [CREDITING_METHODS.md](domain-knowledge/CREDITING_METHODS.md) | Cap/participation methods |
| [COMPETITIVE_ANALYSIS.md](domain-knowledge/COMPETITIVE_ANALYSIS.md) | Competitor analysis |
| [GLOSSARY.md](domain-knowledge/GLOSSARY.md) | Term definitions |
| [TDE_SCHEMA.md](domain-knowledge/TDE_SCHEMA.md) | Transaction data schema |
| [WINK_SCHEMA.md](domain-knowledge/WINK_SCHEMA.md) | Wink data schema |

### fundamentals/
Foundational knowledge.

| File | Purpose |
|------|---------|
| [ECONOMETRICS_PRIMER.md](fundamentals/ECONOMETRICS_PRIMER.md) | Econometrics basics |
| [TIME_SERIES_PRIMER.md](fundamentals/TIME_SERIES_PRIMER.md) | Time series fundamentals |
| [PYTHON_BEST_PRACTICES.md](fundamentals/PYTHON_BEST_PRACTICES.md) | Python guidelines |

### integration/
Cross-cutting concerns and lessons.

| File | Purpose |
|------|---------|
| [LESSONS_LEARNED.md](integration/LESSONS_LEARNED.md) | Critical traps and solutions |
| [CROSS_PRODUCT_COMPARISON.md](integration/CROSS_PRODUCT_COMPARISON.md) | Product differences |
| [HUB_PATTERN_REFERENCES.md](integration/HUB_PATTERN_REFERENCES.md) | Pattern references |

### methodology/
Validation and workflow guides.

| File | Purpose |
|------|---------|
| [EQUIVALENCE_WORKFLOW.md](methodology/EQUIVALENCE_WORKFLOW.md) | Mathematical equivalence |
| [TOLERANCE_REFERENCE.md](methodology/TOLERANCE_REFERENCE.md) | Precision standards |
| [VALIDATOR_SELECTION.md](methodology/VALIDATOR_SELECTION.md) | Choosing validators |
| [feature_engineering_guide.md](methodology/feature_engineering_guide.md) | Feature engineering |
| [validation_guidelines.md](methodology/validation_guidelines.md) | Validation approach |

### onboarding/
Getting started guides.

| File | Purpose |
|------|---------|
| [GETTING_STARTED.md](onboarding/GETTING_STARTED.md) | 2-hour onboarding |
| [day_one_checklist.md](onboarding/day_one_checklist.md) | First day tasks |
| [MENTAL_MODEL.md](onboarding/MENTAL_MODEL.md) | System understanding |
| [FIRST_MODEL_GUIDE.md](onboarding/FIRST_MODEL_GUIDE.md) | First model walkthrough |
| [COMMON_TASKS.md](onboarding/COMMON_TASKS.md) | Frequent operations |
| [TROUBLESHOOTING.md](onboarding/TROUBLESHOOTING.md) | Problem resolution |
| [AWS_SETUP_GUIDE.md](onboarding/AWS_SETUP_GUIDE.md) | AWS configuration |

### operations/
Production operations.

| File | Purpose |
|------|---------|
| [DEPLOYMENT_CHECKLIST.md](operations/DEPLOYMENT_CHECKLIST.md) | Deployment steps |
| [MONITORING_GUIDE.md](operations/MONITORING_GUIDE.md) | System monitoring |
| [EMERGENCY_PROCEDURES.md](operations/EMERGENCY_PROCEDURES.md) | Incident response |
| [DATA_QUALITY_MONITORING.md](operations/DATA_QUALITY_MONITORING.md) | Data health |
| [PERFORMANCE_TUNING.md](operations/PERFORMANCE_TUNING.md) | Optimization |

### practices/
Development practices and checklists.

| File | Purpose |
|------|---------|
| [LEAKAGE_CHECKLIST.md](practices/LEAKAGE_CHECKLIST.md) | Data leakage prevention |
| [ANTI_PATTERNS.md](practices/ANTI_PATTERNS.md) | Common mistakes |
| [data_leakage_prevention.md](practices/data_leakage_prevention.md) | Leakage theory |
| [testing.md](practices/testing.md) | 6-layer validation |
| [git.md](practices/git.md) | Git workflow |
| [sessions.md](practices/sessions.md) | Session management |

### reference/
Configuration and technical reference.

| File | Purpose |
|------|---------|
| [CONFIGURATION_REFERENCE.md](reference/CONFIGURATION_REFERENCE.md) | Config options |

### research/
Research and technical deep-dives.

| File | Purpose |
|------|---------|
| [TD05_OLS_vs_Ridge.md](research/TD05_OLS_vs_Ridge.md) | OLS vs Ridge analysis |
| [leakage_audit_TEMPLATE.md](research/leakage_audit_TEMPLATE.md) | Audit template |

---

## Cross-References

- **Project root**: [../README.md](../README.md)
- **Quick start**: [../QUICK_START.md](../QUICK_START.md)
- **CLAUDE.md**: [../CLAUDE.md](../CLAUDE.md)
- **Changelog**: [../CHANGELOG_REFACTORING.md](../CHANGELOG_REFACTORING.md)
- **Contributing**: [../CONTRIBUTING.md](../CONTRIBUTING.md)

---

## Backward Compatibility

Legacy `knowledge/` paths still work via symlinks:
- `knowledge/domain/` → `docs/domain-knowledge/`
- `knowledge/analysis/` → `docs/analysis/`
- `knowledge/integration/` → `docs/integration/`
- `knowledge/methodology/` → `docs/methodology/`
- `knowledge/practices/` → `docs/practices/`

See [../knowledge/README.md](../knowledge/README.md) for details.

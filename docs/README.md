# Documentation Guide

Complete documentation for the RILA Price Elasticity Model, organized by role and project phase.

## Quick Navigation by Role

### For New Data Scientists

**Fast path (5 minutes):**
1. [QUICK_START.md](../QUICK_START.md) - Run your first model

**Complete onboarding (2-4 hours):**
1. [QUICK_START.md](../QUICK_START.md) (5 min)
2. [onboarding/GETTING_STARTED.md](onboarding/GETTING_STARTED.md) (2 hours)
3. [onboarding/COMMON_TASKS.md](onboarding/COMMON_TASKS.md) (30 min)
4. [onboarding/day_one_checklist.md](onboarding/day_one_checklist.md) (interactive)

**Week 1 deep dives:**
- [domain-knowledge/RILA_ECONOMICS.md](domain-knowledge/RILA_ECONOMICS.md) - What is a RILA?
- [analysis/CAUSAL_FRAMEWORK.md](analysis/CAUSAL_FRAMEWORK.md) - Why this model works
- [onboarding/MENTAL_MODEL.md](onboarding/MENTAL_MODEL.md) - System architecture

### For Business Stakeholders

**Executive overview:**
1. [business/executive_summary.md](business/executive_summary.md) - 1-page overview
2. [business/methodology_report.md](business/methodology_report.md) - Technical methodology
3. [business/rai_governance.md](business/rai_governance.md) - RAI000038 compliance

**Strategic context:**
- [domain-knowledge/RILA_ECONOMICS.md](domain-knowledge/RILA_ECONOMICS.md) - Market dynamics
- [analysis/CAUSAL_FRAMEWORK.md](analysis/CAUSAL_FRAMEWORK.md) - Economic theory

### For Model Developers

**Architecture:**
1. [architecture/MULTI_PRODUCT_DESIGN.md](architecture/MULTI_PRODUCT_DESIGN.md) - System design
2. [development/MODULE_HIERARCHY.md](development/MODULE_HIERARCHY.md) - Code organization
3. [development/CODING_STANDARDS.md](development/CODING_STANDARDS.md) - Style guide

**Implementation:**
- [methodology/feature_engineering_guide.md](methodology/feature_engineering_guide.md) - 598 features
- [onboarding/COMMON_TASKS.md](onboarding/COMMON_TASKS.md) - Practical examples
- [development/TECHNICAL_DEBT.md](development/TECHNICAL_DEBT.md) - Known issues

### For Model Validators

**Validation framework:**
1. [practices/LEAKAGE_CHECKLIST.md](practices/LEAKAGE_CHECKLIST.md) - **MANDATORY** pre-deployment
2. [methodology/validation_guidelines.md](methodology/validation_guidelines.md) - Complete workflow
3. [business/methodology_report.md](business/methodology_report.md) - Performance benchmarks

**Emergency procedures:**
- [operations/EMERGENCY_PROCEDURES.md](operations/EMERGENCY_PROCEDURES.md) - Crisis response
- [methodology/FAILURE_INVESTIGATION.md](methodology/FAILURE_INVESTIGATION.md) - Debug guide

---

## Documentation by Project Phase

### Planning & Design

**Understanding the problem:**
- [domain-knowledge/RILA_ECONOMICS.md](domain-knowledge/RILA_ECONOMICS.md) - Product fundamentals
- [analysis/CAUSAL_FRAMEWORK.md](analysis/CAUSAL_FRAMEWORK.md) - Econometric foundation
- [analysis/BASELINE_COMPARISON.md](analysis/BASELINE_COMPARISON.md) - Benchmark models

**System design:**
- [architecture/MULTI_PRODUCT_DESIGN.md](architecture/MULTI_PRODUCT_DESIGN.md) - Multi-product strategy
- [architecture/DATA_ARCHITECTURE.md](architecture/DATA_ARCHITECTURE.md) - Data flow
- [integration/LESSONS_LEARNED.md](integration/LESSONS_LEARNED.md) - Past decisions

### Implementation

**Getting started:**
- [QUICK_START.md](../QUICK_START.md) - 5-minute setup
- [onboarding/GETTING_STARTED.md](onboarding/GETTING_STARTED.md) - Complete onboarding
- [onboarding/day_one_checklist.md](onboarding/day_one_checklist.md) - First day plan

**Development:**
- [development/MODULE_HIERARCHY.md](development/MODULE_HIERARCHY.md) - Code structure
- [development/CODING_STANDARDS.md](development/CODING_STANDARDS.md) - Style guide
- [development/TESTING_GUIDE.md](development/TESTING_GUIDE.md) - Comprehensive testing strategy
- [onboarding/OFFLINE_DEVELOPMENT.md](onboarding/OFFLINE_DEVELOPMENT.md) - Offline development workflow
- [methodology/feature_engineering_guide.md](methodology/feature_engineering_guide.md) - Feature pipeline
- [onboarding/COMMON_TASKS.md](onboarding/COMMON_TASKS.md) - Code examples

### Validation & Deployment

**Pre-deployment validation:**
- [practices/LEAKAGE_CHECKLIST.md](practices/LEAKAGE_CHECKLIST.md) - **MANDATORY** leakage check
- [methodology/validation_guidelines.md](methodology/validation_guidelines.md) - Complete validation
- [business/methodology_report.md](business/methodology_report.md) - Performance requirements

**Deployment:**
- [operations/DEPLOYMENT_GUIDE.md](operations/DEPLOYMENT_GUIDE.md) - Production setup
- [business/rai_governance.md](business/rai_governance.md) - Governance compliance
- [operations/EMERGENCY_PROCEDURES.md](operations/EMERGENCY_PROCEDURES.md) - Emergency contacts

### Maintenance & Operations

**Deployment:**
- [operations/DEPLOYMENT_CHECKLIST.md](operations/DEPLOYMENT_CHECKLIST.md) - **Complete deployment procedures** (manual)
- [practices/LEAKAGE_CHECKLIST.md](practices/LEAKAGE_CHECKLIST.md) - **MANDATORY** pre-deployment validation

**Monitoring:**
- [operations/MONITORING_GUIDE.md](operations/MONITORING_GUIDE.md) - AWS CloudWatch monitoring setup
- [operations/DATA_QUALITY_MONITORING.md](operations/DATA_QUALITY_MONITORING.md) - Data quality gates
- [operations/PERFORMANCE_TUNING.md](operations/PERFORMANCE_TUNING.md) - Optimization strategies

**Incident Response:**
- [operations/EMERGENCY_PROCEDURES.md](operations/EMERGENCY_PROCEDURES.md) - **Critical incident playbooks**
- [methodology/FAILURE_INVESTIGATION.md](methodology/FAILURE_INVESTIGATION.md) - Debugging guide
- [onboarding/TROUBLESHOOTING.md](onboarding/TROUBLESHOOTING.md) - Common issues

**Evolution:**
- [integration/LESSONS_LEARNED.md](integration/LESSONS_LEARNED.md) - Historical context
- [development/TECHNICAL_DEBT.md](development/TECHNICAL_DEBT.md) - Known issues
- [analysis/FUTURE_ENHANCEMENTS.md](analysis/FUTURE_ENHANCEMENTS.md) - Roadmap

---

## Directory Structure

### `/docs/analysis/` - Economic & Statistical Analysis
Technical analysis documenting the econometric foundation and model design decisions.

- `CAUSAL_FRAMEWORK.md` - Why price elasticity models work for RILAs
- `FEATURE_RATIONALE.md` - Economic justification for 598 features
- `BASELINE_COMPARISON.md` - Benchmark model performance
- `FUTURE_ENHANCEMENTS.md` - Planned improvements

### `/docs/architecture/` - System Design
High-level system architecture and design patterns.

- `MULTI_PRODUCT_DESIGN.md` - Multi-product abstraction strategy
- `DATA_ARCHITECTURE.md` - Data pipeline and storage
- `BOOTSTRAP_METHODOLOGY.md` - Uncertainty quantification approach

### `/docs/business/` - Stakeholder Documentation
Non-technical documentation for business stakeholders and governance.

- `executive_summary.md` - 1-page business overview
- `methodology_report.md` - Complete technical methodology
- `rai_governance.md` - RAI000038 compliance documentation

### `/docs/development/` - Developer Guidelines
Code organization, standards, and development practices.

- `MODULE_HIERARCHY.md` - Package and module structure
- `CODING_STANDARDS.md` - Python style guide
- `TESTING_GUIDE.md` - Comprehensive testing strategy (unit, integration, E2E, property-based, performance)
- `TECHNICAL_DEBT.md` - Known issues and workarounds

### `/docs/domain-knowledge/` - Domain Context
Business domain knowledge essential for model development.

- `RILA_ECONOMICS.md` - How RILAs work
- `COMPETITIVE_LANDSCAPE.md` - Market dynamics
- `SALES_DRIVERS.md` - Factors affecting RILA sales

### `/docs/integration/` - Project History
Historical context and integration lessons.

- `LESSONS_LEARNED.md` - Past decisions and rationale
- `MIGRATION_NOTES.md` - v1 to v2 migration

### `/docs/methodology/` - Technical Methodology
Detailed technical procedures and methodologies.

- `feature_engineering_guide.md` - 598-feature pipeline
- `validation_guidelines.md` - Complete validation framework
- `FAILURE_INVESTIGATION.md` - Debugging procedures

### `/docs/onboarding/` - New User Guides
Step-by-step guides for new team members with curated learning paths.

- `USER_JOURNEYS.md` - **Guided documentation pathways by role** (NEW - 8 journeys)
- `GETTING_STARTED.md` - 2-hour complete onboarding
- `OFFLINE_DEVELOPMENT.md` - **Complete offline development workflow** (fixtures, testing, AWS integration)
- `day_one_checklist.md` - Interactive first-day checklist
- `FIRST_MODEL_GUIDE.md` - Hands-on model training walkthrough
- `NOTEBOOK_QUICKSTART.md` - Notebook usage guide
- `COMMON_TASKS.md` - Practical code examples
- `MENTAL_MODEL.md` - System architecture overview
- `TROUBLESHOOTING.md` - Common issues and solutions

### `/docs/operations/` - Production Operations [WARN] **CRITICAL**
Production deployment, monitoring, and incident response procedures.

- `DEPLOYMENT_CHECKLIST.md` - **Complete manual deployment procedures** (920 lines)
- `MONITORING_GUIDE.md` - AWS CloudWatch monitoring and alerting (800+ lines)
- `EMERGENCY_PROCEDURES.md` - Incident response playbooks (1073 lines)
- `PERFORMANCE_TUNING.md` - Bootstrap optimization and AWS scaling (500+ lines)
- `DATA_QUALITY_MONITORING.md` - Data quality gates and drift detection (400+ lines)

### `/docs/practices/` - Best Practices
Critical practices and checklists for model development.

- `LEAKAGE_CHECKLIST.md` - **MANDATORY** data leakage prevention
- `TESTING_STRATEGY.md` - Test coverage approach

---

## Quick Reference

### Most Important Documents

**Before deploying any model:**
1. [practices/LEAKAGE_CHECKLIST.md](practices/LEAKAGE_CHECKLIST.md) - **MANDATORY**
2. [methodology/validation_guidelines.md](methodology/validation_guidelines.md)

**Understanding the system:**
1. [business/executive_summary.md](business/executive_summary.md) - Start here
2. [onboarding/MENTAL_MODEL.md](onboarding/MENTAL_MODEL.md) - System overview
3. [architecture/MULTI_PRODUCT_DESIGN.md](architecture/MULTI_PRODUCT_DESIGN.md) - Design

**Daily development:**
1. [onboarding/OFFLINE_DEVELOPMENT.md](onboarding/OFFLINE_DEVELOPMENT.md) - Offline workflow
2. [onboarding/COMMON_TASKS.md](onboarding/COMMON_TASKS.md) - Code examples
3. [development/MODULE_HIERARCHY.md](development/MODULE_HIERARCHY.md) - Where is what
4. [development/TESTING_GUIDE.md](development/TESTING_GUIDE.md) - Testing strategy
5. [development/CODING_STANDARDS.md](development/CODING_STANDARDS.md) - Style

### Key Performance Metrics

**RILA 6Y20B (Production):**
- R²: 78.37% (benchmark: 57.54%)
- MAPE: 12.74% (benchmark: 16.40%)
- 95% CI Coverage: 94.4%

See [business/methodology_report.md](business/methodology_report.md) for details.

---

## Contributing to Documentation

When adding or updating documentation:

1. **Choose the right directory** based on audience and purpose
2. **Link liberally** to related documents
3. **Use concrete examples** over abstract theory
4. **Keep it current** - update as code changes
5. **Test your examples** - ensure code works

For questions about documentation structure, see [development/CODING_STANDARDS.md](development/CODING_STANDARDS.md).

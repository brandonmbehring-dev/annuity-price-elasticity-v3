# Documentation User Journeys

**Purpose**: Guided documentation pathways for different user roles and goals
**Last Updated**: 2026-01-29
**Target Audiences**: New data scientists, business stakeholders, validators, developers, operators

---

## Overview

This document provides curated documentation journeys for different roles and use cases. Each journey is designed to get you to your goal efficiently, with time estimates and clear next steps.

**How to Use This Guide**:
1. Find your role or goal in the list below
2. Follow the numbered steps in order
3. Time estimates help you plan your day
4. Each journey is self-contained and tested

---

## Quick Navigation

**I want to...**

- [Run a price elasticity model ASAP](#journey-1-run-model-asap) → 10 minutes
- [Onboard as a new data scientist](#journey-2-new-data-scientist-onboarding) → 2-4 hours
- [Understand the business context](#journey-3-business-stakeholder-evaluation) → 1 hour
- [Validate model before production](#journey-4-production-deployment-validation) → 3-4 hours
- [Develop new features](#journey-5-feature-development) → 2-3 hours
- [Respond to a production incident](#journey-6-production-incident-response) → 10-30 minutes
- [Deploy a model to production](#journey-7-model-deployment) → 3-4 hours
- [Monitor production models](#journey-8-production-monitoring) → 15-30 minutes daily

---

## Journey 1: Run Model ASAP

**Goal**: Generate price elasticity predictions in 10 minutes
**Role**: Anyone who needs predictions quickly
**Time**: 10 minutes
**Prerequisites**: AWS access configured, SageMaker notebook running

### Step 1: Quick Start (5 minutes)
**Read**: [../../QUICK_START.md](../../QUICK_START.md)
- Sets up environment
- Verifies data access
- Runs first model

### Step 2: Run Inference Notebook (5 minutes)
**Execute**: `notebooks/production/rila_6y20b/01_price_elasticity_inference.ipynb`
- Loads production model
- Generates predictions with 95% CI
- Exports elasticity curve

### Success Criteria
- [ ] Predictions generated for current week
- [ ] Elasticity curve chart displays
- [ ] Output CSV exported to `outputs/production/rila_6y20b/`

### Next Steps
- For deeper understanding → [Journey 2](#journey-2-new-data-scientist-onboarding)
- For common tasks → [COMMON_TASKS.md](COMMON_TASKS.md)
- For troubleshooting → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## Journey 2: New Data Scientist Onboarding

**Goal**: Comprehensive onboarding from zero to productive contributor
**Role**: New data scientist joining the team
**Time**: 2-4 hours (can split across multiple sessions)
**Prerequisites**: AWS access, SageMaker notebook, basic Python/pandas knowledge

### Step 1: Getting Started (30 minutes)
**Read**: [GETTING_STARTED.md](GETTING_STARTED.md)
- System overview and architecture
- Environment setup
- Data sources and pipeline
- Key concepts (bootstrap ensemble, Ridge regression, AIC selection)

### Step 2: Day One Checklist (1 hour)
**Complete**: [day_one_checklist.md](day_one_checklist.md)
- Interactive checklist with hands-on tasks
- Verify environment setup
- Run first model
- Explore key notebooks
- Test basic operations

### Step 3: Mental Model (20 minutes)
**Read**: [MENTAL_MODEL.md](MENTAL_MODEL.md)
- How the system fits together
- Key design decisions and rationale
- Common workflows
- Where to find things

### Step 4: First Model Guide (1 hour hands-on)
**Work Through**: [FIRST_MODEL_GUIDE.md](FIRST_MODEL_GUIDE.md)
- Step-by-step model training
- Feature engineering walkthrough
- Bootstrap ensemble configuration
- Validation and interpretation

### Step 5: Common Tasks Reference (30 minutes)
**Browse**: [COMMON_TASKS.md](COMMON_TASKS.md)
- Everyday operations
- Code snippets you'll use often
- Quick reference for future tasks

### Success Criteria
- [ ] Can run production inference notebook independently
- [ ] Understand bootstrap ensemble and why we use it
- [ ] Can explain AIC-based feature selection
- [ ] Know where to find documentation for specific tasks
- [ ] Comfortable navigating repository structure

### Next Steps
- Deep dive into methodology → [../business/methodology_report.md](../business/methodology_report.md)
- Learn validation procedures → [../practices/LEAKAGE_CHECKLIST.md](../practices/LEAKAGE_CHECKLIST.md)
- Explore architecture → [../architecture/MULTI_PRODUCT_DESIGN.md](../architecture/MULTI_PRODUCT_DESIGN.md)

---

## Journey 3: Business Stakeholder Evaluation

**Goal**: Understand model methodology, performance, and business value
**Role**: Business stakeholder, product manager, or executive
**Time**: 1 hour
**Prerequisites**: None (business-focused, minimal technical detail)

### Step 1: Executive Summary (5 minutes)
**Read**: [../business/executive_summary.md](../business/executive_summary.md)
- What the model does
- Business value and ROI
- Key performance metrics
- Current status and products

### Step 2: Methodology Report (45 minutes)
**Read**: [../business/methodology_report.md](../business/methodology_report.md)
- **Focus on**: Sections 1-2 (Objective, Background) and Section 8 (Results & Strategic Applications)
- Business objectives and strategic context
- Model approach (high-level, non-technical)
- Performance metrics and validation
- Strategic applications for rate-setting

### Step 3: Responsible AI Governance (10 minutes)
**Skim**: [../business/rai_governance.md](../business/rai_governance.md)
- Compliance documentation (RAI000038)
- Model risk management
- Validation framework
- Quarterly review procedures

### Success Criteria
- [ ] Can explain model business value to others
- [ ] Understand key performance metrics (R², MAPE, coverage)
- [ ] Know how model informs rate-setting decisions
- [ ] Aware of governance and validation procedures

### Next Steps
- For business review meetings → [../business/executive_summary.md](../business/executive_summary.md) (bring to meetings)
- For technical deep-dive → [Journey 2](#journey-2-new-data-scientist-onboarding)
- For deployment status → [../operations/MONITORING_GUIDE.md](../operations/MONITORING_GUIDE.md)

---

## Journey 4: Production Deployment Validation

**Goal**: Validate model meets all requirements before production deployment
**Role**: Model validator, QA engineer, or deployment reviewer
**Time**: 3-4 hours
**Prerequisites**: Model candidate ready for deployment, validation access

### Step 1: Validation Guidelines (30 minutes)
**Read**: [../methodology/validation_guidelines.md](../methodology/validation_guidelines.md)
- Complete validation framework
- Performance metric thresholds
- Economic constraint validation
- Temporal stability requirements

### Step 2: Leakage Checklist (1 hour - MANDATORY)
**Execute**: [../practices/LEAKAGE_CHECKLIST.md](../practices/LEAKAGE_CHECKLIST.md)
- Run ALL 9 mandatory checks
- Shuffled target test
- Temporal boundary check
- Competitor lag check
- Coefficient sign check
- Out-of-sample validation
- Complete sign-off form

[WARN] **CRITICAL**: Any failure = BLOCK deployment. Do not skip checks.

### Step 3: Run Validation Tests (1 hour)
**Execute**:
```bash
# Run complete validation suite
pytest tests/validation/ -v

# Run leakage gates
make leakage-audit

# Run baseline validation
pytest tests/baselines/ -v
```

### Step 4: Deployment Checklist (1 hour)
**Follow**: [../operations/DEPLOYMENT_CHECKLIST.md](../operations/DEPLOYMENT_CHECKLIST.md)
- Pre-deployment validation (Phases 1-4)
- Data quality checks
- Business logic validation
- Stakeholder review and sign-off

### Success Criteria
- [ ] All leakage checks PASS
- [ ] All automated tests PASS
- [ ] Performance metrics exceed thresholds (R² > 50%, MAPE < 20%)
- [ ] Economic constraints validated (correct coefficient signs)
- [ ] Deployment checklist completed and signed off

### Next Steps
- If validation passed → [Journey 7](#journey-7-model-deployment)
- If validation failed → [../operations/EMERGENCY_PROCEDURES.md](../operations/EMERGENCY_PROCEDURES.md)
- For monitoring setup → [../operations/MONITORING_GUIDE.md](../operations/MONITORING_GUIDE.md)

---

## Journey 5: Feature Development

**Goal**: Develop new features or modify existing codebase
**Role**: Software engineer, ML engineer, or data scientist contributor
**Time**: 2-3 hours initial learning, then ongoing development
**Prerequisites**: Repository cloned, development environment set up

### Step 1: Module Hierarchy (20 minutes)
**Read**: [../development/MODULE_HIERARCHY.md](../development/MODULE_HIERARCHY.md)
- Codebase organization
- Module responsibilities
- Import patterns
- Where to add new code

### Step 2: API Reference (30 minutes)
**Read**: [../api/API_REFERENCE.md](../api/API_REFERENCE.md)
- Public APIs for each module
- Function signatures and parameters
- Return types and examples
- Integration patterns

### Step 3: Coding Standards (30 minutes)
**Read**: [../development/CODING_STANDARDS.md](../development/CODING_STANDARDS.md)
- Code style guide
- Naming conventions
- Documentation requirements
- Error handling patterns
- Example compliance audit

### Step 4: Testing Guide (30 minutes)
**Read**: [../development/TESTING_GUIDE.md](../development/TESTING_GUIDE.md)
- Test structure and organization
- Writing new tests
- Running test suites
- Test coverage requirements

### Step 5: Hands-On Development (ongoing)
**Workflow**:
1. Create feature branch from `main`
2. Implement feature following coding standards
3. Write tests (unit + integration)
4. Run validation: `pytest tests/ -v`
5. Update documentation if public API changes
6. Create pull request with description

### Success Criteria
- [ ] Understand module organization
- [ ] Can navigate API reference
- [ ] Code follows style guide
- [ ] Tests written and passing
- [ ] Documentation updated

### Next Steps
- For complex features → [../architecture/MULTI_PRODUCT_DESIGN.md](../architecture/MULTI_PRODUCT_DESIGN.md)
- For troubleshooting → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- For deployment → [Journey 7](#journey-7-model-deployment)

---

## Journey 6: Production Incident Response

**Goal**: Respond to and resolve production incidents quickly
**Role**: On-call engineer, model owner, or incident responder
**Time**: 10-30 minutes (depends on severity)
**Prerequisites**: Production access, ability to execute rollback

### Step 1: Identify Incident Type (2 minutes)
**Quick Reference**: [../operations/EMERGENCY_PROCEDURES.md](../operations/EMERGENCY_PROCEDURES.md) (Quick Reference table)
- Model predictions fail → P0, rollback immediately
- MAPE > 25% sustained → P0, rollback + investigate
- Data pipeline failure → P1, use workaround
- Infrastructure outage → P1, restore access

### Step 2: Execute Immediate Response (5-15 minutes)
**Follow**: Incident-specific playbook in [../operations/EMERGENCY_PROCEDURES.md](../operations/EMERGENCY_PROCEDURES.md)

**For model failures (P0)**:
1. Notify stakeholders (use email template)
2. Execute rollback procedure
3. Verify rollback success
4. Communicate restoration

**For data failures (P1)**:
1. Check data freshness/quality
2. Use previous week's data if needed
3. Notify data engineering
4. Document workaround

**For infrastructure issues (P1)**:
1. Check AWS status
2. Verify credentials
3. Use local data cache if needed
4. Escalate to infrastructure team

### Step 3: Root Cause Analysis (after incident resolved)
**Document**: Create incident report
- What happened (timeline)
- Impact (business, technical)
- Root cause
- Resolution steps taken
- Prevention plan

### Step 4: Post-Mortem (within 3 days for P0, 1 week for P1)
**Process**: [../operations/EMERGENCY_PROCEDURES.md](../operations/EMERGENCY_PROCEDURES.md) (Post-Mortem Process section)
- Schedule meeting with stakeholders
- Review incident report
- Identify gaps in procedures
- Update documentation
- Implement preventive measures

### Success Criteria
- [ ] Incident resolved within SLA (P0: 4 hours, P1: 1 day)
- [ ] Stakeholders notified and updated
- [ ] Root cause identified
- [ ] Post-mortem scheduled
- [ ] Documentation updated to prevent recurrence

### Next Steps
- For monitoring → [../operations/MONITORING_GUIDE.md](../operations/MONITORING_GUIDE.md)
- For rollback details → [../operations/DEPLOYMENT_CHECKLIST.md](../operations/DEPLOYMENT_CHECKLIST.md) (Rollback Procedures)

---

## Journey 7: Model Deployment

**Goal**: Deploy validated model to production safely
**Role**: Model owner, deployment engineer
**Time**: 3-4 hours (includes validation, deployment, verification)
**Prerequisites**: Model validated and approved, backup plan ready

### Step 1: Pre-Deployment Validation (2 hours)
**Follow**: [../operations/DEPLOYMENT_CHECKLIST.md](../operations/DEPLOYMENT_CHECKLIST.md) (Phases 1-4)
- Data leakage validation (30 min)
- Model performance validation (45 min)
- Data quality validation (30 min)
- Business logic validation (30 min)

### Step 2: Pre-Deployment Preparation (30 minutes)
**Follow**: [../operations/DEPLOYMENT_CHECKLIST.md](../operations/DEPLOYMENT_CHECKLIST.md) (Phase 5)
- Backup current production model
- Stage new model
- Smoke test staged model
- Verify deployment checklist complete

### Step 3: Production Deployment (15 minutes)
**Follow**: [../operations/DEPLOYMENT_CHECKLIST.md](../operations/DEPLOYMENT_CHECKLIST.md) (Phase 6)
- Deploy to production (atomic swap)
- Update production symlinks
- Log deployment

### Step 4: Post-Deployment Validation (30 minutes)
**Follow**: [../operations/DEPLOYMENT_CHECKLIST.md](../operations/DEPLOYMENT_CHECKLIST.md) (Phase 7)
- Immediate post-deployment tests
- Sanity check predictions
- Compare new vs. old model outputs
- Generate deployment report

### Step 5: Monitoring Setup (15 minutes)
**Follow**: [../operations/DEPLOYMENT_CHECKLIST.md](../operations/DEPLOYMENT_CHECKLIST.md) (Phase 8)
- Enable performance monitoring
- Document baseline metrics
- Schedule first review (1 week post-deployment)
- Set up alert thresholds

### Success Criteria
- [ ] All validation checks PASS
- [ ] Production model deployed successfully
- [ ] Post-deployment tests PASS
- [ ] Predictions reasonable (< 20% change from previous)
- [ ] Monitoring configured
- [ ] Stakeholders notified

### Next Steps
- Week 1: Daily monitoring → [../operations/MONITORING_GUIDE.md](../operations/MONITORING_GUIDE.md)
- Week 2: Business review meeting
- Ongoing: Biweekly refresh cycle

### Rollback Plan
If issues arise:
- Execute rollback within 15 minutes
- Follow [../operations/DEPLOYMENT_CHECKLIST.md](../operations/DEPLOYMENT_CHECKLIST.md) (Rollback Procedures)
- Notify stakeholders immediately
- Schedule post-mortem

---

## Journey 8: Production Monitoring

**Goal**: Monitor production model health and performance
**Role**: Model owner, operations engineer
**Time**: 15-30 minutes daily, 1 hour biweekly
**Prerequisites**: Production model deployed, monitoring access

### Daily Monitoring (15 minutes)
**Follow**: [../operations/MONITORING_GUIDE.md](../operations/MONITORING_GUIDE.md) (Daily section)
- Check AWS CloudWatch for infrastructure issues
- Verify S3 bucket accessibility
- Review CloudWatch logs for errors
- Check data freshness (TDE, WINK)

### Weekly Monitoring (30 minutes)
**Follow**: [../operations/MONITORING_GUIDE.md](../operations/MONITORING_GUIDE.md) (Weekly section)
- Compare predictions vs. actuals
- Calculate weekly MAPE and bias
- Check confidence interval coverage
- Review feature distributions
- Update weekly monitoring dashboard

### Biweekly Full Validation (1 hour)
**Follow**: [../operations/MONITORING_GUIDE.md](../operations/MONITORING_GUIDE.md) (Biweekly section)
- Full model performance validation
- Rolling 13-week MAPE analysis
- Coefficient stability check
- Data quality validation
- Feature drift detection
- Generate biweekly monitoring report

### Alert Response
**If issues detected**:
- Follow [../operations/EMERGENCY_PROCEDURES.md](../operations/EMERGENCY_PROCEDURES.md)
- Use incident-specific playbooks
- Escalate per severity level (P0/P1/P2)

### Success Criteria
- [ ] Daily checks completed consistently
- [ ] Weekly MAPE < 20%
- [ ] No unresolved P0/P1 alerts
- [ ] Biweekly reports generated and shared
- [ ] Model performance stable (no drift)

### Next Steps
- For drift issues → [../operations/DATA_QUALITY_MONITORING.md](../operations/DATA_QUALITY_MONITORING.md)
- For performance issues → [../operations/PERFORMANCE_TUNING.md](../operations/PERFORMANCE_TUNING.md)
- For incidents → [../operations/EMERGENCY_PROCEDURES.md](../operations/EMERGENCY_PROCEDURES.md)

---

## Role-Based Quick Links

### Data Scientist
**Core Reading**:
1. [GETTING_STARTED.md](GETTING_STARTED.md)
2. [MENTAL_MODEL.md](MENTAL_MODEL.md)
3. [FIRST_MODEL_GUIDE.md](FIRST_MODEL_GUIDE.md)
4. [COMMON_TASKS.md](COMMON_TASKS.md)

**Reference**:
- [../methodology/feature_engineering_guide.md](../methodology/feature_engineering_guide.md)
- [../methodology/validation_guidelines.md](../methodology/validation_guidelines.md)
- [../api/API_REFERENCE.md](../api/API_REFERENCE.md)

### Business Stakeholder
**Core Reading**:
1. [../business/executive_summary.md](../business/executive_summary.md)
2. [../business/methodology_report.md](../business/methodology_report.md) (focus on Sections 1-2, 8)

**Reference**:
- [../business/rai_governance.md](../business/rai_governance.md)
- [../operations/MONITORING_GUIDE.md](../operations/MONITORING_GUIDE.md)

### Model Validator
**Core Reading**:
1. [../practices/LEAKAGE_CHECKLIST.md](../practices/LEAKAGE_CHECKLIST.md) [WARN] MANDATORY
2. [../methodology/validation_guidelines.md](../methodology/validation_guidelines.md)
3. [../operations/DEPLOYMENT_CHECKLIST.md](../operations/DEPLOYMENT_CHECKLIST.md)

**Reference**:
- [../development/TESTING_GUIDE.md](../development/TESTING_GUIDE.md)
- [../operations/EMERGENCY_PROCEDURES.md](../operations/EMERGENCY_PROCEDURES.md)

### Software Engineer
**Core Reading**:
1. [../development/MODULE_HIERARCHY.md](../development/MODULE_HIERARCHY.md)
2. [../development/CODING_STANDARDS.md](../development/CODING_STANDARDS.md)
3. [../api/API_REFERENCE.md](../api/API_REFERENCE.md)
4. [../development/TESTING_GUIDE.md](../development/TESTING_GUIDE.md)

**Reference**:
- [../architecture/MULTI_PRODUCT_DESIGN.md](../architecture/MULTI_PRODUCT_DESIGN.md)
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

### Operations / Model Owner
**Core Reading**:
1. [../operations/DEPLOYMENT_CHECKLIST.md](../operations/DEPLOYMENT_CHECKLIST.md)
2. [../operations/MONITORING_GUIDE.md](../operations/MONITORING_GUIDE.md)
3. [../operations/EMERGENCY_PROCEDURES.md](../operations/EMERGENCY_PROCEDURES.md)

**Reference**:
- [../operations/PERFORMANCE_TUNING.md](../operations/PERFORMANCE_TUNING.md)
- [../operations/DATA_QUALITY_MONITORING.md](../operations/DATA_QUALITY_MONITORING.md)
- [../practices/LEAKAGE_CHECKLIST.md](../practices/LEAKAGE_CHECKLIST.md)

---

## Tips for Using This Guide

### Time Management
- **Short on time?** Start with Journey 1 (10 minutes)
- **Half day available?** Complete Journey 2 Steps 1-3
- **Full day available?** Complete entire Journey 2 or Journey 4

### Learning Styles
- **Hands-on learner?** Jump to [FIRST_MODEL_GUIDE.md](FIRST_MODEL_GUIDE.md)
- **Top-down learner?** Start with [MENTAL_MODEL.md](MENTAL_MODEL.md)
- **Reference-oriented?** Use [COMMON_TASKS.md](COMMON_TASKS.md)

### Getting Help
- **Stuck?** Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Emergency?** Use [../operations/EMERGENCY_PROCEDURES.md](../operations/EMERGENCY_PROCEDURES.md)
- **Questions?** Ask team, check documentation, or create an issue

---

## Feedback

Help us improve these journeys:
- Which journey did you follow?
- How long did it actually take?
- What was unclear or missing?
- What could be improved?

Document feedback in repository issues or share with model owner.

---

## Related Documentation

### Getting Started
- [GETTING_STARTED.md](GETTING_STARTED.md) - Complete onboarding
- [QUICK_START.md](../../QUICK_START.md) - 5-minute quick start
- [day_one_checklist.md](day_one_checklist.md) - Interactive checklist

### Operations
- [../operations/DEPLOYMENT_CHECKLIST.md](../operations/DEPLOYMENT_CHECKLIST.md) - Deployment procedures
- [../operations/MONITORING_GUIDE.md](../operations/MONITORING_GUIDE.md) - Monitoring and alerting
- [../operations/EMERGENCY_PROCEDURES.md](../operations/EMERGENCY_PROCEDURES.md) - Incident response

### Business Context
- [../business/executive_summary.md](../business/executive_summary.md) - Executive overview
- [../business/methodology_report.md](../business/methodology_report.md) - Technical methodology

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-29 | Initial user journeys documentation |

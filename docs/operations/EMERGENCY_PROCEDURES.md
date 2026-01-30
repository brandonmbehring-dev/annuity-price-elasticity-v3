# Emergency Procedures and Incident Response

**Purpose**: Critical incident response procedures for production RILA price elasticity models
**Last Updated**: 2026-01-29
**Status**: Production operational procedures
**Scope**: Production incidents, data quality failures, model failures, infrastructure outages

---

## Overview

This document provides comprehensive emergency response procedures for critical incidents affecting production RILA price elasticity models. All procedures emphasize safety, clear communication, and systematic resolution with post-mortem learning.

**Emergency Philosophy**:
- **Safety First**: Prevent business harm before fixing root cause
- **Clear Communication**: Keep stakeholders informed
- **Systematic Resolution**: Follow documented procedures
- **Learn and Improve**: Every incident improves the system

**Response Time Targets**:
- **CRITICAL (P0)**: Acknowledge within 15 minutes, response within 1 hour
- **HIGH (P1)**: Acknowledge within 1 hour, response within 4 hours
- **MEDIUM (P2)**: Acknowledge within 4 hours, response within 1 business day

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Incident Severity Levels](#incident-severity-levels)
3. [Production Model Incidents](#production-model-incidents)
4. [Data Pipeline Failures](#data-pipeline-failures)
5. [Infrastructure Outages](#infrastructure-outages)
6. [Data Leakage Response](#data-leakage-response)
7. [Rollback Procedures](#rollback-procedures)
8. [Communication Procedures](#communication-procedures)
9. [Post-Mortem Process](#post-mortem-process)
10. [Recovery Verification](#recovery-verification)

---

## Quick Reference

| Emergency Type | Severity | First Action | Detailed Section |
|----------------|----------|--------------|------------------|
| Model predictions fail | P0 | Rollback to previous model | [Production Model Incidents](#production-model-incidents) |
| MAPE > 25% sustained | P0 | Halt business consumption + rollback | [Production Model Incidents](#production-model-incidents) |
| Data leakage detected | P0 | HALT model immediately | [Data Leakage Response](#data-leakage-response) |
| Coefficient signs flip | P0 | Economic validation failure, rollback | [Production Model Incidents](#production-model-incidents) |
| TDE data pipeline down | P1 | Use previous week's data | [Data Pipeline Failures](#data-pipeline-failures) |
| WINK data missing | P1 | Manual competitor rate entry | [Data Pipeline Failures](#data-pipeline-failures) |
| AWS S3 access failure | P1 | Check credentials, use local backup | [Infrastructure Outages](#infrastructure-outages) |
| MAPE 20-25% single week | P2 | Investigate, monitor | [Production Model Incidents](#production-model-incidents) |
| Test failures after merge | P2 | Identify commit with bisect | [Rollback Procedures](#rollback-procedures) |
| Import errors | P2 | Check dependencies | [Infrastructure Outages](#infrastructure-outages) |

---

## Incident Severity Levels

### P0 - CRITICAL (Production Down)

**Definition**: Production model unavailable or producing unsafe outputs that could cause business harm

**Response Time**: Acknowledge within 15 minutes, begin response within 1 hour

**Examples**:
- Model predictions fail to generate
- MAPE > 25% sustained for 2+ weeks
- Data leakage confirmed
- Coefficient signs flip (economic constraints violated)
- Predictions consistently outside [0, 5000] range

**Response Team**:
- Incident Commander: Model Owner
- Technical Lead: Model Owner
- Business Liaison: Rate Setting Team Lead
- Communications: Model Risk Team

**Communication**:
- Immediate notification to Rate Setting Team
- Hourly updates until resolved
- Post-mortem within 3 business days

**Resolution Target**: Within 4 hours (may involve rollback)

### P1 - HIGH (Degraded Performance)

**Definition**: Production model functioning but with degraded performance or data quality issues

**Response Time**: Acknowledge within 1 hour, begin response within 4 hours

**Examples**:
- MAPE 20-25% for single week
- Data pipeline delays (> 7 days stale)
- AWS infrastructure degradation (high CPU/memory)
- Competitor data missing for 1-2 carriers
- Confidence intervals miscalibrated (coverage < 85% or > 98%)

**Response Team**:
- Incident Lead: Model Owner
- Support: Data Engineering (if data issue)
- Notification: Rate Setting Team (advisory)

**Communication**:
- Notification to stakeholders within 2 hours
- Daily updates until resolved
- Brief post-mortem (1 page summary)

**Resolution Target**: Within 1 business day

### P2 - MEDIUM (Minor Issues)

**Definition**: Non-critical issues that don't immediately impact production but require attention

**Response Time**: Acknowledge within 4 hours, begin response within 1 business day

**Examples**:
- Test failures in development
- Single-week MAPE 15-20%
- Feature distribution shifts (1.5-2.0σ)
- Documentation gaps discovered
- Non-critical dependency issues

**Response Team**:
- Owner: Model Owner
- Escalate only if worsens

**Communication**:
- Document in monitoring log
- Mention in next biweekly review
- No formal post-mortem unless pattern emerges

**Resolution Target**: Within 3 business days

---

## Production Model Incidents

### Incident Type: Model Predictions Fail (P0)

**Symptoms**:
- Python exceptions when running inference notebook
- Model.predict() returns NaN or None
- Output files not generated
- Jupyter kernel crashes during prediction

**Immediate Response** (within 15 minutes):

**Step 1: Confirm Incident**
```bash
# Attempt to load production model
cd ${REPO_ROOT}

python -c "
import sys
sys.path.insert(0, '.')
from src.models.inference import PriceElasticityInference

try:
    model = PriceElasticityInference.load('rila_6y20b')
    print('✓ Model loads successfully')
except Exception as e:
    print(f'✗ Model load FAILED: {e}')
    sys.exit(1)
"
```

**Step 2: Notify Stakeholders**
```
Subject: [P0 INCIDENT] RILA 6Y20B Model Predictions Failing

The production RILA 6Y20B price elasticity model is currently unable
to generate predictions due to [ERROR TYPE].

Status: INCIDENT RESPONSE IN PROGRESS
Impact: Weekly sales forecasts unavailable
ETA: Rollback within 1 hour, root cause analysis ongoing

Incident Commander: [Model Owner Name]
Next Update: [Time + 1 hour]

DO NOT use model outputs for rate-setting decisions until further notice.
```

**Step 3: Execute Rollback** (see [Manual Rollback Process](#manual-rollback-process-15-minutes))

**Step 4: Verify Rollback Success**
```python
# Verify rolled-back model works
from src.models.inference import PriceElasticityInference

rolled_back_model = PriceElasticityInference.load("rila_6y20b")

# Generate test prediction
import pandas as pd
test_features = pd.read_csv('data/processed/latest_features.csv').iloc[-1:]

prediction = rolled_back_model.predict(test_features)

print(f"✓ Rollback successful: Prediction = {prediction['mean'].iloc[0]:.0f}")
print(f"Restored version: {rolled_back_model.metadata['version']}")
```

**Root Cause Analysis** (within 4 hours):
- Review deployment log for changes
- Check CloudWatch logs for exceptions
- Verify data pipeline integrity
- Test model with sample data locally
- Document findings in incident report

**Follow-Up Actions**:
- Schedule post-mortem (within 3 business days)
- Update deployment checklist if gap found
- Add automated smoke tests to prevent recurrence

---

### Incident Type: MAPE > 25% Sustained (P0)

**Symptoms**:
- Weekly MAPE exceeds 25% for 2+ consecutive weeks
- 13-week rolling MAPE > 20% and increasing
- Systematic over/under-prediction (bias > 15%)

**Immediate Response** (within 1 hour):

**Step 1: Validate Measurements**
```python
# Verify MAPE calculation is correct
from src.monitoring.performance import calculate_weekly_metrics

# Recalculate from source data
actuals = pd.read_parquet('data/processed/tde_actuals.csv')
predictions = pd.read_csv('outputs/production/rila_6y20b/latest_predictions.csv')

metrics = calculate_weekly_metrics(predictions, actuals, window_weeks=2)

print(f"Confirmed MAPE: {metrics['mape']:.2%}")
print(f"Prediction bias: {metrics['bias']:.2%}")
print(f"Sample size: {len(actuals)} weeks")

# Check if data quality issue (e.g., incomplete actuals)
if len(actuals) < 2:
    print("⚠️  WARNING: Insufficient data for MAPE calculation")
```

**Step 2: Determine Root Cause**

**Possible Causes**:

**A. Data Quality Issue** (most common):
```bash
# Check data freshness
python -c "
import pandas as pd
from datetime import datetime, timedelta

actuals = pd.read_parquet('data/processed/tde_actuals.csv')
latest_date = actuals['date'].max()
days_old = (datetime.now() - latest_date).days

if days_old > 14:
    print(f'✗ Data stale ({days_old} days old)')
    print('Action: Refresh TDE data before rollback decision')
else:
    print(f'✓ Data fresh ({days_old} days old)')
"
```

**B. Market Regime Change**:
- Check for major economic events (Fed rate changes, market volatility)
- Review competitor rate movements (unusual activity?)
- Verify VIX, Treasury rates within normal bounds

**C. Model Drift**:
```python
# Check coefficient stability
current_model = PriceElasticityInference.load("rila_6y20b")
coef_history = pd.read_csv('outputs/monitoring/coefficient_log.csv')

# Compare current vs. historical coefficients
for feat in current_model.feature_names_:
    current_coef = current_model.get_coefficients()[feat]
    historical_mean = coef_history[feat].mean()
    pct_change = (current_coef - historical_mean) / historical_mean

    if abs(pct_change) > 0.50:
        print(f"⚠️  {feat}: {pct_change:.1%} change from historical mean")
```

**Step 3: Decision Tree**

```
Is data quality issue?
├─ YES → Fix data pipeline, rerun predictions
│         (Do NOT rollback model)
│
└─ NO → Is market regime change?
        ├─ YES → Keep model, document regime change
        │         Monitor for another 2 weeks
        │         (Do NOT rollback unless > 30% MAPE)
        │
        └─ NO → Model drift detected
                → ROLLBACK to previous model
                → Schedule emergency retraining session
```

**Step 4: Communication**

**If Rollback Needed**:
```
Subject: [P0 INCIDENT] RILA 6Y20B Model Rollback Due to Performance Degradation

The production RILA 6Y20B model has been rolled back due to sustained
forecast accuracy degradation (MAPE > 25% for 2 weeks).

Root Cause: [Model drift / Data quality / Market regime]
Rollback Status: COMPLETE
Restored Version: [Previous version]
Current Status: Production predictions available using previous model

Impact:
- Rate-setting guidance available (previous model)
- Forecast accuracy: [Previous model MAPE]
- Retraining scheduled: [Date]

Next Steps:
- Root cause analysis: [ETA]
- Model retraining: [ETA]
- Re-deployment: [ETA]

Incident Commander: [Name]
Next Update: [Time]
```

---

### Incident Type: Coefficient Signs Flip (P0)

**Symptoms**:
- Own rate coefficient becomes negative (β < 0)
- Competitor rate coefficient becomes positive (β > 0)
- Economic constraint validation fails

**This is ALWAYS a critical error** - indicates model has lost economic validity

**Immediate Response** (within 30 minutes):

**Step 1: Confirm Economic Violation**
```python
from src.models.inference import PriceElasticityInference

model = PriceElasticityInference.load("rila_6y20b")
coefs = model.get_coefficients()

# Check economic constraints
violations = []

if coefs.get('prudential_rate_current', 1) < 0:
    violations.append(f"Own rate negative: {coefs['prudential_rate_current']:.4f}")

comp_features = [k for k in coefs.keys() if 'competitor' in k.lower()]
for feat in comp_features:
    if coefs[feat] > 0:
        violations.append(f"Competitor rate positive: {feat} = {coefs[feat]:.4f}")

if violations:
    print("✗ CRITICAL: Economic constraints violated!")
    for v in violations:
        print(f"  - {v}")
else:
    print("✓ Economic constraints satisfied (false alarm)")
```

**Step 2: IMMEDIATE Rollback** (No Investigation First)
```bash
# This is a safety issue - rollback immediately
echo "CRITICAL: Economic constraints violated - executing immediate rollback"

# See [Manual Rollback Process](#manual-rollback-process-15-minutes)
# Execute steps 1-4 immediately
```

**Step 3: Halt Business Consumption**
```
Subject: [P0 CRITICAL] RILA 6Y20B Model UNSAFE - Immediate Rollback

The production RILA 6Y20B model has failed economic validation checks.
Model coefficients have wrong signs, indicating loss of economic validity.

ACTION REQUIRED: DO NOT USE current model outputs for ANY decisions.

Rollback Status: IN PROGRESS (ETA: 15 minutes)
Restored Version: [Previous safe version]

This is a safety issue - rate-setting decisions should use previous
model outputs until further notice.

Incident Commander: [Name]
Model Risk: [Name] (notified)
Next Update: Upon rollback completion
```

**Step 4: Root Cause Investigation** (after rollback)

**Common Causes**:
1. **Data corruption** - Check recent data loads for anomalies
2. **Feature engineering bug** - Lag structure may have been broken
3. **Training data contamination** - Leakage or data quality issue
4. **Bootstrap sampling issue** - Unlucky sample composition (rare)

**Investigation Script**:
```python
# Investigate root cause after rollback
from src.validation.economic_constraints import validate_coefficients

# Re-run training with diagnostics
from src.models.training import train_with_diagnostics

diagnostics = train_with_diagnostics(
    training_data=latest_training_data,
    validation_data=latest_validation_data
)

print("Training Diagnostics:")
print(f"  Coefficient signs: {diagnostics['coefficient_signs']}")
print(f"  Feature correlations: {diagnostics['feature_correlations']}")
print(f"  Data quality: {diagnostics['data_quality']}")
print(f"  Bootstrap stability: {diagnostics['bootstrap_stability']}")
```

**Step 5: Mandatory Post-Mortem**
- Schedule within 24 hours (not 3 days)
- Include Model Risk Team
- Update leakage checklist if gap found
- Add automated economic constraint check to deployment pipeline

---

### Incident Type: Predictions Outside Valid Range (P0)

**Symptoms**:
- Predictions < 0 (impossible)
- Predictions > 5000 contracts/week (extreme outlier)
- Confidence interval bounds include negative values

**Immediate Response** (within 30 minutes):

**Step 1: Confirm Invalid Predictions**
```python
predictions = pd.read_csv('outputs/production/rila_6y20b/latest_predictions.csv')

# Check bounds
invalid_predictions = predictions[
    (predictions['mean'] < 0) |
    (predictions['mean'] > 5000) |
    (predictions['lower'] < 0)
]

if len(invalid_predictions) > 0:
    print(f"✗ CRITICAL: {len(invalid_predictions)} invalid predictions detected")
    print(invalid_predictions[['date', 'mean', 'lower', 'upper']])
else:
    print("✓ All predictions within valid range")
```

**Step 2: IMMEDIATE Rollback**
- Follow [Manual Rollback Process](#manual-rollback-process-15-minutes)
- This is a safety issue - rollback before investigation

**Step 3: Root Cause Investigation** (after rollback)

**Common Causes**:
1. **Log transformation issue** - Model predicts log(sales), exponentiation may overflow
2. **Feature values out of range** - Input rates outside [0.005, 0.045] training bounds
3. **Bootstrap aggregation bug** - Mean/CI calculation error
4. **Data type error** - Integer overflow or float precision issue

**Investigation**:
```python
# Check input features for anomalies
latest_features = pd.read_csv('data/processed/latest_features.csv')

print("Feature Ranges:")
for col in latest_features.columns:
    if 'rate' in col.lower():
        print(f"  {col}: [{latest_features[col].min():.4f}, {latest_features[col].max():.4f}]")

# Expected: rates between 0.005 and 0.045
# If outside this range, data quality issue
```

---

## Data Pipeline Failures

### Incident Type: TDE Sales Data Unavailable (P1)

**Symptoms**:
- S3 bucket empty or inaccessible
- Data refresh job fails
- Most recent data > 14 days old

**Response** (within 4 hours):

**Step 1: Assess Data Staleness**
```bash
# Check S3 bucket
aws s3 ls s3://pruvpcaws031-east/rila/sales/latest/ --profile cross-account

# Check local copy
ls -lh data/raw/tde_sales/latest/

# Determine age of most recent data
python -c "
import pandas as pd
from datetime import datetime

try:
    sales = pd.read_parquet('data/processed/tde_sales_latest.parquet')
    latest = sales['application_signed_date'].max()
    days_old = (datetime.now() - latest).days
    print(f'Most recent sales data: {latest} ({days_old} days old)')
except Exception as e:
    print(f'Cannot load sales data: {e}')
"
```

**Step 2: Decision Tree**

```
Data age < 7 days?
├─ YES → Use existing data for this refresh cycle
│         Notify Rate Setting Team (no immediate action)
│
└─ NO → Data age 7-14 days?
        ├─ YES → Use existing data WITH WARNING
        │         Flag predictions as "based on stale data"
        │         Escalate to data engineering
        │
        └─ NO → Data age > 14 days
                → CRITICAL: Cannot refresh model
                → Use previous week's predictions
                → Escalate to P1 incident
```

**Step 3: Workaround Procedures**

**Use Previous Week's Predictions** (if data > 14 days old):
```bash
# Copy previous week's outputs
cp outputs/production/rila_6y20b/predictions/2026-01-22.csv \
   outputs/production/rila_6y20b/latest_predictions.csv

# Add warning flag
echo "WARNING: Using previous week predictions due to data pipeline failure" > \
   outputs/production/rila_6y20b/DATA_STALENESS_WARNING.txt

# Notify stakeholders
cat <<EOF
Subject: [P1] RILA 6Y20B Using Previous Week Predictions

TDE sales data pipeline is currently unavailable (> 14 days stale).

Workaround: Using previous week's predictions for rate-setting guidance.

Data Pipeline Status: [Status from data engineering]
Expected Resolution: [ETA from data engineering]
Business Impact: Moderate (predictions from [date])

Rate Setting Team: Please use previous week's elasticity analysis.
New predictions will be generated once data pipeline restored.

Incident Owner: [Name]
Next Update: [Time]
EOF
```

**Step 4: Escalation to Data Engineering**
- Open ticket with data engineering team
- Provide S3 paths, error messages, recent changes
- Request ETA for data pipeline restoration
- Daily follow-up until resolved

---

### Incident Type: WINK Competitive Data Missing (P1)

**Symptoms**:
- Missing competitor rates for 1+ carriers
- WINK data refresh fails
- Competitor features have NaN values

**Response** (within 4 hours):

**Step 1: Identify Missing Carriers**
```python
# Check carrier coverage
wink_data = pd.read_parquet('data/processed/wink_rates_latest.parquet')

carriers = [
    'Athene', 'Brighthouse', 'Equitable', 'Ameriprise',
    'Jackson', 'Lincoln', 'Symetra', 'Transamerica'
]

recent = wink_data[wink_data['rate_effective_date'] >= datetime.now() - timedelta(days=30)]

for carrier in carriers:
    count = len(recent[recent['carrier'] == carrier])
    if count == 0:
        print(f"✗ MISSING: {carrier} (no data in last 30 days)")
    elif count < 5:
        print(f"⚠️  SPARSE: {carrier} ({count} records)")
    else:
        print(f"✓ OK: {carrier} ({count} records)")
```

**Step 2: Manual Competitor Rate Entry**

**If 1-2 carriers missing**:
```python
# Manual rate entry from public sources
# WINK alternative: carrier websites, rate sheets

manual_rates = {
    'Athene': 0.0325,  # Verified from [source] on [date]
    'Jackson': 0.0310,  # Verified from [source] on [date]
}

# Append to WINK data
from src.data.competitive import append_manual_rates

wink_data_updated = append_manual_rates(
    wink_data=wink_data,
    manual_rates=manual_rates,
    effective_date=datetime.now(),
    source="Manual entry due to WINK outage",
    verified_by="[Your Name]"
)

# Proceed with model refresh using manual data
wink_data_updated.to_parquet('data/processed/wink_rates_latest_manual.parquet')
```

**Documentation Required**:
```
Manual Competitor Rate Entry Log

Date: 2026-01-29
Reason: WINK data pipeline failure
Missing Carriers: Athene, Jackson
Data Source: [Carrier websites / Rate sheets / Other]
Verified By: [Name]
Effective Date: 2026-01-29

Rates Entered:
- Athene 6Y20B: 3.25% (source: athene.com/rates)
- Jackson 6Y20B: 3.10% (source: jackson.com/annuity-rates)

Notes: Manual rates used only for this refresh cycle. WINK pipeline
restoration expected [date]. Predictions flagged as "manual competitor data".
```

**If 3+ carriers missing**:
- **DO NOT** proceed with model refresh
- Competitor data too incomplete for reliable predictions
- Use previous week's predictions (see TDE failure workaround above)
- Escalate to P1 incident

---

## Infrastructure Outages

### Incident Type: AWS S3 Access Failure (P1)

**Symptoms**:
- S3 bucket list commands fail
- Access Denied errors
- Connection timeout errors

**Response** (within 4 hours):

**Step 1: Diagnose Access Issue**
```bash
# Check AWS credentials
aws sts get-caller-identity --profile cross-account

# Expected output: Account ID, UserId, Arn
# If error: Credentials expired or invalid

# Test S3 access specifically
aws s3 ls s3://pruvpcaws031-east/rila/ --profile cross-account

# Common errors:
# - Access Denied: IAM role issue
# - InvalidAccessKeyId: Credentials expired
# - Connection timeout: Network issue
```

**Step 2: Quick Fixes**

**Credentials Expired**:
```bash
# Refresh AWS credentials (cross-account assume role)
# This depends on your AWS SSO / credential setup

# Example (adjust for your setup):
aws sso login --profile cross-account

# Or re-assume role:
aws sts assume-role \
  --role-arn arn:aws:iam::ACCOUNT:role/ROLE_NAME \
  --role-session-name emergency-access \
  --profile default

# Update ~/.aws/credentials with temporary credentials
```

**Network Issue**:
```bash
# Check VPC endpoint connectivity (if using)
curl -I https://s3.us-east-1.amazonaws.com

# Check security group rules
aws ec2 describe-security-groups --group-ids sg-XXXXX
```

**IAM Role Issue**:
```bash
# Verify IAM role has S3 permissions
aws iam get-role-policy \
  --role-name YOUR_ROLE \
  --policy-name S3AccessPolicy
```

**Step 3: Fallback to Local Data**

**If S3 access cannot be restored quickly**:
```bash
# Use local data cache for model operations
export USE_LOCAL_DATA=true

# Verify local data exists
ls -lh data/raw/tde_sales/
ls -lh data/raw/wink_rates/
ls -lh data/processed/

# Run model with local data
python -c "
from src.models.inference import PriceElasticityInference
model = PriceElasticityInference.load_from_local('models/production/rila_6y20b/')
print('✓ Model loaded from local storage')
"
```

**Step 4: Escalation**
- If credentials issue: Contact AWS account administrator
- If network issue: Contact infrastructure team
- If IAM issue: Open AWS support ticket (if permissions required)

---

### Incident Type: SageMaker Instance Failure (P1)

**Symptoms**:
- Jupyter notebook won't load
- Instance status "Failed" in AWS console
- SSH connection refused

**Response** (within 4 hours):

**Step 1: Check Instance Status**
```bash
# From local terminal (not SageMaker)
aws sagemaker describe-notebook-instance \
  --notebook-instance-name [YOUR_INSTANCE_NAME] \
  --region us-east-1

# Check Status field: InService, Stopped, Failed
```

**Step 2: Instance Recovery**

**If Status = "Stopped"**:
```bash
# Simply start the instance
aws sagemaker start-notebook-instance \
  --notebook-instance-name [YOUR_INSTANCE_NAME] \
  --region us-east-1

# Wait 5-10 minutes for startup
```

**If Status = "Failed"**:
```bash
# Stop instance (if not already stopped)
aws sagemaker stop-notebook-instance \
  --notebook-instance-name [YOUR_INSTANCE_NAME] \
  --region us-east-1

# Wait for complete stop
aws sagemaker wait notebook-instance-stopped \
  --notebook-instance-name [YOUR_INSTANCE_NAME]

# Start instance
aws sagemaker start-notebook-instance \
  --notebook-instance-name [YOUR_INSTANCE_NAME]
```

**If Restart Fails**:
- Escalate to AWS support (infrastructure team)
- In meantime: Spin up new SageMaker instance
- Clone repository to new instance
- Continue work (data/models should be in S3, not instance storage)

**Step 3: Data Recovery Check**

**Verify critical files not lost**:
```bash
# After instance restarts, check for work loss
cd ${REPO_ROOT}

# Check git status (uncommitted work may be lost)
git status

# Check recent model artifacts
ls -lh models/production/rila_6y20b/
ls -lh outputs/production/rila_6y20b/

# If files missing: Restore from S3 or git
git reset --hard HEAD  # Discard local changes
git pull origin feature/refactor-eda-notebooks
```

---

## Data Leakage Response

### 1. Git Rollback (Code Issues)

**Use when**: Broken code merged, tests failing, import errors.

```bash
# Preview what will be rolled back
./scripts/emergency-rollback.sh --dry-run HEAD~1

# Execute rollback (creates backup branch first)
./scripts/emergency-rollback.sh HEAD~1

# Or rollback to specific commit
./scripts/emergency-rollback.sh abc1234
```

**The script automatically**:
1. Creates backup branch `backup/[branch]_[timestamp]`
2. Rolls back to target commit
3. Validates imports
4. Shows undo instructions

### 2. Dependency Rollback

**Use when**: Package update breaks functionality.

```bash
# Check current environment
pip freeze > broken_env.txt

# Reinstall from known-good requirements
pip install -r requirements.txt --force-reinstall

# If using pyproject.toml
pip install -e ".[all]" --force-reinstall
```

### 3. Baseline Rollback

**Use when**: Mathematical equivalence tests failing after legitimate change.

```bash
# Capture new baseline (after verifying change is correct)
python scripts/capture_baselines.py --output tests/baselines/

# Or restore from git
git checkout HEAD~1 -- tests/baselines/
```

---

## Leakage Response

### Immediate Actions

**If leakage detected** (R2 > 0.3, shuffled target test fails, etc.):

1. **HALT**: Do not use model for any decisions
2. **Document**: Record what triggered the alert
3. **Investigate**: Follow audit checklist

```bash
# Run automated leakage gates
make leakage-audit

# Or manually
python -c "
from src.validation.leakage_gates import run_all_gates
report = run_all_gates(r_squared=0.35)
print(report)
"
```

### Investigation Checklist

- [ ] Check for lag-0 competitor features: `make pattern-check`
- [ ] Verify train/test temporal split
- [ ] Audit feature engineering for future data
- [ ] Run shuffled target test
- [ ] Review recent changes: `git log --oneline -20`

### Resolution Steps

1. Identify leakage source
2. Fix feature engineering or data split
3. Regenerate features
4. Re-run all leakage gates
5. Document in `.tracking/decisions.md`

---

## Test Failure Triage

### Identifying Breaking Commit

```bash
# Start bisect
git bisect start HEAD <last-known-good-commit>

# Mark current as bad
git bisect bad

# Run test to check each commit
git bisect run pytest tests/test_specific.py -x

# When done
git bisect reset
```

### Common Test Failures

| Failure Type | Likely Cause | Fix |
|--------------|--------------|-----|
| Import errors | Missing dependency | `pip install -e ".[all]"` |
| Baseline mismatch | Code change | Verify change, update baseline |
| Fixture not found | Path issue | Check `tests/fixtures/` |
| AWS credential error | Environment | Use `environment=fixture` |

---

## Data Issues

### Corrupt Fixture Data

```bash
# Regenerate fixtures from source
python scripts/capture_baselines.py --fixtures-only

# Or restore from git
git checkout HEAD~1 -- tests/fixtures/
```

### Missing AWS Data

```bash
# Verify AWS credentials
aws sts get-caller-identity

# Test S3 access
aws s3 ls s3://your-bucket/

# Fallback to local fixtures
# In notebook:
interface = create_interface("6Y20B", environment="fixture")
```

---

## Performance Issues

### Slow Tests

```bash
# Run quick tests only
make quick-check

# Skip slow tests
pytest tests/ -m "not slow"

# Profile slow test
pytest tests/test_slow.py --profile
```

### Memory Issues

```bash
# Identify memory hogs
python -m memory_profiler scripts/your_script.py

# Reduce test data size
pytest tests/ --fixtures-small
```

---

## Communication Checklist

### When to Escalate

- [ ] Production model affected
- [ ] Data leakage confirmed (not just suspected)
- [ ] Multiple team members blocked
- [ ] Cannot rollback cleanly

### What to Document

In `.tracking/decisions.md`:
```markdown
## YYYY-MM-DD: [Emergency Type]

**What happened**: [Brief description]
**Impact**: [Who/what was affected]
**Root cause**: [Why it happened]
**Resolution**: [What was done]
**Prevention**: [How to prevent recurrence]
```

---

## Recovery Verification

After any emergency resolution:

```bash
# Full validation suite
make verify

# Quick smoke test
make quick-check

# Run leakage gates
make leakage-audit

# Check pattern compliance
make pattern-check
```

---

## Contacts

| Role | Contact | When to Contact |
|------|---------|-----------------|
| Data Owner | [Name] | Data quality issues |
| ML Lead | [Name] | Model/methodology questions |
| DevOps | [Name] | Infrastructure issues |

---

## Related Documents

- `CLAUDE.md` - Project conventions
- `knowledge/practices/LEAKAGE_CHECKLIST.md` - Pre-deployment checklist
- `knowledge/integration/LESSONS_LEARNED.md` - Historical issues
- `.tracking/decisions.md` - Decision log

# Production Deployment Checklist

**Purpose**: MANDATORY validation and deployment procedures for RILA price elasticity models
**Last Updated**: 2026-01-29
**Status**: REQUIRED - Complete ALL checks before production deployment
**Deployment Type**: Manual deployment with manual rollback capability

---

## Overview

This checklist ensures safe, validated deployment of price elasticity models to production. All checks must pass before deployment approval. This document covers manual deployment procedures currently in use.

**Total Time Required**: 3-4 hours for complete deployment cycle

---

## Pre-Deployment Validation (MANDATORY)

### Phase 1: Data Leakage Validation (30 minutes)

**CRITICAL**: Run the complete leakage checklist BEFORE any deployment preparation.

See [../practices/LEAKAGE_CHECKLIST.md](../practices/LEAKAGE_CHECKLIST.md) for complete details.

**Required Checks**:
- [ ] Shuffled target test (model should FAIL on random targets)
- [ ] Temporal boundary check (no future data in training)
- [ ] Competitor lag check (minimum 2-week lag enforced)
- [ ] Suspicious results check (performance not "too good")
- [ ] Feature construction audit (lags computed within splits)
- [ ] Coefficient sign check (economic theory validation)
- [ ] Buffer level control (RILA: properly controlled)
- [ ] Market-share weighting (RILA: weighted competitor means)
- [ ] Out-of-sample validation (reasonable degradation)

**Pass Criteria**: ALL checks must pass. Any failure = BLOCK deployment.

**Documentation**: Complete sign-off form in LEAKAGE_CHECKLIST.md

---

### Phase 2: Model Performance Validation (45 minutes)

#### 2.1 Statistical Performance Thresholds

**Minimum Requirements**:
```
R² > 50%              (current production: 78.37%)
MAPE < 20%            (current production: 12.74%)
Coverage: 90-97%      (current production: 94.4%)
```

**Validation Script**:
```python
# Run in: notebooks/production/rila_6y20b/01_price_elasticity_inference.ipynb

from src.models.inference import PriceElasticityInference

# Load latest model
model = PriceElasticityInference.load("rila_6y20b")

# Evaluate performance
metrics = model.evaluate_performance(X_test, y_test)

print(f"R²: {metrics['r2']:.2%}")
print(f"MAPE: {metrics['mape']:.2%}")
print(f"95% CI Coverage: {metrics['coverage']:.2%}")

# Check thresholds
assert metrics['r2'] > 0.50, f"R² too low: {metrics['r2']:.2%}"
assert metrics['mape'] < 0.20, f"MAPE too high: {metrics['mape']:.2%}"
assert 0.90 <= metrics['coverage'] <= 0.97, f"Coverage out of range: {metrics['coverage']:.2%}"
```

**Checklist**:
- [ ] R² exceeds 50% threshold
- [ ] MAPE below 20% threshold
- [ ] 95% CI coverage between 90-97%
- [ ] Volatility-weighted metrics acceptable (degradation < 5%)

#### 2.2 Economic Constraint Validation

**Required Coefficient Signs**:
```
Own rate (Prudential):        β > 0 (quality signaling)
Competitor rate (lagged):     β < 0 (competitive pressure)
Sales momentum (lagged):      β > 0 (contract persistence)
```

**Validation Script**:
```python
# Check coefficient signs
coefs = model.get_coefficients()

# Own rate check
pru_coef = coefs['prudential_rate_current']
assert pru_coef > 0, f"Own rate coefficient wrong sign: {pru_coef}"

# Competitor rate check
comp_coef = coefs['competitor_top5_t2']  # 2-week lag
assert comp_coef < 0, f"Competitor coefficient wrong sign: {comp_coef}"

# Sales momentum check
sales_coef = coefs['sales_target_contract_t5']  # 5-week lag
assert sales_coef > 0, f"Sales momentum coefficient wrong sign: {sales_coef}"

print("✓ All coefficient signs consistent with economic theory")
```

**Checklist**:
- [ ] All coefficients have theoretically expected signs
- [ ] Statistical significance confirmed (p < 0.05)
- [ ] Bootstrap stability check (100% sign consistency across samples)

#### 2.3 Temporal Stability Analysis

**13-Week Rolling MAPE Check**:
```python
# Check for model drift over time
from src.validation.temporal import rolling_mape_analysis

rolling_metrics = rolling_mape_analysis(model, data, window=13)

# Recent performance should be stable
recent_mape = rolling_metrics['mape'].iloc[-13:].mean()
historical_mape = rolling_metrics['mape'].mean()

drift = abs(recent_mape - historical_mape) / historical_mape

assert drift < 0.15, f"Model drift detected: {drift:.1%} increase in MAPE"
print(f"✓ Model stability confirmed (drift: {drift:.1%})")
```

**Checklist**:
- [ ] 13-week rolling MAPE stable (< 15% drift)
- [ ] No systematic under/over-prediction in recent weeks
- [ ] Performance consistent across volatility regimes

---

### Phase 3: Data Quality Validation (30 minutes)

#### 3.1 Data Freshness Check

**Verify Recent Data Availability**:
```bash
# Check TDE sales data
aws s3 ls s3://pruvpcaws031-east/rila/sales/latest/ --profile cross-account

# Check WINK competitive rates
aws s3 ls s3://pruvpcaws031-east/rila/wink/latest/ --profile cross-account

# Verify data is within 7 days old
python -c "
from datetime import datetime, timedelta
import pandas as pd

sales_date = pd.read_parquet('data/processed/sales_latest.parquet')['date'].max()
days_old = (datetime.now() - sales_date).days

assert days_old <= 7, f'Sales data is {days_old} days old (max: 7)'
print(f'✓ Data freshness confirmed: {days_old} days old')
"
```

**Checklist**:
- [ ] TDE sales data within 7 days old
- [ ] WINK competitive rates within 7 days old
- [ ] Economic indicators (FRED) within 2 days old
- [ ] No missing carrier data (all 8 major carriers present)

#### 3.2 Data Completeness Check

**Verify No Missing Critical Features**:
```python
# Check for missing data in recent window
from src.data.validation import check_data_completeness

completeness = check_data_completeness(data, window_weeks=26)

# Critical features must be complete
critical_features = [
    'prudential_rate_current',
    'competitor_top5_t2',
    'sales_target_contract_t5',
    'market_vix',
    'treasury_5yr'
]

for feat in critical_features:
    missing_pct = completeness[feat]['missing_pct']
    assert missing_pct < 0.01, f"{feat} has {missing_pct:.1%} missing data"

print("✓ All critical features complete")
```

**Checklist**:
- [ ] < 1% missing data in critical features (last 26 weeks)
- [ ] No data quality flags in recent observations
- [ ] Market share weights sum to 100% (quarterly verification)
- [ ] Holiday adjustments applied correctly

#### 3.3 Feature Distribution Check

**Verify No Anomalous Feature Values**:
```python
# Check for distribution shifts
from src.validation.data_quality import check_feature_distributions

distribution_checks = check_feature_distributions(
    train_data=train_data,
    recent_data=recent_data,
    features=model.feature_names_
)

# Flag features with > 2 std deviation shift
anomalies = [
    feat for feat, shift in distribution_checks.items()
    if abs(shift) > 2.0
]

if anomalies:
    print(f"⚠️  Warning: Anomalous features detected: {anomalies}")
    print("Review distributions before deployment")
else:
    print("✓ All feature distributions normal")
```

**Checklist**:
- [ ] No features with > 2σ distribution shift
- [ ] Rate ranges within historical bounds (50-450 bps)
- [ ] Sales volumes within expected range
- [ ] No data entry errors detected

---

### Phase 4: Business Logic Validation (30 minutes)

#### 4.1 Scenario Testing

**Test Common Rate Change Scenarios**:
```python
# Test model predictions for standard rate scenarios
scenarios = {
    'rate_cut_50bp': {'prudential_rate_current': current_rate - 0.50},
    'rate_cut_100bp': {'prudential_rate_current': current_rate - 1.00},
    'rate_increase_50bp': {'prudential_rate_current': current_rate + 0.50},
    'rate_increase_100bp': {'prudential_rate_current': current_rate + 1.00},
}

for scenario_name, rate_change in scenarios.items():
    prediction = model.predict_scenario(
        baseline_features=current_features,
        rate_change=rate_change
    )

    print(f"{scenario_name}:")
    print(f"  Expected Sales: {prediction['mean']:.0f}")
    print(f"  95% CI: [{prediction['lower']:.0f}, {prediction['upper']:.0f}]")
    print(f"  Change vs Baseline: {prediction['pct_change']:.1%}")
```

**Expected Behavior**:
- Rate cuts → Higher sales (positive elasticity)
- Rate increases → Lower sales (negative elasticity)
- Prediction intervals widen for larger rate changes
- No predictions outside [0, 5000] contracts/week range

**Checklist**:
- [ ] Rate increase scenarios show sales decrease
- [ ] Rate decrease scenarios show sales increase
- [ ] Elasticity magnitude reasonable (not extreme)
- [ ] Confidence intervals widen appropriately

#### 4.2 Stakeholder Review

**Required Sign-Off**:
- [ ] Model validator review completed
- [ ] Annuity Rate Setting Team briefed on results
- [ ] Model Risk Team notified of deployment
- [ ] Documentation updated (RAI000038 version bump)

**Review Materials**:
1. Model performance summary (metrics, charts)
2. Coefficient economic validation
3. Scenario analysis results
4. Known limitations and assumptions

**Sign-Off Form**:
```
DEPLOYMENT APPROVAL

Product: RILA 6Y20B
Model Version: __________
Deployment Date: __________

Validated By: ______________________ Date: __________
Approved By (Rate Setting): ______________________ Date: __________
Approved By (Model Risk): ______________________ Date: __________

DEPLOYMENT AUTHORIZED: [ ] YES  [ ] NO
```

---

## Deployment Execution (Manual Process)

### Phase 5: Pre-Deployment Preparation (30 minutes)

#### 5.1 Backup Current Production Model

**CRITICAL**: Always backup before deployment to enable rollback.

```bash
# Create timestamped backup
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/home/sagemaker-user/model_backups/rila_6y20b/${BACKUP_DATE}"

mkdir -p ${BACKUP_DIR}

# Backup current production model
cp -r models/production/rila_6y20b/* ${BACKUP_DIR}/

# Backup current production predictions
cp outputs/production/rila_6y20b/latest_predictions.csv ${BACKUP_DIR}/

# Document backup
echo "Backup created: ${BACKUP_DATE}" >> ${BACKUP_DIR}/backup_log.txt
echo "Previous model version: $(cat models/production/rila_6y20b/version.txt)" >> ${BACKUP_DIR}/backup_log.txt

echo "✓ Backup complete: ${BACKUP_DIR}"
```

**Checklist**:
- [ ] Current production model backed up with timestamp
- [ ] Current production predictions backed up
- [ ] Backup directory documented in deployment log
- [ ] Backup tested (can load successfully)

#### 5.2 Stage New Model

**Copy validated model to staging location**:
```bash
# Stage new model
STAGING_DIR="models/staging/rila_6y20b"
mkdir -p ${STAGING_DIR}

# Copy validated model artifacts
cp models/candidate/rila_6y20b/model.pkl ${STAGING_DIR}/
cp models/candidate/rila_6y20b/metadata.json ${STAGING_DIR}/
cp models/candidate/rila_6y20b/feature_names.txt ${STAGING_DIR}/

# Copy validation results
cp validation/rila_6y20b/latest_validation_report.html ${STAGING_DIR}/

# Create version file
NEW_VERSION="v3.1_$(date +%Y%m%d)"
echo ${NEW_VERSION} > ${STAGING_DIR}/version.txt

echo "✓ New model staged: ${NEW_VERSION}"
```

**Checklist**:
- [ ] Model artifacts copied to staging
- [ ] Validation results included
- [ ] Version file created
- [ ] Staging directory ready for promotion

#### 5.3 Smoke Test Staged Model

**Verify staged model loads and runs**:
```python
# Test staged model before production deployment
import sys
sys.path.insert(0, '/home/sagemaker-user/RILA_6Y20B_refactored')

from src.models.inference import PriceElasticityInference

# Load staged model
staged_model = PriceElasticityInference.load_from_path(
    "models/staging/rila_6y20b/model.pkl"
)

# Run smoke tests
print("Running smoke tests...")

# 1. Model loads successfully
assert staged_model is not None, "Model failed to load"
print("✓ Model loads")

# 2. Can generate predictions
test_prediction = staged_model.predict(test_features)
assert test_prediction is not None, "Prediction failed"
print(f"✓ Predictions work: {test_prediction['mean']:.0f}")

# 3. Feature names match training
assert set(staged_model.feature_names_) == set(train_features.columns), "Feature mismatch"
print("✓ Features match")

# 4. Can generate confidence intervals
assert 'lower' in test_prediction and 'upper' in test_prediction, "CI generation failed"
print("✓ Confidence intervals work")

print("\n✓ All smoke tests passed - staged model ready for deployment")
```

**Checklist**:
- [ ] Staged model loads without errors
- [ ] Can generate predictions
- [ ] Feature names match expected
- [ ] Confidence intervals generate correctly

---

### Phase 6: Production Deployment (15 minutes)

#### 6.1 Deploy to Production

**Promote staged model to production**:
```bash
# CRITICAL: This is the actual deployment step
# Double-check all validation passed before running

PRODUCTION_DIR="models/production/rila_6y20b"
DEPLOYMENT_LOG="logs/deployments/deployment_$(date +%Y%m%d_%H%M%S).log"

echo "=== PRODUCTION DEPLOYMENT ===" | tee -a ${DEPLOYMENT_LOG}
echo "Date: $(date)" | tee -a ${DEPLOYMENT_LOG}
echo "Deployer: $(whoami)" | tee -a ${DEPLOYMENT_LOG}
echo "New Version: $(cat models/staging/rila_6y20b/version.txt)" | tee -a ${DEPLOYMENT_LOG}
echo "Previous Version: $(cat ${PRODUCTION_DIR}/version.txt)" | tee -a ${DEPLOYMENT_LOG}

# Atomic swap: rename staging to production
mv ${PRODUCTION_DIR} ${PRODUCTION_DIR}_old_$(date +%Y%m%d_%H%M%S)
mv models/staging/rila_6y20b ${PRODUCTION_DIR}

# Verify deployment
if [ -f "${PRODUCTION_DIR}/model.pkl" ]; then
    echo "✓ Deployment successful" | tee -a ${DEPLOYMENT_LOG}
    echo "Production model updated: $(cat ${PRODUCTION_DIR}/version.txt)" | tee -a ${DEPLOYMENT_LOG}
else
    echo "✗ Deployment FAILED - rolling back" | tee -a ${DEPLOYMENT_LOG}
    # Automatic rollback
    mv ${PRODUCTION_DIR}_old_$(date +%Y%m%d_%H%M%S) ${PRODUCTION_DIR}
    exit 1
fi
```

**Checklist**:
- [ ] Staging directory renamed to production
- [ ] Old production directory archived with timestamp
- [ ] Deployment logged with timestamp and version
- [ ] No errors during file operations

#### 6.2 Update Production Symlinks

**Update latest symlinks for notebooks**:
```bash
# Update symlinks used by production notebooks
cd outputs/production/rila_6y20b/

# Update model symlink
ln -sf ../../../models/production/rila_6y20b/model.pkl latest_model.pkl

# Verify symlink
if [ -L "latest_model.pkl" ]; then
    echo "✓ Symlink updated"
else
    echo "✗ Symlink update failed"
    exit 1
fi
```

**Checklist**:
- [ ] latest_model.pkl symlink updated
- [ ] Symlink points to correct production model
- [ ] Notebooks can access new model via symlink

---

### Phase 7: Post-Deployment Validation (30 minutes)

#### 7.1 Immediate Post-Deployment Tests

**Verify production model works end-to-end**:
```python
# Run production inference notebook to verify deployment
# Location: notebooks/production/rila_6y20b/01_price_elasticity_inference.ipynb

from src.models.inference import PriceElasticityInference

# Load production model (via symlink)
production_model = PriceElasticityInference.load("rila_6y20b")

# Verify version
deployed_version = production_model.metadata['version']
print(f"Deployed version: {deployed_version}")

# Run full inference pipeline
latest_data = load_latest_data()
predictions = production_model.predict(latest_data)

# Generate standard outputs
elasticity_curve = production_model.generate_elasticity_curve(
    rate_range=range(50, 451, 25)
)

# Export for business review
predictions.to_csv('outputs/production/rila_6y20b/latest_predictions.csv')
elasticity_curve.to_csv('outputs/production/rila_6y20b/latest_elasticity_curve.csv')

print("✓ Production inference complete")
print(f"  Current week prediction: {predictions['mean'].iloc[-1]:.0f} contracts")
print(f"  95% CI: [{predictions['lower'].iloc[-1]:.0f}, {predictions['upper'].iloc[-1]:.0f}]")
```

**Checklist**:
- [ ] Production model loads correctly
- [ ] Latest predictions generated successfully
- [ ] Elasticity curve exported
- [ ] Output files created in production directory

#### 7.2 Sanity Check Predictions

**Compare new vs. old model predictions**:
```python
# Load backup model for comparison
backup_model = PriceElasticityInference.load_from_path(
    f"{BACKUP_DIR}/model.pkl"
)

# Compare predictions on same data
new_predictions = production_model.predict(current_week_features)
old_predictions = backup_model.predict(current_week_features)

prediction_diff = (new_predictions['mean'] - old_predictions['mean']) / old_predictions['mean']

print(f"Prediction difference: {prediction_diff:.1%}")

# Flag if predictions differ dramatically
if abs(prediction_diff) > 0.20:
    print(f"⚠️  WARNING: Predictions differ by {prediction_diff:.1%}")
    print("   Review before business consumption")
else:
    print(f"✓ Predictions reasonable ({prediction_diff:.1%} difference)")
```

**Expected Behavior**:
- Predictions should be similar (< 20% difference) unless major data changes
- Large differences require explanation before business consumption

**Checklist**:
- [ ] New predictions generated
- [ ] Predictions compared to previous model
- [ ] Large differences explained and documented
- [ ] Predictions within reasonable bounds

#### 7.3 Generate Deployment Report

**Create deployment summary for stakeholders**:
```python
# Generate automated deployment report
from src.reporting.deployment import generate_deployment_report

report = generate_deployment_report(
    old_model=backup_model,
    new_model=production_model,
    validation_results=validation_metrics,
    deployment_timestamp=DEPLOYMENT_TIMESTAMP
)

# Export report
report.to_html('outputs/deployment_reports/deployment_report_{DEPLOYMENT_TIMESTAMP}.html')

print("✓ Deployment report generated")
```

**Report Contents**:
- Deployment timestamp and version
- Validation metrics comparison (old vs new)
- Prediction comparison on recent data
- Known issues and limitations
- Next monitoring milestones

**Checklist**:
- [ ] Deployment report generated
- [ ] Report shared with Rate Setting Team
- [ ] Report archived in deployment_reports/
- [ ] Known issues documented

---

### Phase 8: Monitoring Setup (15 minutes)

#### 8.1 Enable Performance Monitoring

**Set up CloudWatch metrics (manual process)**:
```bash
# Document current model performance for monitoring baseline
python scripts/log_model_metrics.py \
    --model-path models/production/rila_6y20b/model.pkl \
    --metrics-file outputs/monitoring/baseline_metrics.json

# Expected output:
# {
#   "deployment_date": "2026-01-29",
#   "model_version": "v3.1_20260129",
#   "r2_score": 0.7837,
#   "mape": 0.1274,
#   "ci_coverage": 0.944
# }
```

**Monitoring Thresholds**:
```
MAPE > 15%:  WARNING (investigate within 24 hours)
MAPE > 20%:  CRITICAL (consider rollback)
R² < 60%:    WARNING (investigate within 24 hours)
R² < 50%:    CRITICAL (consider rollback)
Coverage < 85% or > 98%:  WARNING (miscalibrated intervals)
```

**Checklist**:
- [ ] Baseline metrics documented
- [ ] Monitoring thresholds configured
- [ ] Alert recipients configured (Rate Setting Team, Model Owner)
- [ ] First monitoring checkpoint scheduled (1 week post-deployment)

#### 8.2 Schedule First Review

**Biweekly Business Review Cycle**:
- **Week 1 Post-Deployment**: Intensive monitoring
  - Daily prediction review (manual)
  - Check for anomalies in elasticity curves
  - Verify confidence intervals calibrated

- **Week 2 Post-Deployment**: Business review meeting
  - Present deployment report to Rate Setting Team
  - Review prediction accuracy on first real week
  - Discuss any issues or observations

- **Ongoing**: Biweekly refresh cycle
  - Data refresh Tuesday AM
  - Model retrain Tuesday PM
  - Validation Wednesday
  - Business review Thursday

**Checklist**:
- [ ] Week 1 daily monitoring scheduled
- [ ] Week 2 business review scheduled
- [ ] Calendar invites sent to stakeholders
- [ ] Monitoring dashboard link shared

---

## Rollback Procedures

### When to Rollback

**Immediate Rollback Triggers** (execute within 1 hour):
- Predictions fail to generate (technical failure)
- Predictions consistently outside [0, 5000] range (logic failure)
- Model loads with errors in production notebooks
- Critical data pipeline failure detected

**Urgent Rollback Triggers** (execute within 24 hours):
- MAPE > 25% on first real week (accuracy failure)
- Confidence intervals completely miscalibrated (> 98% or < 85% coverage)
- Coefficient signs flip (economic constraint violation)
- Stakeholder loss of confidence in predictions

**Planned Rollback Triggers** (execute within 1 week):
- MAPE consistently > 20% over 2 weeks (sustained degradation)
- Better model candidate becomes available
- Data quality issues discovered in training data
- Business requirements change

### Manual Rollback Process (15 minutes)

**Step 1: Identify Backup to Restore**
```bash
# List available backups
ls -lh /home/sagemaker-user/model_backups/rila_6y20b/

# Identify most recent pre-deployment backup
ROLLBACK_SOURCE="/home/sagemaker-user/model_backups/rila_6y20b/20260129_143000"

# Verify backup integrity
python scripts/verify_backup.py --backup-dir ${ROLLBACK_SOURCE}
```

**Step 2: Execute Rollback**
```bash
# Create rollback log
ROLLBACK_LOG="logs/rollbacks/rollback_$(date +%Y%m%d_%H%M%S).log"

echo "=== PRODUCTION ROLLBACK ===" | tee -a ${ROLLBACK_LOG}
echo "Date: $(date)" | tee -a ${ROLLBACK_LOG}
echo "Operator: $(whoami)" | tee -a ${ROLLBACK_LOG}
echo "Reason: [ENTER REASON HERE]" | tee -a ${ROLLBACK_LOG}
echo "Restoring from: ${ROLLBACK_SOURCE}" | tee -a ${ROLLBACK_LOG}

# Archive failed deployment
PRODUCTION_DIR="models/production/rila_6y20b"
mv ${PRODUCTION_DIR} models/failed_deployments/rila_6y20b_$(date +%Y%m%d_%H%M%S)

# Restore backup to production
cp -r ${ROLLBACK_SOURCE} ${PRODUCTION_DIR}

# Verify rollback
if [ -f "${PRODUCTION_DIR}/model.pkl" ]; then
    RESTORED_VERSION=$(cat ${PRODUCTION_DIR}/version.txt)
    echo "✓ Rollback successful to version: ${RESTORED_VERSION}" | tee -a ${ROLLBACK_LOG}
else
    echo "✗ ROLLBACK FAILED - CRITICAL ERROR" | tee -a ${ROLLBACK_LOG}
    exit 1
fi
```

**Step 3: Verify Rolled-Back Model**
```python
# Test rolled-back model works
from src.models.inference import PriceElasticityInference

rolled_back_model = PriceElasticityInference.load("rila_6y20b")

# Run smoke tests
test_prediction = rolled_back_model.predict(current_week_features)
print(f"✓ Rolled-back model working: {test_prediction['mean']:.0f}")

# Verify version
print(f"Restored version: {rolled_back_model.metadata['version']}")
```

**Step 4: Notify Stakeholders**
- [ ] Send rollback notification to Rate Setting Team
- [ ] Document rollback reason in incident log
- [ ] Update RAI000038 documentation if needed
- [ ] Schedule post-mortem meeting (within 3 days)

**Rollback Communication Template**:
```
Subject: RILA 6Y20B Model Rollback - [Date]

The RILA 6Y20B price elasticity model has been rolled back to the previous
production version due to [REASON].

Rollback Details:
- Deployment Date: [DATE]
- Rollback Date: [DATE]
- Restored Version: [VERSION]
- Reason: [DETAILED REASON]

Current Status:
- Production model: [VERSION] (previous stable version)
- Predictions: Available and validated
- Business Impact: [DESCRIBE IMPACT]

Next Steps:
- Root cause analysis: [DATE]
- Remediation plan: [DATE]
- Re-deployment timeline: [DATE]

Contact [Model Owner] with any questions.
```

---

## Post-Deployment Monitoring

### Week 1: Intensive Monitoring

**Daily Checks** (15 minutes each day):
- [ ] Model generates predictions without errors
- [ ] Predictions within expected range [0, 5000]
- [ ] Confidence intervals reasonable width
- [ ] No anomalous feature values in input data

**Monitoring Script**:
```python
# Run daily monitoring checks
from src.monitoring.daily_checks import run_daily_monitoring

results = run_daily_monitoring(
    model_path="models/production/rila_6y20b/model.pkl",
    date=today
)

# Flag any issues
if results['any_failures']:
    print("⚠️  Daily monitoring detected issues:")
    for issue in results['issues']:
        print(f"  - {issue}")
    # Send alert email
    send_monitoring_alert(results)
else:
    print("✓ Daily monitoring passed all checks")
```

### Week 2: Business Review

**Review Meeting Agenda** (1 hour):
1. Deployment summary (5 min)
2. First-week performance results (15 min)
3. Prediction accuracy vs. actuals (20 min)
4. Known issues and limitations (10 min)
5. Q&A and feedback (10 min)

**Required Materials**:
- Deployment report
- Actual sales vs. predicted (first week)
- Elasticity curve validation
- Confidence interval calibration chart

### Ongoing: Biweekly Monitoring

**Every 2 Weeks** (aligned with rate-setting cycle):
- [ ] Refresh data from TDE/WINK
- [ ] Retrain model with latest data
- [ ] Run full validation suite
- [ ] Generate updated predictions
- [ ] Business review meeting

**Performance Drift Detection**:
```python
# 13-week rolling MAPE monitoring
from src.monitoring.drift_detection import detect_drift

drift_report = detect_drift(
    model=production_model,
    data=latest_26_weeks,
    window=13
)

if drift_report['drift_detected']:
    print(f"⚠️  Model drift detected: {drift_report['drift_magnitude']:.1%}")
    print("Consider model retraining or rollback")
else:
    print("✓ Model performance stable")
```

---

## Emergency Contacts

**Model Owner**: Brandon Behring (brandon.behring@prudential.com)
**Rate Setting Team**: annuity-rate-setting@prudential.com
**Model Risk**: annuities-model-risk@prudential.com
**Technical Support**: data-science-platform@prudential.com

**Emergency Hotline**: [INTERNAL NUMBER]
**After-Hours Support**: [ON-CALL ROTATION]

---

## Related Documentation

### Validation Framework
- [../practices/LEAKAGE_CHECKLIST.md](../practices/LEAKAGE_CHECKLIST.md) - **MANDATORY** pre-deployment
- [../methodology/validation_guidelines.md](../methodology/validation_guidelines.md) - Complete validation
- [EMERGENCY_PROCEDURES.md](EMERGENCY_PROCEDURES.md) - Crisis response protocols

### Business Context
- [../business/methodology_report.md](../business/methodology_report.md) - Technical methodology
- [../business/rai_governance.md](../business/rai_governance.md) - RAI000038 compliance
- [../business/executive_summary.md](../business/executive_summary.md) - Business overview

### Development
- [../development/TESTING_GUIDE.md](../development/TESTING_GUIDE.md) - Test procedures
- [../onboarding/COMMON_TASKS.md](../onboarding/COMMON_TASKS.md) - Common operations

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-29 | Initial deployment checklist (manual deployment procedures) |

---

## Notes

**Current Deployment Process**: Manual deployment with manual rollback capability
- No CI/CD automation yet
- No GitHub Actions workflows
- Manual validation and promotion
- Manual monitoring and alerting

**Future Enhancements** (when CI/CD implemented):
- Automated testing gates
- Automated model promotion
- Automated rollback triggers
- Automated performance monitoring
- Integration testing in staging environment

**For now**: This checklist documents best practices for manual deployment to ensure safe, validated releases to production.

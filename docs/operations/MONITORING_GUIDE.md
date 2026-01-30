# Production Monitoring Guide

**Purpose**: Real-time monitoring and alerting for RILA price elasticity models in production
**Last Updated**: 2026-01-29
**Focus**: AWS CloudWatch-based monitoring with manual processes
**Status**: Production operational procedures

---

## Overview

This guide describes monitoring procedures for production RILA price elasticity models. Currently using manual monitoring processes with AWS CloudWatch for infrastructure metrics. Automated alerting and dashboards to be implemented in future phases.

**Monitoring Philosophy**:
- **Proactive**: Catch issues before they impact business
- **Actionable**: Every alert should have clear response procedure
- **Tiered**: Warning vs. Critical severity levels
- **Manual**: Currently manual monitoring, automation roadmap included

---

## Table of Contents

1. [Monitoring Layers](#monitoring-layers)
2. [AWS CloudWatch Monitoring](#aws-cloudwatch-monitoring)
3. [Model Performance Monitoring](#model-performance-monitoring)
4. [Data Quality Monitoring](#data-quality-monitoring)
5. [Drift Detection](#drift-detection)
6. [Alert Response Procedures](#alert-response-procedures)
7. [Monitoring Schedule](#monitoring-schedule)
8. [Dashboards and Reports](#dashboards-and-reports)

---

## Monitoring Layers

### Layer 1: Infrastructure (AWS CloudWatch)
**What**: EC2 instance health, S3 access, network connectivity
**Frequency**: Real-time (CloudWatch metrics)
**Ownership**: AWS infrastructure, SageMaker notebooks
**Alerting**: AWS SNS (to be configured)

### Layer 2: Data Pipeline
**What**: Data freshness, completeness, quality
**Frequency**: Daily (manual checks), Biweekly (automated pipeline)
**Ownership**: Data engineering, model owner
**Alerting**: Manual review (automation planned)

### Layer 3: Model Performance
**What**: Prediction accuracy, coefficient stability, confidence interval calibration
**Frequency**: Weekly actuals comparison, Biweekly full validation
**Ownership**: Model owner, Rate Setting Team
**Alerting**: Manual review with escalation procedures

### Layer 4: Business Impact
**What**: Prediction vs. actual sales, elasticity curve reasonableness
**Frequency**: Weekly business review
**Ownership**: Rate Setting Team, Model owner
**Alerting**: Business stakeholder notification

---

## AWS CloudWatch Monitoring

### Current Infrastructure Setup

**SageMaker Notebook Instance**: ml.t3.2xlarge
**S3 Buckets**:
- Source data: `s3://pruvpcaws031-east/rila/`
- DVC remote: `s3://pruvpcaws031-east/dvc-storage/`
- Outputs: `s3://prudential-annuities-analytics/rila/outputs/`

**IAM Role**: Cross-account access with read permissions

### CloudWatch Metrics to Monitor

#### 1. EC2 Instance Metrics

**Access CloudWatch Console**:
```
AWS Console → CloudWatch → Metrics → EC2 → Per-Instance Metrics
→ Filter by instance ID: [your-sagemaker-instance-id]
```

**Key Metrics**:

**CPU Utilization**:
- **Threshold**: > 80% sustained for > 30 minutes = WARNING
- **Threshold**: > 95% sustained for > 10 minutes = CRITICAL
- **Expected**: < 50% during normal operations, spikes to 60-70% during bootstrap training

**Memory Utilization** (requires CloudWatch agent):
- **Threshold**: > 85% = WARNING
- **Threshold**: > 95% = CRITICAL
- **Expected**: 40-60% during normal operations, 70-80% during bootstrap ensemble training

**Disk I/O**:
- **Metric**: DiskReadBytes, DiskWriteBytes
- **Expected**: Elevated during data loading, low otherwise
- **Alert**: Sustained high I/O (> 100 MB/s for > 1 hour) = investigate

**Network**:
- **Metric**: NetworkIn, NetworkOut
- **Expected**: Elevated during S3 data transfers
- **Alert**: Network errors > 0 = investigate connectivity

**Manual Check Script**:
```bash
# Check current instance metrics (run from SageMaker terminal)
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --dimensions Name=InstanceId,Value=$(ec2-metadata --instance-id | cut -d " " -f 2) \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average \
  --region us-east-1

# Expected output: JSON with Average CPUUtilization over last hour
```

#### 2. S3 Bucket Metrics

**Access CloudWatch Console**:
```
AWS Console → CloudWatch → Metrics → S3 → Storage Metrics
→ Filter by bucket: pruvpcaws031-east
```

**Key Metrics**:

**BucketSizeBytes**:
- **Monitor**: Data growth over time
- **Expected**: Steady growth (weekly data additions)
- **Alert**: Unexpected size jumps (> 20% in 1 day) = data quality issue

**NumberOfObjects**:
- **Monitor**: File count in source buckets
- **Expected**: Incremental growth (new weekly files)
- **Alert**: Object count decrease = data deletion (investigate)

**AllRequests** (request metrics, must be enabled):
- **Monitor**: S3 access patterns
- **Expected**: Spikes during biweekly data refresh
- **Alert**: Failed requests > 5% = access issues

**Manual Check Script**:
```bash
# Check S3 bucket access
aws s3 ls s3://pruvpcaws031-east/rila/sales/latest/ --profile cross-account

# Verify file count and size
aws s3 ls s3://pruvpcaws031-east/rila/sales/ --recursive --summarize | tail -2

# Expected output:
# Total Objects: XXXX
# Total Size: YYYY MB
```

#### 3. CloudWatch Logs

**SageMaker Notebook Logs**:
```
AWS Console → CloudWatch → Logs → Log Groups
→ /aws/sagemaker/NotebookInstances/[your-instance-name]
```

**What to Monitor**:
- Python exceptions and errors
- Data loading warnings
- Model training failures
- Out-of-memory errors

**Manual Check**:
```bash
# View recent logs (last 1 hour)
aws logs tail /aws/sagemaker/NotebookInstances/[instance-name] --since 1h --follow

# Filter for errors
aws logs filter-log-events \
  --log-group-name /aws/sagemaker/NotebookInstances/[instance-name] \
  --start-time $(date -d '1 day ago' +%s)000 \
  --filter-pattern "ERROR"
```

### CloudWatch Alarms (To Be Configured)

**Recommended Alarms** (manual setup via AWS Console):

**High CPU Alarm**:
```
Metric: CPUUtilization
Threshold: > 80%
Duration: 2 consecutive periods (10 minutes)
Action: SNS notification to model owner
```

**High Memory Alarm** (requires CloudWatch agent):
```
Metric: MemoryUtilization
Threshold: > 90%
Duration: 2 consecutive periods (10 minutes)
Action: SNS notification to model owner
```

**S3 Access Failure Alarm**:
```
Metric: 4xx/5xx errors
Threshold: > 10 errors in 5 minutes
Action: SNS notification to model owner + infrastructure team
```

**Setup Instructions**:
1. AWS Console → CloudWatch → Alarms → Create Alarm
2. Select metric (e.g., EC2 CPUUtilization)
3. Define threshold and duration
4. Configure SNS notification (email/Slack)
5. Name alarm: "RILA-6Y20B-HighCPU-Warning"

---

## Model Performance Monitoring

### Weekly Performance Checks

**When**: Every Monday AM (review previous week's predictions vs. actuals)

**Process**:
```python
# Location: notebooks/production/rila_6y20b/monitoring/weekly_performance_check.ipynb

import pandas as pd
from src.monitoring.performance import calculate_weekly_metrics

# Load predictions from last week
predictions = pd.read_csv('outputs/production/rila_6y20b/predictions/2026-01-22.csv')

# Load actuals from TDE (available Monday AM with 2-day lag)
actuals = load_tde_actuals(week_ending='2026-01-26')

# Calculate metrics
metrics = calculate_weekly_metrics(predictions, actuals)

print(f"Weekly MAPE: {metrics['mape']:.2%}")
print(f"Weekly Bias: {metrics['bias']:.2%}")
print(f"Coverage: {metrics['coverage']:.2%}")
```

**Thresholds**:
```
Weekly MAPE < 20%:     OK
Weekly MAPE 20-25%:    WARNING (investigate)
Weekly MAPE > 25%:     CRITICAL (consider rollback)

Bias between -10% and +10%:  OK (unbiased)
Bias < -10% or > +10%:       WARNING (systematic error)

Coverage 90-97%:       OK (well-calibrated)
Coverage < 90% or > 97%:  WARNING (miscalibrated intervals)
```

**Documentation**:
```python
# Log weekly metrics to monitoring log
metrics_log = pd.DataFrame({
    'date': [today],
    'mape': [metrics['mape']],
    'bias': [metrics['bias']],
    'coverage': [metrics['coverage']],
    'actuals': [actuals['sales'].sum()],
    'predicted': [predictions['mean'].sum()]
})

metrics_log.to_csv(
    'outputs/monitoring/weekly_metrics_log.csv',
    mode='a',
    header=False,
    index=False
)
```

### Rolling 13-Week MAPE Monitoring

**When**: Biweekly (as part of model refresh cycle)

**Purpose**: Detect model drift over time

**Process**:
```python
# Calculate rolling 13-week MAPE
from src.monitoring.drift_detection import calculate_rolling_mape

rolling_metrics = calculate_rolling_mape(
    predictions_log='outputs/monitoring/weekly_metrics_log.csv',
    window=13
)

# Plot trend
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(rolling_metrics['date'], rolling_metrics['rolling_mape'])
plt.axhline(y=0.15, color='orange', linestyle='--', label='Warning (15%)')
plt.axhline(y=0.20, color='red', linestyle='--', label='Critical (20%)')
plt.title('13-Week Rolling MAPE - RILA 6Y20B')
plt.xlabel('Date')
plt.ylabel('MAPE')
plt.legend()
plt.grid(True)
plt.savefig('outputs/monitoring/rolling_mape_trend.png')

# Alert if trending up
recent_mape = rolling_metrics['rolling_mape'].iloc[-4:].mean()  # Last month
historical_mape = rolling_metrics['rolling_mape'].iloc[:-4].mean()

drift = (recent_mape - historical_mape) / historical_mape

if drift > 0.15:
    print(f"[WARN]  WARNING: MAPE drift detected ({drift:.1%} increase)")
    print("   Investigate data quality or consider model retraining")
else:
    print(f"[PASS] Model performance stable (drift: {drift:.1%})")
```

### Coefficient Stability Monitoring

**When**: Biweekly (after model retraining)

**Purpose**: Ensure economic constraints remain valid

**Process**:
```python
# Track coefficient values over time
from src.models.inference import PriceElasticityInference

current_model = PriceElasticityInference.load("rila_6y20b")
coefs = current_model.get_coefficients()

# Log coefficients
coef_log = pd.DataFrame({
    'date': [today],
    'prudential_rate': [coefs['prudential_rate_current']],
    'competitor_top5': [coefs['competitor_top5_t2']],
    'sales_momentum': [coefs['sales_target_contract_t5']]
})

coef_log.to_csv(
    'outputs/monitoring/coefficient_log.csv',
    mode='a',
    header=False,
    index=False
)

# Check sign consistency
assert coefs['prudential_rate_current'] > 0, "Own rate coefficient flipped sign!"
assert coefs['competitor_top5_t2'] < 0, "Competitor coefficient flipped sign!"
assert coefs['sales_target_contract_t5'] > 0, "Sales momentum coefficient flipped sign!"

print("[PASS] All coefficient signs consistent with economic theory")
```

**Alert Triggers**:
- Coefficient sign flips = **CRITICAL** (immediate investigation)
- Coefficient magnitude changes > 50% = **WARNING** (review model)
- Coefficient becomes statistically insignificant (p > 0.05) = **WARNING**

### R² and MAPE Tracking

**When**: Biweekly (after model validation)

**Process**:
```python
# Track model performance metrics over time
performance_log = pd.DataFrame({
    'date': [today],
    'model_version': [current_model.metadata['version']],
    'r2_score': [validation_metrics['r2']],
    'mape': [validation_metrics['mape']],
    'ci_coverage': [validation_metrics['coverage']],
    'n_training_weeks': [len(train_data)],
    'n_features': [len(current_model.feature_names_)]
})

performance_log.to_csv(
    'outputs/monitoring/performance_log.csv',
    mode='a',
    header=False,
    index=False
)

# Check against thresholds
assert validation_metrics['r2'] > 0.50, f"R² below threshold: {validation_metrics['r2']:.2%}"
assert validation_metrics['mape'] < 0.20, f"MAPE above threshold: {validation_metrics['mape']:.2%}"

print(f"[PASS] Model performance acceptable (R²={validation_metrics['r2']:.2%}, MAPE={validation_metrics['mape']:.2%})")
```

---

## Data Quality Monitoring

### Data Freshness Checks

**When**: Daily (automated) + Biweekly before model refresh

**TDE Sales Data Freshness**:
```python
# Check most recent sales data
import pandas as pd
from datetime import datetime, timedelta

sales_data = pd.read_parquet('data/processed/tde_sales_latest.parquet')
latest_date = sales_data['application_signed_date'].max()
days_old = (datetime.now() - latest_date).days

# Alert thresholds
if days_old > 14:
    print(f"[FAIL] CRITICAL: Sales data {days_old} days old (expected ≤ 14 days)")
    print("  Action: Check TDE data pipeline")
elif days_old > 7:
    print(f"[WARN]  WARNING: Sales data {days_old} days old (expected ≤ 7 days)")
    print("  Action: Verify data refresh schedule")
else:
    print(f"[PASS] Sales data fresh ({days_old} days old)")
```

**WINK Competitive Rates Freshness**:
```python
# Check most recent competitive rates
wink_data = pd.read_parquet('data/processed/wink_rates_latest.parquet')
latest_date = wink_data['rate_effective_date'].max()
days_old = (datetime.now() - latest_date).days

if days_old > 14:
    print(f"[FAIL] CRITICAL: WINK data {days_old} days old (expected ≤ 14 days)")
    print("  Action: Check WINK data pipeline")
elif days_old > 7:
    print(f"[WARN]  WARNING: WINK data {days_old} days old (expected ≤ 7 days)")
else:
    print(f"[PASS] WINK data fresh ({days_old} days old)")
```

**Economic Indicators Freshness** (FRED):
```python
# Check Treasury rates and VIX
import pandas_datareader as pdr

try:
    # Fetch latest DGS5 (5-year Treasury)
    treasury = pdr.get_data_fred('DGS5', start=datetime.now() - timedelta(days=7))
    latest_treasury = treasury.index[-1]
    days_old = (datetime.now() - latest_treasury).days

    if days_old > 3:
        print(f"[WARN]  WARNING: Treasury data {days_old} days old")
    else:
        print(f"[PASS] Economic data fresh ({days_old} days old)")

except Exception as e:
    print(f"[FAIL] CRITICAL: Cannot fetch economic data: {e}")
```

### Data Completeness Checks

**When**: Biweekly before model training

**Missing Data Detection**:
```python
# Check for missing critical features
from src.data.validation import check_data_completeness

completeness = check_data_completeness(
    data=model_data,
    window_weeks=26  # Check last 6 months
)

critical_features = [
    'prudential_rate_current',
    'competitor_top5_t2',
    'sales_target_contract_t5',
    'market_vix',
    'treasury_5yr'
]

for feat in critical_features:
    missing_pct = completeness[feat]['missing_pct']

    if missing_pct > 0.05:
        print(f"[FAIL] CRITICAL: {feat} has {missing_pct:.1%} missing data")
    elif missing_pct > 0.01:
        print(f"[WARN]  WARNING: {feat} has {missing_pct:.1%} missing data")
    else:
        print(f"[PASS] {feat} complete ({missing_pct:.1%} missing)")
```

**Carrier Coverage Check** (RILA-specific):
```python
# Verify all 8 major carriers have recent data
carriers = [
    'Athene', 'Brighthouse', 'Equitable', 'Ameriprise',
    'Jackson', 'Lincoln', 'Symetra', 'Transamerica'
]

recent_data = wink_data[wink_data['rate_effective_date'] >= datetime.now() - timedelta(days=30)]

for carrier in carriers:
    carrier_data = recent_data[recent_data['carrier'] == carrier]

    if len(carrier_data) == 0:
        print(f"[FAIL] CRITICAL: No recent data for {carrier} (last 30 days)")
    elif len(carrier_data) < 10:
        print(f"[WARN]  WARNING: Sparse data for {carrier} ({len(carrier_data)} records)")
    else:
        print(f"[PASS] {carrier} coverage adequate ({len(carrier_data)} records)")
```

### Feature Distribution Monitoring

**When**: Biweekly before model training

**Purpose**: Detect anomalous feature values or distribution shifts

**Process**:
```python
# Check for feature distribution shifts
from src.monitoring.data_quality import check_feature_distributions

# Compare recent 13 weeks vs. historical
recent_data = model_data[model_data['date'] >= datetime.now() - timedelta(weeks=13)]
historical_data = model_data[model_data['date'] < datetime.now() - timedelta(weeks=13)]

distribution_shifts = check_feature_distributions(
    recent_data=recent_data,
    historical_data=historical_data,
    features=current_model.feature_names_
)

# Flag features with significant shifts (> 2 std deviations)
anomalies = {
    feat: shift
    for feat, shift in distribution_shifts.items()
    if abs(shift) > 2.0
}

if anomalies:
    print("[WARN]  WARNING: Anomalous feature distributions detected:")
    for feat, shift in anomalies.items():
        print(f"  - {feat}: {shift:.2f}σ shift")
    print("  Action: Review recent data for quality issues")
else:
    print("[PASS] All feature distributions normal")
```

**Rate Range Validation**:
```python
# Check rates are within expected bounds
prudential_rates = recent_data['prudential_rate_current']
competitor_rates = recent_data['competitor_top5_t2']

# Expected range: 50 to 450 basis points (0.5% to 4.5%)
assert prudential_rates.min() >= 0.005, f"Prudential rate too low: {prudential_rates.min():.2%}"
assert prudential_rates.max() <= 0.045, f"Prudential rate too high: {prudential_rates.max():.2%}"
assert competitor_rates.min() >= 0.005, f"Competitor rate too low: {competitor_rates.min():.2%}"
assert competitor_rates.max() <= 0.045, f"Competitor rate too high: {competitor_rates.max():.2%}"

print("[PASS] Rate ranges within expected bounds")
```

---

## Drift Detection

### Model Drift

**Definition**: Model performance degrades over time due to changing market dynamics

**Detection**:
- 13-week rolling MAPE increases > 15% relative to historical baseline
- R² decreases > 10% relative to validation baseline
- Systematic prediction bias emerges (> 10% over/under prediction)

**Response**:
1. Investigate recent data quality
2. Review for market regime changes (e.g., Fed policy shifts)
3. Consider model retraining with expanded feature set
4. If severe (MAPE > 25%), consider rollback to previous model

### Data Drift

**Definition**: Input feature distributions shift significantly from training data

**Detection**:
- Feature mean/std shifts > 2σ from historical distribution
- New rate ranges outside training bounds (< 50bp or > 450bp)
- Competitor set changes (carriers enter/exit market)

**Response**:
1. Validate data pipeline integrity
2. Check for data source changes (TDE/WINK schema updates)
3. Retrain model if shifts are persistent (> 4 weeks)
4. Update feature engineering if new market structure

### Concept Drift

**Definition**: Underlying relationship between features and sales changes (e.g., customer behavior shifts)

**Detection**:
- Coefficient magnitudes change significantly (> 50% change)
- New features become important (AIC-based selection changes)
- Economic constraint violations (coefficient signs flip)

**Response**:
1. Conduct root cause analysis (market research, stakeholder interviews)
2. Review causal framework assumptions
3. Major refactor may be needed if fundamental market dynamics change
4. Document regime change and rationale for model updates

---

## Alert Response Procedures

### Severity Levels

**INFO**: Informational, no action required
- Weekly monitoring report generated
- Data refresh completed successfully
- Model retraining completed successfully

**WARNING**: Investigate within 24 hours
- MAPE 20-25% on single week
- Data freshness 7-14 days old
- Feature distribution shifts 1.5-2.0σ
- Coefficient magnitude changes 30-50%

**CRITICAL**: Investigate immediately (within 1 hour)
- MAPE > 25% on single week or > 20% sustained for 2+ weeks
- Data freshness > 14 days old
- Model predictions fail to generate
- Coefficient signs flip
- AWS infrastructure failures (CPU > 95%, memory > 95%)

### Escalation Procedures

**Level 1: Model Owner** (Brandon Behring)
- All WARNING and CRITICAL alerts
- Daily monitoring of CRITICAL issues until resolved
- Responsible for root cause analysis

**Level 2: Rate Setting Team**
- CRITICAL alerts affecting business decisions
- Weekly MAPE > 25% (affects rate-setting guidance)
- Model rollback decisions

**Level 3: Model Risk Team**
- Economic constraint violations (coefficient signs flip)
- Sustained performance degradation (> 4 weeks)
- Model rollback or major refactor

**Level 4: Infrastructure Team**
- AWS infrastructure failures
- S3 access issues
- Network connectivity problems

### Response Templates

**WARNING Alert Template**:
```
Subject: [WARNING] RILA 6Y20B Monitoring Alert

Issue: [Brief description]
Severity: WARNING
Detected: [Timestamp]
Metric: [Specific metric and value]
Threshold: [What threshold was exceeded]

Details:
[Detailed description of the issue]

Investigation Timeline:
- Root cause analysis: Within 24 hours
- Resolution target: Within 3 business days

Action Items:
1. [Specific action]
2. [Specific action]

Contact [Model Owner] with questions.
```

**CRITICAL Alert Template**:
```
Subject: [CRITICAL] RILA 6Y20B Production Issue

Issue: [Brief description]
Severity: CRITICAL
Detected: [Timestamp]
Impact: [Business impact description]

Immediate Actions Taken:
- [Action 1]
- [Action 2]

Current Status:
[Description of current situation]

Next Steps:
- [Immediate next step with ETA]
- [Follow-up actions]

Estimated Resolution: [Timeline]

Escalation: [Who has been notified]

Updates will be provided every [frequency] until resolved.

Contact [Model Owner] immediately with questions.
```

---

## Monitoring Schedule

### Daily (Manual)
**Time**: 9:00 AM ET
**Duration**: 15 minutes
**Owner**: Model Owner

**Tasks**:
- [ ] Check AWS CloudWatch for infrastructure issues (CPU, memory, disk)
- [ ] Verify S3 bucket accessibility
- [ ] Review CloudWatch logs for errors (past 24 hours)
- [ ] Check data freshness (TDE, WINK)

**Output**: Daily monitoring log entry (if issues found)

### Weekly (Manual)
**Time**: Monday 10:00 AM ET
**Duration**: 30 minutes
**Owner**: Model Owner

**Tasks**:
- [ ] Compare previous week predictions vs. actuals
- [ ] Calculate weekly MAPE and bias
- [ ] Check confidence interval coverage
- [ ] Review feature distributions for anomalies
- [ ] Update weekly monitoring dashboard

**Output**: Weekly monitoring report (shared with Rate Setting Team if issues)

### Biweekly (Manual)
**Time**: Tuesday 2:00 PM ET (aligned with model refresh)
**Duration**: 1 hour
**Owner**: Model Owner

**Tasks**:
- [ ] Full model performance validation (R², MAPE, coverage)
- [ ] Rolling 13-week MAPE analysis
- [ ] Coefficient stability check
- [ ] Data quality validation (completeness, freshness)
- [ ] Feature distribution monitoring
- [ ] Generate biweekly monitoring report

**Output**: Biweekly monitoring report + model validation sign-off

### Quarterly (Manual)
**Time**: First Tuesday of quarter
**Duration**: 2 hours
**Owner**: Model Owner + Model Risk Team

**Tasks**:
- [ ] Comprehensive performance review (12-week lookback)
- [ ] Drift detection analysis (model, data, concept drift)
- [ ] Economic constraint validation across all periods
- [ ] Business impact assessment (actual vs. predicted sales)
- [ ] Documentation review and updates (RAI000038)
- [ ] Monitoring procedure audit

**Output**: Quarterly model health report + recommendations

---

## Dashboards and Reports

### Weekly Monitoring Dashboard (Manual)

**Location**: `outputs/monitoring/dashboards/weekly_dashboard.html`

**Generated by**:
```python
# notebooks/production/rila_6y20b/monitoring/generate_weekly_dashboard.ipynb

from src.reporting.dashboards import generate_weekly_dashboard

dashboard = generate_weekly_dashboard(
    predictions_log='outputs/monitoring/weekly_metrics_log.csv',
    actuals_log='data/processed/tde_actuals.csv',
    lookback_weeks=13
)

dashboard.save('outputs/monitoring/dashboards/weekly_dashboard.html')
```

**Contents**:
1. **Performance Summary** (KPIs)
   - Latest week MAPE, bias, coverage
   - 4-week rolling averages
   - Year-to-date performance

2. **Predictions vs. Actuals Chart**
   - Line chart: predicted vs. actual sales (last 13 weeks)
   - Confidence interval bands
   - Highlight weeks with large errors

3. **Rolling MAPE Trend**
   - 13-week rolling MAPE
   - Warning/critical threshold lines
   - Color-coded by severity

4. **Feature Distributions**
   - Histograms for key features (current vs. historical)
   - Highlight distribution shifts

5. **Alert Summary**
   - Active warnings and critical issues
   - Resolution status

### Biweekly Model Health Report (Manual)

**Location**: `outputs/monitoring/reports/model_health_YYYYMMDD.pdf`

**Generated by**:
```python
# notebooks/production/rila_6y20b/monitoring/generate_health_report.ipynb

from src.reporting.model_health import generate_health_report

report = generate_health_report(
    model=current_model,
    validation_metrics=validation_metrics,
    monitoring_logs='outputs/monitoring/',
    lookback_weeks=26
)

report.save_pdf('outputs/monitoring/reports/model_health_20260129.pdf')
```

**Contents**:
1. Executive Summary (1 page)
2. Model Performance Metrics (R², MAPE, coverage trends)
3. Coefficient Stability Analysis
4. Data Quality Assessment
5. Drift Detection Results
6. Open Issues and Action Items
7. Recommendations

**Distribution**:
- Model Owner (always)
- Rate Setting Team (if issues flagged)
- Model Risk Team (quarterly)

### Monthly Business Review Deck (Manual)

**Location**: `outputs/business_reports/monthly_review_YYYYMM.pptx`

**Owner**: Rate Setting Team + Model Owner

**Contents**:
1. Model Performance Summary (metrics vs. targets)
2. Business Impact Analysis (rate decisions informed by model)
3. Competitive Intelligence Insights
4. Forecast Accuracy Assessment
5. Known Issues and Mitigation Plans
6. Next Month Outlook

**Meeting**: Monthly business review (1st Thursday of month, 1 hour)

---

## Automation Roadmap (Future)

### Phase 1: Automated Data Quality Monitoring (Q2 2026)
- Automated data freshness checks (cron job)
- Automated completeness validation
- Email alerts for data quality issues

### Phase 2: Automated Performance Monitoring (Q3 2026)
- Automated weekly MAPE calculation (predictions vs. actuals)
- Automated rolling metrics tracking
- CloudWatch custom metrics for model performance

### Phase 3: Real-Time Alerting (Q4 2026)
- CloudWatch alarms for performance thresholds
- SNS/Slack integration for real-time alerts
- On-call rotation for critical issues

### Phase 4: Interactive Dashboards (Q1 2027)
- Web-based monitoring dashboard (Streamlit/Dash)
- Real-time metric visualization
- Self-service reporting for stakeholders

---

## Related Documentation

### Operations
- [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) - Pre-deployment validation
- [EMERGENCY_PROCEDURES.md](EMERGENCY_PROCEDURES.md) - Crisis response protocols
- [DATA_QUALITY_MONITORING.md](DATA_QUALITY_MONITORING.md) - Detailed data monitoring (to be created)

### Validation
- [../practices/LEAKAGE_CHECKLIST.md](../practices/LEAKAGE_CHECKLIST.md) - Pre-deployment checks
- [../methodology/validation_guidelines.md](../methodology/validation_guidelines.md) - Complete validation

### Business Context
- [../business/methodology_report.md](../business/methodology_report.md) - Technical methodology
- [../business/executive_summary.md](../business/executive_summary.md) - Business overview

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-29 | Initial monitoring guide (AWS CloudWatch + manual processes) |

---

## Notes

**Current State**: Manual monitoring with AWS CloudWatch for infrastructure
- No automated alerting yet (manual checks)
- No automated dashboards (manual report generation)
- CloudWatch alarms to be configured
- SNS notifications to be set up

**Key Limitation**: Manual processes require discipline and consistency
- Daily checks may be skipped
- Alert response depends on manual review
- No 24/7 monitoring coverage

**Mitigation**: Clear schedules, ownership, and escalation procedures documented to ensure consistency until automation implemented.

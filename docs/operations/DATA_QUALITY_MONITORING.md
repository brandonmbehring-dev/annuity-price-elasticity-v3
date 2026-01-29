# Data Quality Monitoring and Pipeline Health

**Purpose**: Monitor data quality, detect data drift, and ensure pipeline health for RILA models
**Last Updated**: 2026-01-29
**Status**: Production data quality procedures
**Scope**: TDE sales data, WINK competitive rates, FRED economic indicators

---

## Overview

High-quality data is critical for reliable price elasticity models. This guide provides comprehensive data quality monitoring procedures, drift detection methods, and pipeline health checks to ensure production models receive clean, timely, and complete data.

**Data Quality Pillars**:
1. **Freshness**: Data age < 7 days (target), < 14 days (acceptable)
2. **Completeness**: < 1% missing values in critical features
3. **Accuracy**: Values within expected ranges, no anomalies
4. **Consistency**: Data schema and formats stable over time
5. **Timeliness**: Data arrives on expected schedule

---

## Table of Contents

1. [Data Sources Overview](#data-sources-overview)
2. [Data Freshness Monitoring](#data-freshness-monitoring)
3. [Data Completeness Checks](#data-completeness-checks)
4. [Data Accuracy Validation](#data-accuracy-validation)
5. [Data Drift Detection](#data-drift-detection)
6. [Pipeline Health Monitoring](#pipeline-health-monitoring)
7. [Automated Quality Tests](#automated-quality-tests)
8. [Alerting and Escalation](#alerting-and-escalation)

---

## Data Sources Overview

### TDE Sales Data

**Source**: Transaction Data Engine (TDE) system
**S3 Location**: `s3://pruvpcaws031-east/rila/sales/`
**Refresh Frequency**: Daily (with 2-day processing lag)
**Critical Fields**:
- `application_signed_date` (temporal marker)
- `contract_value` (sales volume)
- `product_code` (6Y20B filter)
- `buffer_percentage` (20% filter)
- `cap_rate` (Prudential rate)

**Expected Schema**:
```python
{
    'application_signed_date': 'datetime64[ns]',
    'contract_issue_date': 'datetime64[ns]',
    'contract_value': 'float64',
    'product_code': 'str',
    'buffer_percentage': 'float64',
    'cap_rate': 'float64',
    'strategy': 'str'
}
```

**Data Volume**: ~1.4 million records (2021-present), ~500-800 new contracts/week

### WINK Competitive Rates

**Source**: WINK competitive intelligence platform
**S3 Location**: `s3://pruvpcaws031-east/rila/wink/`
**Refresh Frequency**: Weekly (Monday AM)
**Critical Fields**:
- `carrier` (8 major carriers tracked)
- `product_name` (RILA 6Y20B equivalents)
- `cap_rate` (competitor rates)
- `rate_effective_date` (temporal marker)
- `market_share_weight` (quarterly updated)

**Expected Carriers**:
```python
carriers = [
    'Athene', 'Brighthouse', 'Equitable', 'Ameriprise',
    'Jackson', 'Lincoln', 'Symetra', 'Transamerica'
]
```

**Data Volume**: ~1 million observations, ~50-100 rate updates/week

### FRED Economic Indicators

**Source**: Federal Reserve Economic Data (FRED) API
**Refresh Frequency**: Daily
**Critical Series**:
- `DGS5` - 5-Year Treasury Constant Maturity Rate
- `VIXCLS` - CBOE Volatility Index
- `CPIAUCSL` - Consumer Price Index

**Expected Ranges**:
```python
expected_ranges = {
    'DGS5': (0.01, 0.10),      # 1% to 10%
    'VIXCLS': (10.0, 80.0),    # VIX typically 10-80
    'CPIAUCSL': (200, 400)     # CPI index range
}
```

---

## Data Freshness Monitoring

### Daily Freshness Check

**When**: Daily at 9:00 AM ET (automated script)

**Check Script**:
```python
import pandas as pd
from datetime import datetime, timedelta

def check_data_freshness(source='all'):
    """Check freshness of all data sources."""

    results = {}

    # TDE Sales Data
    if source in ['all', 'tde']:
        sales_data = pd.read_parquet('data/processed/tde_sales_latest.parquet')
        latest_date = sales_data['application_signed_date'].max()
        days_old = (datetime.now() - latest_date).days

        status = 'OK' if days_old <= 7 else ('WARNING' if days_old <= 14 else 'CRITICAL')

        results['tde_sales'] = {
            'latest_date': latest_date,
            'days_old': days_old,
            'status': status,
            'records': len(sales_data)
        }

    # WINK Competitive Rates
    if source in ['all', 'wink']:
        wink_data = pd.read_parquet('data/processed/wink_rates_latest.parquet')
        latest_date = wink_data['rate_effective_date'].max()
        days_old = (datetime.now() - latest_date).days

        status = 'OK' if days_old <= 7 else ('WARNING' if days_old <= 14 else 'CRITICAL')

        results['wink_rates'] = {
            'latest_date': latest_date,
            'days_old': days_old,
            'status': status,
            'records': len(wink_data)
        }

    # FRED Economic Data
    if source in ['all', 'fred']:
        import pandas_datareader as pdr

        try:
            treasury = pdr.get_data_fred('DGS5', start=datetime.now() - timedelta(days=7))
            latest_date = treasury.index[-1]
            days_old = (datetime.now() - latest_date).days

            status = 'OK' if days_old <= 3 else ('WARNING' if days_old <= 7 else 'CRITICAL')

            results['fred_economic'] = {
                'latest_date': latest_date,
                'days_old': days_old,
                'status': status,
                'series': ['DGS5', 'VIXCLS', 'CPIAUCSL']
            }
        except Exception as e:
            results['fred_economic'] = {
                'status': 'CRITICAL',
                'error': str(e)
            }

    return results

# Run check
freshness_results = check_data_freshness()

# Print summary
for source, result in freshness_results.items():
    print(f"\n{source.upper()}:")
    print(f"  Status: {result['status']}")
    print(f"  Latest Date: {result.get('latest_date', 'N/A')}")
    print(f"  Days Old: {result.get('days_old', 'N/A')}")
```

**Alert Thresholds**:
```
Days Old ≤ 7:   OK (green)
Days Old 7-14:  WARNING (yellow) - notify model owner
Days Old > 14:  CRITICAL (red) - escalate to data engineering
```

### Freshness Monitoring Log

**Log Location**: `outputs/monitoring/data_freshness_log.csv`

```python
# Log freshness check results
freshness_log = pd.DataFrame({
    'timestamp': [datetime.now()],
    'tde_days_old': [freshness_results['tde_sales']['days_old']],
    'tde_status': [freshness_results['tde_sales']['status']],
    'wink_days_old': [freshness_results['wink_rates']['days_old']],
    'wink_status': [freshness_results['wink_rates']['status']],
    'fred_days_old': [freshness_results['fred_economic']['days_old']],
    'fred_status': [freshness_results['fred_economic']['status']]
})

freshness_log.to_csv(
    'outputs/monitoring/data_freshness_log.csv',
    mode='a',
    header=False,
    index=False
)
```

---

## Data Completeness Checks

### Missing Data Detection

**When**: Biweekly before model training

**Check Script**:
```python
def check_data_completeness(data, window_weeks=26):
    """Check for missing data in critical features."""

    # Focus on recent data (last 26 weeks)
    recent_data = data[data['date'] >= datetime.now() - timedelta(weeks=window_weeks)]

    completeness_report = {}

    for col in recent_data.columns:
        missing_count = recent_data[col].isna().sum()
        missing_pct = missing_count / len(recent_data)

        completeness_report[col] = {
            'missing_count': missing_count,
            'missing_pct': missing_pct,
            'status': 'OK' if missing_pct < 0.01 else ('WARNING' if missing_pct < 0.05 else 'CRITICAL')
        }

    return pd.DataFrame(completeness_report).T

# Run completeness check
model_data = pd.read_csv('data/processed/model_features.csv')
completeness = check_data_completeness(model_data, window_weeks=26)

# Print critical features only
critical_features = [
    'prudential_rate_current',
    'competitor_top5_t2',
    'sales_target_contract_t5',
    'market_vix',
    'treasury_5yr'
]

print("Critical Feature Completeness:")
for feat in critical_features:
    status = completeness.loc[feat, 'status']
    missing_pct = completeness.loc[feat, 'missing_pct']
    print(f"  {feat}: {status} ({missing_pct:.2%} missing)")
```

**Alert Thresholds**:
```
Missing < 1%:   OK (acceptable for time series)
Missing 1-5%:   WARNING (investigate cause)
Missing > 5%:   CRITICAL (data quality issue, do not train)
```

### Carrier Coverage Check (WINK-Specific)

**Check Script**:
```python
def check_carrier_coverage(wink_data, window_days=30):
    """Verify all 8 major carriers have recent data."""

    expected_carriers = [
        'Athene', 'Brighthouse', 'Equitable', 'Ameriprise',
        'Jackson', 'Lincoln', 'Symetra', 'Transamerica'
    ]

    recent = wink_data[wink_data['rate_effective_date'] >= datetime.now() - timedelta(days=window_days)]

    coverage = {}

    for carrier in expected_carriers:
        carrier_data = recent[recent['carrier'] == carrier]
        record_count = len(carrier_data)

        if record_count == 0:
            status = 'CRITICAL'
        elif record_count < 5:
            status = 'WARNING'
        else:
            status = 'OK'

        coverage[carrier] = {
            'record_count': record_count,
            'latest_date': carrier_data['rate_effective_date'].max() if record_count > 0 else None,
            'status': status
        }

    return pd.DataFrame(coverage).T

# Run carrier coverage check
wink_data = pd.read_parquet('data/processed/wink_rates_latest.parquet')
carrier_coverage = check_carrier_coverage(wink_data, window_days=30)

print("\nCarrier Coverage (last 30 days):")
print(carrier_coverage)

# Alert if any carrier missing
critical_carriers = carrier_coverage[carrier_coverage['status'] == 'CRITICAL']
if len(critical_carriers) > 0:
    print(f"\n⚠️  WARNING: {len(critical_carriers)} carriers missing data")
    print(critical_carriers[['record_count', 'latest_date']])
```

---

## Data Accuracy Validation

### Range Validation

**Check Script**:
```python
def validate_data_ranges(data):
    """Validate data values within expected ranges."""

    validations = {}

    # Cap rates: 50 to 450 basis points (0.5% to 4.5%)
    if 'prudential_rate_current' in data.columns:
        pru_rates = data['prudential_rate_current']
        out_of_range = ((pru_rates < 0.005) | (pru_rates > 0.045)).sum()

        validations['prudential_rate'] = {
            'out_of_range_count': out_of_range,
            'out_of_range_pct': out_of_range / len(data),
            'min': pru_rates.min(),
            'max': pru_rates.max(),
            'status': 'OK' if out_of_range == 0 else 'WARNING'
        }

    # Sales volumes: 0 to 5000 contracts/week
    if 'sales_volume' in data.columns:
        sales = data['sales_volume']
        out_of_range = ((sales < 0) | (sales > 5000)).sum()

        validations['sales_volume'] = {
            'out_of_range_count': out_of_range,
            'out_of_range_pct': out_of_range / len(data),
            'min': sales.min(),
            'max': sales.max(),
            'status': 'OK' if out_of_range == 0 else 'CRITICAL'  # Impossible values
        }

    # VIX: 10 to 80 (typical range)
    if 'market_vix' in data.columns:
        vix = data['market_vix']
        out_of_range = ((vix < 10) | (vix > 80)).sum()

        validations['market_vix'] = {
            'out_of_range_count': out_of_range,
            'out_of_range_pct': out_of_range / len(data),
            'min': vix.min(),
            'max': vix.max(),
            'status': 'WARNING' if out_of_range > 0 else 'OK'  # Unusual but possible
        }

    return pd.DataFrame(validations).T

# Run range validation
range_validation = validate_data_ranges(model_data)

print("\nData Range Validation:")
print(range_validation[['min', 'max', 'out_of_range_count', 'status']])
```

### Duplicate Detection

**Check Script**:
```python
def detect_duplicates(data, key_columns):
    """Detect duplicate records in data."""

    duplicates = data[data.duplicated(subset=key_columns, keep=False)]

    if len(duplicates) > 0:
        print(f"⚠️  WARNING: {len(duplicates)} duplicate records detected")
        print(f"Duplicated on: {key_columns}")
        print(duplicates[key_columns].head(10))

        return {
            'duplicate_count': len(duplicates),
            'duplicate_pct': len(duplicates) / len(data),
            'status': 'WARNING' if len(duplicates) < 10 else 'CRITICAL'
        }
    else:
        print("✓ No duplicates detected")
        return {'duplicate_count': 0, 'status': 'OK'}

# Check for duplicates in TDE sales
sales_data = pd.read_parquet('data/processed/tde_sales_latest.parquet')
duplicate_check = detect_duplicates(
    sales_data,
    key_columns=['application_signed_date', 'contract_value', 'product_code']
)
```

### Schema Validation

**Check Script**:
```python
def validate_schema(data, expected_schema):
    """Validate data schema matches expected."""

    schema_issues = []

    # Check columns exist
    for col, dtype in expected_schema.items():
        if col not in data.columns:
            schema_issues.append(f"Missing column: {col}")
        elif data[col].dtype != dtype:
            schema_issues.append(f"Column {col}: expected {dtype}, got {data[col].dtype}")

    # Check for unexpected columns
    unexpected_cols = set(data.columns) - set(expected_schema.keys())
    if unexpected_cols:
        schema_issues.append(f"Unexpected columns: {unexpected_cols}")

    if schema_issues:
        print("⚠️  Schema Validation Issues:")
        for issue in schema_issues:
            print(f"  - {issue}")
        return {'status': 'WARNING', 'issues': schema_issues}
    else:
        print("✓ Schema validation passed")
        return {'status': 'OK', 'issues': []}

# Expected TDE schema
expected_tde_schema = {
    'application_signed_date': 'datetime64[ns]',
    'contract_value': 'float64',
    'product_code': 'object',
    'buffer_percentage': 'float64',
    'cap_rate': 'float64'
}

schema_validation = validate_schema(sales_data, expected_tde_schema)
```

---

## Data Drift Detection

### Feature Distribution Monitoring

**Purpose**: Detect shifts in feature distributions over time

**Script**:
```python
from scipy import stats

def detect_feature_drift(recent_data, historical_data, features):
    """Detect distribution shifts using Kolmogorov-Smirnov test."""

    drift_report = {}

    for feat in features:
        # KS test: Are distributions significantly different?
        statistic, p_value = stats.ks_2samp(
            historical_data[feat].dropna(),
            recent_data[feat].dropna()
        )

        # Calculate mean shift in standard deviations
        historical_mean = historical_data[feat].mean()
        historical_std = historical_data[feat].std()
        recent_mean = recent_data[feat].mean()

        mean_shift_sigma = (recent_mean - historical_mean) / historical_std

        # Determine status
        if abs(mean_shift_sigma) > 2.0:
            status = 'CRITICAL'
        elif abs(mean_shift_sigma) > 1.5:
            status = 'WARNING'
        else:
            status = 'OK'

        drift_report[feat] = {
            'ks_statistic': statistic,
            'p_value': p_value,
            'mean_shift_sigma': mean_shift_sigma,
            'status': status
        }

    return pd.DataFrame(drift_report).T

# Detect drift in critical features
model_data = pd.read_csv('data/processed/model_features.csv')

recent = model_data[model_data['date'] >= datetime.now() - timedelta(weeks=13)]
historical = model_data[model_data['date'] < datetime.now() - timedelta(weeks=13)]

critical_features = [
    'prudential_rate_current',
    'competitor_top5_t2',
    'sales_target_contract_t5',
    'market_vix'
]

drift_report = detect_feature_drift(recent, historical, critical_features)

print("\nFeature Drift Detection:")
print(drift_report[['mean_shift_sigma', 'p_value', 'status']])

# Alert on significant drift
drift_issues = drift_report[drift_report['status'].isin(['WARNING', 'CRITICAL'])]
if len(drift_issues) > 0:
    print(f"\n⚠️  {len(drift_issues)} features showing distribution drift")
```

### Sales Volume Drift

**Purpose**: Detect anomalous sales patterns

**Script**:
```python
def detect_sales_drift(sales_data, window_weeks=13):
    """Detect anomalous sales patterns."""

    # Aggregate to weekly sales
    weekly_sales = sales_data.groupby(
        pd.Grouper(key='application_signed_date', freq='W')
    )['contract_value'].sum()

    # Calculate rolling statistics
    rolling_mean = weekly_sales.rolling(window=window_weeks).mean()
    rolling_std = weekly_sales.rolling(window=window_weeks).std()

    # Detect anomalies (> 2 std dev from rolling mean)
    z_scores = (weekly_sales - rolling_mean) / rolling_std
    anomalies = weekly_sales[abs(z_scores) > 2]

    if len(anomalies) > 0:
        print(f"⚠️  {len(anomalies)} anomalous weeks detected:")
        print(anomalies)

        return {
            'anomaly_count': len(anomalies),
            'anomaly_dates': anomalies.index.tolist(),
            'status': 'WARNING' if len(anomalies) < 3 else 'CRITICAL'
        }
    else:
        print("✓ No sales anomalies detected")
        return {'anomaly_count': 0, 'status': 'OK'}

# Run sales drift detection
sales_data = pd.read_parquet('data/processed/tde_sales_latest.parquet')
sales_drift = detect_sales_drift(sales_data, window_weeks=13)
```

---

## Pipeline Health Monitoring

### End-to-End Pipeline Test

**Purpose**: Verify complete data pipeline from S3 to model features

**Script**:
```python
def test_pipeline_end_to_end():
    """Test complete data pipeline."""

    tests_passed = 0
    tests_failed = 0

    # Test 1: S3 Access
    try:
        import boto3
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket='pruvpcaws031-east', Prefix='rila/sales/', MaxKeys=1)
        print("✓ Test 1: S3 access successful")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 1: S3 access failed: {e}")
        tests_failed += 1

    # Test 2: Raw Data Loading
    try:
        sales_raw = pd.read_parquet('data/raw/tde_sales/latest/')
        assert len(sales_raw) > 100000, "Insufficient records"
        print(f"✓ Test 2: Raw data loaded ({len(sales_raw)} records)")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 2: Raw data loading failed: {e}")
        tests_failed += 1

    # Test 3: Data Processing
    try:
        from src.data.pipeline import process_sales_data
        sales_processed = process_sales_data(sales_raw)
        assert len(sales_processed) > 0, "Processing produced empty data"
        print(f"✓ Test 3: Data processing successful ({len(sales_processed)} records)")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 3: Data processing failed: {e}")
        tests_failed += 1

    # Test 4: Feature Engineering
    try:
        from src.features.engineering import engineer_features
        features = engineer_features(sales_processed)
        assert features.shape[1] > 500, "Insufficient features generated"
        print(f"✓ Test 4: Feature engineering successful ({features.shape[1]} features)")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 4: Feature engineering failed: {e}")
        tests_failed += 1

    # Test 5: Model Loading
    try:
        from src.models.inference import PriceElasticityInference
        model = PriceElasticityInference.load('rila_6y20b')
        assert model is not None
        print("✓ Test 5: Model loading successful")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 5: Model loading failed: {e}")
        tests_failed += 1

    # Summary
    print(f"\nPipeline Health: {tests_passed}/{tests_passed + tests_failed} tests passed")

    if tests_failed == 0:
        return {'status': 'OK', 'tests_passed': tests_passed}
    elif tests_failed <= 1:
        return {'status': 'WARNING', 'tests_passed': tests_passed, 'tests_failed': tests_failed}
    else:
        return {'status': 'CRITICAL', 'tests_passed': tests_passed, 'tests_failed': tests_failed}

# Run end-to-end test
pipeline_health = test_pipeline_end_to_end()
```

---

## Automated Quality Tests

### pytest Data Quality Suite

**Location**: `tests/data_quality/test_data_quality.py`

```python
import pytest
import pandas as pd
from datetime import datetime, timedelta

class TestDataQuality:
    """Automated data quality tests."""

    def test_tde_data_freshness(self):
        """TDE data should be < 14 days old."""
        sales = pd.read_parquet('data/processed/tde_sales_latest.parquet')
        latest_date = sales['application_signed_date'].max()
        days_old = (datetime.now() - latest_date).days

        assert days_old <= 14, f"TDE data is {days_old} days old (max: 14)"

    def test_wink_data_freshness(self):
        """WINK data should be < 14 days old."""
        wink = pd.read_parquet('data/processed/wink_rates_latest.parquet')
        latest_date = wink['rate_effective_date'].max()
        days_old = (datetime.now() - latest_date).days

        assert days_old <= 14, f"WINK data is {days_old} days old (max: 14)"

    def test_critical_features_complete(self):
        """Critical features should have < 1% missing data."""
        model_data = pd.read_csv('data/processed/model_features.csv')

        critical_features = [
            'prudential_rate_current',
            'competitor_top5_t2',
            'sales_target_contract_t5'
        ]

        for feat in critical_features:
            missing_pct = model_data[feat].isna().sum() / len(model_data)
            assert missing_pct < 0.01, f"{feat} has {missing_pct:.2%} missing (max: 1%)"

    def test_carrier_coverage(self):
        """All 8 carriers should have data in last 30 days."""
        wink = pd.read_parquet('data/processed/wink_rates_latest.parquet')
        recent = wink[wink['rate_effective_date'] >= datetime.now() - timedelta(days=30)]

        expected_carriers = [
            'Athene', 'Brighthouse', 'Equitable', 'Ameriprise',
            'Jackson', 'Lincoln', 'Symetra', 'Transamerica'
        ]

        for carrier in expected_carriers:
            count = len(recent[recent['carrier'] == carrier])
            assert count > 0, f"No recent data for {carrier}"

    def test_rate_ranges(self):
        """Cap rates should be between 0.5% and 4.5%."""
        model_data = pd.read_csv('data/processed/model_features.csv')

        pru_rate = model_data['prudential_rate_current']
        assert pru_rate.min() >= 0.005, f"Min rate too low: {pru_rate.min()}"
        assert pru_rate.max() <= 0.045, f"Max rate too high: {pru_rate.max()}"

# Run tests
# pytest tests/data_quality/test_data_quality.py -v
```

---

## Alerting and Escalation

### Alert Priority Matrix

| Issue | Severity | Response Time | Escalation |
|-------|----------|---------------|------------|
| Data > 14 days old | P1 | 4 hours | Data Engineering |
| Missing 2+ carriers | P1 | 4 hours | WINK vendor |
| Feature drift > 2σ | P2 | 1 day | Model Owner |
| Missing < 5% data | P2 | 1 day | Data Engineering |
| Schema changes | P1 | 4 hours | Data Engineering + Model Owner |
| Pipeline test failures | P1 | 4 hours | Model Owner |

### Email Alert Template

```
Subject: [P1 DATA QUALITY] {Issue Description}

Issue: {Brief description}
Severity: P1
Detected: {Timestamp}
Data Source: {TDE/WINK/FRED}

Impact:
- {Business impact}
- {Technical impact}

Metrics:
- {Specific metrics and thresholds}

Investigation:
- {Initial findings}
- {Suspected root cause}

Action Items:
1. {Immediate action}
2. {Follow-up action}

ETA: {Expected resolution time}

Contact {Model Owner} with questions.
```

---

## Related Documentation

### Operations
- [MONITORING_GUIDE.md](MONITORING_GUIDE.md) - Performance monitoring
- [EMERGENCY_PROCEDURES.md](EMERGENCY_PROCEDURES.md) - Incident response
- [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) - Pre-deployment validation

### Validation
- [../practices/LEAKAGE_CHECKLIST.md](../practices/LEAKAGE_CHECKLIST.md) - Data leakage checks
- [../methodology/validation_guidelines.md](../methodology/validation_guidelines.md) - Model validation

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-29 | Initial data quality monitoring guide |

---

## Summary

**Daily Checks**:
- Data freshness (TDE, WINK, FRED)
- S3 bucket accessibility

**Biweekly Checks** (before model training):
- Data completeness (< 1% missing)
- Carrier coverage (all 8 carriers)
- Range validation (rates, sales volumes)
- Feature drift detection (> 2σ shift)
- End-to-end pipeline test

**Automated Tests**:
- pytest data quality suite
- Run as part of model training workflow
- Fail fast if data quality issues detected

**Key Principle**: Never train models on bad data. Data quality checks are gates, not suggestions.

# AWS Setup Guide

**Configure AWS credentials for production data access.**

---

## Overview

Production RILA data is stored in AWS S3. You'll need:
1. AWS CLI installed
2. IAM credentials or role assumption
3. Network access (VPN if required)

**For testing/development**: Use `environment="fixture"` instead—no AWS needed!

---

## Prerequisites

### 1. Install AWS CLI

```bash
# macOS (Homebrew)
brew install awscli

# Linux (apt)
sudo apt-get install awscli

# Windows (MSI installer)
# Download from https://aws.amazon.com/cli/

# Verify installation
aws --version
```

### 2. Get Your Credentials

Contact your team lead for:
- **IAM User credentials** (Access Key ID + Secret Access Key), or
- **IAM Role ARN** for role assumption
- **Bucket name** and path permissions

---

## Configuration Methods

### Method 1: AWS CLI Configuration (Simplest)

```bash
# Run interactive setup
aws configure

# Enter when prompted:
AWS Access Key ID [None]: YOUR_ACCESS_KEY
AWS Secret Access Key [None]: YOUR_SECRET_KEY
Default region name [None]: us-east-1
Default output format [None]: json
```

This creates `~/.aws/credentials` and `~/.aws/config`.

### Method 2: Environment Variables

```bash
# Add to ~/.bashrc or ~/.zshrc
export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"
export AWS_DEFAULT_REGION="us-east-1"

# Reload shell
source ~/.bashrc
```

### Method 3: Named Profiles (Multiple Accounts)

```bash
# Configure a named profile
aws configure --profile rila-production

# Use the profile
export AWS_PROFILE=rila-production

# Or specify in code
aws_config = {
    "profile_name": "rila-production",
    ...
}
```

---

## Role Assumption (STS)

If using IAM role assumption (common in enterprise setups):

### 1. Configure the Source Profile

```bash
# ~/.aws/credentials
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
```

### 2. Configure Role Assumption

```bash
# ~/.aws/config
[profile rila-data]
role_arn = arn:aws:iam::123456789012:role/RILADataAccess
source_profile = default
region = us-east-1
```

### 3. Test Role Assumption

```bash
aws sts get-caller-identity --profile rila-data
```

### 4. Use in Code

```python
aws_config = {
    "sts_endpoint_url": "https://sts.us-east-1.amazonaws.com",
    "role_arn": "arn:aws:iam::123456789012:role/RILADataAccess",
    "xid": "your_employee_id",  # If required by your org
    "bucket_name": "prudential-rila-data",
}

interface = create_interface(
    "6Y20B",
    environment="aws",
    adapter_kwargs={"config": aws_config}
)
```

---

## Verifying Access

### Check S3 Access

```bash
# List buckets (if permitted)
aws s3 ls

# List contents of data bucket
aws s3 ls s3://your-bucket-name/

# Download a test file
aws s3 cp s3://your-bucket-name/test-file.txt ./test-file.txt
```

### Check from Python

```python
import boto3

# Create S3 client
s3 = boto3.client('s3')

# List objects
response = s3.list_objects_v2(
    Bucket='your-bucket-name',
    Prefix='rila/',
    MaxKeys=10
)

for obj in response.get('Contents', []):
    print(obj['Key'])
```

### Test the Interface

```python
from src.notebooks import create_interface

aws_config = {
    "sts_endpoint_url": "https://sts.us-east-1.amazonaws.com",
    "role_arn": "arn:aws:iam::123456789012:role/RILADataAccess",
    "xid": "your_xid",
    "bucket_name": "your-bucket",
}

try:
    interface = create_interface(
        "6Y20B",
        environment="aws",
        adapter_kwargs={"config": aws_config}
    )
    df = interface.load_data()
    print(f"Successfully loaded {len(df)} rows")
except Exception as e:
    print(f"Error: {e}")
```

---

## Common Issues

### "NoCredentialsError: Unable to locate credentials"

**Cause**: No credentials configured.

**Fix**:
```bash
# Verify credentials file exists
cat ~/.aws/credentials

# Or check environment variables
echo $AWS_ACCESS_KEY_ID
```

### "ExpiredTokenException"

**Cause**: Temporary credentials expired.

**Fix**:
```bash
# If using role assumption, re-assume the role
aws sts get-session-token --profile your-profile
```

### "AccessDenied" or "403 Forbidden"

**Cause**: IAM permissions insufficient.

**Required permissions**:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket-name",
                "arn:aws:s3:::your-bucket-name/*"
            ]
        }
    ]
}
```

### "Connection timed out"

**Cause**: Network access blocked.

**Fix**:
1. Connect to VPN (if required by your organization)
2. Check firewall rules
3. Verify endpoint URL is correct

---

## Security Best Practices

### DO

- Use IAM roles instead of long-lived access keys when possible
- Rotate access keys regularly
- Use least-privilege permissions
- Store credentials in `~/.aws/credentials`, not in code

### DON'T

- Commit credentials to git
- Share access keys via email/Slack
- Use root account credentials
- Store credentials in notebook cells

### .gitignore Verification

Ensure these patterns are in `.gitignore`:

```
# AWS credentials
.aws/
*.pem
credentials.json
secrets.json
```

---

## Environment-Specific Config

### Development (Local Machine)

```python
# Use fixtures for development
interface = create_interface("6Y20B", environment="fixture")
```

### Staging/Testing (AWS but non-prod)

```python
staging_config = {
    "bucket_name": "staging-rila-data",
    # ... other config
}
interface = create_interface("6Y20B", environment="aws",
                            adapter_kwargs={"config": staging_config})
```

### Production

```python
prod_config = {
    "bucket_name": "production-rila-data",
    # ... other config
}
interface = create_interface("6Y20B", environment="aws",
                            adapter_kwargs={"config": prod_config})
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Configure credentials | `aws configure` |
| Test credentials | `aws sts get-caller-identity` |
| List buckets | `aws s3 ls` |
| Download file | `aws s3 cp s3://bucket/key ./local` |
| Use named profile | `export AWS_PROFILE=profile-name` |

---

## Getting Help

If you encounter issues:

1. Check this guide's "Common Issues" section
2. Verify with `aws sts get-caller-identity`
3. Contact your team lead for IAM/permissions issues
4. See `TROUBLESHOOTING.md` for general debugging

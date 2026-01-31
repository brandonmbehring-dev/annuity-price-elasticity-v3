# Changelog

All notable changes to the annuity-price-elasticity project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation suite (TESTING_STRATEGY, VALIDATION_EVIDENCE, COMMON_PITFALLS)
- AI collaboration methodology documentation
- Sphinx documentation infrastructure with napoleon extension

## [3.0.0] - 2026-01-XX

### Added
- **Multi-Product Architecture**: Support for RILA (6Y20B, 6Y10B, 10Y20B), FIA, and MYGA products
- **Dependency Injection Pattern**: Clean separation between data sources and business logic
  - `S3Adapter` for production AWS access
  - `LocalAdapter` for development
  - `FixtureAdapter` for testing
- **UnifiedNotebookInterface**: Single interface for all notebook operations
- **Aggregation Strategies**: Product-specific competitor aggregation
  - `WeightedAggregation` (RILA)
  - `TopNAggregation` (FIA)
  - `FirmLevelAggregation` (MYGA)
- **Leakage Gates**: 5 automated validation gates
  - Shuffled target test
  - R-squared threshold check
  - Improvement threshold check
  - Lag-0 feature detection
  - Temporal boundary check
- **Property-Based Testing**: 10 Hypothesis-based test files
- **Anti-Pattern Test Suite**: 5 comprehensive leakage detection test files
- **Bootstrap Inference**: 10,000-sample bootstrap with 95% CI coverage

### Changed
- Migrated from monolithic architecture to modular DI-based design
- Test count: 2,467 -> 6,126 tests
- Documentation: 34,557+ lines across 96+ files
- Coverage: 55% -> 70%+ (core modules >80%)

### Fixed
- Lag-0 competitor feature detection now comprehensive
- Temporal boundary validation enforced in all CV splits
- Market-weight leakage prevented via trailing windows
- contract_issue_date replaced with application_signed_date (eliminated 110-day lookahead)

### Removed
- Legacy monolithic notebook structure
- Direct AWS calls in notebooks (replaced with adapters)
- Synthetic data fallbacks (fail-fast pattern enforced)

## [2.0.0] - 2025-06-XX

### Added
- Bootstrap Ridge Ensemble with 1,000 samples
- FIA methodology framework (alpha)
- Temporal cross-validation
- Basic leakage detection

### Changed
- Improved feature engineering pipeline
- Enhanced coefficient sign validation

### Fixed
- Own-rate coefficient sign (now correctly positive)

## [1.0.0] - 2025-01-XX

### Added
- Initial RILA price elasticity model
- Basic OLS regression
- Quarterly reporting pipeline

---

## Migration Guide: v2 -> v3

### Breaking Changes

1. **Data Loading**: Direct AWS calls replaced with adapter pattern
   ```python
   # v2 (deprecated)
   df = load_from_s3(bucket, key)

   # v3 (recommended)
   from src.notebooks import create_interface
   interface = create_interface("6Y20B", environment="aws")
   df = interface.load_data()
   ```

2. **Feature Names**: Standardized naming convention
   - Old: `competitor_rate`, `comp_mean_lag_0`
   - New: `competitor_weighted_t2`, `competitor_mean_t3`

3. **Configuration**: Centralized in `ProductConfig`
   ```python
   # v3
   from src.config import ProductConfig
   config = ProductConfig.for_product("6Y20B")
   ```

### Upgrade Steps

1. Update imports to use `src.notebooks.interface`
2. Replace direct data loading with adapter pattern
3. Update feature names to standardized format
4. Run `make test-all` to verify compatibility

# Quick Reference - Task-Oriented Claude Code Workflows

**Purpose:** Immediate, actionable guidance for common tasks. Each section <100 lines, highly focused.

---

## Adding New Features

### Before Starting
- [ ] **Pattern Check**: Which canonical pattern does this extend? (See MODULE_HIERARCHY.md)
- [ ] **Entry Point**: Use `src.config.product_config`, `src.data.adapters`, or `src.features.aggregation`
- [ ] **Validation Baseline**: Run `make quick-check` (current state)

### Implementation Steps
1. **Define Configuration** (if needed):
   ```python
   # In appropriate config file, add TypedDict
   from src.config.product_config import build_product_config
   ```

2. **Implement Core Function**:
   ```python
   # Follow atomic function pattern (30-50 lines max)
   # Add context anchor if creating new key module
   # Use absolute imports: from src.module import Class
   ```

3. **Integration**:
   ```python
   # Add to UnifiedNotebookInterface if user-facing
   # Use create_interface factory for new product types
   # Update config builders if new configurations needed
   ```

### Validation Checklist
- [ ] `make quick-check` (must pass)
- [ ] `make test` (all tests passing)
- [ ] Mathematical equivalence maintained (if refactoring)
- [ ] Integration tested with create_interface()
- [ ] Follows DRY principles (no code duplication)

### Documentation Updates
- [ ] Update MODULE_HIERARCHY.md (if new module or pattern)
- [ ] Add context anchor to new key modules
- [ ] Update CLAUDE.md (if new entry point)

---

## Debugging Issues

### Issue Investigation Protocol
1. **State the Problem Clearly**:
   ```
   Expected: [Specific behavior]
   Observed: [Actual behavior]
   Context: [When/where it occurs]
   ```

2. **Check v2 Architecture Compliance**:
   ```bash
   make quick-check        # Fast smoke test
   make verify-quick       # Stub hunter + type validation
   ```

3. **Trace Architectural Flow**:
   - Notebook → create_interface() factory → UnifiedNotebookInterface
   - Data adapter layer (S3/Local/Fixture)
   - Aggregation strategy (RILA/FIA/MYGA)
   - Check each handoff point for data/config integrity

### Solution Implementation
- [ ] **Single Pattern Rule**: Use only canonical implementations
- [ ] **No Competing Approaches**: Don't create alternative solutions
- [ ] **Test Against Original Issue**: Verify fix addresses root cause
- [ ] **Regression Check**: Ensure no new issues introduced

### Validation Protocol
```bash
# Must pass all checks before considering issue resolved:
make quick-check         # Fast validation
make test               # Full test suite
make verify-quick       # Stub + type checks
python -c "from src.notebooks import create_interface; print('OK')"  # Import integrity
```

### When Stuck
1. **Check Context Anchors**: Review objectives in affected modules
2. **Review MODULE_HIERARCHY.md**: Confirm correct architectural boundaries
3. **Trace create_interface()**: Understand factory pattern wiring
4. **Validate Assumptions**: Use `assert` statements to verify data state

---

## Modifying Architecture

### Impact Assessment (Required First Step)
- [ ] **Affected Patterns**: List all canonical patterns that change
- [ ] **Multi-Product Impact**: Which products (RILA/FIA/MYGA) are affected?
- [ ] **Breaking Changes**: Any interface modifications to UnifiedNotebookInterface?
- [ ] **Migration Path**: How to update existing factory calls?

### Safe Change Protocol
1. **Document First**: Update MODULE_HIERARCHY.md before implementing
2. **Single Implementation**: Implement new pattern in one place first
3. **Validate Integration**: Test with `make quick-check`
4. **Migrate Existing**: Update all affected code to new pattern
5. **Clean Up**: Remove old patterns completely

### Validation Requirements
- [ ] All products still execute correctly (test-rila, test-fia)
- [ ] `make quick-check` passes with new patterns
- [ ] Mathematical equivalence maintained (regression tests)
- [ ] Documentation reflects new architecture
- [ ] Context anchors updated to reflect changes

---

## Configuration Changes

### Configuration System Rules
- **ALWAYS use**: `src.config.product_config` builders
- **NEVER use**: Direct TypedDict instantiation
- **SINGLE SOURCE**: All configs built through canonical functions

### Adding New Configuration Type
```python
# 1. Add TypedDict to config file
from typing import TypedDict

class NewAnalysisConfig(TypedDict):
    enabled: bool
    method: str
    parameters: dict[str, Any]

# 2. Add builder function to product_config.py
def build_new_analysis_config(**kwargs) -> NewAnalysisConfig:
    """Build type-safe configuration for new analysis."""
    return NewAnalysisConfig(
        enabled=kwargs.get('enabled', True),
        method=kwargs.get('method', 'default'),
        parameters=kwargs.get('parameters', {})
    )
```

### Modifying Existing Configuration
- [ ] **Check Usage**: Find all callers with grep/search
- [ ] **Backward Compatibility**: Ensure existing code still works
- [ ] **Default Values**: Provide sensible defaults for new fields
- [ ] **Validation**: Add type checks and business rule validation

---

## Working with Dependencies and Adapters

### Data Source Adapters (Dependency Injection)
```python
from src.data.adapters import get_adapter
from src.notebooks import create_interface

# Create interface with specific adapter (DI pattern)
interface = create_interface(
    product_code="6Y20B",
    environment="aws",
    adapter_kwargs={"config": aws_config}
)

# Or use local/fixture for testing
interface = create_interface(
    product_code="6Y20B",
    environment="fixture"
)

data = interface.load_data()
results = interface.run_inference(data)
```

### Aggregation Strategies
```python
from src.features.aggregation import get_strategy

# Select product-specific strategy
strategy = get_strategy(product_type="RILA")
aggregated_df = strategy.aggregate(competitor_data, weights)
```

### Product Methodologies
```python
from src.products import get_methodology

# Get product-specific constraint rules
methodology = get_methodology("RILA")
validated_params = methodology.validate_constraints(params)
```

---

## Validation Checkpoints

### Quick Smoke Test (30 seconds)
```bash
make quick-check
# Checks: imports, core protocols, pattern compliance
```

### Full Test Suite
```bash
make test              # All tests
make test-rila         # RILA-specific tests
make test-fia          # FIA-specific tests
make coverage          # Generate coverage report
```

### Verification Tooling
```bash
make verify            # Full: equivalence + stubs + types
make verify-quick      # Fast: stubs + types only
make stub-hunter       # Find placeholder implementations
make hardcode-scan     # Find hardcoded product strings
```

### When to Use Each
- **After data filtering**: Run `make quick-check`
- **After adapter changes**: Run `make test-rila` + `make test-fia`
- **After aggregation changes**: Run `make test`
- **After feature engineering**: Run `make verify-quick`
- **Before committing**: Run `make verify-quick` + `make test`

---

## Production Validation Checkpoints

### Validation Pattern
```python
from src.validation.production_validators import (
    run_production_validation_checkpoint,
    ValidationConfig
)

config: ValidationConfig = {
    'checkpoint_name': 'my_checkpoint',
    'version': 1,
    'project_root': '/path/to/project',
    'strict_schema': True,
    'growth_config': {
        'min_growth_pct': 0.0,
        'max_growth_pct': 20.0,
        'warn_on_shrinkage': True,
        'warn_on_high_growth': True
    },
    'critical_columns': None
}

result = run_production_validation_checkpoint(
    df=my_dataframe,
    config=config,
    date_column='date',
    business_rules={}
)
```

### When to Use
- **After data filtering**: Verify expected record counts and schema preservation
- **After data cleanup**: Validate business rules and data quality
- **After aggregation**: Check aggregation ratios and preservation
- **After feature engineering**: Ensure critical features were created correctly

---

## Multi-Product Testing Strategy

### Testing by Product
```bash
make test-rila         # Test RILA products (6Y20B, 6Y10B, 10Y20B)
make test-fia          # Test FIA products
```

### Testing Specific Factory Calls
```python
from src.notebooks import create_interface

# Test RILA with fixture data
rila_interface = create_interface("6Y20B", environment="fixture")
assert rila_interface.product_code == "6Y20B"

# Test FIA with fixture data
fia_interface = create_interface("FIA_PRODUCT", environment="fixture")
assert fia_interface.product_type == "FIA"
```

### Validating Multi-Product Support
- [ ] Adapter pattern works for all products
- [ ] Aggregation strategies handle product-specific logic
- [ ] Methodologies enforce product constraints
- [ ] Configuration builders support all product types

---

## Performance Optimization

### Before Starting Complex Tasks
- [ ] **Review Objective**: Clear success criteria defined?
- [ ] **Identify Patterns**: Which canonical patterns apply?
- [ ] **Check Baseline**: Run `make quick-check` for current state
- [ ] **Plan Handoffs**: Note logical breakpoints between modules

### During Long Tasks
- [ ] **Use Checkpoints**: Save progress every 2-3 phases
- [ ] **Validate Incrementally**: Run `make quick-check` as you go
- [ ] **Maintain Context**: If confused, review context anchors and MODULE_HIERARCHY.md
- [ ] **Stay Focused**: Resist adding "just one more thing"

### After Task Completion
- [ ] **Quick Validation**: `make quick-check` (must pass)
- [ ] **Full Test Suite**: `make test` (all passing)
- [ ] **Documentation Update**: If patterns changed, update guides
- [ ] **Knowledge Capture**: Note new patterns discovered

---

## Emergency Procedures

### When Things Break
1. **STOP**: Don't make more changes until you understand the issue
2. **Isolate**: Identify exactly what changed since last working state
3. **Rollback Option**: Use git to return to known good state if needed
4. **Validate Fix**: Ensure solution addresses root cause, not symptoms

### Factory Pattern Issues
```bash
# Validate create_interface factory:
python -c "from src.notebooks import create_interface; print('Factory: OK')"

# Test with fixture data:
python -c "from src.notebooks import create_interface; i = create_interface('6Y20B', environment='fixture'); print('Fixture: OK')"
```

### Adapter Integration Issues
```bash
# Check available adapters:
python -c "from src.data.adapters import get_adapter; print(get_adapter('aws'))"

# Validate DI pattern:
make verify-quick
```

### Mathematical Equivalence Failures
- [ ] **Regression Test**: `make validate` or `make verify`
- [ ] **Data Integrity**: Verify input data unchanged
- [ ] **Algorithm Logic**: Check for unintended modifications
- [ ] **Configuration Drift**: Ensure configs match original implementation

---

## Success Indicators

### Task Completion (Green Lights)
- ✅ `make quick-check`: Passes with 0 errors
- ✅ `make test`: All tests passing
- ✅ `make verify-quick`: Stub/type validation clean
- ✅ Documentation: Updated and consistent
- ✅ No rework required: Task completed on first attempt

### Repository Health (Maintenance)
- ✅ Regular `make quick-check`: Consistent clean results
- ✅ Documentation currency: All guides reflect current state
- ✅ Multi-product support: All products testable
- ✅ Performance parity: Claude Code performs as expected

---

## Entry Points Quick Lookup

| Task | Import | Function |
|------|--------|----------|
| **Create Analysis Interface** | `from src.notebooks import create_interface` | `create_interface(product_code, environment)` |
| **Get Data Adapter** | `from src.data.adapters import get_adapter` | `get_adapter(adapter_type)` |
| **Get Aggregation Strategy** | `from src.features.aggregation import get_strategy` | `get_strategy(product_type)` |
| **Get Product Methodology** | `from src.products import get_methodology` | `get_methodology(product_type)` |
| **Build Configuration** | `from src.config.product_config import build_*` | `build_product_config(...)` |

---

**Remember**: This repository implements multi-product architecture patterns. Use dependency injection consistently (adapters, strategies, methodologies) and trust the systems we've built (quick-check, verify tooling, canonical patterns) for optimal performance.

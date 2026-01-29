# Decision Log: RILA Price Elasticity

**Why we made key modeling and architectural decisions.**

This log captures the reasoning behind important choices. When you wonder "why did they do it this way?", check here first.

---

## Modeling Decisions

### D1: Use Lagged Competitor Rates (Not Lag-0)

**Date**: 2025 (inherited from FIA project)

**Decision**: Use competitor rates at t-1 or earlier; forbid lag-0.

**Context**: Initial models used simultaneous competitor rates, showing strong correlations.

**Alternatives Considered**:
1. Use lag-0 competitors (rejected)
2. Use lag-1+ competitors (selected)
3. Use instrumental variables (considered for future)

**Rationale**:
- Lag-0 creates simultaneity bias: both C_t and Sales_t respond to market conditions
- Lagged competitors break this simultaneity
- Econometrically standard approach (Angrist & Pischke, Wooldridge)

**Trade-offs**:
- Lose some predictive power (lag-0 would correlate more)
- Gain causal identification (what we actually need)

**See**: `knowledge/analysis/CAUSAL_FRAMEWORK.md`, Section 3.3

---

### D2: Allow Own Rate at Lag-0

**Date**: 2025 (inherited from FIA project)

**Decision**: Own rate (prudential_rate_current) IS allowed at t=0.

**Context**: Unlike competitors, we control our own rate.

**Rationale**:
- Prudential sets rates before observing application-date sales
- Contract-issue-date lag (19-76 days) creates identification window
- Treatment variable by definition can be at t=0

**See**: `knowledge/analysis/CAUSAL_FRAMEWORK.md`, Section 3.1

---

### D3: Expect Positive Own-Rate Coefficient

**Date**: 2025

**Decision**: Require P_coefficient > 0 in economic constraints.

**Context**: Cap rate is fundamentally different from traditional "price."

**Rationale**:
- Cap rate is a YIELD (customer benefit), not a cost
- Higher cap rate = better return = more attractive = more sales
- This is yield elasticity, not price elasticity

**Alternatives Considered**:
1. Allow negative coefficient (rejected - economically wrong)
2. Require positive coefficient (selected)

**See**: `knowledge/domain/RILA_ECONOMICS.md`

---

### D4: Use Market-Share Weighted Competitor Aggregation (RILA)

**Date**: 2026-01 (v2 development)

**Decision**: Aggregate competitor rates using market-share weighting for RILA.

**Context**: FIA used simple top-N mean; RILA market is more concentrated.

**Rationale**:
- RILA: Top 3 carriers hold ~60% market share
- Allianz alone is ~35%
- Weighting by share reflects actual competitive pressure
- Simple mean would underweight dominant competitors

**Alternatives Considered**:
1. Simple top-N mean (used for FIA, rejected for RILA)
2. Market-share weighted (selected)
3. Exclude Allianz entirely (considered, not implemented)

**Trade-offs**:
- More complex to implement
- Requires market share data (additional data dependency)
- Better reflects actual competition

**See**: `knowledge/integration/LESSONS_LEARNED.md`, Trap #5

---

### D5: Use Logit Transform for Sales

**Date**: 2025 (inherited)

**Decision**: Transform sales via logit scaling before modeling.

**Context**: Sales are bounded (can't be negative, have practical maximum).

**Formula**:
```python
sales_scaled = 0.95 * sales / max(sales)
sales_logit = logit(sales_scaled)
```

**Rationale**:
- Accounts for saturation effects (can't grow infinitely)
- Produces approximately normal distribution
- Standard in demand modeling

**Open Questions**:
- 0.95 saturation parameter is somewhat arbitrary
- Could test sensitivity to this parameter

**See**: `knowledge/analysis/CAUSAL_FRAMEWORK.md`, Section 5.1

---

### D6: Include Buffer Level as Control (RILA-Specific)

**Date**: 2026-01

**Decision**: Include buffer level as stratification/control variable.

**Context**: RILA products vary by buffer (10%, 15%, 20%, 25%).

**Rationale**:
- Different buffers attract different buyer segments
- Risk tolerance varies by buffer level
- Elasticity may differ by buffer
- Must control to avoid confounding

**Alternatives Considered**:
1. Ignore buffer (rejected - omits important variable)
2. Stratify analysis by buffer (partial - for interaction testing)
3. Include as control (selected)

**See**: `knowledge/integration/LESSONS_LEARNED.md`, Trap #4

---

## Architectural Decisions

### A1: Use Dependency Injection Pattern

**Date**: 2026-01 (v2 architecture)

**Decision**: Implement adapter pattern for data sources.

**Context**: Need to run same analysis on fixtures, local, and AWS data.

**Components**:
- `DataSourceAdapter` protocol (abstract interface)
- `S3Adapter`, `LocalAdapter`, `FixtureAdapter` (implementations)
- Factory function to create based on environment

**Rationale**:
- Same analysis code works for all data sources
- Easy testing without AWS access
- Clean separation of concerns

**Trade-offs**:
- More code to write upfront
- Additional abstraction layer
- Worth it for flexibility

**See**: `docs/architecture/MULTI_PRODUCT_DESIGN.md`

---

### A2: Single UnifiedNotebookInterface Entry Point

**Date**: 2026-01

**Decision**: Create one interface for all notebook operations.

**Context**: V1 had scattered functions across multiple modules.

**Rationale**:
- Easier onboarding (one entry point)
- Consistent API across products
- Encapsulates complexity

**Trade-offs**:
- Interface may grow large over time
- Requires careful method decomposition

**See**: `src/notebooks/interface.py`

---

### A3: Use Protocol-Based Abstractions

**Date**: 2026-01

**Decision**: Define interfaces using Python `Protocol` classes.

**Context**: Need stable contracts between components.

**Rationale**:
- Structural subtyping (duck typing with type hints)
- No runtime inheritance required
- Clear contracts for implementers

**See**: `src/core/protocols.py`

---

### A4: Store Fixtures in Git LFS

**Date**: 2025

**Decision**: Track fixture parquet files with Git LFS.

**Context**: Fixtures are 74MB+ (RILA) + 14MB+ (FIA).

**Rationale**:
- Can't commit large binaries to regular Git
- Need fixtures for offline testing
- LFS handles large files efficiently

**Trade-offs**:
- Requires LFS setup for new developers
- Storage costs

---

### A5: Custom Exception Hierarchy

**Date**: 2026-01

**Decision**: Create exception hierarchy with business context.

**Context**: Generic exceptions don't convey impact or action.

**Design**:
```python
class ElasticityBaseError(Exception):
    business_impact: str
    required_action: str
```

**Rationale**:
- Errors tell you what went wrong AND what to do
- Consistent messaging across codebase
- Better for debugging and support

**See**: `src/core/exceptions.py`

---

## Data Decisions

### DD1: Use Application-Date Sales (Not Contract-Issue-Date)

**Date**: 2025 (inherited)

**Decision**: Aggregate sales by application-signed date.

**Context**: TDE has both application and contract-issue dates.

**Rationale**:
- Application date = when customer decided
- Contract-issue date = when paperwork processed (19-76 days later)
- Application date is closer to the decision being influenced by rates

**Trade-offs**:
- Application-date data may have revisions
- Contract-issue-date is more stable but lagged

---

### DD2: Holiday Mask (Exclude Dec/Jan Boundary)

**Date**: 2025 (inherited)

**Decision**: Exclude days 1-12 and 360-366 from analysis.

**Context**: Application-date sales show anomalies around holidays.

**Rationale**:
- Late December: offices closed, $0 sales
- Early January: catch-up spike
- These aren't rate-driven fluctuations

**Trade-offs**:
- Lose ~5% of observations
- Avoid spurious patterns

**See**: `knowledge/analysis/FEATURE_RATIONALE.md`, Section 3

---

### DD3: Data Maturity Threshold (60 Days)

**Date**: 2025

**Decision**: Don't use data less than 60 days old.

**Context**: Application-date sales can be revised as contracts process.

**Rationale**:
- Recent data is incomplete
- Revisions can be substantial
- Wait for data to "mature"

**Trade-offs**:
- Lose most recent 60 days
- More stable estimates

---

## Future Considerations

### F1: VIX as Explicit Control

**Status**: Under investigation

**Hypothesis**: VIX may be especially important for RILA (downside protection value).

**Decision needed**: Whether to include VIX explicitly vs rely on DGS5 as proxy.

---

### F2: Buffer-Elasticity Interaction

**Status**: Under investigation

**Hypothesis**: Elasticity may vary by buffer level.

**Decision needed**: Whether to estimate separate models by buffer or use interaction terms.

---

### F3: FIA Cross-Elasticity

**Status**: Not controlled

**Risk**: RILA and FIA may be substitutes.

**Decision needed**: Whether to include FIA rates as features in RILA model.

---

## Decision Template

When making new decisions, document using this template:

```markdown
### [Code]: [Short Title]

**Date**: YYYY-MM

**Decision**: [What we decided]

**Context**: [Background and problem]

**Alternatives Considered**:
1. [Option A] (why rejected/selected)
2. [Option B] (why rejected/selected)

**Rationale**: [Why this choice]

**Trade-offs**:
- [Pro/con 1]
- [Pro/con 2]

**See**: [Link to detailed documentation]
```

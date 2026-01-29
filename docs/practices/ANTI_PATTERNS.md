# Anti-Patterns: What to Avoid

**Purpose**: Document common mistakes that lead to poor outcomes.
**Last Updated**: 2026-01-20
**Source**: Synthesized from CLAUDE_BEST_PRACTICES_2025.md and project experience

---

## The 4 Key Anti-Patterns

### 1. Overengineering

**The mistake**: Creating factories, abstractions, or complex architectures for simple problems.

**Symptoms**:
- AbstractFactory for a single product type
- 10 classes where 2 functions would suffice
- "Design patterns" that add complexity without value

**Why it happens**:
- Anticipating requirements that may never materialize
- Copying patterns from larger codebases
- "Best practices" applied without context

**The fix**:
- Start simple, refactor when needed
- YAGNI (You Aren't Gonna Need It)
- `/simple-fix` for one-off problems

**Example**:
```python
# WRONG: Overengineered
class RateCalculatorFactory:
    def create_calculator(self, product_type):
        if product_type == "RILA":
            return RILARateCalculator()
        # ... only ever used for RILA

# CORRECT: Simple
def calculate_rila_rate(data):
    return data['cap_rate'] - data['spread']
```

---

### 2. Bloated Context

**The mistake**: CLAUDE.md files that are too long, causing Claude Code to miss important information.

**Symptoms**:
- CLAUDE.md > 15KB
- Duplicate information across files
- Outdated instructions that contradict current practice

**Why it happens**:
- Adding without removing
- Not pruning after refactors
- "More context is always better" assumption

**The fix**:
- Target: CLAUDE.md < 13KB
- Use references to knowledge/ instead of inline content
- Periodic pruning of outdated sections

**Measurement**:
```bash
wc -c CLAUDE.md  # Should be < 13000
```

---

### 3. Graceful Degradation (Silent Failure)

**The mistake**: Returning `None` or default values when errors occur, hiding problems.

**Symptoms**:
```python
def get_data():
    try:
        return fetch_from_database()
    except Exception:
        return None  # SILENT FAILURE!
```

**Why it's dangerous**:
- Bugs propagate silently
- Debugging becomes impossible
- Users see incorrect results without warning

**The fix**:
- NEVER return `None` on critical failures
- Raise explicit exceptions with context
- Fail fast, fail loud

**Correct pattern**:
```python
def get_data():
    try:
        return fetch_from_database()
    except DatabaseError as e:
        raise DataFetchError(f"Failed to fetch data: {e}") from e
```

---

### 4. "Too Good" Results

**The mistake**: Accepting model improvements >20% without investigation.

**Symptoms**:
- R² jumps from 0.25 to 0.65
- MAE drops 30% with minor code change
- "Breakthrough" results on first implementation

**Why it's almost always wrong**:
- Data leakage (90% of cases)
- Bug in evaluation code
- Train/test overlap
- Future information in features

**The fix**:
- HALT when improvement > 20%
- Run shuffled target test
- Audit feature construction
- Check date boundaries

**Protocol**:
```python
def validate_improvement(baseline_metric, new_metric):
    improvement = (baseline_metric - new_metric) / baseline_metric
    if improvement > 0.20:
        raise SuspiciousResultError(
            f"Improvement of {improvement:.0%} exceeds 20% threshold. "
            "Investigate for data leakage before accepting."
        )
```

---

## RILA-Specific Anti-Patterns

### 5. Using Simple Competitor Means

**The mistake**: Copying FIA's simple top-N mean for RILA.

**Why wrong**: RILA market is concentrated; simple means underweight dominant players.

**Fix**: Use market-share weighted means.

### 6. Ignoring Buffer Level

**The mistake**: Treating all RILA products as homogeneous.

**Why wrong**: 10% and 25% buffer products attract different buyers with different price sensitivity.

**Fix**: Include buffer level as control or stratify analysis.

### 7. Expecting MYGA-Like Elasticity

**The mistake**: Expecting 10x sales swings from small rate changes.

**Why wrong**: RILA has smoother response than MYGA due to complexity.

**Fix**: Expect moderate elasticity (10-30% changes, not 10x).

---

## Detection Checklist

Before committing code, verify:

- [ ] No factories/abstractions for single-use cases
- [ ] CLAUDE.md < 13KB
- [ ] No `return None` on error paths
- [ ] No silent exception handling
- [ ] No >20% improvement without investigation
- [ ] No lag-0 competitor features
- [ ] Buffer level controlled (RILA)
- [ ] Market-share weighting used (RILA)

---

## Related Documents

- `knowledge/practices/data_leakage_prevention.md` - Leakage detection
- `knowledge/practices/LEAKAGE_CHECKLIST.md` - Pre-deployment gates
- `knowledge/integration/LESSONS_LEARNED.md` - Critical traps
- `CLAUDE.md` - Project conventions

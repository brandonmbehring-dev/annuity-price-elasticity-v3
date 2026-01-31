# AI Collaboration Methodology

**Version**: 1.0.0 | **Last Updated**: 2026-01-31

This document describes how Claude Code was used effectively in developing the annuity price elasticity v3 system. It captures patterns, workflows, validation approaches, and lessons learned for future AI-assisted development.

---

## Executive Summary

The v3 repository was developed through extensive collaboration with Claude Code, an AI-powered coding assistant. This document captures:

- **Effective patterns** for AI-assisted development
- **Validation workflows** to ensure quality
- **Anti-patterns** to avoid
- **Metrics** demonstrating success

### Key Results

| Metric | Achievement |
|--------|-------------|
| Tests written | 6,126+ |
| Test coverage | 70%+ |
| Documentation files | 96+ |
| Leakage gates | 5 automated |
| Time to production | ~4 weeks |

---

## Section 1: Effective Prompting Patterns

### 1.1 Context-First Prompting

**Pattern**: Provide domain context before asking for code.

**Why It Works**:
- AI understands constraints before generating code
- Reduces back-and-forth iterations
- Produces domain-appropriate solutions

**Example**:
```
Context: In annuity price elasticity modeling, we estimate how
customer demand responds to rate changes. A critical constraint
is that competitor rates must be lagged (minimum 2 weeks) to
avoid simultaneity bias.

Task: Implement a feature engineering function that creates
competitor rate features with configurable lags.
```

### 1.2 Constraint-Driven Development

**Pattern**: State constraints explicitly upfront.

**Why It Works**:
- Prevents common mistakes
- Ensures code meets business requirements
- Reduces review cycles

**Example**:
```
Constraints:
1. Competitor features MUST use minimum lag of 2 (no lag-0)
2. Own rate coefficient MUST be positive in final model
3. All features MUST be computed using training data only
4. Function MUST raise explicit errors (no silent failures)

Now implement: validate_feature_lags(features: List[str]) -> bool
```

### 1.3 Test-First Requests

**Pattern**: Ask for tests alongside implementation.

**Why It Works**:
- Ensures testable code from start
- Documents expected behavior
- Catches edge cases early

**Example**:
```
Implement detect_lag0_features() with these tests:
1. test_detects_lag0_competitor: Returns True for "competitor_t0"
2. test_allows_lagged_competitor: Returns False for "competitor_t2"
3. test_allows_own_rate_t0: Returns False for "prudential_rate_t0"
4. test_handles_empty_list: Returns False for []
```

### 1.4 Anti-Pattern Documentation Requests

**Pattern**: Ask AI to document anti-patterns alongside solutions.

**Why It Works**:
- Prevents future mistakes
- Creates learning resources
- Encodes domain knowledge

**Example**:
```
Implement scaling in the pipeline. Also document:
1. The WRONG way (scaling before split)
2. Why it's wrong (test data leakage)
3. How to detect the mistake
4. The test that would catch it
```

---

## Section 2: Validation Workflows

### 2.1 Iterative Validation

**Workflow**:
1. Generate code with AI
2. Run tests immediately
3. Review economic constraints
4. Iterate on failures

**Implementation**:
```bash
# Quick validation cycle
make quick-check  # 30 seconds
# If pass, continue; if fail, iterate

# Full validation after feature complete
make test-all  # 7 minutes
```

### 2.2 Economic Constraint Verification

**Workflow**:
1. AI generates model code
2. Human reviews coefficient signs
3. AI generates validation tests
4. Tests become permanent gates

**Example Output**:
```
Model coefficients:
  own_rate: +0.085 (expected: positive) PASS
  competitor: -0.031 (expected: negative) PASS
  vix: -0.009 (expected: negative) PASS
```

### 2.3 Documentation Verification

**Workflow**:
1. AI generates code + docstrings
2. Human reviews accuracy
3. AI generates usage examples
4. Examples become integration tests

**Checklist**:
- [ ] Docstring matches implementation
- [ ] Parameters documented with types
- [ ] Raises documented for exceptions
- [ ] Examples are runnable

### 2.4 Code Review Protocol

**Human-in-the-Loop Steps**:
1. AI generates implementation
2. Human reviews for:
   - Economic correctness
   - Domain appropriateness
   - Security concerns
   - Performance implications
3. AI addresses feedback
4. Human approves for commit

---

## Section 3: Effective AI Collaboration Patterns

### 3.1 Incremental Development

**Pattern**: Build features incrementally, validating at each step.

**Anti-Pattern**: Ask for entire system at once.

**Why**:
- Easier to catch errors early
- AI maintains better context
- Humans can course-correct

**Example Session**:
```
Session 1: Define data adapter interface
Session 2: Implement S3 adapter
Session 3: Implement fixture adapter
Session 4: Add adapter tests
Session 5: Integration tests
```

### 3.2 Reference Implementation Requests

**Pattern**: Ask AI to study existing code before generating new code.

**Example**:
```
Before implementing the FIA methodology:
1. Read src/products/rila_methodology.py
2. Note the constraint enforcement pattern
3. Apply same pattern to FIA with different constraints
```

### 3.3 Explicit Error Handling

**Pattern**: Always request explicit error handling.

**Default AI Behavior**: May use silent fallbacks or return None.

**Better Prompt**:
```
Implement load_data() with these error behaviors:
- FileNotFoundError: Raise with helpful message
- Empty DataFrame: Raise ValueError with context
- Missing columns: Raise ValueError listing missing columns
- NEVER: Return None or use synthetic fallback data
```

### 3.4 Test Organization Requests

**Pattern**: Request organized test structure upfront.

**Example**:
```
Create tests for UnifiedNotebookInterface organized as:
tests/unit/notebooks/
├── test_interface_init.py
├── test_interface_load.py
├── test_interface_inference.py
└── test_interface_validation.py

Each file should have:
1. Fixture for test data
2. Unit tests for happy path
3. Unit tests for error cases
4. Integration test with mock adapter
```

---

## Section 4: Anti-Patterns to Avoid

### 4.1 Vague Requests

**Anti-Pattern**:
```
Make the code better.
```

**Better**:
```
Refactor calculate_elasticity() to:
1. Split into 3 functions (validate, compute, format)
2. Add type hints to all parameters
3. Add docstring with example
4. Ensure no function exceeds 30 lines
```

### 4.2 Missing Context

**Anti-Pattern**:
```
Write a function to aggregate competitor rates.
```

**Better**:
```
Context: RILA products use market-share weighted competitor
aggregation, while FIA products use simple top-N mean.

Write a function to aggregate competitor rates that:
1. Supports both strategies via dependency injection
2. Never uses lag-0 rates (minimum lag-2)
3. Returns NaN for periods with insufficient data
```

### 4.3 Skipping Tests

**Anti-Pattern**:
```
Just implement the feature, I'll write tests later.
```

**Better**:
```
Implement detect_lag0_features() with accompanying tests.
Run the tests before considering the task complete.
```

### 4.4 Ignoring Edge Cases

**Anti-Pattern**:
```
Implement data loading.
```

**Better**:
```
Implement data loading handling these edge cases:
1. File doesn't exist
2. File exists but is empty
3. File has wrong schema
4. File has NaN values in key columns
5. File has duplicate rows
```

---

## Section 5: Quality Metrics

### 5.1 Test Quality

| Metric | Target | Achieved |
|--------|--------|----------|
| Total tests | 5,000+ | 6,126 |
| Coverage (core) | 80% | 82.4% |
| Coverage (validation) | 90% | 91.2% |
| Property tests | 6+ files | 10 files |
| Anti-pattern tests | 4+ files | 5 files |

### 5.2 Documentation Quality

| Metric | Target | Achieved |
|--------|--------|----------|
| Docstring coverage | 85% | 87% |
| README complete | Yes | Yes |
| API reference | 80% | 70% (in progress) |
| Common pitfalls guide | Yes | Yes |
| Testing strategy doc | Yes | Yes |

### 5.3 Code Quality

| Metric | Target | Achieved |
|--------|--------|----------|
| Type hints | 100% public API | ~95% |
| Function length | <50 lines | ~90% compliant |
| Ruff linting | 0 errors | 0 errors |
| mypy strict | 0 errors | ~5 warnings |

---

## Section 6: Session Management

### 6.1 Session Tracking

Each development session was tracked with:
- Session number and date
- Goals for the session
- Accomplished tasks
- Next steps

**Location**: `sessions/` directory

### 6.2 Decision Documentation

Major decisions were documented with:
- Decision description
- Alternatives considered
- Rationale for choice
- Validation approach

**Location**: `.tracking/decisions.md`

### 6.3 Context Preservation

Between sessions, context was preserved via:
- `CURRENT_WORK.md` - Active work state
- `CLAUDE.md` - Project instructions
- Session files - Historical context

---

## Section 7: Lessons Learned

### 7.1 What Worked Well

1. **Test-first development**: AI-generated tests caught many issues early
2. **Explicit constraints**: Clear constraints produced better code
3. **Incremental development**: Small steps were easier to validate
4. **Anti-pattern documentation**: Prevented recurring mistakes
5. **Reference implementations**: AI learned from existing patterns

### 7.2 What Could Improve

1. **Initial API design**: More upfront design would reduce refactoring
2. **Performance profiling**: Should include earlier in workflow
3. **Integration testing**: Could be more comprehensive
4. **Documentation timing**: Write docs alongside code, not after

### 7.3 Unexpected Benefits

1. **Knowledge capture**: AI queries forced clear documentation
2. **Consistency**: AI maintained consistent patterns across codebase
3. **Test coverage**: AI naturally generated comprehensive tests
4. **Code review assistance**: AI identified issues during development

---

## Section 8: Recommended Workflow

### 8.1 Feature Development

```
1. Define feature requirements (human)
2. Document constraints (human + AI)
3. Design API (human + AI)
4. Write tests first (AI)
5. Implement feature (AI)
6. Run validation (automated)
7. Human review (human)
8. Document lessons (AI)
9. Commit (human approval)
```

### 8.2 Bug Investigation

```
1. Describe symptom (human)
2. Gather context (AI reads files)
3. Form hypothesis (AI + human)
4. Write reproducing test (AI)
5. Implement fix (AI)
6. Verify fix (automated)
7. Add regression test (AI)
8. Document root cause (AI)
```

### 8.3 Refactoring

```
1. Identify target (human)
2. Capture baseline metrics (automated)
3. Write characterization tests (AI)
4. Refactor incrementally (AI)
5. Verify mathematical equivalence (1e-12)
6. Update documentation (AI)
7. Final validation (automated)
```

---

## Section 9: Tools and Infrastructure

### 9.1 Development Environment

- **IDE**: VS Code with Claude Code extension
- **Testing**: pytest with hypothesis
- **Linting**: ruff, mypy
- **Coverage**: pytest-cov

### 9.2 CI/CD Integration

- **Pre-commit hooks**: ruff, basic tests
- **CI pipeline**: Full test suite + notebook validation
- **Deployment gates**: Leakage audit must pass

### 9.3 Documentation Generation

- **Docstrings**: Google/NumPy style (napoleon)
- **API docs**: Sphinx with autodoc
- **Guides**: Markdown with cross-references

---

## Section 10: Future Recommendations

### 10.1 Process Improvements

1. **Formal design documents** for major features
2. **Performance budgets** defined upfront
3. **API review checkpoints** before implementation
4. **Regular knowledge sync** sessions

### 10.2 Tooling Improvements

1. **Automated API breaking change detection**
2. **Continuous documentation generation**
3. **Performance regression testing**
4. **Visual diff for notebook outputs**

### 10.3 Knowledge Management

1. **Maintain decision log** for all major choices
2. **Document "why not" alongside "why"**
3. **Create runnable examples** for all patterns
4. **Regular anti-pattern gallery updates**

---

## Appendix A: Prompt Templates

### Feature Request Template
```
Context: [Domain context]

Constraints:
1. [Constraint 1]
2. [Constraint 2]

Task: [Specific task]

Tests to include:
1. [Test case 1]
2. [Test case 2]

Anti-patterns to avoid:
1. [Anti-pattern 1]
```

### Bug Investigation Template
```
Symptom: [What's happening]

Expected: [What should happen]

Context:
- File: [path]
- Function: [name]
- Line: [number if known]

Recent changes: [If relevant]

Suspected cause: [If known]
```

### Refactoring Template
```
Target: [File/function to refactor]

Goal: [What improvement]

Constraints:
1. Mathematical equivalence (1e-12)
2. No API changes
3. Tests must pass

Current metrics:
- Lines: [N]
- Complexity: [M]

Target metrics:
- Lines: [N']
- Complexity: [M']
```

---

## References

1. **CLAUDE.md**: Project-specific AI instructions
2. **CURRENT_WORK.md**: Active work tracking
3. **Testing Strategy**: `docs/development/TESTING_STRATEGY.md`
4. **Common Pitfalls**: `docs/guides/COMMON_PITFALLS.md`
5. **Decision Log**: `.tracking/decisions.md`

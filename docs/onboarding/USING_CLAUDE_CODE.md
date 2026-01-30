# Using Claude Code in This Repository

**AI-assisted development for the RILA price elasticity codebase.**

This repository is optimized for working with Claude Code, Anthropic's AI coding assistant. This guide explains how to use it effectively.

---

## What is Claude Code?

Claude Code is an AI assistant that:
- Reads and understands code
- Makes edits across multiple files
- Runs commands and tests
- Follows project-specific guidelines

**Key feature**: Claude Code reads `CLAUDE.md` files to understand project context, patterns, and constraints.

---

## How This Repository Is Configured

### The CLAUDE.md File

The root `CLAUDE.md` file contains:

```
 Repository Root
└── CLAUDE.md           ← Claude's primary guidance document
    ├── Project phase (DEVELOPMENT)
    ├── Core principles (fail fast, DI, DRY, zero regression)
    ├── Entry points (what code to use)
    ├── Code quality standards
    ├── Architecture overview
    └── Links to knowledge/ docs
```

**Claude automatically reads this** at the start of each session. It knows:
- Current project phase
- Supported products (6Y20B, 6Y10B, 10Y20B)
- Forbidden patterns (lag-0 competitors)
- Where to find documentation

### The Knowledge Base

Claude has access to the full `knowledge/` directory:

```
knowledge/
├── domain/           # Product economics, schemas
│   ├── RILA_ECONOMICS.md
│   └── GLOSSARY.md
├── analysis/         # Causal framework, features
│   └── CAUSAL_FRAMEWORK.md
├── practices/        # Leakage prevention, testing
│   └── LEAKAGE_CHECKLIST.md
└── integration/      # Cross-project learnings
    └── LESSONS_LEARNED.md
```

**Claude can search and reference these docs** when answering questions or making changes.

---

## Starting a Claude Code Session

### First Time Setup

1. **Install Claude Code** (if not already installed):
   ```bash
   # Via npm
   npm install -g @anthropic-ai/claude-code

   # Or via Homebrew (macOS)
   brew install claude-code
   ```

2. **Navigate to the repository**:
   ```bash
   cd annuity-price-elasticity-v3
   ```

3. **Start Claude Code**:
   ```bash
   claude
   ```

### Session Start Checklist

When you start a session, Claude will:
1. Read `CLAUDE.md`
2. Understand the project context
3. Be ready to help with code

**Good first prompt**:
```
What products are currently supported, and where is the main entry point?
```

---

## Effective Prompting Patterns

### Pattern 1: Be Specific About Files

**Less effective**:
```
"Fix the interface"
```

**More effective**:
```
"In src/notebooks/interface.py, the run_inference method
raises an error when features is None. Add a default to
auto-detect features from the data."
```

### Pattern 2: Reference Documentation

**Less effective**:
```
"Make sure the coefficients are correct"
```

**More effective**:
```
"Per CAUSAL_FRAMEWORK.md, validate that run_inference rejects
any features matching lag-0 competitor patterns (_t0, _current)."
```

### Pattern 3: Ask for Explanation

Claude can explain code and decisions:

```
"Explain why the UnifiedNotebookInterface uses dependency
injection for the data adapter. What problem does this solve?"
```

```
"Walk me through the _is_competitor_lag_zero method.
What patterns does it detect?"
```

### Pattern 4: Request Tests

```
"Add a unit test for the validate_coefficients method
in src/notebooks/interface.py. Include cases for:
1. All coefficients pass
2. Own rate coefficient is negative (should fail)
3. Competitor coefficient is positive (should fail)"
```

### Pattern 5: Multi-Step Tasks

Break complex tasks into steps:

```
"I want to add support for the 6Y15B product. Let's do this:
1. First, show me the current product config structure
2. Then add the new product config
3. Create fixture data for testing
4. Run the test suite to validate"
```

---

## What Claude Knows About This Codebase

### Domain Knowledge

Claude understands:
- **RILA products**: Cap rates, buffers, yields vs prices
- **Causal framework**: Why lag-0 competitors are forbidden
- **Economic constraints**: Positive own rate, negative competitor coefficients

### Code Patterns

Claude knows the preferred patterns:
- **Dependency injection** for adapters
- **Fail-fast** error handling (no silent failures)
- **Type hints** on all functions
- **20-50 line functions** maximum

### What Claude Won't Do

Per `CLAUDE.md` guidelines, Claude will:
- **Refuse** to use lag-0 competitor features
- **Warn** if creating synthetic data fallbacks
- **Stop** if proposing changes that break equivalence tests

---

## Common Tasks with Claude

### Task: Run Inference

```
"Create a fixture-based interface for 6Y20B and run inference.
Show me the coefficients and validate the economic signs."
```

### Task: Debug an Error

```
"I'm getting 'Lag-0 competitor features detected' when running
inference. Here's my feature list: [competitor_mid_t0, ...].
Why is this happening and how do I fix it?"
```

### Task: Add a Feature

```
"I want to add VIX (volatility index) as a control variable.
Per CAUSAL_FRAMEWORK.md, this is an open investigation item.
Where should I add this feature, and what lag should I use?"
```

### Task: Review Code

```
"Review src/notebooks/interface.py for:
1. Compliance with CLAUDE.md standards
2. Potential data leakage issues
3. Missing error handling"
```

### Task: Write Documentation

```
"Write a docstring for the _prepare_analysis_data method
in interface.py. Follow NumPy docstring format."
```

---

## Claude Code Slash Commands

Claude Code supports special commands:

| Command | Purpose |
|---------|---------|
| `/help` | Show available commands |
| `/clear` | Clear conversation history |
| `/compact` | Summarize conversation to save context |

---

## Troubleshooting

### Claude Seems Confused About the Project

**Reset context**:
```
/clear
```

Then restart with:
```
"Please re-read CLAUDE.md and summarize the current project phase
and supported products."
```

### Claude Made a Change I Don't Want

**Undo with git**:
```bash
git checkout -- path/to/file
```

Or ask Claude:
```
"Revert the last change to interface.py"
```

### Claude Doesn't Know About a File

**Point it to the file**:
```
"Read src/core/exceptions.py and summarize the exception hierarchy."
```

### Claude's Test Doesn't Work

**Be explicit about the test framework**:
```
"Write a pytest test (not unittest) for the validate_coefficients
method. Use fixtures from tests/fixtures/rila/."
```

---

## Best Practices

### 1. Review Before Committing

Always review Claude's changes before committing:
```bash
git diff
```

### 2. Run Tests After Changes

```bash
make quick-check  # Fast validation
make test         # Full suite
```

### 3. Document the "Why"

Ask Claude to explain its reasoning:
```
"Why did you implement it this way instead of [alternative]?"
```

### 4. Iterate

Don't expect perfect results on the first try:
```
"That's close, but the error message should include the
feature name. Can you update it?"
```

### 5. Use Knowledge Base References

Point Claude to specific docs:
```
"Per LESSONS_LEARNED.md Trap #2, update the feature validation
to reject lag-0 competitor features."
```

---

## Example Session

Here's what a productive session might look like:

```
You: What's the current state of the interface module?

Claude: [reads interface.py, summarizes key methods]

You: I need to add a method to export results to CSV.
     Follow the existing export_results pattern.

Claude: [proposes export_to_csv method]

You: Good, but add validation that the results dict has
     the required 'coefficients' key.

Claude: [updates with validation]

You: Now add a test for this method.

Claude: [creates test file]

You: Run the tests.

Claude: [runs pytest, shows results]

You: Looks good. Create a commit with the message
     "feat: Add CSV export for inference results"

Claude: [creates commit]
```

---

## Further Reading

- **Claude Code Documentation**: https://docs.anthropic.com/claude-code
- **This Repository's CLAUDE.md**: Root of this repo
- **Knowledge Base**: `knowledge/INDEX.md`

---

## Summary

1. **Claude reads CLAUDE.md** for project context automatically
2. **Be specific** in prompts—reference files, methods, and docs
3. **Ask for explanations** to understand reasoning
4. **Review and test** before committing
5. **Iterate** if the first result isn't perfect

Claude Code is a powerful tool for this codebase. Use it to accelerate development while maintaining quality.

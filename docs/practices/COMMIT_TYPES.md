# Commit Types Reference

**Standard commit type prefixes for this repository.**

Based on [lever_of_archimedes/patterns/git.md](~/Claude/lever_of_archimedes/patterns/git.md).

---

## Commit Types

| Type | Purpose | Example |
|------|---------|---------|
| `feat` | New feature or capability | `feat: Add multi-product support` |
| `fix` | Bug fix | `fix: Handle None in session stats` |
| `refactor` | Code restructuring (no behavior change) | `refactor: Extract CardFactory module` |
| `test` | Adding or modifying tests | `test: Add comprehensive CardFactory tests` |
| `docs` | Documentation only | `docs: Add CFA study roadmap` |
| `migrate` | Hub-and-spoke migrations | `migrate: Hub-and-spoke architecture` |
| `plan` | Plan document creation | `plan: Add implementation plan` |

---

## Commit Message Format

```
[type]: [description]

[Optional body with 2-3 sentences of context]

[Optional bullets for key changes]
- Change 1
- Change 2

[Optional "Next:" for session-based commits]
Next: What comes next

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## Examples

### Feature
```
feat: Add Product Registry for multi-product support

Centralized product configuration with type-safe access.
Replaces scattered product definitions across modules.

- Created src/core/product_registry.py
- Migrated 6Y20B, 6Y10B, 10Y20B definitions
- Added registry validation on startup

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Fix
```
fix: Address pandas 3.0 deprecation warnings

Updated inplace operations to assignment pattern.
Boolean dtype handling for numpy 2.0 compatibility.

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Documentation
```
docs: Add master navigation and roadmap

Tier 2 documentation improvements:
- docs/INDEX.md: Comprehensive hybrid navigation (221 lines)
- ROADMAP.md: Project goals and milestone tracking

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## Rules

1. **Always include `Co-Authored-By`** when AI-assisted
2. **Use present tense** in subject line ("Add feature" not "Added feature")
3. **Keep subject under 72 characters**
4. **Body explains WHY**, not just what
5. **No emojis** (except attribution line if organization allows)

---

## Related

- [git.md](git.md) - Full git workflow patterns
- [lever_of_archimedes patterns](~/Claude/lever_of_archimedes/patterns/git.md) - Canonical source

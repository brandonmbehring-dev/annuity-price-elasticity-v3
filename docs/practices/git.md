# Git Commit Pattern

**Purpose**: Consistent, informative commit messages with AI attribution.

**Integration**: Works with session-based workflow and burst methodology.

---

## Commit Message Format

```
[type]: [description]

[Optional body with 2-3 sentences of context]

[Optional bullets for key changes]
- [Change 1]
- [Change 2]

[Optional "Next:" for session-based commits]
Next: [What comes next]

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## Commit Types

**feat**: New feature or capability
```
feat: Add Ethics card generation for CFA Level III

Generated 20 scenario-based cards covering Standards I-VII.
Cards export to Anki .apkg format for spaced repetition.

- Created JSON schema for CFA cards
- Implemented json_to_anki.py converter
- Added ethics_standards_batch1.json (20 cards)

Next: Generate Asset Manager Code cards
```

**fix**: Bug fix
```
fix: Handle None value in session stats average_accuracy

Session start crashed when stats.average_accuracy was None.
Added null check before isnan() call.
```

**refactor**: Code restructuring (no functionality change)
```
refactor: Extract card generation into CardFactory module

Moved card creation logic from scripts into reusable module.
Maintains same functionality, improves testability.
```

**test**: Adding or modifying tests
```
test: Add comprehensive tests for CardFactory

- Unit tests for all card types
- Integration tests for Anki export
- Edge case tests for invalid input

Coverage: 0% → 85%
```

**docs**: Documentation only
```
docs: Add CFA study roadmap with 326 LOS breakdown

Organized by topic area and reading number.
Includes time estimates and priority levels.
```

**migrate**: Hub-and-spoke architecture migrations
```
migrate: Hub-and-spoke architecture (archimedes_lever/shared)

- Updated CLAUDE.md to reference shared patterns
- Reduced context size by 60% (45KB → 18KB)
- Maintains all functionality (validated)

See migration_baseline_2025-11-24.md for details
```

**plan**: Plan document creation (Large Task Protocol)
```
plan: Add CFA review system implementation plan

Created docs/plans/active/cfa_review_2025-11-23_15-30.md.
Outlines 6-layer validation and session workflow.
```

---

## Session-Based Commits

**Pattern**: One commit per session (unless multi-day)

**Format**:
```
Session NNN: [Session description]

[2-3 sentences summarizing work]

- [Key accomplishment 1]
- [Key accomplishment 2]

Next: [Next session plan]

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Example**:
```
Session 001: Ethics Standards I-VII (30 min)

Generated 30 Ethics cards for CFA Level III review.
Includes 20 standard cards and 10 complex multi-standard scenarios.
Validated JSON to Anki export workflow.

- Created ethics_standards_batch1.json (20 cards)
- Created ethics_complex_scenarios.json (10 cards)
- Implemented json_to_anki.py converter
- Exported 2 .apkg decks ready for Anki import

Next: Session 002 - Asset Manager Code and GIPS cards

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## When to Commit

**After `/good-enough` check**:
- Code works in production
- Better than before
- Meets immediate need
- Auto-suggest commit

**After burst completion**:
- Timer ended, work complete
- Tests pass
- Ready to ship

**Never during hyperfocus**:
- Breaks flow
- Wait for burst end

---

## Auto-Suggest Behavior

**AI assistant protocol**:

1. **After `/good-enough`**: Suggest creating commit
2. **Show proposed message**: Let user review
3. **Wait for approval**: Never commit without consent
4. **Execute**: Run git add + commit with approved message

**User can**:
- Approve as-is
- Modify message
- Defer commit
- Handle manually

---

## Git Safety Protocol

**NEVER** (without explicit permission):
- Update git config
- Run destructive commands (push --force, hard reset)
- Skip hooks (--no-verify, --no-gpg-sign)
- Force push to main/master
- Amend commits (unless fixing pre-commit hook issues)

**Always check authorship before amending**:
```bash
git log -1 --format='%an %ae'
# Only amend if YOUR commit, not others'
```

---

## Integration with Workflows

### Brandon Burst
```
1. /reality-check -> Verify need
2. /one-thing -> Identify task
3. /burst 25 -> Execute
4. /good-enough -> Verify
5. Auto-suggest commit -> Approve
6. Commit created -> Move to next burst
```

### Session Workflow
```
1. Start session -> Update CURRENT_WORK.md
2. Work in bursts -> Multiple focused periods
3. End session -> Update SESSION_*.md
4. Session commit -> One commit for entire session
```

### Large Task Protocol
```
1. Create plan -> Commit plan document
2. Execute phases -> Commit after each phase
3. Complete -> Final commit references plan
```

---

## Commit Message Quality

**Good**:
- Clear type prefix
- Describes WHAT changed
- Body explains WHY (if not obvious)
- Bullets for multiple changes
- "Next:" for context continuity

**Bad**:
- "WIP" or "temp" (use feature branches)
- "Fixed stuff" (too vague)
- No context (what/why unclear)
- Emoji spam (one robot emoji only in attribution line)

---

**Next**: Use this pattern for ALL commits. Consistency enables automation and tracking.

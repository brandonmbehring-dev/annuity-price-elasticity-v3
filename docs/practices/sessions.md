# Session-Based Workflow Pattern

**Purpose**: Track work in focused bursts with clear history and resumability.
**Last Updated**: 2026-01-20
**Copied from**: ~/Claude/lever_of_archimedes/patterns/sessions.md

**Works with**: Brandon Burst™ methodology (25-min sessions)

---

## Core Files

### 1. CURRENT_WORK.md

**Purpose**: 30-second context switching and resume

**Location**: Project root

**Format**:
```markdown
## Right Now
[One sentence about current work]

## Why
[One sentence why it matters]

## Next Step
[One action, achievable in 25-min burst]

## Context When I Return
[Breadcrumbs for future you]
```

**Update When**:
- Starting work session
- Switching contexts
- Ending work session

**Read When**:
- Resuming after break
- Starting new day

### 2. SESSION_*.md

**Purpose**: Document each focused work session

**Location**: `sessions/SESSION_NNN_description_YYYY-MM-DD_HH-MM.md`

**Format**:
```markdown
# Session NNN: [Description]

**Date**: YYYY-MM-DD HH:MM
**Duration**: XX minutes (planned/actual)
**Type**: [Implementation, Refactoring, Bug Fix, etc.]

**Status**:  IN PROGRESS | [DONE] COMPLETE

---

## Objectives

**Primary**:
- [Main goal]

**Secondary**:
- [Optional goals]

---

## Work Completed

- [x] Task 1
- [x] Task 2
- [ ] Task 3 (deferred)

---

## Next Session Plan

Continue with:
1. [Next task 1]
2. [Next task 2]

---

## Files Created/Modified

**Created**:
- [file1]
- [file2]

**Modified**:
- [file3]
```

---

## Workflow

### Starting a Session

1. Create `sessions/SESSION_NNN_description_$(date +%Y-%m-%d_%H-%M).md`
2. Fill template
3. Update CURRENT_WORK.md with "Right Now"

### During Session

1. **Focus** on session objectives (ONE thing at a time)
2. **Update** SESSION_*.md as you complete tasks
3. **Document** files created/modified
4. **No scope creep** - stay within session objectives

### Ending Session

1. Update SESSION_*.md status to [DONE] COMPLETE
2. Add completion timestamp
3. Update CURRENT_WORK.md with session results
4. Git commit with session summary

---

## Session Types

| Type | Description | Duration |
|------|-------------|----------|
| Implementation | Write new features | 25-120 min |
| Refactoring | Improve code structure | 25-60 min |
| Bug Fix | Reproduce, fix, test | 25-60 min |
| Research | Explore, document | 25-60 min |
| Documentation | Write/update docs | 25-30 min |

---

## Integration with Git

**Commit Pattern**:
```
Session NNN: [Session description]

[2-3 sentences summarizing work]

- [Key accomplishment 1]
- [Key accomplishment 2]

Next: [Next session plan]

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**One commit per session** (unless session spans multiple work periods).

---

## Benefits

1. **Clear history** - Git log shows session-by-session progress
2. **Easy resume** - CURRENT_WORK.md + last SESSION_*.md provide full context
3. **ADHD-friendly** - External memory, no mental overhead
4. **Accountability** - Documents what was done and why

---

**Next**: Use this pattern for burst-based workflow.

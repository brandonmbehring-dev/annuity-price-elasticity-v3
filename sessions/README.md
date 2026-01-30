# Session Logs

**Purpose**: Track work sessions for continuity and historical reference.

**Pattern**: From [lever_of_archimedes/patterns/sessions.md](~/Claude/lever_of_archimedes/patterns/sessions.md)

---

## Structure

```
sessions/
├── active/           # In-progress sessions
│   └── SESSION_NNN_description_YYYY-MM-DD.md
├── archive/          # Completed sessions
│   └── SESSION_NNN_description_YYYY-MM-DD.md
└── README.md         # This file
```

## Session File Template

```markdown
# Session NNN: [Description]

**Date**: YYYY-MM-DD
**Duration**: XX minutes (planned/actual)
**Status**: ⏳ IN PROGRESS | ✅ COMPLETE

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

## Files Modified

**Created**:
- [file1]

**Modified**:
- [file2]

---

## Decisions Made

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| [Decision] | [Why] | [What else] |

---

## Next Session Plan

1. [Next task 1]
2. [Next task 2]
```

## Workflow

### Starting a Session

1. Create `sessions/active/SESSION_NNN_description_YYYY-MM-DD.md`
2. Fill in objectives
3. Update `CURRENT_WORK.md` with "Right Now"

### During Session

1. Focus on session objectives
2. Update SESSION file as you complete tasks
3. Document files created/modified
4. Record decisions made

### Ending Session

1. Update SESSION file status to ✅ COMPLETE
2. Move from `active/` to `archive/`
3. Update `CURRENT_WORK.md` with results
4. Commit with session summary

## Benefits

- **Clear history**: Git log shows session-by-session progress
- **Easy resume**: CURRENT_WORK.md + last SESSION provide full context
- **Decision tracking**: Rationale preserved for future reference

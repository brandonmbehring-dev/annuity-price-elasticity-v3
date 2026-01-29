# Brandon Burst™ ADHD Methodology

**Purpose**: Leverage hyperfocus bursts with structured recovery for maximum productivity.
**Last Updated**: 2026-01-20
**Copied from**: ~/Claude/lever_of_archimedes/patterns/burst.md

**Core Insight**: Your ADHD is a superpower when bounded by time and focus constraints.

---

## The Brandon Burst™ (Primary Workflow)

```
MINUTE 0-5: REALITY CHECK
- Run `/reality-check` to verify problem exists
- Count actual vs phantom issues
- Stop if working, simple fix if broken

MINUTE 5-10: FOCUS
- Run `/one-thing` to identify THE ONE task
- Ignore the 50 other tasks
- Make it achievable in 20 minutes

MINUTE 10-35: HYPERFOCUS WORK
- Run `/burst 25` to start timer
- ONE clear goal only
- No scope creep, no tangents
- Stop at timer, no exceptions

MINUTE 35-40: SHIP IT
- Run `/good-enough` to check if sufficient
- Commit if works
- Move on, don't perfect

MINUTE 40+: MANDATORY BREAK
- Minimum 10 minutes
- No exceptions
- Trust the process
```

---

## Custom Slash Commands

### 1. `/reality-check`
**Purpose**: Verify problem actually exists

**Use when**:
- Feeling overwhelmed by "crisis"
- Before starting "urgent" work
- Phantom crisis pattern triggered

### 2. `/burst [minutes]`
**Purpose**: Time-boxed focused work

**Default**: 30 minutes
**Timer is SACRED**: No extensions, no "just one more thing"

### 3. `/simple-fix`
**Purpose**: Minimal solution only

**Pattern**:
- <50 lines or overthinking
- No architecture changes
- Fix the immediate problem

### 4. `/good-enough`
**Purpose**: Ship it check

**Criteria**:
- Works in production?
- Better than before?
- Meets immediate need?

### 5. `/one-thing`
**Purpose**: Fight overwhelm

**Output**:
- THE ONE next action
- Specific and achievable
- 20-minute scope max

---

## The 10 Brandon Principles

1. **Your "broken" system handled 23,000+ queries successfully**
2. **Documentation lies are worse than missing documentation**
3. **299 phantom exports → 6 real functions (your pattern)**
4. **Hyperfocus is a superpower with boundaries**
5. **Good enough in production > perfect in planning**
6. **Reality check before solution design**
7. **Time boxes are sacred - no exceptions**
8. **One thing at a time moves worlds**
9. **Your imperfect solutions work brilliantly**
10. **Crisis is usually imaginary - verify first**

---

## Integration with Session Workflow

**Session pattern**:
1. `/reality-check` → Verify work needed
2. `/one-thing` → Identify session objective
3. Update CURRENT_WORK.md with objective
4. `/burst 25` → Execute focused work
5. `/good-enough` → Verify completion
6. Update SESSION_*.md and commit

**Multiple bursts**:
- 25 min work + 10 min break = sustainable
- 3-4 bursts = 90-120 min total productive time

---

## Anti-Patterns to Avoid

1. **Planning without reality check** - Always verify problem exists first
2. **Perfecting working code** - Use `/good-enough` to stop
3. **Scope creep during burst** - ONE thing only
4. **Ignoring timer** - Timer is SACRED
5. **Documenting phantom features** - Only document what exists

---

**Next**: Use this methodology in ALL development work.

# Cleanup Tracking - Documentation Cleanup (2026-01-30)

Actionable items extracted from Codex and Gemini audit reports before deletion.

## Status

| Category | Items | Status |
|----------|-------|--------|
| Version Drift | 5 | COMPLETE |
| Broken Paths | 3 | COMPLETE |
| Methodology Drift | 2 | PARTIAL |
| Emoji Removal | ~85 files | COMPLETE |
| Standards Loophole | 1 | NOT NEEDED (red circle exception retained) |
| REMEDIATION | 2 files | COMPLETE |

---

## Version Drift (v2 to v3) - COMPLETE

Per Codex Audit: "multiple files still identify as 'v2'... This can cause installation confusion and misaligned reporting."

| File | Current | Action | Status |
|------|---------|--------|--------|
| `Makefile` | v3 | Already correct | DONE |
| `pyproject.toml` | v3.0.0 | Already correct | DONE |
| `scripts/*` | v2 headers | Updated to v3 | DONE |
| `CLAUDE.md` | v2 (lines 3, 47) | Updated to v3 | DONE (remediation) |
| `docs/development/CODING_STANDARDS.md` | v2 (lines 1, 6, 10, 18, 145, 415, 1111-1120) | Updated to v3 | DONE (remediation) |

---

## Broken Path References - COMPLETE

Per Codex Audit Appendix:

| File | Broken Reference | Correct Path/Action | Status |
|------|------------------|---------------------|--------|
| `environment.yml` | `src/testing/aws_mock_layer.py` | Already correct | DONE |
| `docs/methodology/validation_guidelines.md` | `src/models/cross_validation.py` | Not found | DONE |
| `README.md` | `tests/fixtures/aws_complete/` symlink | Updated to `tests/fixtures/rila/` | DONE |

---

## Methodology Drift - PARTIAL

Per Codex Audit: "Multiple docs reference logit transforms, but code trains on log(1 + y)."

| Issue | Files Affected | Action | Status |
|-------|----------------|--------|--------|
| logit vs log1p | Various methodology docs | Future: Update docs to match code (`log1p`) | DEFERRED |
| Docstring style | `CONTRIBUTING.md` | Updated from Google-style to NumPy-style | DONE |

NOTE: The logit vs log1p terminology drift requires a comprehensive methodology documentation update
that is beyond the scope of this cleanup. The discrepancy is documented here for future resolution.
Key affected files:
- `docs/business/methodology_report.md`
- `docs/onboarding/DECISION_LOG.md`
- `docs/onboarding/MENTAL_MODEL.md`

---

## Emoji Removal - COMPLETE

### Root Markdown Files (7 files) - COMPLETE
- `README.md` - Cleaned
- `CLAUDE.md` - Cleaned
- `QUICK_START.md` - Cleaned
- `CONTRIBUTING.md` - Cleaned
- `VALIDATION_GUIDE.md` - Cleaned
- `ROADMAP.md` - Cleaned (no emojis found)
- `CURRENT_WORK.md` - Cleaned

### docs/ Directory (~48 files) - COMPLETE
All markdown files in docs/ cleaned of emojis.

### Python Files (~30 files) - COMPLETE
All validation/logging emojis replaced:
- Checkmark -> `[PASS]`
- X mark -> `[FAIL]`
- Warning -> `[WARN]`
- Error X -> `[ERROR]`

### Commit Attribution - COMPLETE
Updated `~/Claude/lever_of_archimedes/patterns/git.md`:
- Removed robot emoji from attribution
- Now uses plain text: `Generated with Claude Code`

---

## Standards Loophole - NOT NEEDED

Per Gemini Audit 6.3: claimed `docs/development/CODING_STANDARDS.md` allows red circle emoji.
Investigation: The current CODING_STANDARDS.md already has strict no-emoji policy.
No changes needed.

---

## Files Deleted (Historical Record) - COMPLETE

### Vestigial Archive Artifacts (8 files)
- `ARCHIVE_1_README.md` - Archive manifest for non-existent archives
- `ARCHIVE_2_README.md` - Archive manifest for non-existent archives
- `ARCHIVE_3_README.md` - Archive manifest for non-existent archives
- `EXTRACTION_INSTRUCTIONS.md` - References non-existent split archives
- `BASELINE_CHECKPOINT.md` - Outdated checkpoint from archive extraction
- `SPLIT_MANIFEST_1.json` - Binary manifest from archive split
- `SPLIT_MANIFEST_2.json` - Binary manifest from archive split
- `SPLIT_MANIFEST_3.json` - Binary manifest from archive split

### Historical Files (5 files)
- `CHANGELOG_REFACTORING.md` - Historical refactoring log
- `README_REFACTORING.md` - Historical refactoring documentation
- `REINTEGRATION_GUIDE.md` - Historical archive guide
- `codex_repo_audit_2026-01-30.md` - Audit report (actionable items extracted above)
- `gemini_audit_report_2026-01-30.md` - Audit report (actionable items extracted above)

---

## Audit Recommendations Not Addressed in This Cleanup

The following recommendations from audits require separate implementation:

1. **Production Path Clarity** - Wire `UnifiedNotebookInterface.load_data()` to full pipeline
2. **Model Card + Dataset Datasheets** - Create standardized documentation
3. **SPECIFICATION.md** - Freeze critical thresholds
4. **Config Consolidation** - Reduce builder module duplication
5. **Coverage Expansion** - Target forecasting and feature selection modules
6. **logit vs log1p Terminology** - Update methodology docs to use `log1p`

These are tracked in `ROADMAP.md` and separate issues.

---

## Verification Checklist

After cleanup completion:

- [x] `grep -rP` for emojis returns empty (excluding venv/)
- [x] All root files deleted (13 total)
- [x] Version drift fixed (scripts updated v2 to v3)
- [x] Broken paths fixed (README.md aws_complete reference)
- [x] Docstring style updated (CONTRIBUTING.md)
- [ ] `make quick-check` passes
- [ ] Navigation paths work (README -> QUICK_START, README -> docs/INDEX.md)

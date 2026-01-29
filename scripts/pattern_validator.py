#!/usr/bin/env python3
"""
Pattern Validator for Annuity Price Elasticity v2.

Validates codebase patterns to prevent anti-pattern regression:
1. Import hygiene - No triple-fallback imports, proper dependency handling
2. Constraint compliance - Proper sign constraints for economic variables
3. No competing implementations - Single canonical implementation per pattern
4. Interface usage - No direct engine access, must use interfaces/pipelines

Usage:
    python scripts/pattern_validator.py --path src/
    python scripts/pattern_validator.py --path src/ --fix-suggestions
    python scripts/pattern_validator.py --check-constraints

Exit Codes:
    0 - All validations passed
    1 - Validation errors found
    2 - Configuration error
"""

import argparse
import ast
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Dict, Tuple


# =============================================================================
# CONFIGURATION
# =============================================================================

# Allowed try/except ImportError patterns (for optional dependencies)
ALLOWED_OPTIONAL_IMPORTS = frozenset({
    "mlflow",
    "graphviz",
    "boto3",
    "awscli",
    "ipywidgets",
    "plotly",
})

# Engine modules that should not be imported directly (use interfaces instead)
RESTRICTED_DIRECT_IMPORTS = {
    "src.features.selection.engines.aic_engine": "Use src.features.selection instead",
    "src.features.selection.engines.constraints_engine": "Use src.features.selection instead",
    "src.features.selection.engines.bootstrap_engine": "Use src.features.selection instead",
    "src.features.selection.engines.ridge_cv_engine": "Use src.features.selection instead",
}

# Patterns that indicate lag-0 competitor usage (FORBIDDEN)
LAG0_PATTERNS = [
    r"C_lag_?0\b",
    r"competitor.*lag.*0",
    r"lag_0.*competitor",
    r"C_t\b(?!_)",  # C_t but not C_t_1, C_t_minus_1
]

# Required constraint patterns
REQUIRED_CONSTRAINTS = {
    "prudential": {"expected_sign": "positive", "rationale": "Own rate attracts customers"},
    "competitor": {"expected_sign": "negative", "rationale": "Substitution effect"},
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ValidationIssue:
    """Single validation issue found in codebase."""

    file_path: str
    line_number: int
    issue_type: str
    message: str
    severity: str  # ERROR, WARNING
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        severity_marker = "ERROR" if self.severity == "ERROR" else "WARN"
        location = f"{self.file_path}:{self.line_number}"
        return f"[{severity_marker}] {location}: {self.issue_type} - {self.message}"


@dataclass
class ValidationReport:
    """Complete validation report for the codebase."""

    issues: List[ValidationIssue] = field(default_factory=list)
    files_scanned: int = 0
    patterns_checked: int = 0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "ERROR")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "WARNING")

    @property
    def passed(self) -> bool:
        return self.error_count == 0

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "Pattern Validation Report",
            "=" * 60,
            f"Files scanned: {self.files_scanned}",
            f"Patterns checked: {self.patterns_checked}",
            f"Errors: {self.error_count}",
            f"Warnings: {self.warning_count}",
            "-" * 60,
        ]

        if self.issues:
            for issue in sorted(self.issues, key=lambda x: (x.severity != "ERROR", x.file_path)):
                lines.append(str(issue))
                if issue.suggestion:
                    lines.append(f"    Suggestion: {issue.suggestion}")
        else:
            lines.append("All patterns validated successfully!")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# IMPORT HYGIENE CHECKER
# =============================================================================

class ImportHygieneChecker:
    """Check for problematic import patterns."""

    def __init__(self, allowed_optionals: Set[str] = ALLOWED_OPTIONAL_IMPORTS):
        self.allowed_optionals = allowed_optionals

    def check_file(self, file_path: Path) -> List[ValidationIssue]:
        """Check a single file for import hygiene issues."""
        issues = []

        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError) as e:
            return [ValidationIssue(
                file_path=str(file_path),
                line_number=0,
                issue_type="FILE_READ_ERROR",
                message=str(e),
                severity="WARNING",
            )]

        lines = content.split("\n")

        # Check for triple-fallback imports (more than 2 try/except for same import)
        issues.extend(self._check_triple_fallback(file_path, content, lines))

        # Check for direct engine imports
        issues.extend(self._check_restricted_imports(file_path, lines))

        return issues

    def _check_triple_fallback(
        self, file_path: Path, content: str, lines: List[str]
    ) -> List[ValidationIssue]:
        """Detect triple-fallback import patterns."""
        issues = []

        # Count try/except ImportError blocks
        try_import_pattern = re.compile(
            r"try:\s*\n\s*(?:from\s+\S+\s+)?import\s+(\S+).*?\nexcept\s+ImportError",
            re.MULTILINE
        )

        import_fallback_counts = defaultdict(list)
        for match in try_import_pattern.finditer(content):
            module = match.group(1).split(".")[0]
            line_num = content[:match.start()].count("\n") + 1
            import_fallback_counts[module].append(line_num)

        for module, line_nums in import_fallback_counts.items():
            if len(line_nums) > 2:
                issues.append(ValidationIssue(
                    file_path=str(file_path),
                    line_number=line_nums[0],
                    issue_type="TRIPLE_FALLBACK_IMPORT",
                    message=f"Module '{module}' has {len(line_nums)} fallback imports",
                    severity="ERROR",
                    suggestion="Consolidate import fallbacks into a single location",
                ))
            elif module not in self.allowed_optionals and len(line_nums) > 1:
                issues.append(ValidationIssue(
                    file_path=str(file_path),
                    line_number=line_nums[0],
                    issue_type="UNEXPECTED_FALLBACK_IMPORT",
                    message=f"Module '{module}' uses fallback import but is not in allowed list",
                    severity="WARNING",
                    suggestion=f"Add '{module}' to ALLOWED_OPTIONAL_IMPORTS if truly optional",
                ))

        return issues

    def _check_restricted_imports(
        self, file_path: Path, lines: List[str]
    ) -> List[ValidationIssue]:
        """Check for direct imports of restricted modules."""
        issues = []

        # Skip if we're in an engine file, __init__.py, or internal orchestration
        skip_patterns = ["engines", "__init__.py", "pipeline_orchestrator", "interface_environment", "stability_analysis"]
        if any(pattern in str(file_path) for pattern in skip_patterns):
            return issues

        for line_num, line in enumerate(lines, 1):
            for restricted_module, suggestion in RESTRICTED_DIRECT_IMPORTS.items():
                if restricted_module in line and "import" in line:
                    issues.append(ValidationIssue(
                        file_path=str(file_path),
                        line_number=line_num,
                        issue_type="DIRECT_ENGINE_IMPORT",
                        message=f"Direct import of internal engine module: {restricted_module}",
                        severity="WARNING",
                        suggestion=suggestion,
                    ))

        return issues


# =============================================================================
# LAG-0 CONSTRAINT CHECKER
# =============================================================================

class Lag0ConstraintChecker:
    """Check for forbidden lag-0 competitor feature usage."""

    def __init__(self, patterns: List[str] = LAG0_PATTERNS):
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

    def check_file(self, file_path: Path) -> List[ValidationIssue]:
        """Check a single file for lag-0 competitor usage."""
        issues = []

        # Skip test files, documentation, and validation tooling
        skip_patterns = ["test", ".md", "validation", "pattern_validator", "leakage_gates"]
        if any(pattern in str(file_path).lower() for pattern in skip_patterns):
            return issues

        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            return issues

        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue

            for pattern in self.compiled_patterns:
                if pattern.search(line):
                    issues.append(ValidationIssue(
                        file_path=str(file_path),
                        line_number=line_num,
                        issue_type="LAG0_COMPETITOR_USAGE",
                        message="Potential lag-0 competitor feature detected (violates causal identification)",
                        severity="ERROR",
                        suggestion="Use lagged competitor rates (C_lag_1+) instead",
                    ))
                    break  # One issue per line

        return issues


# =============================================================================
# CONSTRAINT DEFINITION CHECKER
# =============================================================================

class ConstraintDefinitionChecker:
    """Verify constraint definitions match required patterns."""

    def __init__(self, required_constraints: Dict = REQUIRED_CONSTRAINTS):
        self.required_constraints = required_constraints

    def check_file(self, file_path: Path) -> List[ValidationIssue]:
        """Check constraint definitions in a file."""
        issues = []

        # Only check files that might contain constraint definitions
        if "constraint" not in file_path.name.lower() and "selection" not in file_path.name.lower():
            return issues

        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            return issues

        # Check for constraint rules
        if "create_default_constraint_rules" in content or "ConstraintRule" in content:
            issues.extend(self._validate_constraint_signs(file_path, content))

        return issues

    def _validate_constraint_signs(
        self, file_path: Path, content: str
    ) -> List[ValidationIssue]:
        """Validate that constraint signs match expected values."""
        issues = []
        lines = content.split("\n")

        # Check for prudential positive constraint
        if "prudential" in content.lower():
            has_positive = any(
                "positive" in line.lower() and "prudential" in content[max(0, content.find(line)-200):content.find(line)+200].lower()
                for line in lines
            )
            if not has_positive and "expected_sign" in content:
                # Find where prudential is mentioned
                for line_num, line in enumerate(lines, 1):
                    if "prudential" in line.lower() and "negative" in line.lower():
                        issues.append(ValidationIssue(
                            file_path=str(file_path),
                            line_number=line_num,
                            issue_type="WRONG_CONSTRAINT_SIGN",
                            message="Prudential coefficient should be POSITIVE (yield, not price)",
                            severity="ERROR",
                            suggestion="See knowledge/integration/LESSONS_LEARNED.md Trap #1",
                        ))

        # Check for competitor negative constraint
        if "competitor" in content.lower():
            has_negative = any(
                "negative" in line.lower() and "competitor" in content[max(0, content.find(line)-200):content.find(line)+200].lower()
                for line in lines
            )
            if not has_negative and "expected_sign" in content and "competitor" in content.lower():
                for line_num, line in enumerate(lines, 1):
                    if "competitor" in line.lower() and "positive" in line.lower() and "expected" in line.lower():
                        issues.append(ValidationIssue(
                            file_path=str(file_path),
                            line_number=line_num,
                            issue_type="WRONG_CONSTRAINT_SIGN",
                            message="Competitor coefficient should be NEGATIVE (substitution effect)",
                            severity="ERROR",
                            suggestion="See knowledge/analysis/CAUSAL_FRAMEWORK.md",
                        ))

        return issues


# =============================================================================
# COMPETING IMPLEMENTATION CHECKER
# =============================================================================

class CompetingImplementationChecker:
    """Check for multiple implementations of the same functionality."""

    # Patterns that should have single canonical implementation
    CANONICAL_PATTERNS = {
        "MathematicalEquivalence": "src/testing/",  # Allow split modules in same package
        "DataAdapter": "src/data/adapters/",
        "UnifiedNotebookInterface": "src/notebooks/",
    }

    def check_directory(self, base_path: Path) -> List[ValidationIssue]:
        """Check for competing implementations across directory."""
        issues = []
        implementation_locations: Dict[str, List[Tuple[Path, int]]] = defaultdict(list)

        for py_file in base_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)
            except (SyntaxError, UnicodeDecodeError):
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for pattern, canonical in self.CANONICAL_PATTERNS.items():
                        if pattern.lower() in node.name.lower():
                            implementation_locations[pattern].append(
                                (py_file, node.lineno)
                            )

        # Report competing implementations
        for pattern, locations in implementation_locations.items():
            canonical = self.CANONICAL_PATTERNS[pattern]

            # Filter out canonical location
            non_canonical = [
                (p, ln) for p, ln in locations
                if canonical not in str(p)
            ]

            if len(non_canonical) > 0 and len(locations) > 1:
                for file_path, line_num in non_canonical:
                    issues.append(ValidationIssue(
                        file_path=str(file_path),
                        line_number=line_num,
                        issue_type="COMPETING_IMPLEMENTATION",
                        message=f"Potential competing implementation of '{pattern}'",
                        severity="WARNING",
                        suggestion=f"Canonical implementation is in: {canonical}",
                    ))

        return issues


# =============================================================================
# MAIN VALIDATOR
# =============================================================================

class PatternValidator:
    """Main validator orchestrating all checks."""

    def __init__(self):
        self.import_checker = ImportHygieneChecker()
        self.lag0_checker = Lag0ConstraintChecker()
        self.constraint_checker = ConstraintDefinitionChecker()
        self.competing_checker = CompetingImplementationChecker()

    def validate(self, path: Path) -> ValidationReport:
        """Run all validations on the given path."""
        report = ValidationReport()

        if not path.exists():
            report.issues.append(ValidationIssue(
                file_path=str(path),
                line_number=0,
                issue_type="PATH_NOT_FOUND",
                message=f"Path does not exist: {path}",
                severity="ERROR",
            ))
            return report

        # Get all Python files
        if path.is_file():
            py_files = [path] if path.suffix == ".py" else []
        else:
            py_files = list(path.rglob("*.py"))
            py_files = [f for f in py_files if "__pycache__" not in str(f)]

        report.files_scanned = len(py_files)
        report.patterns_checked = 4  # Number of check types

        # Run file-level checks
        for py_file in py_files:
            report.issues.extend(self.import_checker.check_file(py_file))
            report.issues.extend(self.lag0_checker.check_file(py_file))
            report.issues.extend(self.constraint_checker.check_file(py_file))

        # Run directory-level checks
        base_path = path if path.is_dir() else path.parent
        report.issues.extend(self.competing_checker.check_directory(base_path))

        return report


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate codebase patterns for anti-pattern prevention.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("src"),
        help="Path to validate (default: src/)",
    )
    parser.add_argument(
        "--fix-suggestions",
        action="store_true",
        help="Show fix suggestions for each issue",
    )
    parser.add_argument(
        "--warnings-as-errors",
        action="store_true",
        help="Treat warnings as errors (exit code 1 on warnings)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output if there are issues",
    )

    args = parser.parse_args()

    validator = PatternValidator()
    report = validator.validate(args.path)

    if not args.quiet or report.issues:
        print(report)

    # Determine exit code
    if report.error_count > 0:
        sys.exit(1)
    elif args.warnings_as_errors and report.warning_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

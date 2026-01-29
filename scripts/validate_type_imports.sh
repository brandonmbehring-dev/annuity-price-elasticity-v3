#!/bin/bash
# Validate type imports after TypedDict deletion
#
# Verifies no dangling imports after TypedDict cleanup.
# Run as part of pre-commit or CI.
#
# Usage:
#   bash scripts/validate_type_imports.sh

set -e

echo "Type Import Validator"
echo "====================="

# Check if we're in the project root
if [ ! -f "pyproject.toml" ] && [ ! -d "src" ]; then
    echo "ERROR: Run from project root directory"
    exit 1
fi

errors=0

# Check 1: FeatureSelectionResults should come from selection_types.py after refactoring
# This will only be relevant after Phase 3 deletes the TypedDict
echo "Checking FeatureSelectionResults imports..."

# Files that should NOT import FeatureSelectionResults from core.types
# (After Phase 3, it will be deleted from there)
violations=$(grep -rn "from src.core.types import.*FeatureSelectionResults" src/ 2>/dev/null || true)

if [ -n "$violations" ]; then
    # Check if core/types.py still has the TypedDict (not yet deleted)
    if grep -q "class FeatureSelectionResults" src/core/types.py 2>/dev/null; then
        echo "  INFO: FeatureSelectionResults still exists in core/types.py (pre-Phase 3)"
        echo "  These imports will need updating after Phase 3:"
        echo "$violations" | head -5
    else
        echo "  ERROR: Imports from deleted TypedDict location:"
        echo "$violations"
        errors=$((errors + 1))
    fi
else
    echo "  PASS: No problematic FeatureSelectionResults imports"
fi

# Check 2: Verify selection_types.py exists and has the dataclass
echo "Checking selection_types.py..."
if [ -f "src/features/selection_types.py" ]; then
    if grep -q "class FeatureSelectionResults" src/features/selection_types.py; then
        echo "  PASS: FeatureSelectionResults dataclass exists in selection_types.py"
    else
        echo "  WARNING: FeatureSelectionResults not found in selection_types.py"
    fi
else
    echo "  WARNING: selection_types.py not found"
fi

# Check 3: No imports of non-existent types
echo "Checking for common type import errors..."

# Check for InferenceResults imports that might be wrong
inference_imports=$(grep -rn "from src.core.types import.*InferenceResults" src/ 2>/dev/null | wc -l)
echo "  InferenceResults imports from core.types: $inference_imports"

# Check 4: Verify no circular imports (basic check)
echo "Checking for potential circular imports..."
circular_check=$(grep -rn "from src.notebooks.interface import" src/core/ src/features/ src/models/ 2>/dev/null || true)
if [ -n "$circular_check" ]; then
    echo "  WARNING: Potential circular import - src.notebooks.interface imported from:"
    echo "$circular_check"
fi

echo ""
echo "Type import validation complete."
if [ $errors -gt 0 ]; then
    echo "FAIL: $errors error(s) found"
    exit 1
else
    echo "PASS: No critical errors"
    exit 0
fi

#!/bin/bash
# Emergency Rollback Script for Annuity Price Elasticity v3
#
# Usage:
#   ./scripts/emergency-rollback.sh [--dry-run] [commit-sha]
#
# Options:
#   --dry-run       Show what would be done without making changes
#   commit-sha      Specific commit to rollback to (default: HEAD~1)
#
# This script:
#   1. Creates a backup branch of current state
#   2. Identifies the target commit
#   3. Rolls back to that commit
#   4. Validates the rollback
#
# SAFETY: This script will NOT force-push. Manual intervention required for remote.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

DRY_RUN=false
TARGET_COMMIT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            TARGET_COMMIT=$1
            shift
            ;;
    esac
done

# Default to HEAD~1 if no commit specified
if [[ -z "$TARGET_COMMIT" ]]; then
    TARGET_COMMIT="HEAD~1"
fi

echo "=============================================="
echo "Emergency Rollback Script"
echo "=============================================="
echo ""

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Not in a git repository${NC}"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Get current commit
CURRENT_COMMIT=$(git rev-parse HEAD)
CURRENT_SHORT=$(git rev-parse --short HEAD)
echo "Current commit: $CURRENT_SHORT"

# Resolve target commit
if ! TARGET_SHA=$(git rev-parse "$TARGET_COMMIT" 2>/dev/null); then
    echo -e "${RED}ERROR: Cannot resolve commit: $TARGET_COMMIT${NC}"
    exit 1
fi
TARGET_SHORT=$(git rev-parse --short "$TARGET_SHA")
echo "Target commit:  $TARGET_SHORT"

# Show what's being rolled back
echo ""
echo "Commits to be rolled back:"
git log --oneline "$TARGET_SHA".."$CURRENT_COMMIT" | head -10
echo ""

# Check for uncommitted changes
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo -e "${YELLOW}WARNING: Uncommitted changes detected${NC}"
    if [[ "$DRY_RUN" == "false" ]]; then
        read -p "Stash uncommitted changes? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git stash push -m "Emergency rollback stash $(date +%Y%m%d_%H%M%S)"
            echo "Changes stashed."
        else
            echo -e "${RED}Aborting: Commit or stash your changes first${NC}"
            exit 1
        fi
    fi
fi

# Create backup branch
BACKUP_BRANCH="backup/${CURRENT_BRANCH}_$(date +%Y%m%d_%H%M%S)"
echo ""
echo "Creating backup branch: $BACKUP_BRANCH"

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}[DRY RUN] Would create branch: $BACKUP_BRANCH${NC}"
else
    git branch "$BACKUP_BRANCH"
    echo -e "${GREEN}Backup branch created${NC}"
fi

# Perform rollback
echo ""
echo "Rolling back to $TARGET_SHORT..."

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}[DRY RUN] Would run: git reset --hard $TARGET_SHA${NC}"
else
    git reset --hard "$TARGET_SHA"
    echo -e "${GREEN}Rollback complete${NC}"
fi

# Validate rollback
echo ""
echo "Validating rollback..."

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}[DRY RUN] Would run validation checks${NC}"
else
    # Check imports
    if python3 -c "from src.notebooks import create_interface" 2>/dev/null; then
        echo -e "${GREEN}  Import check: PASSED${NC}"
    else
        echo -e "${YELLOW}  Import check: FAILED (may need to reinstall)${NC}"
    fi

    # Check pattern validator
    if python3 scripts/pattern_validator.py --path src/ --quiet 2>/dev/null; then
        echo -e "${GREEN}  Pattern check: PASSED${NC}"
    else
        echo -e "${YELLOW}  Pattern check: FAILED or not available${NC}"
    fi
fi

# Summary
echo ""
echo "=============================================="
echo "Rollback Summary"
echo "=============================================="
echo "Previous commit: $CURRENT_SHORT"
echo "Current commit:  $TARGET_SHORT"
echo "Backup branch:   $BACKUP_BRANCH"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}This was a DRY RUN - no changes were made${NC}"
else
    echo -e "${GREEN}Rollback complete!${NC}"
    echo ""
    echo "To undo this rollback:"
    echo "  git reset --hard $CURRENT_COMMIT"
    echo ""
    echo "To push changes to remote (CAUTION - coordinate with team):"
    echo "  git push --force-with-lease origin $CURRENT_BRANCH"
fi

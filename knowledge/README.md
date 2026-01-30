# Knowledge Directory (Symlinks)

**Status**: Backward compatibility layer - canonical location is `docs/`

This directory contains symlinks to `docs/` subdirectories for backward compatibility.
Existing documentation references to `knowledge/` paths will continue to work.

## Symlink Mapping

| Symlink | Target |
|---------|--------|
| `domain/` | `docs/domain-knowledge/` |
| `analysis/` | `docs/analysis/` |
| `integration/` | `docs/integration/` |
| `methodology/` | `docs/methodology/` |
| `practices/` | `docs/practices/` |

## For New Documentation

**Always use `docs/` paths for new documentation:**
- New files → `docs/<category>/`
- New references → `docs/<category>/<file>.md`

## Platform Notes

- **Linux/macOS**: Symlinks work natively
- **Windows**: Requires Git Bash, WSL, or Developer Mode with symlink support

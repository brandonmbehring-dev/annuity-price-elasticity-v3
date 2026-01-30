#!/usr/bin/env python3
"""
Domain Knowledge Search (FTS5) for Annuity Price Elasticity v3.

Indexes all markdown files and Python docstrings for full-text search.
Uses SQLite FTS5 for fast, ranked search results.

Usage:
    # Index the knowledge base
    python scripts/domain_search.py index

    # Search for a term
    python scripts/domain_search.py search "cap rate"
    python scripts/domain_search.py search "leakage"

    # Show index statistics
    python scripts/domain_search.py stats

Database stored at: .domain_search.db (gitignored)
"""

import argparse
import ast
import re
import sqlite3
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_DB_PATH = ".domain_search.db"
KNOWLEDGE_DIRS = ["knowledge/", "docs/"]
CODE_DIRS = ["src/"]
MARKDOWN_EXTENSIONS = [".md"]
PYTHON_EXTENSIONS = [".py"]

# FTS5 tokenizer configuration
FTS5_TOKENIZE = "porter unicode61"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SearchResult:
    """Single search result."""

    file_path: str
    content_type: str  # 'markdown', 'docstring', 'comment'
    snippet: str
    rank: float
    line_number: Optional[int] = None

    def __str__(self) -> str:
        location = f"{self.file_path}"
        if self.line_number:
            location += f":{self.line_number}"
        return f"[{self.content_type}] {location}\n  {self.snippet[:200]}..."


# =============================================================================
# DATABASE SETUP
# =============================================================================

def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Get database connection with FTS5 support."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    """Create FTS5 tables for full-text search."""
    conn.executescript("""
        -- Main FTS5 table for content
        CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(
            file_path,
            content_type,
            content,
            line_number UNINDEXED,
            tokenize = 'porter unicode61'
        );

        -- Metadata table for file tracking
        CREATE TABLE IF NOT EXISTS file_metadata (
            file_path TEXT PRIMARY KEY,
            last_indexed TEXT,
            file_hash TEXT,
            content_type TEXT,
            doc_count INTEGER
        );

        -- Index statistics
        CREATE TABLE IF NOT EXISTS index_stats (
            stat_name TEXT PRIMARY KEY,
            stat_value TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()


# =============================================================================
# CONTENT EXTRACTION
# =============================================================================

def extract_markdown_content(file_path: Path) -> List[Dict[str, Any]]:
    """Extract content from markdown file."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    docs = []

    # Split into sections by headers
    sections = re.split(r'^(#{1,6}\s+.+)$', content, flags=re.MULTILINE)

    current_section = ""
    line_num = 1

    for i, section in enumerate(sections):
        if section.strip():
            if section.startswith("#"):
                current_section = section.strip()
            else:
                # Add section content
                docs.append({
                    "file_path": str(file_path),
                    "content_type": "markdown",
                    "content": f"{current_section}\n{section.strip()}"[:2000],
                    "line_number": line_num,
                })

        line_num += section.count("\n")

    # If no sections found, add whole file
    if not docs:
        docs.append({
            "file_path": str(file_path),
            "content_type": "markdown",
            "content": content[:2000],
            "line_number": 1,
        })

    return docs


def extract_docstrings(file_path: Path) -> List[Dict[str, Any]]:
    """Extract docstrings from Python file."""
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)
    except (SyntaxError, UnicodeDecodeError):
        return []

    docs = []

    for node in ast.walk(tree):
        docstring = None
        name = ""

        if isinstance(node, ast.Module):
            docstring = ast.get_docstring(node)
            name = "Module"
            line_num = 1
        elif isinstance(node, ast.ClassDef):
            docstring = ast.get_docstring(node)
            name = f"Class {node.name}"
            line_num = node.lineno
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            docstring = ast.get_docstring(node)
            name = f"Function {node.name}"
            line_num = node.lineno

        if docstring:
            docs.append({
                "file_path": str(file_path),
                "content_type": "docstring",
                "content": f"{name}: {docstring[:1500]}",
                "line_number": line_num,
            })

    return docs


def extract_comments(file_path: Path) -> List[Dict[str, Any]]:
    """Extract significant comments from Python file."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    docs = []
    lines = content.split("\n")

    # Look for block comments (multiple # lines together)
    block_start = None
    block_lines = []

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("#!"):
            if block_start is None:
                block_start = i
            block_lines.append(stripped[1:].strip())
        else:
            if len(block_lines) >= 2:  # Only significant comment blocks
                docs.append({
                    "file_path": str(file_path),
                    "content_type": "comment",
                    "content": " ".join(block_lines)[:1000],
                    "line_number": block_start,
                })
            block_start = None
            block_lines = []

    return docs


# =============================================================================
# INDEXING
# =============================================================================

def index_file(conn: sqlite3.Connection, file_path: Path) -> int:
    """Index a single file, return number of documents added."""
    docs = []

    if file_path.suffix in MARKDOWN_EXTENSIONS:
        docs = extract_markdown_content(file_path)
    elif file_path.suffix in PYTHON_EXTENSIONS:
        docs = extract_docstrings(file_path)
        # Optionally add comments (can be noisy)
        # docs.extend(extract_comments(file_path))

    if not docs:
        return 0

    # Remove existing entries for this file
    conn.execute("DELETE FROM content_fts WHERE file_path = ?", (str(file_path),))

    # Insert new entries
    for doc in docs:
        conn.execute(
            "INSERT INTO content_fts (file_path, content_type, content, line_number) VALUES (?, ?, ?, ?)",
            (doc["file_path"], doc["content_type"], doc["content"], doc.get("line_number")),
        )

    # Update metadata
    import hashlib
    from datetime import datetime

    file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
    conn.execute(
        """INSERT OR REPLACE INTO file_metadata
           (file_path, last_indexed, file_hash, content_type, doc_count)
           VALUES (?, ?, ?, ?, ?)""",
        (str(file_path), datetime.now().isoformat(), file_hash, docs[0]["content_type"], len(docs)),
    )

    return len(docs)


def index_directory(
    conn: sqlite3.Connection,
    base_path: Path,
    dirs: List[str],
    extensions: List[str],
) -> Tuple[int, int]:
    """Index all files in directories. Returns (files, documents)."""
    files_indexed = 0
    docs_indexed = 0

    for dir_name in dirs:
        dir_path = base_path / dir_name
        if not dir_path.exists():
            continue

        for ext in extensions:
            for file_path in dir_path.rglob(f"*{ext}"):
                if "__pycache__" in str(file_path):
                    continue
                count = index_file(conn, file_path)
                if count > 0:
                    files_indexed += 1
                    docs_indexed += count

    return files_indexed, docs_indexed


def build_index(base_path: Path, db_path: str = DEFAULT_DB_PATH) -> Dict[str, int]:
    """Build complete search index."""
    conn = get_connection(db_path)
    create_schema(conn)

    stats = {"markdown_files": 0, "python_files": 0, "total_docs": 0}

    # Index markdown (knowledge base)
    md_files, md_docs = index_directory(
        conn, base_path, KNOWLEDGE_DIRS, MARKDOWN_EXTENSIONS
    )
    stats["markdown_files"] = md_files
    stats["total_docs"] += md_docs

    # Index Python docstrings
    py_files, py_docs = index_directory(
        conn, base_path, CODE_DIRS, PYTHON_EXTENSIONS
    )
    stats["python_files"] = py_files
    stats["total_docs"] += py_docs

    # Update stats table
    from datetime import datetime

    for key, value in stats.items():
        conn.execute(
            "INSERT OR REPLACE INTO index_stats (stat_name, stat_value, updated_at) VALUES (?, ?, ?)",
            (key, str(value), datetime.now().isoformat()),
        )

    conn.commit()
    conn.close()

    return stats


# =============================================================================
# SEARCH
# =============================================================================

def search(
    query: str,
    db_path: str = DEFAULT_DB_PATH,
    limit: int = 10,
    content_type: Optional[str] = None,
) -> List[SearchResult]:
    """Search the index."""
    conn = get_connection(db_path)

    # Build query with optional content type filter
    sql = """
        SELECT file_path, content_type, snippet(content_fts, 2, '>>>', '<<<', '...', 50), rank, line_number
        FROM content_fts
        WHERE content_fts MATCH ?
    """

    params = [query]

    if content_type:
        sql += " AND content_type = ?"
        params.append(content_type)

    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    results = []
    for row in conn.execute(sql, params):
        results.append(SearchResult(
            file_path=row[0],
            content_type=row[1],
            snippet=row[2],
            rank=row[3],
            line_number=row[4],
        ))

    conn.close()
    return results


def get_stats(db_path: str = DEFAULT_DB_PATH) -> Dict[str, str]:
    """Get index statistics."""
    conn = get_connection(db_path)

    stats = {}
    try:
        for row in conn.execute("SELECT stat_name, stat_value FROM index_stats"):
            stats[row[0]] = row[1]

        # Get total document count
        count = conn.execute("SELECT COUNT(*) FROM content_fts").fetchone()[0]
        stats["indexed_documents"] = str(count)

    except sqlite3.OperationalError:
        stats["error"] = "Index not built. Run: python scripts/domain_search.py index"

    conn.close()
    return stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Domain knowledge search (FTS5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    index_parser = subparsers.add_parser("index", help="Build search index")
    index_parser.add_argument(
        "--path", type=Path, default=Path("."), help="Base path"
    )
    index_parser.add_argument(
        "--db", type=str, default=DEFAULT_DB_PATH, help="Database path"
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search the index")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=10, help="Max results"
    )
    search_parser.add_argument(
        "--type", type=str, choices=["markdown", "docstring", "comment"],
        help="Filter by content type"
    )
    search_parser.add_argument(
        "--db", type=str, default=DEFAULT_DB_PATH, help="Database path"
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show index statistics")
    stats_parser.add_argument(
        "--db", type=str, default=DEFAULT_DB_PATH, help="Database path"
    )

    args = parser.parse_args()

    if args.command == "index":
        print(f"Building index from {args.path}...")
        stats = build_index(args.path, args.db)
        print(f"Indexed {stats['markdown_files']} markdown files")
        print(f"Indexed {stats['python_files']} Python files")
        print(f"Total documents: {stats['total_docs']}")
        print(f"Database: {args.db}")

    elif args.command == "search":
        results = search(args.query, args.db, args.limit, args.type)
        if not results:
            print(f"No results for: {args.query}")
            return

        print(f"Found {len(results)} results for: {args.query}\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}")
            print()

    elif args.command == "stats":
        stats = get_stats(args.db)
        print("Index Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

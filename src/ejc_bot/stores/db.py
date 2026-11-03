"""SQLite access for the event schedule."""

import sqlite3
from pathlib import Path

MAX_ROWS = 50


def connect_readonly(path: Path) -> sqlite3.Connection:
    """Open the database read-only, so even a malicious LLM-written query cannot modify it."""
    if not path.exists():
        raise FileNotFoundError(f"Database '{path}' not found. Run `ejc-ingest-schedule` first.")
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def run_select(path: Path, query: str, max_rows: int = MAX_ROWS) -> list[dict]:
    """Run a single SELECT statement and return at most ``max_rows`` rows as dicts."""
    if not query.strip().lower().startswith(("select", "with")):
        raise ValueError("Only SELECT queries are permitted on the schedule database.")

    conn = connect_readonly(path)
    try:
        cursor = conn.execute(query)
        col_names = [col[0] for col in cursor.description]
        return [dict(zip(col_names, row, strict=True)) for row in cursor.fetchmany(max_rows)]
    finally:
        conn.close()


def describe_schema(path: Path, table: str = "events") -> str:
    """Column list of ``table``, used to keep the SQL tool description in sync with the DB."""
    conn = connect_readonly(path)
    try:
        columns = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    return ", ".join(col[1] for col in columns)

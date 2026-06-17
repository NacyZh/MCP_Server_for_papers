"""Schema migration helpers for local RAG storage."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict

SQLITE_SCHEMA_VERSION = 4
CHROMA_SCHEMA_VERSION = 1

CHROMA_METADATA_FILENAME = "rag_chroma_schema.json"


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})")}


def _ensure_schema_version_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            component  TEXT PRIMARY KEY,
            version    INTEGER NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def get_sqlite_schema_version(conn: sqlite3.Connection) -> int:
    _ensure_schema_version_table(conn)
    row = conn.execute(
        "SELECT version FROM schema_version WHERE component = ?",
        ("sqlite",),
    ).fetchone()
    return int(row["version"]) if row else 0


def set_sqlite_schema_version(conn: sqlite3.Connection, version: int) -> None:
    _ensure_schema_version_table(conn)
    conn.execute(
        """
        INSERT INTO schema_version(component, version, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(component) DO UPDATE SET
            version = excluded.version,
            updated_at = excluded.updated_at
        """,
        ("sqlite", int(version)),
    )


def migrate_sqlite(conn: sqlite3.Connection) -> None:
    """Apply idempotent SQLite migrations up to the current schema version."""

    version = get_sqlite_schema_version(conn)
    if version > SQLITE_SCHEMA_VERSION:
        raise RuntimeError(
            f"SQLite schema version {version} is newer than supported version {SQLITE_SCHEMA_VERSION}."
        )

    _migrate_sqlite_v1(conn)
    _migrate_sqlite_v2(conn)
    _migrate_sqlite_v3(conn)
    _migrate_sqlite_v4(conn)
    set_sqlite_schema_version(conn, SQLITE_SCHEMA_VERSION)
    conn.commit()


def _migrate_sqlite_v1(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            authors     TEXT,
            abstract    TEXT,
            publish_year INTEGER,
            local_path  TEXT,
            tags        TEXT,
            add_time    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _migrate_sqlite_v2(conn: sqlite3.Connection) -> None:
    existing_cols = _table_columns(conn, "papers")
    for col_name, col_type in (
        ("content_hash", "TEXT"),
        ("doi", "TEXT"),
        ("normalized_title", "TEXT"),
    ):
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE papers ADD COLUMN {col_name} {col_type}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_content_hash ON papers(content_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_normalized_title ON papers(normalized_title)")


def _migrate_sqlite_v3(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_chunks (
            paper_id    TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_type  TEXT,
            section_name TEXT,
            section_title TEXT,
            content     TEXT NOT NULL,
            PRIMARY KEY (paper_id, chunk_index)
        )
        """
    )
    existing_cols = _table_columns(conn, "paper_chunks")
    for col_name, col_type in (
        ("chunk_type", "TEXT"),
        ("section_name", "TEXT"),
        ("section_title", "TEXT"),
    ):
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE paper_chunks ADD COLUMN {col_name} {col_type}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_chunks_paper_id ON paper_chunks(paper_id)")


def _migrate_sqlite_v4(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_sections (
            paper_id                TEXT NOT NULL,
            section_id              TEXT NOT NULL,
            section_name            TEXT NOT NULL,
            normalized_section_name TEXT NOT NULL,
            section_order           INTEGER NOT NULL,
            chunk_count             INTEGER NOT NULL,
            char_count              INTEGER NOT NULL,
            content_hash            TEXT NOT NULL,
            created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (paper_id, section_id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_sections_paper_id ON paper_sections(paper_id)")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_profiles (
            paper_id      TEXT NOT NULL,
            language      TEXT NOT NULL,
            profile_json  TEXT NOT NULL,
            source_hash   TEXT NOT NULL,
            model_name    TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            status        TEXT NOT NULL,
            error_code    TEXT,
            error_message TEXT,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (paper_id, language)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_summaries (
            paper_id       TEXT NOT NULL,
            language       TEXT NOT NULL,
            detail_level   TEXT NOT NULL,
            summary_json   TEXT NOT NULL,
            source_hash    TEXT NOT NULL,
            model_name     TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            status         TEXT NOT NULL,
            error_code     TEXT,
            error_message  TEXT,
            created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (paper_id, language, detail_level)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS section_summaries (
            paper_id       TEXT NOT NULL,
            section_id     TEXT NOT NULL,
            section_name   TEXT NOT NULL,
            language       TEXT NOT NULL,
            detail_level   TEXT NOT NULL,
            summary_text   TEXT NOT NULL,
            source_hash    TEXT NOT NULL,
            model_name     TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            status         TEXT NOT NULL,
            error_code     TEXT,
            error_message  TEXT,
            created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (paper_id, section_id, language, detail_level)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_section_summaries_paper ON section_summaries(paper_id, language, detail_level)"
    )


def chroma_metadata_path(chroma_dir: str | os.PathLike[str]) -> Path:
    return Path(chroma_dir) / CHROMA_METADATA_FILENAME


def ensure_chroma_schema(chroma_dir: str | os.PathLike[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Create or validate the Chroma sidecar schema metadata file."""

    path = chroma_metadata_path(chroma_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = {
        "schema_version": CHROMA_SCHEMA_VERSION,
        **metadata,
    }
    if not path.exists():
        path.write_text(json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8")
        return expected

    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid Chroma schema metadata file: {path}") from exc

    version = int(current.get("schema_version", 0))
    if version > CHROMA_SCHEMA_VERSION:
        raise RuntimeError(
            f"Chroma schema version {version} is newer than supported version {CHROMA_SCHEMA_VERSION}."
        )
    if version < CHROMA_SCHEMA_VERSION:
        raise RuntimeError(
            f"Chroma schema version {version} requires migration to {CHROMA_SCHEMA_VERSION}."
        )
    return current

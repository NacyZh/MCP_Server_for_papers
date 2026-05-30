"""SQLite metadata store for paper information."""

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

from scholar_agent.config import conf

# Columns that are conceptually part of the schema but were added in later
# migrations. ``_init_tables`` will ALTER existing databases to add any that
# are missing, so old papers.db files keep working without manual migration.
_DEDUP_COLUMNS = (
    ("content_hash", "TEXT"),
    ("doi", "TEXT"),
    ("normalized_title", "TEXT"),
)


class PaperDB:
    """Manages paper metadata (title, authors, abstract, etc.) in SQLite."""

    def __init__(self):
        self.db_path = os.path.join(conf.DB_DIR, "papers.db")
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_tables()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    id          TEXT PRIMARY KEY,
                    title       TEXT NOT NULL,
                    authors     TEXT,
                    abstract    TEXT,
                    publish_year INTEGER,
                    local_path  TEXT,
                    tags        TEXT,
                    content_hash TEXT,
                    doi         TEXT,
                    normalized_title TEXT,
                    add_time    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Idempotent migration for databases created before the dedup
            # columns existed. SQLite has no "ADD COLUMN IF NOT EXISTS".
            existing_cols = {row["name"] for row in cursor.execute("PRAGMA table_info(papers)")}
            for col_name, col_type in _DEDUP_COLUMNS:
                if col_name not in existing_cols:
                    cursor.execute(f"ALTER TABLE papers ADD COLUMN {col_name} {col_type}")

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_content_hash ON papers(content_hash)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_normalized_title ON papers(normalized_title)"
            )
            conn.commit()

    def add_paper(
        self,
        paper_id,
        title,
        authors="",
        abstract="",
        publish_year=None,
        local_path="",
        tags="",
        content_hash: str = "",
        doi: str = "",
        normalized_title: str = "",
    ):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO papers
                (id, title, authors, abstract, publish_year, local_path, tags,
                 content_hash, doi, normalized_title)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper_id,
                    title,
                    authors,
                    abstract,
                    publish_year,
                    local_path,
                    tags,
                    content_hash or None,
                    doi or None,
                    normalized_title or None,
                ),
            )
            conn.commit()
            return True

    def update_dedup_fields(
        self,
        paper_id: str,
        content_hash: Optional[str] = None,
        doi: Optional[str] = None,
        normalized_title: Optional[str] = None,
    ) -> bool:
        """Update only the dedup-related columns for a given paper."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE papers
                SET content_hash = COALESCE(?, content_hash),
                    doi = COALESCE(?, doi),
                    normalized_title = COALESCE(?, normalized_title)
                WHERE id = ?
                """,
                (content_hash, doi, normalized_title, paper_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_paper(self, paper_id):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def search_papers(self, keyword):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            search_term = f"%{keyword}%"
            cursor.execute(
                """
                SELECT id, title, authors, publish_year, tags
                FROM papers
                WHERE title LIKE ? OR authors LIKE ? OR abstract LIKE ? OR tags LIKE ?
                ORDER BY add_time DESC
                """,
                (search_term, search_term, search_term, search_term),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_all_papers(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, title, authors, publish_year, tags, local_path,
                       content_hash, doi, normalized_title, add_time
                FROM papers
                ORDER BY add_time DESC
                """
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_by_content_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        if not content_hash:
            return None
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM papers WHERE content_hash = ? LIMIT 1",
                (content_hash,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        if not doi:
            return None
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM papers WHERE doi = ? LIMIT 1",
                (doi,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_by_normalized_title(self, normalized_title: str) -> List[Dict[str, Any]]:
        if not normalized_title:
            return []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM papers
                WHERE normalized_title = ?
                ORDER BY add_time DESC
                """,
                (normalized_title,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def delete_paper(self, paper_id):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
            conn.commit()


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    conf.check_config()
    db = PaperDB()
    print("1. Database initialized successfully!")

    results = db.search_papers("RAG")
    print(f"2. Papers matching 'RAG':\n{json.dumps(results, indent=2, ensure_ascii=False)}")

    all_papers = db.get_all_papers()
    print(f"3. Total papers in database: {len(all_papers)}")

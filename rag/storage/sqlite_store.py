"""SQLite metadata store for paper information."""

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from config import conf
from rag.storage.migrations import SQLITE_SCHEMA_VERSION, get_sqlite_schema_version, migrate_sqlite


def _normalize_section_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "unknown").strip().lower()).strip("_")
    return normalized or "unknown"


class PaperDB:
    """Manages paper metadata (title, authors, abstract, etc.) in SQLite."""

    def __init__(self):
        self.db_path = os.path.join(conf.DB_DIR, "papers.db")
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_tables()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_tables(self):
        with self._get_connection() as conn:
            migrate_sqlite(conn)

    def _init_chunk_table(self, best_effort: bool = False) -> bool:
        try:
            with self._get_connection() as conn:
                migrate_sqlite(conn)
                conn.commit()
            return True
        except sqlite3.OperationalError:
            if best_effort:
                return False
            raise

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

    def replace_paper_chunks(self, paper_id: str, chunks: List[Any]) -> int:
        self._init_chunk_table()
        cleaned = []
        for idx, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                content = str(chunk.get("content") or "").strip()
                if not content:
                    continue
                cleaned.append(
                    {
                        "index": int(chunk.get("chunk_index", idx) or idx),
                        "content": content,
                        "chunk_type": str(chunk.get("chunk_type") or "text_chunk"),
                        "section_name": str(chunk.get("section_name") or "unknown"),
                        "section_title": str(chunk.get("section_title") or "Unknown"),
                    }
                )
            else:
                content = str(chunk or "").strip()
                if not content:
                    continue
                cleaned.append(
                    {
                        "index": idx,
                        "content": content,
                        "chunk_type": "text_chunk",
                        "section_name": "unknown",
                        "section_title": "Unknown",
                    }
                )
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM paper_chunks WHERE paper_id = ?", (paper_id,))
            cursor.execute("DELETE FROM paper_sections WHERE paper_id = ?", (paper_id,))
            cursor.executemany(
                """
                INSERT INTO paper_chunks
                (paper_id, chunk_index, chunk_type, section_name, section_title, content)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        paper_id,
                        item["index"],
                        item["chunk_type"],
                        item["section_name"],
                        item["section_title"],
                        item["content"],
                    )
                    for item in cleaned
                ],
            )
            self._rebuild_paper_sections(conn, paper_id)
            conn.commit()
        return len(cleaned)

    def _rebuild_paper_sections(self, conn: sqlite3.Connection, paper_id: str) -> None:
        rows = conn.execute(
            """
            SELECT section_title, section_name, MIN(chunk_index) AS first_chunk,
                   COUNT(*) AS chunk_count, SUM(LENGTH(content)) AS char_count,
                   GROUP_CONCAT(content, char(10) || char(10)) AS content
            FROM paper_chunks
            WHERE paper_id = ?
            GROUP BY section_title, section_name
            ORDER BY first_chunk ASC
            """,
            (paper_id,),
        ).fetchall()
        for idx, row in enumerate(rows):
            section_name = row["section_title"] or row["section_name"] or "Unknown"
            normalized = _normalize_section_name(section_name)
            section_id = f"{idx + 1:04d}_{normalized}"
            content_hash = hashlib.sha256(str(row["content"] or "").encode("utf-8")).hexdigest()
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_sections
                (paper_id, section_id, section_name, normalized_section_name,
                 section_order, chunk_count, char_count, content_hash, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    paper_id,
                    section_id,
                    section_name,
                    normalized,
                    idx,
                    int(row["chunk_count"] or 0),
                    int(row["char_count"] or 0),
                    content_hash,
                ),
            )

    def get_paper_chunks(
        self,
        paper_id: str,
        limit: int = 8,
        section: str = "",
    ) -> List[Dict[str, Any]]:
        if not self._init_chunk_table(best_effort=True):
            return []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            params: List[Any] = [paper_id]
            where = "paper_id = ?"
            if section:
                where += " AND (section_name LIKE ? OR section_title LIKE ?)"
                like = f"%{section}%"
                params.extend([like, like])
            params.append(max(1, int(limit)))
            cursor.execute(
                f"""
                SELECT paper_id, chunk_index, chunk_type, section_name, section_title, content
                FROM paper_chunks
                WHERE {where}
                ORDER BY chunk_index ASC
                LIMIT ?
                """,
                params,
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_paper_sections(self, paper_id: str) -> List[str]:
        if not self._init_chunk_table(best_effort=True):
            return []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            has_section_table = cursor.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_sections'"
            ).fetchone()
            if has_section_table:
                cursor.execute(
                    """
                    SELECT section_name, chunk_count
                    FROM paper_sections
                    WHERE paper_id = ?
                    ORDER BY section_order ASC
                    """,
                    (paper_id,),
                )
                sections = [
                    f"{(row['section_name'] or 'Unknown')} ({row['chunk_count']})"
                    for row in cursor.fetchall()
                ]
                if sections:
                    return sections
            cursor.execute(
                """
                SELECT section_title, section_name, COUNT(*) AS chunk_count
                FROM paper_chunks
                WHERE paper_id = ?
                GROUP BY section_title, section_name
                ORDER BY MIN(chunk_index)
                """,
                (paper_id,),
            )
            sections = []
            for row in cursor.fetchall():
                title = row["section_title"] or row["section_name"] or "Unknown"
                sections.append(f"{title} ({row['chunk_count']})")
            return sections

    def get_paper_outline(
        self,
        paper_id: str,
        language: str = "en",
        detail_level: str = "medium",
    ) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            paper = conn.execute("SELECT id, title FROM papers WHERE id = ?", (paper_id,)).fetchone()
            if not paper:
                return None
            rows = conn.execute(
                """
                SELECT s.section_id, s.section_name, s.normalized_section_name,
                       s.section_order, s.chunk_count, s.char_count, s.content_hash,
                       CASE WHEN ss.status = 'ready' THEN 1 ELSE 0 END AS has_summary
                FROM paper_sections s
                LEFT JOIN section_summaries ss
                  ON ss.paper_id = s.paper_id
                 AND ss.section_id = s.section_id
                 AND ss.language = ?
                 AND ss.detail_level = ?
                WHERE s.paper_id = ?
                ORDER BY s.section_order ASC
                """,
                (language, detail_level, paper_id),
            ).fetchall()
            if not rows:
                self._rebuild_paper_sections(conn, paper_id)
                conn.commit()
                rows = conn.execute(
                    """
                    SELECT s.section_id, s.section_name, s.normalized_section_name,
                           s.section_order, s.chunk_count, s.char_count, s.content_hash,
                           CASE WHEN ss.status = 'ready' THEN 1 ELSE 0 END AS has_summary
                    FROM paper_sections s
                    LEFT JOIN section_summaries ss
                      ON ss.paper_id = s.paper_id
                     AND ss.section_id = s.section_id
                     AND ss.language = ?
                     AND ss.detail_level = ?
                    WHERE s.paper_id = ?
                    ORDER BY s.section_order ASC
                    """,
                    (language, detail_level, paper_id),
                ).fetchall()
            return {
                "paper_id": paper["id"],
                "title": paper["title"],
                "sections": [
                    {
                        "section_id": row["section_id"],
                        "section": row["section_name"],
                        "normalized_section_name": row["normalized_section_name"],
                        "section_order": row["section_order"],
                        "chunk_count": row["chunk_count"],
                        "char_count": row["char_count"],
                        "content_hash": row["content_hash"],
                        "has_summary": bool(row["has_summary"]),
                    }
                    for row in rows
                ],
            }

    def get_source_hash(self, paper_id: str) -> str:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT content_hash FROM paper_sections
                WHERE paper_id = ?
                ORDER BY section_order ASC
                """,
                (paper_id,),
            ).fetchall()
            if not rows:
                self._rebuild_paper_sections(conn, paper_id)
                conn.commit()
                rows = conn.execute(
                    """
                    SELECT content_hash FROM paper_sections
                    WHERE paper_id = ?
                    ORDER BY section_order ASC
                    """,
                    (paper_id,),
                ).fetchall()
            payload = "\n".join(row["content_hash"] for row in rows)
            return hashlib.sha256(payload.encode("utf-8")).hexdigest() if payload else ""

    def get_profile(self, paper_id: str, language: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT profile_json, source_hash, model_name, prompt_version, status,
                       error_code, error_message, updated_at
                FROM paper_profiles
                WHERE paper_id = ? AND language = ?
                """,
                (paper_id, language),
            ).fetchone()
            if not row:
                return None
            profile = json.loads(row["profile_json"])
            return {
                **profile,
                "paper_id": paper_id,
                "language": language,
                "source_hash": row["source_hash"],
                "model_name": row["model_name"],
                "prompt_version": row["prompt_version"],
                "status": row["status"],
                "error_code": row["error_code"],
                "error_message": row["error_message"],
                "updated_at": row["updated_at"],
            }

    def get_summary(self, paper_id: str, language: str, detail_level: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT summary_json, source_hash, model_name, prompt_version, status,
                       error_code, error_message, updated_at
                FROM paper_summaries
                WHERE paper_id = ? AND language = ? AND detail_level = ?
                """,
                (paper_id, language, detail_level),
            ).fetchone()
            if not row:
                return None
            summary = json.loads(row["summary_json"])
            return {
                "paper_id": paper_id,
                "language": language,
                "detail_level": detail_level,
                "summary_status": row["status"],
                "summary": summary,
                "source_hash": row["source_hash"],
                "model_name": row["model_name"],
                "prompt_version": row["prompt_version"],
                "error_code": row["error_code"],
                "error_message": row["error_message"],
                "updated_at": row["updated_at"],
            }

    def get_section_summaries(self, paper_id: str, language: str, detail_level: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT section_id, section_name, summary_text, source_hash, status
                FROM section_summaries
                WHERE paper_id = ? AND language = ? AND detail_level = ?
                ORDER BY section_id ASC
                """,
                (paper_id, language, detail_level),
            ).fetchall()
            return [dict(row) for row in rows]

    def build_summary_cache(
        self,
        paper_id: str,
        language: str = "en",
        detail_levels: Optional[List[str]] = None,
        force_rebuild: bool = False,
        summary_generator: Any = None,
    ) -> Dict[str, Any]:
        if summary_generator is None:
            raise ValueError("summary_generator is required for summary cache generation")

        detail_levels = detail_levels or ["short", "medium", "long"]
        paper = self.get_paper(paper_id)
        if not paper:
            return {"status": "failed", "error_code": "PAPER_NOT_FOUND", "paper_id": paper_id}

        source_hash = self.get_source_hash(paper_id)
        if not source_hash:
            return {"status": "failed", "error_code": "PAPER_HAS_NO_CHUNKS", "paper_id": paper_id}

        with self._get_connection() as conn:
            section_rows = conn.execute(
                """
                SELECT s.section_id, s.section_name, s.content_hash,
                       GROUP_CONCAT(c.content, char(10) || char(10)) AS content
                FROM paper_sections s
                JOIN paper_chunks c
                  ON c.paper_id = s.paper_id
                 AND (c.section_title = s.section_name OR c.section_name = s.normalized_section_name)
                WHERE s.paper_id = ?
                GROUP BY s.section_id, s.section_name, s.content_hash, s.section_order
                ORDER BY s.section_order ASC
                """,
                (paper_id,),
            ).fetchall()
            if not section_rows:
                return {"status": "failed", "error_code": "PAPER_HAS_NO_CHUNKS", "paper_id": paper_id}

            built = []
            skipped = []
            for detail_level in detail_levels:
                model_name = getattr(summary_generator, "model_name", conf.RAG_SUMMARY_MODEL_NAME)
                prompt_version = getattr(summary_generator, "prompt_version", conf.RAG_SUMMARY_PROMPT_VERSION)
                existing = conn.execute(
                    """
                    SELECT source_hash, model_name, prompt_version FROM paper_summaries
                    WHERE paper_id = ? AND language = ? AND detail_level = ? AND status = 'ready'
                    """,
                    (paper_id, language, detail_level),
                ).fetchone()
                if (
                    existing
                    and existing["source_hash"] == source_hash
                    and existing["model_name"] == model_name
                    and existing["prompt_version"] == prompt_version
                    and not force_rebuild
                ):
                    skipped.append(detail_level)
                    continue

                try:
                    section_summaries = [
                        {
                            "section_id": row["section_id"],
                            "section_name": row["section_name"],
                            "summary": self._build_section_summary(
                                summary_generator,
                                row["section_name"],
                                row["content"],
                                language,
                                detail_level,
                            ),
                            "source_hash": row["content_hash"],
                        }
                        for row in section_rows
                    ]
                    profile = self._build_profile(summary_generator, paper, section_summaries, language)
                    summary = self._build_paper_summary(
                        summary_generator,
                        profile,
                        section_summaries,
                        language,
                        detail_level,
                    )
                except TimeoutError as exc:
                    return {
                        "status": "failed",
                        "error_code": "SUMMARY_GENERATION_TIMEOUT",
                        "error_message": str(exc),
                        "paper_id": paper_id,
                        "language": language,
                    }
                except Exception as exc:
                    error_code = (
                        "SUMMARY_MODEL_UNAVAILABLE"
                        if exc.__class__.__name__ == "SummaryModelUnavailableError"
                        else "SUMMARY_GENERATION_FAILED"
                    )
                    return {
                        "status": "failed",
                        "error_code": error_code,
                        "error_message": str(exc),
                        "paper_id": paper_id,
                        "language": language,
                    }

                conn.execute(
                    """
                    INSERT INTO paper_profiles
                    (paper_id, language, profile_json, source_hash, model_name,
                     prompt_version, status, error_code, error_message, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'ready', NULL, NULL, CURRENT_TIMESTAMP)
                    ON CONFLICT(paper_id, language) DO UPDATE SET
                        profile_json = excluded.profile_json,
                        source_hash = excluded.source_hash,
                        model_name = excluded.model_name,
                        prompt_version = excluded.prompt_version,
                        status = excluded.status,
                        error_code = NULL,
                        error_message = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        paper_id,
                        language,
                        json.dumps(profile, ensure_ascii=False),
                        source_hash,
                        model_name,
                        prompt_version,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO paper_summaries
                    (paper_id, language, detail_level, summary_json, source_hash, model_name,
                     prompt_version, status, error_code, error_message, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'ready', NULL, NULL, CURRENT_TIMESTAMP)
                    ON CONFLICT(paper_id, language, detail_level) DO UPDATE SET
                        summary_json = excluded.summary_json,
                        source_hash = excluded.source_hash,
                        model_name = excluded.model_name,
                        prompt_version = excluded.prompt_version,
                        status = excluded.status,
                        error_code = NULL,
                        error_message = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        paper_id,
                        language,
                        detail_level,
                        json.dumps(summary, ensure_ascii=False),
                        source_hash,
                        model_name,
                        prompt_version,
                    ),
                )
                for item in section_summaries:
                    conn.execute(
                        """
                        INSERT INTO section_summaries
                        (paper_id, section_id, section_name, language, detail_level,
                         summary_text, source_hash, model_name, prompt_version,
                         status, error_code, error_message, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', NULL, NULL, CURRENT_TIMESTAMP)
                        ON CONFLICT(paper_id, section_id, language, detail_level) DO UPDATE SET
                            section_name = excluded.section_name,
                            summary_text = excluded.summary_text,
                            source_hash = excluded.source_hash,
                            model_name = excluded.model_name,
                            prompt_version = excluded.prompt_version,
                            status = excluded.status,
                            error_code = NULL,
                            error_message = NULL,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (
                            paper_id,
                            item["section_id"],
                            item["section_name"],
                            language,
                            detail_level,
                            item["summary"],
                            item["source_hash"],
                            model_name,
                            prompt_version,
                        ),
                    )
                built.append(detail_level)
            conn.commit()

        return {
            "status": "ready",
            "paper_id": paper_id,
            "language": language,
            "detail_levels_built": built,
            "detail_levels_skipped": skipped,
            "source_hash": source_hash,
            "model_name": getattr(summary_generator, "model_name", conf.RAG_SUMMARY_MODEL_NAME),
            "prompt_version": getattr(summary_generator, "prompt_version", conf.RAG_SUMMARY_PROMPT_VERSION),
        }

    def _build_section_summary(
        self,
        summary_generator: Any,
        section_name: str,
        content: str,
        language: str,
        detail_level: str,
    ) -> str:
        return str(
            summary_generator.summarize_section(
                section_name=section_name,
                content=content,
                language=language,
                detail_level=detail_level,
            )
        )

    def _build_profile(
        self,
        summary_generator: Any,
        paper: Dict[str, Any],
        section_summaries: List[Dict[str, str]],
        language: str,
    ) -> Dict[str, Any]:
        return dict(
            summary_generator.build_profile(
                paper=paper,
                section_summaries=section_summaries,
                language=language,
            )
        )

    def _build_paper_summary(
        self,
        summary_generator: Any,
        profile: Dict[str, Any],
        section_summaries: List[Dict[str, str]],
        language: str,
        detail_level: str,
    ) -> Dict[str, str]:
        return dict(
            summary_generator.build_paper_summary(
                profile=profile,
                section_summaries=section_summaries,
                language=language,
                detail_level=detail_level,
            )
        )

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
            results = [dict(row) for row in rows]
            self._attach_sections(conn, results)
            return results

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
            results = [dict(row) for row in rows]
            self._attach_sections(conn, results)
            return results

    def get_schema_version(self) -> int:
        with self._get_connection() as conn:
            return get_sqlite_schema_version(conn)

    @property
    def supported_schema_version(self) -> int:
        return SQLITE_SCHEMA_VERSION

    def _attach_sections(self, conn: sqlite3.Connection, papers: List[Dict[str, Any]]) -> None:
        """Attach cached section summaries to paper rows in one query."""
        if not papers:
            return
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_chunks'"
        ).fetchone()
        if not table_exists:
            for paper in papers:
                paper["sections"] = []
            return
        ids = [paper["id"] for paper in papers if paper.get("id")]
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"""
            SELECT paper_id, section_title, section_name, COUNT(*) AS chunk_count,
                   MIN(chunk_index) AS first_chunk
            FROM paper_chunks
            WHERE paper_id IN ({placeholders})
            GROUP BY paper_id, section_title, section_name
            ORDER BY first_chunk
            """,
            ids,
        ).fetchall()
        sections_by_paper: Dict[str, List[str]] = {paper_id: [] for paper_id in ids}
        for row in rows:
            title = row["section_title"] or row["section_name"] or "Unknown"
            sections_by_paper.setdefault(row["paper_id"], []).append(f"{title} ({row['chunk_count']})")
        for paper in papers:
            paper["sections"] = sections_by_paper.get(paper["id"], [])

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
            cursor.execute("DELETE FROM section_summaries WHERE paper_id = ?", (paper_id,))
            cursor.execute("DELETE FROM paper_summaries WHERE paper_id = ?", (paper_id,))
            cursor.execute("DELETE FROM paper_profiles WHERE paper_id = ?", (paper_id,))
            cursor.execute("DELETE FROM paper_sections WHERE paper_id = ?", (paper_id,))
            cursor.execute("DELETE FROM paper_chunks WHERE paper_id = ?", (paper_id,))
            cursor.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
            conn.commit()

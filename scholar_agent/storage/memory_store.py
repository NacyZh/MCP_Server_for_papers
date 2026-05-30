"""Persistent memory store for multi-agent sessions."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Dict, List

from scholar_agent.config import conf

AgentMemorySnapshot = Dict[str, Any]
ExpertOutputSnapshot = Dict[str, Any]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_json_list(raw: str | None) -> List[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _json_list(values: List[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


def _dedupe_keep_newest(values: List[str], limit: int) -> List[str]:
    result: List[str] = []
    seen = set()
    for raw in values:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        result.append(value)
        seen.add(value)
        if len(result) >= limit:
            break
    return result


class AgentMemoryStore:
    """SQLite-backed rolling memory for one multi-agent conversation session."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or conf.AGENT_MEMORY_DB
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_tables()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        with closing(self._get_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_memory (
                    session_id       TEXT PRIMARY KEY,
                    summary          TEXT NOT NULL DEFAULT '',
                    user_preferences TEXT NOT NULL DEFAULT '[]',
                    recent_topics    TEXT NOT NULL DEFAULT '[]',
                    recent_paper_ids TEXT NOT NULL DEFAULT '[]',
                    active_skills    TEXT NOT NULL DEFAULT '[]',
                    recent_code_project_path TEXT NOT NULL DEFAULT '',
                    recent_code_project_slug TEXT NOT NULL DEFAULT '',
                    recent_code_delivery_status TEXT NOT NULL DEFAULT '',
                    recent_code_validation_evidence TEXT NOT NULL DEFAULT '',
                    turn_count       INTEGER NOT NULL DEFAULT 0,
                    created_at       TEXT NOT NULL,
                    updated_at       TEXT NOT NULL
                )
                """
            )
            self._ensure_agent_memory_columns(cursor)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_memory_events (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id      TEXT NOT NULL,
                    role            TEXT NOT NULL,
                    content_preview TEXT NOT NULL,
                    metadata        TEXT NOT NULL DEFAULT '{}',
                    created_at      TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_memory_events_session "
                "ON agent_memory_events(session_id, id)"
            )
            conn.commit()

    @staticmethod
    def _ensure_agent_memory_columns(cursor: sqlite3.Cursor) -> None:
        cursor.execute("PRAGMA table_info(agent_memory)")
        existing = {str(row[1]) for row in cursor.fetchall()}
        columns = {
            "recent_code_project_path": "TEXT NOT NULL DEFAULT ''",
            "recent_code_project_slug": "TEXT NOT NULL DEFAULT ''",
            "recent_code_delivery_status": "TEXT NOT NULL DEFAULT ''",
            "recent_code_validation_evidence": "TEXT NOT NULL DEFAULT ''",
        }
        for name, ddl in columns.items():
            if name not in existing:
                cursor.execute(f"ALTER TABLE agent_memory ADD COLUMN {name} {ddl}")

    def load(self, session_id: str) -> AgentMemorySnapshot:
        """Load a memory snapshot. Returns an empty snapshot for new sessions."""
        session_id = self.normalize_session_id(session_id)
        with closing(self._get_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM agent_memory WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
        if not row:
            return {
                "session_id": session_id,
                "summary": "",
                "user_preferences": [],
                "recent_topics": [],
                "recent_paper_ids": [],
                "active_skills": [],
                "recent_code_project_path": "",
                "recent_code_project_slug": "",
                "recent_code_delivery_status": "",
                "recent_code_validation_evidence": "",
                "turn_count": 0,
            }
        return {
            "session_id": session_id,
            "summary": row["summary"] or "",
            "user_preferences": _load_json_list(row["user_preferences"]),
            "recent_topics": _load_json_list(row["recent_topics"]),
            "recent_paper_ids": _load_json_list(row["recent_paper_ids"]),
            "active_skills": _load_json_list(row["active_skills"]),
            "recent_code_project_path": row["recent_code_project_path"] or "",
            "recent_code_project_slug": row["recent_code_project_slug"] or "",
            "recent_code_delivery_status": row["recent_code_delivery_status"] or "",
            "recent_code_validation_evidence": row["recent_code_validation_evidence"] or "",
            "turn_count": int(row["turn_count"] or 0),
        }

    def save(self, memory: AgentMemorySnapshot) -> None:
        """Persist a full memory snapshot."""
        session_id = self.normalize_session_id(memory.get("session_id", "default"))
        now = _now_iso()
        with closing(self._get_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO agent_memory
                    (session_id, summary, user_preferences, recent_topics,
                     recent_paper_ids, active_skills, recent_code_project_path,
                     recent_code_project_slug, recent_code_delivery_status,
                     recent_code_validation_evidence, turn_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary = excluded.summary,
                    user_preferences = excluded.user_preferences,
                    recent_topics = excluded.recent_topics,
                    recent_paper_ids = excluded.recent_paper_ids,
                    active_skills = excluded.active_skills,
                    recent_code_project_path = excluded.recent_code_project_path,
                    recent_code_project_slug = excluded.recent_code_project_slug,
                    recent_code_delivery_status = excluded.recent_code_delivery_status,
                    recent_code_validation_evidence = excluded.recent_code_validation_evidence,
                    turn_count = excluded.turn_count,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    memory.get("summary", ""),
                    _json_list(memory.get("user_preferences", [])),
                    _json_list(memory.get("recent_topics", [])),
                    _json_list(memory.get("recent_paper_ids", [])),
                    _json_list(memory.get("active_skills", [])),
                    memory.get("recent_code_project_path", ""),
                    memory.get("recent_code_project_slug", ""),
                    memory.get("recent_code_delivery_status", ""),
                    memory.get("recent_code_validation_evidence", ""),
                    int(memory.get("turn_count", 0) or 0),
                    now,
                    now,
                ),
            )
            conn.commit()

    def update_after_run(
        self,
        session_id: str,
        user_message: str,
        final_answer: str,
        expert_outputs: List[ExpertOutputSnapshot] | None = None,
        active_skills: List[str] | None = None,
    ) -> AgentMemorySnapshot:
        """Update rolling memory after a completed multi-agent run."""
        memory = self.load(session_id)
        expert_outputs = expert_outputs or []

        paper_ids = self._extract_paper_ids(expert_outputs)
        code_project = self._extract_recent_code_project(expert_outputs)
        topic = self._topic_from_message(user_message)
        memory["summary"] = self._roll_summary(
            previous=memory.get("summary", ""),
            user_message=user_message,
            final_answer=final_answer,
            expert_outputs=expert_outputs,
        )
        memory["user_preferences"] = _dedupe_keep_newest(memory.get("user_preferences", []), limit=12)
        memory["recent_topics"] = _dedupe_keep_newest(
            ([topic] if topic else []) + memory.get("recent_topics", []),
            limit=conf.AGENT_MEMORY_MAX_TOPICS,
        )
        memory["recent_paper_ids"] = _dedupe_keep_newest(
            paper_ids + memory.get("recent_paper_ids", []),
            limit=conf.AGENT_MEMORY_MAX_PAPER_IDS,
        )
        memory["active_skills"] = _dedupe_keep_newest(
            list(active_skills or []) + memory.get("active_skills", []),
            limit=12,
        )
        if code_project:
            memory.update(code_project)
        memory["turn_count"] = int(memory.get("turn_count", 0) or 0) + 1
        self.save(memory)
        self._append_event(session_id, "user", user_message, {"paper_ids": paper_ids})
        if final_answer:
            self._append_event(session_id, "assistant", final_answer, {"active_skills": active_skills or []})
        self._trim_events(session_id)
        return memory

    def clear(self, session_id: str) -> None:
        """Remove all persisted memory for a session."""
        session_id = self.normalize_session_id(session_id)
        with closing(self._get_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM agent_memory WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM agent_memory_events WHERE session_id = ?", (session_id,))
            conn.commit()

    @staticmethod
    def normalize_session_id(session_id: str | None) -> str:
        """Keep session ids stable and safe for storage/checkpoint keys."""
        raw = str(session_id or "").strip() or "default"
        cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw)
        return cleaned[:120] or "default"

    @staticmethod
    def format_for_prompt(memory: AgentMemorySnapshot, max_chars: int | None = None) -> str:
        """Convert memory into a compact prompt section."""
        if not memory:
            return ""
        max_chars = conf.AGENT_MEMORY_PROMPT_MAX_CHARS if max_chars is None else max_chars
        lines = []
        summary = str(memory.get("summary", "")).strip()
        if summary:
            lines.append(f"会话摘要: {summary}")
        preferences = memory.get("user_preferences", [])
        if preferences:
            lines.append(f"用户偏好: {', '.join(preferences[:8])}")
        topics = memory.get("recent_topics", [])
        if topics:
            lines.append(f"近期主题: {', '.join(topics[:5])}")
        paper_ids = memory.get("recent_paper_ids", [])
        if paper_ids:
            lines.append(f"近期论文ID: {', '.join(paper_ids[:10])}")
        skills = memory.get("active_skills", [])
        if skills:
            lines.append(f"近期技能: {', '.join(skills[:8])}")
        code_project_path = str(memory.get("recent_code_project_path", "")).strip()
        if code_project_path:
            lines.append(f"最近代码项目路径: {code_project_path}")
        code_project_slug = str(memory.get("recent_code_project_slug", "")).strip()
        if code_project_slug:
            lines.append(f"最近代码项目名: {code_project_slug}")
        code_status = str(memory.get("recent_code_delivery_status", "")).strip()
        if code_status:
            lines.append(f"最近代码交付状态: {code_status}")
        code_evidence = str(memory.get("recent_code_validation_evidence", "")).strip()
        if code_evidence:
            lines.append(f"最近代码验证证据: {code_evidence[:500]}")
        if not lines:
            return ""
        text = "\n".join(lines)
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text

    def _append_event(self, session_id: str, role: str, content: str, metadata: Dict[str, Any]) -> None:
        preview = str(content or "").strip()[:1600]
        if not preview:
            return
        with closing(self._get_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO agent_memory_events
                    (session_id, role, content_preview, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (self.normalize_session_id(session_id), role, preview, json.dumps(metadata, ensure_ascii=False), _now_iso()),
            )
            conn.commit()

    def _trim_events(self, session_id: str) -> None:
        with closing(self._get_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM agent_memory_events
                WHERE session_id = ?
                  AND id NOT IN (
                      SELECT id FROM agent_memory_events
                      WHERE session_id = ?
                      ORDER BY id DESC
                      LIMIT ?
                  )
                """,
                (
                    self.normalize_session_id(session_id),
                    self.normalize_session_id(session_id),
                    conf.AGENT_MEMORY_MAX_EVENTS,
                ),
            )
            conn.commit()

    @staticmethod
    def _extract_paper_ids(expert_outputs: List[ExpertOutputSnapshot]) -> List[str]:
        paper_ids: List[str] = []
        for output in expert_outputs:
            metadata = output.get("metadata", {}) or {}
            raw_ids = metadata.get("paper_ids", [])
            if isinstance(raw_ids, str):
                raw_ids = [raw_ids]
            if isinstance(raw_ids, list):
                paper_ids.extend(str(pid) for pid in raw_ids if str(pid).strip())
        return _dedupe_keep_newest(paper_ids, limit=conf.AGENT_MEMORY_MAX_PAPER_IDS)

    @staticmethod
    def _extract_recent_code_project(expert_outputs: List[ExpertOutputSnapshot]) -> Dict[str, str]:
        for output in reversed(expert_outputs):
            if output.get("expert_name") != "code_builder":
                continue
            metadata = output.get("metadata", {}) or {}
            project_path = str(metadata.get("code_project_path", "")).strip()
            if not project_path:
                continue
            evidence = str(
                metadata.get("validation_evidence", "")
                or metadata.get("tool_trace", "")
                or ""
            ).strip()
            return {
                "recent_code_project_path": project_path,
                "recent_code_project_slug": str(metadata.get("code_project_slug", "")).strip(),
                "recent_code_delivery_status": str(metadata.get("delivery_status", "")).strip(),
                "recent_code_validation_evidence": evidence[:1200],
            }
        return {}

    @staticmethod
    def _topic_from_message(user_message: str) -> str:
        normalized = re.sub(r"\s+", " ", str(user_message or "")).strip()
        return normalized[:120]

    @staticmethod
    def _roll_summary(
        previous: str,
        user_message: str,
        final_answer: str,
        expert_outputs: List[ExpertOutputSnapshot],
    ) -> str:
        experts = []
        for output in expert_outputs:
            name = str(output.get("expert_name", "")).strip()
            if name and name not in experts:
                experts.append(name)
        answer_preview = re.sub(r"\s+", " ", str(final_answer or "")).strip()[:360]
        user_preview = re.sub(r"\s+", " ", str(user_message or "")).strip()[:220]
        turn_line = f"- 用户问: {user_preview}"
        if experts:
            turn_line += f"；参与专家: {', '.join(experts)}"
        if answer_preview:
            turn_line += f"；回复要点: {answer_preview}"

        combined = "\n".join(part for part in [previous.strip(), turn_line] if part)
        if len(combined) <= conf.AGENT_MEMORY_SUMMARY_MAX_CHARS:
            return combined
        return combined[-conf.AGENT_MEMORY_SUMMARY_MAX_CHARS :]


_memory_store: AgentMemoryStore | None = None


def get_agent_memory_store() -> AgentMemoryStore:
    """Return the process-wide memory store singleton."""
    global _memory_store
    if _memory_store is None:
        _memory_store = AgentMemoryStore()
    return _memory_store

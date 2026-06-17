"""Paper ingestion and search orchestration layer."""

import hashlib
import os
import re
import sqlite3
import unicodedata
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from config import conf
from rag.core.logging import get_logger
from rag.storage.sqlite_store import PaperDB

logger = get_logger(__name__)

if TYPE_CHECKING:
    from rag.plugins.pdf_parser import PaperParser
    from rag.storage.vector_store import VectorDB

# Allowed values of the ``on_duplicate`` policy passed to :meth:`PaperManager.ingest_pdf`.
DUPLICATE_POLICIES = ("skip", "replace", "keep_both")

# How many bytes of the file to read at a time when computing SHA256.
_HASH_CHUNK_BYTES = 1 << 16  # 64 KiB


def _canonical_section_name(title: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(title or "unknown").strip().lower()).strip("_")
    return value or "unknown"


class PaperManager:
    """Orchestrates PDF ingestion, metadata storage, and vector search."""

    def __init__(self, enable_hyde: Optional[bool] = None):
        self.db = PaperDB()
        self.vdb: Optional["VectorDB"] = None
        self.parser: Optional["PaperParser"] = None
        self.last_ingested_paper_id: str = ""

        # HyDE is opt-in via config (default on). Expander itself is lazy
        # so constructing PaperManager never triggers an LLM load.
        if enable_hyde is None:
            enable_hyde = bool(getattr(conf, "ENABLE_HYDE", True))
        self._enable_hyde = bool(enable_hyde)
        self._hyde = None

    def _get_vector_db(self) -> "VectorDB":
        if self.vdb is None:
            from rag.storage.vector_store import VectorDB

            self.vdb = VectorDB()
        return self.vdb

    def _get_parser(self):
        if self.parser is None:
            from rag.plugins.pdf_parser import PaperParser

            self.parser = PaperParser(marker_device=conf.PAPER_PARSER_DEVICE)
        return self.parser

    def _get_hyde(self):
        if not self._enable_hyde:
            return None
        if self._hyde is None:
            from rag.plugins.hyde import get_hyde_expander

            self._hyde = get_hyde_expander()
        return self._hyde

    def ingest_pdf(
        self,
        pdf_path,
        paper_id=None,
        tags="",
        on_duplicate: str = "skip",
    ):
        """Parse a PDF and ingest it into the SQLite metadata store and Chroma vector store.

        Duplicate detection runs in two stages:

        1. **Pre-parse**: SHA256 of the file bytes is compared against
           ``papers.content_hash``. This is fast and catches re-imports of the
           same file regardless of filename.
        2. **Post-parse**: DOI (when present) and a normalised title are
           compared against the database. This catches the same paper saved
           under different filenames or paper versions.

        Args:
            pdf_path: Absolute or relative path to the PDF.
            paper_id: Optional explicit id; auto-generated if omitted.
            tags: Free-form tag string forwarded to SQLite.
            on_duplicate: Resolution policy when a duplicate is found:
                ``"skip"`` (default) returns ``(False, msg)`` without writing;
                ``"replace"`` deletes the existing entry and ingests the new
                file; ``"keep_both"`` ingests anyway, leaving both entries.
        """
        if on_duplicate not in DUPLICATE_POLICIES:
            return False, (
                f"Invalid on_duplicate={on_duplicate!r}; expected one of {DUPLICATE_POLICIES}."
            )
        if not os.path.exists(pdf_path):
            return False, f"File not found: {pdf_path}"

        self.last_ingested_paper_id = ""
        filename = os.path.basename(pdf_path)

        stage = "hash_pdf"
        # ------------------------------------------------------------------
        # Stage 1: cheap pre-parse hash check.
        # ------------------------------------------------------------------
        try:
            content_hash = self._compute_content_hash(pdf_path)
        except OSError as exc:
            logger.exception(
                "[paper] ingestion failed stage=%s path=%s error=%s",
                stage,
                pdf_path,
                exc,
            )
            return False, f"Ingestion failed at stage={stage}: {type(exc).__name__}: {exc}"

        pre_dup = self._find_duplicate(content_hash=content_hash)
        if pre_dup is not None:
            decision = self._resolve_duplicate(pre_dup, on_duplicate, filename)
            if decision is not None:
                return decision  # "skip" or invalid path — we're done.

        if not paper_id:
            paper_id = f"local_{uuid.uuid4().hex[:8]}"

        stage = "parse_pdf"
        try:
            # 1. Parse PDF
            parser = self._get_parser()
            parsed_data = parser.process_paper(pdf_path)
            parser_backend = (
                parsed_data.get("parser_backend")
                or (parsed_data.get("meta") or {}).get("parser_backend")
                or "unknown"
            )
            logger.info(
                "[paper] parsed pdf paper_id=%s backend=%s chunks=%s path=%s",
                paper_id,
                parser_backend,
                len(parsed_data.get("chunks", [])),
                pdf_path,
            )
            chunk_records = [
                {**chunk, "chunk_index": idx}
                for idx, chunk in enumerate(parsed_data.get("chunks", []))
                if str(chunk.get("content") or "").strip()
            ]
            chunks = [co["content"] for co in chunk_records]
            if not chunks:
                return False, "Failed to extract text from PDF (possibly image-only PDF)."

            meta = parsed_data.get("meta") or {}
            title = meta.get("title") or filename
            author = meta.get("author") or "Unknown"
            doi = self._normalize_doi(meta.get("doi") or "")
            normalized_title = self._normalize_title(title)
            # Skip title-based dedup when the "title" is just the filename:
            # parsers fall back to filename when no title is detected, and we
            # don't want two unrelated papers to collide on a generic name.
            title_for_dedup = (
                normalized_title
                if normalized_title and normalized_title != self._normalize_title(filename)
                else ""
            )

            # ------------------------------------------------------------------
            # Stage 2: post-parse meta-based check (DOI / normalised title).
            # ------------------------------------------------------------------
            post_dup = self._find_duplicate(doi=doi, normalized_title=title_for_dedup)
            # Avoid re-flagging the same row that pre-stage already cleared.
            if post_dup is not None and pre_dup is not None and post_dup["match"]["id"] == pre_dup["match"]["id"]:
                post_dup = None
            if post_dup is not None:
                decision = self._resolve_duplicate(post_dup, on_duplicate, filename)
                if decision is not None:
                    return decision

            # 2. Store metadata in SQLite
            stage = "sqlite_store"
            self.db.add_paper(
                paper_id=paper_id,
                title=title,
                authors=author,
                abstract=f"The paper has been automatically split into {len(chunks)} text chunks.",
                publish_year=None,
                local_path=pdf_path,
                tags=tags,
                content_hash=content_hash,
                doi=doi,
                normalized_title=normalized_title,
            )
            self.db.replace_paper_chunks(paper_id, chunk_records)

            # 3. Store embeddings in ChromaDB
            stage = "vector_store"
            self._get_vector_db().add_chunks(paper_id, chunk_records)

            self.last_ingested_paper_id = str(paper_id)
            return True, f"Successfully ingested: [{title}] with {len(chunks)} chunks via {parser_backend} parser."

        except Exception as e:
            logger.exception(
                "[paper] ingestion failed stage=%s paper_id=%s path=%s error=%s",
                stage,
                paper_id,
                pdf_path,
                e,
            )
            cleanup_errors = []
            try:
                self.db.delete_paper(paper_id)
            except Exception as cleanup_exc:
                logger.exception(
                    "[paper] ingestion cleanup failed target=sqlite paper_id=%s error=%s",
                    paper_id,
                    cleanup_exc,
                )
                cleanup_errors.append(f"sqlite cleanup: {cleanup_exc}")
            if self.vdb is not None:
                try:
                    self.vdb.delete_paper(paper_id)
                except Exception as cleanup_exc:
                    logger.exception(
                        "[paper] ingestion cleanup failed target=vector paper_id=%s error=%s",
                        paper_id,
                        cleanup_exc,
                    )
                    cleanup_errors.append(f"vector cleanup: {cleanup_exc}")
            cleanup_suffix = f"; cleanup_errors={cleanup_errors}" if cleanup_errors else ""
            return False, f"Ingestion failed at stage={stage}: {type(e).__name__}: {e}{cleanup_suffix}"

    def ingest_directory(
        self,
        directory: str | Path,
        *,
        recursive: bool = False,
        on_duplicate: str = "skip",
        max_files: int = 200,
        dry_run: bool = False,
        tags: str = "Agent-Added",
    ) -> Dict[str, Any]:
        """Import all PDFs from a directory, preserving per-file outcomes.

        The caller is responsible for path boundary checks. This method keeps
        the storage orchestration reusable for CLI/tests/tools without knowing
        about MCP security policy.
        """
        if on_duplicate not in DUPLICATE_POLICIES:
            return {
                "status": "failed",
                "error_code": "INVALID_DUPLICATE_POLICY",
                "message": f"Invalid on_duplicate={on_duplicate!r}; expected one of {DUPLICATE_POLICIES}.",
                "directory": str(directory),
                "recursive": bool(recursive),
                "dry_run": bool(dry_run),
                "max_files": max_files,
                "total_found": 0,
                "processed": 0,
                "imported": 0,
                "skipped": 0,
                "failed": 0,
                "limited": False,
                "results": [],
            }

        try:
            limit = int(max_files)
        except (TypeError, ValueError):
            limit = 200
        limit = max(1, min(limit, 1000))

        root = Path(directory).expanduser().resolve()
        if not root.exists():
            return {
                "status": "failed",
                "error_code": "DIRECTORY_NOT_FOUND",
                "message": f"Directory not found: {root}",
                "directory": str(root),
                "recursive": bool(recursive),
                "dry_run": bool(dry_run),
                "max_files": limit,
                "total_found": 0,
                "processed": 0,
                "imported": 0,
                "skipped": 0,
                "failed": 0,
                "limited": False,
                "results": [],
            }
        if not root.is_dir():
            return {
                "status": "failed",
                "error_code": "DIRECTORY_REQUIRED",
                "message": f"Path is not a directory: {root}",
                "directory": str(root),
                "recursive": bool(recursive),
                "dry_run": bool(dry_run),
                "max_files": limit,
                "total_found": 0,
                "processed": 0,
                "imported": 0,
                "skipped": 0,
                "failed": 0,
                "limited": False,
                "results": [],
            }

        candidates = root.rglob("*") if recursive else root.iterdir()
        pdfs = sorted(
            (path for path in candidates if path.is_file() and path.suffix.lower() == ".pdf"),
            key=lambda path: path.relative_to(root).as_posix().lower(),
        )
        selected = pdfs[:limit]
        results: List[Dict[str, Any]] = []

        if dry_run:
            for path in selected:
                results.append(
                    {
                        "filename": path.name,
                        "relative_path": path.relative_to(root).as_posix(),
                        "status": "found",
                        "message": "PDF would be imported.",
                    }
                )
            return {
                "status": "succeeded",
                "message": f"Found {len(pdfs)} PDF file(s); dry_run selected {len(selected)}.",
                "directory": str(root),
                "recursive": bool(recursive),
                "dry_run": True,
                "max_files": limit,
                "total_found": len(pdfs),
                "processed": 0,
                "imported": 0,
                "skipped": len(selected),
                "failed": 0,
                "limited": len(pdfs) > len(selected),
                "results": results,
            }

        imported = skipped = failed = 0
        for path in selected:
            item = {
                "filename": path.name,
                "relative_path": path.relative_to(root).as_posix(),
            }
            try:
                ok, message = self.ingest_pdf(str(path), tags=tags, on_duplicate=on_duplicate)
            except Exception as exc:
                logger.exception("[paper] batch ingestion exception path=%s error=%s", path, exc)
                ok = False
                message = f"Ingestion failed: {type(exc).__name__}: {exc}"

            item["message"] = message
            if ok:
                imported += 1
                item["status"] = "imported"
                paper_id = str(getattr(self, "last_ingested_paper_id", "") or "")
                if paper_id:
                    item["paper_id"] = paper_id
            elif "Duplicate detected" in str(message):
                skipped += 1
                item["status"] = "skipped"
            else:
                failed += 1
                item["status"] = "failed"
            results.append(item)

        processed = len(selected)
        return {
            "status": "succeeded",
            "message": (
                f"Processed {processed}/{len(pdfs)} PDF file(s): "
                f"imported={imported}, skipped={skipped}, failed={failed}."
            ),
            "directory": str(root),
            "recursive": bool(recursive),
            "dry_run": False,
            "max_files": limit,
            "total_found": len(pdfs),
            "processed": processed,
            "imported": imported,
            "skipped": skipped,
            "failed": failed,
            "limited": len(pdfs) > processed,
            "results": results,
        }

    # ------------------------------------------------------------------
    # Duplicate-detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_content_hash(pdf_path: str) -> str:
        """Streaming SHA256 of the file at ``pdf_path``."""
        h = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(_HASH_CHUNK_BYTES), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Lower-case, strip punctuation/whitespace, keep CJK + alphanumerics.

        The result is suitable for equality-based dedup: papers whose titles
        differ only in case, punctuation or surrounding whitespace will hash
        to the same value.
        """
        if not title:
            return ""
        t = unicodedata.normalize("NFKC", str(title))
        t = t.lower().strip()
        t = re.sub(r"\.pdf$", "", t)
        # Keep word characters and CJK; drop everything else (punct, spaces).
        t = re.sub(r"[^\w\u4e00-\u9fa5]+", "", t, flags=re.UNICODE)
        return t

    @staticmethod
    def _normalize_doi(doi: str) -> str:
        """Trim, lower-case, and strip any URL prefix from a DOI string."""
        if not doi:
            return ""
        d = str(doi).strip().lower()
        d = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", d)
        d = d.rstrip(".,;")
        return d

    def _find_duplicate(
        self,
        *,
        content_hash: str = "",
        doi: str = "",
        normalized_title: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Return the first matching paper row, with the reason it matched.

        Lookup order: ``content_hash`` (strongest) -> ``doi`` -> ``normalized_title``.
        """
        if content_hash:
            row = self.db.get_by_content_hash(content_hash)
            if row:
                return {"reason": "content_hash", "match": row}
        if doi:
            row = self.db.get_by_doi(doi)
            if row:
                return {"reason": "doi", "match": row}
        if normalized_title:
            rows = self.db.get_by_normalized_title(normalized_title)
            if rows:
                return {"reason": "normalized_title", "match": rows[0]}
        return None

    def _resolve_duplicate(
        self,
        dup: Dict[str, Any],
        policy: str,
        new_filename: str,
    ):
        """Apply ``policy`` to a detected duplicate.

        Returns ``(False, msg)`` to short-circuit ingestion when the caller
        should stop, or ``None`` to let ingestion proceed (``replace`` after
        deleting the old row, ``keep_both`` without modification).
        """
        existing = dup["match"]
        existing_id = existing.get("id", "<unknown>")
        existing_title = existing.get("title", "<untitled>")
        reason = dup["reason"]

        if policy == "skip":
            msg = (
                f"Duplicate skipped: '{new_filename}' matches existing "
                f"paper_id={existing_id} ('{existing_title}') by {reason}."
            )
            return False, msg

        if policy == "replace":
            self.delete_paper(existing_id)
            return None

        if policy == "keep_both":
            return None

        # Defensive: unknown policy. ingest_pdf validated this up front, so
        # reaching here would indicate a programming error.
        return False, f"Unknown duplicate policy: {policy!r}"

    def find_duplicate_for_path(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """Public helper: would ingesting ``pdf_path`` collide with an existing paper?

        Computes the content hash without parsing the PDF, so this is cheap
        enough for UI "check before upload" flows. Returns ``None`` if no
        duplicate is found.
        """
        if not os.path.exists(pdf_path):
            return None
        try:
            content_hash = self._compute_content_hash(pdf_path)
        except OSError:
            return None
        return self._find_duplicate(content_hash=content_hash)

    # ------------------------------------------------------------------
    # Bulk dedup utilities for already-polluted databases
    # ------------------------------------------------------------------

    def backfill_metadata(self) -> Dict[str, int]:
        """Recompute hash / normalised title / DOI for legacy rows.

        Rows ingested before the dedup columns existed have ``NULL`` in those
        fields, which means duplicate detection cannot match them. This walks
        the database, re-hashes any file that is still on disk, and re-derives
        a normalised title from the stored title.

        Returns a small report dict: ``{"updated", "missing_files", "scanned"}``.
        """
        rows = self.db.get_all_papers()
        scanned = 0
        updated = 0
        missing_files = 0

        for row in rows:
            scanned += 1
            needs_hash = not row.get("content_hash")
            needs_norm_title = not row.get("normalized_title")
            if not (needs_hash or needs_norm_title):
                continue

            content_hash: Optional[str] = None
            local_path = row.get("local_path") or ""
            if needs_hash:
                if local_path and os.path.exists(local_path):
                    try:
                        content_hash = self._compute_content_hash(local_path)
                    except OSError:
                        content_hash = None
                else:
                    missing_files += 1

            normalized_title: Optional[str] = None
            if needs_norm_title:
                normalized_title = self._normalize_title(row.get("title") or "") or None

            if content_hash is None and normalized_title is None:
                continue

            if self.db.update_dedup_fields(
                paper_id=row["id"],
                content_hash=content_hash,
                normalized_title=normalized_title,
            ):
                updated += 1

        return {"scanned": scanned, "updated": updated, "missing_files": missing_files}

    def deduplicate_database(
        self,
        strategy: str = "report",
        backfill: bool = True,
    ) -> Dict[str, Any]:
        """Find and optionally remove duplicate papers across the whole database.

        Args:
            strategy: ``"report"`` (default) returns the duplicate groups
                without modifying anything. ``"prune"`` keeps the **most
                recently added** paper in each group and deletes the rest from
                both SQLite and ChromaDB.
            backfill: When ``True`` (default), call :meth:`backfill_metadata`
                first so legacy rows participate in dedup grouping.

        Returns:
            Dict with keys ``groups`` (list of duplicate groups), ``pruned``
            (list of deleted paper ids; empty for ``"report"``) and
            ``total_papers`` (size of the database before pruning).
        """
        if strategy not in {"report", "prune"}:
            raise ValueError(f"strategy must be 'report' or 'prune', got {strategy!r}")

        if backfill:
            self.backfill_metadata()

        rows = self.db.get_all_papers()
        groups = self._group_duplicates(rows)

        pruned: List[str] = []
        if strategy == "prune":
            for group in groups:
                # ``members`` is sorted newest-first by add_time (matches
                # get_all_papers ordering); keep index 0, delete the rest.
                for victim in group["members"][1:]:
                    ok, _ = self.delete_paper(victim["id"])
                    if ok:
                        pruned.append(victim["id"])

        return {
            "strategy": strategy,
            "groups": groups,
            "pruned": pruned,
            "total_papers": len(rows),
        }

    @staticmethod
    def _group_duplicates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group ``rows`` by the strongest available equivalence signal.

        Priority of signals: ``content_hash`` > ``doi`` > ``normalized_title``.
        Each row is assigned to a group via union-find so transitive matches
        (e.g. row A shares hash with B, B shares DOI with C) end up together.
        """
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Initialise every row as its own component.
        for row in rows:
            parent[row["id"]] = row["id"]

        # Build buckets per signal and union members within each bucket.
        for signal in ("content_hash", "doi", "normalized_title"):
            buckets: Dict[str, List[str]] = {}
            for row in rows:
                val = row.get(signal)
                if not val:
                    continue
                buckets.setdefault(val, []).append(row["id"])
            for ids in buckets.values():
                if len(ids) < 2:
                    continue
                first = ids[0]
                for other in ids[1:]:
                    union(first, other)

        # Collect components with size > 1 as duplicate groups.
        components: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            root = find(row["id"])
            components.setdefault(root, []).append(row)

        groups: List[Dict[str, Any]] = []
        for members in components.values():
            if len(members) < 2:
                continue
            # Determine which signals tie this group together for the report.
            reasons = []
            for signal in ("content_hash", "doi", "normalized_title"):
                seen = {m.get(signal) for m in members if m.get(signal)}
                if len(seen) == 1:
                    reasons.append(signal)
            groups.append({"reasons": reasons, "members": members})

        # Surface largest groups first - they're usually the most worth fixing.
        groups.sort(key=lambda g: len(g["members"]), reverse=True)
        return groups

    def search_knowledge(
        self,
        query,
        n_results=3,
        use_hyde: Optional[bool] = None,
        retrieval_mode: str = "hybrid",
        rerank: bool = True,
    ):
        """Search vector store and enrich results with SQLite metadata.

        Args:
            query: The user's search query.
            n_results: Number of top chunks to return.
            use_hyde: Override the instance-level HyDE toggle for this call.
                ``None`` uses the manager default (from config).
        """
        embed_query: Optional[str] = None
        should_use_hyde = self._enable_hyde if use_hyde is None else bool(use_hyde)
        if should_use_hyde:
            hyde = self._get_hyde()
            if hyde is not None:
                passage = hyde.expand(query)
                if passage:
                    # Concatenate original query + hypothesis so the dense
                    # embedding captures both the intent and evidence-like
                    # phrasing. BM25 still runs on the raw query.
                    embed_query = f"{query}\n\n{passage}"

        results = self._get_vector_db().search(
            query,
            n_results=n_results,
            embed_query=embed_query,
            mode=retrieval_mode,
            rerank=rerank,
        )
        if not results:
            return []

        final_results = []
        for res in results:
            paper_id = res["paper_id"]
            meta = self.db.get_paper(paper_id) if paper_id else None
            title = meta["title"] if meta else "Unknown title"

            final_results.append(
                {
                    "paper_id": paper_id,
                    "chunk_id": res.get("chunk_id"),
                    "title": title,
                    "score": res.get("rerank_score", res.get("rrf_score", 0.0)),
                    "section_name": res.get("section_name", "unknown"),
                    "section_title": res.get("section_title", "Unknown"),
                    "content": res["content"],
                }
            )

        return final_results

    def get_chunks_for_paper_ids(
        self,
        paper_ids: List[str],
        max_chunks_per_paper: int = 8,
        section: str = "",
    ) -> List[Dict[str, Any]]:
        """Fetch chunks for known local paper IDs without using a free-text query."""
        final_results = []
        for paper_id in paper_ids:
            pid = str(paper_id or "").strip()
            if not pid:
                continue
            meta = self.db.get_paper(pid)
            title = meta["title"] if meta else "Unknown title"
            rows = self.db.get_paper_chunks(pid, limit=max_chunks_per_paper, section=section)
            if not rows and meta:
                rows = self._backfill_chunks_from_pdf(pid, meta, max_chunks_per_paper, section=section)
            if not rows:
                continue
            for row in rows:
                final_results.append(
                    {
                        "paper_id": pid,
                        "chunk_id": f"{pid}:{row.get('chunk_index', 0)}",
                        "title": title,
                        "score": 1.0,
                        "section_name": row.get("section_name") or "unknown",
                        "section_title": row.get("section_title") or "Unknown",
                        "content": row["content"],
                    }
                )
        return final_results

    def _backfill_chunks_from_pdf(
        self,
        paper_id: str,
        meta: Dict[str, Any],
        limit: int,
        section: str = "",
    ) -> List[Dict[str, Any]]:
        local_path = str(meta.get("local_path") or "")
        if not local_path or not os.path.exists(local_path):
            return []
        try:
            import fitz
        except ImportError:
            return []

        try:
            texts: List[str] = []
            with fitz.open(local_path) as doc:
                for page in doc:
                    text = page.get_text("text").strip()
                    if text:
                        texts.append(text)
            chunks = self._split_text_chunks("\n\n".join(texts), section_title="FullText")
            if not chunks:
                return []
            self.db.replace_paper_chunks(paper_id, chunks)
            rows = self.db.get_paper_chunks(paper_id, limit=limit, section=section)
            return rows
        except Exception as exc:
            logger.warning("[paper] pdf chunk backfill failed paper_id=%s error=%s", paper_id, exc)
            return []

    def list_all(self):
        papers = self.db.get_all_papers()
        for paper in papers:
            paper["sections"] = self.db.get_paper_sections(paper["id"])
        return papers

    def delete_paper(self, paper_id):
        if not paper_id:
            return False, "paper_id is required"
        self.db.delete_paper(paper_id)
        if self.vdb is not None:
            self.vdb.delete_paper(paper_id)
        return True, f"Paper {paper_id} has been removed from the local metadata store."

    @staticmethod
    def _split_text_chunks(
        text: str,
        chunk_chars: int = 2200,
        overlap: int = 200,
        section_title: str = "FullText",
    ) -> List[Dict[str, Any]]:
        normalized = re.sub(r"\s+\n", "\n", str(text or "")).strip()
        if not normalized:
            return []
        chunks: List[Dict[str, Any]] = []
        start = 0
        while start < len(normalized):
            end = min(len(normalized), start + chunk_chars)
            if end < len(normalized):
                boundary = normalized.rfind("\n\n", start, end)
                if boundary > start + chunk_chars // 2:
                    end = boundary
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(
                    {
                        "content": chunk,
                        "chunk_type": "text_chunk",
                        "section_name": _canonical_section_name(section_title),
                        "section_title": section_title,
                    }
                )
            if end >= len(normalized):
                break
            start = max(end - overlap, start + 1)
        return chunks



def get_chunks_for_paper_ids_readonly(
    paper_ids: List[str],
    max_chunks_per_paper: int = 8,
    section: str = "",
) -> List[Dict[str, Any]]:
    """Read paper chunks without opening SQLite/Chroma in writable mode.

    This is a recovery-safe path for MCP clients on Windows when a stale
    SQLite journal prevents normal database initialization.
    """
    db_path = os.path.join(conf.DB_DIR, "papers.db")
    if not os.path.exists(db_path):
        return []

    uri_path = Path(db_path).resolve().as_posix()
    uri = f"file:{uri_path}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        has_chunk_table = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_chunks'"
            ).fetchone()
        )
        chunk_cols = set()
        if has_chunk_table:
            chunk_cols = {row["name"] for row in conn.execute("PRAGMA table_info(paper_chunks)").fetchall()}
        has_section_columns = {"content", "section_name", "section_title", "chunk_type"}.issubset(chunk_cols)
        rows = []
        for paper_id in paper_ids:
            pid = str(paper_id or "").strip()
            if not pid:
                continue
            row = conn.execute(
                "SELECT id, title, local_path FROM papers WHERE id = ?",
                (pid,),
            ).fetchone()
            if not row:
                continue
            meta = dict(row)
            if has_chunk_table and has_section_columns:
                params: List[Any] = [pid]
                where = "paper_id = ?"
                if section:
                    where += " AND (section_name LIKE ? OR section_title LIKE ?)"
                    like = f"%{section}%"
                    params.extend([like, like])
                params.append(max(1, int(max_chunks_per_paper)))
                cached_chunks = conn.execute(
                    f"""
                    SELECT chunk_index, content, section_name, section_title, chunk_type
                    FROM paper_chunks
                    WHERE {where}
                    ORDER BY chunk_index ASC
                    LIMIT ?
                    """,
                    params,
                ).fetchall()
                if cached_chunks:
                    for chunk_row in cached_chunks:
                        rows.append(
                            {
                                **meta,
                                "chunk_index": chunk_row["chunk_index"],
                                "chunk_content": chunk_row["content"],
                                "section_name": chunk_row["section_name"],
                                "section_title": chunk_row["section_title"],
                                "chunk_type": chunk_row["chunk_type"],
                            }
                        )
                    continue
            rows.append(meta)

    final: List[Dict[str, Any]] = []
    found_paper = bool(rows)
    for row in rows:
        pid = row["id"]
        title = row.get("title") or "Unknown title"
        if row.get("chunk_content"):
            chunks = [
                {
                    "content": row["chunk_content"],
                    "section_name": row.get("section_name") or "unknown",
                    "section_title": row.get("section_title") or "Unknown",
                    "chunk_type": row.get("chunk_type") or "text_chunk",
                }
            ]
        else:
            chunks = _extract_pdf_chunks_readonly(row.get("local_path") or "", max_chunks_per_paper, section=section)
        for chunk in chunks:
            final.append(
                {
                    "paper_id": pid,
                    "chunk_id": f"{pid}:{row.get('chunk_index', 0)}",
                    "title": title,
                    "score": 1.0,
                    "section_name": chunk.get("section_name", "unknown"),
                    "section_title": chunk.get("section_title", "Unknown"),
                    "content": chunk["content"],
                }
            )
    if not final and found_paper:
        return [
            {
                "paper_id": row["id"],
                "title": row.get("title") or "Unknown title",
                "score": 1.0,
                "section_name": "not_found",
                "section_title": "Section not found",
                "content": f"No chunks matched section filter: {section}",
                "_section_miss": True,
            }
            for row in rows[:1]
        ]
    return final


def list_papers_readonly(query: str = "", limit: int = 50) -> List[Dict[str, Any]]:
    db_path = os.path.join(conf.DB_DIR, "papers.db")
    if not os.path.exists(db_path):
        return []

    uri_path = Path(db_path).resolve().as_posix()
    uri = f"file:{uri_path}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        has_chunk_table = bool(
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_chunks'"
            ).fetchone()
        )
        chunk_cols = set()
        if has_chunk_table:
            chunk_cols = {row["name"] for row in conn.execute("PRAGMA table_info(paper_chunks)").fetchall()}
        has_sections = {"section_name", "section_title"}.issubset(chunk_cols)

        params: List[Any] = []
        where = ""
        if query:
            where = "WHERE title LIKE ? OR authors LIKE ? OR abstract LIKE ? OR tags LIKE ?"
            like = f"%{query}%"
            params.extend([like, like, like, like])
        params.append(max(1, int(limit)))
        rows = conn.execute(
            f"""
            SELECT id, title, authors, publish_year, tags, local_path,
                   content_hash, doi, normalized_title, add_time
            FROM papers
            {where}
            ORDER BY add_time DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        papers = [dict(row) for row in rows]
        if has_sections:
            for paper in papers:
                section_rows = conn.execute(
                    """
                    SELECT section_title, section_name, COUNT(*) AS chunk_count
                    FROM paper_chunks
                    WHERE paper_id = ?
                    GROUP BY section_title, section_name
                    ORDER BY MIN(chunk_index)
                    """,
                    (paper["id"],),
                ).fetchall()
                paper["sections"] = [
                    f"{(row['section_title'] or row['section_name'] or 'Unknown')} ({row['chunk_count']})"
                    for row in section_rows
                ]
        else:
            for paper in papers:
                paper["sections"] = _infer_pdf_sections_readonly(paper.get("local_path") or "")[:12]
        return papers


def _infer_pdf_sections_readonly(pdf_path: str) -> List[str]:
    chunks = _extract_pdf_chunks_readonly(pdf_path, limit=200)
    seen: List[str] = []
    for chunk in chunks:
        title = chunk.get("section_title") or ""
        if title and title not in seen:
            seen.append(title)
    return seen


def _extract_pdf_chunks_readonly(pdf_path: str, limit: int, section: str = "") -> List[Dict[str, Any]]:
    if not pdf_path or not os.path.exists(pdf_path):
        return []
    try:
        import fitz
    except ImportError:
        return []

    pages: List[str] = []
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text = page.get_text("text").strip()
                if text:
                    pages.append(text)
    except Exception as exc:
        logger.warning("[paper] readonly pdf extraction failed path=%s error=%s", pdf_path, exc)
        return []

    chunks = _split_pdf_text_into_section_chunks("\n\n".join(pages))
    if section:
        needle = section.lower()
        chunks = [
            chunk
            for chunk in chunks
            if needle in chunk.get("section_name", "").lower()
            or needle in chunk.get("section_title", "").lower()
        ]
    return chunks[: max(1, int(limit))]


def _split_pdf_text_into_section_chunks(text: str) -> List[Dict[str, Any]]:
    normalized = re.sub(r"\n{3,}", "\n\n", str(text or "")).strip()
    if not normalized:
        return []
    section_pattern = re.compile(
        r"(?im)^(?:[IVXLC]+\.|\d+\.?)\s+"
        r"(Abstract|Introduction|Related Work|System Model|Proposed Method|Method|Methodology|"
        r"Receiver Design|Simulation|Simulation Results|Experiment|Experiments|Results|Conclusion|References)\b.*$"
    )
    matches = list(section_pattern.finditer(normalized))
    if not matches:
        return PaperManager._split_text_chunks(normalized, section_title="FullText")

    chunks: List[Dict[str, Any]] = []
    first_start = matches[0].start()
    if first_start > 0:
        chunks.extend(PaperManager._split_text_chunks(normalized[:first_start], section_title="FrontMatter"))
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        title = match.group(0).strip()
        body = normalized[start:end].strip()
        chunks.extend(PaperManager._split_text_chunks(body, section_title=title))
    return chunks

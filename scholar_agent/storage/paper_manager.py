"""Paper ingestion and search orchestration layer."""

import glob
import hashlib
import os
import re
import time
import unicodedata
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from scholar_agent.config import conf
from scholar_agent.core.logging import get_logger
from scholar_agent.storage.sqlite_store import PaperDB

logger = get_logger(__name__)

if TYPE_CHECKING:
    from scholar_agent.plugins.pdf_parser import PaperParser
    from scholar_agent.storage.vector_store import VectorDB

# Allowed values of the ``on_duplicate`` policy passed to :meth:`PaperManager.ingest_pdf`.
DUPLICATE_POLICIES = ("skip", "replace", "keep_both")

# How many bytes of the file to read at a time when computing SHA256.
_HASH_CHUNK_BYTES = 1 << 16  # 64 KiB


class PaperManager:
    """Orchestrates PDF ingestion, metadata storage, and vector search."""

    def __init__(self, enable_hyde: Optional[bool] = None):
        self.db = PaperDB()
        self.vdb: Optional["VectorDB"] = None
        self.parser: Optional["PaperParser"] = None

        # HyDE is opt-in via config (default on). Expander itself is lazy
        # so constructing PaperManager never triggers an LLM load.
        if enable_hyde is None:
            enable_hyde = bool(getattr(conf, "ENABLE_HYDE", True))
        self._enable_hyde = bool(enable_hyde)
        self._hyde = None

    def _get_vector_db(self) -> "VectorDB":
        if self.vdb is None:
            from scholar_agent.storage.vector_store import VectorDB

            self.vdb = VectorDB()
        return self.vdb

    def _get_parser(self):
        if self.parser is None:
            from scholar_agent.plugins.pdf_parser import PaperParser

            self.parser = PaperParser(marker_device=conf.PAPER_PARSER_DEVICE)
        return self.parser

    def _get_hyde(self):
        if not self._enable_hyde:
            return None
        if self._hyde is None:
            from scholar_agent.plugins.hyde import get_hyde_expander

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
           under different filenames or arxiv versions.

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

        filename = os.path.basename(pdf_path)

        # ------------------------------------------------------------------
        # Stage 1: cheap pre-parse hash check.
        # ------------------------------------------------------------------
        try:
            content_hash = self._compute_content_hash(pdf_path)
        except OSError as exc:
            return False, f"Failed to read file for hashing: {exc}"

        pre_dup = self._find_duplicate(content_hash=content_hash)
        if pre_dup is not None:
            decision = self._resolve_duplicate(pre_dup, on_duplicate, filename)
            if decision is not None:
                return decision  # "skip" or invalid path — we're done.

        if not paper_id:
            paper_id = f"local_{uuid.uuid4().hex[:8]}"
        logger.info(f"[paper] ingest start filename={filename} paper_id={paper_id}")

        try:
            # 1. Parse PDF
            parser = self._get_parser()
            parsed_data = parser.process_paper(pdf_path)
            chunks = [co["content"] for co in parsed_data.get("chunks", [])]
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

            # 3. Store embeddings in ChromaDB
            self._get_vector_db().add_chunks(paper_id, chunks)

            return True, f"Successfully ingested: [{title}] with {len(chunks)} chunks."

        except Exception as e:
            self.db.delete_paper(paper_id)
            if self.vdb is not None:
                self.vdb.delete_paper(paper_id)
            return False, f"Ingestion failed: {e}"

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
            logger.info(f"[paper] {msg}")
            return False, msg

        if policy == "replace":
            logger.info(
                f"[paper] replacing existing paper_id={existing_id} "
                f"(matched by {reason}) with new file {new_filename!r}"
            )
            self.delete_paper(existing_id)
            return None

        if policy == "keep_both":
            logger.info(
                f"[paper] keep_both policy: ingesting {new_filename!r} alongside "
                f"existing paper_id={existing_id} (matched by {reason})"
            )
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

        logger.info(
            f"[paper] backfill done scanned={scanned} updated={updated} missing_files={missing_files}"
        )
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

        logger.info(
            f"[paper] dedup strategy={strategy} groups={len(groups)} pruned={len(pruned)}"
        )
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

    def search_knowledge(self, query, n_results=3, use_hyde: Optional[bool] = None):
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

        results = self._get_vector_db().search(query, n_results=n_results, embed_query=embed_query)
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
                    "title": title,
                    "score": res.get("rerank_score", res.get("rrf_score", 0.0)),
                    "content": res["content"],
                }
            )

        return final_results

    def get_chunks_for_paper_ids(
        self,
        paper_ids: List[str],
        max_chunks_per_paper: int = 8,
    ) -> List[Dict[str, Any]]:
        """Fetch chunks for known local paper IDs without using a free-text query."""
        results = self._get_vector_db().get_chunks_by_paper_ids(
            paper_ids,
            max_chunks_per_paper=max_chunks_per_paper,
        )
        final_results = []
        for res in results:
            paper_id = res["paper_id"]
            meta = self.db.get_paper(paper_id) if paper_id else None
            title = meta["title"] if meta else "Unknown title"
            final_results.append(
                {
                    "paper_id": paper_id,
                    "title": title,
                    "score": 1.0,
                    "content": res["content"],
                }
            )
        return final_results

    def list_all(self):
        return self.db.get_all_papers()

    def delete_paper(self, paper_id):
        if not paper_id:
            return False, "paper_id is required"
        self.db.delete_paper(paper_id)
        self._get_vector_db().delete_paper(paper_id)
        return True, f"Paper {paper_id} has been completely removed."

# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    from scholar_agent.config import conf

    conf.check_config()
    manager = PaperManager()
    pdf_pattern = os.path.join(conf.PAPERS_DIR, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)

    if not pdf_files:
        print(f"\nNo PDF files found in {conf.PAPERS_DIR}.")
        print("Copy some academic PDF files to this directory before running the test.")
    else:
        print(f"\nFound {len(pdf_files)} PDF files. Starting ingestion...")
        print("=" * 60)

        for pdf_path in pdf_files:
            start_time = time.time()
            success, msg = manager.ingest_pdf(pdf_path)
            cost_time = time.time() - start_time
            print(f"{msg} (time: {cost_time:.2f}s)\n")

        all_papers = manager.list_all()
        print("=" * 60)
        print(f"Total papers in database: {len(all_papers)}")
        for p in all_papers:
            short_title = p["title"][:40] + "..." if len(p["title"]) > 40 else p["title"]
            print(f"  - ID: {p['id']} | Title: {short_title} | Tags: {p['tags']}")
        print("=" * 60)

        # Dedup scan: should report 0 groups on a clean ingestion run.
        dedup_report = manager.deduplicate_database(strategy="report")
        print(
            f"Duplicate scan: {len(dedup_report['groups'])} group(s) found "
            f"across {dedup_report['total_papers']} papers."
        )
        for grp in dedup_report["groups"]:
            ids = ", ".join(m["id"] for m in grp["members"])
            print(f"  reasons={grp['reasons']} members=[{ids}]")
        print("=" * 60)

        query = "sparse code multiple access"
        start_time = time.time()
        results = manager.search_knowledge(query, n_results=3)
        cost_time = time.time() - start_time

        if results:
            print(f"\nSearch completed (time: {cost_time:.2f}s), top 3 results:")
            for i, res in enumerate(results):
                short_title = res["title"][:40] + "..." if len(res["title"]) > 40 else res["title"]
                print(f"\n[{i + 1}] Source: 《{short_title}》 (score: {res['score']:.4f})")
                preview = res["content"].replace("\n", " ")[:150]
                print(f"    Preview: {preview}...")
        else:
            print("\nNo relevant content found in vector store.")

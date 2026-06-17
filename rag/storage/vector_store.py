"""Vector database with ChromaDB (dense) + BM25 (sparse) + BGE reranker.

Retrieval pipeline:
    1. Dense semantic search via BGE-M3 over ChromaDB.
    2. Sparse keyword search via BM25 with bilingual tokenization.
    3. Reciprocal Rank Fusion (RRF) of the two candidate lists.
    4. Cross-encoder rerank (bge-reranker-v2-m3) on the fused top-k.
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

if __package__ is None:
    _proj = Path(__file__).resolve().parent
    while _proj and not (_proj / "pyproject.toml").exists():
        _proj = _proj.parent
    if _proj and str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))

import chromadb

from config import conf
from rag.core.logging import get_logger
from rag.storage import model_manager
from rag.storage.migrations import CHROMA_SCHEMA_VERSION, ensure_chroma_schema

logger = get_logger(__name__)

# Reciprocal Rank Fusion constant. k=60 is the canonical value from the
# original TREC paper (Cormack et al., 2009) and is used by Elasticsearch,
# Vespa, Weaviate and LlamaIndex by default.
RRF_K = 60


class VectorDB:
    """Hybrid search combining dense embeddings (ChromaDB + BGE-M3), BM25 keyword search,
    and cross-encoder reranking (bge-reranker-v2-m3)."""

    def __init__(self, collection_name="papers"):
        from FlagEmbedding import BGEM3FlagModel
        from rank_bm25 import BM25Okapi

        chroma_path = os.path.join(conf.DB_DIR, "chroma")
        self._bm25_cls = BM25Okapi
        embed_model_path = model_manager.ensure_bge_embedding_model()
        reranker_model_path = conf.BGE_RERANKER_MODEL_PATH
        self.reranker_model_path = reranker_model_path
        self.reranker = None

        os.makedirs(conf.DB_DIR, exist_ok=True)
        self.schema_metadata = ensure_chroma_schema(
            chroma_path,
            {
                "collection_name": collection_name,
                "embedding_model_path": str(embed_model_path),
                "reranker_model_path": str(self.reranker_model_path),
            },
        )
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # BM25 in-memory corpus caches. Rebuilt on add/delete so the sparse
        # index never drifts from ChromaDB.
        self._bm25_ids: List[str] = []
        self._bm25_chunks: List[str] = []
        self._bm25_metadatas: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._rebuild_bm25()

        try:
            self.ef = BGEM3FlagModel(model_name_or_path=embed_model_path)
        except Exception as e:
            logger.warning("[vector] model load failed: %s", e)
            logger.warning("[vector] expected embedding model at: %s", embed_model_path)
            raise

    @property
    def supported_schema_version(self) -> int:
        return CHROMA_SCHEMA_VERSION

    def _get_reranker(self):
        if self.reranker is None:
            from FlagEmbedding import FlagReranker

            model_path = model_manager.ensure_bge_reranker_model()
            self.reranker_model_path = model_path
            self.reranker = FlagReranker(model_name_or_path=model_path, use_fp16=True)
        return self.reranker

    @staticmethod
    def _bilingual_tokenizer(text1: str) -> List[str]:
        stop_words = {
            "的", "了", "和", "是", "在", "我", "有", "就", "都", "而", "及", "与",
            "the", "a", "an", "and", "or", "is", "are", "was", "were", "in", "on", "at",
            "to", "of", "for", "with", "this", "that", "it", "you", "me",
        }
        text = re.sub(r"\s+", " ", text1.strip())
        tokens = []
        parts = re.findall(r"[\u4e00-\u9fa5]+|[a-zA-Z0-9]+", text)

        for part in parts:
            if re.match(r"^[\u4e00-\u9fa5]+$", part):
                import jieba

                tokens.extend(jieba.lcut(part))
            else:
                tokens.append(part.lower())

        return [t.strip() for t in tokens if len(t.strip()) > 1 and t.strip() not in stop_words]

    def _rebuild_bm25(self) -> None:
        """Refresh the in-memory BM25 index from the current ChromaDB state."""
        snapshot = self.collection.get()
        self._bm25_ids = list(snapshot.get("ids") or [])
        self._bm25_chunks = list(snapshot.get("documents") or [])
        self._bm25_metadatas = list(snapshot.get("metadatas") or [])

        if not self._bm25_chunks:
            self.bm25 = None
            return

        tokenized = [self._bilingual_tokenizer(chunk) for chunk in self._bm25_chunks]
        self.bm25 = self._bm25_cls(tokenized)

    def add_chunks(self, paper_id, chunks):
        if not chunks:
            return

        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                content = str(chunk.get("content") or "").strip()
                if not content:
                    continue
                documents.append(content)
                metadatas.append(
                    {
                        "paper_id": paper_id,
                        "chunk_index": int(chunk.get("chunk_index", idx) or idx),
                        "chunk_type": str(chunk.get("chunk_type") or "text_chunk"),
                        "section_name": str(chunk.get("section_name") or "unknown"),
                        "section_title": str(chunk.get("section_title") or "Unknown"),
                    }
                )
            else:
                content = str(chunk or "").strip()
                if not content:
                    continue
                documents.append(content)
                metadatas.append(
                    {
                        "paper_id": paper_id,
                        "chunk_index": idx,
                        "chunk_type": "text_chunk",
                        "section_name": "unknown",
                        "section_title": "Unknown",
                    }
                )
        if not documents:
            return

        ids = [f"{paper_id}_chunk_{meta['chunk_index']}" for meta in metadatas]
        embeddings = self.ef.encode(documents)

        self.collection.add(
            embeddings=embeddings["dense_vecs"],
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        self._rebuild_bm25()

    def search(
        self,
        query_text: str,
        n_results: int = 100,
        hybrid: bool = True,
        embed_query: Optional[str] = None,
        mode: str = "",
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks with a selected retrieval strategy.

        Args:
            query_text: The user's original query; used for BM25 tokenisation
                and for the cross-encoder reranker.
            n_results: Final number of chunks to return.
            hybrid: Backward-compatible selector. If ``False`` and ``mode`` is
                empty, dense retrieval is used.
            embed_query: Optional alternate string used *only* for the dense
                embedding step. Allows callers (e.g. HyDE) to retrieve with an
                enriched passage while still reranking against the original
                short question.
            mode: One of ``hybrid``, ``dense``, ``bm25``. Empty preserves the
                previous ``hybrid`` boolean behavior.
            rerank: Whether to apply the cross-encoder reranker to candidates.
        """
        n_results = max(1, int(n_results))
        mode = (mode or ("hybrid" if hybrid else "dense")).strip().lower()
        if mode not in {"hybrid", "dense", "bm25"}:
            raise ValueError(f"Unsupported retrieval mode: {mode!r}")

        # Retrieve a wider pool from each ranker so RRF has enough signal.
        candidate_pool = max(n_results * 10, 40)

        if mode == "dense":
            dense_ranked = self._dense_search(embed_query or query_text, candidate_pool)
            return self._rerank_or_slice(query_text, dense_ranked, n_results, rerank)

        if mode == "bm25":
            sparse_ranked = self._bm25_search(query_text, candidate_pool)
            return self._rerank_or_slice(query_text, sparse_ranked, n_results, rerank)

        dense_ranked = self._dense_search(embed_query or query_text, candidate_pool)
        sparse_ranked = self._bm25_search(query_text, candidate_pool)

        fused = self._rrf_fuse([dense_ranked, sparse_ranked], k=RRF_K)
        # Cap the rerank input to keep latency bounded; 40 pairs is <1s on GPU.
        rerank_pool = min(40, max(n_results * 6, 24))
        return self._rerank_or_slice(query_text, fused[:rerank_pool], n_results, rerank)

    def _dense_search(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        encode_result = self.ef.encode(query_text)
        query_embedding = (
            encode_result["dense_vecs"] if isinstance(encode_result, dict) else encode_result
        )
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        ranked: List[Dict[str, Any]] = []
        for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            ranked.append(
                {
                    "chunk_id": chunk_id,
                    "paper_id": (meta or {}).get("paper_id"),
                    "section_name": (meta or {}).get("section_name", "unknown"),
                    "section_title": (meta or {}).get("section_title", "Unknown"),
                    "chunk_type": (meta or {}).get("chunk_type", "text_chunk"),
                    "content": doc,
                    "distance": dist,
                    "source": "dense",
                }
            )
        return ranked

    def _bm25_search(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        if self.bm25 is None or not self._bm25_chunks:
            return []
        query_keywords = self._bilingual_tokenizer(query_text)
        if not query_keywords:
            return []

        doc_scores = self.bm25.get_scores(query_keywords)
        # argsort descending; BM25Okapi returns a numpy array when installed
        # with numpy, and a plain list otherwise — handle both.
        indexed = sorted(
            range(len(doc_scores)),
            key=lambda i: float(doc_scores[i]),
            reverse=True,
        )

        ranked: List[Dict[str, Any]] = []
        for idx in indexed[:top_k]:
            score = float(doc_scores[idx])
            if score <= 0:
                break  # remaining docs have no keyword overlap at all
            meta = self._bm25_metadatas[idx] or {}
            ranked.append(
                {
                    "chunk_id": self._bm25_ids[idx],
                    "paper_id": meta.get("paper_id"),
                    "section_name": meta.get("section_name", "unknown"),
                    "section_title": meta.get("section_title", "Unknown"),
                    "chunk_type": meta.get("chunk_type", "text_chunk"),
                    "content": self._bm25_chunks[idx],
                    "bm25_score": score,
                    "source": "bm25",
                }
            )
        return ranked

    @staticmethod
    def _rrf_fuse(
        ranked_lists: Iterable[List[Dict[str, Any]]],
        k: int = RRF_K,
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion over multiple ranked lists.

        score(d) = Σ_r 1 / (k + rank_r(d))   with rank starting at 1.

        Items are deduplicated by ``chunk_id`` (falling back to a content
        prefix if a ranker did not expose ids).
        """
        scores: Dict[str, float] = {}
        items: Dict[str, Dict[str, Any]] = {}

        for ranked in ranked_lists:
            for rank, item in enumerate(ranked, start=1):
                key = item.get("chunk_id") or f"content::{(item.get('content') or '')[:96]}"
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
                if key not in items:
                    # Store the first-seen copy; later occurrences only
                    # contribute to the fused score, not to item metadata.
                    items[key] = dict(item)

        fused = []
        for key, score in scores.items():
            enriched = items[key]
            enriched["rrf_score"] = score
            fused.append(enriched)
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused

    def _rerank(
        self,
        query_text: str,
        candidates: List[Dict[str, Any]],
        n_results: int,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        pairs = [[query_text, res["content"]] for res in candidates]
        try:
            scores = self._get_reranker().compute_score(pairs)
        except Exception as exc:
            logger.warning("[vector] reranker unavailable, returning pre-rerank order: %s", exc)
            return candidates[:n_results]
        if isinstance(scores, (int, float)):
            scores = [float(scores)]
        for i, s in enumerate(scores):
            candidates[i]["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:n_results]

    def _rerank_or_slice(
        self,
        query_text: str,
        candidates: List[Dict[str, Any]],
        n_results: int,
        rerank: bool,
    ) -> List[Dict[str, Any]]:
        if rerank:
            return self._rerank(query_text, candidates, n_results)
        return candidates[:n_results]

    def delete_paper(self, paper_id):
        self.collection.delete(where={"paper_id": paper_id})
        self._rebuild_bm25()

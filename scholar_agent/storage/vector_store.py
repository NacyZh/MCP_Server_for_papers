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
import jieba
from rank_bm25 import BM25Okapi

from scholar_agent.config import conf
from scholar_agent.core.logging import get_logger

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

        chroma_path = os.path.join(conf.DB_DIR, "chroma")
        embed_model_path = conf.BGE_M3_MODEL_PATH
        reranker_model_path = conf.BGE_RERANKER_MODEL_PATH
        self.reranker_model_path = reranker_model_path
        self.reranker = None

        os.makedirs(conf.DB_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        os.makedirs(os.path.dirname(embed_model_path), exist_ok=True)
        if not self._model_source_exists(embed_model_path):
            raise FileNotFoundError(
                "BGE-M3 embedding model not found. "
                f"Expected local path: {embed_model_path}. "
                "Set BGE_M3_MODEL_PATH in .env to the local model directory before importing PDFs."
            )

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
            logger.info(f"[vector] model load failed: {e}")
            logger.info(f"[vector] expected embedding model at: {embed_model_path}")
            raise

        logger.info(f"[vector] chroma ready collection={collection_name}")

    def _get_reranker(self):
        if self.reranker is None:
            from FlagEmbedding import FlagReranker

            if not self._model_source_exists(self.reranker_model_path):
                raise FileNotFoundError(
                    "BGE reranker model not found. "
                    f"Expected local path: {self.reranker_model_path}. "
                    "Set BGE_RERANKER_MODEL_PATH in .env to enable reranking."
                )

            self.reranker = FlagReranker(model_name_or_path=self.reranker_model_path, use_fp16=True)
        return self.reranker

    @staticmethod
    def _model_source_exists(model_name_or_path: str) -> bool:
        path = Path(str(model_name_or_path)).expanduser()
        if path.is_absolute() or os.sep in str(model_name_or_path) or (os.altsep and os.altsep in str(model_name_or_path)):
            return path.exists()
        return True

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
        self.bm25 = BM25Okapi(tokenized)

    def add_chunks(self, paper_id, chunks):
        if not chunks:
            return

        ids = [f"{paper_id}_chunk_{chunk_id}" for chunk_id in range(len(chunks))]
        metadatas = [{"paper_id": paper_id, "chunk_index": chunk_idx} for chunk_idx in range(len(chunks))]
        embeddings = self.ef.encode(chunks)

        logger.info(f"[vector] adding chunks count={len(chunks)}")
        self.collection.add(
            embeddings=embeddings["dense_vecs"],
            documents=chunks,
            metadatas=metadatas,
            ids=ids,
        )
        self._rebuild_bm25()
        logger.info(f"[vector] add done paper_id={paper_id}")

    def search(
        self,
        query_text: str,
        n_results: int = 100,
        hybrid: bool = True,
        embed_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieve-then-rerank.

        Args:
            query_text: The user's original query; used for BM25 tokenisation
                and for the cross-encoder reranker.
            n_results: Final number of chunks to return.
            hybrid: If ``False``, skips BM25 and RRF and returns pure dense
                results (still reranked by the cross-encoder).
            embed_query: Optional alternate string used *only* for the dense
                embedding step. Allows callers (e.g. HyDE) to retrieve with an
                enriched passage while still reranking against the original
                short question.
        """
        logger.info(
            f"[vector] search query='{query_text[:80]}' n_results={n_results} hybrid={hybrid}"
            + (" hyde=on" if embed_query and embed_query != query_text else "")
        )

        n_results = max(1, int(n_results))
        # Retrieve a wider pool from each ranker so RRF has enough signal.
        candidate_pool = max(n_results * 10, 40)

        dense_ranked = self._dense_search(embed_query or query_text, candidate_pool)

        if not hybrid:
            return self._rerank(query_text, dense_ranked, n_results)

        sparse_ranked = self._bm25_search(query_text, candidate_pool)

        fused = self._rrf_fuse([dense_ranked, sparse_ranked], k=RRF_K)
        # Cap the rerank input to keep latency bounded; 40 pairs is <1s on GPU.
        rerank_pool = min(40, max(n_results * 6, 24))
        return self._rerank(query_text, fused[:rerank_pool], n_results)

    def get_chunks_by_paper_ids(
        self,
        paper_ids: List[str],
        max_chunks_per_paper: int = 8,
    ) -> List[Dict[str, Any]]:
        """Return stored chunks for specific local paper IDs without semantic search."""
        chunks: List[Dict[str, Any]] = []
        for paper_id in paper_ids:
            pid = str(paper_id or "").strip()
            if not pid:
                continue
            result = self.collection.get(where={"paper_id": pid})
            ids = result.get("ids") or []
            documents = result.get("documents") or []
            metadatas = result.get("metadatas") or []
            paper_chunks: List[Dict[str, Any]] = []
            for chunk_id, doc, meta in zip(ids, documents, metadatas):
                meta = meta or {}
                paper_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "paper_id": meta.get("paper_id") or pid,
                        "chunk_index": meta.get("chunk_index", 0),
                        "content": doc,
                        "source": "paper_id",
                    }
                )
            paper_chunks.sort(key=lambda item: int(item.get("chunk_index") or 0))
            chunks.extend(paper_chunks[:max_chunks_per_paper])
        return chunks

    # ------------------------------------------------------------------
    # Retrieval stages
    # ------------------------------------------------------------------

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
            logger.info("[vector] reranker unavailable, returning pre-rerank order: %s", exc)
            return candidates[:n_results]
        if isinstance(scores, (int, float)):
            scores = [float(scores)]
        for i, s in enumerate(scores):
            candidates[i]["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:n_results]

    def delete_paper(self, paper_id):
        self.collection.delete(where={"paper_id": paper_id})
        self._rebuild_bm25()
        logger.info(f"[vector] deleted paper_id={paper_id}")


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    conf.check_config()
    from scholar_agent.storage.sqlite_store import PaperDB

    vdb = VectorDB()
    db = PaperDB()
    query = "SCMA"
    result = vdb.search(query, n_results=5)
    for r in result:
        print(f"{r}\n")

import importlib
import sys
import types
from pathlib import Path


class FakeCollection:
    def __init__(self):
        self.add_calls = []

    def get(self):
        return {"ids": [], "documents": [], "metadatas": []}

    def add(self, **kwargs):
        self.add_calls.append(kwargs)


class FakeClient:
    def __init__(self, path):
        self.path = path
        self.collection = FakeCollection()

    def get_or_create_collection(self, name):
        return self.collection


class FakeEmbeddingModel:
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path

    def encode(self, texts):
        if isinstance(texts, str):
            return {"dense_vecs": [0.1, 0.2]}
        return {"dense_vecs": [[0.1, 0.2] for _ in texts]}


def test_vector_db_add_chunks_does_not_load_reranker(monkeypatch):
    fake_flag_embedding = types.ModuleType("FlagEmbedding")
    fake_flag_embedding.BGEM3FlagModel = FakeEmbeddingModel

    def fail_reranker(*args, **kwargs):
        raise AssertionError("FlagReranker should not load during ingestion")

    fake_flag_embedding.FlagReranker = fail_reranker
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_flag_embedding)

    fake_chromadb = types.ModuleType("chromadb")
    fake_chromadb.PersistentClient = FakeClient
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)

    fake_jieba = types.ModuleType("jieba")
    fake_jieba.lcut = lambda text: list(text)
    monkeypatch.setitem(sys.modules, "jieba", fake_jieba)

    fake_rank_bm25 = types.ModuleType("rank_bm25")
    fake_rank_bm25.BM25Okapi = lambda tokens: object()
    monkeypatch.setitem(sys.modules, "rank_bm25", fake_rank_bm25)

    sys.modules.pop("scholar_agent.storage.vector_store", None)
    vector_store = importlib.import_module("scholar_agent.storage.vector_store")
    monkeypatch.setattr(vector_store.conf, "BGE_M3_MODEL_PATH", str(Path(".").resolve()))
    monkeypatch.setattr(vector_store.conf, "BGE_RERANKER_MODEL_PATH", str(Path(".").resolve()))

    db = vector_store.VectorDB()
    db.add_chunks("paper_1", ["chunk one", "chunk two"])

    assert db.reranker is None
    assert db.collection.add_calls


def test_vector_db_reports_missing_local_embedding_model(monkeypatch):
    fake_flag_embedding = types.ModuleType("FlagEmbedding")
    fake_flag_embedding.BGEM3FlagModel = FakeEmbeddingModel
    fake_flag_embedding.FlagReranker = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_flag_embedding)

    fake_chromadb = types.ModuleType("chromadb")
    fake_chromadb.PersistentClient = FakeClient
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)

    fake_jieba = types.ModuleType("jieba")
    fake_jieba.lcut = lambda text: list(text)
    monkeypatch.setitem(sys.modules, "jieba", fake_jieba)

    fake_rank_bm25 = types.ModuleType("rank_bm25")
    fake_rank_bm25.BM25Okapi = lambda tokens: object()
    monkeypatch.setitem(sys.modules, "rank_bm25", fake_rank_bm25)

    sys.modules.pop("scholar_agent.storage.vector_store", None)
    vector_store = importlib.import_module("scholar_agent.storage.vector_store")
    monkeypatch.setattr(vector_store.conf, "BGE_M3_MODEL_PATH", str((Path(".") / "__missing_bge_m3__").resolve()))
    monkeypatch.setattr(vector_store.conf, "BGE_RERANKER_MODEL_PATH", str((Path(".") / "__missing_reranker__").resolve()))

    try:
        vector_store.VectorDB()
    except FileNotFoundError as exc:
        assert "BGE-M3 embedding model not found" in str(exc)
    else:
        raise AssertionError("missing local embedding model should fail clearly")

import importlib
import json
import sys
import types
from pathlib import Path


class FakeCollection:
    def __init__(self):
        self.add_calls = []
        self.get_result = {"ids": [], "documents": [], "metadatas": []}

    def get(self, where=None):
        return self.get_result

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


def write_minimal_model_files(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")


def patch_model_manager_conf(vector_store, monkeypatch, **values):
    for key, value in values.items():
        monkeypatch.setattr(vector_store.conf, key, value)
        monkeypatch.setattr(vector_store.model_manager.conf, key, value)


def test_vector_db_add_chunks_does_not_load_reranker(monkeypatch, tmp_path):
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

    sys.modules.pop("rag.storage.vector_store", None)
    vector_store = importlib.import_module("rag.storage.vector_store")
    model_path = tmp_path / "models" / "bge-m3"
    reranker_path = tmp_path / "models" / "bge-reranker-v2-m3"
    write_minimal_model_files(model_path)
    write_minimal_model_files(reranker_path)
    patch_model_manager_conf(
        vector_store,
        monkeypatch,
        DB_DIR=str(tmp_path / "db"),
        BGE_M3_MODEL_PATH=str(model_path),
        BGE_RERANKER_MODEL_PATH=str(reranker_path),
    )

    db = vector_store.VectorDB()
    db.add_chunks(
        "paper_1",
        [
            {"content": "chunk one", "section_name": "introduction", "section_title": "Introduction"},
            {"content": "chunk two", "section_name": "method", "section_title": "Method"},
        ],
    )

    assert db.reranker is None
    assert db.supported_schema_version == 1
    assert db.collection.add_calls
    metadata = db.collection.add_calls[0]["metadatas"][0]
    assert metadata["section_name"] == "introduction"
    assert metadata["section_title"] == "Introduction"


def test_vector_db_reports_missing_local_embedding_model(monkeypatch, tmp_path):
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

    sys.modules.pop("rag.storage.vector_store", None)
    vector_store = importlib.import_module("rag.storage.vector_store")
    patch_model_manager_conf(
        vector_store,
        monkeypatch,
        DB_DIR=str(tmp_path / "db"),
        BGE_M3_MODEL_PATH=str(tmp_path / "__missing_bge_m3__"),
        BGE_RERANKER_MODEL_PATH=str(tmp_path / "__missing_reranker__"),
        BGE_AUTO_DOWNLOAD=False,
    )

    try:
        vector_store.VectorDB()
    except FileNotFoundError as exc:
        assert "BGE-M3 embedding model is not available" in str(exc)
    else:
        raise AssertionError("missing local embedding model should fail clearly")


def test_vector_db_downloads_missing_embedding_model(monkeypatch, tmp_path):
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

    downloads = []

    def fake_snapshot_download(repo_id, revision, local_dir, local_dir_use_symlinks=False):
        downloads.append((repo_id, revision, Path(local_dir).name, local_dir_use_symlinks))
        write_minimal_model_files(Path(local_dir))

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    sys.modules.pop("rag.storage.vector_store", None)
    vector_store = importlib.import_module("rag.storage.vector_store")
    model_path = tmp_path / "models" / "bge-m3"
    patch_model_manager_conf(
        vector_store,
        monkeypatch,
        BGE_M3_MODEL_PATH=str(model_path),
        BGE_M3_MODEL_REPO="BAAI/bge-m3",
        BGE_M3_MODEL_REVISION="main",
        BGE_RERANKER_MODEL_PATH=str(tmp_path),
        BGE_AUTO_DOWNLOAD=True,
        BGE_OFFLINE_MODE=False,
        BGE_MODEL_LOCK_TIMEOUT_SEC=2,
        BGE_MODEL_LOCK_STALE_SEC=2,
    )

    db = vector_store.VectorDB()

    assert db.ef.model_name_or_path == str(model_path)
    assert downloads[0][0] == "BAAI/bge-m3"
    assert downloads[0][1] == "main"
    assert downloads[0][2].startswith(".download-bge-m3-")
    assert downloads[0][3] is False
    assert (model_path / ".scholaragent_model_ready.json").exists()


def test_vector_db_downloads_missing_reranker_lazily(monkeypatch, tmp_path):
    fake_flag_embedding = types.ModuleType("FlagEmbedding")
    fake_flag_embedding.BGEM3FlagModel = FakeEmbeddingModel

    class FakeReranker:
        def __init__(self, model_name_or_path, use_fp16=True):
            self.model_name_or_path = model_name_or_path
            self.use_fp16 = use_fp16

        def compute_score(self, pairs):
            return [1.0 for _ in pairs]

    fake_flag_embedding.FlagReranker = FakeReranker
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

    downloads = []

    def fake_snapshot_download(repo_id, revision, local_dir, local_dir_use_symlinks=False):
        downloads.append((repo_id, revision, Path(local_dir).name, local_dir_use_symlinks))
        write_minimal_model_files(Path(local_dir))

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    sys.modules.pop("rag.storage.vector_store", None)
    vector_store = importlib.import_module("rag.storage.vector_store")
    reranker_path = tmp_path / "models" / "bge-reranker-v2-m3"
    patch_model_manager_conf(
        vector_store,
        monkeypatch,
        BGE_M3_MODEL_PATH=str(tmp_path),
        BGE_RERANKER_MODEL_PATH=str(reranker_path),
        BGE_RERANKER_MODEL_REPO="BAAI/bge-reranker-v2-m3",
        BGE_RERANKER_MODEL_REVISION="main",
        BGE_AUTO_DOWNLOAD=True,
        BGE_OFFLINE_MODE=False,
        BGE_MODEL_LOCK_TIMEOUT_SEC=2,
        BGE_MODEL_LOCK_STALE_SEC=2,
    )

    db = vector_store.VectorDB()
    reranker = db._get_reranker()

    assert reranker.model_name_or_path == str(reranker_path)
    reranker_downloads = [item for item in downloads if item[0] == "BAAI/bge-reranker-v2-m3"]
    assert reranker_downloads[0][1] == "main"
    assert reranker_downloads[0][2].startswith(".download-bge-reranker-v2-m3-")
    assert reranker_downloads[0][3] is False
    assert (reranker_path / ".scholaragent_model_ready.json").exists()


def test_vector_db_writes_chroma_schema_metadata(monkeypatch, tmp_path):
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

    sys.modules.pop("rag.storage.vector_store", None)
    vector_store = importlib.import_module("rag.storage.vector_store")
    model_path = tmp_path / "models" / "bge-m3"
    reranker_path = tmp_path / "models" / "bge-reranker-v2-m3"
    write_minimal_model_files(model_path)
    write_minimal_model_files(reranker_path)
    patch_model_manager_conf(
        vector_store,
        monkeypatch,
        DB_DIR=str(tmp_path / "db"),
        BGE_M3_MODEL_PATH=str(model_path),
        BGE_RERANKER_MODEL_PATH=str(reranker_path),
    )

    vector_store.VectorDB(collection_name="papers")

    metadata_path = tmp_path / "db" / "chroma" / "rag_chroma_schema.json"
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["collection_name"] == "papers"

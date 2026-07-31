"""ChromaDB 客户端 —— 封装 PersistentClient，提供同义词知识库的 CRUD

使用 BAAI/bge-small-zh-v1.5 中文 embedding 模型，支持中英文混合检索。
"""

import os as _os
if not _os.environ.get("HF_ENDPOINT"):
    _os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path

from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from backend.core.config import get_settings
from backend.core.logger import get_logger

# 中文 embedding 模型（轻量 24MB，中英文兼顾）
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# 相似度阈值：bge 模型余弦相似度通常在 0.3-0.9 范围
SIMILARITY_THRESHOLD = 0.4

_embedding_unavailable: bool = False


class ChromaClient:
    """ChromaDB 持久化客户端封装"""

    def __init__(self):
        settings = get_settings()
        self._persist_dir = Path(settings.chromadb.persist_dir)
        self._collection_name = settings.chromadb.collection_name
        self._log = get_logger()
        self._client: PersistentClient | None = None
        self._collection = None
        self._embed_fn: SentenceTransformerEmbeddingFunction | None = None
        self._no_embed_collection = None

    @property
    def embed_fn(self) -> SentenceTransformerEmbeddingFunction | None:
        global _embedding_unavailable
        if _embedding_unavailable:
            return None
        if self._embed_fn is None:
            try:
                self._embed_fn = SentenceTransformerEmbeddingFunction(
                    model_name=EMBEDDING_MODEL,
                )
                self._log.info(f"Embedding 模型已加载: {EMBEDDING_MODEL}")
            except Exception as e:
                self._log.warning(f"Embedding 模型加载失败（HuggingFace 不可用），禁用向量检索: {e}")
                _embedding_unavailable = True
                self._embed_fn = None
        return self._embed_fn

    @property
    def client(self) -> PersistentClient:
        if self._client is None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = PersistentClient(
                path=str(self._persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._log.info(f"ChromaDB 已连接: {self._persist_dir}")
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            ef = self.embed_fn
            name = self._collection_name
            if ef is None:
                # 无 embedding 可用：使用不带 embedding 的 collection（仅支持 get/add/关键词检索）
                name = f"{self._collection_name}_noembed"
            try:
                self._collection = self.client.get_or_create_collection(
                    name=name,
                    embedding_function=ef,
                    metadata={"description": "AI技能同义词知识库"},
                )
            except Exception:
                # 如果带 embedding 创建失败，降级为 no-embedding
                self._collection = self.client.get_or_create_collection(
                    name=f"{self._collection_name}_noembed",
                    embedding_function=None,
                    metadata={"description": "AI技能同义词知识库（降级模式）"},
                )
        return self._collection

    def rebuild(self) -> None:
        """切换 embedding 模型后重建集合"""
        try:
            old_coll = self.client.get_collection(self._collection_name)
            old_count = old_coll.count()
        except Exception:
            old_count = 0
        self.client.delete_collection(self._collection_name)
        self._collection = None
        self._log.info(f"ChromaDB 集合已删除 ({old_count} 条)，等待重新种子化")

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        """批量添加文档"""
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        self._log.info(f"ChromaDB 添加 {len(ids)} 条记录")

    def query(self, query_text: str, n_results: int = 3) -> list[str]:
        """检索同义表述（仅返回相似度 >= 阈值的）"""
        if self.embed_fn is None:
            return []  # 无 embedding，跳过向量检索，由调用方回退关键词
        results = self.collection.query(query_texts=[query_text], n_results=n_results)
        distances = results.get("distances", [[]])[0]  # ChromaDB 返回 distance（越小越相似）
        documents = results.get("documents", [[]])[0]

        synonyms: list[str] = []
        for distance, doc in zip(distances, documents):
            # ChromaDB distance: 越小越相似。转换: similarity = 1.0 - distance (近似)
            similarity = 1.0 - distance
            if similarity < SIMILARITY_THRESHOLD:
                continue
            parts = doc.split()
            synonyms.extend(parts[1:])  # 跳过 canonical
        return list(set(synonyms))

    def count(self) -> int:
        """返回集合中的文档数量"""
        return self.collection.count()

    def clear(self) -> None:
        """清空集合（用于重新种子化）"""
        self.client.delete_collection(self._collection_name)
        self._collection = None
        self._log.info("ChromaDB 集合已清空")


# 全局单例
chroma_client = ChromaClient()

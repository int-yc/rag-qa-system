
import os
import json
from typing import List

from config.settings import RETRIEVER_CONFIG
from utils.logger import logger, log_time

class VectorRetriever:

    def __init__(self, embedding_model: str = None):
        self.embedding_model = embedding_model or RETRIEVER_CONFIG["embedding_model"]
        self.top_k = RETRIEVER_CONFIG["top_k"]
        self.store_dir = RETRIEVER_CONFIG["vector_store_dir"]

        self._chunks: List[str] = []
        self._index = None
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            logger.info(f"加载向量模型：{self.embedding_model}")
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model)
        return self._embedder

    @log_time
    def build_index(self, chunks: List[str]):
        self._chunks = chunks
        embedder = self._get_embedder()

        embeddings = embedder.encode(chunks, normalize_embeddings=True)
        import faiss
        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(embeddings)
        logger.info(f"向量索引建立完成，共 {len(chunks)} 个文本块，维度 {dimension}")
        self._save_meta()

    @log_time
    def search(self, query: str, top_k: int = None) -> List[str]:
        indices = self.search_indices(query, top_k)
        results = [self._chunks[i] for i in indices]
        logger.info(f"检索到 {len(results)} 个相关文本块")
        return results

    @log_time
    def search_indices(self, query: str, top_k: int = None) -> List[int]:
        k = top_k or self.top_k
        embedder = self._get_embedder()
        query_vec = embedder.encode([query], normalize_embeddings=True)

        scores, indices = self._index.search(query_vec, k)
        result = []
        for i in indices[0]:
            if i >= 0 and i < len(self._chunks):
                result.append(int(i))
        return result

    def _save_meta(self):
        os.makedirs(self.store_dir, exist_ok=True)
        meta_path = os.path.join(self.store_dir, "chunks.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"文本块元数据已保存：{meta_path}")

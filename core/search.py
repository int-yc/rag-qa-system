
from typing import List, Tuple

from config.settings import SEARCH_CONFIG, RETRIEVER_CONFIG
from utils.logger import logger, log_time

class BM25Retriever:

    def __init__(self, chunks: List[str]):
        import jieba
        from rank_bm25 import BM25Okapi
        self._chunks = chunks

        tokenized = [list(jieba.cut(c)) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        import jieba
        query_tokens = list(jieba.cut(query))
        scores = self._bm25.get_scores(query_tokens)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, float(s)) for idx, s in ranked[:top_k]]

class Reranker:

    def __init__(self, model_name: str = None):
        self.model_name = model_name or SEARCH_CONFIG["rerank_model"]
        self._model = None

    def _get_model(self):
        if self._model is None:
            logger.info(f"加载重排模型：{self.model_name}")
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidate_indices: List[int],
               chunks: List[str], top_k: int) -> List[int]:
        model = self._get_model()

        pairs = [(query, chunks[i]) for i in candidate_indices]
        scores = model.predict(pairs)

        ranked = sorted(zip(candidate_indices, scores), key=lambda x: x[1], reverse=True)
        result = [idx for idx, _ in ranked[:top_k]]
        logger.info(f"重排完成，最终保留 {len(result)} 个文本块")
        return result

class HybridRetriever:

    def __init__(self, vector_retriever, chunks: List[str]):
        self.vector = vector_retriever
        self.bm25 = BM25Retriever(chunks)
        self.reranker = Reranker() if SEARCH_CONFIG["use_rerank"] else None
        self._chunks = chunks

    def _reciprocal_rank_fusion(self, ranked_lists: List[List[int]]) -> List[int]:
        from collections import defaultdict
        k = SEARCH_CONFIG["rrf_k"]
        score_map = defaultdict(float)
        for ranked in ranked_lists:
            for rank, idx in enumerate(ranked, start=1):
                score_map[idx] += 1.0 / (k + rank)

        return [idx for idx, _ in sorted(score_map.items(), key=lambda x: x[1], reverse=True)]

    @log_time
    def search(self, query: str, top_k: int = None) -> List[str]:
        final_k = top_k or SEARCH_CONFIG["final_top_k"]
        coarse_k = SEARCH_CONFIG["coarse_top_k"]

        vector_idx = self.vector.search_indices(query, coarse_k)
        bm25_idx = [idx for idx, _ in self.bm25.search(query, coarse_k)]

        fused = self._reciprocal_rank_fusion([vector_idx, bm25_idx])
        candidates = fused[:SEARCH_CONFIG["candidate_top_k"]]

        if self.reranker is not None:
            final_idx = self.reranker.rerank(query, candidates, self._chunks, final_k)
        else:
            final_idx = candidates[:final_k]

        results = [self._chunks[i] for i in final_idx]
        logger.info(f"混合检索返回 {len(results)} 个文本块")
        return results

if __name__ == "__main__":

    chunks = ["大语言模型是基于深度学习的大规模神经网络。",
              "RAG 是检索增强生成，用于解决模型幻觉。",
              "向量数据库用于存储和检索语义向量。"]
    import jieba
    bm = BM25Retriever(chunks)
    print("BM25 检索测试：", [chunks[i] for i, s in bm.search("向量数据库", 2)])

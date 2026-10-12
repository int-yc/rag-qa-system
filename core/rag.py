
from typing import List

from config.settings import LLM_CONFIG, RETRIEVER_CONFIG, SEARCH_CONFIG
from core.loader import DocumentLoader, TextSplitter
from core.retriever import VectorRetriever
from core.search import HybridRetriever
from core.llm import LLMClient
from utils.logger import logger, log_time

SYSTEM_PROMPT = """你是一个专业的文档问答助手。
请严格基于【参考文档】中的内容回答用户的问题。
要求：
1. 如果参考文档里有答案，就准确引用文档内容回答；
2. 如果参考文档里没有答案，明确回答"根据现有文档无法回答"，不要编造；
3. 回答要简洁、有条理，可以分点说明。"""

class RAGPipeline:

    def __init__(self):
        self.loader = DocumentLoader()
        self.splitter = TextSplitter()
        self.retriever = VectorRetriever()
        self.hybrid_retriever = None
        self._chunks: List[str] = []
        self.llm = LLMClient()
        self._ready = False

    @log_time
    def build(self):
        logger.info("开始构建知识库索引...")
        documents = self.loader.load_all()
        if not documents:
            logger.error("data 目录下没有找到文档，请先放入 txt/pdf 文件")
            return
        chunks = self.splitter.split_documents(documents)
        self._chunks = chunks
        self.retriever.build_index(chunks)

        if SEARCH_CONFIG.get("use_hybrid", False):
            self.hybrid_retriever = HybridRetriever(self.retriever, chunks)
        self._ready = True
        logger.info("知识库索引构建完成 ✅")

    def _search(self, question: str):
        if self.hybrid_retriever is not None:
            return self.hybrid_retriever.search(question)
        return self.retriever.search(question)

    @log_time
    def ask(self, question: str) -> str:
        if not self._ready:
            self.build()

        related_chunks = self._search(question)
        context = "\n\n".join(related_chunks)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"【参考文档】\n{context}\n\n【用户问题】\n{question}"},
        ]

        answer = self.llm.chat(messages)
        return answer

    def get_related(self, question: str) -> List[str]:
        return self._search(question)

if __name__ == "__main__":

    pipeline = RAGPipeline()
    pipeline.build()
    while True:
        q = input("\n请输入问题（输入 exit 退出）：").strip()
        if q.lower() == "exit":
            break
        print("\n" + "=" * 50)
        print("回答：", pipeline.ask(q))
        print("=" * 50)

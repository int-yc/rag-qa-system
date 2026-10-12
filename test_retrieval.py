
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.loader import DocumentLoader, TextSplitter
from core.retriever import VectorRetriever

def main():
    print("=" * 60)
    print("步骤1：加载文档")
    loader = DocumentLoader()
    docs = loader.load_all()
    print(f"读取到 {len(docs)} 个文档")

    print("=" * 60)
    print("步骤2：文本切分")
    splitter = TextSplitter()
    chunks = splitter.split_documents(docs)
    print(f"切分成 {len(chunks)} 个文本块")
    if chunks:
        print("示例文本块：", chunks[0][:80], "...")

    print("=" * 60)
    print("步骤3：向量化 + 建立 FAISS 索引")
    retriever = VectorRetriever()
    retriever.build_index(chunks)

    print("=" * 60)
    print("步骤4：语义检索测试")
    test_queries = [
        "什么是RAG？",
        "向量数据库是用来做什么的？",
        "如何让大模型不产生幻觉？",
        "Transformer的核心机制是什么？",
    ]
    for q in test_queries:
        results = retriever.search(q, top_k=1)
        print(f"\n问题：{q}")
        print(f"最相关文本块：{results[0][:100]}...")
    print("\n✅ 检索链路测试通过！")

if __name__ == "__main__":
    main()


import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.loader import DocumentLoader, TextSplitter
from core.retriever import VectorRetriever
from core.search import HybridRetriever

GOLDEN_SET = [

    ("什么是大语言模型？", ["大规模神经网络", "Transformer"]),
    ("RAG 是怎么工作的？", ["检索增强生成", "向量化", "重排"]),
    ("微调和 RAG 有什么区别？", ["微调", "Fine-tuning"]),
    ("向量数据库是做什么用的？", ["向量数据库", "ANN", "相似度"]),
    ("怎么评估大模型的效果？", ["召回率", "命中率", "MRR"]),
    ("Transformer 的核心机制是什么？", ["自注意力", "Self-Attention"]),

    ("LoRA 的全称是什么？", ["Low-Rank Adaptation"]),
    ("QLoRA 把模型量化到了多少位？", ["4bit"]),
    ("FAISS 底层用的是哪种检索算法？", ["ANN", "近似最近邻"]),
    ("知识蒸馏的英文是什么？", ["Knowledge Distillation"]),
    ("Function Calling 是什么？", ["Function Calling", "工具调用"]),
    ("LangGraph 是什么框架？", ["LangGraph"]),
    ("GPTQ 是什么技术？", ["GPTQ"]),
    ("MRR 指标全称是什么？", ["MRR", "倒数排名"]),
]

def build_retrievers():
    loader = DocumentLoader()
    splitter = TextSplitter()
    chunks = splitter.split_documents(loader.load_all())

    vector_retriever = VectorRetriever()
    vector_retriever.build_index(chunks)

    hybrid_retriever = HybridRetriever(vector_retriever, chunks)
    return vector_retriever, hybrid_retriever, chunks

def hit_at_k(results, key_words, k):
    return any(key in "".join(results[:k]) for key in key_words)

def main():
    print("=" * 66)
    print("加载文档、切分并构建索引...")
    vector_retriever, hybrid_retriever, chunks = build_retrievers()
    print(f"共 {len(chunks)} 个文本块，测试问题 {len(GOLDEN_SET)} 个")

    K_VALUES = [1, 3]
    total = len(GOLDEN_SET)
    stats = {k: {"vec": 0, "hyb": 0} for k in K_VALUES}

    print("\n" + "=" * 66)
    print("逐题评估（✅=命中  ❌=未命中）\n")

    for q, keys in GOLDEN_SET:
        vec_top = vector_retriever.search(q, top_k=3)
        hyb_top = hybrid_retriever.search(q, top_k=3)
        for k in K_VALUES:
            stats[k]["vec"] += int(hit_at_k(vec_top, keys, k))
            stats[k]["hyb"] += int(hit_at_k(hyb_top, keys, k))
        v1 = "✅" if hit_at_k(vec_top, keys, 1) else "❌"
        h1 = "✅" if hit_at_k(hyb_top, keys, 1) else "❌"
        v3 = "✅" if hit_at_k(vec_top, keys, 3) else "❌"
        h3 = "✅" if hit_at_k(hyb_top, keys, 3) else "❌"
        print(f"{q:<28} Hit@1:{v1}/{h1}   Hit@3:{v3}/{h3}   (纯向量/混合)")

    print("\n" + "=" * 66)
    print("【汇总对比】")
    for k in K_VALUES:
        vec_rate = stats[k]["vec"] / total * 100
        hyb_rate = stats[k]["hyb"] / total * 100
        diff = hyb_rate - vec_rate
        print(f"  Hit@{k}: 纯向量 {stats[k]['vec']}/{total} = {vec_rate:.1f}%  |  "
              f"混合检索 {stats[k]['hyb']}/{total} = {hyb_rate:.1f}%  |  提升 {diff:+.1f} 个百分点")

    print("\n评估完成。")

if __name__ == "__main__":
    main()

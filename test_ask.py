
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.rag import RAGPipeline

pipeline = RAGPipeline()
pipeline.build()
print("\n" + "=" * 50)
answer = pipeline.ask("什么是 RAG？")
print("问答回答：")
print(answer)
print("=" * 50)

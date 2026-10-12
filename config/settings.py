
LLM_CONFIG = {
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "api_key": "your-api-key-here",
    "model_name": "doubao-seed-2-1-pro-260915",
    "temperature": 0.7,
    "max_tokens": 512,
}

RETRIEVER_CONFIG = {
    "embedding_model": "BAAI/bge-small-zh-v1.5",
    "top_k": 3,
    "chunk_size": 300,
    "chunk_overlap": 50,
    "vector_store_dir": "./vector_store",
}

PATH_CONFIG = {
    "data_dir": "./data",
}

SEARCH_CONFIG = {
    "use_hybrid": True,
    "use_rerank": True,
    "coarse_top_k": 10,
    "rrf_k": 60,
    "candidate_top_k": 8,
    "final_top_k": 3,
    "rerank_model": "BAAI/bge-reranker-base",
}


"""
config/settings.py
Central registry of all selectable components and pipeline defaults.
Adding a new provider = add one entry here, implement the class, register in the factory.
"""

from dataclasses import dataclass, field
from typing import Dict, Any

# Component registries  (display-name → internal key)

LLM_OPTIONS: Dict[str, str] = {
    "Gemini 2.5 Flash":      "gemini-2.5-flash",
    "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
    "Gemini 2.5 Pro":        "gemini-2.5-pro",
    "Gemini 2.0 Flash":      "gemini-2.0-flash",
    "Gemini 2.0 Flash Lite": "gemini-2.0-flash-lite",
    "Gemini 3.5 Flash":      "gemini-3.5-flash",
    "Gemini 3 Pro":          "gemini-3-pro-preview",
    "Gemini 3 Flash":        "gemini-3-flash-preview",
}

EMBEDDING_OPTIONS: Dict[str, str] = {
    "MiniLM (all-MiniLM-L6-v2)":        "all-MiniLM-L6-v2",
    "BGE Small (bge-small-en-v1.5)":     "BAAI/bge-small-en-v1.5",
    "BGE Base (bge-base-en-v1.5)":       "BAAI/bge-base-en-v1.5",
    "BGE Large (bge-large-en-v1.5)":     "BAAI/bge-large-en-v1.5",
}

VECTORSTORE_OPTIONS: Dict[str, str] = {
    "FAISS":   "faiss",
    "ChromaDB": "chroma",
}

RETRIEVAL_OPTIONS: Dict[str, str] = {
    "Hybrid (BM25 + Embeddings)": "hybrid",
    "Similarity Search":           "similarity",
    "MMR (Max Marginal Relevance)": "mmr",
    "Multi-Query":                 "multiquery",
}

RERANKER_OPTIONS: Dict[str, str] = {
    "None":                        "none",
    "BGE Base Reranker":           "BAAI/bge-reranker-base",
    "BGE Large Reranker":          "BAAI/bge-reranker-large",
    "MiniLM Cross-Encoder":        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "MXBAI Reranker Large":        "mixedbread-ai/mxbai-rerank-large-v1",
}

# Pipeline defaults

@dataclass
class PipelineConfig:
    # LLM
    llm_display:        str = "Gemini 2.5 Flash"
    llm_model:          str = "gemini-2.5-flash"
    llm_provider:       str = "gemini"         

    # Embeddings
    embedding_display:  str = "MiniLM (all-MiniLM-L6-v2)"
    embedding_model:    str = "all-MiniLM-L6-v2"

    # Vector store
    vectorstore_display: str = "FAISS"
    vectorstore_type:    str = "faiss"

    # Retrieval
    retrieval_display:  str = "Hybrid (BM25 + Embeddings)"
    retrieval_method:   str = "hybrid"

    # Reranker
    reranker_display:   str = "BGE Base Reranker"
    reranker_model:     str = "BAAI/bge-reranker-base"

    # Chunking
    chunk_size:         int = 1000
    chunk_overlap:      int = 200

    # Retrieval sizes
    hybrid_fetch_k:     int = 20
    rerank_top_n:       int = 6


def llm_provider_from_model(model_key: str) -> str:
    """Infer the provider string from a model key."""
    if model_key.startswith("gemini"):
        return "gemini"
    if model_key.startswith("gpt"):
        return "openai"
    if model_key.startswith("claude"):
        return "anthropic"
    return "gemini"

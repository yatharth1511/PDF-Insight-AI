"""vectorstores/factory.py — Return the correct vector store backend."""

import logging
from vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)


def get_vector_store(store_type: str, persist_dir: str = "faiss_index") -> BaseVectorStore:
    """
    Instantiate a vector store by type key.

    Args:
        store_type:  'faiss' | 'chroma'
        persist_dir: Directory used for persistence.

    Returns:
        A fresh (unbuilt) BaseVectorStore instance.
    """
    if store_type == "faiss":
        from vectorstores.faiss_store import FAISSStore
        return FAISSStore()

    elif store_type == "chroma":
        from vectorstores.chroma_store import ChromaStore
        return ChromaStore(persist_dir="chroma_index")

    else:
        logger.warning(f"Unknown vector store '{store_type}'; defaulting to FAISS.")
        from vectorstores.faiss_store import FAISSStore
        return FAISSStore()

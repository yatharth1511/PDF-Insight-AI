"""vectorstores/base.py — Abstract vector store interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseVectorStore(ABC):
    """
    All vector store backends implement this interface.
    Swap FAISS for ChromaDB / Qdrant / Milvus without touching retrieval logic.
    """

    @abstractmethod
    def build(self, chunks: List[Dict[str, Any]], embeddings: "np.ndarray") -> None:
        """Index chunks using pre-computed embeddings."""
        ...

    @abstractmethod
    def save(self, directory: str) -> None:
        """Persist index to disk."""
        ...

    @abstractmethod
    def load(self, directory: str) -> bool:
        """Load a saved index. Returns True on success."""
        ...

    @abstractmethod
    def similarity_search(self, query_embedding: "np.ndarray", k: int) -> List[Dict[str, Any]]:
        """Return top-k chunks by cosine similarity."""
        ...

    @abstractmethod
    def mmr_search(
        self,
        query_embedding: "np.ndarray",
        k: int,
        fetch_k: int = 30,
        lambda_mult: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Return top-k chunks via Maximal Marginal Relevance."""
        ...

    @property
    @abstractmethod
    def chunk_count(self) -> int:
        """Number of indexed chunks."""
        ...

    @property
    def chunks(self) -> List[Dict[str, Any]]:
        """All stored chunks with metadata."""
        return []

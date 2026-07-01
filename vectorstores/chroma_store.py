"""vectorstores/chroma_store.py — ChromaDB-backed vector store."""

import logging
import numpy as np
from typing import List, Dict, Any
from vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaStore(BaseVectorStore):
    """
    Vector store backed by ChromaDB (persistent local mode).
    Implements the same interface as FAISSStore so the retriever
    doesn't need to know which backend is active.
    """

    COLLECTION_NAME = "pdf_insight"

    def __init__(self, persist_dir: str = "chroma_index"):
        self._persist_dir = persist_dir
        self._client      = None
        self._collection  = None
        self._chunks: List[Dict[str, Any]] = []

    def _init_client(self) -> None:
        import chromadb
        self._client     = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def build(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        self._chunks = chunks
        self._init_client()

        # Clear existing data
        try:
            self._client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        ids        = [str(c["chunk_id"]) for c in chunks]
        documents  = [c["text"] for c in chunks]
        metadatas  = [
            {"filename": c["filename"], "page_number": c["page_number"]}
            for c in chunks
        ]
        embs = embeddings.tolist()

        # Chroma has a 5461-item batch limit
        batch = 5000
        for i in range(0, len(ids), batch):
            self._collection.add(
                ids=ids[i:i+batch],
                documents=documents[i:i+batch],
                metadatas=metadatas[i:i+batch],
                embeddings=embs[i:i+batch],
            )
        logger.info(f"ChromaDB collection built: {len(chunks)} items")

    def save(self, directory: str) -> None:
        # ChromaDB auto-persists; just save chunk list for metadata retrieval
        import pickle, os
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "chroma_meta.pkl"), "wb") as f:
            pickle.dump(self._chunks, f)

    def load(self, directory: str) -> bool:
        import pickle, os
        meta_path = os.path.join(directory, "chroma_meta.pkl")
        if not os.path.exists(meta_path):
            return False
        with open(meta_path, "rb") as f:
            self._chunks = pickle.load(f)
        try:
            self._init_client()
            return self._collection.count() > 0
        except Exception as e:
            logger.error(f"ChromaDB load failed: {e}")
            return False

    def similarity_search(self, query_embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text":         doc,
                "filename":     meta.get("filename", "?"),
                "page_number":  meta.get("page_number", "?"),
                "chunk_id":     -1,
                "dense_score":  1.0 - dist,  # convert cosine distance → similarity
            })
        return chunks

    def mmr_search(self, query_embedding, k, fetch_k=30, lambda_mult=0.5):
        # ChromaDB doesn't natively support MMR; fall back to similarity
        logger.warning("ChromaDB MMR not supported; using similarity search.")
        return self.similarity_search(query_embedding, k)

    @property
    def chunk_count(self) -> int:
        try:
            return self._collection.count() if self._collection else 0
        except Exception:
            return len(self._chunks)

    @property
    def chunks(self) -> List[Dict[str, Any]]:
        return self._chunks

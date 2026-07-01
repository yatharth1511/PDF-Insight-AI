"""vectorstores/faiss_store.py — FAISS-backed vector store."""

import os
import pickle
import logging
import numpy as np
import faiss
from typing import List, Dict, Any
from vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class FAISSStore(BaseVectorStore):
    """
    Dense vector store backed by FAISS IndexFlatIP (cosine on L2-normalised vecs).
    Also stores raw chunks + BM25 index (built externally by the hybrid retriever).
    """

    FAISS_FILE = "index.faiss"
    META_FILE  = "metadata.pkl"

    def __init__(self):
        self._index:  faiss.Index | None  = None
        self._chunks: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Build / persist / restore
    # ------------------------------------------------------------------

    def build(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        self._chunks = chunks
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype("float32"))
        logger.info(f"FAISS index built: {len(chunks)} vectors, dim={dim}")

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self._index, os.path.join(directory, self.FAISS_FILE))
        with open(os.path.join(directory, self.META_FILE), "wb") as f:
            pickle.dump(self._chunks, f)
        logger.info(f"FAISS index saved to {directory}")

    def load(self, directory: str) -> bool:
        fp = os.path.join(directory, self.FAISS_FILE)
        mp = os.path.join(directory, self.META_FILE)
        if not (os.path.exists(fp) and os.path.exists(mp)):
            return False
        self._index = faiss.read_index(fp)
        with open(mp, "rb") as f:
            self._chunks = pickle.load(f)
        logger.info(f"FAISS index loaded from {directory}: {len(self._chunks)} chunks")
        return True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def similarity_search(self, query_embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        q = query_embedding.reshape(1, -1).astype("float32")
        scores, indices = self._index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self._chunks[idx])
            chunk["dense_score"] = float(score)
            results.append(chunk)
        return results

    def mmr_search(
        self,
        query_embedding: np.ndarray,
        k: int,
        fetch_k: int = 30,
        lambda_mult: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Maximal Marginal Relevance: balances relevance with diversity.
        lambda_mult=1.0 → pure relevance; 0.0 → pure diversity.
        """
        fetch_k = min(fetch_k, self.chunk_count)
        q = query_embedding.reshape(1, -1).astype("float32")
        scores, indices = self._index.search(q, fetch_k)

        # Get embeddings for candidates
        valid = [(s, i) for s, i in zip(scores[0], indices[0]) if i != -1]
        if not valid:
            return []

        cand_indices = [i for _, i in valid]

        # Reconstruct candidate vectors from FAISS
        cand_vecs = np.zeros((len(cand_indices), self._index.d), dtype="float32")
        for row, idx in enumerate(cand_indices):
            self._index.reconstruct(int(idx), cand_vecs[row])

        selected_idx = []
        remaining    = list(range(len(cand_indices)))

        for _ in range(min(k, len(remaining))):
            if not remaining:
                break
            if not selected_idx:
                # Pick highest relevance first
                best = max(remaining, key=lambda r: float(scores[0][r]))
            else:
                # MMR score = λ * relevance - (1-λ) * max_similarity_to_selected
                sel_vecs = cand_vecs[selected_idx]
                best, best_score = None, -1e9
                for r in remaining:
                    rel  = lambda_mult * float(scores[0][r])
                    sim  = float(np.max(cand_vecs[r] @ sel_vecs.T))
                    div  = (1 - lambda_mult) * sim
                    mmr  = rel - div
                    if mmr > best_score:
                        best_score, best = mmr, r
            selected_idx.append(best)
            remaining.remove(best)

        results = []
        for r in selected_idx:
            orig_idx = cand_indices[r]
            chunk    = dict(self._chunks[orig_idx])
            chunk["dense_score"] = float(scores[0][r])
            results.append(chunk)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> List[Dict[str, Any]]:
        return self._chunks

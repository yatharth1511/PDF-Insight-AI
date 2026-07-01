"""
retrievers/hybrid.py
All retrieval strategies in one module.
Strategy is selected at call time via the 'method' parameter.

Strategies:
  similarity  – pure FAISS cosine
  mmr         – Maximal Marginal Relevance (diversity-aware)
  hybrid      – BM25 sparse + FAISS dense, fused with RRF
  multiquery  – query decomposition → parallel hybrid → deduplicate → merge
"""

from __future__ import annotations

import logging
import pickle
import os
from typing import List, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)

BM25_FILE = "bm25.pkl"

# BM25 helpers

def build_bm25(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    tokenized = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized)


def save_bm25(bm25: BM25Okapi, directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, BM25_FILE), "wb") as f:
        pickle.dump(bm25, f)


def load_bm25(directory: str) -> BM25Okapi | None:
    path = os.path.join(directory, BM25_FILE)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

# RRF fusion

def _rrf_fuse(
    dense_ranks: Dict[int, int],
    sparse_ranks: Dict[int, int],
    n_candidates: int,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    rrf_k: int = 60,
) -> Dict[int, float]:
    all_ids = set(dense_ranks) | set(sparse_ranks)
    fused   = {}
    for cid in all_ids:
        dr = dense_ranks.get(cid,  n_candidates + 1)
        sr = sparse_ranks.get(cid, n_candidates + 1)
        fused[cid] = (
            dense_weight  / (rrf_k + dr) +
            sparse_weight / (rrf_k + sr)
        )
    return fused

# Query decomposition (multi-part question splitting)

def decompose_query(query: str, llm=None) -> List[str]:
    """
    Split a compound question into atomic sub-queries.
    Uses the LLM if provided; falls back to simple heuristic splitting.
    """
    if llm is not None:
        try:
            system = (
                "You are a query decomposition expert. "
                "Split the user question into 2-4 simple, independent sub-questions "
                "that together cover all parts of the original question. "
                "Return ONLY a JSON array of strings, nothing else. "
                'Example: ["What is X?", "What is Y?", "How do X and Y compare?"]'
            )
            response = llm.chat(system=system, messages=[], user_message=query, max_tokens=300)
            import json, re
            # Extract JSON array from response
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                sub_queries = json.loads(match.group())
                if isinstance(sub_queries, list) and sub_queries:
                    logger.info(f"Query decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
                    return sub_queries
        except Exception as e:
            logger.warning(f"LLM query decomposition failed: {e}; using heuristic split.")

    # Heuristic: split on 'and', 'also', question marks, semicolons
    import re
    parts = re.split(r'\?(?=\s)', query)
    parts = [p.strip().rstrip('?').strip() for p in parts if p.strip()]
    if len(parts) > 1:
        parts = [p + '?' for p in parts]
        logger.info(f"Heuristic decomposition: {parts}")
        return parts

    return [query]

# Main retriever

class Retriever:
    """
    Unified retriever supporting similarity, MMR, hybrid, and multi-query strategies.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embed_model:  SentenceTransformer,
        bm25:         BM25Okapi | None = None,
    ):
        self.vs          = vector_store
        self.embed_model = embed_model
        self.bm25        = bm25

    def _encode(self, text: str) -> np.ndarray:
        return self.embed_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")[0]

    # Public retrieve method

    def retrieve(
        self,
        query: str,
        method: str   = "hybrid",
        k: int        = 20,
        llm           = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve candidate chunks using the selected strategy.

        Args:
            query:  User question (may be compound).
            method: 'similarity' | 'mmr' | 'hybrid' | 'multiquery'
            k:      Number of candidates to return.
            llm:    Optional LLM used for multi-query decomposition.

        Returns:
            List of chunk dicts, each with a score field.
        """
        if method == "similarity":
            return self._similarity(query, k)

        elif method == "mmr":
            return self._mmr(query, k)

        elif method == "hybrid":
            return self._hybrid(query, k)

        elif method == "multiquery":
            return self._multiquery(query, k, llm)

        else:
            logger.warning(f"Unknown retrieval method '{method}'; using hybrid.")
            return self._hybrid(query, k)

    # Strategy implementations

    def _similarity(self, query: str, k: int) -> List[Dict[str, Any]]:
        q_emb = self._encode(query)
        return self.vs.similarity_search(q_emb, k)

    def _mmr(self, query: str, k: int) -> List[Dict[str, Any]]:
        q_emb = self._encode(query)
        return self.vs.mmr_search(q_emb, k, fetch_k=k * 3)

    def _hybrid(self, query: str, k: int) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            logger.warning("BM25 index not available; falling back to similarity.")
            return self._similarity(query, k)

        chunks = self.vs.chunks
        n      = min(k * 2, len(chunks))

        # --- Dense ---
        q_emb = self._encode(query)
        d_scores, d_indices = self.vs._index.search(
            q_emb.reshape(1, -1), n
        ) if hasattr(self.vs, '_index') else (None, None)

        # Use similarity_search as fallback if _index not accessible
        if d_indices is None:
            dense_results  = self.vs.similarity_search(q_emb, n)
            dense_ranks    = {i: rank + 1 for rank, i in enumerate(
                [chunks.index(c) for c in dense_results]
            )}
        else:
            dense_ranks = {
                int(idx): rank + 1
                for rank, idx in enumerate(d_indices[0])
                if idx != -1
            }

        # --- Sparse ---
        bm25_scores  = self.bm25.get_scores(query.lower().split())
        sparse_order = np.argsort(bm25_scores)[::-1][:n]
        sparse_ranks = {int(idx): rank + 1 for rank, idx in enumerate(sparse_order)}

        # --- RRF ---
        fused      = _rrf_fuse(dense_ranks, sparse_ranks, n)
        sorted_ids = sorted(fused, key=fused.__getitem__, reverse=True)[:k]

        results = []
        for cid in sorted_ids:
            chunk = dict(chunks[cid])
            chunk["hybrid_score"] = fused[cid]
            results.append(chunk)

        logger.info(f"Hybrid retrieval: {len(results)} candidates")
        return results

    def _multiquery(self, query: str, k: int, llm=None) -> List[Dict[str, Any]]:
        """Decompose query → retrieve per sub-query → deduplicate → merge."""
        sub_queries = decompose_query(query, llm)
        seen_ids:  set            = set()
        all_chunks: List[Dict]    = []

        # Collect candidates from each sub-query (hybrid where possible)
        per_q = max(k, 10)
        for sq in sub_queries:
            candidates = self._hybrid(sq, per_q) if self.bm25 else self._similarity(sq, per_q)
            for chunk in candidates:
                cid = chunk.get("chunk_id", id(chunk))
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    chunk["sub_query"] = sq
                    all_chunks.append(chunk)

        # Sort by hybrid/dense score descending, return top k
        all_chunks.sort(
            key=lambda c: c.get("hybrid_score", c.get("dense_score", 0.0)),
            reverse=True,
        )
        logger.info(
            f"Multi-query: {len(sub_queries)} sub-queries → {len(all_chunks)} unique candidates"
        )
        return all_chunks[:k]

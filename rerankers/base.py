"""
rerankers/base.py
Abstract reranker interface + concrete implementations.

Supported:
  - NoneReranker   (passthrough)
  - BGEReranker    (BAAI/bge-reranker-* via FlagEmbedding)
  - CrossEncoder   (sentence-transformers CrossEncoder)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Base

class BaseReranker(ABC):
    """All rerankers implement this interface."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int,
    ) -> List[Dict[str, Any]]:
        """
        Score and reorder chunks; return the top_n highest-scored.

        Each returned chunk is annotated with 'rerank_score' (float).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def healthy(self) -> bool:
        """Override to False if model failed to load."""
        return True

# Passthrough (no reranking)

class NoneReranker(BaseReranker):
    """Simply returns the first top_n chunks unchanged."""

    @property
    def name(self) -> str:
        return "None (passthrough)"

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int,
    ) -> List[Dict[str, Any]]:
        for c in chunks:
            c["rerank_score"] = c.get("hybrid_score", 0.0)
        return chunks[:top_n]


# BGE Reranker (FlagEmbedding)

class BGEReranker(BaseReranker):
    """
    Uses BAAI/bge-reranker-* via the FlagEmbedding library.
    Supports: bge-reranker-base, bge-reranker-large, mxbai-rerank-large-v1
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model      = None
        self._healthy    = False
        self._load()

    def _load(self) -> None:
        try:
            from FlagEmbedding import FlagReranker
            logger.info(f"Loading BGE reranker: {self._model_name}")
            self._model   = FlagReranker(self._model_name, use_fp16=True)
            self._healthy = True
            logger.info(f"BGE reranker loaded successfully: {self._model_name}")
        except ImportError as e:
            logger.error(
                f"FlagEmbedding not installed. Install with: pip install FlagEmbedding. Error: {e}"
            )
        except Exception as e:
            logger.error(f"BGE reranker '{self._model_name}' failed to load: {e}", exc_info=True)

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def healthy(self) -> bool:
        return self._healthy

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int,
    ) -> List[Dict[str, Any]]:
        if not self._healthy or not self._model:
            logger.warning(f"BGE reranker not healthy; returning first {top_n} chunks unchanged.")
            return chunks[:top_n]

        try:
            pairs  = [[query, c["text"]] for c in chunks]
            scores = self._model.compute_score(pairs, normalize=True)

            # scores may be a flat list or numpy array
            if hasattr(scores, "tolist"):
                scores = scores.tolist()

            for chunk, score in zip(chunks, scores):
                chunk["rerank_score"] = float(score)

            ranked = sorted(chunks, key=lambda c: c.get("rerank_score", 0.0), reverse=True)
            logger.info(
                f"BGE reranker scored {len(chunks)} chunks; "
                f"top score={ranked[0]['rerank_score']:.4f}"
            )
            return ranked[:top_n]

        except Exception as e:
            logger.error(f"BGE reranker scoring failed: {e}", exc_info=True)
            return chunks[:top_n]


# CrossEncoder (sentence-transformers)

class CrossEncoderReranker(BaseReranker):
    """
    Uses sentence-transformers CrossEncoder.
    Supports: cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model      = None
        self._healthy    = False
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            logger.info(f"Loading CrossEncoder reranker: {self._model_name}")
            self._model   = CrossEncoder(self._model_name)
            self._healthy = True
            logger.info(f"CrossEncoder loaded: {self._model_name}")
        except Exception as e:
            logger.error(f"CrossEncoder '{self._model_name}' failed to load: {e}", exc_info=True)

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def healthy(self) -> bool:
        return self._healthy

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int,
    ) -> List[Dict[str, Any]]:
        if not self._healthy or not self._model:
            logger.warning("CrossEncoder not healthy; returning first chunks unchanged.")
            return chunks[:top_n]

        try:
            pairs  = [(query, c["text"]) for c in chunks]
            scores = self._model.predict(pairs)

            for chunk, score in zip(chunks, scores):
                chunk["rerank_score"] = float(score)

            ranked = sorted(chunks, key=lambda c: c.get("rerank_score", 0.0), reverse=True)
            return ranked[:top_n]

        except Exception as e:
            logger.error(f"CrossEncoder scoring failed: {e}", exc_info=True)
            return chunks[:top_n]

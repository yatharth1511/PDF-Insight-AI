"""rerankers/factory.py — Instantiate the right reranker from a model key."""

import logging
import streamlit as st
from rerankers.base import BaseReranker, NoneReranker, BGEReranker, CrossEncoderReranker

logger = logging.getLogger(__name__)

# Models that use BGE reranker library vs CrossEncoder library
_BGE_MODELS = {
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-large",
    "mixedbread-ai/mxbai-rerank-large-v1",
}

_CE_MODELS = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
}


@st.cache_resource(show_spinner=False)
def get_reranker(model_key: str) -> BaseReranker:
    """
    Load and cache a reranker by model key.
    Cache is keyed by model_key so switching rerankers loads correctly.

    Args:
        model_key: Internal model identifier (e.g. 'BAAI/bge-reranker-base').

    Returns:
        A BaseReranker instance (always returns something; may be NoneReranker on failure).
    """
    if model_key == "none":
        logger.info("Reranker: None (passthrough)")
        return NoneReranker()

    if model_key in _BGE_MODELS:
        logger.info(f"Reranker: BGEReranker({model_key})")
        r = BGEReranker(model_key)
        if not r.healthy:
            logger.warning(f"BGEReranker failed; falling back to NoneReranker.")
            return NoneReranker()
        return r

    if model_key in _CE_MODELS:
        logger.info(f"Reranker: CrossEncoderReranker({model_key})")
        r = CrossEncoderReranker(model_key)
        if not r.healthy:
            logger.warning(f"CrossEncoderReranker failed; falling back to NoneReranker.")
            return NoneReranker()
        return r

    logger.warning(f"Unknown reranker key '{model_key}'; using NoneReranker.")
    return NoneReranker()

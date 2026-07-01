"""
embeddings/factory.py
Loads and caches SentenceTransformer embedding models.
Cached by model name so switching models in the UI triggers a reload.
"""

import logging
import streamlit as st
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def get_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Load and cache a SentenceTransformer model.

    The cache key is the model_name, so switching to a different model
    correctly loads a new instance.

    Args:
        model_name: HuggingFace model name or path.

    Returns:
        A loaded SentenceTransformer instance.

    Raises:
        RuntimeError: If the model fails to load.
    """
    logger.info(f"Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        logger.info(f"Embedding model loaded: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model '{model_name}': {e}")
        raise RuntimeError(f"Could not load embedding model '{model_name}': {e}") from e


def embed_texts(model: SentenceTransformer, texts: list[str]) -> "np.ndarray":
    """
    Encode a list of texts to normalized float32 embeddings.

    Args:
        model: A loaded SentenceTransformer.
        texts: List of strings to embed.

    Returns:
        numpy array of shape (len(texts), dim), float32, L2-normalised.
    """
    import numpy as np
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    return embeddings

"""llms/factory.py — Factory that returns the correct LLM adapter."""

import os
import logging
from llms.base import BaseLLM
from config.settings import llm_provider_from_model

logger = logging.getLogger(__name__)


def get_llm(model_key: str, api_keys: dict) -> BaseLLM:
    """
    Instantiate the right LLM adapter for a given model key.

    Args:
        model_key: Internal model identifier (e.g. 'gemini-2.0-flash').
        api_keys:  Dict with keys 'google', 'openai', 'anthropic'.

    Returns:
        An instance of BaseLLM.

    Raises:
        ValueError:  Unknown provider.
        RuntimeError: Missing API key.
    """
    provider = llm_provider_from_model(model_key)

    if provider == "gemini":
        from llms.gemini_llm import GeminiLLM
        key = api_keys.get("google") or os.getenv("GOOGLE_API_KEY", "")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set.")
        return GeminiLLM(model=model_key, api_key=key)

    elif provider == "openai":
        from llms.openai_llm import OpenAILLM
        key = api_keys.get("openai") or os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return OpenAILLM(model=model_key, api_key=key)

    elif provider == "anthropic":
        from llms.anthropic_llm import AnthropicLLM
        key = api_keys.get("anthropic") or os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        return AnthropicLLM(model=model_key, api_key=key)

    else:
        raise ValueError(f"Unknown LLM provider '{provider}' for model '{model_key}'.")

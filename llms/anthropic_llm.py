"""llms/anthropic_llm.py — Anthropic Claude adapter."""

import logging
from typing import List, Dict
import anthropic
from llms.base import BaseLLM

logger = logging.getLogger(__name__)


class AnthropicLLM(BaseLLM):
    def __init__(self, model: str, api_key: str):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model  = model
        logger.info(f"AnthropicLLM initialised with model={model}")

    @property
    def model_name(self) -> str:
        return self._model

    def chat(
        self,
        system: str,
        messages: List[Dict[str, str]],
        user_message: str,
        max_tokens: int = 1500,
        temperature: float = 0.2,
    ) -> str:
        history = list(messages[-8:])
        history.append({"role": "user", "content": user_message})

        resp = self._client.messages.create(
            model=self._model,
            system=system,
            messages=history,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.content[0].text.strip()

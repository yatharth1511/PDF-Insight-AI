"""llms/openai_llm.py — OpenAI adapter."""

import logging
from typing import List, Dict
from openai import OpenAI
from llms.base import BaseLLM

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    def __init__(self, model: str, api_key: str):
        self._client = OpenAI(api_key=api_key)
        self._model  = model
        logger.info(f"OpenAILLM initialised with model={model}")

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
        history = [{"role": "system", "content": system}]
        history += messages[-8:]
        history.append({"role": "user", "content": user_message})

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=history,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

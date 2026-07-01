"""llms/gemini_llm.py — Google Gemini adapter."""

import logging
from typing import List, Dict
import google.generativeai as genai
from llms.base import BaseLLM

logger = logging.getLogger(__name__)


class GeminiLLM(BaseLLM):
    def __init__(self, model: str, api_key: str):
        genai.configure(api_key=api_key)
        self._model = model
        logger.info(f"GeminiLLM initialised with model={model}")

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
        gemini = genai.GenerativeModel(
            model_name=self._model,
            system_instruction=system,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        # Convert history to Gemini format
        history = []
        for m in messages[-8:]:  # keep last 4 exchanges
            role = "user" if m["role"] == "user" else "model"
            history.append({"role": role, "parts": [m["content"]]})

        chat_session = gemini.start_chat(history=history)
        response = chat_session.send_message(user_message)
        return response.text.strip()

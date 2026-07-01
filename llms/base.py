"""llms/base.py — Abstract base class for all LLM providers."""

from abc import ABC, abstractmethod
from typing import List, Dict


class BaseLLM(ABC):
    """All LLM adapters implement this interface."""

    @abstractmethod
    def chat(
        self,
        system: str,
        messages: List[Dict[str, str]],
        user_message: str,
        max_tokens: int = 1500,
        temperature: float = 0.2,
    ) -> str:
        """
        Send a chat request.

        Args:
            system:       System/instruction prompt.
            messages:     Prior conversation turns [{role, content}].
            user_message: The current user turn.
            max_tokens:   Max tokens in the reply.
            temperature:  Sampling temperature.

        Returns:
            The assistant reply as a plain string.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""
        ...

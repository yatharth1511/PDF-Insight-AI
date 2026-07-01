"""utils/qa_engine.py — Citation-aware RAG question answering."""

import logging
from typing import List, Dict, Any, Tuple
from llms.base import BaseLLM

logger = logging.getLogger(__name__)


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format chunks as numbered SOURCE blocks for citation."""
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[SOURCE {i}] File: {c.get('filename','?')} | Page: {c.get('page_number','?')}\n"
            f"{c.get('text','')}"
        )
    return "\n\n".join(lines)


_SYSTEM = (
    "You are an expert document analyst. "
    "Answer the user's question using ONLY the provided source passages. "
    "After each factual claim add an inline citation [SOURCE N] matching the numbered source. "
    "Cite multiple sources where applicable, e.g. [SOURCE 1][SOURCE 3]. "
    "End your answer with a '### Citations' section listing each referenced source "
    "with its filename and page number. "
    "If the answer is not in the sources, say so — never invent information."
)


def answer_question(
    query: str,
    reranked_chunks: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]],
    llm: BaseLLM,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generate a citation-aware answer.

    Args:
        query:                The user question.
        reranked_chunks:      Top chunks after reranking.
        conversation_history: Prior [{role, content}] turns.
        llm:                  Any BaseLLM instance.

    Returns:
        (answer_text, cited_chunks)
    """
    context      = _format_context(reranked_chunks)
    user_message = (
        f"CONTEXT SOURCES:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer using only the sources above. "
        "Cite each source inline as [SOURCE N] and add a Citations section at the end."
    )

    try:
        answer = llm.chat(
            system=_SYSTEM,
            messages=conversation_history[-8:],
            user_message=user_message,
            max_tokens=1500,
            temperature=0.2,
        )
    except Exception as e:
        logger.error(f"LLM answer generation failed: {e}", exc_info=True)
        answer = f"Error generating answer: {e}"

    # Map inline citations back to source chunks
    cited = [
        c for i, c in enumerate(reranked_chunks, 1)
        if f"[SOURCE {i}]" in answer
    ]
    return answer, cited

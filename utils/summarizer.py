"""utils/summarizer.py — Document summarization via any BaseLLM."""

import logging
from typing import List, Dict, Any, Tuple
from llms.base import BaseLLM

logger = logging.getLogger(__name__)


def _sample_context(chunks: List[Dict[str, Any]], max_chars: int = 14000) -> str:
    if not chunks:
        return ""
    step    = max(1, len(chunks) // 50)
    sampled = chunks[::step]
    context = "\n\n---\n\n".join(c["text"] for c in sampled)
    return context[:max_chars]


def generate_summaries(
    chunks: List[Dict[str, Any]],
    llm: BaseLLM,
) -> Tuple[str, str]:
    """
    Generate short + detailed summaries.

    Args:
        chunks: Indexed document chunks.
        llm:    Any BaseLLM instance.

    Returns:
        (short_summary, detailed_summary)
    """
    context = _sample_context(chunks)
    if not context.strip():
        return "No content to summarize.", ""

    system = "You are an expert document analyst. Be factual, clear and concise."

    # Short
    short_prompt = (
        "Based on the following document excerpts, write a concise summary of 5–10 sentences "
        "covering the core topics, key findings, and purpose.\n\n"
        f"EXCERPTS:\n{context}\n\nSHORT SUMMARY:"
    )
    try:
        short = llm.chat(system=system, messages=[], user_message=short_prompt, max_tokens=500)
    except Exception as e:
        short = f"Could not generate short summary: {e}"
        logger.error(short)

    # Detailed
    detailed_prompt = (
        "Based on the following document excerpts, write a detailed structured summary "
        "with bullet points grouped under these headings: "
        "**Main Topics**, **Key Points**, **Important Details**, **Conclusions**.\n\n"
        f"EXCERPTS:\n{context}\n\nDETAILED SUMMARY:"
    )
    try:
        detailed = llm.chat(system=system, messages=[], user_message=detailed_prompt, max_tokens=1200)
    except Exception as e:
        detailed = f"Could not generate detailed summary: {e}"
        logger.error(detailed)

    return short, detailed

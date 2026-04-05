from __future__ import annotations

from langchain_ollama import ChatOllama

from rag_app.config import CONFIG
from rag_app.logging_utils import get_logger

LOGGER = get_logger("llm")


def build_chat_llm() -> ChatOllama:
    LOGGER.info(
        "Initializing ChatOllama with model '%s' via '%s' and timeout=%ss",
        CONFIG.ollama_model_name,
        CONFIG.ollama_base_url,
        CONFIG.graph_llm_timeout_seconds,
    )
    return ChatOllama(
        model=CONFIG.ollama_model_name,
        base_url=CONFIG.ollama_base_url,
        temperature=0,
        timeout=CONFIG.graph_llm_timeout_seconds,
    )

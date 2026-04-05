from __future__ import annotations

from functools import lru_cache

from langchain_ollama import OllamaEmbeddings

from rag_app.config import CONFIG
from rag_app.logging_utils import get_logger

LOGGER = get_logger("embeddings")


@lru_cache(maxsize=1)
def build_embeddings() -> OllamaEmbeddings:
    LOGGER.info(
        "Initializing Ollama embeddings with model '%s' via '%s'",
        CONFIG.ollama_embedding_model_name,
        CONFIG.ollama_base_url,
    )
    return OllamaEmbeddings(
        model=CONFIG.ollama_embedding_model_name,
        base_url=CONFIG.ollama_base_url,
    )

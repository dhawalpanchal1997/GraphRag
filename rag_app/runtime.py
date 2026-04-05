from __future__ import annotations

import logging
import os
import warnings

from rag_app.config import CONFIG


def configure_runtime() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    logging.basicConfig(
        level=getattr(logging, CONFIG.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    warnings.filterwarnings(
        "ignore",
        message=r"Accessing `__path__` from `\.models\..*Returning `__path__` instead\..*",
    )

    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("unstructured").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:  # noqa: BLE001
        pass

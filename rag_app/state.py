from __future__ import annotations

import streamlit as st

from rag_app.config import CONFIG


def init_session_state() -> None:
    defaults = {
        "uploaded_paths": [],
        "indexed_files": [],
        "chat_history": [],
        "index_ready": False,
        "graph_ready": False,
        "graph_stats": {},
        "chunk_size": CONFIG.chunk_size,
        "chunk_overlap": CONFIG.chunk_overlap,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

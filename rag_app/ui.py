from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_app.chat import ChatService
from rag_app.config import CONFIG
from rag_app.document_loader import load_pdf_documents, save_uploaded_files, split_documents
from rag_app.embeddings import build_embeddings
from rag_app.logging_utils import get_logger
from rag_app.neo4j_graphrag_service import Neo4jGraphRAGConnectionError, Neo4jGraphRAGService
from rag_app.state import init_session_state
from rag_app.vector_store import QdrantService, VectorStoreConnectionError

LOGGER = get_logger("ui")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .app-kicker {
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.72rem;
            font-weight: 700;
            color: #8a6a2f;
            margin-bottom: 0.35rem;
        }
        .app-hero {
            padding: 1.2rem 1.4rem;
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-radius: 18px;
            background:
                radial-gradient(circle at top left, rgba(255, 219, 153, 0.55), transparent 34%),
                linear-gradient(135deg, rgba(255, 247, 230, 0.95), rgba(247, 249, 252, 0.96));
            margin-bottom: 1rem;
        }
        .app-hero h1 {
            margin: 0;
            font-size: 2.15rem;
            line-height: 1.05;
        }
        .app-hero p {
            margin: 0.7rem 0 0;
            color: #475467;
            font-size: 1rem;
            max-width: 58rem;
        }
        .section-label {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            font-weight: 700;
            color: #667085;
            margin-bottom: 0.45rem;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .section-copy {
            color: #667085;
            font-size: 0.95rem;
            margin-bottom: 0.9rem;
        }
        .result-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .result-copy {
            color: #667085;
            font-size: 0.85rem;
            margin-bottom: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_page_header() -> None:
    st.markdown(
        """
        <div class="app-hero">
            <div class="app-kicker">Local RAG Workspace</div>
            <h1>Compare vector, graph, and hybrid answers in one place.</h1>
            <p>
                Upload PDFs, build a fast Qdrant index plus a Neo4j knowledge graph, and then inspect how each retrieval
                workflow answers the same question side by side.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_status_strip() -> None:
    stats = st.session_state.graph_stats
    metric_cols = st.columns(4, gap="medium")

    metric_cols[0].metric("Uploaded Files", len(st.session_state.uploaded_paths))
    metric_cols[1].metric("Vector Index", "Ready" if st.session_state.index_ready else "Waiting")
    metric_cols[2].metric("Neo4j Graph", "Ready" if st.session_state.graph_ready else "Waiting")
    metric_cols[3].metric("Graph Nodes", stats.get("nodes", 0) if stats else 0)


def _render_chat_history() -> None:
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(turn["question"])

        with st.chat_message("assistant"):
            _render_response_columns(turn["responses"])


def _render_response_columns(responses: dict) -> None:
    headings = {
        "vector": "Vector RAG",
        "graph": "GraphRAG",
        "hybrid": "Hybrid",
    }
    subtitles = {
        "vector": "Qdrant similarity retrieval grounded with Ollama",
        "graph": "Neo4j graph retrieval grounded by graph context",
        "hybrid": "Neo4j graph plus vector evidence merged together",
    }
    columns = st.columns(3, gap="medium")

    for column, mode in zip(columns, ("vector", "graph", "hybrid")):
        payload = responses.get(mode, {})
        with column:
            with st.container(border=True):
                st.markdown(f"<div class='result-title'>{headings[mode]}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='result-copy'>{subtitles[mode]}</div>", unsafe_allow_html=True)
                if payload.get("error"):
                    st.error(payload["error"])
                else:
                    st.markdown(payload.get("answer") or "No answer returned.")
                    retrieval_path = payload.get("retrieval_path")
                    if retrieval_path:
                        st.caption(f"Retrieval path: {retrieval_path}")
                    if payload.get("sources"):
                        names = ", ".join(Path(source).name for source in payload["sources"])
                        st.caption(f"Sources: {names}")
                    _render_workflow_evidence(payload, mode)


def _render_workflow_evidence(payload: dict, mode: str) -> None:
    qdrant_chunks = payload.get("qdrant_chunks") or []
    neo4j_context = payload.get("neo4j_context") or []
    neo4j_edges = payload.get("neo4j_edges") or []

    if not qdrant_chunks and not neo4j_context and not neo4j_edges:
        return

    with st.expander("Retrieved Evidence", expanded=False):
        if qdrant_chunks:
            st.caption("Qdrant chunks")
            for chunk in qdrant_chunks:
                source_name = Path(chunk["source"]).name
                st.markdown(f"**{source_name}**  `{chunk['chunk_id']}`")
                st.write(chunk["content"][:700])

        if neo4j_context:
            st.caption("Neo4j context lines")
            for line in neo4j_context:
                st.write(f"- {line}")

        if neo4j_edges:
            st.caption("Neo4j graph edges")
            st.dataframe(neo4j_edges, use_container_width=True, hide_index=True)

        if mode == "graph" and not neo4j_context:
            st.info("This graph response did not receive explicit Neo4j context lines.")


def _handle_upload() -> None:
    with st.container(border=True):
        st.markdown("<div class='section-label'>Step 1</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Upload PDF documents</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-copy'>Choose one or more PDFs. Saving a new batch resets the previous index and chat history.</div>",
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if st.button("Save Uploaded Documents", use_container_width=True):
            if not uploaded_files:
                LOGGER.warning("Upload requested without selecting any PDF files")
                st.warning("Select at least one PDF before saving.")
                return

            saved_paths = save_uploaded_files(uploaded_files)
            st.session_state.uploaded_paths = saved_paths
            st.session_state.index_ready = False
            st.session_state.graph_ready = False
            st.session_state.graph_stats = {}
            st.session_state.indexed_files = []
            st.session_state.chat_history = []
            LOGGER.info("Upload workflow complete; %s file(s) ready for indexing", len(saved_paths))
            st.success(f"Saved {len(saved_paths)} document(s) to {CONFIG.upload_dir}.")

        if st.session_state.uploaded_paths:
            st.caption("Ready for indexing")
            for path in st.session_state.uploaded_paths:
                st.write(Path(path).name)


def _handle_indexing() -> None:
    with st.container(border=True):
        st.markdown("<div class='section-label'>Step 2</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Build indexes</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-copy'>Create the Qdrant vector index first, then build the Neo4j graph so the comparison panel has all three retrieval paths available.</div>",
            unsafe_allow_html=True,
        )

        st.caption("Chunk settings")
        chunk_setting_cols = st.columns(2, gap="medium")
        with chunk_setting_cols[0]:
            st.number_input(
                "Chunk size",
                min_value=200,
                step=100,
                key="chunk_size",
                help="Maximum characters per chunk before embedding and graph indexing.",
            )

        if st.session_state.chunk_overlap >= st.session_state.chunk_size:
            st.session_state.chunk_overlap = max(st.session_state.chunk_size // 5, 0)

        with chunk_setting_cols[1]:
            st.number_input(
                "Chunk overlap",
                min_value=0,
                max_value=max(int(st.session_state.chunk_size) - 1, 0),
                step=50,
                key="chunk_overlap",
                help="Shared characters between neighboring chunks.",
            )

        if st.button("Generate Indexes", type="primary", use_container_width=True):
            if not st.session_state.uploaded_paths:
                LOGGER.warning("Indexing requested before any uploaded files were saved")
                st.warning("Upload and save PDF documents first.")
                return

            try:
                LOGGER.info("Indexing workflow started for %s file(s)", len(st.session_state.uploaded_paths))
                selected_chunk_size = int(st.session_state.chunk_size)
                selected_chunk_overlap = int(st.session_state.chunk_overlap)
                progress_bar = st.progress(0.0, text="Preparing documents...")
                status_placeholder = st.empty()

                with st.spinner("Loading PDFs, indexing vectors, and building the Neo4j graph..."):
                    documents = load_pdf_documents(st.session_state.uploaded_paths)
                    chunks = split_documents(
                        documents,
                        chunk_size=selected_chunk_size,
                        chunk_overlap=selected_chunk_overlap,
                    )
                    progress_bar.progress(
                        0.15,
                        text=(
                            f"Loaded and split {len(chunks)} chunks "
                            f"(size={selected_chunk_size}, overlap={selected_chunk_overlap})"
                        ),
                    )
                    embeddings = build_embeddings()
                    vector_service = QdrantService(embeddings=embeddings)
                    vector_service.index_documents(chunks)
                    progress_bar.progress(0.4, text="Vector index stored in Qdrant")
                    neo4j_service = Neo4jGraphRAGService()
                    graph_result = neo4j_service.rebuild_graph(
                        chunks,
                        source_chunk_size=selected_chunk_size,
                        progress_callback=lambda current, total: (
                            progress_bar.progress(
                                0.4 + (0.6 * current / max(total, 1)),
                                text=f"Building Neo4j graph: chunk {current} of {total}",
                            ),
                            status_placeholder.caption(f"Processing Neo4j GraphRAG pipeline for chunk {current}/{total}"),
                        ),
                    )

                st.session_state.index_ready = True
                st.session_state.graph_ready = True
                st.session_state.graph_stats = {
                    "documents": graph_result.documents_processed,
                    "chunks": graph_result.chunks,
                    "nodes": graph_result.nodes,
                    "relationships": graph_result.relationships,
                    "skipped_chunks": graph_result.skipped_chunks,
                    "chunk_size": selected_chunk_size,
                    "chunk_overlap": selected_chunk_overlap,
                }
                st.session_state.indexed_files = [Path(path).name for path in st.session_state.uploaded_paths]
                st.session_state.chat_history = []
                st.success(
                    (
                        f"Indexed {len(chunks)} chunks into Qdrant and built a Neo4j graph with "
                        f"{graph_result.nodes} nodes / {graph_result.relationships} relationships."
                    )
                )
                if graph_result.skipped_chunks:
                    st.warning(
                        f"Neo4j skipped {graph_result.skipped_chunks} chunk(s) because they timed out or failed during graph extraction."
                    )
                progress_bar.progress(1.0, text="Indexing complete")
                status_placeholder.empty()
                LOGGER.info(
                    "Indexing workflow finished: %s chunk(s), %s node(s), %s relationship(s), %s skipped chunk(s)",
                    len(chunks),
                    graph_result.nodes,
                    graph_result.relationships,
                    graph_result.skipped_chunks,
                )
            except VectorStoreConnectionError as exc:
                LOGGER.exception("Indexing workflow failed while connecting to Qdrant")
                st.error(str(exc))
            except Neo4jGraphRAGConnectionError as exc:
                LOGGER.exception("Indexing workflow failed while connecting to Neo4j")
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Indexing workflow failed unexpectedly")
                st.error(f"Indexing failed: {exc}")

        if st.session_state.indexed_files:
            st.caption("Indexed files")
            for file_name in st.session_state.indexed_files:
                st.write(file_name)


def _render_system_panel() -> None:
    with st.container(border=True):
        st.markdown("<div class='section-label'>Workspace</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>System status</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-copy'>Keep these local services running while you work.</div>",
            unsafe_allow_html=True,
        )

        st.write(f"`Qdrant`  {CONFIG.qdrant_url}")
        st.write(f"`Neo4j`  {CONFIG.neo4j_uri}")
        st.write(f"`Ollama chat`  {CONFIG.ollama_model_name}")
        st.write(f"`Ollama embeddings`  {CONFIG.ollama_embedding_model_name}")

        if st.session_state.graph_stats:
            stats = st.session_state.graph_stats
            st.divider()
            st.caption("Current graph snapshot")
            st.write(f"{stats['documents']} document(s)")
            st.write(f"{stats['chunks']} chunk node(s)")
            st.write(f"{stats['nodes']} total node(s)")
            st.write(f"{stats['relationships']} relationship(s)")
            st.write(f"{stats.get('skipped_chunks', 0)} skipped chunk(s)")
            if "chunk_size" in stats:
                st.write(f"Chunk size {stats['chunk_size']} / overlap {stats['chunk_overlap']}")


def _render_graph_explorer() -> None:
    st.markdown("<div class='section-label'>Knowledge Graph</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Inspect what Neo4j actually stored</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-copy'>Use this to verify that the graph contains entities and relationships from the uploaded PDF, instead of relying only on the final answer text.</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.graph_ready:
        st.info("Build the indexes first to inspect the Neo4j graph.")
        return

    try:
        overview = Neo4jGraphRAGService().get_graph_overview(limit=12)
    except Neo4jGraphRAGConnectionError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to load Neo4j graph overview")
        st.error(f"Could not load the graph overview: {exc}")
        return

    metric_cols = st.columns(4, gap="medium")
    metric_cols[0].metric("Graph Nodes", overview["stats"]["nodes"])
    metric_cols[1].metric("Relationships", overview["stats"]["relationships"])
    metric_cols[2].metric("Chunk Nodes", overview["stats"]["chunks"])
    metric_cols[3].metric("Uploaded PDFs", len(st.session_state.uploaded_paths))

    left_col, mid_col, right_col = st.columns([0.9, 0.9, 1.4], gap="medium")
    with left_col:
        st.caption("Node labels")
        st.dataframe(overview["labels"], use_container_width=True, hide_index=True)
    with mid_col:
        st.caption("Relationship types")
        st.dataframe(overview["relationships"], use_container_width=True, hide_index=True)
    with right_col:
        st.caption("Graph edge preview")
        for edge in overview["edges"]:
            st.write(f"{edge['source']} --{edge['relationship']}--> {edge['target']}")


def _handle_chat() -> None:
    with st.container(border=True):
        st.markdown("<div class='section-label'>Step 3</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Ask once, compare three answers</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='section-copy'>Each question runs `{CONFIG.ollama_model_name}` across vector, graph, and hybrid retrieval so you can see tradeoffs immediately.</div>",
            unsafe_allow_html=True,
        )

        _render_chat_history()

    user_question = st.chat_input("Ask a question about the indexed documents")
    if not user_question:
        return

    with st.chat_message("user"):
        st.markdown(user_question)

    if not st.session_state.index_ready or not st.session_state.graph_ready:
        assistant_message = "Generate indexes before starting the chat."
        LOGGER.warning("Chat requested before indexes were ready")
        with st.chat_message("assistant"):
            st.markdown(assistant_message)
        return

    try:
        LOGGER.info("Chat workflow started for question: %s", user_question)
        with st.spinner("Retrieving context and generating an answer..."):
            embeddings = build_embeddings()
            vector_service = QdrantService(embeddings=embeddings)
            vector_store = vector_service.get_vector_store()
            retriever = vector_store.as_retriever(search_kwargs={"k": CONFIG.retrieval_k})
            neo4j_service = Neo4jGraphRAGService()
            chat_service = ChatService(retriever=retriever, neo4j_service=neo4j_service)
            responses = chat_service.ask_all(user_question)

        st.session_state.chat_history.append(
            {
                "question": user_question,
                "responses": responses,
            }
        )

        with st.chat_message("assistant"):
            _render_response_columns(responses)
        LOGGER.info("Chat workflow finished successfully")
    except VectorStoreConnectionError as exc:
        LOGGER.exception("Chat workflow failed while connecting to Qdrant")
        st.error(str(exc))
    except Neo4jGraphRAGConnectionError as exc:
        LOGGER.exception("Chat workflow failed while connecting to Neo4j")
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Chat workflow failed unexpectedly")
        st.error(f"Chat request failed: {exc}")


def run_app() -> None:
    LOGGER.info("Starting Streamlit UI")
    st.set_page_config(page_title="Local RAG with Streamlit", layout="wide")
    init_session_state()
    _inject_styles()
    _render_page_header()
    _render_status_strip()
    st.divider()

    control_col, build_col, system_col = st.columns([1.15, 1.15, 0.8], gap="large")

    with control_col:
        _handle_upload()

    with build_col:
        _handle_indexing()

    with system_col:
        _render_system_panel()

    st.divider()
    _render_graph_explorer()
    st.divider()
    _handle_chat()

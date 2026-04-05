from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from rag_app.llm import build_chat_llm
from rag_app.logging_utils import get_logger
from rag_app.neo4j_graphrag_service import Neo4jGraphRAGService

LOGGER = get_logger("chat")


@dataclass(slots=True)
class ChatService:
    retriever: object
    neo4j_service: Neo4jGraphRAGService

    def ask_vector(self, question: str) -> dict[str, Any]:
        LOGGER.info("Running vector workflow for question: %s", question)
        vector_docs = self.retriever.invoke(question)
        prompt = self._build_vector_prompt(question, vector_docs)
        result = build_chat_llm().invoke(prompt)

        sources = []
        for doc in vector_docs:
            source = doc.metadata.get("source")
            if source and source not in sources:
                sources.append(source)

        LOGGER.info("Vector workflow completed with %s source(s)", len(sources))
        return {
            "answer": self._coerce_content(result.content),
            "sources": sources,
            "error": None,
            "retrieval_path": "qdrant_similarity",
            "qdrant_chunks": self._serialize_vector_docs(vector_docs),
            "neo4j_context": [],
            "neo4j_edges": [],
        }

    def ask_graph(self, question: str) -> dict[str, Any]:
        LOGGER.info("Running graph workflow for question: %s", question)
        graph_result = self.neo4j_service.ask_graph(question)
        LOGGER.info("Graph workflow completed with %s source(s)", len(graph_result.sources))
        return {
            "answer": graph_result.answer,
            "sources": graph_result.sources,
            "error": None,
            "retrieval_path": graph_result.retrieval_mode,
            "qdrant_chunks": [],
            "neo4j_context": graph_result.context_lines,
            "neo4j_edges": graph_result.preview_edges,
        }

    def ask_hybrid(self, question: str) -> dict[str, Any]:
        LOGGER.info("Running hybrid workflow for question: %s", question)
        vector_docs = self.retriever.invoke(question)
        graph_result = self.neo4j_service.get_graph_context(question)
        prompt = self._build_hybrid_prompt(question, vector_docs, graph_result.context_lines)
        result = build_chat_llm().invoke(prompt)

        sources = list(graph_result.sources)
        for doc in vector_docs:
            source = doc.metadata.get("source")
            if source and source not in sources:
                sources.append(source)

        answer = self._coerce_content(result.content)
        LOGGER.info("Hybrid workflow completed with %s source(s)", len(sources))
        return {
            "answer": answer,
            "sources": sources,
            "error": None,
            "retrieval_path": f"qdrant_plus_{graph_result.retrieval_mode}",
            "qdrant_chunks": self._serialize_vector_docs(vector_docs),
            "neo4j_context": graph_result.context_lines,
            "neo4j_edges": graph_result.preview_edges,
        }

    def ask(self, question: str, mode: str) -> dict[str, Any]:
        if mode == "graph":
            return self.ask_graph(question)
        if mode == "hybrid":
            return self.ask_hybrid(question)
        return self.ask_vector(question)

    def ask_all(self, question: str) -> dict[str, dict[str, Any]]:
        modes = ("vector", "graph", "hybrid")
        results: dict[str, dict[str, Any]] = {}
        LOGGER.info("Executing side-by-side workflows for question: %s", question)

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_mode = {
                executor.submit(self.ask, question, mode): mode
                for mode in modes
            }

            for future in as_completed(future_to_mode):
                mode = future_to_mode[future]
                try:
                    results[mode] = future.result()
                    LOGGER.info("Workflow '%s' finished successfully", mode)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Workflow '%s' failed", mode)
                    results[mode] = {
                        "answer": "",
                        "sources": [],
                        "error": str(exc),
                        "retrieval_path": "failed",
                        "qdrant_chunks": [],
                        "neo4j_context": [],
                        "neo4j_edges": [],
                    }

        LOGGER.info("Completed side-by-side workflows for question")
        return {mode: results[mode] for mode in modes}

    def _build_vector_prompt(
        self,
        question: str,
        vector_docs: list[Document],
    ) -> list[SystemMessage | HumanMessage]:
        context = "\n\n".join(
            [
                f"Chunk from {Path(doc.metadata.get('source', 'unknown-source')).name}:\n{doc.page_content}"
                for doc in vector_docs
            ]
        )
        if not context:
            context = "No vector context retrieved."

        return [
            SystemMessage(
                content=(
                    "Answer the user's question using only the supplied document context. "
                    "If the context is insufficient, say that clearly."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{question}\n\n"
                    f"Context:\n{context}\n\n"
                    "Provide a concise grounded answer."
                )
            ),
        ]

    def _build_hybrid_prompt(
        self,
        question: str,
        vector_docs: list[Document],
        graph_context: list[str],
    ) -> list[SystemMessage | HumanMessage]:
        vector_context = "\n\n".join(
            [
                f"Qdrant chunk from {Path(doc.metadata.get('source', 'unknown-source')).name}:\n{doc.page_content}"
                for doc in vector_docs
            ]
        )
        if not vector_context:
            vector_context = "No Qdrant vector context retrieved."

        graph_text = "\n".join(graph_context) if graph_context else "No Neo4j graph context retrieved."

        return [
            SystemMessage(
                content=(
                    "Answer the user's question using both the vector context and the graph context. "
                    "Prefer direct evidence. If the two disagree, say so clearly."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{question}\n\n"
                    f"Qdrant vector context:\n{vector_context}\n\n"
                    f"Neo4j graph context:\n{graph_text}\n\n"
                    "Provide a concise grounded answer."
                )
            ),
        ]

    @staticmethod
    def _coerce_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(getattr(item, "text", str(item)) for item in content)
        return str(content)

    @staticmethod
    def _serialize_vector_docs(vector_docs: list[Document]) -> list[dict[str, str]]:
        serialized_docs: list[dict[str, str]] = []
        for doc in vector_docs:
            serialized_docs.append(
                {
                    "source": doc.metadata.get("source", "unknown-source"),
                    "chunk_id": doc.metadata.get("chunk_id", "unknown-chunk"),
                    "content": doc.page_content,
                }
            )
        return serialized_docs

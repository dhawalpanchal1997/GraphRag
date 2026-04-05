from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
import re
import threading
from typing import Callable

from langchain_core.documents import Document
from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError
from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings as Neo4jOllamaEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.indexes import create_vector_index, drop_index_if_exists
from neo4j_graphrag.llm.ollama_llm import OllamaLLM
from neo4j_graphrag.retrievers import Text2CypherRetriever
from langchain_core.messages import HumanMessage, SystemMessage

from rag_app.config import CONFIG
from rag_app.llm import build_chat_llm
from rag_app.logging_utils import get_logger

LOGGER = get_logger("neo4j_graphrag")

_QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


class Neo4jGraphRAGConnectionError(RuntimeError):
    """Raised when Neo4j is not reachable."""


@dataclass(slots=True)
class Neo4jGraphBuildResult:
    documents_processed: int
    nodes: int
    relationships: int
    chunks: int
    skipped_chunks: int = 0


@dataclass(slots=True)
class GraphContextResult:
    answer: str = ""
    context_lines: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    retrieval_mode: str = "unknown"
    preview_edges: list[dict[str, str]] = field(default_factory=list)


class Neo4jGraphRAGService:
    def connect(self):
        LOGGER.info("Connecting to Neo4j at '%s'", CONFIG.neo4j_uri)
        try:
            driver = GraphDatabase.driver(
                CONFIG.neo4j_uri,
                auth=(CONFIG.neo4j_username, CONFIG.neo4j_password),
            )
            driver.verify_connectivity()
            LOGGER.info("Connected to Neo4j successfully")
            return driver
        except (Neo4jError, DriverError, ValueError, OSError) as exc:
            LOGGER.exception("Failed to connect to Neo4j")
            raise Neo4jGraphRAGConnectionError(
                "Unable to connect to Neo4j. Start the Docker container and verify the credentials."
            ) from exc

    def rebuild_graph(
        self,
        chunks: list[Document],
        source_chunk_size: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Neo4jGraphBuildResult:
        if not chunks:
            raise ValueError("No document chunks were provided for Neo4j GraphRAG indexing.")

        driver = self.connect()
        try:
            source_paths = {
                str(chunk.metadata.get("source"))
                for chunk in chunks
                if chunk.metadata.get("source")
            }
            LOGGER.info(
                "Rebuilding Neo4j graph from %s chunk(s) across %s source file(s)",
                len(chunks),
                len(source_paths),
            )
            self._clear_graph(driver)

            total_chunks = len(chunks)
            skipped_chunks = 0
            for index, chunk in enumerate(chunks, start=1):
                source = str(chunk.metadata.get("source", ""))
                file_name = Path(source).name if source else "uploaded-document"
                chunk_id = str(chunk.metadata.get("chunk_id", f"chunk-{index - 1}"))
                document_key = f"{source}#{chunk_id}" if source else chunk_id
                LOGGER.info("Processing Neo4j KG pipeline for chunk '%s' (%s/%s)", chunk_id, index, total_chunks)
                try:
                    self._run_async(
                        lambda source=source, file_name=file_name, chunk_id=chunk_id, document_key=document_key, chunk_text=chunk.page_content, chunk_position=index - 1: self._build_pipeline(
                            driver,
                            from_pdf=False,
                            source_chunk_size=source_chunk_size,
                        ).run_async(
                            file_path=document_key,
                            text=chunk_text,
                            document_metadata={
                                "source": source,
                                "file_name": file_name,
                                "chunk_id": chunk_id,
                                "chunk_index": str(chunk_position),
                            },
                        ),
                        timeout_seconds=CONFIG.graph_chunk_timeout_seconds,
                    )
                except TimeoutError:
                    skipped_chunks += 1
                    LOGGER.warning(
                        "Timed out while building graph for chunk '%s' after %ss; skipping it",
                        chunk_id,
                        CONFIG.graph_chunk_timeout_seconds,
                    )
                except Exception as exc:  # noqa: BLE001
                    skipped_chunks += 1
                    LOGGER.warning("Failed to build graph for chunk '%s'; skipping it: %s", chunk_id, exc)
                if progress_callback is not None:
                    progress_callback(index, total_chunks)

            self._ensure_vector_index(driver)
            stats = self._fetch_stats(driver)
            LOGGER.info(
                "Neo4j graph rebuild complete: %s document(s), %s node(s), %s relationship(s), %s chunk(s)",
                len(source_paths) or 1,
                stats["nodes"],
                stats["relationships"],
                stats["chunks"],
            )
            return Neo4jGraphBuildResult(
                documents_processed=len(source_paths) or 1,
                nodes=stats["nodes"],
                relationships=stats["relationships"],
                chunks=stats["chunks"],
                skipped_chunks=skipped_chunks,
            )
        finally:
            driver.close()

    def ask_graph(self, question: str) -> GraphContextResult:
        LOGGER.info("Running Neo4j graph retrieval for question: %s", question)
        graph_result = self.get_graph_context(question)
        graph_result.answer = self._synthesize_fallback_answer(question, graph_result.context_lines, mode="graph")
        LOGGER.info("Neo4j graph retrieval completed with %s source(s)", len(graph_result.sources))
        return graph_result

    def get_graph_context(self, question: str) -> GraphContextResult:
        LOGGER.info("Retrieving Neo4j graph context for question: %s", question)
        driver = self.connect()
        try:
            llm = self._build_neo4j_llm()
            retriever = Text2CypherRetriever(
                driver=driver,
                llm=llm,
                neo4j_database=CONFIG.neo4j_database,
            )
            graph_rag = GraphRAG(retriever=retriever, llm=llm)
            result = graph_rag.search(
                query_text=question,
                return_context=True,
                response_fallback="I could not find enough graph context to answer that question.",
            )
            context_lines = self._extract_context_lines(result.retriever_result.items if result.retriever_result else [])
            sources = self._extract_sources(result.retriever_result.items if result.retriever_result else [])
            if not context_lines:
                LOGGER.info("Text2Cypher returned no graph context; switching to deterministic fallback")
                rows = self._search_graph_neighborhood(driver, question)
                context_lines, sources = self._format_graph_rows(rows)
                return GraphContextResult(
                    context_lines=context_lines,
                    sources=sources,
                    retrieval_mode="deterministic_fallback",
                    preview_edges=self._preview_edges_from_rows(rows),
                )
            LOGGER.info("Neo4j graph context retrieval completed with %s context line(s)", len(context_lines))
            return GraphContextResult(
                context_lines=context_lines,
                sources=sources,
                retrieval_mode="text2cypher",
                preview_edges=self.get_graph_preview(question=question, limit=max(CONFIG.graph_max_edges, 8)),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Text2Cypher graph retrieval failed, using deterministic graph fallback: %s", exc)
            rows = self._search_graph_neighborhood(driver, question)
            context_lines, sources = self._format_graph_rows(rows)
            return GraphContextResult(
                context_lines=context_lines,
                sources=sources,
                retrieval_mode="deterministic_fallback",
                preview_edges=self._preview_edges_from_rows(rows),
            )
        finally:
            driver.close()

    def _clear_graph(self, driver) -> None:
        LOGGER.info("Clearing existing Neo4j graph data")
        driver.execute_query(
            "MATCH (n) DETACH DELETE n",
            database_=CONFIG.neo4j_database,
        )
        try:
            drop_index_if_exists(
                driver,
                CONFIG.neo4j_vector_index_name,
                neo4j_database=CONFIG.neo4j_database,
            )
        except Exception:  # noqa: BLE001
            LOGGER.debug("Neo4j vector index '%s' did not need dropping", CONFIG.neo4j_vector_index_name)

    def _ensure_vector_index(self, driver) -> None:
        dimensions = len(self._build_neo4j_embedder().embed_query("dimension probe"))
        LOGGER.info(
            "Creating Neo4j vector index '%s' with %s dimensions",
            CONFIG.neo4j_vector_index_name,
            dimensions,
        )
        create_vector_index(
            driver,
            CONFIG.neo4j_vector_index_name,
            label=CONFIG.neo4j_chunk_label,
            embedding_property=CONFIG.neo4j_chunk_embedding_property,
            dimensions=dimensions,
            similarity_fn="cosine",
            neo4j_database=CONFIG.neo4j_database,
        )

    def _build_pipeline(self, driver, from_pdf: bool, source_chunk_size: int) -> SimpleKGPipeline:
        LOGGER.info("Initializing Neo4j SimpleKGPipeline with schema mode '%s'", CONFIG.neo4j_schema_mode)
        return SimpleKGPipeline(
            llm=self._build_neo4j_llm(),
            driver=driver,
            embedder=self._build_neo4j_embedder(),
            schema=CONFIG.neo4j_schema_mode,
            from_pdf=from_pdf,
            text_splitter=FixedSizeSplitter(
                chunk_size=max(source_chunk_size * 2, 5000),
                chunk_overlap=0,
            ),
            on_error="IGNORE",
            perform_entity_resolution=True,
            neo4j_database=CONFIG.neo4j_database,
        )

    def _build_neo4j_llm(self) -> OllamaLLM:
        return OllamaLLM(
            model_name=CONFIG.ollama_model_name,
            host=CONFIG.ollama_base_url,
            model_params={"options": {"temperature": 0}},
            timeout=CONFIG.graph_llm_timeout_seconds,
        )

    def _build_neo4j_embedder(self) -> Neo4jOllamaEmbeddings:
        return Neo4jOllamaEmbeddings(
            model=CONFIG.ollama_embedding_model_name,
            host=CONFIG.ollama_base_url,
        )

    def _fetch_stats(self, driver) -> dict[str, int]:
        node_records, _, _ = driver.execute_query(
            "MATCH (n) RETURN count(n) AS count",
            database_=CONFIG.neo4j_database,
        )
        relationship_records, _, _ = driver.execute_query(
            "MATCH ()-[r]->() RETURN count(r) AS count",
            database_=CONFIG.neo4j_database,
        )
        chunk_records, _, _ = driver.execute_query(
            f"MATCH (c:{CONFIG.neo4j_chunk_label}) RETURN count(c) AS count",
            database_=CONFIG.neo4j_database,
        )
        return {
            "nodes": int(node_records[0]["count"]) if node_records else 0,
            "relationships": int(relationship_records[0]["count"]) if relationship_records else 0,
            "chunks": int(chunk_records[0]["count"]) if chunk_records else 0,
        }

    def get_graph_overview(self, limit: int = 12) -> dict[str, object]:
        LOGGER.info("Fetching Neo4j graph overview")
        driver = self.connect()
        try:
            stats = self._fetch_stats(driver)
            label_rows, _, _ = driver.execute_query(
                """
                MATCH (n)
                RETURN labels(n) AS labels, count(*) AS count
                ORDER BY count DESC
                LIMIT $limit
                """,
                limit=limit,
                database_=CONFIG.neo4j_database,
            )
            relationship_rows, _, _ = driver.execute_query(
                """
                MATCH ()-[r]->()
                RETURN type(r) AS relationship, count(*) AS count
                ORDER BY count DESC
                LIMIT $limit
                """,
                limit=limit,
                database_=CONFIG.neo4j_database,
            )

            return {
                "stats": stats,
                "labels": [
                    {"labels": ", ".join(row["labels"]), "count": int(row["count"])}
                    for row in label_rows
                ],
                "relationships": [
                    {"relationship": row["relationship"], "count": int(row["count"])}
                    for row in relationship_rows
                ],
                "edges": self.get_graph_preview(limit=limit),
            }
        finally:
            driver.close()

    def get_graph_preview(self, question: str | None = None, limit: int = 10) -> list[dict[str, str]]:
        driver = self.connect()
        try:
            if question:
                rows = self._search_graph_neighborhood(driver, question)
                return self._preview_edges_from_rows(rows[:limit])

            rows, _, _ = driver.execute_query(
                """
                MATCH (n)-[r]->(m)
                RETURN
                  labels(n) AS source_labels,
                  properties(n) AS source_props,
                  type(r) AS relationship,
                  labels(m) AS target_labels,
                  properties(m) AS target_props
                LIMIT $limit
                """,
                limit=limit,
                database_=CONFIG.neo4j_database,
            )
            return [
                {
                    "source": self._best_entity_name(row.get("source_props") or {}),
                    "source_labels": ", ".join(row.get("source_labels") or []),
                    "relationship": row.get("relationship") or "RELATED_TO",
                    "target": self._best_entity_name(row.get("target_props") or {}),
                    "target_labels": ", ".join(row.get("target_labels") or []),
                }
                for row in rows
            ]
        finally:
            driver.close()

    def _extract_sources(self, items) -> list[str]:
        sources: list[str] = []
        for item in items:
            metadata = getattr(item, "metadata", None) or {}

            candidate_values = []
            for key in ("source", "path", "file_path", "document_path", "file_name"):
                value = metadata.get(key)
                if isinstance(value, str):
                    candidate_values.append(value)

            for value in metadata.values():
                if isinstance(value, str) and value.lower().endswith(".pdf"):
                    candidate_values.append(value)

            for source in candidate_values:
                if source not in sources:
                    sources.append(source)

        return sources

    def _search_graph_neighborhood(self, driver, question: str) -> list[dict]:
        search_terms = self._extract_search_terms(question)
        LOGGER.info("Running deterministic graph neighborhood search with %s term(s)", len(search_terms))
        rows, _, _ = driver.execute_query(
            """
            UNWIND $terms AS term
            MATCH (n)
            WHERE any(
              k IN keys(n)
              WHERE k <> $embedding_property
                AND n[k] IS NOT NULL
                AND (
                  n[k] IS :: STRING
                  OR n[k] IS :: INTEGER
                  OR n[k] IS :: FLOAT
                  OR n[k] IS :: BOOLEAN
                )
                AND toLower(toString(n[k])) CONTAINS term
            )
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN DISTINCT
              labels(n) AS node_labels,
              properties(n) AS node_props,
              type(r) AS rel_type,
              labels(m) AS neighbor_labels,
              properties(m) AS neighbor_props
            LIMIT $limit
            """,
            terms=search_terms,
            limit=max(CONFIG.graph_max_edges, 8),
            embedding_property=CONFIG.neo4j_chunk_embedding_property,
            database_=CONFIG.neo4j_database,
        )

        if rows:
            return rows

        fallback_rows, _, _ = driver.execute_query(
            """
            MATCH (n)-[r]-(m)
            RETURN DISTINCT
              labels(n) AS node_labels,
              properties(n) AS node_props,
              type(r) AS rel_type,
              labels(m) AS neighbor_labels,
              properties(m) AS neighbor_props
            LIMIT $limit
            """,
            limit=max(CONFIG.graph_max_edges, 8),
            database_=CONFIG.neo4j_database,
        )
        return fallback_rows

    def _format_graph_rows(self, rows) -> tuple[list[str], list[str]]:
        context_lines: list[str] = []
        sources: list[str] = []

        for row in rows:
            node_props = row.get("node_props") or {}
            neighbor_props = row.get("neighbor_props") or {}
            node_name = self._best_entity_name(node_props)
            neighbor_name = self._best_entity_name(neighbor_props)
            rel_type = row.get("rel_type") or "RELATED_TO"

            if neighbor_name:
                context_lines.append(f"{node_name} --{rel_type}-- {neighbor_name}")
            else:
                context_lines.append(f"{node_name} [matched node]")

            for props in (node_props, neighbor_props):
                source = self._extract_source_from_props(props)
                if source and source not in sources:
                    sources.append(source)

        if not context_lines:
            context_lines.append("No graph neighborhood context was found.")

        return context_lines, sources

    def _preview_edges_from_rows(self, rows) -> list[dict[str, str]]:
        preview_edges: list[dict[str, str]] = []
        for row in rows:
            node_props = row.get("node_props") or {}
            neighbor_props = row.get("neighbor_props") or {}
            preview_edges.append(
                {
                    "source": self._best_entity_name(node_props),
                    "source_labels": ", ".join(row.get("node_labels") or []),
                    "relationship": row.get("rel_type") or "RELATED_TO",
                    "target": self._best_entity_name(neighbor_props),
                    "target_labels": ", ".join(row.get("neighbor_labels") or []),
                }
            )
        return preview_edges

    @staticmethod
    def _extract_context_lines(items) -> list[str]:
        context_lines: list[str] = []
        for item in items:
            content = getattr(item, "content", None)
            if isinstance(content, str) and content.strip():
                context_lines.append(content.strip())
        return context_lines

    def _synthesize_fallback_answer(self, question: str, context_lines: list[str], mode: str) -> str:
        context = "\n".join(context_lines) if context_lines else "No graph context found."
        response = build_chat_llm().invoke(
            [
                SystemMessage(
                    content=(
                        "You are answering from Neo4j retrieval context. "
                        "Use only the provided context. If it is sparse or uncertain, say so clearly."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Mode: {mode}\n\n"
                        f"Question:\n{question}\n\n"
                        f"Context:\n{context}\n\n"
                        "Provide a concise grounded answer."
                    )
                ),
            ]
        )
        return self._coerce_content(response.content)

    @staticmethod
    def _extract_search_terms(question: str) -> list[str]:
        terms: list[str] = []
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]+", question.lower()):
            if token in _QUESTION_STOPWORDS or len(token) < 3:
                continue
            if token not in terms:
                terms.append(token)
        return terms or [question.strip().lower()]

    @staticmethod
    def _best_entity_name(properties: dict) -> str:
        for key in ("name", "title", "id", "path", "file_name"):
            value = properties.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return str(properties)[:120]

    @staticmethod
    def _extract_source_from_props(properties: dict) -> str | None:
        for key in ("source", "path", "file_path", "document_path", "file_name"):
            value = properties.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None

    @staticmethod
    def _coerce_content(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(getattr(item, "text", str(item)) for item in content)
        return str(content)

    @staticmethod
    def _run_async(coroutine_factory: Callable[[], object], timeout_seconds: float | None = None):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine_factory())

        result: dict[str, object] = {}
        error: dict[str, BaseException] = {}

        def runner() -> None:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                result["value"] = loop.run_until_complete(coroutine_factory())
            except BaseException as exc:  # noqa: BLE001
                error["value"] = exc
            finally:
                loop.close()

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            raise TimeoutError(f"Timed out waiting for async graph task after {timeout_seconds}s")

        if "value" in error:
            raise error["value"]
        return result.get("value")

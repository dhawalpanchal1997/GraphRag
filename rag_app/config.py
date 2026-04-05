from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(slots=True)
class AppConfig:
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    upload_dir_name: str = "data/uploads"
    graph_dir_name: str = "data/graph"
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_embedding_model_name: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    ollama_model_name: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password12345")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    neo4j_vector_index_name: str = os.getenv("NEO4J_VECTOR_INDEX_NAME", "chunk-vector-index")
    neo4j_schema_mode: str = os.getenv("NEO4J_SCHEMA_MODE", "FREE")
    neo4j_chunk_label: str = "Chunk"
    neo4j_chunk_embedding_property: str = "embedding"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_k: int = 4
    graph_seed_nodes: int = 3
    graph_max_edges: int = 8
    graph_max_triplets_per_chunk: int = 5
    graph_llm_timeout_seconds: float = 20.0
    graph_chunk_timeout_seconds: float = float(os.getenv("GRAPH_CHUNK_TIMEOUT_SECONDS", "60"))
    request_timeout: float = 10.0

    @property
    def upload_dir(self) -> Path:
        return self.base_dir / self.upload_dir_name

    @property
    def graph_dir(self) -> Path:
        return self.base_dir / self.graph_dir_name

    @property
    def graph_index_path(self) -> Path:
        return self.graph_dir / "knowledge_graph.json"


CONFIG = AppConfig()

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import exceptions as qdrant_exceptions
from qdrant_client.http.models import Distance, VectorParams

from rag_app.config import CONFIG
from rag_app.logging_utils import get_logger

LOGGER = get_logger("vector_store")


class VectorStoreConnectionError(RuntimeError):
    """Raised when Qdrant is not reachable."""


@dataclass(slots=True)
class QdrantService:
    embeddings: Embeddings

    def connect(self) -> QdrantClient:
        LOGGER.info("Connecting to Qdrant at '%s'", CONFIG.qdrant_url)
        try:
            client = QdrantClient(url=CONFIG.qdrant_url, timeout=CONFIG.request_timeout)
            client.get_collections()
            LOGGER.info("Connected to Qdrant successfully")
            return client
        except (qdrant_exceptions.UnexpectedResponse, ValueError, OSError) as exc:
            LOGGER.exception("Failed to connect to Qdrant")
            raise VectorStoreConnectionError(
                "Unable to connect to Qdrant. Start the Docker container and verify the URL."
            ) from exc

    def _embedding_dimension(self) -> int:
        dimension = len(self.embeddings.embed_query("dimension probe"))
        LOGGER.info("Detected embedding dimension: %s", dimension)
        return dimension

    def recreate_collection(self, client: QdrantClient) -> None:
        collection_name = CONFIG.qdrant_collection
        LOGGER.info("Recreating Qdrant collection '%s'", collection_name)
        if client.collection_exists(collection_name):
            LOGGER.info("Deleting existing collection '%s'", collection_name)
            client.delete_collection(collection_name)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self._embedding_dimension(),
                distance=Distance.COSINE,
            ),
        )
        LOGGER.info("Created collection '%s'", collection_name)

    def index_documents(self, documents: list[Document]) -> QdrantVectorStore:
        if not documents:
            raise ValueError("No document chunks were created for indexing.")

        LOGGER.info("Indexing %s document chunk(s) into Qdrant", len(documents))
        client = self.connect()
        self.recreate_collection(client)

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=CONFIG.qdrant_collection,
            embedding=self.embeddings,
        )
        vector_store.add_documents(documents)
        LOGGER.info("Finished indexing %s chunk(s) into collection '%s'", len(documents), CONFIG.qdrant_collection)
        return vector_store

    def get_vector_store(self) -> QdrantVectorStore:
        client = self.connect()
        if not client.collection_exists(CONFIG.qdrant_collection):
            raise ValueError(
                f"Collection '{CONFIG.qdrant_collection}' does not exist yet. Generate embeddings first."
            )

        LOGGER.info("Opening existing Qdrant collection '%s' for retrieval", CONFIG.qdrant_collection)
        return QdrantVectorStore(
            client=client,
            collection_name=CONFIG.qdrant_collection,
            embedding=self.embeddings,
        )

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from uuid import uuid4

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_app.config import CONFIG
from rag_app.logging_utils import get_logger

LOGGER = get_logger("document_loader")


def ensure_upload_dir() -> Path:
    CONFIG.upload_dir.mkdir(parents=True, exist_ok=True)
    return CONFIG.upload_dir


def save_uploaded_files(uploaded_files: Iterable) -> list[Path]:
    saved_paths: list[Path] = []
    upload_dir = ensure_upload_dir()
    file_names = [uploaded_file.name for uploaded_file in uploaded_files]
    LOGGER.info("Saving %s uploaded file(s): %s", len(file_names), ", ".join(file_names) or "<none>")

    for uploaded_file in uploaded_files:
        file_name = f"{uuid4().hex}_{uploaded_file.name}"
        destination = upload_dir / file_name
        destination.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(destination)
        LOGGER.info("Saved upload '%s' to '%s'", uploaded_file.name, destination)

    return saved_paths


def load_pdf_documents(file_paths: Iterable[Path]) -> list[Document]:
    documents: list[Document] = []

    for file_path in file_paths:
        LOGGER.info("Loading PDF document from '%s'", file_path)
        loader = UnstructuredPDFLoader(str(file_path), mode="elements", languages=["eng"])
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source"] = str(file_path)
        documents.extend(loaded_docs)
        LOGGER.info("Loaded %s document element(s) from '%s'", len(loaded_docs), file_path.name)

    LOGGER.info("Loaded %s total document element(s)", len(documents))
    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    resolved_chunk_size = chunk_size or CONFIG.chunk_size
    resolved_chunk_overlap = chunk_overlap if chunk_overlap is not None else CONFIG.chunk_overlap
    LOGGER.info(
        "Splitting %s document element(s) with chunk_size=%s and chunk_overlap=%s",
        len(documents),
        resolved_chunk_size,
        resolved_chunk_overlap,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=resolved_chunk_size,
        chunk_overlap=resolved_chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk-{index}"

    LOGGER.info("Created %s chunk(s) for indexing", len(chunks))
    return chunks

from __future__ import annotations

import math
import uuid

from config import Settings
from services.embedding_service import EmbeddingService


class VectorService:
    """Embedded Chroma collection with an explicit isolated collection namespace."""

    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingService | None = None,
        client=None,
        collection_name: str | None = None,
    ):
        self.settings = settings
        self.embedder = embedder or EmbeddingService(settings)
        self.collection_name = collection_name or f"paper_chunks_{uuid.uuid4().hex[:16]}"
        settings.ensure_data_dirs()
        if client is None:
            import chromadb

            # No HTTP Chroma service. Collection names below isolate Streamlit sessions/runs;
            # Chroma's in-memory system can itself be shared by clients in one Python process.
            client = chromadb.EphemeralClient()
        self.client = client
        self.collection = self.client.get_or_create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[dict], batch_size: int = 32) -> int:
        if not chunks:
            return 0
        if batch_size < 1 or batch_size > 128:
            raise ValueError("batch_size must be between 1 and 128")
        unique: dict[str, dict] = {}
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id") or "")
            text = str(chunk.get("text") or "")
            metadata = chunk.get("metadata") or {}
            if not chunk_id or not text or not metadata.get("paper_id"):
                raise ValueError("Every indexed chunk needs a chunk_id, text, and paper_id.")
            old = unique.get(chunk_id)
            if old and (old["text"] != text or old["metadata"] != metadata):
                raise ValueError(f"Conflicting chunk payloads share chunk_id {chunk_id}.")
            unique.setdefault(chunk_id, {"chunk_id": chunk_id, "text": text, "metadata": metadata})

        existing = self.collection.get(ids=list(unique), include=["documents", "metadatas"])
        existing_by_id = {
            chunk_id: (document, metadata or {})
            for chunk_id, document, metadata in zip(
                existing.get("ids", []), existing.get("documents", []), existing.get("metadatas", []),
            )
        }
        items = []
        for chunk_id, chunk in unique.items():
            old = existing_by_id.get(chunk_id)
            expected = {**chunk["metadata"], "chunk_id": chunk_id}
            if old:
                old_text, old_metadata = old
                if old_text != chunk["text"] or old_metadata != expected:
                    raise ValueError(f"Stored chunk {chunk_id} conflicts with new content or citation metadata.")
                continue
            items.append(chunk)
        indexed = 0
        for offset in range(0, len(items), batch_size):
            batch = items[offset:offset + batch_size]
            embeddings = self.embedder.embed([chunk["text"] for chunk in batch])
            if len(embeddings) != len(batch):
                raise RuntimeError("Embedding provider returned a mismatched vector count.")
            if any(not vector or any(not math.isfinite(float(value)) for value in vector) for vector in embeddings):
                raise RuntimeError("Embedding provider returned an empty or non-finite vector.")
            metadata_rows = []
            for chunk in batch:
                metadata = {**chunk["metadata"], "chunk_id": chunk["chunk_id"]}
                metadata_rows.append({key: value for key, value in metadata.items() if value is not None})
            self.collection.upsert(
                ids=[chunk["chunk_id"] for chunk in batch],
                embeddings=embeddings,
                documents=[chunk["text"] for chunk in batch],
                metadatas=metadata_rows,
            )
            indexed += len(batch)
        return indexed

    def cleanup(self) -> None:
        """Delete this session/run collection to release in-memory Chroma records."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            # Cleanup is best-effort; indexing and retrieval outcomes are reported separately.
            pass

    def query(self, query: str, limit: int = 8, paper_ids: list[str] | None = None) -> list[dict]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if not query.strip() or (paper_ids is not None and not paper_ids):
            return []
        where = {"paper_id": {"$in": sorted(set(paper_ids))}} if paper_ids is not None else None
        vectors = self.embedder.embed([query])
        if len(vectors) != 1 or not vectors[0]:
            raise RuntimeError("Embedding provider returned an invalid query vector.")
        vector = vectors[0]
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        rows = []
        for doc, metadata, distance in zip(
            (result.get("documents") or [[]])[0],
            (result.get("metadatas") or [[]])[0],
            (result.get("distances") or [[]])[0],
        ):
            rows.append({"text": doc, "metadata": metadata, "score": 1.0 - float(distance)})
        return rows

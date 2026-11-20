from __future__ import annotations

from config import Settings
from services.embedding_service import EmbeddingService


class VectorService:
    def __init__(self, settings: Settings, embedder: EmbeddingService | None = None):
        self.settings = settings
        self.embedder = embedder or EmbeddingService(settings)
        settings.ensure_data_dirs()
        import chromadb

        self.client = chromadb.PersistentClient(path=str(settings.data_dir / "chroma"))
        self.collection = self.client.get_or_create_collection(
            "paper_chunks",
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[dict], batch_size: int = 32) -> int:
        if not chunks:
            return 0
        if batch_size < 1 or batch_size > 128:
            raise ValueError("batch_size must be between 1 and 128")
        indexed = 0
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset:offset + batch_size]
            embeddings = self.embedder.embed([chunk["text"] for chunk in batch])
            if len(embeddings) != len(batch):
                raise RuntimeError("Embedding provider returned a mismatched vector count.")
            self.collection.upsert(
                ids=[chunk["chunk_id"] for chunk in batch],
                embeddings=embeddings,
                documents=[chunk["text"] for chunk in batch],
                metadatas=[{**chunk["metadata"], "chunk_id": chunk["chunk_id"]} for chunk in batch],
            )
            indexed += len(batch)
        return indexed

    def query(self, query: str, limit: int = 8, paper_ids: list[str] | None = None) -> list[dict]:
        where = {"paper_id": {"$in": paper_ids}} if paper_ids else None
        vector = self.embedder.embed([query])[0]
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

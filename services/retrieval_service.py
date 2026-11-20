from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any

from schemas.models import Paper


def chunk_pages(
    paper: Paper, pages: list[dict], chunk_words: int = 220, overlap_words: int = 35,
) -> list[dict[str, Any]]:
    chunks = []
    for item in pages:
        words = item.get("text", "").split()
        if not words:
            continue
        page = item.get("page")
        start = 0
        index = 0
        while start < len(words):
            text = " ".join(words[start:start + chunk_words])
            if not text:
                break
            raw_id = f"{paper.paper_id}|{page}|{index}|{text[:100]}"
            chunk_id = hashlib.sha256(raw_id.encode()).hexdigest()
            chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "page": int(page) if page is not None else -1,
                    "source_url": paper.url or "",
                    "location": item.get("location", "full-text" if page is not None else "abstract"),
                },
            })
            if start + chunk_words >= len(words):
                break
            start += max(1, chunk_words - overlap_words)
            index += 1
    return chunks


def keyword_retrieve(
    query: str, chunks: list[dict], limit: int = 8, paper_ids: list[str] | None = None,
) -> list[dict]:
    query_terms = set(re.findall(r"[a-z0-9]+", query.casefold()))
    if not query_terms:
        return []
    scored = []
    for chunk in chunks:
        meta = chunk["metadata"]
        if paper_ids and meta.get("paper_id") not in paper_ids:
            continue
        terms = re.findall(r"[a-z0-9]+", chunk["text"].casefold())
        counts = Counter(terms)
        overlap = query_terms & set(counts)
        if overlap:
            score = sum(1 + min(counts[t], 3) * 0.15 for t in overlap) / len(query_terms)
            scored.append({**chunk, "score": score})
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:limit]


def hybrid_retrieve(
    query: str,
    chunks: list[dict],
    vector_service: Any = None,
    limit: int = 8,
    paper_ids: list[str] | None = None,
) -> tuple[list[dict], str | None]:
    lexical = keyword_retrieve(query, chunks, limit=max(limit, 1), paper_ids=paper_ids)
    if vector_service is None:
        return lexical, None
    try:
        semantic = vector_service.query(query, limit=limit, paper_ids=paper_ids)
    except Exception as exc:
        return lexical, f"Semantic retrieval unavailable ({type(exc).__name__}); keyword retrieval used."
    merged: dict[str, dict] = {}
    for row in semantic + lexical:
        key = row.get("chunk_id") or row.get("metadata", {}).get("chunk_id")
        if not key:
            # Chroma metadata retains IDs in the result IDs, when provided by adapter.
            key = hashlib.sha256((row.get("text", "") + str(row.get("metadata"))).encode()).hexdigest()
        old = merged.get(key)
        if old is None or row.get("score", 0) > old.get("score", 0):
            merged[key] = row
    return sorted(merged.values(), key=lambda x: x.get("score", 0), reverse=True)[:limit], None

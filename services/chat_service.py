from __future__ import annotations

import re

from schemas.models import Paper
from services.retrieval_service import hybrid_retrieve


def answer_from_papers(
    question: str,
    papers: list[Paper],
    chunks: list[dict],
    vector_service=None,
    selected_paper_ids: list[str] | None = None,
    limit: int = 6,
) -> dict:
    allowed = {p.paper_id for p in papers}
    selected = [pid for pid in (selected_paper_ids or []) if pid in allowed]
    rows, warning = hybrid_retrieve(
        question, chunks, vector_service=vector_service, limit=limit,
        paper_ids=selected or None,
    )
    paper_by_id = {p.paper_id: p for p in papers}
    evidence = []
    for row in rows:
        metadata = row.get("metadata", {})
        paper = paper_by_id.get(metadata.get("paper_id"))
        if not paper:
            continue
        text = row.get("text", "")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) >= 20]
        query_terms = set(re.findall(r"[a-z0-9]+", question.casefold()))
        selected_sentence = next((s for s in sentences if query_terms & set(re.findall(r"[a-z0-9]+", s.casefold()))), text[:600])
        page = metadata.get("page")
        citation = f"{paper.title}, p. {page}" if isinstance(page, int) and page > 0 else f"{paper.title}, abstract"
        evidence.append({"paper_id": paper.paper_id, "title": paper.title, "page": page if isinstance(page, int) and page > 0 else None, "quote": selected_sentence, "citation": citation})
    if not evidence:
        answer = "I do not have sufficient retrieved paper evidence to answer that question. Select papers with accessible abstracts or full text, or run a research search first."
    else:
        answer = "\n\n".join(f"> {item['quote']}\n\n— {item['citation']}" for item in evidence)
    return {"answer": answer, "evidence": evidence, "warning": warning}

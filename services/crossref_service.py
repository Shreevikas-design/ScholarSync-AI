from __future__ import annotations

from typing import Any

from schemas.models import Paper
from services.metadata import clean_abstract, make_id, norm_doi


def parse_crossref(data: dict[str, Any]) -> list[Paper]:
    papers = []
    for item in data.get("message", {}).get("items", []):
        titles = item.get("title") or []
        title = (titles[0] if titles else "").strip()
        if not title:
            continue
        authors = [
            " ".join(x for x in (author.get("given"), author.get("family")) if x)
            for author in item.get("author", [])
        ]
        dates = item.get("published-print") or item.get("published-online") or item.get("issued") or {}
        parts = dates.get("date-parts") or [[]]
        year = parts[0][0] if parts and parts[0] else None
        doi = norm_doi(item.get("DOI"))
        pdfs = [
            link.get("URL") for link in item.get("link", [])
            if "pdf" in (link.get("content-type") or "").casefold() and link.get("URL")
        ]
        paper_url = f"https://doi.org/{doi}" if doi else item.get("URL")
        papers.append(Paper(
            paper_id=make_id(doi, None, title),
            title=title,
            authors=[x for x in authors if x],
            abstract=clean_abstract(item.get("abstract")),
            year=int(year) if year else None,
            doi=doi,
            url=paper_url,
            pdf_url=pdfs[0] if pdfs else None,
            source="Crossref",
            source_ids=[f"Crossref:{item.get('DOI', '')}"],
            is_preprint=(item.get("type") or "").casefold() == "posted-content",
        ))
    return papers

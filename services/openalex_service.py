from __future__ import annotations

from typing import Any

from schemas.models import Paper
from services.metadata import abstract_from_index, make_id, norm_doi


def parse_openalex(data: dict[str, Any]) -> list[Paper]:
    papers = []
    for item in data.get("results", []):
        title = (item.get("title") or "").strip()
        if not title:
            continue
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in item.get("authorships", [])
            if a.get("author", {}).get("display_name")
        ]
        locations = [item.get("best_oa_location") or {}, item.get("primary_location") or {}]
        pdf = next((loc.get("pdf_url") for loc in locations if loc.get("pdf_url")), None)
        doi = norm_doi(item.get("doi"))
        year = item.get("publication_year")
        papers.append(Paper(
            paper_id=make_id(doi, None, title),
            title=title,
            authors=authors,
            abstract=abstract_from_index(item.get("abstract_inverted_index")),
            year=int(year) if year else None,
            doi=doi,
            url=item.get("doi") or item.get("id"),
            pdf_url=pdf,
            source="OpenAlex",
            source_ids=[f"OpenAlex:{item.get('id', '')}"],
            is_preprint=(item.get("type") or "").casefold() == "preprint",
        ))
    return papers

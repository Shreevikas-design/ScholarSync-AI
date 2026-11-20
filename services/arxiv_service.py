from __future__ import annotations

import xml.etree.ElementTree as ET

from schemas.models import Paper
from services.metadata import make_id


def parse_arxiv(raw: str) -> list[Paper]:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("a:entry", ns):
        paper_url = entry.findtext("a:id", default="", namespaces=ns)
        arxiv_id = paper_url.rsplit("/", 1)[-1].split("v")[0]
        title = " ".join(entry.findtext("a:title", default="", namespaces=ns).split())
        if not title:
            continue
        abstract = " ".join(entry.findtext("a:summary", default="", namespaces=ns).split())
        published = entry.findtext("a:published", default="", namespaces=ns)
        year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None
        pdf = next(
            (link.get("href") for link in entry.findall("a:link", ns)
             if link.get("title") == "pdf" or link.get("type") == "application/pdf"),
            None,
        )
        authors = [element.findtext("a:name", default="", namespaces=ns) for element in entry.findall("a:author", ns)]
        papers.append(Paper(
            paper_id=make_id(None, arxiv_id, title),
            title=title,
            authors=[author for author in authors if author],
            abstract=abstract,
            year=year,
            url=paper_url,
            pdf_url=pdf,
            source="arXiv",
            source_ids=[f"arXiv:{arxiv_id}"],
            is_preprint=True,
            published_date=published or None,
        ))
    return papers

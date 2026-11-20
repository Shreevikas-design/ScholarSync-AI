from __future__ import annotations

import re
from typing import Iterable

from schemas.models import Paper


def reference_text(paper: Paper) -> str:
    authors = ", ".join(paper.authors[:6]) or "Author not supplied by source"
    if len(paper.authors) > 6:
        authors += ", et al."
    parts = [authors, f"({paper.year})" if paper.year else "(year not supplied)", paper.title]
    if paper.doi:
        parts.append(f"https://doi.org/{paper.doi.removeprefix('https://doi.org/')}")
    elif paper.url:
        parts.append(paper.url)
    return ". ".join(parts) + "."


def citation_map(papers: Iterable[Paper]) -> dict[str, str]:
    return {paper.paper_id: f"[{i}]" for i, paper in enumerate(papers, start=1)}


def validate_citations(text: str, papers: Iterable[Paper]) -> list[str]:
    allowed = set(citation_map(papers).values())
    found = re.findall(r"\[(\d+)\]", text)
    return [f"[{n}]" for n in found if f"[{n}]" not in allowed]


def bibtex_export(papers: Iterable[Paper]) -> str:
    entries = []
    for index, paper in enumerate(papers, start=1):
        key = re.sub(r"[^A-Za-z0-9]+", "", (paper.authors[0] if paper.authors else "paper").split()[-1])
        key = f"{key or 'paper'}{paper.year or 'nd'}{index}"
        def esc(value: str) -> str:
            return value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        fields = {
            "title": paper.title,
            "author": " and ".join(paper.authors),
            "year": str(paper.year or ""),
            "doi": paper.doi or "",
            "url": paper.url or (f"https://doi.org/{paper.doi}" if paper.doi else ""),
        }
        lines = [f"@article{{{key},"]
        lines += [f"  {name} = {{{esc(value)}}}," for name, value in fields.items() if value]
        lines.append("}")
        entries.append("\n".join(lines))
    return "\n\n".join(entries) + ("\n" if entries else "")

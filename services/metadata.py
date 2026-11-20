from __future__ import annotations

import hashlib
import html
import re


def norm_doi(value: str | None) -> str | None:
    if not value:
        return None
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value.strip(), flags=re.I)
    return value.casefold() or None


def abstract_from_index(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    words = []
    for word, positions in index.items():
        words.extend((int(position), word) for position in positions)
    return " ".join(word for _, word in sorted(words))


def clean_abstract(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"<[^>]+>", " ", html.unescape(value))
    return " ".join(value.split())


def title_id(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()


def make_id(doi: str | None, arxiv_id: str | None, title: str) -> str:
    if doi:
        return "doi:" + norm_doi(doi)
    if arxiv_id:
        return "arxiv:" + arxiv_id.casefold()
    return "title:" + hashlib.sha256(title_id(title).encode()).hexdigest()[:24]

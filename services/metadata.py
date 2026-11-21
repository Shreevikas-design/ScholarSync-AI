from __future__ import annotations

import hashlib
import html
import re


def norm_doi(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    value = re.sub(r"^(?:doi\s*:\s*|https?://(?:dx\.)?doi\.org/)", "", value, flags=re.I)
    value = value.strip().rstrip(".,;)]}>").casefold()
    return value if re.match(r"^10\.\d{4,9}/\S+$", value) else None


def norm_arxiv(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().casefold()
    value = re.sub(r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/", "", value)
    value = re.sub(r"^arxiv\s*:\s*", "", value)
    value = value.split("?", 1)[0].removesuffix(".pdf")
    value = re.sub(r"v\d+$", "", value)
    return value if re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})", value) else None


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

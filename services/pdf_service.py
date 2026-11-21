from __future__ import annotations

import hashlib
import ipaddress
import logging
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pymupdf as fitz
import httpx

from config import Settings

log = logging.getLogger(__name__)


@dataclass
class PageText:
    page: int
    text: str


@dataclass
class PDFResult:
    ok: bool
    path: str | None
    pages: list[PageText]
    metadata: dict[str, str]
    error: str | None = None
    truncated: bool = False


def _public_http_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        addresses = socket.getaddrinfo(parsed.hostname, None)
        return all(ipaddress.ip_address(row[4][0]).is_global for row in addresses)
    except (OSError, ValueError):
        return False


def clean_text(text: str) -> str:
    lines = [" ".join(line.replace(chr(0), " ").split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _remove_repeated_marginalia(pages: list[PageText]) -> list[PageText]:
    if len(pages) < 3:
        return pages
    counts: dict[str, int] = {}
    for page in pages:
        lines = page.text.splitlines()
        for line in set(lines[:2] + lines[-2:]):
            normalized = " ".join(line.casefold().split())
            if len(normalized) >= 5:
                counts[normalized] = counts.get(normalized, 0) + 1
    threshold = max(3, int(len(pages) * 0.4))
    repeated = {line for line, count in counts.items() if count >= threshold}
    if not repeated:
        return pages
    cleaned = []
    for page in pages:
        lines = page.text.splitlines()
        kept = [line for line in lines if " ".join(line.casefold().split()) not in repeated]
        cleaned.append(PageText(page=page.page, text="\n".join(kept)))
    return [page for page in cleaned if page.text.strip()]


def extract_pdf_bytes(data: bytes, settings: Settings, target: Path | None = None) -> PDFResult:
    if len(data) > settings.max_pdf_mb * 1024 * 1024:
        return PDFResult(False, None, [], {}, "PDF exceeds configured size limit.")
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            if doc.is_encrypted:
                return PDFResult(False, None, [], {}, "Encrypted PDF cannot be processed.")
            metadata = {k: str(v or "") for k, v in (doc.metadata or {}).items() if v}
            pages = []
            page_count = doc.page_count
            truncated = page_count > settings.max_pdf_pages
            for index, page in enumerate(doc, start=1):
                if index > settings.max_pdf_pages:
                    break
                text = clean_text(page.get_text("text"))
                if text:
                    pages.append(PageText(page=index, text=text))
            pages = _remove_repeated_marginalia(pages)
            if not pages:
                return PDFResult(False, None, [], metadata, "PDF contains no extractable text.")
        saved = None
        if target:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            saved = str(target)
        warning = f"PDF page limit reached; extracted the first {settings.max_pdf_pages} of {page_count} pages." if truncated else None
        return PDFResult(True, saved, pages, metadata, error=warning, truncated=truncated)
    except Exception as exc:
        log.info("PDF extraction failed: %s", type(exc).__name__)
        return PDFResult(False, None, [], {}, f"Invalid or unreadable PDF ({type(exc).__name__}).")


def download_pdf(url: str, paper_id: str, settings: Settings) -> PDFResult:
    if not _public_http_url(url):
        return PDFResult(False, None, [], {}, "PDF URL is not a public HTTP(S) address.")
    digest = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:24]
    target = settings.data_dir / "papers" / f"{digest}.pdf"
    if target.exists():
        result = extract_pdf_bytes(target.read_bytes(), settings)
        if result.ok:
            result.path = str(target)
            return result
        target.unlink(missing_ok=True)
    current_url = url
    try:
        with httpx.Client(
            timeout=settings.request_timeout,
            follow_redirects=False,
            headers={"User-Agent": settings.user_agent},
        ) as client:
            for _ in range(4):
                if not _public_http_url(current_url):
                    return PDFResult(False, None, [], {}, "PDF redirect target is not a public HTTP(S) address.")
                with client.stream("GET", current_url) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")
                        if not location:
                            return PDFResult(False, None, [], {}, "PDF redirect omitted its destination.")
                        current_url = urljoin(current_url, location)
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").casefold()
                    if "pdf" not in content_type and not current_url.casefold().split("?")[0].endswith(".pdf"):
                        return PDFResult(False, None, [], {}, "Source did not return a PDF content type.")
                    limit = settings.max_pdf_mb * 1024 * 1024
                    chunks = []
                    size = 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > limit:
                            return PDFResult(False, None, [], {}, "PDF exceeds configured size limit.")
                        chunks.append(chunk)
                return extract_pdf_bytes(b"".join(chunks), settings, target)
        return PDFResult(False, None, [], {}, "PDF source exceeded the redirect limit.")
    except httpx.HTTPStatusError as exc:
        return PDFResult(False, None, [], {}, f"PDF source returned HTTP {exc.response.status_code}.")
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        return PDFResult(False, None, [], {}, f"PDF download failed ({type(exc).__name__}).")

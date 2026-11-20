from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import time
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from config import Settings
from services.arxiv_service import parse_arxiv
from services.crossref_service import parse_crossref
from services.metadata import norm_doi, title_id
from services.openalex_service import parse_openalex
from schemas.models import Paper, ResearchPlan

log = logging.getLogger(__name__)


@dataclass
class SearchOptions:
    start_year: int | None = None
    end_year: int | None = None
    max_papers: int = 20
    include_openalex: bool = True
    include_arxiv: bool = True
    include_crossref: bool = True


@dataclass
class DiscoveryResult:
    papers: list[Paper]
    failed_calls: int = 0
    api_calls: int = 0
    errors: list[str] = field(default_factory=list)
    cache_hits: int = 0


class AcademicSearchService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.ensure_data_dirs()
        self._request_state = threading.local()

    def _cache_path(self, key: str) -> Path:
        return self.settings.data_dir / "cache" / (hashlib.sha256(key.encode()).hexdigest() + ".json")

    def _get(self, url: str, params: dict[str, Any], content: str = "json") -> tuple[Any, int, bool]:
        self._request_state.error = None
        cache_key = json.dumps([url, params, content], sort_keys=True, ensure_ascii=True)
        path = self._cache_path(cache_key)
        if self.settings.cache_ttl_hours > 0 and path.exists():
            age = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
            if age <= self.settings.cache_ttl_hours * 3600:
                try:
                    return json.loads(path.read_text(encoding="utf-8")), 0, True
                except (OSError, json.JSONDecodeError):
                    pass
        attempts = self.settings.max_retries + 1
        transient_statuses = (429, 500, 502, 503, 504)
        calls = 0
        final_error = "request failed"
        for attempt in range(attempts):
            try:
                calls += 1
                with httpx.Client(
                    timeout=self.settings.request_timeout,
                    headers={"User-Agent": self.settings.user_agent, "Accept": "application/json, application/atom+xml"},
                    follow_redirects=True,
                ) as client:
                    response = client.get(url, params=params)
                if response.status_code in transient_statuses and attempt + 1 < attempts:
                    try:
                        delay = float(response.headers.get("Retry-After", 0.5 * (2 ** attempt)))
                    except (TypeError, ValueError):
                        delay = 0.5 * (2 ** attempt)
                    time.sleep(min(max(delay, 0.1), 5.0))
                    continue
                response.raise_for_status()
                body = response.json() if content == "json" else response.text
                if self.settings.cache_ttl_hours > 0:
                    try:
                        path.write_text(json.dumps(body, ensure_ascii=True), encoding="utf-8")
                    except OSError:
                        log.warning("Could not write the local search cache.")
                return body, calls, False
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError, ValueError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                final_error = f"HTTP {status}" if status is not None else type(exc).__name__
                retryable = (
                    isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))
                    or status in transient_statuses
                )
                if not retryable or attempt + 1 >= attempts:
                    break
                time.sleep(min(0.25 * (2 ** attempt), 1.0))
        self._request_state.error = final_error
        log.warning("Academic API request failed for host=%s: %s", urlparse(url).netloc, final_error)
        return None, calls, False

    def _openalex(self, query: str, options: SearchOptions) -> tuple[list[Paper], int, bool, str | None]:
        params: dict[str, Any] = {"search": query, "per-page": min(max(options.max_papers, 10), 50)}
        if options.start_year and options.end_year:
            params["filter"] = f"from_publication_date:{options.start_year}-01-01,to_publication_date:{options.end_year}-12-31"
        data, calls, hit = self._get("https://api.openalex.org/works", params)
        if not data:
            return [], calls, hit, "OpenAlex request failed (" + (getattr(self._request_state, "error", None) or "unknown error") + ")."
        return parse_openalex(data), calls, hit, None

    def _crossref(self, query: str, options: SearchOptions) -> tuple[list[Paper], int, bool, str | None]:
        filters = []
        if options.start_year:
            filters.append(f"from-pub-date:{options.start_year}-01-01")
        if options.end_year:
            filters.append(f"until-pub-date:{options.end_year}-12-31")
        params: dict[str, Any] = {"query.bibliographic": query, "rows": min(max(options.max_papers, 10), 50)}
        if filters:
            params["filter"] = ",".join(filters)
        data, calls, hit = self._get("https://api.crossref.org/works", params)
        if not data:
            return [], calls, hit, "Crossref request failed (" + (getattr(self._request_state, "error", None) or "unknown error") + ")."
        return parse_crossref(data), calls, hit, None

    def _arxiv(self, query: str, options: SearchOptions) -> tuple[list[Paper], int, bool, str | None]:
        params = {
            "search_query": f"all:{query}", "start": 0,
            "max_results": min(max(options.max_papers, 10), 50), "sortBy": "relevance",
        }
        raw, calls, hit = self._get("https://export.arxiv.org/api/query", params, content="xml")
        if not raw:
            return [], calls, hit, "arXiv request failed (" + (getattr(self._request_state, "error", None) or "unknown error") + ")."
        papers = parse_arxiv(raw)
        if not papers and "<entry" in raw:
            return [], calls, hit, "arXiv returned invalid Atom XML."
        return papers, calls, hit, None

    def search(self, plan: ResearchPlan, options: SearchOptions) -> DiscoveryResult:
        queries = list(dict.fromkeys(q.strip() for q in plan.search_queries if q.strip()))[:6]
        jobs = []
        for query in queries:
            if options.include_openalex:
                jobs.append((query, "OpenAlex", self._openalex))
            if options.include_arxiv:
                jobs.append((query, "arXiv", self._arxiv))
            if options.include_crossref:
                jobs.append((query, "Crossref", self._crossref))
        raw: list[Paper] = []
        errors: list[str] = []
        calls = hits = failed = 0
        with ThreadPoolExecutor(max_workers=min(6, max(1, len(jobs)))) as pool:
            futures = {pool.submit(fn, query, options): (query, source) for query, source, fn in jobs}
            for future in as_completed(futures):
                query, source = futures[future]
                try:
                    papers, count, hit, error = future.result()
                    calls += count
                    hits += int(hit)
                    raw.extend(papers)
                    if error:
                        failed += 1
                        errors.append(f"{source} ({query[:80]}): {error}")
                except Exception as exc:
                    failed += 1
                    errors.append(f"{source} ({query[:80]}): {type(exc).__name__}")
        filtered = [
            p for p in raw
            if (options.start_year is None or (p.year is not None and p.year >= options.start_year))
            and (options.end_year is None or (p.year is not None and p.year <= options.end_year))
        ]
        unique = deduplicate(filtered)
        ranked = rank_papers(unique, plan.keywords, plan.objective)
        return DiscoveryResult(
            papers=ranked[:options.max_papers],
            failed_calls=failed,
            api_calls=calls,
            errors=errors,
            cache_hits=hits,
        )


def deduplicate(papers: list[Paper]) -> list[Paper]:
    result: list[Paper] = []
    alias: dict[str, int] = {}
    for paper in papers:
        keys = {f"title:{title_id(paper.title)}"}
        doi = norm_doi(paper.doi)
        if doi:
            keys.add(f"doi:{doi}")
        for source_id in paper.source_ids:
            if source_id.casefold().startswith("arxiv:"):
                keys.add("arxiv:" + source_id.split(":", 1)[1].casefold())
        target = next((alias[key] for key in keys if key in alias), None)
        if target is None:
            target = len(result)
            result.append(paper.model_copy(deep=True))
        else:
            old = result[target]
            result[target] = old.model_copy(update={
                "authors": old.authors or paper.authors,
                "abstract": old.abstract or paper.abstract,
                "year": old.year or paper.year,
                "doi": old.doi or paper.doi,
                "url": old.url or paper.url,
                "pdf_url": old.pdf_url or paper.pdf_url,
                "source_ids": list(dict.fromkeys(old.source_ids + paper.source_ids + [paper.source])),
                "is_preprint": old.is_preprint or paper.is_preprint,
            })
        for key in keys:
            alias[key] = target
    return result


def _tokens(text: str) -> set[str]:
    stop = {"the", "and", "for", "with", "from", "into", "using", "based", "study", "analysis", "of", "in", "on", "to"}
    return {t for t in re.findall(r"[a-z0-9]+", text.casefold()) if len(t) > 2 and t not in stop}


def rank_papers(papers: list[Paper], keywords: list[str], objective: str) -> list[Paper]:
    terms = _tokens(" ".join(keywords) + " " + objective)
    if not terms:
        return sorted(papers, key=lambda p: p.year or 0, reverse=True)
    def score(paper: Paper) -> float:
        title = _tokens(paper.title)
        abstract = _tokens(paper.abstract)
        title_coverage = len(title & terms) / len(terms)
        abstract_coverage = len(abstract & terms) / len(terms)
        return round(min(1.0, 0.72 * title_coverage + 0.28 * abstract_coverage), 4)
    return sorted(
        (p.model_copy(update={"relevance_score": score(p)}) for p in papers),
        key=lambda p: (p.relevance_score, p.year or 0),
        reverse=True,
    )

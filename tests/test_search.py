from __future__ import annotations

from agents.planner import heuristic_plan
from schemas.models import Paper
from services.search_service import AcademicSearchService, SearchOptions, deduplicate, rank_papers
from services.metadata import norm_arxiv, norm_doi


def test_planner_creates_valid_queries():
    plan = heuristic_plan("Wi-Fi access point placement optimization", 2020, 2025)
    assert plan.search_queries[0] == "Wi-Fi access point placement optimization"
    assert plan.keywords
    assert "2020" in plan.scope


def test_identifier_normalization_handles_public_urls_versions_and_punctuation():
    assert norm_doi("https://doi.org/10.1145/ABC.123.") == "10.1145/abc.123"
    assert norm_doi("not-a-doi") is None
    assert norm_arxiv("https://arxiv.org/abs/1101.5027v3") == "1101.5027"
    assert norm_arxiv("arXiv:cs/9901001v2") == "cs/9901001"


def test_openalex_abstract_inverted_index_parsing(settings_factory, monkeypatch, real_paper):
    service = AcademicSearchService(settings_factory())
    record = {
        "title": "A Wi-Fi placement study",
        "authorships": [{"author": {"display_name": "A. Author"}}],
        "publication_year": 2022,
        "doi": "https://doi.org/10.1234/test",
        "abstract_inverted_index": {"Wi-Fi": [0], "placement": [1], "works.": [2]},
        "best_oa_location": {"pdf_url": "https://example.org/paper.pdf"},
        "id": "https://openalex.org/W1",
        "type": "article",
    }
    monkeypatch.setattr(service, "_get", lambda *a, **k: ({"results": [record]}, 1, False))
    papers, calls, cache_hit, error = service._openalex("Wi-Fi placement", SearchOptions())
    assert calls == 1 and not cache_hit and error is None
    assert papers[0].abstract == "Wi-Fi placement works."
    assert papers[0].doi == "10.1234/test"
    assert papers[0].pdf_url.endswith(".pdf")


def test_openalex_retry_after_is_bounded_and_preserves_success(settings_factory, monkeypatch):
    responses = [
        type("Response", (), {"status_code": 429, "headers": {"Retry-After": "0"}, "json": lambda self: {"error": "rate limited"}})(),
        type("Response", (), {"status_code": 200, "headers": {}, "json": lambda self: {"results": []}, "raise_for_status": lambda self: None})(),
    ]
    calls = []
    class Client:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, params=None):
            calls.append((url, params))
            return responses.pop(0)
    monkeypatch.setattr("services.search_service.httpx.Client", Client)
    monkeypatch.setattr("services.search_service.wait_for_host", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("services.search_service.time.sleep", lambda *_args: None)
    service = AcademicSearchService(settings_factory(max_retries=2))
    data, count, cached = service._get("https://api.openalex.org/works", {"search": "wireless"})
    assert data == {"results": []} and count == 2 and not cached
    assert len(calls) == 2


def test_openalex_daily_budget_error_is_not_retried(settings_factory, monkeypatch):
    class Response:
        status_code = 429
        headers = {"Retry-After": "0"}
        def json(self): return {"error": "429", "reason": "daily_budget", "retryAfter": 1}
    class Client:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, *args, **kwargs): return Response()
    monkeypatch.setattr("services.search_service.httpx.Client", Client)
    monkeypatch.setattr("services.search_service.wait_for_host", lambda *_args, **_kwargs: None)
    service = AcademicSearchService(settings_factory(max_retries=4))
    data, count, _ = service._get("https://api.openalex.org/works", {"search": "wireless"})
    assert data is None and count == 1
    assert "daily API budget exhausted" in service._request_state.error


def test_search_cache_avoids_repeating_successful_provider_call(settings_factory, monkeypatch):
    calls = []
    class Response:
        status_code = 200
        headers = {}
        def json(self): return {"results": []}
        def raise_for_status(self): pass
    class Client:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, *args, **kwargs):
            calls.append(1)
            return Response()
    monkeypatch.setattr("services.search_service.httpx.Client", Client)
    monkeypatch.setattr("services.search_service.wait_for_host", lambda *_args, **_kwargs: None)
    service = AcademicSearchService(settings_factory(cache_ttl_hours=24))
    first = service._get("https://api.openalex.org/works", {"search": "wireless"})
    second = service._get("https://api.openalex.org/works", {"search": "wireless"})
    assert first[1:] == (1, False)
    assert second[1:] == (0, True)
    assert len(calls) == 1


def test_concurrent_identical_requests_share_one_cache_fill(settings_factory, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    calls = []
    class Response:
        status_code = 200
        headers = {}
        def json(self): return {"results": []}
        def raise_for_status(self): pass
    class Client:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, *args, **kwargs):
            calls.append(1)
            return Response()
    monkeypatch.setattr("services.search_service.httpx.Client", Client)
    monkeypatch.setattr("services.search_service.wait_for_host", lambda *_args, **_kwargs: None)
    service = AcademicSearchService(settings_factory(cache_ttl_hours=24))
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: service._get("https://api.openalex.org/works", {"search": "shared"}), range(3)))
    assert len(calls) == 1
    assert sum(cache_hit for _, _, cache_hit in results) == 2


def test_search_query_deduplication_is_case_and_whitespace_insensitive(settings_factory):
    plan = heuristic_plan("Wi-Fi access point placement", None, None).model_copy(update={
        "search_queries": ["wireless  access points", " WIRELESS access   points "]
    })
    seen = []
    service = AcademicSearchService(settings_factory())
    service._openalex = lambda query, options: (seen.append(query) or [], 1, False, None)
    result = service.search(plan, SearchOptions(include_arxiv=False, include_crossref=False))
    assert seen == ["wireless access points"]
    assert result.api_calls == 1


def test_crossref_parsing(settings_factory, monkeypatch):
    service = AcademicSearchService(settings_factory())
    item = {
        "title": ["Optimal Access Point Placement for Multi-AP mmWave WLANs"],
        "author": [{"given": "A", "family": "Researcher"}],
        "DOI": "10.1145/3345768.3355914",
        "issued": {"date-parts": [[2019, 10, 1]]},
        "URL": "https://doi.org/10.1145/3345768.3355914",
        "type": "proceedings-article",
    }
    monkeypatch.setattr(service, "_get", lambda *a, **k: ({"message": {"items": [item]}}, 1, False))
    papers, _, _, error = service._crossref("AP placement WLAN", SearchOptions())
    assert error is None
    assert papers[0].year == 2019
    assert papers[0].authors == ["A Researcher"]
    assert papers[0].doi == "10.1145/3345768.3355914"


def test_arxiv_atom_parsing(settings_factory, monkeypatch):
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>https://arxiv.org/abs/1101.5027v1</id>
        <updated>2011-01-25T00:00:00Z</updated>
        <published>2011-01-25T00:00:00Z</published>
        <title>Modeling and Solving AP Location and Frequency Assignment</title>
        <summary>We present optimization approaches for Wi-Fi network planning.</summary>
        <author><name>Test Author</name></author>
        <link title="pdf" type="application/pdf" href="https://arxiv.org/pdf/1101.5027"/>
      </entry>
    </feed>"""
    service = AcademicSearchService(settings_factory())
    monkeypatch.setattr(service, "_get", lambda *a, **k: (xml, 1, False))
    papers, _, _, error = service._arxiv("access point", SearchOptions())
    assert error is None
    assert papers[0].paper_id == "arxiv:1101.5027"
    assert papers[0].is_preprint
    assert papers[0].pdf_url.endswith("1101.5027")


def test_search_respects_year_range_and_source_errors(settings_factory, monkeypatch, real_paper):
    service = AcademicSearchService(settings_factory())
    current = real_paper.model_copy(update={"paper_id": "doi:10.0000/current", "doi": "10.0000/current", "year": 2022})
    service._openalex = lambda query, options: ([real_paper, current], 1, False, None)
    service._arxiv = lambda query, options: ([], 1, False, "arXiv request failed.")
    service._crossref = lambda query, options: ([], 1, False, None)
    result = service.search(
        heuristic_plan("Wi-Fi AP placement", 2020, 2025),
        SearchOptions(start_year=2020, end_year=2025, max_papers=5),
    )
    assert [p.paper_id for p in result.papers] == ["doi:10.0000/current"]
    assert result.failed_calls == len(heuristic_plan("Wi-Fi AP placement", 2020, 2025).search_queries)
    assert result.errors


def test_deduplicate_by_doi_and_normalized_title(real_paper):
    same_doi = real_paper.model_copy(update={"source": "Crossref", "source_ids": ["Crossref:10.1234/X"], "doi": "10.1234/X", "paper_id": "doi:10.1234/x"})
    by_title = real_paper.model_copy(update={"paper_id": "title:other", "source": "OpenAlex", "source_ids": ["OpenAlex:W1"]})
    result = deduplicate([real_paper, same_doi, by_title])
    assert len(result) == 1
    assert "Crossref:10.1234/X" in result[0].source_ids
    assert "OpenAlex" in result[0].source_ids


def test_deduplication_uses_arxiv_alias_and_exact_titles_only(real_paper):
    doi_record = real_paper.model_copy(update={
        "paper_id": "doi:10.5555/alpha", "doi": "10.5555/alpha", "title": "A revised metadata title",
        "source_ids": ["arXiv:1101.5027v3"],
    })
    unrelated = real_paper.model_copy(update={
        "paper_id": "title:nearby", "title": "Modeling AP Location and Frequency Assignments for Wi-Fi Networks",
        "source_ids": ["OpenAlex:W999"],
    })
    result = deduplicate([real_paper, doi_record, unrelated])
    assert len(result) == 2
    assert next(p for p in result if p.doi).paper_id == "doi:10.5555/alpha"


def test_relevance_ranking_uses_title_and_abstract(real_paper):
    other = real_paper.model_copy(update={"paper_id": "arxiv:other", "title": "Unrelated topic", "abstract": "This work studies batteries."})
    ranked = rank_papers([other, real_paper], ["wifi", "access", "point", "placement"], "Wi-Fi network placement")
    assert ranked[0].paper_id == real_paper.paper_id
    assert ranked[0].relevance_score > ranked[1].relevance_score

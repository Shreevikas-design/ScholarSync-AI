from __future__ import annotations

from agents.planner import heuristic_plan
from schemas.models import Paper
from services.search_service import AcademicSearchService, SearchOptions, deduplicate, rank_papers


def test_planner_creates_valid_queries():
    plan = heuristic_plan("Wi-Fi access point placement optimization", 2020, 2025)
    assert plan.search_queries[0] == "Wi-Fi access point placement optimization"
    assert plan.keywords
    assert "2020" in plan.scope


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


def test_relevance_ranking_uses_title_and_abstract(real_paper):
    other = real_paper.model_copy(update={"paper_id": "arxiv:other", "title": "Unrelated topic", "abstract": "This work studies batteries."})
    ranked = rank_papers([other, real_paper], ["wifi", "access", "point", "placement"], "Wi-Fi network placement")
    assert ranked[0].paper_id == real_paper.paper_id
    assert ranked[0].relevance_score > ranked[1].relevance_score

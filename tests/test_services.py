from __future__ import annotations

import json
from pathlib import Path

import pymupdf as fitz
import pytest

from evaluation.metrics import mean_reciprocal_rank, precision_at_k, recall_at_k
from schemas.models import EvidenceSpan
from services.citation_service import bibtex_export, citation_map, validate_citations
from services.embedding_service import EmbeddingService
from services.llm_service import LLMService, ProviderError, TokenBudgetError
from services.pdf_service import _public_http_url, download_pdf, extract_pdf_bytes
from services.report_service import export_pdf, safe_filename
from services.retrieval_service import chunk_pages, hybrid_retrieve, keyword_retrieve
from services.vector_service import VectorService


def test_pdf_extracts_page_text_and_metadata(settings_factory):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This paper presents a verified public research method for Wi-Fi planning.")
    data = doc.tobytes()
    doc.close()
    result = extract_pdf_bytes(data, settings_factory())
    assert result.ok
    assert result.pages[0].page == 1
    assert "Wi-Fi planning" in result.pages[0].text


def test_pdf_rejects_invalid_file(settings_factory):
    result = extract_pdf_bytes(b"not a pdf", settings_factory())
    assert not result.ok
    assert "Invalid" in result.error


def test_pdf_rejects_oversize_file(settings_factory):
    result = extract_pdf_bytes(b"x" * (1024 * 1024 + 1), settings_factory(max_pdf_mb=1))
    assert not result.ok
    assert "size limit" in result.error


def test_pdf_rejects_local_url():
    assert not _public_http_url("http://127.0.0.1/test.pdf")
    assert not _public_http_url("file:///tmp/test.pdf")


def test_pdf_download_uses_public_metadata_url(settings_factory, real_paper, monkeypatch):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "A public PDF test response with extractable Wi-Fi evidence.")
    pdf = doc.tobytes()
    doc.close()

    class Response:
        status_code = 200
        headers = {"content-type": "application/pdf"}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_bytes(self): yield pdf

    class Client:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def stream(self, *args, **kwargs): return Response()

    monkeypatch.setattr("services.pdf_service._public_http_url", lambda url: True)
    monkeypatch.setattr("services.pdf_service.httpx.Client", Client)
    paper = real_paper.model_copy(update={"paper_id": "arxiv:download-fixture", "pdf_url": "https://example.org/file.pdf"})
    result = download_pdf(paper.pdf_url, paper.paper_id, settings_factory())
    assert result.ok and result.path
    assert result.pages[0].page == 1


def test_pdf_redirect_to_private_host_is_blocked(settings_factory, real_paper, monkeypatch):
    calls = []
    class Response:
        status_code = 302
        headers = {"location": "http://127.0.0.1/private.pdf"}
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class Client:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def stream(self, method, url):
            calls.append(url)
            return Response()
    monkeypatch.setattr("services.pdf_service.httpx.Client", Client)
    monkeypatch.setattr("services.pdf_service._public_http_url", lambda url: url.startswith("https://"))
    paper = real_paper.model_copy(update={"paper_id": "arxiv:redirect-fixture"})
    result = download_pdf("https://example.org/file.pdf", paper.paper_id, settings_factory())
    assert not result.ok
    assert len(calls) == 1


def test_page_aware_chunking_and_keyword_retrieval(real_paper):
    pages = [{"page": 4, "text": " ".join(["Wi-Fi access point placement optimization."] * 100)}]
    chunks = chunk_pages(real_paper, pages, chunk_words=30, overlap_words=5)
    assert len(chunks) > 1
    assert all(c["metadata"]["page"] == 4 for c in chunks)
    assert all(c["metadata"]["paper_id"] == real_paper.paper_id for c in chunks)
    found = keyword_retrieve("access point placement", chunks, limit=2)
    assert found and found[0]["metadata"]["page"] == 4


def test_hybrid_retrieval_exposes_semantic_fallback(real_paper):
    chunks = chunk_pages(real_paper, [{"page": None, "location": "abstract", "text": real_paper.abstract}])
    results, warning = hybrid_retrieve("frequency assignment", chunks)
    assert results
    assert warning is None


def test_vector_service_batches_embeddings_and_preserves_metadata(settings_factory):
    class Embedder:
        def __init__(self):
            self.batch_lengths = []

        def embed(self, texts):
            self.batch_lengths.append(len(texts))
            return [[float(index + 1), 1.0, 0.5] for index, _ in enumerate(texts)]

    embedder = Embedder()
    service = VectorService(settings_factory(), embedder=embedder)
    chunks = [
        {
            "chunk_id": f"chunk-{index}",
            "text": f"Evidence for paper {index}",
            "metadata": {"paper_id": f"paper-{index}", "page": index + 1, "source_url": f"https://example.org/{index}"},
        }
        for index in range(5)
    ]

    assert service.add(chunks, batch_size=2) == 5
    assert embedder.batch_lengths == [2, 2, 1]
    rows = service.query("evidence", limit=1, paper_ids=["paper-4"])
    assert len(rows) == 1
    assert rows[0]["metadata"]["paper_id"] == "paper-4"
    assert rows[0]["metadata"]["page"] == 5
    assert rows[0]["metadata"]["source_url"] == "https://example.org/4"


def test_citation_validation_and_bibtex(real_paper):
    mapping = citation_map([real_paper])
    assert mapping[real_paper.paper_id] == "[1]"
    assert validate_citations("Supported [1], invalid [4].", [real_paper]) == ["[4]"]
    bib = bibtex_export([real_paper])
    assert "@article{" in bib and real_paper.title in bib
    assert "1101.5027" in bib


def test_llm_provider_selection_does_not_silently_fallback(settings_factory):
    settings = settings_factory(provider="groq")
    llm = LLMService(settings)
    with pytest.raises(ProviderError, match="not configured"):
        llm.complete_text("analyze")
    assert llm.usage.failed_calls == 0
    assert llm.usage.llm_calls == 0
    assert llm.usage.errors
    template = LLMService(settings_factory(provider="template"))
    with pytest.raises(ProviderError, match="Template mode"):
        template.complete_text("prompt")


def test_token_context_preflight(settings_factory):
    settings = settings_factory(provider="groq", max_context_tokens=1024, max_output_tokens=900)
    llm = LLMService(settings)
    with pytest.raises(TokenBudgetError, match="Prompt budget exceeded"):
        llm.complete_text("x" * 1000, max_tokens=900)
    assert llm.usage.budget_rejections == 1
    assert llm.usage.llm_calls == 0


def test_provider_500_token_overflow_is_classified(settings_factory, monkeypatch):
    class Response:
        status_code = 500
        headers = {}
        text = '{"error":"too many tokens for this model context length"}'

    class Client:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, *args, **kwargs): return Response()

    monkeypatch.setattr("httpx.Client", Client)
    settings = settings_factory(provider="groq", groq_api_key="test-secret", max_context_tokens=4096, max_output_tokens=512)
    llm = LLMService(settings)
    with pytest.raises(TokenBudgetError, match="HTTP 500"):
        llm.complete_text("A small prompt", max_tokens=100)
    assert llm.usage.failed_calls == 1
    assert llm.usage.llm_calls == 1
    assert "too many tokens" not in llm.usage.errors[0]


def test_sentence_transformer_selection_is_explicit(settings_factory):
    settings = settings_factory(embedding_provider="sentence-transformers")
    embedder = EmbeddingService(settings)
    assert embedder.settings.embedding_provider == "sentence-transformers"


def test_report_pdf_export_and_filename(tmp_path, real_paper):
    from agents.planner import heuristic_plan
    from agents.paper_analyzer import PaperAnalyzer
    from agents.comparative_analyst import ComparativeAnalyst
    from agents.gap_analyzer import ResearchGapAnalyzer
    from agents.writer import LiteratureReviewWriter
    from services.llm_service import LLMService

    settings = __import__("config").Settings(provider="template", data_dir=tmp_path, max_context_tokens=2048, max_output_tokens=512)
    analysis = PaperAnalyzer(settings, LLMService(settings)).run(real_paper)
    plan = heuristic_plan("Wi-Fi access point placement", 2011, 2026)
    comparison = ComparativeAnalyst().run([real_paper], [analysis])
    gaps = ResearchGapAnalyzer().run([analysis])
    report = LiteratureReviewWriter().run("Wi-Fi access point placement", plan, [real_paper], [analysis], comparison, gaps, 2011, 2026)
    path = export_pdf(report, tmp_path / "report.pdf")
    assert path.exists() and path.stat().st_size > 1000
    check = fitz.open(path)
    text = "\n".join(page.get_text() for page in check)
    assert "Abstract" in text and "References" in text and "Methodological comparison" in text
    assert safe_filename(report.title, "pdf").endswith(".pdf")


def test_retrieval_metrics():
    retrieved = ["a", "b", "c"]
    relevant = {"b", "z"}
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert mean_reciprocal_rank(retrieved, relevant) == 0.5

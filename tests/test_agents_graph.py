from __future__ import annotations

from agents.comparative_analyst import ComparativeAnalyst
from agents.critic import EvidenceCritic
from agents.gap_analyzer import ResearchGapAnalyzer
from agents.paper_analyzer import PaperAnalyzer
from agents.planner import heuristic_plan
from agents.supervisor import ResearchRequest
from agents.writer import LiteratureReviewWriter
from graph.workflow import build_workflow
from schemas.models import Critique, CriticIssue, EvidenceSpan
from services.llm_service import LLMService
from services.search_service import DiscoveryResult


class FakeSearch:
    def __init__(self, papers):
        self.papers = papers
    def search(self, plan, options):
        return DiscoveryResult(papers=self.papers, api_calls=3)


def test_request_schema_bounds_and_validates_sources():
    request = ResearchRequest(topic="Access point placement", start_year=2020, end_year=2024, max_papers=10)
    assert request.max_papers == 10
    try:
        ResearchRequest(topic="tiny")
    except ValueError:
        pass
    else:
        raise AssertionError("short topic should be rejected")
    try:
        ResearchRequest(topic="Access point placement", include_openalex=False, include_arxiv=False, include_crossref=False)
    except ValueError:
        pass
    else:
        raise AssertionError("empty source selection should be rejected")


def test_analysis_is_abstract_only_and_evidence_is_exact(settings_factory, real_paper):
    settings = settings_factory()
    analysis = PaperAnalyzer(settings, LLMService(settings)).run(real_paper)
    assert analysis.analysis_mode == "abstract-only"
    assert analysis.objective != "Not reported in accessible text."
    assert all(item.quote in real_paper.abstract for item in analysis.evidence)
    assert all(item.page is None and item.location == "abstract" for item in analysis.evidence)


def test_metadata_only_analysis_does_not_claim_an_abstract(settings_factory, real_paper):
    paper = real_paper.model_copy(update={"abstract": "", "pdf_url": None})
    settings = settings_factory()
    analysis = PaperAnalyzer(settings, LLMService(settings)).run(paper)
    assert analysis.analysis_mode == "metadata-only"
    assert analysis.objective == "Not reported in accessible text."
    assert not analysis.evidence


def test_page_limited_analysis_is_marked_partial_full_text(settings_factory, real_paper):
    settings = settings_factory()
    analysis = PaperAnalyzer(settings, LLMService(settings)).run(
        real_paper, [{"page": 1, "text": real_paper.abstract}], partial_full_text=True,
    )
    assert analysis.analysis_mode == "partial-full-text"


def test_analyzer_rejects_an_invented_page_number(settings_factory, real_paper):
    settings = settings_factory(provider="groq")
    template_settings = settings_factory()
    template = PaperAnalyzer(template_settings, LLMService(template_settings)).run(real_paper)

    class WrongPageModel:
        def generate_json(self, *args, **kwargs):
            broken = template.evidence[0].model_copy(update={"page": 99})
            return template.model_copy(update={"evidence": [broken]})

    analysis = PaperAnalyzer(settings, WrongPageModel()).run(real_paper)
    assert analysis.analysis_mode == "abstract-only"
    assert analysis.error and "template extraction used" in analysis.error
    assert all(span.page is None for span in analysis.evidence)


def test_comparison_warns_when_numeric_conditions_are_missing(settings_factory, real_paper):
    settings = settings_factory()
    analysis = PaperAnalyzer(settings, LLMService(settings)).run(real_paper)
    result = ComparativeAnalyst().run([real_paper], [analysis])
    assert len(result.items) == 1
    assert "Only one analyzed paper" in result.items[0].comparability_note
    assert result.items[0].strengths


def test_author_reported_limitation_requires_quote(settings_factory, real_paper):
    paper = real_paper.model_copy(update={
        "abstract": "This paper proposes a Wi-Fi placement model. A limitation is that the model assumes static demand."
    })
    settings = settings_factory()
    analysis = PaperAnalyzer(settings, LLMService(settings)).run(paper)
    gaps = ResearchGapAnalyzer().run([analysis])
    assert gaps
    assert gaps[0].category == "author-reported limitation"
    assert gaps[0].supporting_evidence[0].quote in paper.abstract
    assert gaps[0].confidence == "low"
    assert gaps[0].evidence_coverage == 1.0


def test_repeated_limitation_confidence_requires_independent_papers(settings_factory, real_paper):
    settings = settings_factory()
    base = PaperAnalyzer(settings, LLMService(settings)).run(real_paper)
    limitation = "The method assumes fixed and static user traffic throughout evaluation."
    analyses = [base.model_copy(update={
        "paper_id": f"arxiv:test{i}",
        "limitations": limitation,
        "evidence": [EvidenceSpan(paper_id=f"arxiv:test{i}", field="limitations", quote=limitation, location="abstract")],
    }) for i in range(3)]
    repeated = next(g for g in ResearchGapAnalyzer().run(analyses) if g.category == "cross-paper observation")
    assert repeated.confidence == "high"
    assert repeated.evidence_coverage == 1.0


def test_system_generated_hypothesis_is_low_confidence_and_cited(settings_factory, real_paper):
    settings = settings_factory()
    first = PaperAnalyzer(settings, LLMService(settings)).run(real_paper).model_copy(update={
        "dataset": "The authors evaluate the method on the NS-3 simulation dataset.",
        "evidence": [EvidenceSpan(
            paper_id=real_paper.paper_id,
            field="dataset",
            quote="The authors evaluate the method on the NS-3 simulation dataset.",
            location="abstract",
        )],
    })
    second = first.model_copy(update={
        "paper_id": "arxiv:1505.06530",
        "dataset": "The experiments use an indoor Wi-Fi testbed dataset.",
        "evidence": [EvidenceSpan(
            paper_id="arxiv:1505.06530",
            field="dataset",
            quote="The experiments use an indoor Wi-Fi testbed dataset.",
            location="abstract",
        )],
    })
    gaps = ResearchGapAnalyzer().run([first, second])
    hypothesis = next(g for g in gaps if g.category == "system-generated hypothesis")
    assert hypothesis.confidence == "low"
    assert len(hypothesis.related_paper_ids) == 2
    assert all(span.paper_id in hypothesis.related_paper_ids for span in hypothesis.supporting_evidence)


def test_critic_accepts_grounded_report(settings_factory, real_paper):
    settings = settings_factory()
    analysis = PaperAnalyzer(settings, LLMService(settings)).run(real_paper)
    plan = heuristic_plan("Wi-Fi access point placement", 2011, 2026)
    comparison = ComparativeAnalyst().run([real_paper], [analysis])
    gaps = ResearchGapAnalyzer().run([analysis])
    report = LiteratureReviewWriter().run(
        "Wi-Fi access point placement", plan, [real_paper], [analysis],
        comparison, gaps, 2011, 2026,
    )
    result = EvidenceCritic().run(
        report, [real_paper], [analysis],
        {real_paper.paper_id: [{"page": None, "location": "abstract", "text": real_paper.abstract}]},
    )
    assert result.accepted
    assert not [i for i in result.issues if i.severity == "error"]


def test_writer_validates_synthesis_evidence_and_reuses_it(settings_factory, real_paper):
    from agents.writer import LiteratureReviewWriter, SynthesisClaim, SynthesisDraft, ThemeSynthesis, EvidenceReference
    settings = settings_factory(provider="groq")
    analysis = PaperAnalyzer(settings_factory(), LLMService(settings_factory())).run(real_paper)

    class FakeLLM:
        def __init__(self):
            self.settings = settings
            self.calls = 0
        def generate_json(self, prompt, schema, max_tokens=None):
            self.calls += 1
            evidence = analysis.evidence[0]
            return SynthesisDraft(themes=[ThemeSynthesis(theme="Optimization", claims=[SynthesisClaim(
                text="The included study frames access point planning as an optimization task.",
                evidence=[EvidenceReference(paper_id=real_paper.paper_id, quote=evidence.quote)],
            )])])

    llm = FakeLLM()
    writer = LiteratureReviewWriter(llm)
    plan = heuristic_plan("Wi-Fi access point placement", 2011, 2026)
    comparison = ComparativeAnalyst().run([real_paper], [analysis])
    report = writer.run("Wi-Fi access point placement", plan, [real_paper], [analysis], comparison, [], 2011, 2026)
    writer.run("Wi-Fi access point placement", plan, [real_paper], [analysis], comparison, [], 2011, 2026)
    assert "The included study frames access point planning" in report.markdown
    assert "[1]" in report.markdown
    assert llm.calls == 1


def test_critic_rejects_quote_absent_from_source(settings_factory, real_paper):
    settings = settings_factory()
    analysis = PaperAnalyzer(settings, LLMService(settings)).run(real_paper)
    bad = analysis.model_copy(update={
        "evidence": [EvidenceSpan(
            paper_id=real_paper.paper_id,
            field="objective",
            quote="This invented objective is not in the source text.",
            location="abstract",
        )],
    })
    plan = heuristic_plan("Wi-Fi access point placement", 2011, 2026)
    comparison = ComparativeAnalyst().run([real_paper], [analysis])
    report = LiteratureReviewWriter().run("Wi-Fi access point placement", plan, [real_paper], [analysis], comparison, [], 2011, 2026)
    result = EvidenceCritic().run(
        report, [real_paper], [bad],
        {real_paper.paper_id: [{"page": None, "location": "abstract", "text": real_paper.abstract}]},
    )
    assert not result.accepted
    assert any("not present" in issue.message for issue in result.issues)


def _run_graph(monkeypatch, settings_factory, real_paper, critic_class=None):
    import graph.workflow as workflow
    class FakeVector:
        def __init__(self, *args, **kwargs): pass
        def add(self, chunks): return len(chunks)
    monkeypatch.setattr(workflow, "VectorService", FakeVector)
    if critic_class:
        monkeypatch.setattr(workflow, "EvidenceCritic", critic_class)
    events = []
    settings = settings_factory()
    graph = build_workflow(settings, on_event=events.append, search_service=FakeSearch([real_paper]))
    state = graph.invoke({
        "topic": "Wi-Fi access point location optimization",
        "start_year": 2010, "end_year": 2026, "max_papers": 10,
        "include_openalex": True, "include_arxiv": True, "include_crossref": True,
        "revision_count": 0, "events": [], "fatal_errors": [],
    }, config={"recursion_limit": 30})
    return state, events


def test_graph_runs_all_agents_and_emits_statuses(monkeypatch, settings_factory, real_paper):
    state, events = _run_graph(monkeypatch, settings_factory, real_paper)
    assert state["report"].references
    assert state["critique"].accepted
    names = {event.agent for event in events}
    assert {"Research Planner", "Paper Discovery", "Paper Analysis", "Evidence Critic", "Citation Validation"} <= names
    assert any(event.status == "Running" for event in events)
    assert any(event.status == "Completed" for event in events)
    assert state["citation_errors"] == []
    assert state["revision_count"] == 0


def test_graph_reuses_analysis_when_paper_fingerprint_matches(monkeypatch, settings_factory, real_paper):
    import graph.workflow as workflow
    class FakeVector:
        def __init__(self, *args, **kwargs): pass
        def add(self, chunks): return len(chunks)
    monkeypatch.setattr(workflow, "VectorService", FakeVector)
    paper = real_paper.model_copy(update={"pdf_url": "https://example.org/paper.pdf", "full_text_status": "abstract-only"})
    settings = settings_factory()
    analysis = PaperAnalyzer(settings, LLMService(settings)).run(paper)
    previous = {
        "papers": [paper], "analyses": [analysis],
        "source_texts": {paper.paper_id: [{"page": None, "location": "abstract", "text": paper.abstract}]},
    }
    monkeypatch.setattr(workflow, "download_pdf", lambda *args: (_ for _ in ()).throw(AssertionError("cached paper should not download again")))
    state = build_workflow(settings, search_service=FakeSearch([paper]), reuse_result=previous).invoke({
        "topic": "Wi-Fi access point location optimization", "start_year": 2010, "end_year": 2026,
        "max_papers": 10, "include_openalex": True, "include_arxiv": True,
        "include_crossref": True, "revision_count": 0, "events": [], "fatal_errors": [],
    }, config={"recursion_limit": 30})
    analysis_event = next(event for event in state["events"] if event.agent == "Paper Analysis" and event.status == "Completed")
    assert "1 reusable analyses reused" in analysis_event.summary
    assert state["analyses"][0] == analysis


def test_critic_revision_loop_is_capped_at_two(monkeypatch, settings_factory, real_paper):
    class AlwaysReject:
        name = "Evidence Critic"
        def run(self, *args):
            return Critique(accepted=False, issues=[CriticIssue(severity="error", message="forced test condition")])
    state, events = _run_graph(monkeypatch, settings_factory, real_paper, AlwaysReject)
    writer_runs = [e for e in events if e.agent == "Literature Review Writer" and e.status == "Completed"]
    assert len(writer_runs) == 3
    assert state["revision_count"] == 2
    assert not state["critique"].accepted

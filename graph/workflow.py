from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Callable

from langgraph.graph import END, START, StateGraph

from agents.comparative_analyst import ComparativeAnalyst
from agents.critic import EvidenceCritic
from agents.gap_analyzer import ResearchGapAnalyzer
from agents.paper_analyzer import PaperAnalyzer
from agents.planner import ResearchPlanner
from agents.writer import LiteratureReviewWriter
from config import Settings
from graph.state import ResearchState
from schemas.models import PaperAnalysis, WorkflowEvent
from services.llm_service import LLMService
from services.pdf_service import download_pdf
from services.retrieval_service import chunk_pages
from services.search_service import AcademicSearchService, SearchOptions, deduplicate
from services.vector_service import VectorService

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_workflow(
    settings: Settings,
    on_event: Callable[[WorkflowEvent], None] | None = None,
    search_service: AcademicSearchService | None = None,
) :
    llm = LLMService(settings)
    searcher = search_service or AcademicSearchService(settings)
    planner = ResearchPlanner(settings, llm)
    analyzer = PaperAnalyzer(settings, llm)
    comparator = ComparativeAnalyst()
    gap_agent = ResearchGapAnalyzer()
    writer = LiteratureReviewWriter()
    critic = EvidenceCritic()

    def event(agent: str, status: str, started: str, summary: str = "", error: str | None = None) -> WorkflowEvent:
        item = WorkflowEvent(
            agent=agent, status=status, started_at=started,
            completed_at=_now() if status in ("Completed", "Failed") else None,
            summary=summary, error=error,
        )
        if on_event:
            on_event(item)
        return item

    def plan_node(state: ResearchState) -> dict:
        started = _now()
        running = event(planner.name, "Running", started)
        try:
            plan = planner.run(state["topic"], state.get("start_year"), state.get("end_year"))
            return {"plan": plan, "events": [running, event(planner.name, "Completed", started, f"{len(plan.search_queries)} queries planned.", planner.last_error)]}
        except Exception as exc:
            log.exception("Planner failed")
            return {"plan": None, "fatal_errors": [f"Planner failed: {type(exc).__name__}"], "events": [running, event(planner.name, "Failed", started, error=str(exc))]}

    def discovery_node(state: ResearchState) -> dict:
        started = _now()
        running = event("Paper Discovery", "Running", started)
        try:
            if not state.get("plan"):
                raise RuntimeError("Research plan is unavailable.")
            result = searcher.search(
                state["plan"],
                SearchOptions(
                    start_year=state.get("start_year"),
                    end_year=state.get("end_year"),
                    max_papers=state.get("max_papers", 20),
                    include_openalex=state.get("include_openalex", True),
                    include_arxiv=state.get("include_arxiv", True),
                    include_crossref=state.get("include_crossref", True),
                ),
            )
            summary = f"{len(result.papers)} unique papers; {result.api_calls} API calls; {result.failed_calls} failures; {result.cache_hits} cache hits."
            return {
                "papers": result.papers, "api_calls": result.api_calls, "failed_api_calls": result.failed_calls,
                "search_errors": result.errors,
                "events": [running, event("Paper Discovery", "Completed", started, summary, "; ".join(result.errors[:3]) or None)],
            }
        except Exception as exc:
            log.exception("Discovery failed")
            return {"papers": [], "api_calls": 0, "failed_api_calls": 1, "search_errors": [str(exc)],
                    "events": [running, event("Paper Discovery", "Failed", started, error=f"{type(exc).__name__}: {exc}")]}

    def analyze_one(paper):
        pages = []
        errors = []
        if paper.pdf_url:
            pdf = download_pdf(paper.pdf_url, paper.paper_id, settings)
            if pdf.ok:
                pages = [{"page": p.page, "text": p.text} for p in pdf.pages]
            elif pdf.error:
                errors.append(pdf.error)
        result = analyzer.run(paper, pages or None)
        paper = paper.model_copy(update={
            "full_text_status": "full-text" if pages else ("abstract-only" if paper.abstract else "failed")
        })
        return paper, result, pages, errors

    def analyze_node(state: ResearchState) -> dict:
        started = _now()
        running = event("Paper Analysis", "Running", started)
        papers = state.get("papers", [])
        if not papers:
            return {
                "analyses": [], "chunks": [],
                "events": [running, event("Paper Analysis", "Completed", started, "No papers were available to analyze.")],
            }
        output = []
        with ThreadPoolExecutor(max_workers=min(4, len(papers))) as pool:
            futures = {pool.submit(analyze_one, paper): paper for paper in papers}
            for future in as_completed(futures):
                source = futures[future]
                try:
                    output.append(future.result())
                except Exception as exc:
                    log.exception("Paper analysis failed for %s", source.paper_id)
                    output.append((source.model_copy(update={"full_text_status": "failed"}), PaperAnalysis(
                        paper_id=source.paper_id, analysis_mode="abstract-only",
                        objective="Not reported in accessible text.", problem_statement="Not reported in accessible text.",
                        methodology="Not reported in accessible text.", algorithm_model="Not reported in accessible text.", computational_complexity="Not reported in accessible text.",
                        dataset="Not reported in accessible text.", experimental_setup="Not reported in accessible text.",
                        baselines="Not reported in accessible text.", metrics="Not reported in accessible text.",
                        results="Not reported in accessible text.", limitations="Not reported in accessible text.",
                        future_work="Not reported in accessible text.", contributions="Not reported in accessible text.",
                        error=f"{type(exc).__name__}: {exc}",
                    ), [], [str(exc)]))
        output.sort(key=lambda x: x[0].relevance_score, reverse=True)
        updated_papers = [x[0] for x in output]
        analyses = [x[1] for x in output]
        chunks = []
        source_texts = {}
        for paper, _, pages, _ in output:
            sources = pages if pages else ([{"page": None, "text": paper.abstract, "location": "abstract"}] if paper.abstract else [])
            source_texts[paper.paper_id] = sources
            chunks.extend(chunk_pages(paper, sources))
        rag_errors = []
        if chunks:
            try:
                VectorService(settings).add(chunks)
            except Exception as exc:
                # Keyword retrieval remains available and the selected embedding failure is disclosed.
                rag_errors.append(f"Vector indexing unavailable: {type(exc).__name__}: {exc}")
                log.info("Vector indexing failed: %s", type(exc).__name__)
        errors = [message for *_, messages in output for message in messages]
        abstract_count = sum(p.full_text_status != "full-text" for p in updated_papers)
        summary = f"{len(analyses)} papers analyzed in parallel; {len(chunks)} page-aware chunks; {abstract_count} abstract/metadata-only."
        if errors:
            summary += f" {len(errors)} PDF access issues."
        return {
            "papers": updated_papers, "analyses": analyses, "chunks": chunks, "source_texts": source_texts, "rag_errors": rag_errors,
            "events": [running, event("Paper Analysis", "Completed", started, summary, "; ".join(errors[:3]) or None)],
        }

    def comparison_node(state: ResearchState) -> dict:
        started = _now()
        running = event(comparator.name, "Running", started)
        value = comparator.run(state.get("papers", []), state.get("analyses", []))
        return {"comparison": value, "events": [running, event(comparator.name, "Completed", started, f"{len(value.items)} papers compared with comparability caveats.")]}

    def gap_node(state: ResearchState) -> dict:
        started = _now()
        running = event(gap_agent.name, "Running", started)
        value = gap_agent.run(state.get("analyses", []))
        return {"gaps": value, "events": [running, event(gap_agent.name, "Completed", started, f"{len(value)} evidence-backed limitations/gaps identified.")]}

    def writer_node(state: ResearchState) -> dict:
        started = _now()
        running = event(writer.name, "Running", started)
        revision = state.get("revision_count", 0)
        report = writer.run(
            state["topic"], state["plan"], state.get("papers", []), state.get("analyses", []),
            state.get("comparison"), state.get("gaps", []), state.get("start_year"), state.get("end_year"),
        )
        next_revision = revision + 1 if state.get("critique") else revision
        return {
            "report": report, "revision_count": next_revision,
            "events": [running, event(writer.name, "Completed", started, f"Structured report drafted (revision {next_revision}; template writer).")],
        }

    def critic_node(state: ResearchState) -> dict:
        started = _now()
        running = event(critic.name, "Running", started)
        result = critic.run(state["report"], state.get("papers", []), state.get("analyses", []), state.get("source_texts", {}))
        summary = "Accepted." if result.accepted else "Revision requested."
        return {"critique": result, "events": [running, event(critic.name, "Completed", started, summary)]}

    def route_critique(state: ResearchState) -> str:
        if not state.get("critique") or state["critique"].accepted:
            return "accept"
        if state.get("revision_count", 0) < 2:
            return "revise"
        return "accept"

    def citation_node(state: ResearchState) -> dict:
        started = _now()
        running = event("Citation Validation", "Running", started)
        from services.citation_service import validate_citations
        errors = validate_citations(state["report"].markdown, state.get("papers", []))
        if errors:
            log.error("Citation validation failed: %s", errors)
        return {
            "citation_errors": errors,
            "llm_usage": {**llm.usage.__dict__.copy(), "total_tokens": llm.usage.total_tokens},
            "events": [running, event("Citation Validation", "Completed" if not errors else "Failed", started,
                             "All report citation IDs map to collected records." if not errors else "Invalid citation identifiers found.",
                             ", ".join(errors) if errors else None)],
        }

    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_node)
    graph.add_node("discover", discovery_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("compare", comparison_node)
    graph.add_node("gaps", gap_node)
    graph.add_node("write", writer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("citations", citation_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "discover")
    graph.add_edge("discover", "analyze")
    graph.add_edge("analyze", "compare")
    graph.add_edge("compare", "gaps")
    graph.add_edge("gaps", "write")
    graph.add_edge("write", "critic")
    graph.add_conditional_edges("critic", route_critique, {"revise": "write", "accept": "citations"})
    graph.add_edge("citations", END)
    return graph.compile()

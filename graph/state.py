from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from schemas.models import ComparativeAnalysis, Critique, LiteratureReport, Paper, PaperAnalysis, ResearchGap, ResearchPlan, WorkflowEvent


class ResearchState(TypedDict, total=False):
    topic: str
    start_year: int | None
    end_year: int | None
    max_papers: int
    include_openalex: bool
    include_arxiv: bool
    include_crossref: bool
    plan: ResearchPlan
    papers: list[Paper]
    analyses: list[PaperAnalysis]
    comparison: ComparativeAnalysis
    gaps: list[ResearchGap]
    report: LiteratureReport
    critique: Critique
    revision_count: int
    citation_errors: list[str]
    search_errors: list[str]
    api_calls: int
    failed_api_calls: int
    provider_status: dict[str, dict[str, Any]]
    llm_usage: dict[str, Any]
    chunks: list[dict[str, Any]]
    source_texts: dict[str, list[dict[str, Any]]]
    rag_errors: list[str]
    events: Annotated[list[WorkflowEvent], operator.add]
    fatal_errors: Annotated[list[str], operator.add]

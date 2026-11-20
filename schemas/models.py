from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ResearchPlan(BaseModel):
    objective: str
    scope: str
    subtopics: list[str] = Field(min_length=1, max_length=8)
    research_questions: list[str] = Field(min_length=1, max_length=8)
    search_queries: list[str] = Field(min_length=1, max_length=12)
    keywords: list[str] = Field(min_length=1, max_length=30)
    expected_evidence: list[str] = Field(min_length=1, max_length=8)


class Paper(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    source: str
    source_ids: list[str] = Field(default_factory=list)
    is_preprint: bool = False
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    full_text_status: Literal["not-fetched", "full-text", "abstract-only", "failed"] = "not-fetched"
    published_date: str | None = None

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("paper title must not be blank")
        return value


class EvidenceSpan(BaseModel):
    paper_id: str
    field: str | None = None
    quote: str = Field(min_length=8, max_length=1200)
    location: Literal["abstract", "full-text"]
    page: int | None = Field(default=None, ge=1)


class PaperAnalysis(BaseModel):
    paper_id: str
    analysis_mode: Literal["full-text", "abstract-only"]
    objective: str
    problem_statement: str
    methodology: str
    algorithm_model: str
    computational_complexity: str
    dataset: str
    experimental_setup: str
    baselines: str
    metrics: str
    results: str
    limitations: str
    future_work: str
    contributions: str
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    error: str | None = None


class ComparisonItem(BaseModel):
    paper_id: str
    methodology: str
    algorithm_model: str
    dataset: str
    metrics: str
    results: str
    computational_complexity: str
    strengths: str
    weaknesses: str
    limitations: str
    comparability_note: str


class ComparativeAnalysis(BaseModel):
    items: list[ComparisonItem] = Field(default_factory=list)
    agreements: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    trends: list[str] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class ResearchGap(BaseModel):
    gap_id: str
    title: str
    description: str
    category: Literal["author-reported limitation", "cross-paper observation", "system-generated hypothesis"]
    supporting_evidence: list[EvidenceSpan] = Field(default_factory=list)
    related_paper_ids: list[str] = Field(default_factory=list)
    why_it_matters: str
    proposed_direction: str
    confidence: Literal["low", "medium", "high"]
    confidence_explanation: str


class CriticIssue(BaseModel):
    severity: Literal["error", "warning", "info"]
    message: str
    paper_ids: list[str] = Field(default_factory=list)


class Critique(BaseModel):
    accepted: bool
    issues: list[CriticIssue] = Field(default_factory=list)


class WorkflowEvent(BaseModel):
    agent: str
    status: Literal["Pending", "Running", "Completed", "Failed"]
    started_at: str
    completed_at: str | None = None
    summary: str = ""
    error: str | None = None


class ReportSection(BaseModel):
    heading: str
    body: str


class LiteratureReport(BaseModel):
    title: str
    abstract: str
    sections: list[ReportSection]
    references: list[str]
    citation_map: dict[str, str]
    markdown: str

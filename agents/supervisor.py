from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, Field, model_validator

from config import Settings
from graph.workflow import build_workflow
from schemas.models import WorkflowEvent
from services.vector_service import VectorService


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=8, max_length=500)
    start_year: int | None = Field(default=None, ge=1900, le=2100)
    end_year: int | None = Field(default=None, ge=1900, le=2100)
    max_papers: int = Field(default=20, ge=1, le=40)
    include_openalex: bool = True
    include_arxiv: bool = True
    include_crossref: bool = True

    @model_validator(mode="after")
    def validate_range_and_sources(self):
        if self.start_year and self.end_year and self.start_year > self.end_year:
            raise ValueError("start_year must not exceed end_year")
        if not any((self.include_openalex, self.include_arxiv, self.include_crossref)):
            raise ValueError("at least one academic search source must be enabled")
        return self


class Supervisor:
    """Validates resource bounds, invokes the graph, and returns complete run state."""

    def __init__(self, settings: Settings, vector_service: VectorService | None = None):
        self.settings = settings
        self.vector_service = vector_service

    def run(
        self,
        request: ResearchRequest,
        on_event: Callable[[WorkflowEvent], None] | None = None,
        reuse_result: dict | None = None,
    ) -> dict:
        graph = build_workflow(
            self.settings, on_event=on_event, vector_service=self.vector_service,
            reuse_result=reuse_result,
        )
        return graph.invoke(
            {
                **request.model_dump(),
                "revision_count": 0,
                "events": [],
                "fatal_errors": [],
            },
            config={"recursion_limit": 30},
        )

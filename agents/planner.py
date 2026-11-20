from __future__ import annotations

import re

from config import Settings
from schemas.models import ResearchPlan
from services.llm_service import LLMService


def _keywords(topic: str) -> list[str]:
    stop = {"the", "and", "for", "with", "from", "using", "based", "into", "to", "of", "in", "on", "a", "an"}
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]+", topic)
    unique = list(dict.fromkeys(w.casefold() for w in words if len(w) > 2 and w.casefold() not in stop))
    return unique[:20] or [topic.strip()]


def heuristic_plan(topic: str, start_year: int | None, end_year: int | None) -> ResearchPlan:
    clean = " ".join(topic.split())
    words = _keywords(clean)
    span = f"Publications from {start_year or 'earliest available'} through {end_year or 'latest available'}."
    queries = [clean]
    if len(words) >= 2:
        queries += [
            f"{clean} methodology evaluation",
            f"{clean} limitations future work",
            " ".join(words[:5]) + " dataset experiments",
        ]
    return ResearchPlan(
        objective=f"Identify and synthesize scholarly evidence about {clean}.",
        scope=f"Focus on peer-reviewed records and clearly labelled preprints. {span} "
              "The review is limited to records returned by selected public academic APIs and accessible text.",
        subtopics=[
            "Research objectives and problem framing",
            "Methods, algorithms, and experimental design",
            "Datasets, metrics, and reported results",
            "Author-reported limitations and future work",
        ],
        research_questions=[
            f"What research problems and objectives are reported for {clean}?",
            "Which methods, datasets, and evaluation measures are reported?",
            "What limitations and future work do the authors state?",
        ],
        search_queries=queries[:6],
        keywords=words,
        expected_evidence=[
            "Bibliographic metadata from academic indexes",
            "Abstract or accessible full-text statements about methods and results",
            "Explicit author-reported limitations where present",
        ],
    )


class ResearchPlanner:
    name = "Research Planner"

    def __init__(self, settings: Settings, llm: LLMService):
        self.settings = settings
        self.llm = llm
        self.last_error: str | None = None

    def run(self, topic: str, start_year: int | None, end_year: int | None) -> ResearchPlan:
        self.last_error = None
        base = heuristic_plan(topic, start_year, end_year)
        if self.settings.provider == "template":
            return base
        prompt = (
            "Plan a reproducible academic literature search. Keep the user's topic central. "
            "Do not assert research findings. Topic: " + topic + "\n"
            "Publication range: " + str(start_year) + " to " + str(end_year) + "\n"
            "Candidate keywords: " + ", ".join(base.keywords)
        )
        try:
            planned = self.llm.generate_json(prompt, ResearchPlan, max_tokens=900)
            return planned if planned.search_queries else base
        except Exception as exc:
            self.last_error = f"LLM planning unavailable; deterministic query plan used ({type(exc).__name__})."
            return base

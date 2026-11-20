from __future__ import annotations

import re

from config import Settings
from schemas.models import EvidenceSpan, Paper, PaperAnalysis
from services.llm_service import LLMService


FIELDS = {
    "objective": ("aim", "objective", "purpose", "this study", "this paper", "we propose"),
    "problem_statement": ("problem", "challenge", "address", "issue", "limitation"),
    "methodology": ("method", "approach", "framework", "we propose", "we develop", "algorithm"),
    "algorithm_model": ("reinforcement learning", "machine learning", "deep learning", "neural", "algorithm", "model"),
    "computational_complexity": ("complexity", "computational cost", "runtime", "time complexity", "space complexity"),
    "dataset": ("dataset", "data set", "trace", "simulation data", "testbed"),
    "experimental_setup": ("experiment", "simulation", "testbed", "scenario", "evaluation setup"),
    "baselines": ("baseline", "compared with", "comparison with", "benchmark"),
    "metrics": ("metric", "throughput", "latency", "accuracy", "energy", "performance", "score"),
    "results": ("result", "achieve", "improve", "outperform", "show that", "demonstrate"),
    "limitations": ("limitation", "however", "challenge", "future work", "remains"),
    "future_work": ("future work", "future research", "further work", "we plan", "open problem"),
    "contributions": ("contribution", "we present", "we introduce", "we propose", "our work"),
}


def _sentences(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if len(x.strip()) >= 12]


def _evidence_field(sentence: str, paper: Paper, field: str, location: str, page: int | None) -> EvidenceSpan:
    quote = sentence.strip()
    if len(quote) > 1000:
        quote = quote[:997].rsplit(" ", 1)[0] + "..."
    return EvidenceSpan(paper_id=paper.paper_id, field=field, quote=quote, location=location, page=page)


def _extract_template(paper: Paper, pages: list[dict] | None) -> PaperAnalysis:
    sources: list[tuple[str, int | None, str]] = []
    if pages:
        sources.extend(("full-text", int(p["page"]), p["text"]) for p in pages if p.get("text"))
    if not sources and paper.abstract:
        sources.append(("abstract", None, paper.abstract))
    all_sentences: list[tuple[str, int | None, str]] = []
    for location, page, text in sources:
        all_sentences.extend((s, page, location) for s in _sentences(text))
    values: dict[str, str] = {}
    evidence: list[EvidenceSpan] = []
    for field, terms in FIELDS.items():
        match = next(
            ((sentence, page, loc) for sentence, page, loc in all_sentences
             if any(term in sentence.casefold() for term in terms)),
            None,
        )
        if match:
            sentence, page, loc = match
            values[field] = sentence
            evidence.append(_evidence_field(sentence, paper, field, loc, page))
        else:
            values[field] = "Not reported in accessible text."
    return PaperAnalysis(
        paper_id=paper.paper_id,
        analysis_mode="full-text" if pages else "abstract-only",
        **values,
        evidence=evidence,
    )


class PaperAnalyzer:
    name = "Paper Analysis"

    def __init__(self, settings: Settings, llm: LLMService):
        self.settings = settings
        self.llm = llm

    def run(self, paper: Paper, pages: list[dict] | None = None) -> PaperAnalysis:
        template = _extract_template(paper, pages)
        if self.settings.provider == "template":
            return template
        raw_pages = pages or ([{"page": None, "text": paper.abstract}] if paper.abstract else [])
        source = "\n".join(f"Page {p.get('page') or 'abstract'}: {p.get('text', '')}" for p in raw_pages)
        source = source[:14000]
        if not source.strip():
            template.error = "No accessible abstract or full text was available for analysis."
            return template
        prompt = (
            "Analyze one scholarly paper. Each non-empty finding must be supported by an exact short quote "
            "from the supplied source. For fields without support, return 'Not reported in accessible text.' "
            "Preserve page numbers exactly. Mark abstract-only when only an abstract is supplied.\n"
            f"Paper: {paper.title}\nMode: {template.analysis_mode}\nSource text:\n{source}"
        )
        try:
            result = self.llm.generate_json(prompt, PaperAnalysis, max_tokens=1000)
            page_sources = {p.get("page"): p.get("text", "") for p in raw_pages if p.get("page") is not None}
            abstract_source = paper.abstract if not pages else ""
            for evidence in result.evidence:
                if evidence.paper_id != paper.paper_id:
                    raise ValueError("LLM evidence referenced a different paper ID.")
                if evidence.location == "abstract":
                    if evidence.page is not None or not abstract_source or evidence.quote not in abstract_source:
                        raise ValueError("LLM supplied an invalid abstract citation or page number.")
                elif evidence.page is None or evidence.page not in page_sources or evidence.quote not in page_sources[evidence.page]:
                    raise ValueError("LLM evidence page or quote did not exactly match the full text.")
            if result.analysis_mode != template.analysis_mode:
                raise ValueError("LLM changed the source access mode.")
            for field in FIELDS:
                value = getattr(result, field)
                if value != "Not reported in accessible text." and not any(e.field == field for e in result.evidence):
                    raise ValueError(f"LLM finding for {field} has no field-matched evidence span.")
            return result
        except Exception as exc:
            template.error = f"LLM analysis unavailable; evidence-template extraction used ({type(exc).__name__})."
            return template

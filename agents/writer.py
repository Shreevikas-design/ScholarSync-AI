from __future__ import annotations

import re
import hashlib

from pydantic import BaseModel, Field

from schemas.models import ComparativeAnalysis, LiteratureReport, Paper, PaperAnalysis, ReportSection, ResearchGap, ResearchPlan
from services.citation_service import citation_map, reference_text
from services.llm_service import LLMService


class EvidenceReference(BaseModel):
    paper_id: str
    quote: str = Field(min_length=8, max_length=1200)


class SynthesisClaim(BaseModel):
    text: str = Field(min_length=20, max_length=1200)
    evidence: list[EvidenceReference] = Field(min_length=1, max_length=5)


class ThemeSynthesis(BaseModel):
    theme: str = Field(min_length=3, max_length=100)
    claims: list[SynthesisClaim] = Field(min_length=1, max_length=6)


class SynthesisDraft(BaseModel):
    themes: list[ThemeSynthesis] = Field(default_factory=list, max_length=8)
    cross_paper_claims: list[SynthesisClaim] = Field(default_factory=list, max_length=8)
    future_directions: list[SynthesisClaim] = Field(default_factory=list, max_length=6)


def _format_analysis(paper: Paper, analysis: PaperAnalysis, cmap: dict[str, str]) -> str:
    marker = cmap.get(paper.paper_id, "")
    status = {
        "abstract-only": "Abstract-only analysis",
        "metadata-only": "Metadata-only analysis",
        "partial-full-text": "Partial full-text analysis",
    }.get(analysis.analysis_mode, "Full-text analysis")
    lines = [f"**{paper.title}** {marker} ({status}; {paper.year or 'year not supplied'})."]
    for name in ("objective", "methodology", "algorithm_model", "computational_complexity", "dataset", "sensor_types", "feature_extraction", "metrics", "results", "contributions", "limitations", "future_work"):
        value = getattr(analysis, name)
        if value != "Not reported in accessible text.":
            suffix = f" {marker}" if any(e.field == name for e in analysis.evidence) else ""
            lines.append(f"- {name.replace('_', ' ').capitalize()}: {value}{suffix}")
    return "\n".join(lines)


class LiteratureReviewWriter:
    name = "Literature Review Writer"

    def __init__(self, llm: LLMService | None = None):
        self.llm = llm
        self.last_error: str | None = None
        self._synthesis_cache: dict[str, SynthesisDraft | None] = {}
        self._synthesis_error_cache: dict[str, str | None] = {}

    @staticmethod
    def _claim_text(claim: SynthesisClaim, papers: dict[str, Paper], cmap: dict[str, str]) -> str:
        refs = list(dict.fromkeys(cmap[item.paper_id] for item in claim.evidence if item.paper_id in papers and item.paper_id in cmap))
        return claim.text.rstrip() + (" " + " ".join(refs) if refs else "")

    def _synthesize(
        self, topic: str, records: list[Paper], by_id: dict[str, PaperAnalysis],
    ) -> SynthesisDraft | None:
        self.last_error = None
        if not self.llm or self.llm.settings.provider == "template" or not records:
            return None
        output_budget = min(1200, self.llm.settings.max_output_tokens, max(128, self.llm.settings.max_context_tokens // 5))
        budget = max(0, min(16000, (self.llm.settings.max_context_tokens - output_budget - 1800) * 3))
        if budget < 500:
            self.last_error = "The configured model context is too small for bounded cross-paper synthesis."
            return None
        lines = []
        allowed: dict[str, dict[str, str | None]] = {}
        used = 0
        for paper in sorted(records, key=lambda item: item.relevance_score, reverse=True)[:12]:
            analysis = by_id[paper.paper_id]
            excerpts = []
            selected_evidence = []
            priority = {"future_work": 0, "results": 1, "limitations": 2, "methodology": 3, "dataset": 4}
            selected_spans = sorted(analysis.evidence, key=lambda item: priority.get(item.field or "", 5))[:4]
            for evidence in selected_spans:
                quote = evidence.quote
                excerpts.append(f"- {evidence.field or 'evidence'} | {evidence.location} | page {evidence.page or 'n/a'}: {quote}")
                selected_evidence.append((quote, evidence.field))
            block = (
                f"Paper ID: {paper.paper_id}; title: {paper.title}; year: {paper.year or 'not supplied'}; "
                f"analysis scope: {analysis.analysis_mode}\n"
                f"Method: {analysis.methodology}; model: {analysis.algorithm_model}; dataset: {analysis.dataset}; "
                f"metrics: {analysis.metrics}; results: {analysis.results}; limitations: {analysis.limitations}; "
                f"future work: {analysis.future_work}\n" + "\n".join(excerpts)
            )
            if used + len(block) > budget:
                continue
            lines.append(block)
            allowed[paper.paper_id] = {quote: field for quote, field in selected_evidence}
            used += len(block)
        if not lines:
            self.last_error = "The configured context budget could not fit a validated paper summary."
            return None
        prompt = (
            "Synthesize the supplied scholarly paper summaries for this review topic: " + topic + ".\n"
            "Group only themes represented in the supplied evidence. Claims must be cautious and must cite "
            "one or more exact evidence excerpts by copying their Paper ID and quote into the evidence array. "
            "Do not return citation numbers, outside facts, invented findings, or unsupported generalizations. "
            "Only make cross-paper claims when at least two distinct papers support them. Future directions "
            "must follow reported future-work evidence or be framed as a question. Keep claims concise.\n\n"
            + "\n\n".join(lines)
        )
        cache_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if cache_key in self._synthesis_cache:
            self.last_error = self._synthesis_error_cache.get(cache_key)
            return self._synthesis_cache[cache_key]
        try:
            draft = self.llm.generate_json(prompt, SynthesisDraft, max_tokens=min(1200, self.llm.settings.max_output_tokens))
            known = set(allowed)

            def valid_claim(claim: SynthesisClaim, require_multiple: bool = False, future_only: bool = False) -> bool:
                if re.search(r"\[\d+\]", claim.text):
                    return False
                paper_ids = set()
                fields = []
                for ref in claim.evidence:
                    if ref.paper_id not in known:
                        return False
                    matched_fields = [field for supplied, field in allowed[ref.paper_id].items() if ref.quote in supplied or supplied in ref.quote]
                    if not matched_fields:
                        return False
                    fields.extend(matched_fields)
                    paper_ids.add(ref.paper_id)
                if require_multiple and len(paper_ids) < 2:
                    return False
                if future_only and "future_work" not in fields and not re.search(r"\b(could|may|might|whether|potential|investigate)\b", claim.text, re.I):
                    return False
                return True

            def unique_claims(claims: list[SynthesisClaim], multiple: bool = False, future_only: bool = False) -> list[SynthesisClaim]:
                seen = set()
                output = []
                for claim in claims:
                    key = " ".join(claim.text.casefold().split())
                    if key in seen or not valid_claim(claim, multiple, future_only):
                        continue
                    seen.add(key)
                    output.append(claim)
                return output

            draft.themes = [theme.model_copy(update={"claims": unique_claims(theme.claims)})
                            for theme in draft.themes]
            draft.themes = [theme for theme in draft.themes if theme.claims]
            draft.cross_paper_claims = unique_claims(draft.cross_paper_claims, True)
            draft.future_directions = unique_claims(draft.future_directions, future_only=True)
            if not draft.themes and not draft.cross_paper_claims:
                self.last_error = "Structured synthesis had no claims with verifiable evidence references."
                self._synthesis_cache[cache_key] = None
                self._synthesis_error_cache[cache_key] = self.last_error
                return None
            self._synthesis_cache[cache_key] = draft
            self._synthesis_error_cache[cache_key] = None
            return draft
        except Exception as exc:
            self.last_error = f"Evidence synthesis unavailable ({type(exc).__name__}); deterministic evidence summary used."
            self._synthesis_cache[cache_key] = None
            self._synthesis_error_cache[cache_key] = self.last_error
            return None

    def run(
        self, topic: str, plan: ResearchPlan, papers: list[Paper], analyses: list[PaperAnalysis],
        comparison: ComparativeAnalysis, gaps: list[ResearchGap], start_year: int | None, end_year: int | None,
    ) -> LiteratureReport:
        cmap = citation_map(papers)
        by_id = {a.paper_id: a for a in analyses}
        records = [p for p in papers if p.paper_id in by_id]
        theme_groups: dict[str, list[str]] = {}
        for paper in records:
            analysis = by_id[paper.paper_id]
            method_text = (analysis.methodology + " " + analysis.algorithm_model).casefold()
            if "reinforcement learning" in method_text:
                theme = "Reinforcement-learning methods"
            elif any(term in method_text for term in ("machine learning", "deep learning", "neural network")):
                theme = "Machine-learning methods"
            elif any(term in method_text for term in ("optimization", "placement", "algorithm")):
                theme = "Optimization and placement methods"
            elif any(term in method_text for term in ("simulation", "testbed", "experiment")):
                theme = "Simulation and experimental studies"
            else:
                theme = "Other reported methods and objectives"
            theme_groups.setdefault(theme, []).append(_format_analysis(paper, analysis, cmap))
        themed = "\n\n".join("### " + theme + "\n\n" + "\n\n".join(bodies) for theme, bodies in theme_groups.items())
        synthesis = self._synthesize(topic, records, by_id)
        if synthesis:
            paper_lookup = {paper.paper_id: paper for paper in papers}
            generated_themes = []
            for theme in synthesis.themes:
                claims = [self._claim_text(claim, paper_lookup, cmap) for claim in theme.claims]
                generated_themes.append("### " + theme.theme + "\n\n" + "\n\n".join(claims))
            synthesized_paper_ids = {
                item.paper_id for theme in synthesis.themes for claim in theme.claims for item in claim.evidence
            }
            additional = [
                _format_analysis(paper, by_id[paper.paper_id], cmap)
                for paper in records if paper.paper_id not in synthesized_paper_ids
            ]
            if additional:
                generated_themes.append("### Additional retrieved studies\n\n" + "\n\n".join(additional))
            if generated_themes:
                themed = "\n\n".join(generated_themes)
            else:
                self.last_error = self.last_error or "Synthesis returned no validated thematic claims; deterministic evidence summary used."
        if not themed:
            themed = "No records were returned by the selected sources. No research findings or citations are presented."
        rows = []
        for item in comparison.items:
            paper = next((p for p in papers if p.paper_id == item.paper_id), None)
            if paper:
                cells = [
                    paper.title, item.methodology, item.algorithm_model, item.computational_complexity,
                    item.dataset, item.sensor_types, item.feature_extraction, item.experimental_setup, item.baselines,
                    item.metrics, item.results, item.strengths, item.weaknesses,
                    item.comparability_note,
                ]
                cells = [x.replace("|", "/")[:240] for x in cells]
                rows.append("| " + " | ".join(cells) + f" {_cite(paper, cmap)} |")
        header = "| Paper | Methodology | Algorithm/model | Complexity | Dataset | Sensors | Feature extraction | Experimental setup | Baselines | Metrics | Results | Strengths reported | Weaknesses reported | Comparability caveat |"
        separator = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
        table = header + "\n" + separator + "\n" + "\n".join(rows) if rows else "No comparable records are available."
        gap_rows = []
        for gap in gaps:
            snippets = []
            for span in gap.supporting_evidence:
                paper = next((item for item in papers if item.paper_id == span.paper_id), None)
                marker = cmap.get(span.paper_id, "")
                if paper and marker:
                    location = f", p. {span.page}" if span.page else ""
                    snippets.append(f"“{span.quote}” {marker}{location}")
            gap_rows.append(
                f"- **{gap.title}** ({gap.category}; confidence {gap.confidence}; evidence coverage {gap.evidence_coverage:.0%}): {gap.description} "
                + " ".join(cmap.get(pid, "") for pid in gap.related_paper_ids)
                + f" Why it matters: {gap.why_it_matters} Proposed direction: {gap.proposed_direction} "
                + ("Supporting evidence: " + "; ".join(snippets) if snippets else "No supporting quote was available.")
            )
        gap_text = "\n\n".join(gap_rows) or "No evidence-supported research gaps could be established from the accessible text. This is not evidence that no gaps exist."
        major_items = []
        for paper in records:
            analysis = by_id[paper.paper_id]
            finding = analysis.results if analysis.results != "Not reported in accessible text." else analysis.contributions
            if finding != "Not reported in accessible text.":
                major_items.append(f"- {paper.title}: {finding} {_cite(paper, cmap)}")
            if len(major_items) >= 6:
                break
        major_findings = "\n".join(major_items) or "The retrieved abstracts and accessible text did not support a concise cross-paper finding statement."
        if synthesis and synthesis.future_directions:
            future_text = "\n\n".join(self._claim_text(claim, {p.paper_id: p for p in papers}, cmap) for claim in synthesis.future_directions)
        else:
            future_text = "\n\n".join(g.proposed_direction + " " + " ".join(cmap.get(pid, "") for pid in g.related_paper_ids) for g in gaps) or "Future directions cannot be inferred from this evidence set; consult included papers' full text for explicit author proposals."
        citation_list = [f"{cmap[p.paper_id]} {reference_text(p)}" for p in papers]
        source_note = f"Search window: {start_year or 'earliest available'}-{end_year or 'latest available'}. Records: {len(papers)}; partial-full-text analyses: {sum(a.analysis_mode == 'partial-full-text' for a in analyses)}; abstract-only analyses: {sum(a.analysis_mode == 'abstract-only' for a in analyses)}; metadata-only analyses: {sum(a.analysis_mode == 'metadata-only' for a in analyses)}."
        synthesis_text = "\n\n".join(self._claim_text(claim, {p.paper_id: p for p in papers}, cmap) for claim in synthesis.cross_paper_claims) if synthesis else "The retrieved evidence does not establish cross-paper agreement or contradiction."
        if not synthesis_text.strip():
            synthesis_text = "The retrieved evidence does not establish cross-paper agreement or contradiction."
        sections = [
            ReportSection(heading="Introduction", body=f"This review examines {topic}. {source_note} Scope: {plan.scope}"),
            ReportSection(heading="Research background", body=f"Objective: {plan.objective}\n\nQuestions: " + "; ".join(plan.research_questions)),
            ReportSection(heading="Review methodology", body=f"Searches were planned from the topic, queried the enabled academic APIs, deduplicated records by DOI/arXiv ID/exact normalized title, and filtered years to {start_year or 'earliest available'}–{end_year or 'latest available'}. The report includes {len(papers)} retrieved records; analysis scope is reported per paper as full text, partial full text, abstract only, or metadata only."),
            ReportSection(heading="Literature review and thematic literature synthesis", body=themed),
            ReportSection(
                heading="Methodological comparison",
                body="Comparative analysis across consistently reported dimensions; missing fields and incompatible experimental conditions are marked rather than inferred.\n\n" + table,
            ),
            ReportSection(heading="Cross-paper synthesis", body="Agreement, contradiction, and trend statements are limited to supported evidence. " + ("\n\n".join(comparison.agreements + comparison.contradictions + comparison.trends) if comparison.agreements or comparison.contradictions or comparison.trends else synthesis_text) + ("\n\n" + "\n".join(comparison.caveats) if comparison.caveats else "")),
            ReportSection(heading="Major findings", body=major_findings),
            ReportSection(heading="Research limitations", body="Author-reported limitations: " + ("\n\n".join(f"{p.title} {_cite(p, cmap)}: {by_id[p.paper_id].limitations}" for p in records if by_id[p.paper_id].limitations != "Not reported in accessible text.") or "Accessible texts did not report limitations for the remaining records.") + "\n\nCorpus limitation: only returned API records and publicly accessible text are included."),
            ReportSection(heading="Research gaps", body=gap_text),
            ReportSection(heading="Limitations of this review", body="This review is bounded by public-index coverage, the selected query plan, publication window, accessible full or partial text, abstracts, metadata-only records, and the small set of retrieved papers. It should not be read as coverage of the entire research field."),
            ReportSection(heading="Future research directions", body=future_text),
            ReportSection(heading="Conclusion", body=f"The collected records provide a bounded view of evidence about {topic}. Interpret findings according to source coverage and whether each analysis used full text, partial full text, an abstract, or metadata only."),
        ]
        abstract = (f"This evidence-grounded review summarizes {len(papers)} records returned for {topic}. "
                    "It distinguishes full-text, partial-full-text, abstract-only, and metadata-only analyses and limits comparisons to reported fields. "
                    "Missing metadata is not treated as evidence of a research gap.")
        markdown = self._markdown(topic, abstract, sections, citation_list)
        return LiteratureReport(title=f"Literature Review: {topic}", abstract=abstract, sections=sections,
                                references=citation_list, citation_map=cmap, markdown=markdown)

    @staticmethod
    def _markdown(topic: str, abstract: str, sections: list[ReportSection], references: list[str]) -> str:
        lines = [f"# Literature Review: {topic}", "", "## Abstract", "", abstract]
        for section in sections:
            lines.extend(["", f"## {section.heading}", "", section.body])
        lines.extend(["", "## References", ""])
        lines.extend(references or ["No records returned by selected academic sources."])
        lines.append("")
        return "\n".join(lines)


def _cite(paper: Paper, cmap: dict[str, str]) -> str:
    return cmap.get(paper.paper_id, "")

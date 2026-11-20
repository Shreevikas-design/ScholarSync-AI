from __future__ import annotations

from schemas.models import ComparativeAnalysis, LiteratureReport, Paper, PaperAnalysis, ReportSection, ResearchGap, ResearchPlan
from services.citation_service import citation_map, reference_text


def _format_analysis(paper: Paper, analysis: PaperAnalysis, cmap: dict[str, str]) -> str:
    marker = cmap.get(paper.paper_id, "")
    status = "Abstract-only analysis" if analysis.analysis_mode == "abstract-only" else "Full-text analysis"
    lines = [f"**{paper.title}** {marker} ({status}; {paper.year or 'year not supplied'})."]
    for name in ("objective", "methodology", "algorithm_model", "computational_complexity", "dataset", "metrics", "results", "contributions", "limitations", "future_work"):
        value = getattr(analysis, name)
        if value != "Not reported in accessible text.":
            suffix = f" {marker}" if any(e.quote == value for e in analysis.evidence) else ""
            lines.append(f"- {name.replace('_', ' ').capitalize()}: {value}{suffix}")
    return "\n".join(lines)


class LiteratureReviewWriter:
    name = "Literature Review Writer"

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
        if not themed:
            themed = "No records were returned by the selected sources. No research findings or citations are presented."
        rows = []
        for item in comparison.items:
            paper = next((p for p in papers if p.paper_id == item.paper_id), None)
            if paper:
                cells = [
                    paper.title, item.methodology, item.algorithm_model, item.computational_complexity,
                    item.dataset, item.metrics, item.results, item.strengths, item.weaknesses,
                    item.comparability_note,
                ]
                cells = [x.replace("|", "/")[:240] for x in cells]
                rows.append("| " + " | ".join(cells) + f" {_cite(paper, cmap)} |")
        header = "| Paper | Methodology | Algorithm/model | Complexity | Dataset | Metrics | Results | Strengths reported | Weaknesses reported | Comparability caveat |"
        separator = "|---|---|---|---|---|---|---|---|---|---|"
        table = header + "\n" + separator + "\n" + "\n".join(rows) if rows else "No comparable records are available."
        gap_text = "\n\n".join(
            f"- **{g.title}** ({g.category}; confidence {g.confidence}): {g.description} "
            + " ".join(cmap.get(pid, "") for pid in g.related_paper_ids)
            + f" Why it matters: {g.why_it_matters} Proposed direction: {g.proposed_direction}"
            for g in gaps
        ) or "No evidence-supported research gaps could be established from the accessible text. This is not evidence that no gaps exist."
        citation_list = [f"{cmap[p.paper_id]} {reference_text(p)}" for p in papers]
        source_note = f"Search window: {start_year or 'earliest available'}-{end_year or 'latest available'}. Records: {len(papers)}; abstract-only analyses: {sum(a.analysis_mode == 'abstract-only' for a in analyses)}."
        sections = [
            ReportSection(heading="Introduction", body=f"This review examines {topic}. {source_note} Scope: {plan.scope}"),
            ReportSection(heading="Research background", body=f"Objective: {plan.objective}\n\nQuestions: " + "; ".join(plan.research_questions)),
            ReportSection(heading="Literature review organized by themes", body=themed),
            ReportSection(heading="Methodological comparison", body=table),
            ReportSection(heading="Cross-paper synthesis", body="Agreement, contradiction, and trend statements are limited to supported evidence. " + ("\n\n".join(comparison.agreements + comparison.contradictions + comparison.trends) if comparison.agreements or comparison.contradictions or comparison.trends else "The retrieved evidence does not establish cross-paper agreement or contradiction.") + ("\n\n" + "\n".join(comparison.caveats) if comparison.caveats else "")),
            ReportSection(heading="Major findings", body=themed),
            ReportSection(heading="Research limitations", body="Author-reported limitations: " + ("\n\n".join(f"{p.title} {_cite(p, cmap)}: {by_id[p.paper_id].limitations}" for p in records if by_id[p.paper_id].limitations != "Not reported in accessible text.") or "Accessible texts did not report limitations for the remaining records.") + "\n\nCorpus limitation: only returned API records and publicly accessible text are included."),
            ReportSection(heading="Research gaps", body=gap_text),
            ReportSection(heading="Future research directions", body="\n\n".join(g.proposed_direction + " " + " ".join(cmap.get(pid, "") for pid in g.related_paper_ids) for g in gaps) or "Future directions cannot be inferred from this evidence set; consult included papers' full text for explicit author proposals."),
            ReportSection(heading="Conclusion", body=f"The collected records provide a bounded view of evidence about {topic}. Interpret findings according to source coverage and full-text versus abstract-only labels."),
        ]
        abstract = (f"This evidence-grounded review summarizes {len(papers)} records returned for {topic}. "
                    "It distinguishes full-text and abstract-only analyses and limits comparisons to reported fields. "
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

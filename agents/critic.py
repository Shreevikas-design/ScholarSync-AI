from __future__ import annotations

import re

from schemas.models import CriticIssue, Critique, LiteratureReport, Paper, PaperAnalysis
from services.citation_service import validate_citations


class EvidenceCritic:
    name = "Evidence Critic"

    def run(self, report: LiteratureReport, papers: list[Paper], analyses: list[PaperAnalysis], source_texts: dict | None = None) -> Critique:
        issues: list[CriticIssue] = []
        known = {p.paper_id for p in papers}
        for analysis in analyses:
            if analysis.paper_id not in known:
                issues.append(CriticIssue(severity="error", message="Analysis references a paper outside the discovered corpus."))
            if analysis.analysis_mode == "abstract-only" and "abstract-only" not in report.markdown.casefold():
                issues.append(CriticIssue(severity="error", message="Abstract-only analysis was not labelled in the report."))
            source = next((paper for paper in papers if paper.paper_id == analysis.paper_id), None)
            for evidence in analysis.evidence:
                if evidence.paper_id not in known or not evidence.quote:
                    issues.append(CriticIssue(severity="error", message="Evidence span has an invalid paper ID or empty quote.", paper_ids=[evidence.paper_id]))
                    continue
                if source_texts is not None:
                    if evidence.location == "abstract":
                        supported = bool(source and evidence.page is None and evidence.quote in source.abstract)
                    else:
                        supported = any(
                            evidence.page == page.get("page") and evidence.quote in page.get("text", "")
                            for page in source_texts.get(analysis.paper_id, [])
                        )
                    if not supported:
                        issues.append(CriticIssue(severity="error", message="Evidence quote is not present in the supplied source text.", paper_ids=[analysis.paper_id]))
        for line in report.markdown.splitlines():
            if line.startswith("- ") and not re.search(r"\[\d+\]", line):
                issues.append(CriticIssue(severity="error", message="A report bullet claim is missing a collected-paper citation."))
        invalid = validate_citations(report.markdown, papers)
        if invalid:
            issues.append(CriticIssue(severity="error", message="Report contains citation numbers without a matching collected paper: " + ", ".join(sorted(set(invalid)))))
        missing_refs = [f"[{i}]" for i in range(1, len(papers) + 1) if f"[{i}]" not in report.markdown]
        if papers and missing_refs:
            issues.append(CriticIssue(severity="warning", message="Some collected references are not cited in the report: " + ", ".join(missing_refs)))
        required = ("Abstract", "Introduction", "Research background", "Literature review", "Methodological comparison",
                    "Cross-paper synthesis", "Major findings", "Research limitations", "Research gaps",
                    "Future research directions", "Conclusion", "References")
        for heading in required:
            if heading.casefold() not in report.markdown.casefold():
                issues.append(CriticIssue(severity="error", message=f"Required report section is missing: {heading}."))
        if not papers:
            issues.append(CriticIssue(severity="info", message="No papers were returned by the selected sources; report contains no research findings."))
        return Critique(accepted=not any(i.severity == "error" for i in issues), issues=issues)

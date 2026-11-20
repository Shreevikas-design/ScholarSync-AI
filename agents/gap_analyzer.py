from __future__ import annotations

import re

from schemas.models import EvidenceSpan, PaperAnalysis, ResearchGap


class ResearchGapAnalyzer:
    name = "Research Gap Analysis"

    def run(self, analyses: list[PaperAnalysis]) -> list[ResearchGap]:
        gaps: list[ResearchGap] = []
        for analysis in analyses:
            if analysis.limitations != "Not reported in accessible text.":
                evidence = [e for e in analysis.evidence if e.quote == analysis.limitations]
                if evidence:
                    gaps.append(ResearchGap(
                        gap_id=f"AG-{len(gaps)+1:02d}",
                        title="Author-reported limitation",
                        description=analysis.limitations,
                        category="author-reported limitation",
                        supporting_evidence=evidence,
                        related_paper_ids=[analysis.paper_id],
                        why_it_matters="The paper itself identifies this constraint, which bounds how its findings should be interpreted.",
                        proposed_direction=analysis.future_work if analysis.future_work != "Not reported in accessible text." else "Treat this as a candidate direction; the accessible text does not state a specific follow-up.",
                        confidence="high" if analysis.analysis_mode == "full-text" else "medium",
                        confidence_explanation=f"Quoted author text supports this limitation; analysis used {analysis.analysis_mode}.",
                    ))
        groups: dict[str, list[EvidenceSpan]] = {}
        for analysis in analyses:
            if analysis.limitations == "Not reported in accessible text.":
                continue
            for evidence in analysis.evidence:
                if evidence.quote != analysis.limitations:
                    continue
                terms = {t for t in re.findall(r"[a-z]{5,}", evidence.quote.casefold())}
                key = " ".join(sorted(terms))
                if key:
                    groups.setdefault(key, []).append(evidence)
        for evidence in groups.values():
            paper_ids = list(dict.fromkeys(e.paper_id for e in evidence))
            if len(paper_ids) >= 2:
                gaps.append(ResearchGap(
                    gap_id=f"CG-{len(gaps)+1:02d}",
                    title="Repeated author-reported limitation",
                    description="Related retrieved papers report similar limitation language; this is an observed pattern in this corpus, not a field-wide claim.",
                    category="cross-paper observation",
                    supporting_evidence=evidence,
                    related_paper_ids=paper_ids,
                    why_it_matters="A limitation reported independently in multiple included papers may constrain the evidence represented by this corpus.",
                    proposed_direction="Compare the affected conditions across the cited papers and evaluate a remedy under matched experimental conditions.",
                    confidence="medium",
                    confidence_explanation="At least two distinct paper IDs provide matching limitation evidence; corpus coverage remains limited.",
                ))
        dataset_evidence = []
        dataset_paper_ids = []
        dataset_labels = set()
        for analysis in analyses:
            if analysis.dataset == "Not reported in accessible text.":
                continue
            spans = [e for e in analysis.evidence if e.quote == analysis.dataset]
            if not spans:
                continue
            dataset_labels.add(" ".join(analysis.dataset.casefold().split()))
            dataset_paper_ids.append(analysis.paper_id)
            dataset_evidence.extend(spans)
        if len(set(dataset_paper_ids)) >= 2 and len(dataset_labels) >= 2:
            gaps.append(ResearchGap(
                gap_id="SH-01",
                title="Shared benchmark hypothesis",
                description=(
                    "The included papers report different evaluation datasets. This supports a testable question "
                    "about whether method comparisons change under a shared benchmark; it does not establish a "
                    "field-wide research gap."
                ),
                category="system-generated hypothesis",
                supporting_evidence=dataset_evidence,
                related_paper_ids=list(dict.fromkeys(dataset_paper_ids)),
                why_it_matters="Dataset differences can confound comparisons between methods and experimental settings.",
                proposed_direction="Replicate the reported methods on a documented shared benchmark with common metrics and conditions.",
                confidence="low",
                confidence_explanation="This hypothesis is generated from distinct dataset statements in a small retrieved corpus; it needs domain review and broader search.",
            ))
        return gaps

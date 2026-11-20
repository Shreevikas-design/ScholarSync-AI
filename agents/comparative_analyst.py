from __future__ import annotations

from schemas.models import ComparativeAnalysis, ComparisonItem, Paper, PaperAnalysis


class ComparativeAnalyst:
    name = "Comparative Analysis"

    def run(self, papers: list[Paper], analyses: list[PaperAnalysis]) -> ComparativeAnalysis:
        by_id = {a.paper_id: a for a in analyses}
        items = []
        fields = ("dataset", "metrics")
        for paper in papers:
            analysis = by_id.get(paper.paper_id)
            if not analysis:
                continue
            reported = [getattr(analysis, field) != "Not reported in accessible text." for field in fields]
            note = (
                "Numerical results are not directly comparable: datasets or metrics are missing or differ."
                if not all(reported)
                else "Reported metrics and datasets must still be checked for matching units and conditions before numeric comparison."
            )
            items.append(ComparisonItem(
                paper_id=paper.paper_id,
                methodology=analysis.methodology,
                algorithm_model=analysis.algorithm_model,
                dataset=analysis.dataset,
                metrics=analysis.metrics,
                results=analysis.results,
                computational_complexity=analysis.computational_complexity,
                strengths=analysis.contributions,
                weaknesses=analysis.limitations,
                limitations=analysis.limitations,
                comparability_note=note,
            ))
        return ComparativeAnalysis(
            items=items,
            caveats=[
                "The corpus combines records with different access levels; abstract-only fields are incomplete.",
                "No cross-paper numerical comparison is made without matching datasets, metrics, and experimental conditions.",
            ] if items else [],
        )

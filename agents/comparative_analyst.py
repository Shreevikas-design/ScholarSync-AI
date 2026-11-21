from __future__ import annotations

from schemas.models import ComparativeAnalysis, ComparisonItem, Paper, PaperAnalysis


class ComparativeAnalyst:
    name = "Comparative Analysis"

    def run(self, papers: list[Paper], analyses: list[PaperAnalysis]) -> ComparativeAnalysis:
        by_id = {a.paper_id: a for a in analyses}
        items = []
        for paper in papers:
            analysis = by_id.get(paper.paper_id)
            if not analysis:
                continue
            items.append(ComparisonItem(
                paper_id=paper.paper_id,
                methodology=analysis.methodology,
                algorithm_model=analysis.algorithm_model,
                dataset=analysis.dataset,
                sensor_types=analysis.sensor_types,
                feature_extraction=analysis.feature_extraction,
                experimental_setup=analysis.experimental_setup,
                baselines=analysis.baselines,
                metrics=analysis.metrics,
                results=analysis.results,
                computational_complexity=analysis.computational_complexity,
                strengths=analysis.contributions,
                weaknesses=analysis.limitations,
                limitations=analysis.limitations,
                comparability_note="",
            ))
        dimensions = ("dataset", "metrics", "experimental_setup")
        missing = [field for field in dimensions if any(getattr(item, field) == "Not reported in accessible text." for item in items)]
        differing = [
            field for field in dimensions
            if len({" ".join(getattr(item, field).casefold().split()) for item in items
                    if getattr(item, field) != "Not reported in accessible text."}) > 1
        ]
        if len(items) < 2:
            note = "Only one analyzed paper is available; no cross-paper numerical comparison can be made."
        elif missing:
            note = "Qualitative comparison only: missing reported " + ", ".join(missing) + " prevent matched numerical comparison."
        elif differing:
            note = "Qualitative comparison only: reported " + ", ".join(differing) + " differ across papers."
        else:
            note = "Reported conditions appear aligned, but verify units, metric definitions, and implementation details before comparing numerical results."
        items = [item.model_copy(update={"comparability_note": note}) for item in items]
        return ComparativeAnalysis(
            items=items,
            caveats=[
                "The corpus combines records with different access levels; abstract-only fields are incomplete.",
                "No cross-paper numerical comparison is made without matching datasets, metrics, and experimental conditions.",
            ] if items else [],
        )

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from agents.planner import heuristic_plan
from evaluation.metrics import (
    citation_validity, evidence_coverage, mean_reciprocal_rank, precision_at_k,
    recall_at_k, unsupported_claim_count,
)
from schemas.models import LiteratureReport, Paper, PaperAnalysis
from services.citation_service import citation_map
from services.search_service import AcademicSearchService, SearchOptions


def evaluate(dataset_path: Path, search_service: AcademicSearchService, k: int = 5) -> dict:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows = []
    elapsed = 0.0
    for case in dataset["cases"]:
        plan = heuristic_plan(case["query"], None, None)
        started = time.perf_counter()
        result = search_service.search(plan, SearchOptions(max_papers=max(k, 10)))
        elapsed += time.perf_counter() - started
        ids = [p.paper_id for p in result.papers]
        relevant = set(case["relevant_ids"])
        rows.append({
            "query": case["query"],
            "retrieved_ids": ids[:k],
            "relevant_ids": sorted(relevant),
            "precision_at_k": precision_at_k(ids, relevant, k),
            "recall_at_k": recall_at_k(ids, relevant, k),
            "mrr": mean_reciprocal_rank(ids, relevant),
            "failed_api_calls": result.failed_calls,
            "api_calls": result.api_calls,
        })
    count = len(rows) or 1
    return {
        "dataset": dataset["dataset_name"],
        "cases": rows,
        "mean_precision_at_k": sum(r["precision_at_k"] for r in rows) / count,
        "mean_recall_at_k": sum(r["recall_at_k"] for r in rows) / count,
        "mean_mrr": sum(r["mrr"] for r in rows) / count,
        "latency_seconds": elapsed,
        "evaluation_kind": "live retrieval against public academic APIs and manually curated relevance IDs",
    }


def evaluate_generation(report: LiteratureReport, papers: list[Paper], analyses: list[PaperAnalysis]) -> dict:
    cmap = citation_map(papers)
    reverse = {marker: paper_id for paper_id, marker in cmap.items()}
    citations = re.findall(r"\[(\d+)\]", report.markdown)
    citation_ids = [reverse.get(f"[{number}]", "invalid") for number in citations]
    known = {p.paper_id for p in papers}
    quote_by_paper = {a.paper_id: {e.quote for e in a.evidence} for a in analyses}
    claim_rows = []
    for line in report.markdown.splitlines():
        match = re.match(r"^- [^:]+: (.*?)(?: (\[\d+\]))?$", line)
        if not match:
            continue
        claim = match.group(1).strip()
        marker = match.group(2)
        paper_id = reverse.get(marker) if marker else None
        supported = bool(paper_id and claim in quote_by_paper.get(paper_id, set()))
        claim_rows.append((claim, supported))
    claims = [claim for claim, _ in claim_rows]
    supported = {claim for claim, ok in claim_rows if ok}
    return {
        "citation_validity": citation_validity(citation_ids, known),
        "evidence_coverage": evidence_coverage(claims, supported),
        "groundedness_exact_quote_estimate": evidence_coverage(claims, supported),
        "unsupported_claim_count": unsupported_claim_count(claims, supported),
        "audited_bullet_claim_count": len(claim_rows),
        "metric_note": "Exact quote matching is a mechanical estimate, not semantic entailment.",
    }

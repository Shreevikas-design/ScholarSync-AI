from __future__ import annotations

import json
import re
import time
from pathlib import Path

from agents.planner import heuristic_plan
from evaluation.metrics import (
    citation_validity, evidence_coverage, unsupported_claim_count,
)
from schemas.models import LiteratureReport, Paper, PaperAnalysis
from services.citation_service import citation_map
from services.metadata import norm_arxiv, norm_doi
from services.search_service import AcademicSearchService, SearchOptions


def _identity_aliases(value: str) -> set[str]:
    raw = (value or "").strip()
    if not raw:
        return set()
    aliases = {raw.casefold()}
    doi = norm_doi(raw)
    arxiv = norm_arxiv(raw)
    if doi:
        aliases.update({doi.casefold(), f"doi:{doi}".casefold()})
    if arxiv:
        aliases.update({arxiv.casefold(), f"arxiv:{arxiv}".casefold()})
    return aliases


def _paper_aliases(paper: Paper) -> set[str]:
    aliases = set()
    for value in [paper.paper_id, paper.doi or "", *paper.source_ids, paper.url or ""]:
        aliases.update(_identity_aliases(value))
    return aliases


def evaluate(dataset_path: Path, search_service: AcademicSearchService, k: int = 5) -> dict:
    if k < 1:
        raise ValueError("k must be positive")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows = []
    elapsed = 0.0
    for case in dataset["cases"]:
        plan = heuristic_plan(case["query"], None, None)
        started = time.perf_counter()
        result = search_service.search(plan, SearchOptions(max_papers=max(k, 10)))
        elapsed += time.perf_counter() - started
        relevant_ids = list(dict.fromkeys(case.get("relevant_ids", [])))
        relevant_groups = [_identity_aliases(identifier) for identifier in relevant_ids]
        relevant = set().union(*relevant_groups) if relevant_groups else set()
        retrieved = result.papers[:k]
        retrieved_aliases = [_paper_aliases(p) for p in retrieved]
        hit_flags = [bool(aliases & relevant) for aliases in retrieved_aliases]
        matched_ids = [paper.paper_id for paper, hit in zip(retrieved, hit_flags) if hit]
        matched_relevant = [identifier for identifier, group in zip(relevant_ids, relevant_groups)
                            if any(aliases & group for aliases in retrieved_aliases)]
        matched_rank = next((index for index, aliases in enumerate(retrieved_aliases, start=1) if aliases & relevant), None)
        if not relevant:
            status, explanation = "unscorable", "No ground-truth relevant identifiers are present for this query."
        elif result.failed_calls and not result.papers:
            status, explanation = "incomplete", "Search providers failed and returned no records; retrieval metrics are unavailable."
        elif not result.papers:
            status, explanation = "incomplete", "No records were retrieved; this run cannot distinguish no relevant results from search failure."
        elif matched_rank is None:
            status = "partial" if result.failed_calls else "complete"
            explanation = "No ground-truth IDs matched. Check provider coverage and identifier aliases before interpreting this as a true zero." \
                if status == "complete" else "No IDs matched and at least one provider failed; metrics may be incomplete."
        else:
            status = "partial" if result.failed_calls else "complete"
            explanation = "At least one provider failed; metrics reflect only the results returned." if result.failed_calls else "All enabled provider calls completed."
        scoreable = bool(relevant and result.papers)
        rows.append({
            "query": case["query"],
            "status": status,
            "explanation": explanation,
            "retrieved_ids": [p.paper_id for p in retrieved],
            "retrieved_titles": [p.title for p in retrieved],
            "relevant_ids": relevant_ids,
            "matched_ids": matched_ids,
            "matched_relevant_ids": matched_relevant,
            "precision_at_k": sum(hit_flags) / k if scoreable else None,
            "recall_at_k": len(matched_relevant) / len(relevant_groups) if scoreable else None,
            "reciprocal_rank": 1.0 / matched_rank if scoreable and matched_rank else (0.0 if scoreable else None),
            "mrr": 1.0 / matched_rank if scoreable and matched_rank else (0.0 if scoreable else None),
            "failed_api_calls": result.failed_calls,
            "api_calls": result.api_calls,
            "provider_status": result.provider_status,
        })
    scoreable_rows = [r for r in rows if r["precision_at_k"] is not None]
    count = len(scoreable_rows)
    return {
        "dataset": dataset["dataset_name"],
        "k": k,
        "cases": rows,
        "mean_precision_at_k": sum(r["precision_at_k"] for r in scoreable_rows) / count if count else None,
        "mean_recall_at_k": sum(r["recall_at_k"] for r in scoreable_rows) / count if count else None,
        "mean_mrr": sum(r["mrr"] for r in scoreable_rows) / count if count else None,
        "scorable_cases": count,
        "unscorable_cases": len(rows) - count,
        "latency_seconds": elapsed,
        "llm_calls": 0,
        "token_usage": None,
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

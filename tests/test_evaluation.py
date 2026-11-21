from __future__ import annotations

import json

from evaluation.evaluator import evaluate
from services.search_service import DiscoveryResult


class FakeSearch:
    def __init__(self, result):
        self.result = result

    def search(self, plan, options):
        return self.result


def _dataset(tmp_path, relevant_ids):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({
        "dataset_name": "fixture",
        "cases": [{"query": "wifi access point optimization", "relevant_ids": relevant_ids}],
    }), encoding="utf-8")
    return path


def test_retrieval_evaluation_matches_arxiv_alias_on_doi_canonical_paper(tmp_path, real_paper):
    canonical = real_paper.model_copy(update={
        "paper_id": "doi:10.1234/example",
        "doi": "https://doi.org/10.1234/Example.",
        "source_ids": ["arXiv:1101.5027v2", "OpenAlex:W123"],
    })
    result = DiscoveryResult(papers=[canonical], api_calls=3, provider_status={"OpenAlex": {"calls": 1}})
    report = evaluate(_dataset(tmp_path, ["arxiv:1101.5027"]), FakeSearch(result), k=1)
    case = report["cases"][0]
    assert case["status"] == "complete"
    assert case["matched_ids"] == ["doi:10.1234/example"]
    assert case["matched_relevant_ids"] == ["arxiv:1101.5027"]
    assert report["mean_precision_at_k"] == 1.0
    assert report["mean_recall_at_k"] == 1.0
    assert report["mean_mrr"] == 1.0


def test_failed_or_empty_retrieval_is_explained_not_scored_as_zero(tmp_path):
    report = evaluate(
        _dataset(tmp_path, ["arxiv:1101.5027"]),
        FakeSearch(DiscoveryResult(papers=[], failed_calls=2, errors=["OpenAlex unavailable"])),
        k=5,
    )
    assert report["cases"][0]["status"] == "incomplete"
    assert report["cases"][0]["precision_at_k"] is None
    assert report["mean_precision_at_k"] is None
    assert "failed" in report["cases"][0]["explanation"]


def test_no_ground_truth_is_unscorable(tmp_path, real_paper):
    report = evaluate(
        _dataset(tmp_path, []),
        FakeSearch(DiscoveryResult(papers=[real_paper])),
        k=5,
    )
    assert report["cases"][0]["status"] == "unscorable"
    assert report["mean_recall_at_k"] is None

from __future__ import annotations

from collections.abc import Sequence


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(item in relevant for item in list(retrieved[:k])) / k


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    return sum(item in relevant for item in retrieved[:k]) / len(relevant)


def mean_reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    return next((1.0 / i for i, item in enumerate(retrieved, 1) if item in relevant), 0.0)


def citation_validity(cited_ids: Sequence[str], known_ids: set[str]) -> float:
    return sum(item in known_ids for item in cited_ids) / len(cited_ids) if cited_ids else 1.0


def evidence_coverage(claims: Sequence[str], supported_claims: set[str]) -> float:
    return sum(claim in supported_claims for claim in claims) / len(claims) if claims else 1.0


def groundedness(claims: Sequence[str], supported_claims: set[str]) -> float:
    return evidence_coverage(claims, supported_claims)


def unsupported_claim_count(claims: Sequence[str], supported_claims: set[str]) -> int:
    return sum(claim not in supported_claims for claim in claims)

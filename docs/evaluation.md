# Evaluation

## Retrieval

evaluation/dataset.json contains two query-specific positive relevance judgments, each pointing to the corresponding arXiv abstract page. The evaluator searches live public APIs using the topic planner and computes Precision@K, Recall@K, and MRR. API failures, call counts, and elapsed search time are reported.

Judgments are manually seeded and intentionally small. They should be expanded by domain reviewers before making portfolio claims about general retrieval quality. Metrics are computed at runtime and are not hardcoded or represented as results from a prior run.

## Generation and evidence

The metrics module includes citation validity, evidence coverage, groundedness, and unsupported-claim count. Exact quote matching is auditable and repeatable, but it is stricter than semantic entailment and can miss a valid paraphrase. Groundedness from exact-string checks is a mechanical estimate, not a human or LLM-as-judge score.

## System measurements

Workflow results include selected API call/failure counts and LLM calls. Ollama and Groq usage values are included only when the provider returns token usage. Input token estimates use a conservative character heuristic and are not tokenizer-exact. Agent timestamps are recorded in UTC. Latency for individual API calls is bounded by configured timeouts; the evaluation script measures total live search duration.

## Running

       D:\ScholarSync_AI\.venv\Scripts\python.exe -m evaluation.run_evaluation --k 5

The run requires network access to selected public academic APIs. Unit tests use mocked responses and do not use the internet, API keys, or model downloads.

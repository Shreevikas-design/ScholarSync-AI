# Limitations and safeguards

- OpenAlex, Crossref, and arXiv do not cover all publications; metadata and abstracts can be incomplete.
- Full text is downloaded only from a PDF URL supplied by a source and validated as publicly reachable. No paywall is bypassed.
- PDF extraction can miss scanned pages, figures, tables, equations, and complex layouts. Empty pages are skipped and page numbers are retained for extracted text.
- Abstract-only analyses explicitly say so. Missing fields are reported as not available in accessible text.
- Template analysis is transparent keyword/sentence extraction, not deep scientific interpretation. Selecting Ollama or Groq adds model-assisted extraction, but exact quote validation can reject useful paraphrases.
- Comparison does not rank numerical outcomes across incompatible datasets, units, metrics, or experimental conditions.
- Gap detection is deliberately conservative. It extracts author-reported limitations and only reports cross-paper patterns when multiple records provide matching limitation evidence. It cannot establish field-wide novelty.
- Template chat returns evidence excerpts rather than generated synthesis. This favors traceability but can be less conversational.
- Chroma semantic retrieval depends on the configured embedding service. If it is unavailable, keyword retrieval remains available and the indexing/retrieval warning is exposed.
- The token counter is an estimate. Requests that exceed the configured context budget are rejected before transmission. Provider-specific token overflow responses, including 500 responses that mention context limits, are reported without retrying that request.
- LLM-provider output can still be wrong. Structured validation and quote checks lower risk but do not replace human review.
- The evaluation seed has only two manually judged queries and is not statistically representative.
- No live LLM test is claimed. The default run mode is model-free template mode.

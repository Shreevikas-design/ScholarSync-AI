# Architecture

## Component boundaries

- app.py: Streamlit workspace, explicit model/source controls, workflow status, comparison, paper chat, and exports.
- agents/: topic planner, per-paper analyzer, comparative analyst, limitation/gap analyzer, report writer, and evidence critic.
- graph/: typed state and LangGraph orchestration with conditional revision routing.
- services/: academic API clients, LLM provider adapter, PDF extraction, embedding/vector indexing, hybrid retrieval, citation checks, and exports.
- schemas/: validated Pydantic input/output structures.
- evaluation/: retrieval and grounded-generation metrics and a manually judged seed dataset.
- data/: runtime search cache, downloaded public PDFs, lazy embedding-model cache, and generated reports. It is excluded from Git. Vector records are held in isolated in-memory Chroma collections.

## State

The graph carries the original topic and year/source limits, validated plan, normalized paper records, analysis records, comparison, research gaps, draft report, critic decision, revision count, API error counts, citation audit, usage metrics, page-aware chunks, and workflow event log. Streamlit owns one uniquely named Chroma collection per active session/run; the same vector service is injected into analysis and chat. Collections are replaced after a successful new run and are not shared across sessions. Uploaded PDFs and vectors remain in session memory.

## Workflow control

The graph is a compiled LangGraph StateGraph. Search and paper analysis are bounded. Per-paper work is parallelized with a four-worker maximum. The critic routes back to the writer only while fewer than two revisions have been used. Citation validation always runs at the end. Search errors do not invent substitute papers; a zero-result report states that no records were returned.

## Evidence and references

Each extracted analysis finding keeps a quote and its source kind; full-text quotes keep a page number. Full-text, abstract-only, and metadata-only analysis are explicitly labelled. Report markers are assigned from the collected Paper objects and bibliography is generated from that same list. Chat emits retrieved excerpts and derives page labels from chunk metadata. Hybrid retrieval uses reciprocal-rank fusion; keyword fallback is surfaced as its own mode.

## Provider behavior

template is the default and is not an LLM. ollama and groq are explicit settings. The central adapter preflights estimated input tokens, reserves output tokens, records provider usage when supplied, caps each generation operation at two attempts, and classifies context/token errors separately, including provider HTTP 500 responses whose body indicates token overflow. An unreachable Ollama endpoint is circuit-broken for the remainder of that workflow. No provider is silently substituted. Chroma's bundled ONNX embedding function is the cloud-safe default and initializes lazily on first indexing.

OpenAlex requests use process-wide host pacing, bounded exponential retries, Retry-After support, a result cache, optional API-key configuration, and provider-level failure reporting. Each provider can fail without discarding results from the others.

## Sources

- https://reference.langchain.com/python/langgraph/graph/state/StateGraph
- https://docs.openalex.org/
- https://info.arxiv.org/help/api/user-manual.html
- https://www.crossref.org/documentation/retrieve-metadata/rest-api/

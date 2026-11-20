# Architecture

## Component boundaries

- app.py: Streamlit workspace, explicit model/source controls, workflow status, comparison, paper chat, and exports.
- agents/: topic planner, per-paper analyzer, comparative analyst, limitation/gap analyzer, report writer, and evidence critic.
- graph/: typed state and LangGraph orchestration with conditional revision routing.
- services/: academic API clients, LLM provider adapter, PDF extraction, embedding/vector indexing, hybrid retrieval, citation checks, and exports.
- schemas/: validated Pydantic input/output structures.
- evaluation/: retrieval and grounded-generation metrics and a manually judged seed dataset.
- data/: runtime cache, downloaded public PDFs, ChromaDB collection, and generated reports. It is excluded from Git.

## State

The graph carries the original topic and year/source limits, validated plan, normalized paper records, analysis records, comparison, research gaps, draft report, critic decision, revision count, API error counts, citation audit, usage metrics, page-aware chunks, and workflow event log.

## Workflow control

The graph is a compiled LangGraph StateGraph. Search and paper analysis are bounded. Per-paper work is parallelized with a four-worker maximum. The critic routes back to the writer only while fewer than two revisions have been used. Citation validation always runs at the end. Search errors do not invent substitute papers; a zero-result report states that no records were returned.

## Evidence and references

Each extracted analysis finding keeps a quote and its source kind; full-text quotes keep a page number. Abstract-only analysis is explicitly labelled. Report markers are assigned from the collected Paper objects and bibliography is generated from that same list. Chat emits retrieved excerpts and derives page labels from chunk metadata.

## Provider behavior

template is the default and is not an LLM. ollama and groq are explicit settings. The adapter preflights estimated input tokens, reserves output tokens, records provider usage when supplied, bounds retries for transient service errors, and classifies context/token errors separately, including provider HTTP 500 responses whose body indicates token overflow. No provider is silently substituted.

## Sources

- https://reference.langchain.com/python/langgraph/graph/state/StateGraph
- https://docs.openalex.org/
- https://info.arxiv.org/help/api/user-manual.html
- https://www.crossref.org/documentation/retrieve-metadata/rest-api/

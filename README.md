# ScholarSync AI

A reproducible academic literature intelligence workspace built around real public scholarly APIs, a bounded LangGraph workflow, page-aware paper processing, and evidence-audited reports.

ScholarSync is not a general chatbot. Search results and citations come from OpenAlex, arXiv, and Crossref responses. If no LLM is configured, the explicit template mode still runs planning, discovery, extraction, comparison, gap review, and citation checks. It labels abstract-only work and does not invent sources.

## Features

- Multi-query search across OpenAlex, arXiv, and Crossref, with bounded retries and a local response cache.
- DOI, arXiv ID, and normalized-title deduplication with relevance ranking and publication-year filtering.
- Optional public PDF retrieval with URL checks, timeout, file-size and page limits, validation, and page-aware PyMuPDF extraction.
- Separate research planner, discovery, paper analysis, comparative analysis, gap analysis, writer, evidence critic, and citation validation stages.
- LangGraph conditional revision loop capped at two re-drafts.
- Abstract-only analysis labels and exact evidence excerpts.
- ChromaDB semantic retrieval with Ollama or optional Sentence Transformers embeddings, plus lexical retrieval when semantic indexing is unavailable.
- Paper chat returns retrieved source excerpts with paper titles and page numbers when those pages exist.
- Markdown, PDF, and BibTeX exports.
- Live retrieval evaluation against a small manually judged seed dataset.
- Workflow event log, API failure counts, LLM call/token usage where providers report them, and prompt-budget protection.

## Architecture

See docs/architecture.md for modules, state fields, boundaries, and the workflow diagram.

### Agent roles

| Agent | Responsibility |
|---|---|
| Supervisor | Validates request limits, executes the compiled graph, bounds retries and workflow revisions. |
| Research Planner | Produces a validated topic scope, questions, keywords, and search queries. |
| Paper Discovery | Queries enabled public indexes, normalizes records, deduplicates, filters years, and ranks. |
| Paper Analyzer | Extracts paper fields from accessible text; each extracted finding keeps a source quote. |
| Comparative Analyst | Aligns fields and warns when datasets, metrics, or conditions do not support numeric comparisons. |
| Research Gap Analyst | Separates author-reported limitations from cross-paper observations; does not infer gaps from missing metadata. |
| Literature Review Writer | Builds the required report sections from collected metadata and evidence. |
| Evidence Critic | Checks structure, evidence IDs, abstract-only labels, and citation validity; requests at most two revisions. |

Agents are Python modules and use Pydantic schemas. Template mode is deterministic and transparent. Ollama or Groq can be selected for planning and paper analysis. The literature review remains evidence-template based so provider prose cannot add uncited facts.

## Workflow

    Research request
      -> Research planner
      -> OpenAlex / arXiv / Crossref
      -> Normalize, deduplicate, filter, rank
      -> Public PDF retrieval
      -> Parallel paper analysis and page-aware chunks
      -> Comparison
      -> Evidence-supported gap analysis
      -> Report writer
      -> Evidence critic -- revise at most twice --> Report writer
      -> Citation validation
      -> Report and paper chat

The workflow is an actual LangGraph StateGraph with conditional critic routing. Agent event timestamps, statuses, summaries, errors, API call counts, and revision counts are stored in the run result.

## Tech stack

Python 3.11+, LangGraph, Streamlit, ChromaDB, PyMuPDF, Pydantic, HTTPX, ReportLab, pytest, OpenAlex API, arXiv export API, and Crossref REST API.

## Windows setup (D drive)

The project is designed for D:\ScholarSync_AI. Keep the virtual environment, application data, temporary files, pip cache, Python bytecode cache, and Streamlit config under this folder.

1. Install Python 3.11 or later.
2. Open PowerShell and run:

       Set-Location D:\ScholarSync_AI
       .\setup.ps1

3. Edit D:\ScholarSync_AI\.env only if you want custom provider settings.
4. Start the app:

       .\run.ps1

The default template mode does not need a model or API key. Search needs internet access to the selected academic APIs.

## Ollama setup

Ollama is optional. Install it, pull a chat model and an embedding model, then choose Ollama in the sidebar.

       ollama pull qwen2.5:7b
       ollama pull nomic-embed-text

To keep large model weights on D, set OLLAMA_MODELS to D:\ScholarSync_AI\models\ollama for the Ollama server process before starting that process. An already running Windows Ollama service may need to be restarted after changing this variable. ScholarSync does not download Ollama models automatically and never switches to another provider.

## Groq setup

Set SCHOLARSYNC_GROQ_API_KEY in the process environment or enter it in the password field after selecting Groq. The key is never persisted by the UI or written to logs. Groq is optional; no paid API is needed for template mode.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| SCHOLARSYNC_PROVIDER | template, ollama, or groq | template |
| SCHOLARSYNC_OLLAMA_BASE_URL | Local Ollama endpoint | http://127.0.0.1:11434 |
| SCHOLARSYNC_OLLAMA_MODEL | Ollama generation model | qwen2.5:7b |
| SCHOLARSYNC_OLLAMA_EMBED_MODEL | Ollama embedding model | nomic-embed-text |
| SCHOLARSYNC_GROQ_API_KEY | Optional Groq key | unset |
| SCHOLARSYNC_GROQ_MODEL | Groq generation model | openai/gpt-oss-20b |
| SCHOLARSYNC_MAX_CONTEXT_TOKENS | Prompt context ceiling; preflight checked | 8192 |
| SCHOLARSYNC_MAX_OUTPUT_TOKENS | Generation output cap | 1200 |
| SCHOLARSYNC_REQUEST_TIMEOUT | API timeout, seconds | 45 |
| SCHOLARSYNC_DATA_DIR | Data/cache/vector path | ./data |
| SCHOLARSYNC_CACHE_TTL_HOURS | Search cache TTL; 0 disables cache | 24 |
| SCHOLARSYNC_MAX_PDF_MB | PDF download limit | 25 |
| SCHOLARSYNC_MAX_PDF_PAGES | Extraction page limit | 200 |
| SCHOLARSYNC_MAX_RETRIES | Bounded retries for transient failures | 2 |

## Usage

Enter a research topic, publication window, paper limit, and sources in Research Workspace. Start the run and inspect actual agent states, errors, and outputs in Agent Workflow. Review metadata and access labels in Papers, select records in Comparison, inspect gaps, read and download the review, and ask paper-specific questions in Research Chat.

Run the live relevance evaluation from the Evaluation tab or:

       D:\ScholarSync_AI\.venv\Scripts\python.exe -m evaluation.run_evaluation --k 5

The evaluation calls public APIs and writes no canned scores. Results vary with API availability and ranking changes.

## Screenshots

Screenshots can be added here after running the application on the target machine. No screenshots are included in this source delivery.

## Evaluation methodology

See docs/evaluation.md. Retrieval metrics compare live API results with two manually selected positive arXiv IDs and their source pages. This is a small seed benchmark, not a representative benchmark. No scores are claimed in advance. Generation checks can measure citation validity and exact evidence coverage on a completed report; they do not prove semantic entailment.

## Limitations

See docs/limitations.md. Public API indexing is incomplete; many papers are paywalled; abstract extraction is not full-paper review; token estimates are approximate; embeddings require an explicitly selected backend; and automated gap detection is conservative.

## Documentation

- docs/architecture.md
- docs/evaluation.md
- docs/limitations.md

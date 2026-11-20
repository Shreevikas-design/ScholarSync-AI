# Run verification

Checks completed on the target Windows machine. Runtime data, the virtual environment, caches, and the application are located under `D:\ScholarSync_AI`.

- Runtime: Python 3.11.0; available memory observed: approximately 31.8 GB.
- Disk: approximately 157 GB free on D: during initial inspection.
- Dependency installation: completed in `D:\ScholarSync_AI\.venv`; `pip check` reported no broken requirements.
- Compilation: `python -m compileall -q app.py config.py agents graph services schemas evaluation tests` completed successfully.
- Offline tests: 33 passed. Tests do not require API keys, external network access, or model downloads. Chroma indexing and metadata round-trip are tested with a local fixture and fake embeddings.
- Imports: LangGraph, Streamlit, ChromaDB, PyMuPDF, Pydantic, HTTPX, ReportLab, workflow construction, and academic search service imported successfully.
- UI checks: Streamlit AppTest had zero application exceptions; a local Streamlit server returned HTTP 200 for its health endpoint and main page.
- Setup: `setup.ps1` completed and created the virtual environment and `.env` on D:. `run.ps1` syntax was parsed; the equivalent Streamlit launch was smoke-tested.
- Ollama: executable version 0.40.16. Installed local models were `qwen2.5:7b`, `phi3:mini`, and `nomic-embed-text:latest`. A live structured-output request using `qwen2.5:7b` succeeded and reported 121 tokens; a live `nomic-embed-text` request returned a 768-dimension vector.
- RAG: a live Ollama embedding plus Chroma add/query completed for a disposable test chunk; the retrieved result retained page 7 and its source URL. The test collection used a temporary D: path and was removed after verification.
- Academic API check: real OpenAlex, arXiv, and Crossref requests returned records. A live end-to-end template workflow retrieved five papers, analyzed five, generated four evidence-linked gap candidates, and produced a cited report. One Crossref request received HTTP 429 and was bounded and surfaced as a failed API call.
- Evaluation script: `python -m evaluation.run_evaluation --k 5` completed live against the two-case seed set: mean Precision@5 0.10, Recall@5 0.50, and MRR 0.50 over 13 and 15 API requests; measured search latency was about 9.4 seconds. One relevant item was not retrieved for the second query, so this small result highlights a real ranking/retrieval weakness rather than a passing benchmark claim.
- Workflow: live run completed with 16 status events, no failed agent events, citation validation errors, or LLM calls in template mode. The evidence critic accepted the report without a revision.
- Token overflow: unit tests verify local context-budget rejection and classification of a mocked provider HTTP 500 response containing a “too many tokens” error. This confirms error handling but is not a live Groq failure test.
- Secrets: `.env` is ignored by Git. No provider key was configured or exposed during verification.
- Groq: no live Groq request was made because no API key was configured.

External API availability and live search rankings can vary. The checks above are observations from this run, not a guarantee that third-party APIs remain available.

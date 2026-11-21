from __future__ import annotations

import json
import logging
import os
import sys
import time
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for folder in (".tmp", ".pycache", ".streamlit", "models\\huggingface", "models\\ollama"):
    (ROOT / folder).mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(ROOT / ".tmp")
os.environ["TMP"] = str(ROOT / ".tmp")
os.environ["PYTHONPYCACHEPREFIX"] = str(ROOT / ".pycache")
os.environ.setdefault("STREAMLIT_CONFIG_DIR", str(ROOT / ".streamlit"))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(ROOT / "models" / "huggingface" / "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(ROOT / "models" / "sentence-transformers"))
os.environ.setdefault("OLLAMA_MODELS", str(ROOT / "models" / "ollama"))
sys.path.insert(0, str(ROOT))

import streamlit as st

from config import Settings
from evaluation.evaluator import evaluate, evaluate_generation
from agents.supervisor import ResearchRequest, Supervisor
from schemas.models import WorkflowEvent
from services.chat_service import answer_from_papers
from schemas.models import Paper
from services.report_service import export_pdf, safe_filename
from services.pdf_service import extract_pdf_bytes
from services.retrieval_service import chunk_pages
from services.llm_service import LLMService
from services.search_service import AcademicSearchService
from services.vector_service import VectorService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("scholarsync.ui")

st.set_page_config(page_title="ScholarSync AI", page_icon="S", layout="wide")
st.title("ScholarSync AI")
st.caption("Multi-agent academic research and literature intelligence")
st.markdown(
    """
    <style>
    .block-container {max-width: 1500px; padding-top: 1.4rem;}
    div[data-testid="stMetric"] {border: 1px solid color-mix(in srgb, var(--text-color) 18%, transparent);
      border-radius: 12px; padding: 12px; background: color-mix(in srgb, var(--secondary-background-color) 75%, transparent);}
    [data-testid="stSidebar"] {border-right: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);}
    [data-testid="stSidebar"] > div:first-child {padding-top: 1.15rem;}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {line-height: 1.35;}
    .ss-brand {display:flex; align-items:center; gap:.65rem; padding:.35rem .3rem 1rem;
      margin-bottom:.35rem; border-bottom:1px solid color-mix(in srgb,var(--text-color) 9%,transparent);}
    .ss-brand-mark {display:grid; place-items:center; width:2rem; height:2rem; flex:0 0 auto; color:var(--text-color);}
    .ss-brand-mark svg {width:1.8rem; height:1.8rem; fill:none; stroke:currentColor; stroke-width:1.55;
      stroke-linecap:round; stroke-linejoin:round; opacity:.8;}
    .ss-brand-name {font-size:1.08rem; font-weight:650; letter-spacing:-.02em; line-height:1.1;}
    .ss-brand-note {font-size:.73rem; font-weight:450; letter-spacing:.01em; opacity:.62; margin-top:.22rem;}
    .ss-nav-section {font-size:.64rem; font-weight:750; letter-spacing:.13em; opacity:.56;
      padding:.85rem .55rem .28rem;}
    [data-testid="stSidebar"] div.stButton {margin:.12rem 0;}
    [data-testid="stSidebar"] div.stButton > button {justify-content:flex-start; text-align:left;
      border-radius:.68rem; min-height:2.55rem; padding:.45rem .75rem; font-weight:550;
      transition:background .15s ease,border-color .15s ease,transform .15s ease;}
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {border-color:transparent; background:transparent;}
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
      border-color:color-mix(in srgb,#5268df 24%,transparent);
      background:color-mix(in srgb,#5268df 7%,var(--secondary-background-color));}
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
      color:var(--text-color);
      border-color:color-mix(in srgb,#5268df 35%,transparent);
      background:color-mix(in srgb,#5268df 12%,var(--secondary-background-color));
      box-shadow:inset 3px 0 0 #5268df;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "research" not in st.session_state:
    st.session_state.research = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
st.session_state.setdefault("uploaded_papers", [])
st.session_state.setdefault("uploaded_chunks", [])
st.session_state.setdefault("vector_service", None)
st.session_state.setdefault("vector_signature", None)
st.session_state.setdefault("vector_unavailable", None)
st.session_state.setdefault("vector_unavailable_signature", None)
st.session_state.setdefault("session_key", uuid.uuid4().hex[:16])
st.session_state.setdefault("chat_llm", None)
st.session_state.setdefault("chat_llm_signature", None)

pages = [
    "Research Workspace", "Agent Workflow", "Papers", "Comparison",
    "Research Gaps", "Literature Review", "Research Chat", "Evaluation", "Settings",
]
page_icons = {
    "Research Workspace": "⌂", "Agent Workflow": "◷", "Papers": "▤", "Comparison": "⇄",
    "Research Gaps": "⌖", "Literature Review": "≡", "Research Chat": "◉",
    "Evaluation": "▥", "Settings": "⚙",
}


def _configured_secret(name: str) -> str:
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, "")
if "active_page" not in st.session_state or st.session_state.active_page not in pages:
    st.session_state.active_page = "Research Workspace"

def navigate_to(page: str) -> None:
    st.session_state.active_page = page

with st.sidebar:
    st.markdown(
        '<div class="ss-brand"><div class="ss-brand-mark" aria-hidden="true">'
        '<svg viewBox="0 0 32 32"><path d="M4.75 7.25c4.1-1.7 7.8-.9 11.25 1.1 3.45-2 7.15-2.8 11.25-1.1v17.5c-4.1-1.7-7.8-.9-11.25 1.1-3.45-2-7.15-2.8-11.25-1.1V7.25Z"/>'
        '<path d="M16 8.4v17.45M8 11.3c2.5-.6 4.6-.2 6.1.5M8 15c2.5-.6 4.6-.2 6.1.5M18 11.8c1.5-.7 3.6-1.1 6.1-.5M18 15.5c1.5-.7 3.6-1.1 6.1-.5"/></svg></div>'
        '<div><div class="ss-brand-name">ScholarSync</div>'
        '<div class="ss-brand-note">Literature workspace</div></div></div>',
        unsafe_allow_html=True,
    )
    sections = [
        ("RESEARCH", ["Research Workspace", "Agent Workflow"]),
        ("LIBRARY", ["Papers", "Comparison"]),
        ("SYNTHESIS", ["Research Gaps", "Literature Review", "Research Chat"]),
        ("QUALITY", ["Evaluation"]),
    ]
    for section, section_pages in sections:
        st.markdown(f'<div class="ss-nav-section">{section}</div>', unsafe_allow_html=True)
        for page in section_pages:
            st.button(
                f"{page_icons[page]}    {page}",
                key=f"nav_{page.lower().replace(' ', '_')}",
                type="primary" if st.session_state.active_page == page else "secondary",
                use_container_width=True,
                on_click=navigate_to,
                args=(page,),
            )
    st.divider()
    st.button(
        f"{page_icons['Settings']}    Settings",
        key="nav_settings",
        type="primary" if st.session_state.active_page == "Settings" else "secondary",
        use_container_width=True,
        on_click=navigate_to,
        args=("Settings",),
    )
    active_page = st.session_state.active_page

setting_defaults = {
    "settings_provider": os.getenv("SCHOLARSYNC_PROVIDER", "template"),
    "settings_ollama_url": os.getenv("SCHOLARSYNC_OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    "settings_ollama_model": os.getenv("SCHOLARSYNC_OLLAMA_MODEL", "qwen2.5:7b"),
    "settings_groq_model": os.getenv("SCHOLARSYNC_GROQ_MODEL", "openai/gpt-oss-20b"),
    "settings_groq_key": _configured_secret("SCHOLARSYNC_GROQ_API_KEY"),
    "settings_openalex_key": _configured_secret("SCHOLARSYNC_OPENALEX_API_KEY"),
    "settings_context_tokens": int(os.getenv("SCHOLARSYNC_MAX_CONTEXT_TOKENS", "8192")),
    "settings_output_tokens": int(os.getenv("SCHOLARSYNC_MAX_OUTPUT_TOKENS", "1200")),
    "settings_embedding_provider": os.getenv("SCHOLARSYNC_EMBEDDING_PROVIDER", "chroma-default"),
    "settings_embed_model": os.getenv("SCHOLARSYNC_OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    "settings_request_timeout": int(float(os.getenv("SCHOLARSYNC_REQUEST_TIMEOUT", "20"))),
    "settings_max_pdf_mb": int(os.getenv("SCHOLARSYNC_MAX_PDF_MB", "25")),
    "settings_max_pdf_pages": int(os.getenv("SCHOLARSYNC_MAX_PDF_PAGES", "200")),
    "settings_max_retries": int(os.getenv("SCHOLARSYNC_MAX_RETRIES", "2")),
}
settings_state = st.session_state.setdefault("app_settings", {})
for key, value in setting_defaults.items():
    # Migrate settings from an earlier UI run while keeping secrets in session memory only.
    settings_state.setdefault(key, st.session_state.get(key, value))
if settings_state["settings_provider"] not in ("template", "ollama", "groq"):
    settings_state["settings_provider"] = "template"
if settings_state["settings_embedding_provider"] not in ("chroma-default", "ollama", "sentence-transformers"):
    settings_state["settings_embedding_provider"] = "chroma-default"
initial_output_ceiling = max(128, int(settings_state["settings_context_tokens"]) - 128)
settings_state["settings_output_tokens"] = min(
    max(128, int(settings_state["settings_output_tokens"])), initial_output_ceiling,
)

if active_page == "Settings":
    for key in setting_defaults:
        st.session_state.setdefault(f"ui_{key}", settings_state[key])
    st.header("Settings")
    st.caption("Provider, model, retrieval, and resource settings for future research runs.")
    st.selectbox(
        "Generation provider", ["template", "ollama", "groq"], key="ui_settings_provider",
        help="Template mode runs without an LLM. Provider selection is explicit; failures are shown in workflow events.",
    )
    with st.expander("Generation models", expanded=True):
        st.text_input("Ollama URL", key="ui_settings_ollama_url")
        st.text_input("Ollama model", key="ui_settings_ollama_model")
        st.text_input("Groq model", key="ui_settings_groq_model")
        st.text_input(
            "Groq API key", key="ui_settings_groq_key", type="password",
            help="Kept only in this browser session; it is not written to project files or logs.",
        )
        st.text_input(
            "OpenAlex API key", key="ui_settings_openalex_key", type="password",
            help="Optional. Helps with OpenAlex rate limits. It stays in this browser session and is not logged.",
        )
    with st.expander("Context and retrieval", expanded=True):
        st.number_input("Context token limit", 1024, 32768, step=1024, key="ui_settings_context_tokens")
        output_ceiling = max(128, int(st.session_state.ui_settings_context_tokens) - 128)
        st.session_state.ui_settings_output_tokens = min(
            max(128, int(st.session_state.ui_settings_output_tokens)), output_ceiling,
        )
        st.number_input("Maximum output tokens", 128, output_ceiling, step=128, key="ui_settings_output_tokens")
        st.selectbox(
            "Embedding provider", ["chroma-default", "ollama", "sentence-transformers"], key="ui_settings_embedding_provider",
            help="Chroma's bundled ONNX embedding model is the cloud-safe default. Ollama requires a reachable endpoint; Sentence Transformers is optional and downloads a model.",
        )
        st.text_input("Ollama embedding model", key="ui_settings_embed_model")
    with st.expander("Request and PDF limits"):
        st.number_input("Request timeout (seconds)", 5, 180, key="ui_settings_request_timeout")
        st.number_input("Maximum PDF size (MB)", 1, 100, key="ui_settings_max_pdf_mb")
        st.number_input("Maximum PDF pages", 1, 1000, key="ui_settings_max_pdf_pages")
        st.number_input("Transient retries", 0, 5, key="ui_settings_max_retries")
    st.caption(f"Application data stays in: {ROOT / 'data'}")
    for key in setting_defaults:
        settings_state[key] = st.session_state[f"ui_{key}"]

settings = Settings(
    provider=settings_state["settings_provider"],
    ollama_base_url=settings_state["settings_ollama_url"],
    ollama_model=settings_state["settings_ollama_model"],
    groq_model=settings_state["settings_groq_model"],
    groq_api_key=settings_state["settings_groq_key"] or None,
    openalex_api_key=settings_state["settings_openalex_key"] or None,
    max_context_tokens=int(settings_state["settings_context_tokens"]),
    max_output_tokens=int(settings_state["settings_output_tokens"]),
    embedding_provider=settings_state["settings_embedding_provider"],
    ollama_embed_model=settings_state["settings_embed_model"],
    data_dir=ROOT / "data",
    request_timeout=float(settings_state["settings_request_timeout"]),
    max_pdf_mb=int(settings_state["settings_max_pdf_mb"]),
    max_pdf_pages=int(settings_state["settings_max_pdf_pages"]),
    max_retries=int(settings_state["settings_max_retries"]),
)
settings.ensure_data_dirs()

result = st.session_state.research


def _embedding_signature() -> tuple[str, str, str]:
    return (settings.embedding_provider, settings.ollama_base_url, settings.ollama_embed_model)


def _llm_for_session() -> LLMService:
    secret = settings.groq_api_key.get_secret_value() if settings.groq_api_key else ""
    signature = (settings.provider, settings.ollama_base_url, settings.ollama_model, settings.groq_model,
                 hashlib.sha256(secret.encode()).hexdigest() if secret else "", settings.max_context_tokens, settings.max_output_tokens)
    if st.session_state.chat_llm is None or st.session_state.chat_llm_signature != signature:
        st.session_state.chat_llm = LLMService(settings)
        st.session_state.chat_llm_signature = signature
    return st.session_state.chat_llm


def _vector_for_current_corpus(papers: list[Paper], chunks: list[dict], rebuild: bool = False):
    service = st.session_state.vector_service
    signature = _embedding_signature()
    unavailable = st.session_state.get("vector_unavailable")
    if not rebuild and unavailable and st.session_state.vector_unavailable_signature == signature:
        return None, unavailable
    if rebuild or service is None or st.session_state.vector_signature != signature:
        candidate = None
        try:
            candidate = VectorService(settings, collection_name=f"ss_{st.session_state.session_key}_{uuid.uuid4().hex[:12]}")
            candidate.add(chunks)
        except Exception as exc:
            message = f"Semantic indexing failed ({type(exc).__name__}); keyword retrieval remains available."
            st.session_state.vector_unavailable = message
            st.session_state.vector_unavailable_signature = signature
            if candidate:
                candidate.cleanup()
            return None, message
        if service:
            service.cleanup()
        service = candidate
        st.session_state.vector_service = service
        st.session_state.vector_signature = signature
        st.session_state.vector_unavailable = None
        st.session_state.vector_unavailable_signature = None
    return service, None

if active_page == "Research Workspace":
    st.subheader("Define a research run")
    topic = st.text_area(
        "Research topic",
        placeholder="Example: Machine learning-based Wi-Fi network optimization and access point placement",
        height=95,
        max_chars=500,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        start_year = st.number_input("Start year", min_value=1900, max_value=2100, value=2018, step=1)
    with c2:
        end_year = st.number_input("End year", min_value=1900, max_value=2100, value=2026, step=1)
    with c3:
        max_papers = st.slider("Maximum papers", min_value=5, max_value=40, value=15, step=5)
    source_cols = st.columns(3)
    with source_cols[0]:
        include_openalex = st.checkbox("OpenAlex", True)
    with source_cols[1]:
        include_arxiv = st.checkbox("arXiv", True)
    with source_cols[2]:
        include_crossref = st.checkbox("Crossref", True)
    st.caption("Research depth is controlled by the paper limit. The workflow searches up to six planned queries per enabled source.")
    progress_area = st.empty()
    if st.button("Start research", type="primary", use_container_width=True):
        clean_topic = " ".join(topic.split())
        if len(clean_topic) < 8:
            st.error("Enter a research topic with at least 8 characters.")
        elif start_year > end_year:
            st.error("Start year must be less than or equal to end year.")
        elif not any((include_openalex, include_arxiv, include_crossref)):
            st.error("Enable at least one academic search source.")
        else:
            observed: list[WorkflowEvent] = []

            def update_progress(item: WorkflowEvent):
                observed.append(item)
                rows = []
                latest = {}
                for ev in observed:
                    latest[ev.agent] = ev
                for name, ev in latest.items():
                    rows.append(f"{ev.status:<10} | {name:<28} | {ev.summary or ev.error or ''}")
                progress_area.code("\n".join(rows), language="text")

            progress_area.info("Waiting for the first LangGraph stage event…")
            with st.status("Running the LangGraph research workflow", expanded=True) as status:
                try:
                    started_run = time.perf_counter()
                    request = ResearchRequest(
                        topic=clean_topic, start_year=int(start_year), end_year=int(end_year),
                        max_papers=int(max_papers), include_openalex=include_openalex,
                        include_arxiv=include_arxiv, include_crossref=include_crossref,
                    )
                    run_vectors = VectorService(
                        settings,
                        collection_name=f"ss_{st.session_state.session_key}_{uuid.uuid4().hex[:12]}",
                    )
                    previous_vectors = st.session_state.vector_service
                    preindex_errors = []
                    try:
                        run_vectors.add(st.session_state.uploaded_chunks)
                    except Exception as exc:
                        preindex_errors.append(f"Uploaded-paper indexing unavailable: {type(exc).__name__}")
                    result = Supervisor(settings, vector_service=run_vectors).run(
                        request, on_event=update_progress, reuse_result=st.session_state.research,
                    )
                    result["latency_seconds"] = time.perf_counter() - started_run
                    result["ui_events"] = observed
                    result["run_id"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + hashlib.sha256(clean_topic.encode()).hexdigest()[:8]
                    if preindex_errors:
                        result.setdefault("rag_errors", []).extend(preindex_errors)
                    st.session_state.research = result
                    st.session_state.chat_history = []
                    st.session_state.chat_papers = []
                    st.session_state.vector_service = run_vectors
                    st.session_state.vector_signature = _embedding_signature()
                    st.session_state.vector_unavailable = None
                    st.session_state.vector_unavailable_signature = None
                    if previous_vectors:
                        previous_vectors.cleanup()
                    status.update(label="Research workflow finished", state="complete")
                    st.success(
                        f"Run {result['run_id']} | collected {len(result.get('papers', []))} papers; "
                        f"{len(result.get('analyses', []))} analyses; "
                        f"{len(result.get('gaps', []))} evidence-supported limitations/gaps."
                    )
                except Exception as exc:
                    if "run_vectors" in locals():
                        run_vectors.cleanup()
                    log.exception("Research workflow failed")
                    status.update(label="Research workflow failed", state="error")
                    st.error(f"{type(exc).__name__}: {exc}")

result = st.session_state.research

if active_page == "Agent Workflow":
    st.subheader("Observed agent workflow")
    if result:
        if result.get("run_id"):
            st.caption(f"Research run ID: {result['run_id']}")
        events = result.get("ui_events", result.get("events", []))
        if events:
            st.dataframe(
                [{
                    "Agent": e.agent,
                    "Status": e.status,
                    "Started (UTC)": e.started_at,
                    "Completed (UTC)": e.completed_at or "",
                    "Output": e.summary,
                    "Error": e.error or "",
                } for e in events],
                use_container_width=True,
                hide_index=True,
            )
        st.write("Revision passes:", result.get("revision_count", 0), "/ 2")
        critique = result.get("critique")
        if critique:
            st.write("Critic accepted:", critique.accepted)
            for issue in critique.issues:
                (st.error if issue.severity == "error" else st.warning if issue.severity == "warning" else st.info)(issue.message)
        if result.get("search_errors"):
            with st.expander("Search/API errors"):
                st.write("\n".join(result["search_errors"]))
        if result.get("provider_status"):
            with st.expander("Search provider status"):
                st.json(result["provider_status"])
        if result.get("rag_errors"):
            with st.expander("Embedding/indexing errors"):
                st.write("\n".join(result["rag_errors"]))
        if result.get("fatal_errors"):
            st.error("\n".join(result["fatal_errors"]))
        usage = result.get("llm_usage", {})
        st.metric("Workflow latency", f"{result.get('latency_seconds', 0):.2f} s")
        st.caption(
            f"LLM calls {usage.get('llm_calls', 0)} | provider-reported tokens "
            f"{usage.get('total_tokens', 0)} | estimated input tokens {usage.get('estimated_prompt_tokens', 0)} | "
            f"failed API calls {result.get('failed_api_calls', 0)} | token-budget rejections {usage.get('budget_rejections', 0)}"
        )
        errors = usage.get("errors", [])
        if errors:
            st.warning("Provider errors: " + "; ".join(errors))
    else:
        st.info("No run has completed. Start a research run in Research Workspace.")

if active_page == "Papers":
    st.subheader("Collected papers")
    st.caption("Upload a PDF to add page-aware evidence to Research Chat. Bibliographic details are taken from the PDF metadata or fields you provide.")
    upload = st.file_uploader("Upload an accessible PDF", type=["pdf"], key="paper_upload")
    if upload:
        default_title = Path(upload.name).stem
        upload_title = st.text_input("Paper title", value=default_title, key="upload_title")
        upload_authors = st.text_input("Authors (optional; comma separated)", key="upload_authors")
        upload_year = st.number_input("Publication year (optional; use 0 if unknown)", min_value=0, max_value=2100, value=0, key="upload_year")
        upload_doi = st.text_input("DOI (optional)", key="upload_doi")
        if st.button("Index uploaded PDF", key="upload_index", type="primary"):
            contents = upload.getvalue()
            digest = hashlib.sha256(contents).hexdigest()
            paper_id = f"upload:{digest}"
            if any(p.paper_id == paper_id for p in st.session_state.uploaded_papers):
                st.info("This PDF is already in the current session library.")
            else:
                pdf = extract_pdf_bytes(contents, settings)
                if not pdf.ok:
                    st.error(pdf.error or "PDF extraction failed.")
                else:
                    title = " ".join(upload_title.split()) or pdf.metadata.get("title") or default_title
                    paper = Paper(
                        paper_id=paper_id, title=title,
                        authors=[name.strip() for name in upload_authors.split(",") if name.strip()],
                        abstract="", year=int(upload_year) or None, doi=upload_doi.strip() or None,
                        url=None, pdf_url=None, source="User upload", source_ids=[paper_id],
                        full_text_status="partial-full-text" if pdf.truncated else "full-text",
                    )
                    pages = [{"page": p.page, "text": p.text, "location": "full-text"} for p in pdf.pages]
                    chunks = chunk_pages(paper, pages)
                    try:
                        existing_chunks = (result or {}).get("chunks", []) + st.session_state.uploaded_chunks
                        existing_papers = (result or {}).get("papers", []) + st.session_state.uploaded_papers
                        vector, vector_error = _vector_for_current_corpus(existing_papers + [paper], existing_chunks + chunks)
                        if vector:
                            vector.add(chunks)
                        elif vector_error:
                            st.info(vector_error)
                    except Exception as exc:
                        st.info(f"Semantic indexing failed ({type(exc).__name__}); keyword retrieval remains available.")
                    st.session_state.uploaded_papers.append(paper)
                    st.session_state.uploaded_chunks.extend(chunks)
                    if pdf.error:
                        st.warning(pdf.error)
                    st.success(f"Indexed {len(pdf.pages)} readable pages into {len(chunks)} evidence chunks for this session.")
    if result:
        papers = result.get("papers", [])
        uploaded_papers = st.session_state.uploaded_papers
        st.metric("Papers", len(papers) + len(uploaded_papers))
        for index, paper in enumerate(papers, start=1):
            with st.expander(f"{index}. {paper.title} ({paper.year or 'year not supplied'}) | relevance {paper.relevance_score:.3f}"):
                st.write("Authors:", ", ".join(paper.authors) or "Not supplied")
                st.write("Access:", paper.full_text_status, "| Preprint:", "Yes" if paper.is_preprint else "No / not marked as preprint")
                st.write("Abstract:", paper.abstract or "No abstract was supplied by the source.")
                st.write("DOI:", paper.doi or "Not supplied")
                st.write("Source:", paper.source, "| Source IDs:", ", ".join(paper.source_ids))
                if paper.url:
                    st.markdown(f"[Paper source]({paper.url})")
                if paper.pdf_url:
                    st.markdown(f"[Public PDF URL]({paper.pdf_url})")
                analysis = next((a for a in result.get("analyses", []) if a.paper_id == paper.paper_id), None)
                if analysis:
                    st.write("Analysis mode:", analysis.analysis_mode)
                    st.write("Sensors:", analysis.sensor_types)
                    st.write("Feature extraction:", analysis.feature_extraction)
                    st.json(analysis.model_dump(exclude={"evidence"}))
                    if analysis.evidence:
                        st.write("Evidence excerpts")
                        for item in analysis.evidence:
                            loc = f"page {item.page}" if item.page else "abstract"
                            st.markdown(f"> {item.quote}\n\n— {loc}")
                    if analysis.error:
                        st.warning(analysis.error)
        if uploaded_papers:
            st.markdown("#### User-uploaded papers")
            for paper in uploaded_papers:
                with st.expander(paper.title + " · uploaded " + paper.full_text_status):
                    st.write("Authors:", ", ".join(paper.authors) or "Not supplied")
                    st.write("Year:", paper.year or "Not supplied", "· DOI:", paper.doi or "Not supplied")
                    st.caption("User-uploaded metadata; indexed for this Streamlit session only.")
    else:
        if st.session_state.uploaded_papers:
            st.info("No search has run yet. Uploaded papers are available in Research Chat for this session.")
            for paper in st.session_state.uploaded_papers:
                st.write(f"• {paper.title} — full-text uploaded")
        else:
            st.info("Run a search or upload an accessible PDF to populate this tab.")
    export_papers = ((result or {}).get("papers", []) + st.session_state.uploaded_papers)
    if export_papers:
        st.download_button(
            "Export paper metadata (JSON)",
            data=json.dumps([paper.model_dump() for paper in export_papers], ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="scholarsync-paper-metadata.json", mime="application/json",
        )

if active_page == "Comparison":
    st.subheader("Method and result comparison")
    if result and result.get("papers"):
        papers = result["papers"]
        labels = {p.paper_id: f"{p.title} ({p.year or 'n.d.'})" for p in papers}
        selected = st.multiselect("Select papers", list(labels), default=list(labels), format_func=lambda item: labels[item])
        comparison = result.get("comparison")
        if comparison:
            rows = []
            for item in comparison.items:
                if item.paper_id not in selected:
                    continue
                rows.append({
                    "Paper": labels.get(item.paper_id, item.paper_id),
                    "Methodology": item.methodology,
                    "Algorithm/model": item.algorithm_model,
                    "Complexity": item.computational_complexity,
                    "Dataset": item.dataset,
                    "Sensors": item.sensor_types,
                    "Feature extraction": item.feature_extraction,
                    "Experimental setup": item.experimental_setup,
                    "Baselines": item.baselines,
                    "Metrics": item.metrics,
                    "Results": item.results,
                    "Strengths": item.strengths,
                    "Weaknesses": item.weaknesses,
                    "Numeric comparison caveat": item.comparability_note,
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
            for note in comparison.caveats:
                st.info(note)
    else:
        st.info("Run a search first.")

if active_page == "Research Gaps":
    st.subheader("Evidence-supported research limitations and gaps")
    if result:
        gaps = result.get("gaps", [])
        if not gaps:
            st.info("No evidence-supported gaps could be established from accessible text. Missing metadata is not treated as a gap.")
        for gap in gaps:
            with st.expander(f"{gap.gap_id}: {gap.title} | {gap.confidence} confidence"):
                st.write("Category:", gap.category)
                st.write(gap.description)
                st.write("Why it matters:", gap.why_it_matters)
                st.write("Proposed direction:", gap.proposed_direction)
                st.write("Confidence explanation:", gap.confidence_explanation)
                st.write("Evidence coverage:", f"{gap.evidence_coverage:.0%} of analyzed papers")
                st.write("Related paper IDs:", ", ".join(gap.related_paper_ids))
                for evidence in gap.supporting_evidence:
                    paper = next((p for p in result.get("papers", []) if p.paper_id == evidence.paper_id), None)
                    label = paper.title if paper else evidence.paper_id
                    location = f"page {evidence.page}" if evidence.page else "abstract"
                    st.markdown(f"> {evidence.quote}\n\n— {label}, {location}")
    else:
        st.info("Run a search first.")
    if result and result.get("gaps"):
        st.download_button(
            "Export research gaps (JSON)",
            data=json.dumps([gap.model_dump() for gap in result["gaps"]], ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="scholarsync-research-gaps.json", mime="application/json",
        )

if active_page == "Literature Review":
    st.subheader("Literature review")
    if result and result.get("report"):
        report = result["report"]
        st.markdown(report.markdown)
        st.download_button(
            "Download Markdown",
            data=report.markdown.encode("utf-8"),
            file_name=safe_filename(report.title, "md"),
            mime="text/markdown",
        )
        if st.button("Prepare PDF export"):
            target = settings.data_dir / "reports" / safe_filename(report.title, "pdf")
            export_pdf(report, target)
            st.session_state.pdf_bytes = target.read_bytes()
        if st.session_state.pdf_bytes:
            st.download_button(
                "Download PDF",
                data=st.session_state.pdf_bytes,
                file_name=safe_filename(report.title, "pdf"),
                mime="application/pdf",
            )
        cite = st.session_state.research.get("papers", [])
        from services.citation_service import bibtex_export
        st.download_button(
            "Download BibTeX",
            data=bibtex_export(cite).encode("utf-8"),
            file_name=safe_filename(report.title, "bib"),
            mime="application/x-bibtex",
        )
        citation_errors = result.get("citation_errors", [])
        if citation_errors:
            st.error("Citation validation failed: " + ", ".join(citation_errors))
    else:
        st.info("Run a search to draft the review.")

if active_page == "Research Chat":
    st.subheader("Ask the collected papers")
    papers = ((result or {}).get("papers", []) + st.session_state.uploaded_papers)
    chunks = ((result or {}).get("chunks", []) + st.session_state.uploaded_chunks)
    if papers:
        labels = {p.paper_id: f"{p.title} ({p.year or 'n.d.'})" for p in papers}
        if "chat_papers" in st.session_state:
            st.session_state.chat_papers = [pid for pid in st.session_state.chat_papers if pid in labels]
        selected_ids = st.multiselect("Limit search to selected papers", list(labels), format_func=lambda item: labels[item], key="chat_papers")
        question = st.text_input("Question", placeholder="Which datasets and limitations are reported?")
        if st.button("Retrieve evidence", key="chat_submit"):
            if len(question.strip()) < 4:
                st.warning("Enter a question.")
            else:
                vector, vector_error = _vector_for_current_corpus(papers, chunks)
                response = answer_from_papers(
                    question, papers, chunks, vector_service=vector,
                    selected_paper_ids=selected_ids,
                    llm_service=_llm_for_session(),
                )
                if vector_error:
                    response["warning"] = vector_error
                if response.get("warning"):
                    st.info(response["warning"])
                st.session_state.chat_history.append({"question": question, **response})
        for entry in reversed(st.session_state.chat_history):
            st.markdown(f"**Question:** {entry['question']}")
            st.caption(f"Retrieval: {entry.get('retrieval_mode', 'unknown')} · Answer: {entry.get('answer_mode', 'unknown')}")
            st.markdown(entry["answer"])
            st.divider()
    else:
        st.info("Run a research search or upload an accessible PDF before chatting.")

if active_page == "Evaluation":
    st.subheader("Retrieval evaluation")
    st.write("Uses the small manually curated relevance seed in evaluation/dataset.json and performs live searches. Reported scores are computed at run time, not pre-filled.")
    k = st.slider("K", 1, 10, 5)
    if st.button("Run live retrieval evaluation"):
        with st.spinner("Querying enabled academic sources..."):
            try:
                searcher = AcademicSearchService(settings)
                measured = evaluate(ROOT / "evaluation" / "dataset.json", searcher, k=k)
                st.metric("Scorable queries", measured["scorable_cases"])
                left, middle, right = st.columns(3)
                left.metric("Mean Precision@K", "Unavailable" if measured["mean_precision_at_k"] is None else f"{measured['mean_precision_at_k']:.3f}")
                middle.metric("Mean Recall@K", "Unavailable" if measured["mean_recall_at_k"] is None else f"{measured['mean_recall_at_k']:.3f}")
                right.metric("Mean Reciprocal Rank", "Unavailable" if measured["mean_mrr"] is None else f"{measured['mean_mrr']:.3f}")
                st.dataframe([
                    {key: row[key] for key in (
                        "query", "status", "retrieved_ids", "relevant_ids", "matched_ids",
                        "precision_at_k", "recall_at_k", "reciprocal_rank", "explanation",
                    )}
                    for row in measured["cases"]
                ], use_container_width=True, hide_index=True)
                with st.expander("Provider-level evaluation details"):
                    for row in measured["cases"]:
                        st.markdown(f"**{row['query']}** · {row['status']}")
                        st.json({"retrieved_titles": row["retrieved_titles"], "provider_status": row["provider_status"], "API calls": row["api_calls"], "failed API calls": row["failed_api_calls"]})
            except Exception as exc:
                st.error(f"Evaluation failed: {type(exc).__name__}: {exc}")
    if result:
        if result.get("report"):
            st.subheader("Generation evidence checks")
            st.json(evaluate_generation(result["report"], result.get("papers", []), result.get("analyses", [])))
        st.caption(
            f"This workflow run used {result.get('api_calls', 0)} academic API calls and "
            f"{result.get('failed_api_calls', 0)} failed calls. Per-agent start and completion timestamps appear in Agent Workflow."
        )

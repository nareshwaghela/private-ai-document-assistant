"""Private AI Document Assistant — article-faithful Streamlit front end."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from rag_local import (ask, build_index, collect_bytes, encoder, retrieve,
                       EMBEDDING_MODEL)

st.set_page_config(page_title="Private AI Document Assistant", page_icon="📚",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp {background-color: #091221; color:#e1f0ed}
.block-container {max-width: 1100px; padding-top:1.4rem}
.hero {background:linear-gradient(120deg,#133d36,#192744);border:1px solid #348b71;
       border-radius:20px;padding:25px 30px;margin-bottom:20px}
.hero h1 {font-size:2.2rem;color:white; margin:0.4rem 0}
.hero p {color:#c5e9df;margin:0}
.info {font-size:0.9rem;color:#bcd2d4}
</style>
""", unsafe_allow_html=True)
st.markdown("""<div class="hero"><strong>DECENTRALIZE AI HACKATHON • RAG + llama.cpp</strong>
<h1>Private AI Document Assistant</h1>
<p>CPU MiniLM embeddings + FAISS retrieval + optional Qwen2.5-7B inference on llama.cpp.</p></div>""",unsafe_allow_html=True)

# Never allow public visitors to choose arbitrary HTTP destinations (SSRF).
SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
SERVER_KEY = os.environ.get("LLAMA_SERVER_API_KEY", "") or None

@st.cache_resource(show_spinner=False)
def cpu_encoder():
    return encoder()


def server_available():
    import requests
    try:
        resp = requests.get(SERVER_URL + "/health", timeout=2,
                            headers=({"Authorization": f"Bearer {SERVER_KEY}"} if SERVER_KEY else {}))
        return resp.status_code == 200
    except requests.RequestException:
        return False

with st.sidebar:
    st.subheader("Model & environment")
    online = server_available()
    if online:
        st.success("llama.cpp model server connected")
        st.caption("Generation is handled by the connected llama.cpp server. Confirm the served model is Qwen2.5-7B-Instruct Q4_K_M.")
    else:
        st.info("llama.cpp server not connected")
        st.caption("Semantic FAISS retrieval works without an LLM. Model-generated answers need your separately hosted Qwen2.5/llama-server.")
    st.divider()
    st.write("Embedding model (CPU):")
    st.code(EMBEDDING_MODEL)
    top_k = st.slider("Top-k source passages", 1, 8, 4)
    max_tokens = st.slider("Maximum output tokens", 64, 512, 256, step=64)
    st.divider()
    if st.button("Clear indexed documents", use_container_width=True):
        for key in ("index", "records", "files", "result", "source_results"):
            st.session_state.pop(key, None)
        st.rerun()

st.warning("Privacy: this public Streamlit app processes uploads on its hosting server. Do not upload private or confidential documents. A truly private deployment requires running the app and model locally.")

upload_tab, question_tab, detail_tab = st.tabs(["📁 Ingest documents", "💬 Ask documents", "🔬 Architecture & limits"])
with upload_tab:
    st.markdown("### Upload PDF, TXT or Markdown files")
    uploads = st.file_uploader("Select up to 5 files (20 MB maximum each)",
                               type=["pdf", "txt", "md"], accept_multiple_files=True)
    col1, col2 = st.columns([2, 2])
    use_sample = col2.button("Use Project Cedar sample", use_container_width=True)
    process_upload = col1.button("Process uploaded files", type="primary", use_container_width=True)
    files = None
    if use_sample:
        sample = Path(__file__).with_name("sample_docs") / "project_cedar.txt"
        files = [(sample.name, sample.read_bytes())]
    elif process_upload:
        if not uploads:
            st.error("Please select at least one PDF, TXT or Markdown file.")
        elif len(uploads) > 5:
            st.error("Maximum 5 files per session.")
        else:
            files = [(uploaded.name, uploaded.getvalue()) for uploaded in uploads]
            if any(len(raw) > 20 * 1024 * 1024 for _, raw in files):
                st.error("Each file must be 20 MB or smaller.")
                files = None
    if files is not None:
        try:
            with st.spinner("Extracting documents and creating CPU semantic embeddings / FAISS index. First run downloads the MiniLM model..."):
                records = collect_bytes(files)
                index = build_index(records, cpu_encoder())
            st.session_state["index"] = index
            st.session_state["records"] = records
            st.session_state["files"] = [name for name, _ in files]
            st.session_state.pop("result", None)
            st.session_state.pop("source_results", None)
            st.success(f"Indexed {len(records)} excerpt(s) from {len(files)} file(s). Go to Ask documents.")
        except Exception as exc:
            st.error(f"Could not build semantic index: {exc}")
    if "index" in st.session_state:
        a, b = st.columns(2)
        a.metric("Indexed documents", len(st.session_state["files"]))
        b.metric("Indexed excerpts", len(st.session_state["records"]))
        st.caption("Files: " + ", ".join(st.session_state["files"]))

with question_tab:
    if "index" not in st.session_state:
        st.info("Index at least one document first. The Project Cedar sample is built in.")
    else:
        question = st.text_input("Ask a question using your document evidence",
                                 placeholder="Who owns the CSV import validator?")
        st.caption("Every retrieved excerpt includes a filename, page and excerpt ID. Generative answers require a running llama-server.")
        if st.button("Answer with RAG" if online else "Find semantic sources", type="primary"):
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                with st.spinner("Searching semantic embeddings" + (" and generating an answer..." if online else "...")):
                    try:
                        if online:
                            result = ask(question, cpu_encoder(), st.session_state["index"],
                                         st.session_state["records"], SERVER_URL,
                                         top_k=top_k, max_tokens=max_tokens, api_key=SERVER_KEY)
                            st.session_state["result"] = result
                            st.session_state["source_results"] = result["selected"]
                        else:
                            st.session_state["result"] = None
                            st.session_state["source_results"] = retrieve(
                                question, cpu_encoder(), st.session_state["index"],
                                st.session_state["records"], top_k)
                    except Exception as exc:
                        st.session_state["result"] = None
                        st.session_state["source_results"] = []
                        st.error(f"Question processing failed: {exc}")
        result = st.session_state.get("result")
        if result:
            st.markdown("### Model-generated answer")
            st.write(result["answer"])
            a, b, c = st.columns(3)
            a.metric("Retrieval", f"{result['retrieval_s']:.2f}s")
            b.metric("Generation", f"{result['generation_s']:.2f}s")
            c.metric("Prompt tokens", result["formatted_prompt_tokens"])
        elif st.session_state.get("source_results"):
            st.info("**Semantic retrieval only:** The LLM is not connected. The excerpts below are real source text, not a fabricated AI answer.")
        sources = st.session_state.get("source_results", [])
        if sources:
            st.markdown("### Retrieved source passages")
            for i, passage in enumerate(sources, 1):
                with st.expander(f"[S{i}] {passage['file']} · page {passage['page']} · cosine similarity {passage['score']:.3f}", expanded=i==1):
                    st.code(passage["id"],language="text")
                    st.write(passage["text"])

with detail_tab:
    st.markdown("""
### The article's actual architecture
1. **Ingest:** text-based PDF/TXT/Markdown, page-wise PyMuPDF extraction, 160-word chunks and 32-word overlap.
2. **Embed:** `sentence-transformers/all-MiniLM-L6-v2` on CPU, normalized vectors.
3. **Retrieve:** FAISS `IndexFlatIP`, normalized cosine similarity, top-k with source IDs.
4. **Generate (only if connected):** Qwen2.5-7B-Instruct Q4_K_M behind llama.cpp `llama-server`.
5. **Prompt budget:** `POST /apply-template` and `POST /tokenize` ensure selected passages fit the configured 4,096-token context.
6. **Evaluate:** `benchmark.py` records raw answers, citations, token counts and real wall-clock latencies.

### Limits
- Free Streamlit Community Cloud does not offer this 8GB GPU; GPU inference must run on compatible external hardware.
- Scanned PDFs need OCR (not included in the article).
- Citations and model answers need human verification; retrieval similarity is not truth probability.
- This demo does not provide user authentication or sensitive-document isolation.
- The model server should not be exposed without authentication/TLS; for private use, bind to localhost.
""")
st.divider()
st.caption("Faithful article reference implementation • No fabricated 8GB GPU benchmarks • Source-grounded retrieval")

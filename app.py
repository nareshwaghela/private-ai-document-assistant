"""Public Streamlit demo: source-grounded PDF retrieval (not an LLM)."""
import re
from collections import Counter
from io import BytesIO

import fitz
import streamlit as st

st.set_page_config(page_title="Private AI Document Assistant", page_icon="📚", layout="wide")
st.markdown("""
<style>
.stApp {background:#0a1220;color:#e8f5f1}
.block-container{max-width:1100px;padding-top:2rem}
[data-testid="stSidebar"]{background:#101e2d}
.hero{background:linear-gradient(115deg,#123e34,#192b48);border:1px solid #337a62;border-radius:20px;padding:28px;margin-bottom:24px}
.hero h1{color:#fff;font-size:2.4rem;margin:.5rem 0}
.hero p{color:#bfe9d7}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><strong>DECENTRALIZE AI HACKATHON • PUBLIC EVIDENCE DEMO</strong><h1>Private AI Document Assistant</h1><p>Upload PDF documents, find relevant passages, and inspect cited source pages.</p></div>', unsafe_allow_html=True)
st.warning("Public cloud demo: do not upload private or confidential PDFs. Uploaded text is processed by the hosting service.")
st.caption("This deployment is a deterministic retrieval demo, not a generative AI model. The 8GB GPU / llama.cpp implementation needs separate GPU hosting.")

STOPWORDS = set("a an and are as at be but by do for from has have how i in is it of on or our that the their there these this to was were what when where which who why with you your".split())
def tokens(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w)>1 and w not in STOPWORDS]

def make_chunks(files):
    passages=[]
    for name, raw in files:
        try:
            pdf=fitz.open(stream=raw,filetype="pdf")
            if pdf.needs_pass: raise ValueError("Password-protected PDFs are not supported")
            if pdf.page_count>75: raise ValueError("Maximum 75 pages per PDF")
            for number,page in enumerate(pdf,1):
                text=page.get_text("text").strip()
                if not text: continue
                words=text.split()
                for offset in range(0,len(words),155):
                    part=" ".join(words[offset:offset+220])
                    if part.strip(): passages.append({"file":name,"page":number,"text":part})
                if len(passages)>1200: raise ValueError("Maximum 1200 passages exceeded")
            pdf.close()
        except (fitz.FileDataError,fitz.EmptyFileError,ValueError) as error:
            raise ValueError(f"{name}: {error}") from error
    if not passages: raise ValueError("No searchable text found. Scanned PDFs need OCR, which is not included.")
    return passages

def search(question, passages, limit=4):
    terms=Counter(tokens(question))
    if not terms:return []
    scored=[]
    for p in passages:
        body=Counter(tokens(p["text"]))
        score=sum(min(body[t],4)*count for t,count in terms.items())
        if score:scored.append((score,p))
    scored.sort(key=lambda x:-x[0])
    return scored[:limit]

with st.sidebar:
    st.subheader("About this demo")
    st.write("PDF → PyMuPDF → text passages → keyword retrieval → cited excerpts")
    st.write("The article's advanced local version can run Qwen2.5 GGUF with llama.cpp and FAISS separately.")
    st.caption("No files are intentionally written to persistent storage by this app.")
    if st.button("Clear documents and results"):
        for key in ("passages","names"): st.session_state.pop(key,None)
        st.rerun()

uploader,ask,about=st.tabs(["📁 Upload & Index","💬 Ask a Question","🔍 Technical Details"])
with uploader:
    uploaded=st.file_uploader("Upload up to 5 searchable PDFs (max 15 MB each)",type=["pdf"],accept_multiple_files=True)
    if st.button("Process documents",type="primary"):
        if not uploaded:st.error("Upload at least one PDF.")
        elif len(uploaded)>5:st.error("Maximum 5 PDFs.")
        elif any(len(f.getvalue())>15*1024*1024 for f in uploaded):st.error("One or more files exceed 15 MB.")
        elif sum(len(f.getvalue()) for f in uploaded)>30*1024*1024:st.error("Maximum 30 MB combined.")
        else:
            try:
                with st.spinner("Extracting and indexing PDFs..."):
                    st.session_state["passages"]=make_chunks([(f.name,f.getvalue()) for f in uploaded])
                    st.session_state["names"]=[f.name for f in uploaded]
                st.success(f"Indexed {len(st.session_state['passages'])} text passages. Open Ask a Question.")
            except Exception as error:st.error(f"Processing failed: {error}")
    if "passages" in st.session_state:
        a,b=st.columns(2)
        a.metric("Documents",len(st.session_state["names"]))
        b.metric("Passages",len(st.session_state["passages"]))
with ask:
    if "passages" not in st.session_state:
        st.info("Start by uploading and processing a PDF.")
    else:
        question=st.text_input("Question about your documents",placeholder="Which library extracts text from PDFs?")
        if st.button("Find supporting evidence",type="primary"):
            results=search(question,st.session_state["passages"])
            if not results:st.warning("No matching evidence was found. Try a more specific question.")
            else:
                st.markdown("### Retrieved evidence")
                st.info("These are source excerpts, not a model-generated answer. Verify context before relying on them.")
                for i,(score,p) in enumerate(results,1):
                    with st.expander(f"[{i}] {p['file']} — page {p['page']} (match score {score})",expanded=i==1):
                        st.write(p["text"])
with about:
    st.markdown("""
### Architecture
1. Parse PDF text using PyMuPDF.
2. Split extracted pages into overlapping text passages.
3. Score passages using query-term matches.
4. Show ranked excerpts with original filename and page number.

### Important limitations
- This cloud app has **no LLM** and **no FAISS**. It demonstrates the retrieval and citation layer.
- Scanned or image-only PDFs require OCR (not included).
- It is a prototype, not a security-audited document vault.
- An actual Qwen2.5/llama.cpp 8GB-GPU version must be deployed to GPU infrastructure and benchmarked before performance claims.
""")
st.divider()
st.caption("Hackathon prototype • Transparent retrieval • Source citations • No fabricated GPU benchmarks")

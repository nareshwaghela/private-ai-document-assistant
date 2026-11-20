# Private AI Document Assistant — Complete HackerNoon Implementation

**Project for the [Decentralize AI Hackathon](https://hackernoon.com/building-a-private-ai-document-assistant-on-an-8gb-gpu-using-rag-and-llamacpp).**

This repository implements the full reference pipeline described in [the original HackerNoon article](https://hackernoon.com/building-a-private-ai-document-assistant-on-an-8gb-gpu-using-rag-and-llamacpp), not just a keyword-search mockup.

### Architecture

```
PDF / TXT / Markdown ──> PyMuPDF / UTF-8 ──> chunks (160 words, 32 overlap)
 ──> all-MiniLM-L6-v2 SentenceTransformer (CPU) ──> normalized vectors
 ──> FAISS IndexFlatIP (CPU) ──> top-k excerpts with page/file IDs
 ──> llama.cpp /apply-template + /tokenize (real token budget)
 ──> Qwen2.5-7B-Instruct GGUF Q4_K_M (8GB GPU target)
 ──> grounded AI answer + evidence + timings
```

## Current public UI

The existing [Streamlit demo](https://app-ai-document-assistant-fwha5jhla7fkjpqtnqh5df.streamlit.app/) now implements **CPU sentence-transformer embeddings and FAISS semantic retrieval**. It will perform **true model generation only if a working llama.cpp server is configured** via `LLAMA_SERVER_URL` in the Streamlit hosting environment. Free Streamlit Community Cloud itself does **not** host an 8GB GPU, and the absence of the server is clearly disclosed in-app. **Never claim that the deployed app currently runs Qwen2.5 unless the model server actually connects.**

## Run locally (Python 3.11 recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest tests -q
python rag_local.py ingest --docs sample_docs --store store
```

### Run Qwen2.5-7B Q4_K_M on an 8GB NVIDIA GPU

You'll need an NVIDIA driver, CUDA-compatible Docker/NVIDIA Container Toolkit, sufficient RAM/VRAM, internet for the initial weight download, and the `ghcr.io/ggml-org/llama.cpp:server-cuda` image. On Linux, in a separate terminal:

```bash
bash gpu/start_llama.sh
```

This starts the **real** GGUF inference server on host `127.0.0.1:8080` (not a publicly exposed port). It requests the article's 4096-token context and GPU layer offload; verify that the model fully fits the actual card, because other GPU allocations matter. The Docker image's options can change over time; check `llama-server --help` when needed.

Now run:

```bash
python rag_local.py ask "Who owns the CSV import validator?"
python rag_local.py ask "What project budget amount was approved?"
python -m streamlit run app.py
```

The application checks the real model tokenizer using llama.cpp `/apply-template` and `/tokenize` before calling `/v1/chat/completions`. `max_tokens` and the evidence are included in the 4096-token budget. Prompt-injected file content is treated as untrusted data; the LLM has no tools.

## Evaluation: actual measurements only

The article's Project Cedar fixture includes owner **Mira**, deadline **15 November 2026**, blocker **duplicate IDs**, **no approved budget**, and an **unknown Kubernetes cluster** that should make the assistant abstain.

```bash
python rag_local.py ingest --docs sample_docs --store store
python benchmark.py --cases eval_cases.json --runs 5 --out reports/baseline.json
python -m pip freeze > reports/python-environment.txt
# On a separate terminal, capture GPU telemetry during the run:
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,utilization.gpu --format=csv -l 1 > reports/gpu-samples.csv
```

After a real run, `reports/baseline.json` contains raw answers, retrieved passage IDs, tokenizer counts, server token usage (when available), per-question retrieval/generation/total elapsed times, and medians. **No real GPU benchmark is included in source control**—you must run it on actual hardware and manually review answer correctness. Five runs are a modest baseline, not a reliable p95.

## Deployment notes

- **Streamlit Cloud:** CPU-based FAISS retrieval, text extraction and hosted UI. Potentially slow first-time downloads of Sentence Transformers. Community Cloud memory limits apply. Not private for uploaded documents.
- **Local GPU:** Run Streamlit and llama-server on the same machine. This is the closest implementation of the article's privacy-first premise.
- **Remote GPU:** Set server URL only via trusted server-side environment variables; use HTTPS, authentication, and access controls. Do not expose llama-server open to the internet or allow end users to choose arbitrary backend URLs.
- **Nosana:** The GPU image and model runtime can be deployed on compatible GPU infrastructure using Nosana's service configuration; an actual deployment requires a user account, credits, a published container image, and verification. This repository does not assert GPU deployment succeeded.

### Safety and limitations

The demo is **single-user reference code**, not a secure multi-tenant document vault. Avoid confidential uploads to Community Cloud, enforce access control before storing user indexes on disk, and never import untrusted `.faiss` indexes. Scanned/image-only PDFs require OCR. Source citations are audit aids, not guarantees of accuracy.

## Project contents

- `rag_local.py` — article reference CLI, local ingestion, CPU embeddings, FAISS and llama.cpp chat
- `app.py` — Streamlit GUI integrating that exact pipeline
- `benchmark.py` and `eval_cases.json` — raw-run evaluation and five Project Cedar questions
- `sample_docs/project_cedar.txt` — reproducible fixture
- `tests/` — offline unit tests, including token-budget and source metadata
- `gpu/start_llama.sh` — CUDA-based llama.cpp server (requires NVIDIA hardware)
- `.github/workflows/test.yml` — syntax and unit tests (no fake GPU measurements)

## Documentation links

- [Original article](https://hackernoon.com/building-a-private-ai-document-assistant-on-an-8gb-gpu-using-rag-and-llamacpp)
- [llama.cpp server reference](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [Official llama.cpp Docker images](https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md)

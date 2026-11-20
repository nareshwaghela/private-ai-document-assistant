# Private AI Document Assistant — Hackathon Demo

A source-grounded PDF question-answering demo for the Decentralize AI Hackathon.

## Live application

Deploy `app.py` using [Streamlit Community Cloud](https://share.streamlit.io/). The public hosted version provides **document retrieval and cited excerpts**, not local LLM generation. Its optional llama.cpp/Qwen2.5 deployment is a separate GPU task.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

This repository's hosted prototype indexes selectable text in uploaded PDF documents, performs keyword-ranked retrieval, and shows matching page-specific source excerpts. No invented model-generated answers or measured GPU benchmarks are claimed.

**Privacy:** Uploaded documents are processed by the host running this app. Never upload confidential files to the public demo.
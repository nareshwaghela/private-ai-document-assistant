# Deploy to Streamlit Community Cloud

1. Visit https://share.streamlit.io/ and sign in using GitHub.
2. Choose **Create app** or **New app**, select `nareshwaghela/private-ai-document-assistant`, branch `main`, and main file `app.py`.
3. Click **Deploy**. Streamlit will install packages from `requirements.txt` automatically.
4. On the public URL, upload a searchable PDF and click **Process documents**. Switch to **Ask a Question**, enter a query, and click **Find supporting evidence**.

**Important:** GitHub is source hosting, not a Python server. Public Streamlit Cloud offers a CPU retrieval demo only. Full private Qwen2.5 GGUF + llama.cpp GPU generation must be deployed separately to a GPU-compatible platform such as Nosana. The hosted app handles user PDF content on a third-party server; never submit confidential data.

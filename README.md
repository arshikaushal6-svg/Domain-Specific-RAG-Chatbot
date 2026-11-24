# Domain-Specific RAG Chatbot for PDF Question Answering

A Streamlit app for uploading PDFs, extracting text page-by-page, chunking, embedding with Sentence Transformers, indexing/searching with FAISS, and answering questions with source filename and page references.

## Features
- Multiple PDF uploads (20 MB per file)
- PyPDF extraction with page metadata
- LangChain recursive text chunking (850 characters, 120 overlap)
- `all-MiniLM-L6-v2` embeddings
- FAISS vector similarity search (top 4 chunks)
- Source document and page display
- Optional Groq LLM generation with a context-only prompt
- Works without an API key in extractive retrieval mode
- Clear documents and chat

## Recommended environment
Windows 10/11 64-bit and Python 3.11 or 3.12. ML dependencies may not yet support every newly released Python version.

## Run in VS Code (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
If activation is blocked:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```
Open the local URL printed by Streamlit, usually http://localhost:8501. The first run downloads the embedding model and may take a few minutes.

## Optional Groq integration
The app runs without a key, showing retrieved document passages directly. To enable generated answers, copy `.env.example` to `.env`, add your `GROQ_API_KEY`, and restart Streamlit. Install `python-dotenv` is included, but this basic app reads environment variables directly; in PowerShell you can set `$env:GROQ_API_KEY="your-key"` before running Streamlit. Never commit API keys to GitHub.

## Structure
```text
domain_rag_chatbot/
├── app.py
├── rag_pipeline.py
├── document_loader.py
├── vector_store.py
├── prompt.py
├── requirements.txt
├── README.md
├── .env.example
├── documents/sample_handbook.pdf
├── vector_store/
└── tests/test_questions.csv
```

## Workflow
Upload PDFs -> extract pages and metadata -> chunk -> create embeddings -> FAISS index -> retrieve top chunks -> answer from retrieved context -> show source/page.

## Limitations and responsible use
- Scanned/image-only PDFs are not supported; OCR is not included.
- Index exists only in the active app session and is not persisted.
- Without Groq, the system returns retrieved passages rather than a synthesized answer.
- Similarity is a retrieval signal, not proof of answer correctness.
- RAG can still retrieve irrelevant content or generate mistakes. Verify important claims in the original PDF.
- Do not upload confidential documents without permission. Uploaded files/index are not saved by this app, but consider the hosting environment.
- Instructions inside PDFs are treated as untrusted data.

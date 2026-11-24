import streamlit as st
from document_loader import load_uploaded_pdfs
from rag_pipeline import build_rag_index, answer_question

st.set_page_config(page_title="Domain-Specific RAG Chatbot", page_icon="📚", layout="wide")
st.title("📚 Domain-Specific RAG Chatbot")
st.caption("Ask questions about your PDFs and receive answers with document and page references.")
st.warning("For important or high-stakes decisions, verify answers against the original documents.")

if "rag_index" not in st.session_state: st.session_state.rag_index = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

with st.sidebar:
    st.header("Document collection")
    uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        st.write("**Selected documents**")
        for file in uploaded_files: st.write(f"📄 {file.name} ({file.size / 1024:.1f} KB)")
    if st.button("Process documents", type="primary", use_container_width=True):
        if not uploaded_files: st.error("Please upload at least one PDF.")
        elif any(f.size > 20 * 1024 * 1024 for f in uploaded_files): st.error("Each PDF must be 20 MB or smaller.")
        else:
            with st.spinner("Extracting text, creating embeddings, and building the search index..."):
                try:
                    pages = load_uploaded_pdfs(uploaded_files)
                    if not pages: st.error("No readable text found. Scanned PDFs need OCR, not included in this version.")
                    else:
                        st.session_state.rag_index = build_rag_index(pages)
                        st.session_state.chat_history = []
                        st.success(f"Processed {len(uploaded_files)} PDF(s) into {len(st.session_state.rag_index['chunks'])} chunks.")
                except Exception as exc: st.error(f"Could not process documents: {exc}")
    if st.button("Clear documents and chat", use_container_width=True):
        st.session_state.rag_index = None
        st.session_state.chat_history = []
        st.rerun()
    st.divider()
    st.caption("Uploaded PDFs and the FAISS index are held in the current app session, not saved permanently.")

if st.session_state.rag_index is None:
    st.info("Upload PDF documents in the sidebar, then click Process documents.")
    st.markdown("### Workflow\n1. Extract PDF text page by page.\n2. Split text into overlapping chunks.\n3. Create Sentence Transformer embeddings.\n4. Search vectors using FAISS.\n5. Answer from retrieved passages and show sources.")
    st.stop()

st.success("Document index is ready. Ask a question.")
for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("sources"):
            with st.expander("View sources"):
                for src in item["sources"]:
                    st.markdown(f"**{src['source']} — Page {src['page']}**")
                    st.caption(src["text"])

question = st.chat_input("Ask a question about your uploaded documents...")
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"): st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."): result = answer_question(question, st.session_state.rag_index)
        st.markdown(result["answer"])
        if result["sources"]:
            st.markdown("**Sources**")
            for i, src in enumerate(result["sources"], 1):
                with st.expander(f"{i}. {src['source']} — Page {src['page']}"): st.write(src["text"])
        st.caption(result["mode"])
    st.session_state.chat_history.append({"role":"assistant","content":result["answer"],"sources":result["sources"]})

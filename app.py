"""
Financial RAG Chatbot
=====================
A Retrieval-Augmented Generation chatbot for financial document Q&A.
Built with LangChain, OpenAI GPT-4, ChromaDB, and Streamlit.

Author: Satya Sai Prakash Kantamani
GitHub: https://github.com/kantamaniprakash
"""

import os
import html
import streamlit as st
from pathlib import Path

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
import chromadb
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

from chunking import chunk_documents

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .chat-msg-user {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 1rem 1.2rem;
        border-radius: 12px 12px 4px 12px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        color: white;
    }
    .chat-msg-assistant {
        background: #1c2128;
        border: 1px solid #30363d;
        padding: 1rem 1.2rem;
        border-radius: 12px 12px 12px 4px;
        margin: 0.5rem 0;
        max-width: 85%;
        color: #e6edf3;
    }
    .source-chip {
        display: inline-block;
        background: rgba(88,166,255,0.1);
        border: 1px solid rgba(88,166,255,0.3);
        color: #58a6ff;
        padding: 0.2rem 0.7rem;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 0.2rem;
    }
    .metric-card {
        background: #1c2128;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
# Chunking defaults come from the budget-matched benchmark in
# https://github.com/kantamaniprakash/genai-lab (rag-chunking-bench);
# rationale and numbers are in this repo's README under "Chunking configuration".
CHUNK_TOKENS = 256
CHUNK_OVERLAP_SENTENCES = 0
TOP_K_RETRIEVAL = 5
PERSIST_DIR = str(Path(__file__).resolve().parent / "chroma_db")
COLLECTION_NAME = "financial_documents"

# ─── System Prompt ─────────────────────────────────────────────────────────────
FINANCIAL_SYSTEM_PROMPT = """You are an expert financial analyst AI assistant. You answer questions about financial documents (annual reports, 10-Ks, earnings transcripts, analyst reports) using ONLY the provided context.

Rules:
1. Be precise and cite specific numbers, dates, and facts from the context
2. If the context doesn't contain enough information, say so clearly
3. Format financial figures properly (e.g., $1.2B, 15.3%, Q3 FY2024)
4. Highlight key insights and trends when relevant
5. Never hallucinate or invent financial data

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""

# ─── Session State Init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# ─── Helper Functions ──────────────────────────────────────────────────────────

def load_and_chunk_pdfs(uploaded_files: list) -> list:
    """Load PDFs and split into sentence-aware, token-budgeted chunks."""
    import fitz
    from langchain_core.documents import Document

    all_docs = []
    for uploaded_file in uploaded_files:
        pdf_bytes = uploaded_file.getvalue()
        if not pdf_bytes:
            raise ValueError(f"{uploaded_file.name} is empty.")
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                if doc.needs_pass:
                    raise ValueError(f"{uploaded_file.name} is password-protected.")
                docs = []
                for page_num in range(doc.page_count):
                    page_text = doc[page_num].get_text("text").strip()
                    if page_text:
                        docs.append(Document(
                            page_content=page_text,
                            metadata={"source_file": uploaded_file.name, "page": page_num + 1}
                        ))
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(
                f"Could not read '{uploaded_file.name}'. Please upload a valid, text-based PDF."
            ) from exc

        if not docs:
            st.warning(
                f"No selectable text was found in {uploaded_file.name}. "
                "Scanned/image-only PDFs need OCR and are not supported by this version."
            )
            continue

        chunks = chunk_documents(
            docs,
            max_tokens=CHUNK_TOKENS,
            overlap_sentences=CHUNK_OVERLAP_SENTENCES,
        )
        all_docs.extend(chunks)
    return all_docs


def build_vector_store(chunks: list, api_key: str) -> Chroma:
    """Embed chunks and store in ChromaDB."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key
    )
    if not chunks:
        raise ValueError(
            "No readable text was found in the uploaded PDFs. "
            "Try a text-based PDF instead of a scanned image."
        )

    # Rebuild one named collection on each explicit processing run. This avoids
    # silently appending duplicate chunks when users click Process Documents twice.
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    existing_collections = {
        collection.name for collection in client.list_collections()
    }
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(COLLECTION_NAME)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )


def build_qa_chain(vector_store: Chroma, api_key: str) -> ConversationalRetrievalChain:
    """Build the conversational RAG chain."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",   # swap to "gpt-4" for higher accuracy
        temperature=0.1,
        openai_api_key=api_key
    )
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=5  # Keep last 5 turns
    )
    retriever = vector_store.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance for diversity
        search_kwargs={"k": TOP_K_RETRIEVAL, "fetch_k": 15}
    )
    qa_prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=FINANCIAL_SYSTEM_PROMPT
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )
    return chain


def format_sources(source_docs: list) -> str:
    """Format source document citations (HTML-escaped to prevent injection)."""
    seen = set()
    chips = []
    for doc in source_docs:
        src = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "?")
        key = f"{src} p.{page}"
        if key not in seen:
            seen.add(key)
            chips.append(
                f'<span class="source-chip">📄 {html.escape(str(src))} · Page {html.escape(str(page))}</span>'
            )
    return " ".join(chips)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Financial RAG Chatbot")
    st.markdown("*Ask questions about your financial documents*")
    st.divider()

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your OpenAI API key (never stored)"
    )

    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload 10-Ks, earnings transcripts, analyst reports..."
    )

    if uploaded_files:
        if st.button("⚡ Process Documents", type="primary", use_container_width=True):
            if not api_key or not api_key.strip():
                st.warning("Please enter your OpenAI API key above.")
            else:
                with st.spinner("Reading PDFs, creating embeddings, and indexing documents..."):
                    try:
                        chunks = load_and_chunk_pdfs(uploaded_files)
                        if not chunks:
                            raise ValueError(
                                "No readable PDF text was found. Text-based PDFs are supported; "
                                "scanned/image-only PDFs need OCR."
                            )

                        vs = build_vector_store(chunks, api_key.strip())
                        chain = build_qa_chain(vs, api_key.strip())

                        # Update session state only after the full pipeline succeeds.
                        st.session_state.vector_store = vs
                        st.session_state.chain = chain
                        st.session_state.chunk_count = len(chunks)
                        st.session_state.doc_count = len(uploaded_files)
                        st.session_state.messages = []
                        st.success(
                            f"Indexed {len(chunks)} chunks from {len(uploaded_files)} PDF(s). "
                            "You can now ask questions."
                        )
                    except Exception as exc:
                        message = str(exc)
                        lower = message.lower()
                        if "api key" in lower or "authentication" in lower or "401" in lower:
                            st.error(
                                "OpenAI authentication failed. Check that the API key is correct "
                                "and that your OpenAI account has API billing/credits enabled."
                            )
                        elif "429" in lower or "quota" in lower or "rate limit" in lower:
                            st.error(
                                "OpenAI quota/rate limit reached. Check API billing, available "
                                "credits, and usage limits, then try again."
                            )
                        else:
                            st.error(f"Could not process documents: {message}")

    # Stats
    if st.session_state.vector_store:
        st.divider()
        st.markdown("### 📊 Index Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", st.session_state.doc_count)
        with col2:
            st.metric("Chunks", st.session_state.chunk_count)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.chain:
            st.session_state.chain.memory.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **Built and customized by Prem Devnath**

    Stack: Python · LangChain · OpenAI API · ChromaDB · Streamlit

    *Based on an open-source project by [Satya Sai Prakash Kantamani](https://github.com/kantamaniprakash).*
    """)

# ─── Main Chat UI ──────────────────────────────────────────────────────────────
st.markdown("# 💬 Financial Document Q&A")

if not st.session_state.vector_store:
    st.info("👈 Upload financial documents in the sidebar and click **Process Documents** to begin.")
    st.markdown("""
    ### 🚀 What you can do:
    - Upload **10-K annual reports**, earnings call transcripts, analyst research
    - Ask **natural language questions** about financials, risk factors, strategy
    - Get **cited answers** with exact page references
    - Maintain **conversation context** across multiple questions

    ### Example questions to try:
    > *"What was the revenue growth year-over-year?"*
    > *"Summarize the key risk factors"*
    > *"What did management say about AI investments?"*
    > *"Compare operating margins between segments"*
    """)
else:
    # Display chat history (content is HTML-escaped so document/model text
    # cannot inject HTML or scripts into the page)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-msg-user">👤 {html.escape(msg["content"])}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-msg-assistant">🤖 {html.escape(msg["content"])}</div>', unsafe_allow_html=True)
            if "sources" in msg:
                st.markdown(msg["sources"], unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask a question about your financial documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="chat-msg-user">👤 {html.escape(prompt)}</div>', unsafe_allow_html=True)

        with st.spinner("Searching documents and generating answer..."):
            try:
                result = st.session_state.chain.invoke({"question": prompt})
                answer = result["answer"]
                sources_html = format_sources(result.get("source_documents", []))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_html
                })
                st.markdown(f'<div class="chat-msg-assistant">🤖 {html.escape(answer)}</div>', unsafe_allow_html=True)
                if sources_html:
                    st.markdown("**Sources:** " + sources_html, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generating response: {e}")

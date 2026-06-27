"""Retrieval-augmented generation over a private document corpus.

Loads PDFs from a directory, chunks and embeds them into an in-memory Chroma
vector store, and exposes a ``query_private_database`` tool the agent can call to
ground its answers in proprietary research instead of hallucinating.
"""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import get_chat_model, get_embeddings

_retriever = None  # populated by build_retriever()


def build_retriever(pdf_dir: str = "data/reports", k: int = 6):
    """Build (and cache) a retriever over the PDFs in ``pdf_dir``."""
    global _retriever
    path = Path(pdf_dir)
    if not path.exists() or not any(path.glob("*.pdf")):
        raise FileNotFoundError(
            f"No PDFs found in '{pdf_dir}'. Add analyst report PDFs there first."
        )

    docs = PyPDFDirectoryLoader(str(path)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name="private_reports",
    )
    _retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return _retriever


@tool
def query_private_database(query: str) -> str:
    """Answer a question using the private corpus of analyst reports (RAG).

    Use this for proprietary insights — company strategy, AI initiatives,
    research roadmaps — that public sources do not contain.
    """
    if _retriever is None:
        return ("Private database is not initialized. Call build_retriever() and "
                "ensure analyst PDFs are present in data/reports/.")
    try:
        chunks = _retriever.invoke(query)
        if not chunks:
            return "No relevant information found in the analyst reports."
        context = "\n\n---\n\n".join(d.page_content for d in chunks)
        prompt = (
            "You are reviewing internal analyst reports. Answer the question using "
            "ONLY the context below. Cite the company each fact comes from. If the "
            "answer is not in the context, say so.\n\n"
            f"### Context\n{context}\n\n### Question\n{query}"
        )
        return get_chat_model().invoke(prompt).content
    except Exception as e:  # noqa: BLE001
        return f"Error querying private database: {e}"

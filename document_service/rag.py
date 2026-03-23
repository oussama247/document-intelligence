"""
document_service/rag.py
-----------------------
Retrieval-Augmented Generation (RAG) utilities:
  - build_vectorstore : chunk text → embed → persist FAISS index
  - load_vectorstore  : reload the persisted FAISS index from disk
  - ask_question      : retrieve relevant chunks → generate answer via Groq LLM
  - generate_index_report : write a human-readable audit report
"""

import datetime
import os

from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Directory where the FAISS index files (index.faiss + index.pkl) are saved.
# Mounted as a named Docker volume so the index survives container restarts.
INDEX_PATH = "faiss_index"

# Embedding model used both at indexing time and at retrieval time.
# e5-small-v2 is a compact, high-quality multilingual model (~33 MB).
EMBEDDING_MODEL = "intfloat/e5-small-v2"

# LLM used for answer generation
GROQ_MODEL = "llama-3.1-8b-instant"

# Text splitting parameters
CHUNK_SIZE = 1000    # Maximum characters per chunk
CHUNK_OVERLAP = 200  # Overlap between consecutive chunks (preserves context)

# ---------------------------------------------------------------------------
# Module-level singleton — loaded once when the service starts
# ---------------------------------------------------------------------------

# Loading the embedding model takes ~1-2 s and downloads ~33 MB on first run.
# Instantiating it here (module scope) means it is loaded exactly once per
# worker process, instead of on every request.
_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def build_vectorstore(text: str) -> list[str]:
    """
    Split *text* into overlapping chunks, embed them with HuggingFace, and
    persist a FAISS vector index to INDEX_PATH.

    The index is overwritten on every call, so uploading a new document
    replaces the previous one.

    Args:
        text: Full extracted text of the document.

    Returns:
        List of text chunks (useful for reporting and unit-testing).
    """
    # --- 1. Split the document into manageable chunks ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)

    # --- 2. Use the module-level singleton (loaded once at startup) ---
    # --- 3. Build the FAISS index and persist it locally ---
    vectorstore = FAISS.from_texts(chunks, _embeddings)
    vectorstore.save_local(INDEX_PATH)

    return chunks


def load_vectorstore() -> FAISS | None:
    """
    Load the FAISS index from disk if it exists.

    Returns:
        A ready-to-query FAISS vectorstore, or None if no index has been built yet.
    """
    if not os.path.exists(INDEX_PATH):
        return None

    # Reuse the module-level singleton — no reload cost
    return FAISS.load_local(
        INDEX_PATH,
        _embeddings,
        allow_dangerous_deserialization=True,
    )


def ask_question(question: str) -> str:
    """
    Answer *question* using RAG:
      1. Retrieve the top-k most relevant chunks from the FAISS index.
      2. Concatenate them as context.
      3. Ask the Groq LLM to answer using only that context.

    Args:
        question: Natural-language question from the user.

    Returns:
        The LLM-generated answer as a plain string, or an error message if
        no index is available.
    """
    vectorstore = load_vectorstore()

    if vectorstore is None:
        return "No document indexed. Please upload a document first."

    # --- Retrieve relevant chunks ---
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # --- Generate the answer ---
    llm = ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = f"""Answer the question using only the context below.
If the answer cannot be found in the context, say so explicitly.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    return response.content


def generate_index_report(filename: str, text: str, chunks: list[str]) -> None:
    """
    Write a plain-text audit report describing the last indexing operation.

    The report is saved as ``indexing_report.txt`` inside the service working
    directory. It is intended for debugging and traceability, not production
    logging.

    Args:
        filename: Original filename of the uploaded document.
        text:     Full extracted text (used to count characters).
        chunks:   List of text chunks produced by the splitter.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    with open("indexing_report.txt", "w", encoding="utf-8") as f:
        f.write("INDEXING REPORT\n")
        f.write("----------------------------\n\n")
        f.write(f"Timestamp:           {timestamp}\n")
        f.write(f"File:                {filename}\n")
        f.write(f"Characters extracted:{len(text)}\n")
        f.write(f"Chunks created:      {len(chunks)}\n")
        f.write(f"Chunk size:          {CHUNK_SIZE}\n")
        f.write(f"Chunk overlap:       {CHUNK_OVERLAP}\n")
        f.write(f"Embedding model:     {EMBEDDING_MODEL}\n")
        f.write( "Vector store:        FAISS\n")
        f.write( "\nStatus: SUCCESS\n")
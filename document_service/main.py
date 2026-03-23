"""
document_service/main.py
------------------------
FastAPI microservice responsible for:
  - Receiving document uploads (PDF or image)
  - Extracting text via pdfplumber (native) or OCR fallback
  - Building a FAISS vector index for RAG (Retrieval-Augmented Generation)
  - Answering natural-language questions via RAG + Groq LLM
  - Analyzing documents and returning structured JSON output
"""

import json
import os
import re

import pdfplumber
import requests
from fastapi import FastAPI, File, UploadFile

from rag import ask_question, build_vectorstore, generate_index_report

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Document Service",
    description="Handles document ingestion, RAG indexing, Q&A and structured extraction.",
    version="1.0.0",
)

# URL of the companion OCR microservice.
# Switch between the two constants below depending on your runtime environment.
# OCR_URL = "http://localhost:8002/ocr"   # ← local development
OCR_URL = "http://ocr_service:8002/ocr"  # ← Docker Compose network

# Minimum number of characters required to consider a text extraction successful.
# Documents returning fewer characters are treated as empty / scanned and
# forwarded to the OCR service.
MIN_TEXT_LENGTH = 50


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def extract_text_with_fallback(path: str, filename: str) -> str:
    """
    Extract text from a document using a two-step strategy:

    1. For PDFs  → try native text extraction with pdfplumber.
       If the result is shorter than MIN_TEXT_LENGTH (e.g. scanned PDF),
       fall back to the OCR service.
    2. For every other file type → send directly to the OCR service.

    Args:
        path:     Absolute or relative path to the temporary file on disk.
        filename: Original filename (used to detect the .pdf extension).

    Returns:
        Extracted text as a plain string.
    """
    text = ""

    if filename.lower().endswith(".pdf"):
        # --- Primary path: native PDF text extraction ---
        with pdfplumber.open(path) as pdf:
            text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )

        # --- Fallback: OCR when the native extraction is insufficient ---
        if len(text.strip()) < MIN_TEXT_LENGTH:
            with open(path, "rb") as f:
                response = requests.post(OCR_URL, files={"file": f})
            text = response.json()["text"]

    else:
        # --- Non-PDF files: always use OCR ---
        with open(path, "rb") as f:
            response = requests.post(OCR_URL, files={"file": f})
        text = response.json()["text"]

    return text


def clean_llm_json(text: str) -> dict:
    """
    Sanitise the raw string returned by the LLM so it can be parsed as JSON.

    LLMs often wrap their output in markdown code fences (```json … ```) or
    add explanatory prose before/after the JSON object. This function strips
    all of that and returns a Python dict.

    Args:
        text: Raw string output from the LLM.

    Returns:
        Parsed dict on success, or a dict with an "error" key on failure.
    """
    # Remove markdown code fences
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    # Isolate the outermost JSON object (handles leading/trailing noise)
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start : end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON from LLM", "raw_output": text}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/upload", summary="Index a document into the FAISS vector store")
async def upload(file: UploadFile = File(...)):
    """
    Receive a document, extract its text, and build (or overwrite) the
    FAISS vector index used by the /ask endpoint.

    Steps:
      1. Write the uploaded bytes to a temporary file.
      2. Extract text (pdfplumber → OCR fallback).
      3. Split into chunks and embed into FAISS.
      4. Generate an indexing report for audit purposes.
      5. Delete the temporary file.

    Returns:
        {"status": "Document indexed successfully"} on success.
        {"error": "<reason>"}                       on failure.
    """
    temp_path = "temp_file"

    # Persist upload to disk so pdfplumber / OCR can access it by path
    content = await file.read()
    with open(temp_path, "wb") as f:
        f.write(content)

    try:
        text = extract_text_with_fallback(temp_path, file.filename)

        if len(text.strip()) < MIN_TEXT_LENGTH:
            return {"error": "Document vide ou extraction échouée."}

        # Build the vector index and get back the list of text chunks
        chunks = build_vectorstore(text)

        # Write a human-readable report for traceability
        generate_index_report(file.filename, text, chunks)

        return {"status": "Document indexed successfully"}

    finally:
        # Always clean up the temp file, even if an exception occurred
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/ask", summary="Ask a natural-language question about the indexed document")
async def ask(data: dict):
    """
    Answer a question using Retrieval-Augmented Generation (RAG):
      1. Retrieve the most relevant chunks from the FAISS index.
      2. Pass them as context to the Groq LLM.
      3. Return the generated answer.

    Request body:
        {"question": "<your question>"}

    Returns:
        {"Réponse:": "<answer>"} or an error message if no index exists.
    """
    question = data["question"]
    answer = ask_question(question)
    return {"Réponse:": answer}


@app.post("/analyze", summary="Extract structured data from a document as JSON")
async def analyze(file: UploadFile = File(...)):
    """
    Analyse a document with the Groq LLM and return a structured JSON object
    describing the document type, key fields, named entities, and evidence.

    The LLM is instructed to produce valid JSON only (no markdown, no prose),
    conforming to the schema below:

        {
          "doc_type":            string,   // e.g. "contract", "invoice"
          "doc_type_confidence": number,   // 0.0 – 1.0
          "fields":              object,   // key/value pairs of extracted data
          "entities":            array,    // named entities (persons, orgs …)
          "evidence":            array     // [{"field": "…", "text": "…"}, …]
        }

    Dates are normalised to ISO 8601 (YYYY-MM-DD); monetary amounts include
    the currency symbol or code.

    Returns:
        Parsed JSON dict on success, or {"error": "…"} on failure.
    """
    # Lazy import: keeps module-level imports minimal and avoids loading
    # heavyweight dependencies until this endpoint is actually called.
    from langchain_groq import ChatGroq

    temp_path = "temp_file"

    content = await file.read()
    with open(temp_path, "wb") as f:
        f.write(content)

    try:
        text = extract_text_with_fallback(temp_path, file.filename)

        if len(text.strip()) < MIN_TEXT_LENGTH:
            return {"error": "Document vide ou extraction échouée."}

        # Build the structured-extraction prompt.
        # The schema is embedded directly so the LLM knows exactly what to output.
        prompt = f"""
Tu es un système d'extraction d'information.

Réponds UNIQUEMENT avec un JSON valide.
Aucun texte explicatif.
Aucun markdown.

Schéma exact :

{{
  "doc_type": string,
  "doc_type_confidence": number,
  "fields": object,
  "entities": array,
  "evidence": array
}}

Contraintes :
- Dates au format ISO YYYY-MM-DD
- Montants sous forme nombre + devise
- evidence: liste de {{ "field": "...", "text": "..." }}

Document :
{text}
"""

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

        response = llm.invoke(prompt)

        # response is a LangChain AIMessage; .content holds the raw string
        structured_json = clean_llm_json(response.content)

        return structured_json

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

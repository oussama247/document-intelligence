"""
streamlit_app/app.py
--------------------
Streamlit front-end for the Intelligent Document Converter.

This single-page application exposes three features:
  1. Index Document  – uploads a file to the RAG pipeline (/upload)
  2. Ask a Question  – queries the indexed document in natural language (/ask)
  3. Convert to JSON – extracts structured data from the document (/analyze)

The resulting JSON can be previewed and downloaded directly from the page.
"""

import json

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Base URL of the document_service FastAPI backend.
# Switch between the two constants below depending on your runtime environment.
# API_URL = "http://localhost:8000"        # ← local development
API_URL = "http://document_service:8000"  # ← Docker Compose network

# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Intelligent Document Converter", page_icon="📄")
st.title("📄 Intelligent Document Converter")

# Single file uploader shared by all three actions.
# The widget is rendered once at the top so the user uploads only once.
uploaded_file = st.file_uploader(
    "Upload a PDF or image",
    type=["pdf", "png", "jpg", "jpeg", "tiff"],
    help="Supported formats: PDF, PNG, JPEG, TIFF",
)

st.divider()

# ---------------------------------------------------------------------------
# Section 1 – Index Document (RAG)
# ---------------------------------------------------------------------------

st.subheader("📥 Index Document")
st.caption("Embed the document into the vector store so you can ask questions about it.")

if uploaded_file and st.button("Index Document", key="index_btn"):
    with st.spinner("Indexing …"):
        response = requests.post(
            f"{API_URL}/upload",
            files={"file": uploaded_file},
        )

    if response.status_code == 200:
        st.success("Document indexed successfully!")
    else:
        st.error("Indexing failed.")
        st.code(response.text)

st.divider()

# ---------------------------------------------------------------------------
# Section 2 – Ask a Question (RAG)
# ---------------------------------------------------------------------------

st.subheader("💬 Ask a Question")
st.caption(
    "Type a question in any language. The answer is generated from the indexed document."
)

# Example questions (shown as hints below the input)
EXAMPLE_QUESTIONS = [
    "Quelle est la durée du contrat ?",
    "Quel est le montant total du contrat ?",
    "Qui représente AlphaTech ?",
    "Quelles sont les dates de paiement ?",
    "Quel est l'objet du contrat ?",
]

question = st.text_input(
    "Your question",
    placeholder="e.g. What is the contract duration?",
)

with st.expander("💡 Example questions"):
    for q in EXAMPLE_QUESTIONS:
        st.markdown(f"- *{q}*")

if st.button("Ask", key="ask_btn"):
    if not question.strip():
        st.warning("Please enter a question before clicking Ask.")
    else:
        with st.spinner("Searching the document …"):
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question},
            )

        if response.status_code == 200:
            answer_data = response.json()
            # The backend returns {"Réponse:": "<text>"}
            answer = answer_data.get("Réponse:", answer_data)
            st.info(answer)
        else:
            st.error("An error occurred while querying the document.")
            st.code(response.text)

st.divider()

# ---------------------------------------------------------------------------
# Section 3 – Convert to Structured JSON
# ---------------------------------------------------------------------------

st.subheader("🔍 Convert to Structured JSON")
st.caption(
    "Analyse the document and extract key fields, entities, and evidence as JSON."
)

if uploaded_file and st.button("Convert to JSON", key="json_btn"):
    with st.spinner("Analysing document …"):
        response = requests.post(
            f"{API_URL}/analyze",
            files={"file": uploaded_file},
        )

    if response.status_code == 200:
        result = response.json()

        # Store both the parsed dict (for st.json rendering) and the formatted
        # string (for the download button) in session state so they survive
        # a Streamlit rerun triggered by other widgets.
        st.session_state["json_dict"] = result
        st.session_state["json_string"] = json.dumps(
            result,
            indent=4,
            ensure_ascii=False,
        )
    else:
        st.error("Error during analysis.")
        st.code(response.text)

# ---------------------------------------------------------------------------
# Section 4 – Display & Download JSON (rendered whenever data is available)
# ---------------------------------------------------------------------------

if "json_dict" in st.session_state:
    st.subheader("📋 Structured JSON Output")
    st.json(st.session_state["json_dict"])

    st.download_button(
        label="⬇️ Download JSON",
        data=st.session_state["json_string"],
        file_name="analysis_result.json",
        mime="application/json",
        key="download_btn",
    )

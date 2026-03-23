#  Intelligent Document Converter

A containerised multi-service application that turns PDFs and scanned images into structured, queryable data.

## Architecture

```
┌─────────────────────────────────────────────┐
│              Docker Compose                 │
│                                             │
│  ┌─────────────┐      ┌──────────────────┐  │
│  │  Streamlit  │─────▶│ Document Service │  │
│  │   :8501     │      │    FastAPI :8000  │  │
│  └─────────────┘      └────────┬─────────┘  │
│                                │            │
│                       ┌────────▼─────────┐  │
│                       │   OCR Service    │  │
│                       │  FastAPI :8002   │  │
│                       └──────────────────┘  │
└─────────────────────────────────────────────┘
```

| Service | Role |
|---|---|
| `streamlit_app` | Browser UI — upload, Q&A, JSON export |
| `document_service` | Text extraction, FAISS indexing, RAG, LLM analysis |
| `ocr_service` | Tesseract OCR for scanned PDFs and images |

## Features

- **Smart text extraction** — uses native PDF parsing first; falls back to OCR for scanned documents
- **RAG Q&A** — ask natural-language questions about your document (powered by FAISS + Groq LLaMA 3)
- **Structured JSON extraction** — extracts document type, fields, entities and evidence
- **Download results** — export the JSON analysis with one click

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) ≥ 24
- [Docker Compose](https://docs.docker.com/compose/) ≥ 2
- A [Groq](https://console.groq.com/) API key (free tier is sufficient)

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/project-convertisseur.git
cd project-convertisseur
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get a free key at [console.groq.com](https://console.groq.com/).

### 3. Build and start all services

```bash
docker compose up --build
```

First build takes a few minutes (downloads the embedding model ~33 MB and PyTorch CPU).
Subsequent starts are fast:

```bash
docker compose up
```

### 4. Open the UI

Navigate to [http://localhost:8501](http://localhost:8501).

### 5. Stop the stack

```bash
# Stop containers but keep the FAISS index
docker compose down

# Stop and also delete the FAISS index (full reset)
docker compose down -v
```

### Running locally without Docker (optional)

If you want to run services directly on your machine:

**OCR Service**
```bash
cd ocr_service
pip install -r requirements.txt
uvicorn main:app --port 8002
```

**Document Service** — in a second terminal
```bash
cd document_service
pip install -r requirements.txt
# Switch OCR_URL in main.py to http://localhost:8002/ocr
uvicorn main:app --port 8000
```

**Streamlit App** — in a third terminal
```bash
cd streamlit_app
pip install -r requirements.txt
# Switch API_URL in app.py to http://localhost:8000
streamlit run app.py
```

## Usage

1. **Upload** a PDF or image file.
2. Click **Index Document** to embed it into the vector store.
3. Type a question and click **Ask** to query the document with RAG.
4. Click **Convert to JSON** to extract structured data.
5. **Download** the JSON result if needed.

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key — required for LLM calls |

## Project Structure

```
.
├── docker-compose.yml
├── .env.example
├── .gitignore
├── README.md
├── document_service/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py          # FastAPI app — /upload, /ask, /analyze
│   └── rag.py           # FAISS vectorstore + RAG helpers
├── ocr_service/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py          # FastAPI app — /ocr
└── streamlit_app/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py           # Streamlit front-end
```

## API Reference

### Document Service (`http://localhost:8000`)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Index a document (multipart `file`) |
| `POST` | `/ask` | Answer a question — body: `{"question": "…"}` |
| `POST` | `/analyze` | Return structured JSON (multipart `file`) |

### OCR Service (`http://localhost:8002`)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ocr` | Extract text from a PDF or image (multipart `file`) |

## Tech Stack

- **FastAPI** — REST API framework
- **Streamlit** — UI framework
- **pdfplumber** — Native PDF text extraction
- **Tesseract / pdf2image** — OCR engine
- **LangChain + FAISS** — Vector store and RAG pipeline
- **HuggingFace Embeddings** — `intfloat/e5-small-v2`
- **Groq** — LLM inference (LLaMA 3.1 8B Instant)
- **Docker Compose** — Container orchestration

## License

MIT

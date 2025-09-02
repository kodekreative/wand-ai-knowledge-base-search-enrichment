# AI-Powered Knowledge Base Search & Enrichment

This project implements a backend for document ingestion, semantic search, and Q&A using AI.

## Features

- Document ingestion pipeline (TXT files only)
- Semantic search with FAISS
- Q&A via Groq API with Llama
- API for completeness check

## Design Decisions

- **Tech Stack**: Python/FastAPI for async API, sentence-transformers for local embeddings, FAISS for vector search, SQLite for raw storage.
- **Architecture**: Modular with separate modules for DB, ingestion, search, QA, API.
- **Efficiency**: Batch embeddings, in-memory FAISS for fast queries.

## Trade-offs

- Local tools (no cloud) for 24h speed; basic DB (not scalable); no advanced Q&A model; no security (auth/API keys) implemented; used free Groq API with Llama instead of paid GPT-3.5 (may have lower quality or rate limits); only TXT files supported (skipped PDF/other formats for time).

## Loom Demo
- Watch the demo video: [Loom Demo](https://www.loom.com/share/595213f494674572b04e37196c69e764?sid=9ca7ff98-5060-4c6c-a3b9-47ece11b1bd5)

## How to Run

1. Install dependencies: `pip install -r requirements.txt`
2. Set up environment: Copy `.env.example` to `.env` and add your Groq API key, or export `GROQ_API_KEY=your_key`
3. Run server: `uvicorn src.main:app --reload`

## How to Test

- Access interactive API documentation at `http://localhost:8000/docs`
- Upload TXT via POST /ingest
- Search: GET /search?query=...
- Q&A: POST /qa with {"query": "..."}
- Completeness: POST /check_completeness with {"query": "..."}
- Run tests: `pytest`

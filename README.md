# LendRight RAG Chatbot — Backend

A **Retrieval-Augmented Generation (RAG)** chatbot backend built with:

| Layer | Technology |
|---|---|
| API framework | FastAPI |
| Language model | Gemini 2.5 Flash (via `google-genai` SDK) |
| Retrieval | In-memory TF-IDF vector store (no external DB) |
| Knowledge base | Synthetic LendRight Financial data (`data.py`) |
| Accounts & personalization | SQLite (SQLAlchemy) + JWT bearer auth |

---

## Project Structure

```
MicroCredit/
├── backend/
│   ├── static/
│   │   └── index.html    # Frontend UI, served by FastAPI at GET /
│   ├── main.py            # FastAPI app + API routes (incl. /auth/*)
│   ├── rag_engine.py      # RAG pipeline (retrieve → augment → generate)
│   ├── vector_store.py    # Lightweight TF-IDF vector store
│   ├── data.py            # Synthetic loan company knowledge base
│   ├── database.py        # SQLite engine/session setup
│   ├── models.py          # User + LoanAccount SQLAlchemy models
│   ├── security.py        # Password hashing + JWT auth dependencies
│   ├── loan_service.py    # Loan schedule/payment business logic
│   ├── loan_seed.py       # Synthetic loan generator for new accounts
│   ├── requirements.txt
│   ├── .env.example       # Environment variable template
│   └── microcredit.db     # SQLite database (auto-created on startup)
└── README.md
```

The frontend and API are served from the same FastAPI app/port — visiting
`http://localhost:8000/` loads the UI, which talks to the API on the same
origin.

---

## Quick Start

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY and JWT_SECRET_KEY
```

Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
`JWT_SECRET_KEY` signs auth tokens — generate one with
`python -c "import secrets; print(secrets.token_hex(32))"`. Keep it stable
across restarts, or previously issued tokens stop validating.

### 3. Run the server

```bash
# from backend/
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/` for the app UI.  
Interactive API docs: `http://localhost:8000/docs`

---

## API Endpoints

### `GET /health`
Returns service status and the number of indexed document chunks.

### `POST /auth/register`
Create an account (`email`, `password`, `full_name`, optional `phone`). Seeds a
synthetic loan account for the new user and returns a bearer access token.

### `POST /auth/login`
Exchange `email` + `password` for a bearer access token.

### `GET /auth/me`
Returns the signed-in user's profile and loan summary. Requires
`Authorization: Bearer <token>`.

### `POST /chat`
Main RAG endpoint. Request body:

```json
{
  "question": "What is the minimum credit score for a personal loan?",
  "history": [],
  "top_k": 4
}
```

Response:

```json
{
  "answer": "To qualify for a LendRight personal loan, you need a minimum credit score of 600...",
  "sources": [
    {
      "id": "personal_loan_2",
      "category": "Personal Loans",
      "score": 0.82,
      "text": "..."
    }
  ],
  "latency_ms": 1234.5
}
```

The `history` array supports multi-turn conversations — pass previous turns as `{"role": "user"|"assistant", "content": "..."}` objects.

Pass `Authorization: Bearer <token>` (obtained from `/auth/login` or
`/auth/register`) to let the assistant answer questions about the caller's own
loan account (balance, next payment, etc.). It's optional — anonymous requests
still get general answers.

### `GET /documents`
Lists all 24 indexed document chunks from the knowledge base.

---

## How RAG Works Here

1. **Retrieval** — The user's question is converted to a TF-IDF vector and compared (cosine similarity) against all document chunks. The top-`k` most relevant chunks are selected.
2. **Augmentation** — The retrieved chunks are injected into the prompt as `Context information`.
3. **Generation** — Gemini 2.5 Flash reads the context and the question, then generates a grounded answer following the system prompt rules.

---

## Knowledge Base Categories

| Category | # Chunks |
|---|---|
| Company Overview | 2 |
| Personal Loans | 3 |
| Home Equity Loans | 2 |
| Auto Loans | 2 |
| Student Loan Refinancing | 2 |
| Small-Business Loans | 2 |
| Application Process | 3 |
| Rates and Fees | 3 |
| Repayment and Hardship | 2 |
| Security and Privacy | 2 |
| Customer Support | 2 |
| **Total** | **25** |

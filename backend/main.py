"""
FastAPI backend for the LendRight RAG Chatbot.

Endpoints
---------
GET  /             → health check
GET  /health       → health check (JSON)
POST /chat         → main RAG chat endpoint
GET  /documents    → list all indexed documents (for debugging)
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import rag_engine

# ── Load environment variables ────────────────────────────────────────────────

load_dotenv()


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Previous conversation turns (oldest first).",
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of document chunks to retrieve.",
    )


class SourceChunk(BaseModel):
    id: str
    category: str
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    documents_indexed: int


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up: the vector store is already built at import time inside rag_engine
    yield


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="LendRight RAG Chatbot API",
    description=(
        "A retrieval-augmented generation chatbot backed by Gemini 2.5 Flash "
        "that answers questions about LendRight Financial's loan products."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "LendRight RAG Chatbot API is running. See /docs for usage."}


@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health():
    """Return service health and basic metadata."""
    return HealthResponse(
        status="ok",
        model="gemini-2.5-flash",
        documents_indexed=len(rag_engine.DOCUMENTS),
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a question and receive a grounded answer from Gemini 2.5 Flash.

    The *history* field lets you maintain multi-turn conversations: pass the
    previous turns (role + content) and the model will use them as context.
    """
    history = (
        [{"role": m.role, "content": m.content} for m in request.history]
        if request.history
        else None
    )

    try:
        t0 = time.perf_counter()
        result = rag_engine.answer(
            question=request.question,
            chat_history=history,
            k=request.top_k,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Upstream model error: {exc}",
        )

    return ChatResponse(
        answer=result["answer"],
        sources=[SourceChunk(**s) for s in result["sources"]],
        latency_ms=latency_ms,
    )


@app.get("/documents", tags=["Utility"])
async def list_documents():
    """Return all indexed document chunks (useful for debugging)."""
    return {
        "total": len(rag_engine.DOCUMENTS),
        "documents": rag_engine.DOCUMENTS,
    }

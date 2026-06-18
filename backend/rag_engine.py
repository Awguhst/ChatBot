"""
RAG engine: retrieves relevant document chunks and sends them to Gemini 2.5 Flash
via the Google GenAI SDK to generate a grounded answer.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any

from google import genai
from google.genai import types

from vector_store import TFIDFVectorStore
from data import DOCUMENTS

# ── Initialise the vector store once at import time ───────────────────────────

_store = TFIDFVectorStore()
_store.add_documents(DOCUMENTS)

# ── Gemini client (lazy-initialised) ─────────────────────────────────────────

_client: genai.Client | None = None

def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please add it to your .env file."
            )
        _client = genai.Client(api_key=api_key)
    return _client


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Ти си полезен асистент за обслужване на клиенти на Кредит Про ЕАД — \
българска небанкова финансова институция, която предоставя бързи потребителски \
кредити. Разполагаш с авторитетна информация за продуктите, лихвите, таксите, \
условията за допустимост, процесите на кандидатстване и регулаторната рамка на \
компанията, предоставена ти като контекстни пасажи по-долу.

ВАЖНО: Отговаряй ВИНАГИ на български език, независимо на какъв език е зададен въпросът.

Правила:
1. Отговаряй САМО въз основа на предоставения контекст. Не измисляй факти.
2. Ако отговорът не е обхванат от контекста, кажи го ясно и учтиво, \
   и предложи на клиента да се свърже с Кредит Про на национален телефон \
   0700 14 200 или чрез уебсайта creditpro.bg.
3. Бъди кратък, приятелски настроен и професионален.
4. При цитиране на числа (лихвени проценти, такси, суми) бъди точен. \
   Всички парични суми посочвай в евро (€).
5. Не давай лични финансови или правни съвети — насочвай потребителите към \
   съответния канал за обслужване на Кредит Про.
6. При посочване на български нормативни актове (ЗПК, ЗМИП, ОРЗД/GDPR) използвай \
   точните им наименования така, както са посочени в контекста.
7. Ако получиш блок "Лична информация за удостоверения клиент", той се отнася \
   единствено за клиента, който в момента разговаря с теб — използвай го за да \
   отговаряш на въпроси за неговия кредит (баланс, вноски, дата на плащане). \
   Тази информация е поверителна: никога не я споменавай, ако въпросът не я \
   засяга, и не я разкривай по друг начин извън отговора на самия клиент.
"""


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """Return the top-k document chunks most relevant to *query*."""
    return _store.similarity_search(query, k=k)


def build_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] ({chunk.get('category', 'General')})\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def answer(
    question: str,
    chat_history: List[Dict[str, str]] | None = None,
    k: int = 4,
    user_context: str | None = None,
) -> Dict[str, Any]:
    """
    Full RAG pipeline:
      1. Retrieve relevant chunks.
      2. Build an augmented prompt.
      3. Call Gemini 2.5 Flash.
      4. Return the answer + retrieved sources.

    *chat_history* is a list of {"role": "user"|"assistant", "content": "..."} dicts
    representing the conversation so far (NOT including the current question).

    *user_context* is an optional, pre-formatted block of the signed-in customer's
    own account data (e.g. loan balance, next payment date). Only ever pass the
    data belonging to the customer making this request.
    """
    # 1. Retrieve
    chunks = retrieve(question, k=k)
    context = build_context(chunks)

    # 2. Build the full user message with injected context
    personal_block = (
        f"Лична информация за удостоверения клиент:\n{user_context}\n\n"
        if user_context
        else ""
    )
    user_message = (
        f"{personal_block}"
        f"Context information:\n{context}\n\n"
        f"Customer question: {question}"
    )

    # 3. Build message list for the API
    messages: List[types.ContentDict] = []

    if chat_history:
        for turn in chat_history:
            role = "user" if turn["role"] == "user" else "model"
            messages.append({"role": role, "parts": [{"text": turn["content"]}]})

    messages.append({"role": "user", "parts": [{"text": user_message}]})

    # 4. Call Gemini 2.5 Flash
    client = _get_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )

    answer_text = response.text or ""

    return {
        "answer": answer_text,
        "sources": [
            {
                "id": c["id"],
                "category": c.get("category", ""),
                "score": c.get("score", 0.0),
                "text": c["text"],
            }
            for c in chunks
        ],
    }
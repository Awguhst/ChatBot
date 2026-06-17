"""
Lightweight in-memory vector store using TF-IDF embeddings and cosine similarity.
No external vector database is required — everything runs in-process.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Dict, Any


# ── Tokenisation ──────────────────────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    """Lower-case, strip punctuation (Unicode-aware), split on whitespace.

    Uses \\w with re.UNICODE so Cyrillic, Latin and other scripts are all
    preserved — the original [^a-z0-9\\s] pattern stripped every Cyrillic
    character, producing zero-length token lists and zeroed cosine scores.
    """
    text = text.lower()
    # Keep Unicode word characters (letters + digits) and whitespace; drop the rest.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    # Exclude bare underscores that \w includes
    return [t for t in text.split() if t != "_"]


# ── TF-IDF vector store ───────────────────────────────────────────────────────

class TFIDFVectorStore:
    """
    Stores documents as TF-IDF vectors and supports cosine-similarity retrieval.
    """

    def __init__(self) -> None:
        self._documents: List[Dict[str, Any]] = []
        self._tfidf_matrix: List[Dict[str, float]] = []   # one dict per document
        self._idf: Dict[str, float] = {}
        self._vocab: set = set()

    # ── Building the index ────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add a list of documents and rebuild the TF-IDF index.

        Each document dict must contain at least:
            - "id":   unique string identifier
            - "text": the raw text content
        Optional keys (e.g. "category") are preserved and returned with results.
        """
        self._documents = documents
        tokenised = [_tokenise(doc["text"]) for doc in documents]

        # Build vocabulary
        self._vocab = set(tok for tokens in tokenised for tok in tokens)

        # IDF: log((N + 1) / (df + 1)) + 1  (smoothed)
        N = len(documents)
        df: Dict[str, int] = Counter()
        for tokens in tokenised:
            for tok in set(tokens):
                df[tok] += 1

        self._idf = {
            tok: math.log((N + 1) / (df[tok] + 1)) + 1
            for tok in self._vocab
        }

        # TF-IDF vectors (normalised)
        self._tfidf_matrix = []
        for tokens in tokenised:
            tf = Counter(tokens)
            total = len(tokens) or 1
            vec = {
                tok: (count / total) * self._idf[tok]
                for tok, count in tf.items()
            }
            self._tfidf_matrix.append(self._normalise(vec))

    @staticmethod
    def _normalise(vec: Dict[str, float]) -> Dict[str, float]:
        """L2-normalise a sparse vector represented as a dict."""
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm == 0:
            return vec
        return {k: v / norm for k, v in vec.items()}

    # ── Query ─────────────────────────────────────────────────────────────────

    def _query_vector(self, query: str) -> Dict[str, float]:
        """Convert a query string to a normalised TF-IDF vector."""
        tokens = _tokenise(query)
        tf = Counter(tokens)
        total = len(tokens) or 1
        vec = {
            tok: (count / total) * self._idf.get(tok, 0)
            for tok, count in tf.items()
            if tok in self._vocab
        }
        return self._normalise(vec)

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        """Dot product of two already-normalised vectors (= cosine similarity)."""
        # Iterate over the shorter dict for speed
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b[k] for k, v in a.items() if k in b)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Return the top-k most relevant documents for *query*.

        Returns a list of dicts, each being the original document dict augmented
        with a "score" key containing the cosine similarity value.
        """
        if not self._documents:
            return []

        q_vec = self._query_vector(query)
        scores = [
            (self._cosine(q_vec, doc_vec), idx)
            for idx, doc_vec in enumerate(self._tfidf_matrix)
        ]
        scores.sort(reverse=True)

        results = []
        for score, idx in scores[:k]:
            doc = dict(self._documents[idx])   # shallow copy
            doc["score"] = round(score, 4)
            results.append(doc)
        return results
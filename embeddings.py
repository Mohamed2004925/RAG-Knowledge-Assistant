"""Gemini document embeddings for the learning app's current chunk set."""

from __future__ import annotations

import hashlib
import json

from google import genai
from google.genai import types


EMBEDDING_MODEL = "gemini-embedding-001"
BATCH_SIZE = 16


def fingerprint_chunks(source: str, chunks: list[dict], model: str = EMBEDDING_MODEL) -> str:
    """Identify the exact source/chunk/model combination represented by vectors."""
    payload = {
        "model": model,
        "source": source,
        "chunks": [(chunk["chunk_id"], chunk["text"]) for chunk in chunks],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def embed_documents(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """Create retrieval-document vectors in small batches, preserving input order."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        if not response.embeddings or len(response.embeddings) != len(batch):
            raise RuntimeError("Gemini returned an unexpected number of embeddings.")
        vectors.extend([embedding.values for embedding in response.embeddings])
    return vectors


def embed_query(client: genai.Client, question: str) -> list[float]:
    """Create one query vector using the matching retrieval-query task type."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    if not response.embeddings:
        raise RuntimeError("Gemini returned no query embedding.")
    return response.embeddings[0].values


def embed_queries(client: genai.Client, questions: list[str]) -> list[list[float]]:
    """Create query vectors in batches while preserving question order."""
    vectors: list[list[float]] = []
    for start in range(0, len(questions), BATCH_SIZE):
        batch = questions[start : start + BATCH_SIZE]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        if not response.embeddings or len(response.embeddings) != len(batch):
            raise RuntimeError("Gemini returned an unexpected number of query embeddings.")
        vectors.extend([embedding.values for embedding in response.embeddings])
    return vectors

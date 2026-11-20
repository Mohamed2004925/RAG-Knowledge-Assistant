"""Transparent in-memory vector ranking for Chapter 5."""

from math import sqrt


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Measure the angle-based similarity of two vectors."""
    if len(left) != len(right):
        raise ValueError("Vectors must have the same number of dimensions.")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot_product = sum(a * b for a, b in zip(left, right))
    return dot_product / (left_norm * right_norm)


def rank_chunks(
    query_vector: list[float],
    chunks: list[dict],
    document_vectors: list[list[float]],
    top_k: int = 3,
) -> list[dict]:
    """Return the top matching chunks, preserving their source metadata."""
    if len(chunks) != len(document_vectors):
        raise ValueError("Each chunk must have exactly one document embedding.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    scored = [
        {
            "chunk": chunk,
            "score": cosine_similarity(query_vector, vector),
        }
        for chunk, vector in zip(chunks, document_vectors)
    ]
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]

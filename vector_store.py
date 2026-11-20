"""A small SQLite-backed embedding index for local learning and demos."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from retrieval import rank_chunks


DB_PATH = Path(__file__).parent / "data" / "rag_index.sqlite3"


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def _connection(db_path: Path = DB_PATH):
    """Commit successful operations, roll back failures, and always close SQLite."""
    connection = _connect(db_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_store(db_path: Path = DB_PATH) -> None:
    """Create source and chunk tables if they do not exist yet."""
    with _connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                source_name TEXT NOT NULL,
                page_label TEXT NOT NULL,
                chunk_number INTEGER NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                FOREIGN KEY(source_id) REFERENCES sources(source_id)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id)"
        )


def source_identity(source_name: str, pages: list[dict[str, str]]) -> tuple[str, str]:
    """Return stable source ID and content hash for an extracted document."""
    content = "\n".join(
        f"{page.get('label', '')}\n{page.get('text', '')}" for page in pages
    )
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    source_id = hashlib.sha256(
        f"{source_name}\0{content_hash}".encode("utf-8")
    ).hexdigest()[:24]
    return source_id, content_hash


def upsert_document(
    source_id: str,
    source_name: str,
    content_hash: str,
    chunks: list[dict],
    vectors: list[list[float]],
    embedding_model: str,
    db_path: Path = DB_PATH,
) -> None:
    """Replace a source's indexed chunks and vectors in one SQLite transaction."""
    if len(chunks) != len(vectors):
        raise ValueError("Each chunk must have exactly one embedding vector.")
    if not chunks:
        raise ValueError("Cannot index a source with no text chunks.")

    initialize_store(db_path)
    with _connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO sources (source_id, source_name, content_hash, indexed_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(source_id) DO UPDATE SET
                source_name=excluded.source_name,
                content_hash=excluded.content_hash,
                indexed_at=CURRENT_TIMESTAMP
            """,
            (source_id, source_name, content_hash),
        )
        connection.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
        connection.executemany(
            """
            INSERT INTO chunks (
                chunk_id, source_id, source_name, page_label, chunk_number,
                start_char, end_char, text, embedding_json, embedding_model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    f"{source_id}::{chunk['chunk_id']}",
                    source_id,
                    source_name,
                    str(chunk["page"]),
                    int(chunk["chunk_number"]),
                    int(chunk["start_char"]),
                    int(chunk["end_char"]),
                    str(chunk["text"]),
                    json.dumps(vector),
                    embedding_model,
                )
                for chunk, vector in zip(chunks, vectors)
            ],
        )


def list_sources(db_path: Path = DB_PATH) -> list[dict]:
    initialize_store(db_path)
    with _connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT s.source_id, s.source_name, s.indexed_at,
                   COUNT(c.chunk_id) AS chunk_count
            FROM sources s
            LEFT JOIN chunks c ON c.source_id = s.source_id
            GROUP BY s.source_id
            ORDER BY s.source_name COLLATE NOCASE
            """
        ).fetchall()
    return [dict(row) for row in rows]


def load_chunks(source_ids: list[str] | None = None, db_path: Path = DB_PATH) -> list[dict]:
    initialize_store(db_path)
    query = "SELECT * FROM chunks"
    parameters: tuple = ()
    if source_ids is not None:
        if not source_ids:
            return []
        placeholders = ", ".join("?" for _ in source_ids)
        query += f" WHERE source_id IN ({placeholders})"
        parameters = tuple(source_ids)
    query += " ORDER BY source_name COLLATE NOCASE, page_label, chunk_number"

    with _connection(db_path) as connection:
        rows = connection.execute(query, parameters).fetchall()
    chunks = []
    for row in rows:
        item = dict(row)
        item["page"] = item.pop("page_label")
        item["embedding"] = json.loads(item.pop("embedding_json"))
        item["chunk_number"] = int(item["chunk_number"])
        chunks.append(item)
    return chunks


def search_index(
    query_vector: list[float],
    top_k: int = 3,
    source_ids: list[str] | None = None,
    db_path: Path = DB_PATH,
) -> list[dict]:
    """Brute-force cosine search over stored vectors, suitable for a small demo."""
    rows = load_chunks(source_ids, db_path)
    chunks = [
        {
            "chunk_id": row["chunk_id"],
            "source_id": row["source_id"],
            "source": row["source_name"],
            "page": row["page"],
            "chunk_number": row["chunk_number"],
            "start_char": row["start_char"],
            "end_char": row["end_char"],
            "text": row["text"],
        }
        for row in rows
    ]
    vectors = [row["embedding"] for row in rows]
    return rank_chunks(query_vector, chunks, vectors, top_k=top_k)


def delete_source(source_id: str, db_path: Path = DB_PATH) -> None:
    """Remove one source and all of its chunks from the local index."""
    initialize_store(db_path)
    with _connection(db_path) as connection:
        connection.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
        connection.execute("DELETE FROM sources WHERE source_id = ?", (source_id,))

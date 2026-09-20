"""Question fingerprint vs stored fingerprints. Smaller distance = closer match."""

from __future__ import annotations

from dataclasses import dataclass

from expense_rag.embeddings import embed_text
from expense_rag.models import PolicyChunk

RETRIEVE_LIMIT = 3


@dataclass(frozen=True)
class RetrievedChunk:
    section: str
    distance: float
    document: str
    version: str
    section_number: str
    section_title: str
    text: str
    chunk_id: str


def cosine_distance(left: list[float], right: list[float]) -> float:
    """Cosine distance: 1 - cosine similarity. Matches pgvector's <=> operator."""
    if len(left) != len(right):
        raise ValueError("vectors must be the same length")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        raise ValueError("cannot compare a zero vector")
    similarity = dot / (left_norm * right_norm)
    return float(1 - similarity)


def rank_chunks(
    query_vector: list[float],
    chunks: list[PolicyChunk],
    limit: int = RETRIEVE_LIMIT,
) -> list[RetrievedChunk]:
    scored: list[RetrievedChunk] = []
    for chunk in chunks:
        if chunk.embedding is None:
            continue
        scored.append(
            RetrievedChunk(
                section=f"{chunk.section}. {chunk.section_title}",
                distance=cosine_distance(query_vector, chunk.embedding),
                document=chunk.document,
                version=chunk.version,
                section_number=chunk.section,
                section_title=chunk.section_title,
                text=chunk.text,
                chunk_id=chunk.chunk_id,
            )
        )
    scored.sort(key=lambda item: item.distance)
    return scored[:limit]


def retrieve_sql(
    connection,
    query_vector: list[float],
    limit: int = RETRIEVE_LIMIT,
) -> list[RetrievedChunk]:
    rows = connection.execute(
        """
        SELECT
            chunk_id,
            document,
            version,
            section,
            section_title,
            text,
            embedding <=> %s::vector AS distance
        FROM policy_chunks
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector ASC
        LIMIT %s
        """,
        (query_vector, query_vector, limit),
    ).fetchall()
    return [
        RetrievedChunk(
            section=f"{row[3]}. {row[4]}",
            distance=float(row[6]),
            document=row[1],
            version=row[2],
            section_number=row[3],
            section_title=row[4],
            text=row[5],
            chunk_id=row[0],
        )
        for row in rows
    ]


def retrieve(question: str, connection=None, limit: int = RETRIEVE_LIMIT) -> list[RetrievedChunk]:
    query_vector = embed_text(question)
    if connection is not None:
        return retrieve_sql(connection, query_vector, limit)
    from expense_rag.store import connect

    with connect() as owned:
        return retrieve_sql(owned, query_vector, limit)

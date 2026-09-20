"""PostgreSQL helpers for policy chunks."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from expense_rag.models import PolicyChunk

MIGRATION = Path(__file__).resolve().parents[2] / "migrations" / "001_create_chunks.sql"


def database_url() -> str:
    url = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5432/expense_rag")
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


def connect() -> psycopg.Connection:
    connection = psycopg.connect(database_url())
    register_vector(connection)
    return connection


def apply_schema(connection: psycopg.Connection) -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    statements = [part.strip() for part in sql.split(";") if part.strip()]
    for statement in statements:
        connection.execute(statement)
    connection.commit()


def upsert_chunks(connection: psycopg.Connection, chunks: list[PolicyChunk]) -> None:
    for chunk in chunks:
        connection.execute(
            """
            INSERT INTO policy_chunks (
                chunk_id, document, version, section, section_title, text, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                document = EXCLUDED.document,
                version = EXCLUDED.version,
                section = EXCLUDED.section,
                section_title = EXCLUDED.section_title,
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding
            """,
            (
                chunk.chunk_id,
                chunk.document,
                chunk.version,
                chunk.section,
                chunk.section_title,
                chunk.text,
                chunk.embedding,
            ),
        )
    connection.commit()


def count_chunks(connection: psycopg.Connection) -> int:
    row = connection.execute("SELECT COUNT(*) FROM policy_chunks").fetchone()
    assert row is not None
    return int(row[0])

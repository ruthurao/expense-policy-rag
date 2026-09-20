from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (ROOT / "migrations" / "001_create_chunks.sql").read_text(encoding="utf-8")


def test_schema_enables_pgvector() -> None:
    assert "CREATE EXTENSION IF NOT EXISTS vector" in SCHEMA


def test_schema_stores_text_vector_and_metadata() -> None:
    required = [
        "chunk_id",
        "document",
        "version",
        "section",
        "section_title",
        "text",
        "embedding",
        "vector(384)",
    ]
    for item in required:
        assert item in SCHEMA


def test_chunk_id_is_primary_key() -> None:
    assert "chunk_id TEXT PRIMARY KEY" in SCHEMA

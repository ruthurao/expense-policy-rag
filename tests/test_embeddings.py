from dataclasses import replace

from expense_rag.embeddings import EMBEDDING_DIM, embed_text
from expense_rag.ingest import ingest
from expense_rag.models import PolicyChunk


def test_embed_text_returns_full_vector(monkeypatch) -> None:
    monkeypatch.setattr(
        "expense_rag.embeddings._model",
        lambda: _FakeModel(),
    )
    vector = embed_text("Employees may claim up to $65 per day for meals.")
    assert len(vector) == EMBEDDING_DIM
    assert all(isinstance(value, float) for value in vector)


def test_ingest_stores_text_and_embedding(monkeypatch) -> None:
    stored: list[PolicyChunk] = []

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("expense_rag.ingest.embed_text", lambda text: [0.1] * EMBEDDING_DIM)
    monkeypatch.setattr("expense_rag.ingest.connect", lambda: FakeConnection())
    monkeypatch.setattr("expense_rag.ingest.apply_schema", lambda connection: None)
    monkeypatch.setattr(
        "expense_rag.ingest.upsert_chunks",
        lambda connection, chunks: stored.extend(chunks),
    )
    monkeypatch.setattr("expense_rag.ingest.count_chunks", lambda connection: len(stored))

    count = ingest()
    assert count == 6
    assert all(chunk.embedding is not None for chunk in stored)
    assert all(len(chunk.embedding or []) == EMBEDDING_DIM for chunk in stored)
    assert stored[0].section_title == "Meals"


class _FakeModel:
    def encode(self, text: str, normalize_embeddings: bool = True):
        class _Vector(list):
            def tolist(self):
                return list(self)

        return _Vector([0.01] * EMBEDDING_DIM)


def test_chunk_can_hold_embedding() -> None:
    chunk = PolicyChunk(
        chunk_id="expense-policy:v2.0:section-1",
        document="Employee Expense Policy",
        version="2.0",
        section="1",
        section_title="Meals",
        text="Employees may claim up to $65 per day.",
    )
    with_vector = replace(chunk, embedding=[0.12, -0.31, 0.44])
    assert with_vector.embedding == [0.12, -0.31, 0.44]
    assert chunk.embedding is None

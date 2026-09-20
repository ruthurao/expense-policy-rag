"""Read the policy, embed each chunk, and store rows in Postgres."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from expense_rag.chunking import load_policy
from expense_rag.embeddings import embed_text
from expense_rag.store import apply_schema, connect, count_chunks, upsert_chunks


def ingest(policy_path: Path | None = None) -> int:
    chunks = load_policy(policy_path)
    embedded = [
        replace(chunk, embedding=embed_text(chunk.text))
        for chunk in chunks
    ]
    with connect() as connection:
        apply_schema(connection)
        upsert_chunks(connection, embedded)
        return count_chunks(connection)


def main() -> None:
    count = ingest()
    print(f"ingested {count} chunks")


if __name__ == "__main__":
    main()

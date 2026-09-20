"""Ask a question: retrieve evidence, then generate a grounded answer."""

from __future__ import annotations

from typing import Any

from expense_rag.generate import generate_answer
from expense_rag.retrieve import RetrievedChunk, retrieve


def _citation(generated, chunks: list[RetrievedChunk]) -> dict[str, str] | None:
    if generated.citation_section is None:
        return None
    match = next((chunk for chunk in chunks if chunk.section == generated.citation_section), None)
    if match is None:
        return None
    return {
        "document": match.document,
        "version": match.version,
        "section": match.section,
    }


def ask(question: str, retrieve_fn=retrieve, complete=None) -> dict[str, Any]:
    chunks = retrieve_fn(question)
    generated = generate_answer(question, chunks, complete=complete)
    return {
        "answer": generated.answer,
        "citation": _citation(generated, chunks),
        "retrieved_chunks": [
            {"section": chunk.section, "distance": chunk.distance}
            for chunk in chunks
        ],
    }

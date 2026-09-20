"""One stored policy section: text, labels, and optional embedding."""

from __future__ import annotations


from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyChunk:
    chunk_id: str
    document: str
    version: str
    section: str
    section_title: str
    text: str
    embedding: list[float] | None = None

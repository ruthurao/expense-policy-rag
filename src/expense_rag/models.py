"""Stored chunk shape. Embeddings are added in a later phase."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyChunk:
    chunk_id: str
    document: str
    version: str
    section: str
    section_title: str
    text: str

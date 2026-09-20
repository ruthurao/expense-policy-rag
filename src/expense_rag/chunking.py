"""Split the expense policy at numbered section headings."""

from __future__ import annotations

import re
from pathlib import Path

from expense_rag.models import PolicyChunk

TITLE_RE = re.compile(
    r"^#\s+(?P<document>.+?)\s+[—-]\s+Version\s+(?P<version>\S+)\s*$",
    re.MULTILINE,
)
HEADING_RE = re.compile(
    r"^##\s+(?P<section>\d+)\.\s+(?P<title>.+?)\s*$",
    re.MULTILINE,
)

EXPECTED_SECTION_COUNT = 6


def split_policy(markdown: str) -> list[PolicyChunk]:
    """Return one chunk per numbered heading. Never split mid-sentence."""
    title_match = TITLE_RE.search(markdown)
    if title_match is None:
        raise ValueError("policy.md must start with '# Title — Version X.Y'")

    document = title_match.group("document").strip()
    version = title_match.group("version").strip()
    headings = list(HEADING_RE.finditer(markdown))
    if len(headings) != EXPECTED_SECTION_COUNT:
        raise ValueError(
            f"expected {EXPECTED_SECTION_COUNT} numbered sections, found {len(headings)}"
        )

    chunks: list[PolicyChunk] = []
    for index, heading in enumerate(headings):
        body_start = heading.end()
        body_end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        text = markdown[body_start:body_end].strip()
        if not text:
            raise ValueError(f"section {heading.group('section')} has an empty body")
        if "##" in text:
            raise ValueError("section body contains a heading; refusing a mid-section split")

        section = heading.group("section")
        section_title = heading.group("title").strip()
        chunks.append(
            PolicyChunk(
                chunk_id=f"expense-policy:v{version}:section-{section}",
                document=document,
                version=version,
                section=section,
                section_title=section_title,
                text=text,
            )
        )
    return chunks


def load_policy(path: Path | None = None) -> list[PolicyChunk]:
    policy_path = path or Path(__file__).resolve().parents[2] / "policy.md"
    return split_policy(policy_path.read_text(encoding="utf-8"))

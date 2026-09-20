"""Turn retrieved policy excerpts into a grounded, cited answer."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from expense_rag.retrieve import RetrievedChunk

REFUSE_TEXT = "The provided policy does not answer this question."

INSTRUCTION = """Answer the question using only the policy excerpts below.
Include the section that supports your answer.
If the excerpts do not contain the answer, respond:
"The provided policy does not answer this question."

Return JSON with keys:
- "answer": string
- "citation_section": string like "1. Meals", or null if the excerpts do not answer the question
Do not add information that is not in the excerpts.
"""


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    citation_section: str | None


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    excerpts = []
    for chunk in chunks:
        excerpts.append(
            f"Section {chunk.section}\n{chunk.text}"
        )
    joined = "\n\n".join(excerpts) if excerpts else "(no excerpts)"
    return (
        f"{INSTRUCTION}\n\n"
        f"Question: {question}\n\n"
        f"Policy excerpts:\n{joined}"
    )


def _openai_complete(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )
    response = client.chat.completions.create(
        model=os.environ.get("GENERATION_MODEL", "gpt-4o-mini"),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("generation model returned an empty response")
    return content


def parse_generation(raw: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
    data = json.loads(raw)
    answer = str(data.get("answer", "")).strip()
    citation_section = data.get("citation_section")
    if isinstance(citation_section, str):
        citation_section = citation_section.strip() or None
    else:
        citation_section = None

    retrieved_sections = {chunk.section for chunk in chunks}
    refused = (
        not answer
        or answer == REFUSE_TEXT
        or citation_section is None
        or citation_section not in retrieved_sections
    )
    if refused:
        return GeneratedAnswer(answer=REFUSE_TEXT, citation_section=None)
    return GeneratedAnswer(answer=answer, citation_section=citation_section)


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    complete=None,
) -> GeneratedAnswer:
    prompt = build_prompt(question, chunks)
    raw = (complete or _openai_complete)(prompt)
    return parse_generation(raw, chunks)

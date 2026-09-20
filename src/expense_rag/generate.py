"""Turn retrieved policy excerpts into a grounded, cited answer."""

from __future__ import annotations

import json
import os
import re
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

STOPWORDS = {
    "does", "the", "company", "what", "need", "can", "book", "claim",
    "how", "much", "spend", "each", "day", "my", "costs", "do", "for",
    "and", "are", "is", "a", "an", "to", "on", "of", "i",
    "reimburse", "reimbursable",
}

SYNONYMS = {
    "food": ["meal", "meals"],
    "first": ["economy", "airfare", "business"],
    "class": ["airfare", "economy"],
    "airfare": ["airfare", "economy", "business"],
    "hotel": ["hotel", "hotels", "manager"],
    "receipt": ["receipt", "receipts"],
    "taxi": ["taxi", "receipt", "receipts"],
    "limousine": ["luxury", "vehicle", "upgrade"],
    "limo": ["luxury", "vehicle"],
    "upgrade": ["luxury", "upgrade"],
}


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    citation_section: str | None


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    excerpts = [f"Section {chunk.section}\n{chunk.text}" for chunk in chunks]
    joined = "\n\n".join(excerpts) if excerpts else "(no excerpts)"
    return f"{INSTRUCTION}\n\nQuestion: {question}\n\nPolicy excerpts:\n{joined}"


def _question_terms(question: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", question.lower())
    return [word for word in words if word not in STOPWORDS and len(word) > 2]


def _chunk_score(question: str, chunk: RetrievedChunk) -> int:
    blob = f"{chunk.section_title} {chunk.text}".lower()
    score = 0
    for word in _question_terms(question):
        needles = SYNONYMS.get(word, [word])
        if any(needle in blob for needle in needles):
            score += 1
    return score


def extractive_answer(question: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
    scored = [(chunk, _chunk_score(question, chunk)) for chunk in chunks]
    supported = [item for item in scored if item[1] > 0]
    if not supported:
        return GeneratedAnswer(answer=REFUSE_TEXT, citation_section=None)
    best, _ = sorted(supported, key=lambda item: (-item[1], item[0].distance))[0]
    answer = " ".join(best.text.split())
    return GeneratedAnswer(answer=answer, citation_section=best.section)


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


def generate_answer(question: str, chunks: list[RetrievedChunk], complete=None) -> GeneratedAnswer:
    if complete is None and not os.environ.get("OPENAI_API_KEY"):
        return extractive_answer(question, chunks)
    prompt = build_prompt(question, chunks)
    raw = (complete or _openai_complete)(prompt)
    return parse_generation(raw, chunks)

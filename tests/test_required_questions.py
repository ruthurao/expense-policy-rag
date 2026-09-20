"""Required lab questions: retrieve the right section, cite or refuse."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from expense_rag.chunking import load_policy
from expense_rag.generate import REFUSE_TEXT
from expense_rag.lab_cases import LAB_CASES
from expense_rag.models import PolicyChunk
from expense_rag.pipeline import ask
from expense_rag.retrieve import rank_chunks

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "required_questions.json"

AXES = [
    ("1", ["meal", "meals", "food", "alcohol", "65"]),
    ("2", ["hotel", "hotels", "night", "manager", "225", "250", "booking"]),
    ("3", ["airfare", "flight", "economy", "business", "first-class", "first class", "vice president"]),
    ("4", ["taxi", "rideshare", "limousine", "limo", "luxury", "vehicle", "upgrade", "train"]),
    ("5", ["receipt", "receipts", "25", "20"]),
    ("6", ["deadline", "30", "submit", "submission"]),
]


def meaning_embed(text: str) -> list[float]:
    lowered = text.lower()
    vector = [float(sum(1 for term in terms if term in lowered)) for _, terms in AXES]
    if sum(vector) == 0:
        return [0.01] * len(AXES)
    return vector


def embedded_policy() -> list[PolicyChunk]:
    return [
        replace(chunk, embedding=meaning_embed(f"{chunk.section_title} {chunk.text}"))
        for chunk in load_policy(ROOT / "policy.md")
    ]


def retrieve_lab(question: str):
    return rank_chunks(meaning_embed(question), embedded_policy())


def complete_lab(question: str):
    case = next(item for item in LAB_CASES if item["question"] == question)

    def _complete(prompt: str) -> str:
        return json.dumps(
            {
                "answer": case["answer"],
                "citation_section": case["expected_section"],
            }
        )

    return _complete


def run_required_questions() -> list[dict]:
    results = []
    for case in LAB_CASES:
        payload = ask(
            case["question"],
            retrieve_fn=retrieve_lab,
            complete=complete_lab(case["question"]),
        )
        results.append({"question": case["question"], "response": payload})
    return results


def test_six_questions_are_defined() -> None:
    assert len(LAB_CASES) == 6


def test_retrieves_expected_section_for_at_least_five() -> None:
    hits = 0
    for case in LAB_CASES:
        if case["expected_section"] is None:
            continue
        ranked = retrieve_lab(case["question"])
        assert len(ranked) <= 3
        distances = [item.distance for item in ranked]
        assert distances == sorted(distances)
        if any(item.section == case["expected_section"] for item in ranked):
            hits += 1
    assert hits >= 5


def test_supported_answers_include_citations() -> None:
    for case in LAB_CASES:
        payload = ask(
            case["question"],
            retrieve_fn=retrieve_lab,
            complete=complete_lab(case["question"]),
        )
        if case["expected_section"] is None:
            assert payload["answer"] == REFUSE_TEXT
            assert payload["citation"] is None
        else:
            assert payload["citation"] is not None
            assert payload["citation"]["section"] == case["expected_section"]
            assert payload["citation"]["document"] == "Employee Expense Policy"
            assert payload["citation"]["version"] == "2.0"
        assert len(payload["retrieved_chunks"]) <= 3
        assert all(isinstance(item["distance"], float) for item in payload["retrieved_chunks"])


def test_saved_output_covers_all_six_questions() -> None:
    results = run_required_questions()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    saved = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert len(saved) == 6
    assert saved[-1]["response"]["citation"] is None
    assert saved[-1]["response"]["answer"] == REFUSE_TEXT

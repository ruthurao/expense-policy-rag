import json

from expense_rag.generate import REFUSE_TEXT, build_prompt, parse_generation
from expense_rag.pipeline import ask
from expense_rag.retrieve import RetrievedChunk


def _chunk(section: str, title: str, text: str, distance: float) -> RetrievedChunk:
    return RetrievedChunk(
        section=f"{section}. {title}",
        distance=distance,
        document="Employee Expense Policy",
        version="2.0",
        section_number=section,
        section_title=title,
        text=text,
        chunk_id=f"expense-policy:v2.0:section-{section}",
    )


MEALS = _chunk("1", "Meals", "Employees may claim up to $65 per day for meals while traveling overnight.", 0.08)


def test_prompt_requires_excerpts_only() -> None:
    prompt = build_prompt("How much can I spend on food each day?", [MEALS])
    assert "only the policy excerpts" in prompt
    assert REFUSE_TEXT in prompt
    assert MEALS.text in prompt


def test_supported_answer_includes_citation() -> None:
    raw = json.dumps(
        {
            "answer": "Employees may claim up to $65 per day for meals.",
            "citation_section": "1. Meals",
        }
    )
    generated = parse_generation(raw, [MEALS])
    assert generated.citation_section == "1. Meals"
    assert "65" in generated.answer


def test_gym_question_has_no_citation() -> None:
    raw = json.dumps(
        {
            "answer": REFUSE_TEXT,
            "citation_section": None,
        }
    )
    generated = parse_generation(raw, [MEALS])
    assert generated.answer == REFUSE_TEXT
    assert generated.citation_section is None


def test_ask_returns_structured_payload() -> None:
    def fake_retrieve(question: str):
        return [MEALS]

    def fake_complete(prompt: str) -> str:
        return json.dumps(
            {
                "answer": "Employees may claim up to $65 per day for meals.",
                "citation_section": "1. Meals",
            }
        )

    payload = ask("How much can I spend on food each day?", retrieve_fn=fake_retrieve, complete=fake_complete)
    assert payload["citation"] == {
        "document": "Employee Expense Policy",
        "version": "2.0",
        "section": "1. Meals",
    }
    assert payload["retrieved_chunks"][0]["section"] == "1. Meals"
    assert isinstance(payload["retrieved_chunks"][0]["distance"], float)


def test_ask_gym_omits_citation() -> None:
    def fake_retrieve(question: str):
        return [MEALS]

    def fake_complete(prompt: str) -> str:
        return json.dumps({"answer": REFUSE_TEXT, "citation_section": None})

    payload = ask("Does the company reimburse gym memberships?", retrieve_fn=fake_retrieve, complete=fake_complete)
    assert payload["answer"] == REFUSE_TEXT
    assert payload["citation"] is None

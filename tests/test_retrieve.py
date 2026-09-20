from expense_rag.models import PolicyChunk
from expense_rag.retrieve import RETRIEVE_LIMIT, cosine_distance, rank_chunks, retrieve


def _chunk(section: str, title: str, embedding: list[float]) -> PolicyChunk:
    return PolicyChunk(
        chunk_id=f"expense-policy:v2.0:section-{section}",
        document="Employee Expense Policy",
        version="2.0",
        section=section,
        section_title=title,
        text=title,
        embedding=embedding,
    )


def test_cosine_distance_is_smaller_for_closer_vectors() -> None:
    meals = [1.0, 0.0, 0.0]
    close = [0.9, 0.1, 0.0]
    far = [0.0, 0.0, 1.0]
    assert cosine_distance(meals, close) < cosine_distance(meals, far)


def test_rank_returns_at_most_three_sorted_ascending() -> None:
    chunks = [
        _chunk("1", "Meals", [1.0, 0.0, 0.0]),
        _chunk("2", "Hotels", [0.0, 1.0, 0.0]),
        _chunk("3", "Airfare", [0.0, 0.0, 1.0]),
        _chunk("4", "Ground Transportation", [0.2, 0.8, 0.0]),
        _chunk("5", "Receipts", [0.1, 0.1, 0.9]),
        _chunk("6", "Submission Deadline", [0.3, 0.3, 0.3]),
    ]
    ranked = rank_chunks([1.0, 0.0, 0.0], chunks)
    assert len(ranked) == RETRIEVE_LIMIT
    distances = [item.distance for item in ranked]
    assert distances == sorted(distances)
    assert ranked[0].section == "1. Meals"


def test_food_question_ranks_meals_first(monkeypatch) -> None:
    chunks = [
        _chunk("1", "Meals", [1.0, 0.0, 0.0]),
        _chunk("2", "Hotels", [0.0, 1.0, 0.0]),
        _chunk("3", "Airfare", [0.0, 0.0, 1.0]),
        _chunk("4", "Ground Transportation", [0.2, 0.2, 0.8]),
        _chunk("5", "Receipts", [0.1, 0.8, 0.1]),
        _chunk("6", "Submission Deadline", [0.4, 0.4, 0.2]),
    ]
    monkeypatch.setattr("expense_rag.retrieve.embed_text", lambda question: [1.0, 0.0, 0.0])
    monkeypatch.setattr(
        "expense_rag.retrieve.retrieve_sql",
        lambda connection, query_vector, limit: rank_chunks(query_vector, chunks, limit),
    )
    ranked = retrieve("How much can I spend on food each day?", connection=object())
    assert ranked[0].section == "1. Meals"
    assert ranked[0].distance <= ranked[-1].distance
    assert len(ranked) <= 3

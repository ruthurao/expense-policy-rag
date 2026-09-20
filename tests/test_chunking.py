from pathlib import Path

from expense_rag.chunking import load_policy, split_policy

ROOT = Path(__file__).resolve().parents[1]
POLICY = (ROOT / "policy.md").read_text(encoding="utf-8")

EXPECTED = [
    ("1", "Meals", "Employees may claim up to $65 per day"),
    ("2", "Hotels", "Hotels are reimbursable up to $225 per night."),
    ("3", "Airfare", "Employees must purchase economy airfare."),
    ("4", "Ground Transportation", "Luxury vehicle upgrades are not reimbursable."),
    ("5", "Receipts", "Receipts are required for individual expenses of $25 or more."),
    ("6", "Submission Deadline", "within 30 days after travel ends."),
]


def test_creates_exactly_six_structural_chunks() -> None:
    chunks = split_policy(POLICY)
    assert len(chunks) == 6


def test_load_policy_reads_policy_md() -> None:
    chunks = load_policy(ROOT / "policy.md")
    assert [chunk.section_title for chunk in chunks] == [item[1] for item in EXPECTED]


def test_chunk_ids_and_metadata() -> None:
    chunks = split_policy(POLICY)
    for chunk, (section, title, _) in zip(chunks, EXPECTED, strict=True):
        assert chunk.chunk_id == f"expense-policy:v2.0:section-{section}"
        assert chunk.document == "Employee Expense Policy"
        assert chunk.version == "2.0"
        assert chunk.section == section
        assert chunk.section_title == title


def test_text_is_section_body_not_heading() -> None:
    chunks = split_policy(POLICY)
    for chunk, (_, _, snippet) in zip(chunks, EXPECTED, strict=True):
        assert snippet in chunk.text
        assert not chunk.text.startswith("#")


def test_does_not_cut_a_sentence_in_half() -> None:
    chunks = split_policy(POLICY)
    for chunk in chunks:
        assert chunk.text.endswith(".")
        assert "##" not in chunk.text
        assert "\n#" not in chunk.text

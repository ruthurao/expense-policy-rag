"""The six questions the Mini RAG Lab requires."""

from __future__ import annotations

from expense_rag.generate import REFUSE_TEXT

LAB_CASES = [
    {
        "question": "How much can I spend on food each day?",
        "expected_section": "1. Meals",
        "answer": "Employees may claim up to $65 per day for meals.",
    },
    {
        "question": "Can I book first-class airfare?",
        "expected_section": "3. Airfare",
        "answer": "Employees must purchase economy airfare. Business-class airfare requires written approval from a vice president.",
    },
    {
        "question": "My hotel costs $250. What do I need?",
        "expected_section": "2. Hotels",
        "answer": "A manager must approve higher rates before booking.",
    },
    {
        "question": "Do I need a receipt for a $20 taxi?",
        "expected_section": "5. Receipts",
        "answer": "No receipt is required under this policy. Receipts are required for individual expenses of $25 or more.",
    },
    {
        "question": "Can I claim a limousine upgrade?",
        "expected_section": "4. Ground Transportation",
        "answer": "Luxury vehicle upgrades are not reimbursable.",
    },
    {
        "question": "Does the company reimburse gym memberships?",
        "expected_section": None,
        "answer": REFUSE_TEXT,
    },
]

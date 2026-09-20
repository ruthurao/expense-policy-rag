# Expense Policy RAG

Grounded assistant over one employee expense policy. It splits the policy into
six section chunks, stores text + embeddings + metadata in Postgres/pgvector,
retrieves the three closest chunks by cosine distance, and answers only from
that evidence.

## Requirements

- Python 3.12 (the Mac python3 command may be older; use uv)
- uv
- Docker Desktop (for Postgres + pgvector)
- An OpenAI-compatible API key only if you want live generated answers

Automated tests and output/required_questions.json do not need a paid key.

## Setup

    cp .env.example .env
    docker compose up -d
    uv python pin 3.12
    uv sync

Edit .env and set OPENAI_API_KEY for live answers. Leave it blank for tests.

## Ingest

Reads policy.md, creates six structural chunks, embeds each one, and stores
the rows:

    uv run python -m expense_rag.ingest

You should see: ingested 6 chunks.

## Ask a question

    uv run python -m expense_rag.ask "How much can I spend on food each day?"

The JSON has answer, citation, and up to three retrieved_chunks with
numeric cosine distances (smaller is closer).

If the policy does not contain the answer, the app returns:

    The provided policy does not answer this question.

and citation is null.

## Tests

    uv run pytest

Saved output for the six required lab questions:

    output/required_questions.json

Regenerate it:

    PYTHONPATH=src python3 scripts/run_required_questions.py

## Required lab questions

1. How much can I spend on food each day? -> $65 / section 1 Meals
2. Can I book first-class airfare? -> economy, VP approval / section 3 Airfare
3. My hotel costs $250. What do I need? -> manager approval / section 2 Hotels
4. Do I need a receipt for a $20 taxi? -> no receipt under $25 / section 5 Receipts
5. Can I claim a limousine upgrade? -> luxury upgrades not reimbursable / section 4
6. Does the company reimburse gym memberships? -> policy does not answer / no citation

## How the pieces fit

1. Chunk: split on numbered headings. Never cut a sentence in half.
2. Embed: all-MiniLM-L6-v2 (384 numbers per chunk).
3. Store: one Postgres row per chunk (migrations/001_create_chunks.sql).
4. Retrieve: embed the question, ORDER BY embedding <=> query ASC LIMIT 3.
5. Generate: model may use only those excerpts. Cite the supporting section or refuse.

## Submit

Include this repository (or a zip of it):

- Application source under src/expense_rag/
- policy.md
- migrations/001_create_chunks.sql
- Ingest: uv run python -m expense_rag.ingest
- This README
- Tests under tests/ and saved output output/required_questions.json

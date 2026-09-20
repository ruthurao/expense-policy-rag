"""Ask a policy question and print the grounded JSON response."""

from __future__ import annotations

import json
import sys

from dotenv import load_dotenv

from expense_rag.pipeline import ask


def main() -> None:
    load_dotenv()
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("usage: python -m expense_rag.ask \"How much can I spend on food each day?\"")
        raise SystemExit(2)
    print(json.dumps(ask(question), indent=2))


if __name__ == "__main__":
    main()

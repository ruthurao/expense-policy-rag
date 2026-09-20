"""Write saved output for the six required lab questions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tests.test_required_questions import OUTPUT, run_required_questions


def main() -> None:
    results = run_required_questions()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(results)} questions)")


if __name__ == "__main__":
    main()

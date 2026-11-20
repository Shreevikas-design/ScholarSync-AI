from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import Settings
from evaluation.evaluator import evaluate
from services.search_service import AcademicSearchService


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure live ScholarSync retrieval against curated relevance judgments.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("dataset.json"))
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    settings = Settings()
    settings.ensure_data_dirs()
    report = evaluate(args.dataset, AcademicSearchService(settings), k=args.k)
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

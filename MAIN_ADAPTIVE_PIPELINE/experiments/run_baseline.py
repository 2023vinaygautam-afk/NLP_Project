"""
Baseline retrieval for the five fixed queries.

BASELINE = plain keyword search (union of all query terms) with TF-IDF
ranking but NO query understanding, NO domain check, NO adaptive method
choice and NO precision cutoff. This is the naive system that the
adaptive pipeline is compared against.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from retrieval import (
    load_index,
    load_corpus,
    keyword_search,
    rank_documents,
)
from adaptive_pipeline import FIXED_QUERIES

OUTPUT_FILE = BASE_DIR / "results" / "baseline_pipeline_results.json"


def main():
    index_data = load_index()
    inverted_index = index_data["index"]
    corpus = load_corpus()
    total_documents = len(corpus)

    results = []

    for item in FIXED_QUERIES:
        query = item["query"]

        candidates = keyword_search(query, inverted_index)

        ranked = rank_documents(
            candidates, query, inverted_index, total_documents
        )   # no cutoff on purpose

        results.append({
            "query_id": item["query_id"],
            "query": query,
            "retrieved_documents": [r["document_id"] for r in ranked],
        })

        print(f"{item['query_id']}: baseline retrieved {len(ranked)} documents")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
"""
Run once from the MAIN_ADAPTIVE_PIPELINE folder:
    python setup_comparison.py
It (1) patches experiments\baseline_vs_adaptive.py and
(2) creates experiments\run_baseline.py
"""
from pathlib import Path

root = Path(__file__).resolve().parent
target = root / "experiments" / "baseline_vs_adaptive.py"

text = target.read_text(encoding="utf-8")

if "baseline_pipeline_results.json" in text:
    print("baseline_vs_adaptive.py is already patched - skipping.")
else:
    # --- 1. baseline now comes from its own results file -----------------
    old = 'RETRIEVAL_FILE = RESULTS_DIR / "retrieval_results.json"'
    assert old in text, "RETRIEVAL_FILE line not found"
    text = text.replace(
        old, 'RETRIEVAL_FILE = RESULTS_DIR / "baseline_pipeline_results.json"'
    )

    # --- 2. read nested adaptive results ---------------------------------
    old = '''            or item.get("results")
            or []
        )
'''
    assert old in text, "load_results block not found"
    text = text.replace(old, '''            or item.get("results")
            or (item.get("retrieval") or {}).get("results")
            or []
        )

        retrieved = [
            entry.get("document_id", "")
            if isinstance(entry, dict)
            else entry
            for entry in retrieved
        ]
''', 1)

    # --- 3. normalise relevance labels (01_name -> D01) -------------------
    old = '''    retrieved = normalize_documents(retrieved)
'''
    assert old in text, "calculate_metrics block not found"
    text = text.replace(old, '''    retrieved = normalize_documents(retrieved)
    relevant = normalize_documents(relevant)
''', 1)

    target.write_text(text, encoding="utf-8")
    print("Patched:", target)

baseline_code = '''"""
Baseline retrieval for the five fixed queries.
BASELINE = plain keyword search (union of all query terms) with TF-IDF
ranking, but NO query understanding, NO domain check, NO adaptive method
choice and NO precision cutoff.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from retrieval import load_index, load_corpus, keyword_search, rank_documents
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
        )  # no cutoff on purpose

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
'''
(root / "experiments" / "run_baseline.py").write_text(baseline_code, encoding="utf-8")
print("Created:", root / "experiments" / "run_baseline.py")
print("\nDone. Now run:")
print("  python experiments\\run_baseline.py")
print("  python src\\adaptive_pipeline.py")
print("  python experiments\\baseline_vs_adaptive.py")
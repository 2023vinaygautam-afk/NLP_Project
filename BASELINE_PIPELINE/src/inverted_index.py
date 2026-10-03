
from pathlib import Path
from collections import defaultdict
import re
import json
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "documents"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def tokenize(text):
    return re.findall(
        r"\b[a-zA-Z]+(?:[-'][a-zA-Z]+)*\b",
        text.lower()
    )


def build_inverted_index(documents):
    index = defaultdict(set)

    for doc_id, text in documents.items():
        tokens = tokenize(text)

        for token in set(tokens):
            index[token].add(doc_id)

    return index


def main():
    print("\nBASELINE PIPELINE")
    print("MODULE 8: INVERTED INDEX")

    documents = {}

    for file_path in sorted(DOCS_DIR.glob("*.txt")):
        documents[file_path.stem] = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    if not documents:
        print("No documents found.")
        return

    index = build_inverted_index(documents)

    # Convert sets to sorted lists for JSON output
    serializable_index = {
        term: sorted(doc_ids)
        for term, doc_ids in sorted(index.items())
    }

    with open(
        RESULTS_DIR / "inverted_index.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(serializable_index, file, indent=4)

    rows = []

    for term, doc_ids in sorted(index.items()):
        rows.append({
            "Term": term,
            "Document_Frequency": len(doc_ids),
            "Posting_List": ", ".join(sorted(doc_ids))
        })

    index_df = pd.DataFrame(rows)

    index_df.to_csv(
        RESULTS_DIR / "inverted_index.csv",
        index=False
    )

    print(f"\nDocuments indexed: {len(documents)}")
    print(f"Unique indexed terms: {len(index)}")

    print("\nSAMPLE POSTING LISTS")

    for term in list(sorted(index.keys()))[:15]:
        print(f"{term}: {sorted(index[term])}")

    print("\nInverted index created successfully.")
    print("Results saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()

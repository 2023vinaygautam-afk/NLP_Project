
import json
import re
from collections import defaultdict, Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "results" / "cleaned_corpus.json"
OUTPUT_DIR = PROJECT_ROOT / "results"

INDEX_FILE = OUTPUT_DIR / "inverted_index.json"
STATS_FILE = OUTPUT_DIR / "inverted_index_statistics.json"


def load_corpus():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        documents = data.get("documents", [])
    else:
        documents = data

    normalized = []

    for i, doc in enumerate(documents, start=1):
        doc_id = (
            doc.get("doc_id")
            or doc.get("id")
            or doc.get("document_id")
            or f"D{i:02d}"
        )

        text = (
            doc.get("cleaned_text")
            or doc.get("cleaned")
            or doc.get("text")
            or doc.get("content")
            or ""
        )

        name = (
            doc.get("document_name")
            or doc.get("name")
            or doc.get("filename")
            or doc_id
        )

        normalized.append({
            "doc_id": str(doc_id),
            "name": str(name),
            "text": str(text)
        })

    return normalized


def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*\b", text.lower())


def build_index(documents):
    inverted_index = defaultdict(dict)
    document_lengths = {}

    for doc in documents:
        doc_id = doc["doc_id"]
        tokens = tokenize(doc["text"])

        document_lengths[doc_id] = len(tokens)

        frequencies = Counter(tokens)

        for term, frequency in frequencies.items():
            inverted_index[term][doc_id] = frequency

    return dict(inverted_index), document_lengths


def calculate_statistics(index, documents, document_lengths):
    total_terms = sum(document_lengths.values())

    statistics = {
        "total_documents": len(documents),
        "unique_terms": len(index),
        "total_token_occurrences": total_terms,
        "average_document_length": (
            round(total_terms / len(documents), 2)
            if documents else 0
        ),
        "average_document_frequency": (
            round(
                sum(len(postings) for postings in index.values())
                / len(index),
                2
            )
            if index else 0
        ),
        "top_20_terms_by_document_frequency": sorted(
            [
                {
                    "term": term,
                    "document_frequency": len(postings)
                }
                for term, postings in index.items()
            ],
            key=lambda item: (
                -item["document_frequency"],
                item["term"]
            )
        )[:20],
        "document_lengths": document_lengths
    }

    return statistics


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_corpus()

    if not documents:
        raise ValueError("No documents found in cleaned_corpus.json")

    index, document_lengths = build_index(documents)

    statistics = calculate_statistics(
        index,
        documents,
        document_lengths
    )

    output = {
        "description": "Term-to-document inverted index",
        "tokenization": "Lowercase alphanumeric words with internal hyphens/apostrophes",
        "documents": {
            doc["doc_id"]: doc["name"]
            for doc in documents
        },
        "index": index
    }

    with open(INDEX_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4, ensure_ascii=False)

    with open(STATS_FILE, "w", encoding="utf-8") as file:
        json.dump(statistics, file, indent=4, ensure_ascii=False)

    print("=" * 60)
    print("INVERTED INDEX CONSTRUCTION COMPLETED")
    print("=" * 60)
    print(f"Documents indexed: {statistics['total_documents']}")
    print(f"Unique terms: {statistics['unique_terms']}")
    print(f"Total token occurrences: {statistics['total_token_occurrences']}")
    print(f"Average document length: {statistics['average_document_length']}")
    print(f"Average document frequency: {statistics['average_document_frequency']}")

    print("\nSample postings:")

    for term in ["covid", "vaccine", "vaccination", "testing"]:
        if term in index:
            print(f"\n{term}: {index[term]}")

    print("\nFiles saved:")
    print(INDEX_FILE)
    print(STATS_FILE)


if __name__ == "__main__":
    main()

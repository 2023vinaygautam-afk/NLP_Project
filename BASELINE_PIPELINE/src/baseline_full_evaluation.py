
import re
import csv
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "documents"
QUERY_FILE = DATA_DIR / "Query" / "queries_400.csv"
INDEX_FILE = BASE_DIR / "results" / "inverted_index.json"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def tokenize(text):
    return re.findall(
        r"\b[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*\b",
        text.lower()
    )


def load_index():
    with open(INDEX_FILE, "r", encoding="utf-8") as file:
        raw_index = json.load(file)

    return {
        term: set(documents)
        for term, documents in raw_index.items()
    }


def load_documents():
    documents = {}

    for file_path in DATA_DIR.glob("*.txt"):
        documents[file_path.name] = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    return documents


def retrieve_keyword(query, index):
    terms = set(tokenize(query))
    matched = set()

    for term in terms:
        matched.update(index.get(term, set()))

    return matched


def retrieve_and(query, index):
    terms = list(dict.fromkeys(tokenize(query)))

    if not terms:
        return set()

    if any(term not in index for term in terms):
        return set()

    result = set(index[terms[0]])

    for term in terms[1:]:
        result.intersection_update(index[term])

    return result


def retrieve_or(query, index):
    return retrieve_keyword(query, index)


def retrieve_phrase(query, documents):
    phrase = " ".join(tokenize(query))

    if not phrase:
        return set()

    matched = set()

    for document_id, text in documents.items():
        normalized_text = " ".join(tokenize(text))

        if phrase in normalized_text:
            matched.add(document_id)

    return matched


def precision_recall_f1(retrieved, relevant):
    true_positive = len(retrieved & relevant)

    precision = (
        true_positive / len(retrieved)
        if retrieved else 0.0
    )

    recall = (
        true_positive / len(relevant)
        if relevant else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return precision, recall, f1


def precision_at_k(retrieved_ordered, relevant, k=5):
    top_k = retrieved_ordered[:k]

    return (
        sum(doc in relevant for doc in top_k) / k
    )


def recall_at_k(retrieved_ordered, relevant, k=5):
    if not relevant:
        return 0.0

    top_k = retrieved_ordered[:k]

    return (
        sum(doc in relevant for doc in top_k)
        / len(relevant)
    )


def main():
    print("\nBASELINE PIPELINE")
    print("FULL QUERY EVALUATION")

    queries = pd.read_csv(QUERY_FILE)
    index = load_index()
    documents = load_documents()

    # Normalize document IDs to filenames.
    available_ids = {
        name.lower(): name
        for name in documents
    }

    results = []

    for _, row in queries.iterrows():

        query_id = str(row["query_id"])
        query = str(row["query"])
        expected = str(row["expected_document"])

        # Support either filename or extension-free IDs.
        expected_name = expected.strip()

        if not expected_name.lower().endswith(".txt"):
            expected_name += ".txt"

        relevant_name = available_ids.get(
            expected_name.lower(),
            expected_name
        )

        relevant = {relevant_name}

        # Baseline uses fixed keyword retrieval.
        retrieved = retrieve_keyword(query, index)

        # Convert IDs into actual filenames when possible.
        normalized_retrieved = set()

        for doc in retrieved:
            candidate = str(doc)

            if not candidate.lower().endswith(".txt"):
                candidate += ".txt"

            normalized_retrieved.add(
                available_ids.get(candidate.lower(), candidate)
            )

        retrieved_ordered = sorted(normalized_retrieved)

        precision, recall, f1 = precision_recall_f1(
            normalized_retrieved,
            relevant
        )

        p_at_5 = precision_at_k(
            retrieved_ordered,
            relevant,
            5
        )

        r_at_5 = recall_at_k(
            retrieved_ordered,
            relevant,
            5
        )

        results.append({
            "query_id": query_id,
            "query": query,
            "expected_document": relevant_name,
            "retrieval_method": "KEYWORD",
            "retrieved_documents": "; ".join(retrieved_ordered),
            "number_retrieved": len(retrieved_ordered),
            "true_positives": len(normalized_retrieved & relevant),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "precision_at_5": p_at_5,
            "recall_at_5": r_at_5
        })

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        RESULTS_DIR / "baseline_full_query_results.csv",
        index=False
    )

    metric_columns = [
        "precision",
        "recall",
        "f1",
        "precision_at_5",
        "recall_at_5"
    ]

    macro = result_df[metric_columns].mean()

    total_tp = result_df["true_positives"].sum()
    total_retrieved = result_df["number_retrieved"].sum()
    total_relevant = len(result_df)

    micro_precision = (
        total_tp / total_retrieved
        if total_retrieved else 0.0
    )

    micro_recall = total_tp / total_relevant if total_relevant else 0.0

    micro_f1 = (
        2 * micro_precision * micro_recall
        / (micro_precision + micro_recall)
        if micro_precision + micro_recall else 0.0
    )

    summary = pd.DataFrame([
        {"Metric": "Total Queries", "Value": len(result_df)},
        {"Metric": "Macro Precision", "Value": macro["precision"]},
        {"Metric": "Macro Recall", "Value": macro["recall"]},
        {"Metric": "Macro F1", "Value": macro["f1"]},
        {"Metric": "Macro Precision@5", "Value": macro["precision_at_5"]},
        {"Metric": "Macro Recall@5", "Value": macro["recall_at_5"]},
        {"Metric": "Micro Precision", "Value": micro_precision},
        {"Metric": "Micro Recall", "Value": micro_recall},
        {"Metric": "Micro F1", "Value": micro_f1}
    ])

    summary.to_csv(
        RESULTS_DIR / "baseline_full_evaluation_summary.csv",
        index=False
    )

    print("\nEVALUATION SUMMARY")

    for _, row in summary.iterrows():
        print(f"{row['Metric']}: {row['Value']}")

    print("\nQueries evaluated:", len(result_df))
    print("Results saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()

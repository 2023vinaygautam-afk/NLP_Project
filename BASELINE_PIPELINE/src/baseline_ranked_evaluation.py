
import re
import json
import pandas as pd

from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "documents"
QUERY_FILE = DATA_DIR / "Query" / "queries_400.csv"
INDEX_FILE = BASE_DIR / "results" / "inverted_index.json"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [5, 10, 20]

STOPWORDS = {
    "what", "which", "who", "when", "where", "why", "how",
    "does", "do", "did", "is", "are", "was", "were",
    "the", "a", "an", "of", "to", "in", "on", "for",
    "and", "or", "with", "by", "from", "can", "could",
    "may", "might", "will", "would", "should", "be",
    "been", "being", "this", "that", "these", "those",
    "it", "its", "about", "during", "through", "into",
    "help", "support", "use", "used"
}


def tokenize(text):
    return re.findall(
        r"\b[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*\b",
        str(text).lower()
    )


def meaningful_terms(query):
    return [
        term for term in tokenize(query)
        if term not in STOPWORDS
    ]


def load_index():
    with open(INDEX_FILE, "r", encoding="utf-8") as file:
        raw_index = json.load(file)

    return {
        term: set(docs)
        for term, docs in raw_index.items()
    }


def load_documents():
    documents = {}

    for path in DATA_DIR.glob("*.txt"):
        documents[path.stem] = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    return documents


def rank_documents(query, index, documents):
    terms = meaningful_terms(query)

    scores = Counter()

    for term in terms:
        matching_docs = index.get(term, set())

        for doc_id in matching_docs:
            scores[doc_id] += 1

    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0])
    )

    return ranked


def calculate_metrics(ranked_docs, relevant_doc, total_docs):
    retrieved_ids = [doc for doc, score in ranked_docs]

    relevant = {relevant_doc}

    retrieved_set = set(retrieved_ids)

    tp = len(retrieved_set & relevant)

    precision = (
        tp / len(retrieved_set)
        if retrieved_set else 0.0
    )

    recall = (
        tp / len(relevant)
        if relevant else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    for k in K_VALUES:
        effective_k = min(k, total_docs)

        top_k = retrieved_ids[:effective_k]

        relevant_count = sum(
            doc in relevant for doc in top_k
        )

        result[f"precision_at_{k}"] = (
            relevant_count / effective_k
            if effective_k else 0.0
        )

        result[f"recall_at_{k}"] = (
            relevant_count / len(relevant)
            if relevant else 0.0
        )

    return result


def main():
    print("\nBASELINE PIPELINE")
    print("RANKED RETRIEVAL EVALUATION")

    queries = pd.read_csv(QUERY_FILE)
    index = load_index()
    documents = load_documents()

    filename_lookup = {
        filename.lower(): doc_id
        for doc_id, filename in [
            (doc_id, doc_id + ".txt")
            for doc_id in documents
        ]
    }

    results = []

    for _, row in queries.iterrows():

        query_id = row["query_id"]
        query = str(row["query"])

        expected = str(row["expected_document"]).strip()

        if expected.lower().endswith(".txt"):
            expected = expected[:-4]

        expected = filename_lookup.get(
            expected.lower(),
            expected
        )

        ranked = rank_documents(
            query,
            index,
            documents
        )

        metrics = calculate_metrics(
            ranked,
            expected,
            len(documents)
        )

        results.append({
            "query_id": query_id,
            "query": query,
            "expected_document": expected,
            "retrieval_method": "TERM_MATCH_RANKING",
            "retrieved_documents": "; ".join(
                doc for doc, score in ranked
            ),
            "ranking_scores": "; ".join(
                f"{doc}:{score}"
                for doc, score in ranked
            ),
            "number_retrieved": len(ranked),
            **metrics
        })

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        RESULTS_DIR / "baseline_ranked_query_results.csv",
        index=False
    )

    metric_columns = [
        "precision",
        "recall",
        "f1",
        "precision_at_5",
        "recall_at_5",
        "precision_at_10",
        "recall_at_10",
        "precision_at_20",
        "recall_at_20"
    ]

    summary = []

    for metric in metric_columns:
        summary.append({
            "Metric": "Macro " + metric.replace("_", " ").title(),
            "Value": result_df[metric].mean()
        })

    total_tp = (
        result_df["recall"].sum()
    )

    total_relevant = len(result_df)

    total_retrieved = result_df["number_retrieved"].sum()

    micro_precision = (
        total_tp / total_retrieved
        if total_retrieved else 0.0
    )

    micro_recall = (
        total_tp / total_relevant
        if total_relevant else 0.0
    )

    micro_f1 = (
        2 * micro_precision * micro_recall
        / (micro_precision + micro_recall)
        if micro_precision + micro_recall else 0.0
    )

    summary.extend([
        {"Metric": "Micro Precision", "Value": micro_precision},
        {"Metric": "Micro Recall", "Value": micro_recall},
        {"Metric": "Micro F1", "Value": micro_f1},
        {
            "Metric": "Average Retrieved Documents",
            "Value": result_df["number_retrieved"].mean()
        },
        {
            "Metric": "Total Queries",
            "Value": len(result_df)
        }
    ])

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        RESULTS_DIR / "baseline_ranked_evaluation_summary.csv",
        index=False
    )

    print("\nRANKED EVALUATION SUMMARY")

    for _, row in summary_df.iterrows():
        print(
            f"{row['Metric']}: {row['Value']:.4f}"
            if isinstance(row["Value"], (int, float))
            else f"{row['Metric']}: {row['Value']}"
        )

    print("\nFIRST 5 QUERY RESULTS")

    print(
        result_df[
            [
                "query_id",
                "number_retrieved",
                "precision_at_5",
                "precision_at_10",
                "precision_at_20",
                "recall",
                "f1"
            ]
        ].head(5).to_string(index=False)
    )

    print("\nFiles saved:")
    print("baseline_ranked_query_results.csv")
    print("baseline_ranked_evaluation_summary.csv")


if __name__ == "__main__":
    main()

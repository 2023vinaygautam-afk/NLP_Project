
import csv
import json
from pathlib import Path
from statistics import mean

# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"

INPUT_CSV = RESULTS_DIR / "retrieval_results.csv"

OUTPUT_CSV = RESULTS_DIR / "baseline_evaluation_metrics.csv"

OUTPUT_JSON = RESULTS_DIR / "baseline_evaluation_metrics.json"

TOP_K = 5

EXPECTED_QUERY_IDS = {"Q01", "Q02", "Q03", "Q04", "Q05"}


# --------------------------------------------------
# LOAD RETRIEVAL RESULTS
# --------------------------------------------------

def load_results():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Retrieval results not found: {INPUT_CSV}"
        )

    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if len(rows) != 5:
        raise ValueError(
            f"Expected exactly 5 evaluation queries, found {len(rows)}"
        )

    actual_ids = {row["Query_ID"] for row in rows}

    if actual_ids != EXPECTED_QUERY_IDS:
        raise ValueError(
            f"Unexpected query IDs: {actual_ids}"
        )

    return rows


# --------------------------------------------------
# PARSE DOCUMENT IDS
# --------------------------------------------------

def parse_documents(value):
    if not value:
        return []

    return [
        document.strip()
        for document in value.split(";")
        if document.strip()
    ]


def parse_relevant_documents(value):
    if not value:
        return []

    return [
        document.strip()
        for document in value.split(";")
        if document.strip()
    ]


# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------

def precision_at_k(retrieved, relevant, k):
    top_k_documents = retrieved[:k]

    if not top_k_documents:
        return 0.0

    relevant_count = sum(
        1 for document in top_k_documents
        if document in relevant
    )

    return relevant_count / k


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0

    top_k_documents = retrieved[:k]

    relevant_count = sum(
        1 for document in top_k_documents
        if document in relevant
    )

    return relevant_count / len(relevant)


def f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
    ) / (precision + recall)


def reciprocal_rank(retrieved, relevant):
    for rank, document in enumerate(retrieved, start=1):
        if document in relevant:
            return 1 / rank

    return 0.0


def hit_rate(retrieved, relevant):
    return int(any(
        document in relevant
        for document in retrieved
    ))


# --------------------------------------------------
# EVALUATE EACH QUERY
# --------------------------------------------------

def evaluate_query(row):

    query_id = row["Query_ID"]

    query = row["Query"]

    retrieved = parse_documents(
        row["Retrieved_Documents"]
    )

    relevant = parse_relevant_documents(
        row["Relevant_Documents"]
    )

    if not relevant:
        raise ValueError(
            f"No relevance label found for {query_id}"
        )

    if len(set(retrieved)) != len(retrieved):
        raise ValueError(
            f"Duplicate document IDs found for {query_id}"
        )

    p1 = precision_at_k(retrieved, relevant, 1)

    p5 = precision_at_k(retrieved, relevant, TOP_K)

    r5 = recall_at_k(retrieved, relevant, TOP_K)

    f1_5 = f1_score(p5, r5)

    rr = reciprocal_rank(retrieved, relevant)

    hit = hit_rate(retrieved, relevant)

    first_relevant_rank = next(
        (
            rank
            for rank, document in enumerate(retrieved, start=1)
            if document in relevant
        ),
        None
    )

    return {
        "Query_ID": query_id,
        "Query": query,
        "Relevant_Documents": "; ".join(relevant),
        "First_Relevant_Rank": first_relevant_rank,
        "Precision@1": round(p1, 4),
        "Precision@5": round(p5, 4),
        "Recall@5": round(r5, 4),
        "F1@5": round(f1_5, 4),
        "Reciprocal_Rank": round(rr, 4),
        "Hit": hit
    }


# --------------------------------------------------
# AGGREGATE METRICS
# --------------------------------------------------

def calculate_overall_metrics(query_results):

    metric_names = [
        "Precision@1",
        "Precision@5",
        "Recall@5",
        "F1@5",
        "Reciprocal_Rank"
    ]

    overall = {}

    for metric in metric_names:
        overall[metric] = round(
            mean(row[metric] for row in query_results),
            4
        )

    overall["MRR"] = overall.pop("Reciprocal_Rank")

    overall["Hit_Rate"] = round(
        mean(row["Hit"] for row in query_results),
        4
    )

    overall["Total_Queries"] = len(query_results)

    overall["Total_Documents"] = 20

    return overall


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

def save_results(query_results, overall):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = list(query_results[0].keys())

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(query_results)

    output_data = {
        "evaluation_type": "Baseline Keyword TF-IDF-Style Retrieval",
        "evaluation_queries": 5,
        "corpus_documents": 20,
        "top_k": TOP_K,
        "per_query_metrics": query_results,
        "overall_metrics": overall
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output_data,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nEvaluation files saved:")
    print(OUTPUT_CSV)
    print(OUTPUT_JSON)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 60)
    print("BASELINE RETRIEVAL EVALUATION")
    print("=" * 60)

    rows = load_results()

    query_results = [
        evaluate_query(row)
        for row in rows
    ]

    overall = calculate_overall_metrics(
        query_results
    )

    print("\nPER-QUERY EVALUATION")
    print("-" * 60)

    for row in query_results:

        print(f"\nQuery: {row['Query_ID']}")

        print(
            f"Relevant Document: "
            f"{row['Relevant_Documents']}"
        )

        print(
            f"First Relevant Rank: "
            f"{row['First_Relevant_Rank']}"
        )

        print(
            f"Precision@1: {row['Precision@1']}"
        )

        print(
            f"Precision@5: {row['Precision@5']}"
        )

        print(
            f"Recall@5: {row['Recall@5']}"
        )

        print(
            f"F1@5: {row['F1@5']}"
        )

        print(
            f"Reciprocal Rank: "
            f"{row['Reciprocal_Rank']}"
        )

        print(f"Hit: {row['Hit']}")

    print("\n" + "=" * 60)
    print("OVERALL BASELINE METRICS")
    print("=" * 60)

    for metric, value in overall.items():
        print(f"{metric}: {value}")

    save_results(query_results, overall)

    print("\nBaseline evaluation completed successfully.")


if __name__ == "__main__":
    main()

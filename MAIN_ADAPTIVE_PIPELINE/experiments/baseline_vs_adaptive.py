from pathlib import Path
import json
import sys
import pandas as pd


# =====================================================
# 1. PATH CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

sys.path.insert(0, str(BASE_DIR / "src"))
from adaptive_pipeline import FIXED_QUERIES

RETRIEVAL_FILE = RESULTS_DIR / "baseline_pipeline_results.json"
ADAPTIVE_FILE = RESULTS_DIR / "adaptive_pipeline_results.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. FIVE EVALUATION QUERIES AND RELEVANCE LABELS
# =====================================================

RELEVANCE = {
    item["query"]: set(item["expected_documents"])
    for item in FIXED_QUERIES
}


# =====================================================
# 3. DOCUMENT ID NORMALIZATION
# =====================================================

def normalize_document_id(value):

    value = str(value).strip()

    value = value.replace(".txt", "")

    if value.startswith("D") and len(value) == 3:
        return value

    for number in range(1, 21):

        if value.startswith(f"{number:02d}_"):
            return f"D{number:02d}"

    return value


def normalize_documents(value):

    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
        return []

    if isinstance(value, str):

        value = value.strip()

        if not value:
            return []

        try:
            parsed = json.loads(value)

            if isinstance(parsed, list):
                value = parsed

        except json.JSONDecodeError:

            value = value.replace(";", ",")

            value = [
                item.strip()
                for item in value.split(",")
                if item.strip()
            ]

    if isinstance(value, dict):
        value = list(value.keys())

    if not isinstance(value, (list, tuple, set)):
        value = [value]

    return list(dict.fromkeys(
        normalize_document_id(item)
        for item in value
        if str(item).strip()
    ))


# =====================================================
# 4. METRIC CALCULATION
# =====================================================

def calculate_metrics(retrieved, relevant):

    retrieved = normalize_documents(retrieved)
    relevant = normalize_documents(relevant)

    retrieved_set = set(retrieved)
    relevant_set = set(relevant)

    true_positives = retrieved_set & relevant_set

    false_positives = retrieved_set - relevant_set

    false_negatives = relevant_set - retrieved_set

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Retrieved_Count": len(retrieved)
    }


# =====================================================
# 5. LOAD RETRIEVAL RESULTS
# =====================================================

def load_results(file_path):

    if not file_path.exists():

        print(f"\nFile not found: {file_path}")

        return {}

    with open(file_path, "r", encoding="utf-8") as file:

        data = json.load(file)

    if isinstance(data, dict):

        if isinstance(data.get("results"), list):
            data = data["results"]

        elif isinstance(data.get("retrieval_results"), list):
            data = data["retrieval_results"]

        elif isinstance(data.get("queries"), list):
            data = data["queries"]

        else:
            data = [
                {"query": key, **value}
                for key, value in data.items()
                if isinstance(value, dict)
            ]

    result_map = {}

    for item in data:

        if not isinstance(item, dict):
            continue

        query = (
            item.get("query")
            or item.get("Query")
            or item.get("user_query")
        )

        if not query:
            continue

        retrieved = (
            item.get("retrieved_documents")
            or item.get("Retrieved_Documents")
            or item.get("retrieved")
            or item.get("results")
            or (item.get("retrieval") or {}).get("results")
            or []
        )

        # results may be dicts such as {"document_id": "D01", ...}
        retrieved = [
            entry.get("document_id", "")
            if isinstance(entry, dict)
            else entry
            for entry in retrieved
        ]

        result_map[str(query).strip()] = normalize_documents(
            retrieved
        )

    return result_map


# =====================================================
# 6. EVALUATE ONE PIPELINE
# =====================================================

def evaluate_pipeline(result_map, pipeline_name):

    rows = []

    for query_id, (query, relevant) in enumerate(
        RELEVANCE.items(), start=1
    ):

        retrieved = result_map.get(query, [])

        metrics = calculate_metrics(
            retrieved,
            relevant
        )

        rows.append({

            "Query_ID": f"Q{query_id}",

            "Query": query,

            "Pipeline": pipeline_name,

            "Retrieved_Documents": "; ".join(retrieved),

            "Expected_Relevant_Documents": "; ".join(
                sorted(normalize_documents(relevant))
            ),

            "Relevant_Count": len(relevant),

            **metrics

        })

    return pd.DataFrame(rows)


# =====================================================
# 7. MAIN COMPARATIVE EVALUATION
# =====================================================

def main():

    print("\nMAIN ADAPTIVE PIPELINE")
    print("=" * 65)

    print("SAME-QUERY BASELINE VS ADAPTIVE EVALUATION")

    baseline_map = load_results(RETRIEVAL_FILE)

    adaptive_map = load_results(ADAPTIVE_FILE)

    print("\nQueries defined:", len(RELEVANCE))

    print("Baseline query results:", len(baseline_map))

    print("Adaptive query results:", len(adaptive_map))

    baseline = evaluate_pipeline(
        baseline_map,
        "BASELINE"
    )

    adaptive = evaluate_pipeline(
        adaptive_map,
        "ADAPTIVE"
    )

    # -------------------------------------------------
    # 8. MERGE SAME QUERY RESULTS
    # -------------------------------------------------

    comparison = baseline.merge(
        adaptive,
        on=["Query_ID", "Query", "Relevant_Count"],
        suffixes=("_Baseline", "_Adaptive")
    )

    metric_names = [
        "Precision",
        "Recall",
        "F1_Score"
    ]

    for metric in metric_names:

        comparison[f"{metric}_Change"] = (
            comparison[f"{metric}_Adaptive"]
            - comparison[f"{metric}_Baseline"]
        )

    # -------------------------------------------------
    # 9. QUERY-WISE OUTCOME
    # -------------------------------------------------

    comparison["F1_Outcome"] = comparison[
        "F1_Score_Change"
    ].apply(

        lambda value:
        "Improved" if value > 0
        else "Decreased" if value < 0
        else "Unchanged"

    )

    # -------------------------------------------------
    # 10. MACRO METRICS
    # -------------------------------------------------

    summary = {}

    for pipeline_name, frame in [
        ("Baseline", baseline),
        ("Adaptive", adaptive)
    ]:

        summary[pipeline_name] = {

            "Precision": float(frame["Precision"].mean()),

            "Recall": float(frame["Recall"].mean()),

            "F1_Score": float(frame["F1_Score"].mean())

        }

    # -------------------------------------------------
    # 11. ADAPTIVE MINUS BASELINE
    # -------------------------------------------------

    difference = {}

    for metric in metric_names:

        difference[metric] = (
            summary["Adaptive"][metric]
            - summary["Baseline"][metric]
        )

    summary["Adaptive_Minus_Baseline"] = difference

    summary["Query_Wise_Comparison"] = {

        "Improved": int(
            (comparison["F1_Outcome"] == "Improved").sum()
        ),

        "Unchanged": int(
            (comparison["F1_Outcome"] == "Unchanged").sum()
        ),

        "Decreased": int(
            (comparison["F1_Outcome"] == "Decreased").sum()
        )

    }

    summary["Evaluation_Information"] = {

        "Total_Queries": len(RELEVANCE),

        "Baseline_Queries_Found": len(baseline_map),

        "Adaptive_Queries_Found": len(adaptive_map),

        "Same_Query_Set": True,

        "Relevant_Document_Count": "Multiple documents supported"

    }

    # -------------------------------------------------
    # 12. SAVE RESULTS
    # -------------------------------------------------

    baseline.to_csv(
        RESULTS_DIR / "baseline_five_query_evaluation.csv",
        index=False
    )

    adaptive.to_csv(
        RESULTS_DIR / "adaptive_five_query_evaluation.csv",
        index=False
    )

    comparison.to_csv(
        RESULTS_DIR / "same_query_comparison.csv",
        index=False
    )

    with open(
        RESULTS_DIR / "same_query_evaluation_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # -------------------------------------------------
    # 13. DISPLAY RESULTS
    # -------------------------------------------------

    print("\nQUERY-WISE COMPARISON")
    print("=" * 65)

    display_columns = [
        "Query_ID",
        "Precision_Baseline",
        "Precision_Adaptive",
        "Recall_Baseline",
        "Recall_Adaptive",
        "F1_Score_Baseline",
        "F1_Score_Adaptive",
        "F1_Outcome"
    ]

    print(
        comparison[display_columns].to_string(index=False)
    )

    print("\nOVERALL METRICS")
    print("=" * 65)

    for pipeline_name in ["Baseline", "Adaptive"]:

        print(f"\n{pipeline_name.upper()}")

        for metric, value in summary[pipeline_name].items():

            print(f"{metric}: {value:.4f}")

    print("\nADAPTIVE MINUS BASELINE")
    print("=" * 65)

    for metric, value in difference.items():

        print(f"{metric}: {value:+.4f}")

    print("\nGenerated files:")

    print("1. baseline_five_query_evaluation.csv")

    print("2. adaptive_five_query_evaluation.csv")

    print("3. same_query_comparison.csv")

    print("4. same_query_evaluation_summary.json")

    print("\nResults directory:")
    print(RESULTS_DIR)


if __name__ == "__main__":
    main()
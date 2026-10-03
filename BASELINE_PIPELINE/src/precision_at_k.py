
import pandas as pd
from pathlib import Path

# ---------------------------------------
# 1. PATH CONFIGURATION
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

INPUT_FILE = RESULTS_DIR / "baseline_full_query_results.csv"

df = pd.read_csv(INPUT_FILE)

# Updated K values
K_VALUES = [1, 2, 5, 10]


# ---------------------------------------
# 2. PARSE RETRIEVED DOCUMENTS
# ---------------------------------------

def parse_documents(value):

    if pd.isna(value) or not str(value).strip():
        return []

    return [
        doc.strip()
        for doc in str(value).split(";")
        if doc.strip()
    ]


# ---------------------------------------
# 3. PRECISION AT K
# ---------------------------------------

def calculate_precision_at_k(retrieved, relevant, k):

    top_k = retrieved[:k]

    relevant_count = sum(
        doc in relevant for doc in top_k
    )

    # Standard Precision@K denominator
    precision = relevant_count / k

    return precision, relevant_count


# ---------------------------------------
# 4. QUERY-WISE EVALUATION
# ---------------------------------------

results = []

for _, row in df.iterrows():

    retrieved = parse_documents(
        row["retrieved_documents"]
    )

    relevant = {
        str(row["expected_document"]).strip()
    }

    result = {
        "query_id": row["query_id"],
        "query": row["query"],
        "expected_document": row["expected_document"],
        "total_retrieved": len(retrieved)
    }

    for k in K_VALUES:

        precision, relevant_count = calculate_precision_at_k(
            retrieved,
            relevant,
            k
        )

        result[f"Relevant_in_top_{k}"] = relevant_count
        result[f"Precision_at_{k}"] = precision

    results.append(result)


result_df = pd.DataFrame(results)


# ---------------------------------------
# 5. SAVE QUERY-WISE RESULTS
# ---------------------------------------

result_df.to_csv(
    RESULTS_DIR / "precision_at_k_query_results.csv",
    index=False
)


# ---------------------------------------
# 6. AGGREGATE RESULTS
# ---------------------------------------

summary = []

for k in K_VALUES:

    metric = f"Precision_at_{k}"

    summary.append({
        "Metric": f"Macro Precision@{k}",
        "Value": result_df[metric].mean()
    })

    summary.append({
        "Metric": f"Queries with Relevant Result in Top {k}",
        "Value": (
            result_df[f"Relevant_in_top_{k}"] > 0
        ).sum()
    })


summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    RESULTS_DIR / "precision_at_k_summary.csv",
    index=False
)


# ---------------------------------------
# 7. DISPLAY RESULTS
# ---------------------------------------

print("\nBASELINE PRECISION AT K")
print("=" * 60)

print("\nFIRST 5 QUERIES")

columns = [
    "query_id",
    "total_retrieved",
    "Relevant_in_top_1",
    "Precision_at_1",
    "Relevant_in_top_2",
    "Precision_at_2",
    "Relevant_in_top_5",
    "Precision_at_5",
    "Relevant_in_top_10",
    "Precision_at_10"
]

print(
    result_df[columns]
    .head(5)
    .to_string(index=False)
)

print("\nAGGREGATE RESULTS")

for _, row in summary_df.iterrows():

    print(
        f"{row['Metric']}: {row['Value']:.4f}"
    )

print("\nSaved Files:")
print("1. precision_at_k_query_results.csv")
print("2. precision_at_k_summary.csv")

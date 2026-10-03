
import json
import csv
from pathlib import Path

from query_understanding import understand_query
from adaptive_decision_engine import select_retrieval_method

from retrieval import (
    load_index,
    load_corpus,
    retrieve_detailed
)


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "adaptive_pipeline_results.json"
METRICS_JSON = RESULTS_DIR / "adaptive_evaluation_metrics.json"
METRICS_CSV = RESULTS_DIR / "adaptive_evaluation_metrics.csv"

K = 5


# ==================================================
# 2. FIVE FIXED EVALUATION QUERIES
# ==================================================

FIXED_QUERIES = [
    {
        "query_id": "Q01",
        "query": (
            "What are the main causes of COVID-19 "
            "transmission and how can it be prevented?"
        ),
        "expected_document": "D01"
    },
    {
        "query_id": "Q02",
        "query": (
            "How effective are COVID-19 vaccines in "
            "preventing infection and reducing disease severity?"
        ),
        "expected_document": "D03"
    },
    {
        "query_id": "Q03",
        "query": (
            "How does COVID-19 vaccination help protect "
            "individuals and communities from infection?"
        ),
        "expected_document": "D02"
    },
    {
        "query_id": "Q04",
        "query": (
            "What is the relationship between vaccination, "
            "COVID-19 testing, and early detection of infection?"
        ),
        "expected_document": "D16"
    },
    {
        "query_id": "Q05",
        "query": (
            "How do quarantine and contact tracing help "
            "control infectious disease outbreaks?"
        ),
        "expected_document": "D06"
    }
]


# ==================================================
# 3. LOAD PROJECT RESOURCES
# ==================================================

def load_resources():

    print("\nLoading healthcare document corpus...")

    index_result = load_index()

    if isinstance(index_result, tuple):

        index = index_result[0]

        if len(index_result) > 1:
            document_mapping = index_result[1]
        else:
            document_mapping = {}

    else:

        index = index_result
        document_mapping = {}

    corpus = load_corpus()

    if not document_mapping:

        document_mapping = {
            doc_id: doc_name
            for doc_id, doc_name in index["documents"].items()
        }

    print(f"Documents loaded: {len(document_mapping)}")
    print(f"Indexed terms: {len(index['index'])}")

    return index, document_mapping, corpus


# ==================================================
# 4. DOMAIN RELEVANCE CHECK
# ==================================================

def is_domain_relevant(query_info):

    domain_terms = query_info.get("domain_terms", [])

    if isinstance(domain_terms, str):
        domain_terms = [domain_terms]

    return len(domain_terms) > 0


# ==================================================
# 5. OUT-OF-DOMAIN RESPONSE
# ==================================================

def create_out_of_domain_response(query, query_info):

    return {
        "query": query,

        "query_understanding": query_info,

        "adaptive_decision": {
            "method": "ABSTAIN",
            "confidence": 1.0,
            "reason": (
                "No recognized healthcare or public-health "
                "domain terms were identified."
            )
        },

        "retrieval": {
            "method": "NONE",
            "count": 0,
            "results": []
        },

        "status": "OUT_OF_DOMAIN"
    }


# ==================================================
# 6. MAIN ADAPTIVE RETRIEVAL PIPELINE
# ==================================================

def run_adaptive_pipeline(
    query,
    index,
    document_mapping,
    corpus
):

    # STEP 1: Understand query

    query_info = understand_query(query)

    # STEP 2: Check domain relevance

    if not is_domain_relevant(query_info):

        return create_out_of_domain_response(
            query,
            query_info
        )

    # STEP 3: Select retrieval method

    decision = select_retrieval_method(query_info)

    retrieval_method = decision.get(
        "selected_method",
        "KEYWORD"
    )

    # STEP 4: Execute lexical retrieval

    retrieval_output = retrieve_detailed(
        query,
        retrieval_method,
        index,
        corpus
    )

    # STEP 5: Enrich retrieved documents

    enriched_results = []

    for result in retrieval_output.get("results", []):

        doc_id = result.get("document_id")

        document_name = document_mapping.get(
            doc_id,
            "Unknown document"
        )

        enriched_results.append({

            "document_id": doc_id,

            "document_name": document_name,

            "score": result.get("score", 0),

            "matched_terms": result.get(
                "matched_terms",
                []
            ),

            "coverage": result.get(
                "coverage",
                0
            ),

            "total_term_frequency": result.get(
                "total_term_frequency",
                0
            )

        })

    # STEP 6: Determine retrieval status

    if not enriched_results:

        status = "NO_MATCHING_EVIDENCE"

        reason = (
            "The query is within the supported domain, "
            "but no documents matched the selected "
            "lexical retrieval strategy."
        )

    else:

        status = "RETRIEVAL_COMPLETED"

        reason = (
            "Documents were retrieved using the "
            "selected lexical retrieval strategy."
        )

    # STEP 7: Final pipeline output

    final_output = {

        "query": query,

        "query_understanding": query_info,

        "adaptive_decision": {

            "method": retrieval_method,

            "confidence": decision.get(
                "decision_confidence",
                0
            ),

            "reason": decision.get(
                "decision_reason",
                reason
            )

        },

        "retrieval": {

            "method": retrieval_method,

            "count": len(enriched_results),

            "results": enriched_results

        },

        "status": status

    }

    return final_output


# ==================================================
# 7. CALCULATE EVALUATION METRICS
# ==================================================

def calculate_metrics(all_results):

    query_metrics = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for result in all_results:

        query_id = result["query_id"]

        expected = result["expected_document"]

        retrieved = [
            item["document_id"]
            for item in result["retrieval"]["results"]
        ]

        # Remove duplicate IDs while preserving ranking

        retrieved = list(dict.fromkeys(retrieved))

        retrieved_top5 = retrieved[:K]

        relevant_set = {expected}

        retrieved_set = set(retrieved)

        retrieved_top5_set = set(retrieved_top5)

        # Confusion counts

        tp = len(relevant_set & retrieved_set)

        fp = len(retrieved_set - relevant_set)

        fn = len(relevant_set - retrieved_set)

        # Precision and recall over the full returned list

        precision = (
            tp / len(retrieved_set)
            if retrieved_set else 0.0
        )

        recall = (
            tp / len(relevant_set)
            if relevant_set else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        # Precision@1

        precision_at_1 = (
            1.0
            if retrieved and retrieved[0] == expected
            else 0.0
        )

        # Fixed denominator K=5 for consistency

        precision_at_5 = (
            len(relevant_set & retrieved_top5_set) / K
        )

        recall_at_5 = (
            len(relevant_set & retrieved_top5_set)
            / len(relevant_set)
        )

        f1_at_5 = (
            2 * precision_at_5 * recall_at_5
            / (precision_at_5 + recall_at_5)
            if precision_at_5 + recall_at_5 > 0
            else 0.0
        )

        # Reciprocal rank

        expected_rank = (
            retrieved.index(expected) + 1
            if expected in retrieved
            else None
        )

        reciprocal_rank = (
            1 / expected_rank
            if expected_rank is not None
            else 0.0
        )

        hit_at_5 = int(
            expected in retrieved_top5
        )

        expected_retrieved = int(
            expected in retrieved
        )

        row = {

            "query_id": query_id,

            "expected_document": expected,

            "retrieved_documents": ";".join(retrieved),

            "retrieved_count": len(retrieved),

            "expected_document_rank": expected_rank,

            "TP": tp,

            "FP": fp,

            "FN": fn,

            "Precision": precision,

            "Recall": recall,

            "F1": f1,

            "Precision@1": precision_at_1,

            "Precision@5": precision_at_5,

            "Recall@5": recall_at_5,

            "F1@5": f1_at_5,

            "Reciprocal_Rank": reciprocal_rank,

            "Hit@5": hit_at_5,

            "Expected_Document_Retrieved": expected_retrieved

        }

        query_metrics.append(row)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Macro averages

    metric_names = [
        "Precision",
        "Recall",
        "F1",
        "Precision@1",
        "Precision@5",
        "Recall@5",
        "F1@5",
        "Reciprocal_Rank",
        "Hit@5"
    ]

    macro_metrics = {}

    for metric in metric_names:

        macro_metrics[metric] = (
            sum(row[metric] for row in query_metrics)
            / len(query_metrics)
            if query_metrics else 0.0
        )

    # Micro metrics

    micro_precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    micro_recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    micro_f1 = (
        2 * micro_precision * micro_recall
        / (micro_precision + micro_recall)
        if micro_precision + micro_recall > 0
        else 0.0
    )

    successful_queries = sum(
        row["Expected_Document_Retrieved"]
        for row in query_metrics
    )

    summary = {

        "total_queries": len(query_metrics),

        "total_documents": 20,

        "total_true_positives": total_tp,

        "total_false_positives": total_fp,

        "total_false_negatives": total_fn,

        "expected_document_retrieval_rate": (
            successful_queries / len(query_metrics)
            if query_metrics else 0.0
        ),

        "macro_metrics": macro_metrics,

        "micro_metrics": {

            "Precision": micro_precision,

            "Recall": micro_recall,

            "F1": micro_f1

        },

        "evaluation_definition": {

            "relevance": "One fixed relevant document per query",

            "k": K,

            "precision_at_5_denominator": (
                "Fixed K=5, including when fewer than 5 "
                "documents are returned"
            ),

            "reciprocal_rank": (
                "Inverse rank of the expected document, "
                "or zero when not retrieved"
            )

        }

    }

    return query_metrics, summary


# ==================================================
# 8. SAVE EVALUATION REPORTS
# ==================================================

def save_evaluation(
    all_results,
    query_metrics,
    summary
):

    # Save detailed adaptive retrieval output

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Save evaluation summary and per-query metrics

    evaluation_report = {

        "project": "Adaptive Healthcare NLP Retrieval",

        "evaluation_type": "Fixed Five-Query Evaluation",

        "retrieval_type": "Adaptive Lexical Retrieval",

        "summary": summary,

        "per_query_metrics": query_metrics,

        "retrieval_results_file": OUTPUT_FILE.name,

        "limitations": [

            "Evaluation uses five fixed queries.",

            "One relevant document is assigned to each query.",

            "Relevance labels require manual validation.",

            "Retrieval metrics do not measure answer factuality.",

            "Results do not establish statistical significance."

        ]

    }

    with open(
        METRICS_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evaluation_report,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Save per-query metrics to CSV

    if query_metrics:

        with open(
            METRICS_CSV,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(query_metrics[0].keys())
            )

            writer.writeheader()

            writer.writerows(query_metrics)

    print("\nAll evaluation reports saved successfully.")


# ==================================================
# 9. RUN FIVE-QUERY EXPERIMENT
# ==================================================

def run_fixed_evaluation(
    index,
    document_mapping,
    corpus
):

    all_results = []

    print("\n" + "=" * 70)

    print("ADAPTIVE HEALTHCARE NLP PIPELINE")

    print("FIVE-QUERY EXPERIMENT")

    print("=" * 70)

    for item in FIXED_QUERIES:

        query_id = item["query_id"]

        query = item["query"]

        expected_document = item["expected_document"]

        print("\n" + "-" * 70)

        print(f"Query ID: {query_id}")

        print(f"Query: {query}")

        print(
            f"Expected Relevant Document: {expected_document}"
        )

        output = run_adaptive_pipeline(
            query,
            index,
            document_mapping,
            corpus
        )

        retrieved_ids = [
            result["document_id"]
            for result in output["retrieval"]["results"]
        ]

        expected_rank = (
            retrieved_ids.index(expected_document) + 1
            if expected_document in retrieved_ids
            else None
        )

        output["query_id"] = query_id

        output["expected_document"] = expected_document

        output["expected_document_rank"] = expected_rank

        output["expected_document_retrieved"] = (
            expected_document in retrieved_ids
        )

        all_results.append(output)

        print(
            f"Selected Method: "
            f"{output['adaptive_decision']['method']}"
        )

        print(
            f"Retrieved Documents: "
            f"{output['retrieval']['count']}"
        )

        print(
            "Expected Document Rank: "
            f"{expected_rank if expected_rank is not None else 'Not Retrieved'}"
        )

        print(f"Status: {output['status']}")

    # Calculate metrics

    query_metrics, summary = calculate_metrics(all_results)

    # Save reports

    save_evaluation(
        all_results,
        query_metrics,
        summary
    )

    # Print per-query evaluation

    print("\n" + "=" * 70)

    print("PER-QUERY EVALUATION METRICS")

    print("=" * 70)

    for row in query_metrics:

        print(f"\nQuery: {row['query_id']}")

        print(f"Precision: {row['Precision']:.4f}")

        print(f"Recall: {row['Recall']:.4f}")

        print(f"F1: {row['F1']:.4f}")

        print(f"Precision@5: {row['Precision@5']:.4f}")

        print(f"Recall@5: {row['Recall@5']:.4f}")

        print(f"F1@5: {row['F1@5']:.4f}")

        print(
            f"Reciprocal Rank: "
            f"{row['Reciprocal_Rank']:.4f}"
        )

        print(f"Hit@5: {row['Hit@5']}")

    # Print macro summary

    print("\n" + "=" * 70)

    print("FINAL ADAPTIVE EVALUATION SUMMARY")

    print("=" * 70)

    print(f"Total Queries: {summary['total_queries']}")

    print(f"Total Documents: {summary['total_documents']}")

    print(
        "Expected Document Retrieval Rate: "
        f"{summary['expected_document_retrieval_rate']:.2%}"
    )

    print("\nMACRO METRICS")

    for metric, value in summary["macro_metrics"].items():

        print(f"{metric}: {value:.4f}")

    print("\nMICRO METRICS")

    for metric, value in summary["micro_metrics"].items():

        print(f"{metric}: {value:.4f}")

    print("\nOUTPUT FILES")

    print(OUTPUT_FILE)

    print(METRICS_JSON)

    print(METRICS_CSV)

    print("\nADAPTIVE EVALUATION COMPLETED SUCCESSFULLY.")

    print("=" * 70)


# ==================================================
# 10. MAIN
# ==================================================

def main():

    index, document_mapping, corpus = load_resources()

    run_fixed_evaluation(
        index,
        document_mapping,
        corpus
    )


if __name__ == "__main__":

    main()

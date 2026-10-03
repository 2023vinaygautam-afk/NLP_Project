
import json
from pathlib import Path

from query_understanding import understand_query


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "results"

OUTPUT_FILE = OUTPUT_DIR / "adaptive_decision_results.json"


# ============================================================
# ADAPTIVE DECISION ENGINE
# ============================================================

def select_retrieval_method(query_info):
    """
    Select a lexical retrieval strategy using query structure,
    extracted keywords, domain terms, and complexity.
    """

    query_type = query_info["query_type"]

    terms = query_info["terms"]

    keywords = query_info.get("keywords", terms)

    domain_terms = query_info.get("domain_terms", [])

    complexity = query_info["complexity"]

    # --------------------------------------------------------
    # EXPLICIT RETRIEVAL STRATEGY
    # --------------------------------------------------------

    if query_type == "PHRASE":

        method = "PHRASE"

        reason = (
            "Quoted phrase syntax was detected. "
            "Phrase retrieval will be selected."
        )

        confidence = 1.0

    elif query_type == "NOT":

        method = "NOT"

        reason = (
            "An explicit NOT operator was detected. "
            "Exclusion-based retrieval will be selected."
        )

        confidence = 1.0

    elif query_type == "AND":

        method = "AND"

        reason = (
            "An explicit AND operator was detected. "
            "Intersection-based retrieval will be selected."
        )

        confidence = 1.0

    elif query_type == "OR":

        method = "OR"

        reason = (
            "An explicit OR operator was detected. "
            "Union-based retrieval will be selected."
        )

        confidence = 1.0

    else:

        method = "KEYWORD"

        if domain_terms:

            reason = (
                "The query contains domain-specific terms "
                "without explicit Boolean syntax. "
                "Keyword retrieval will be selected."
            )

        else:

            reason = (
                "No explicit Boolean or quoted phrase syntax "
                "was detected. Keyword retrieval will be selected."
            )

        confidence = 0.85

    # --------------------------------------------------------
    # QUERY ANALYSIS
    # --------------------------------------------------------

    if complexity == "HIGH":

        complexity_reason = (
            "The query contains several terms or explicit "
            "Boolean structure."
        )

    elif complexity == "MEDIUM":

        complexity_reason = (
            "The query contains a moderate number of "
            "meaningful terms."
        )

    else:

        complexity_reason = (
            "The query contains a small number of meaningful terms."
        )

    # --------------------------------------------------------
    # DECISION OUTPUT
    # --------------------------------------------------------

    return {
        "selected_method": method,
        "decision_reason": reason,
        "complexity_reason": complexity_reason,
        "query_complexity": complexity,
        "number_of_terms": len(terms),
        "number_of_keywords": len(keywords),
        "number_of_domain_terms": len(domain_terms),
        "decision_confidence": confidence
    }


# ============================================================
# ADAPTIVE DECISION FUNCTION
# ============================================================

def adaptive_decision(query):

    query_info = understand_query(query)

    decision = select_retrieval_method(query_info)

    return {
        **query_info,
        **decision
    }


# ============================================================
# TESTING
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    test_queries = [
        "COVID",
        "vaccine effectiveness",
        '"vaccine effectiveness"',
        "COVID AND vaccination",
        "vaccination OR testing",
        "COVID NOT vaccination",
        "How does contact tracing work?",
        "COVID vaccination and prevention",
        "How does vaccination help prevent infection?",
        "contact tracing AND outbreak investigation"
    ]

    results = []

    for query in test_queries:

        result = adaptive_decision(query)

        results.append(result)

        print("\n" + "=" * 65)

        print(f"Query: {result['query']}")

        print(f"Query Type: {result['query_type']}")

        print(f"Selected Method: {result['selected_method']}")

        print(f"Complexity: {result['query_complexity']}")

        print(f"Keywords: {result['keywords']}")

        print(f"Domain Terms: {result['domain_terms']}")

        print(
            f"Decision Confidence: "
            f"{result['decision_confidence']:.2f}"
        )

        print(f"Reason: {result['decision_reason']}")

        print(f"Complexity Analysis: {result['complexity_reason']}")

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 65)

    print("ADAPTIVE DECISION ENGINE COMPLETED")

    print("\nResults saved to:")

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()

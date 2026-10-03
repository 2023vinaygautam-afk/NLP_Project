
import json
from pathlib import Path

from query_understanding import understand_query


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "results"

OUTPUT_FILE = OUTPUT_DIR / "adaptive_decision_results.json"


# ============================================================
# 2. ADAPTIVE RETRIEVAL DECISION ENGINE
# ============================================================

def select_retrieval_method(query_info):
    """
    Select a lexical retrieval method using:

    1. Query type
    2. Query complexity
    3. Number of meaningful terms
    4. Domain-specific terms
    5. Query intent

    Supported methods:
    KEYWORD, PHRASE, AND, OR, NOT
    """

    query_type = str(
        query_info.get("query_type", "KEYWORD")
    ).upper()

    terms = query_info.get("terms", [])

    keywords = query_info.get("keywords", terms)

    domain_terms = query_info.get("domain_terms", [])

    complexity = str(
        query_info.get("complexity", "LOW")
    ).upper()

    query = str(query_info.get("query", ""))

    number_of_terms = len(terms)

    number_of_keywords = len(keywords)

    number_of_domain_terms = len(domain_terms)

    # --------------------------------------------------------
    # STEP 1: SELECT RETRIEVAL METHOD
    # --------------------------------------------------------

    if query_type == "PHRASE":

        method = "PHRASE"

        reason = (
            "The query contains an explicitly quoted phrase. "
            "Phrase retrieval is selected to preserve term order."
        )

        confidence = 1.0

    elif query_type == "NOT":

        method = "NOT"

        reason = (
            "The query contains an explicit NOT operator. "
            "Exclusion-based lexical retrieval is selected."
        )

        confidence = 1.0

    elif query_type == "AND":

        method = "AND"

        reason = (
            "The query contains an explicit AND operator. "
            "Intersection-based lexical retrieval is selected."
        )

        confidence = 1.0

    elif query_type == "OR":

        method = "OR"

        reason = (
            "The query contains an explicit OR operator. "
            "Union-based lexical retrieval is selected."
        )

        confidence = 1.0

    else:

        method = "KEYWORD"

        if number_of_domain_terms > 0:

            reason = (
                "The query contains healthcare or public-health "
                "domain terms without explicit Boolean syntax. "
                "Keyword retrieval is selected."
            )

        else:

            reason = (
                "No explicit Boolean operator or quoted phrase "
                "was detected. Keyword retrieval is selected."
            )

        confidence = 0.85

    # --------------------------------------------------------
    # STEP 2: ANALYZE QUERY COMPLEXITY
    # --------------------------------------------------------

    if complexity == "HIGH":

        complexity_reason = (
            "The query has high complexity based on the "
            "query-understanding module."
        )

    elif complexity == "MEDIUM":

        complexity_reason = (
            "The query has moderate complexity and contains "
            "multiple meaningful terms."
        )

    else:

        complexity_reason = (
            "The query has relatively low complexity."
        )

    # --------------------------------------------------------
    # STEP 3: DETERMINE RETRIEVAL SCOPE
    # --------------------------------------------------------

    if method in ["AND", "PHRASE"]:

        retrieval_scope = "RESTRICTIVE"

        scope_reason = (
            "The retrieval method applies explicit term "
            "constraints or phrase matching."
        )

    elif method == "OR":

        retrieval_scope = "BROAD"

        scope_reason = (
            "OR retrieval allows documents matching any "
            "specified alternative."
        )

    elif method == "NOT":

        retrieval_scope = "EXCLUSION_BASED"

        scope_reason = (
            "The retrieval method excludes documents "
            "matching the prohibited term."
        )

    else:

        if number_of_keywords >= 5:

            retrieval_scope = "CONTROLLED"

            scope_reason = (
                "The query contains several keywords. "
                "Relevance ranking should prioritize documents "
                "covering more important query terms."
            )

        else:

            retrieval_scope = "STANDARD"

            scope_reason = (
                "Standard keyword retrieval is suitable "
                "for the detected query structure."
            )

    # --------------------------------------------------------
    # STEP 4: DETERMINE EVIDENCE REQUIREMENT
    # --------------------------------------------------------

    if complexity == "HIGH" or number_of_keywords >= 5:

        evidence_requirement = "HIGH"

    elif number_of_keywords >= 3:

        evidence_requirement = "MEDIUM"

    else:

        evidence_requirement = "LOW"

    # --------------------------------------------------------
    # STEP 5: DETERMINE INITIAL ACTION
    # --------------------------------------------------------

    if number_of_keywords == 0:

        initial_action = "REFINE"

        action_reason = (
            "No meaningful keywords were identified. "
            "The query should be refined before retrieval."
        )

    elif method in ["AND", "PHRASE"]:

        initial_action = "RETRIEVE"

        action_reason = (
            "The query contains explicit retrieval constraints "
            "that can be applied directly."
        )

    else:

        initial_action = "RETRIEVE"

        action_reason = (
            "Meaningful query terms are available for "
            "lexical retrieval."
        )

    # --------------------------------------------------------
    # STEP 6: QUERY ANALYSIS SUMMARY
    # --------------------------------------------------------

    return {
        "selected_method": method,

        "decision_reason": reason,

        "query_complexity": complexity,

        "complexity_reason": complexity_reason,

        "number_of_terms": number_of_terms,

        "number_of_keywords": number_of_keywords,

        "number_of_domain_terms": number_of_domain_terms,

        "retrieval_scope": retrieval_scope,

        "scope_reason": scope_reason,

        "evidence_requirement": evidence_requirement,

        "initial_action": initial_action,

        "action_reason": action_reason,

        "decision_confidence": round(confidence, 4)
    }


# ============================================================
# 3. ADAPTIVE DECISION FUNCTION
# ============================================================

def adaptive_decision(query):

    query_info = understand_query(query)

    decision = select_retrieval_method(query_info)

    return {
        **query_info,
        **decision
    }


# ============================================================
# 4. FIXED FIVE-QUERY EVALUATION SET
# ============================================================

EVALUATION_QUERIES = [
    {
        "id": "Q01",
        "query": (
            "What are the main causes of COVID-19 transmission "
            "and how can it be prevented?"
        ),
        "relevant": ["D01"]
    },
    {
        "id": "Q02",
        "query": (
            "How effective are COVID-19 vaccines in preventing "
            "infection and reducing disease severity?"
        ),
        "relevant": ["D03"]
    },
    {
        "id": "Q03",
        "query": (
            "How does COVID-19 vaccination help protect individuals "
            "and communities from infection?"
        ),
        "relevant": ["D02"]
    },
    {
        "id": "Q04",
        "query": (
            "What is the relationship between vaccination, "
            "COVID-19 testing, and early detection of infection?"
        ),
        "relevant": ["D16"]
    },
    {
        "id": "Q05",
        "query": (
            "How do quarantine and contact tracing help control "
            "infectious disease outbreaks?"
        ),
        "relevant": ["D06"]
    }
]


# ============================================================
# 5. RUN FIVE-QUERY DECISION ANALYSIS
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    print("=" * 70)

    print("ADAPTIVE DECISION ENGINE")

    print("FIXED FIVE-QUERY ANALYSIS")

    print("=" * 70)

    for query_data in EVALUATION_QUERIES:

        query_id = query_data["id"]

        query = query_data["query"]

        result = adaptive_decision(query)

        result["query_id"] = query_id

        result["expected_relevant_documents"] = (
            query_data["relevant"]
        )

        results.append(result)

        print("\n" + "-" * 70)

        print(f"Query ID: {query_id}")

        print(f"Query: {result['query']}")

        print(f"Query Type: {result['query_type']}")

        print(f"Selected Method: {result['selected_method']}")

        print(f"Complexity: {result['query_complexity']}")

        print(f"Keywords: {result['keywords']}")

        print(f"Domain Terms: {result['domain_terms']}")

        print(f"Retrieval Scope: {result['retrieval_scope']}")

        print(
            f"Evidence Requirement: "
            f"{result['evidence_requirement']}"
        )

        print(f"Initial Action: {result['initial_action']}")

        print(
            f"Decision Confidence: "
            f"{result['decision_confidence']:.4f}"
        )

        print(f"Decision Reason: {result['decision_reason']}")

        print(f"Scope Reason: {result['scope_reason']}")

        print(f"Action Reason: {result['action_reason']}")

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    output_data = {
        "project": "Adaptive Lexical Retrieval",

        "evaluation_type": "Fixed Five-Query Decision Analysis",

        "total_queries": len(results),

        "retrieval_type": "Lexical",

        "results": results
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output_data,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)

    print("ADAPTIVE DECISION ENGINE COMPLETED")

    print(f"Queries processed: {len(results)}")

    print(f"Results saved to: {OUTPUT_FILE}")

    print("=" * 70)


# ============================================================
# 6. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

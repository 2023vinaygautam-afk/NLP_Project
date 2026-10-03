
import re
import json
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "results"

OUTPUT_FILE = OUTPUT_DIR / "query_understanding_results.json"


# ============================================================
# CONFIGURATION
# ============================================================

BOOLEAN_OPERATORS = {"AND", "OR", "NOT"}

QUESTION_WORDS = {
    "what", "why", "how", "when", "where", "which", "who"
}

STOPWORDS = {
    "what", "why", "how", "when", "where", "which", "who",
    "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "would", "should",
    "the", "a", "an", "of", "to", "for", "in", "on",
    "with", "by", "from", "about", "between",
    "and", "or", "it", "its", "this", "that",
    "help", "main", "relationship", "effective"
}

DOMAIN_TERMS = {
    "covid",
    "covid-19",
    "vaccination",
    "vaccine",
    "vaccines",
    "transmission",
    "infection",
    "epidemiology",
    "incidence",
    "prevalence",
    "outbreak",
    "surveillance",
    "testing",
    "contact",
    "tracing",
    "quarantine",
    "prevention",
    "effectiveness",
    "severity",
    "disease",
    "infectious",
    "epidemiological",
    "variant",
    "variants",
    "wastewater",
    "antiviral",
    "immunity",
    "safety",
    "monitoring"
}


# ============================================================
# TOKEN EXTRACTION
# ============================================================

def extract_terms(query):

    tokens = re.findall(
        r"\b[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*\b",
        query.lower()
    )

    return [
        token
        for token in tokens
        if token.upper() not in BOOLEAN_OPERATORS
    ]


def extract_keywords(terms):

    return [
        term
        for term in terms
        if term not in STOPWORDS
    ]


def extract_domain_terms(terms):

    return [
        term
        for term in terms
        if term in DOMAIN_TERMS
    ]


# ============================================================
# QUERY TYPE DETECTION
# ============================================================

def detect_query_type(query):

    # Explicit quoted phrase
    if re.search(r'"[^"]+"', query):
        return "PHRASE"

    # Boolean operators must be explicitly written
    # in uppercase to distinguish them from ordinary English.
    if re.search(r"\bNOT\b", query):
        return "NOT"

    if re.search(r"\bAND\b", query):
        return "AND"

    if re.search(r"\bOR\b", query):
        return "OR"

    return "KEYWORD"


# ============================================================
# QUERY COMPLEXITY
# ============================================================

def detect_complexity(terms, keywords, query_type):

    if query_type in {"AND", "OR", "NOT"}:
        return "HIGH"

    if len(keywords) <= 2:
        return "LOW"

    if len(keywords) <= 5:
        return "MEDIUM"

    return "HIGH"


# ============================================================
# QUERY UNDERSTANDING
# ============================================================

def understand_query(query):

    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    query_type = detect_query_type(query)

    terms = extract_terms(query)

    keywords = extract_keywords(terms)

    domain_terms = extract_domain_terms(terms)

    question_tokens = re.findall(
        r"\b[a-zA-Z]+\b",
        query.lower()
    )

    question_words_found = [
        word
        for word in question_tokens
        if word in QUESTION_WORDS
    ]

    complexity = detect_complexity(
        terms,
        keywords,
        query_type
    )

    return {
        "query": query,
        "query_type": query_type,
        "terms": terms,
        "keywords": keywords,
        "domain_terms": domain_terms,
        "term_count": len(terms),
        "keyword_count": len(keywords),
        "contains_question_word": bool(question_words_found),
        "question_words": question_words_found,
        "complexity": complexity
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
        "COVID AND vaccination",
        "vaccination OR testing",
        "COVID NOT vaccination",
        '"vaccine effectiveness"',
        "How does contact tracing work?",
        "COVID vaccination and prevention",
        "How does vaccination help prevent infection?",
        "contact tracing AND outbreak investigation"
    ]

    results = [
        understand_query(query)
        for query in test_queries
    ]

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

    print("=" * 65)
    print("QUERY UNDERSTANDING COMPLETED")
    print("=" * 65)

    for result in results:

        print(f"\nQuery: {result['query']}")
        print(f"Type: {result['query_type']}")
        print(f"Terms: {result['terms']}")
        print(f"Keywords: {result['keywords']}")
        print(f"Domain Terms: {result['domain_terms']}")
        print(f"Complexity: {result['complexity']}")

    print("\nResults saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()

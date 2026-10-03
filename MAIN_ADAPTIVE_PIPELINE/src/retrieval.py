import json
import re
import math

from pathlib import Path


# ==================================================
# FILE PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_PATH = BASE_DIR / "results" / "inverted_index.json"

CORPUS_PATH = BASE_DIR / "results" / "cleaned_corpus.json"

OUTPUT_PATH = BASE_DIR / "results" / "retrieval_results.json"


# ==================================================
# CONFIGURATION
# ==================================================

BOOLEAN_OPERATORS = {"AND", "OR", "NOT"}

STOPWORDS = {
    "a", "an", "the",
    "is", "are", "was", "were",
    "do", "does", "did",
    "how", "what", "why",
    "when", "where", "which",
    "who", "whom",
    "can", "could", "would", "should",
    "of", "in", "on", "at", "to",
    "for", "with", "by", "from",
    "about", "and", "or",
    "help", "explain", "describe",
    "tell", "me"
}

# Extra generic words that carry no retrieval signal in this corpus
STOPWORDS |= {
    "main", "relationship", "between", "effective", "it", "its",
    "be", "this", "that", "these", "those", "other", "also"
}

# Precision controls for KEYWORD / OR retrieval
TOP_K = 5                 # never return more than 5 documents
MIN_COVERAGE = 0.5        # doc must match >= 50% of the query terms
RELATIVE_CUTOFF = 0.7     # doc score must be >= 70% of the best score


# ==================================================
# LOAD INVERTED INDEX
# ==================================================

def load_index():

    if not INDEX_PATH.exists():

        raise FileNotFoundError(
            f"Inverted index not found: {INDEX_PATH}"
        )

    with open(
        INDEX_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if not isinstance(data, dict):

        raise ValueError(
            "Invalid inverted index format."
        )

    if "index" not in data:

        raise KeyError(
            "The inverted index JSON does not contain "
            "the required 'index' field."
        )

    if "documents" not in data:

        raise KeyError(
            "The inverted index JSON does not contain "
            "the required 'documents' field."
        )

    return {
        "index": data["index"],
        "documents": data["documents"]
    }


# ==================================================
# LOAD CLEANED CORPUS
# ==================================================

def load_corpus():

    if not CORPUS_PATH.exists():

        raise FileNotFoundError(
            f"Cleaned corpus not found: {CORPUS_PATH}"
        )

    with open(
        CORPUS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    documents = {}

    if isinstance(data, dict):

        data = data.get("documents", data)

    if isinstance(data, list):

        for i, doc in enumerate(data, start=1):

            doc_id = (
                doc.get("document_id")
                or doc.get("doc_id")
                or doc.get("id")
                or f"D{i:02d}"
            )

            text = (
                doc.get("cleaned_text")
                or doc.get("cleaned")
                or doc.get("text")
                or doc.get("content")
                or doc.get("original_text")
                or ""
            )

            documents[doc_id] = str(text).lower()

    elif isinstance(data, dict):

        for doc_id, value in data.items():

            if isinstance(value, dict):

                text = (
                    value.get("cleaned_text")
                    or value.get("cleaned")
                    or value.get("text")
                    or value.get("content")
                    or ""
                )

            else:

                text = str(value)

            documents[doc_id] = str(text).lower()

    else:

        raise ValueError(
            "Unsupported cleaned corpus format."
        )

    return documents


# ==================================================
# TOKENIZATION
# ==================================================

def tokenize(text):

    return re.findall(
        r"\b[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*\b",
        str(text).lower()
    )


# ==================================================
# QUERY NORMALIZATION
# ==================================================

def remove_boolean_operators(text):

    return re.sub(
        r"\b(?:AND|OR|NOT)\b",
        " ",
        text,
        flags=re.IGNORECASE
    )


def get_query_terms(query, remove_stopwords=True):

    query = remove_boolean_operators(query)

    terms = tokenize(query)

    if remove_stopwords:

        terms = [
            term
            for term in terms
            if term not in STOPWORDS
        ]

    return terms


# ==================================================
# QUERY TERM EXPANSION
# ==================================================

def expand_query_term(term, index):

    term = term.lower().strip()

    if term == "covid":

        return ["covid-19", "covid"]

    return [term]


# ==================================================
# GET DOCUMENTS FOR A TERM
# ==================================================

def get_term_documents(term, index):

    matched_documents = set()

    for expanded_term in expand_query_term(
        term,
        index
    ):

        postings = index.get(expanded_term, {})

        if isinstance(postings, dict):

            matched_documents.update(
                postings.keys()
            )

    return matched_documents


# ==================================================
# TERM FREQUENCY
# ==================================================

def get_term_frequency(term, doc_id, index):

    frequency = 0

    for expanded_term in expand_query_term(
        term,
        index
    ):

        postings = index.get(expanded_term, {})

        if not isinstance(postings, dict):
            continue

        value = postings.get(doc_id, 0)

        if isinstance(value, dict):

            value = value.get(
                "frequency",
                value.get("tf", 0)
            )

        try:

            frequency += int(value)

        except (TypeError, ValueError):

            pass

    return frequency


# ==================================================
# KEYWORD SEARCH
# ==================================================

def keyword_search(query, index):

    terms = get_query_terms(query)

    matched_documents = set()

    for term in terms:

        matched_documents.update(
            get_term_documents(term, index)
        )

    return matched_documents


# ==================================================
# PHRASE SEARCH
# ==================================================

def phrase_search(query, corpus):

    phrase = query.strip().strip('"').lower()

    if not phrase:
        return set()

    phrase_tokens = tokenize(phrase)

    if not phrase_tokens:
        return set()

    normalized_phrase = " ".join(phrase_tokens)

    results = set()

    for doc_id, text in corpus.items():

        normalized_text = " ".join(
            tokenize(text)
        )

        if normalized_phrase in normalized_text:

            results.add(doc_id)

    return results


# ==================================================
# AND SEARCH
# ==================================================

def and_search(query, index):

    terms = get_query_terms(query)

    if not terms:
        return set()

    result = get_term_documents(
        terms[0],
        index
    )

    for term in terms[1:]:

        result = result.intersection(
            get_term_documents(term, index)
        )

    return result


# ==================================================
# OR SEARCH
# ==================================================

def or_search(query, index):

    terms = get_query_terms(query)

    result = set()

    for term in terms:

        result.update(
            get_term_documents(term, index)
        )

    return result


# ==================================================
# NOT SEARCH
# ==================================================

def not_search(query, index, all_documents):

    terms = get_query_terms(query)

    if not terms:
        return set()

    if len(terms) == 1:

        excluded = get_term_documents(
            terms[0],
            index
        )

        return all_documents - excluded

    positive_documents = get_term_documents(
        terms[0],
        index
    )

    excluded_documents = set()

    for term in terms[1:]:

        excluded_documents.update(
            get_term_documents(term, index)
        )

    return positive_documents - excluded_documents


# ==================================================
# IDF WEIGHT CALCULATION
# ==================================================

def calculate_idf(term, index, total_documents):

    if total_documents == 0:
        return 0.0

    document_frequency = len(
        get_term_documents(term, index)
    )

    return math.log(
        (total_documents + 1) /
        (document_frequency + 1)
    ) + 1


# ==================================================
# DOCUMENT RANKING
# ==================================================

def rank_documents(doc_ids, query, index, total_documents):

    terms = get_query_terms(query)

    unique_terms = list(dict.fromkeys(terms))

    ranked = []

    for doc_id in doc_ids:

        matched_terms = []

        total_frequency = 0

        weighted_score = 0.0

        for term in unique_terms:

            frequency = get_term_frequency(
                term,
                doc_id,
                index
            )

            if frequency <= 0:
                continue

            matched_terms.append(term)

            total_frequency += frequency

            idf = calculate_idf(
                term,
                index,
                total_documents
            )

            tf_weight = 1 + math.log(frequency)

            weighted_score += (
                tf_weight * idf
            )

        coverage = (

            len(matched_terms) / len(unique_terms)

            if unique_terms

            else 0

        )

        score = (
            weighted_score
            + 2 * coverage
        )

        ranked.append({

            "document_id": doc_id,

            "score": round(score, 4),

            "matched_terms": sorted(matched_terms),

            "coverage": round(coverage, 4),

            "total_term_frequency": total_frequency,

            "weighted_score": round(
                weighted_score,
                4
            )

        })

    ranked.sort(

        key=lambda item: (
            -item["score"],
            item["document_id"]
        )

    )

    return ranked


# ==================================================
# CUTOFF (PRECISION CONTROL)
# ==================================================

def apply_cutoff(ranked, method):

    # AND / PHRASE / NOT are already restrictive.
    # KEYWORD / OR return a union, so they need a cutoff.
    if method not in {"KEYWORD", "OR"} or not ranked:
        return ranked

    best = ranked[0]["score"]

    kept = [
        item for item in ranked
        if item["coverage"] >= MIN_COVERAGE
        and item["score"] >= RELATIVE_CUTOFF * best
    ]

    return kept[:TOP_K]


# ==================================================
# DETAILED RETRIEVAL
# ==================================================

def retrieve_detailed(query, method, index, corpus):

    method = method.upper().strip()

    all_documents = set(corpus.keys())

    if "index" in index and isinstance(index["index"], dict):
        inverted_index = index["index"]
    else:
        inverted_index = index

    total_documents = len(corpus)

    if method == "KEYWORD":
        candidates = keyword_search(query, inverted_index)

    elif method == "PHRASE":
        candidates = phrase_search(query, corpus)

    elif method == "AND":
        candidates = and_search(query, inverted_index)

    elif method == "OR":
        candidates = or_search(query, inverted_index)

    elif method == "NOT":
        candidates = not_search(query, inverted_index, all_documents)

    else:
        raise ValueError(
            f"Unsupported retrieval method: {method}"
        )

    ranked = rank_documents(
        candidates,
        query,
        inverted_index,
        total_documents
    )

    ranked = apply_cutoff(ranked, method)

    return {
        "query": query,
        "method": method,
        "retrieved_count": len(ranked),
        "results": ranked
    }


# ==================================================
# MAIN RETRIEVAL FUNCTION (returns document IDs only)
# ==================================================

def retrieve(query, method, index, corpus):

    details = retrieve_detailed(query, method, index, corpus)

    return [item["document_id"] for item in details["results"]]


# ==================================================
# TEST RETRIEVAL
# ==================================================

def main():

    index_data = load_index()

    inverted_index = index_data["index"]

    document_mapping = index_data["documents"]

    corpus = load_corpus()

    test_queries = [

        ("COVID", "KEYWORD"),

        ('"vaccine effectiveness"', "PHRASE"),

        ("COVID AND vaccination", "AND"),

        ("vaccination OR testing", "OR"),

        ("COVID NOT vaccination", "NOT")

    ]

    results = []

    print("\n========== RETRIEVAL ENGINE ==========\n")

    print("Documents:", len(document_mapping))

    print("Indexed terms:", len(inverted_index))

    for query, method in test_queries:

        details = retrieve_detailed(
            query,
            method,
            index_data,
            corpus
        )

        print("\nQuery:", query)

        print("Method:", method)

        print(
            "Retrieved count:",
            details["retrieved_count"]
        )

        for item in details["results"]:

            doc_id = item["document_id"]

            document_name = document_mapping.get(
                doc_id,
                doc_id
            )

            print(

                f"{doc_id} | {document_name} | "
                f"Score: {item['score']} | "
                f"Coverage: {item['coverage']} | "
                f"Matched terms: {item['matched_terms']}"

            )

        print("-" * 65)

        results.append({

            **details,

            "retrieved_documents": [

                document_mapping.get(
                    item["document_id"],
                    item["document_id"]
                )

                for item in details["results"]

            ]

        })

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nRETRIEVAL ENGINE COMPLETED")

    print("Results saved to:", OUTPUT_PATH)


if __name__ == "__main__":

    main()

import json
import csv
import re
from pathlib import Path
from collections import Counter, defaultdict


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
INDEX_FILE = BASE_DIR / "results" / "inverted_index.json"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = RESULTS_DIR / "retrieval_results.csv"
JSON_OUTPUT = RESULTS_DIR / "baseline_retrieval_results.json"


# ==========================================
# FIXED FIVE EVALUATION QUERIES
# ==========================================

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


# ==========================================
# TEXT PREPROCESSING
# ==========================================

def tokenize(text):

    text = text.lower()

    tokens = re.findall(r"\b[a-z0-9]+\b", text)

    return tokens


# ==========================================
# DOCUMENT LOADING
# ==========================================

def load_documents():

    documents = {}

    for file_path in sorted(DOCUMENTS_DIR.glob("*.txt")):

        content = file_path.read_text(encoding="utf-8")

        match = re.match(r"(\d+)", file_path.stem)

        if match:
            doc_id = f"D{int(match.group(1)):02d}"
        else:
            doc_id = file_path.stem

        documents[doc_id] = content

    if len(documents) != 20:
        raise ValueError(
            f"Expected 20 documents, found {len(documents)}"
        )

    return documents


# ==========================================
# LOAD INVERTED INDEX
# ==========================================

def load_index():

    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"Inverted index not found: {INDEX_FILE}"
        )

    with open(INDEX_FILE, "r", encoding="utf-8") as file:
        index = json.load(file)

    return index


# ==========================================
# NORMALIZE DOCUMENT IDS
# ==========================================

def normalize_doc_id(doc_id):

    doc_id = str(doc_id)

    match = re.search(r"(\d+)", doc_id)

    if match:
        return f"D{int(match.group(1)):02d}"

    return doc_id


# ==========================================
# LEXICAL RETRIEVAL
# ==========================================

def retrieve_documents(query, documents, index):

    query_tokens = tokenize(query)

    scores = defaultdict(float)

    for token in query_tokens:

        if token not in index:
            continue

        postings = index[token]

        if isinstance(postings, dict):

            for doc_id, frequency in postings.items():

                normalized_id = normalize_doc_id(doc_id)

                if normalized_id in documents:
                    scores[normalized_id] += float(frequency)

        elif isinstance(postings, list):

            for item in postings:

                if isinstance(item, dict):

                    doc_id = (
                        item.get("doc_id")
                        or item.get("document_id")
                        or item.get("doc")
                    )

                    frequency = (
                        item.get("tf")
                        or item.get("frequency")
                        or item.get("count")
                        or 1
                    )

                    if doc_id is not None:

                        normalized_id = normalize_doc_id(doc_id)

                        if normalized_id in documents:
                            scores[normalized_id] += float(frequency)

                else:

                    normalized_id = normalize_doc_id(item)

                    if normalized_id in documents:
                        scores[normalized_id] += 1.0

    # Include all documents, including zero-score documents
    for doc_id in documents:
        scores.setdefault(doc_id, 0.0)

    ranked_documents = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0])
    )

    return ranked_documents


# ==========================================
# RUN BASELINE RETRIEVAL
# ==========================================

def run_baseline():

    documents = load_documents()

    index = load_index()

    csv_rows = []
    json_results = []

    print("\nBASELINE LEXICAL RETRIEVAL")
    print("=" * 60)

    print(f"Total documents: {len(documents)}")
    print(f"Total evaluation queries: {len(EVALUATION_QUERIES)}")

    for item in EVALUATION_QUERIES:

        query_id = item["id"]
        query = item["query"]
        relevant_docs = item["relevant"]

        ranked = retrieve_documents(
            query,
            documents,
            index
        )

        ranked_ids = [
            doc_id for doc_id, score in ranked
        ]

        relevant_ranks = {}

        for relevant_id in relevant_docs:

            if relevant_id in ranked_ids:

                relevant_ranks[relevant_id] = (
                    ranked_ids.index(relevant_id) + 1
                )

        expected_retrieved = all(
            doc_id in ranked_ids
            for doc_id in relevant_docs
        )

        row = {
            "Query_ID": query_id,
            "Query": query,
            "Query_Type": "Fixed Evaluation Query",
            "Relevant_Documents": "; ".join(relevant_docs),
            "Retrieved_Documents": "; ".join(ranked_ids),
            "Number_of_Results": len(ranked_ids),
            "Expected_Document_Retrieved": expected_retrieved,
            "Relevant_Document_Ranks": relevant_ranks
        }

        csv_rows.append(row)

        json_results.append({
            "query_id": query_id,
            "query": query,
            "relevant_documents": relevant_docs,
            "ranked_documents": [
                {
                    "document_id": doc_id,
                    "score": score,
                    "rank": rank
                }
                for rank, (doc_id, score) in enumerate(
                    ranked, start=1
                )
            ],
            "relevant_document_ranks": relevant_ranks,
            "expected_document_retrieved": expected_retrieved
        })

        print(f"\n{query_id}: {query}")
        print(f"Relevant document: {relevant_docs}")
        print(f"Relevant document rank: {relevant_ranks}")
        print(f"Retrieved documents: {len(ranked_ids)}")

    # Save CSV
    with open(
        CSV_OUTPUT,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Query_ID",
                "Query",
                "Query_Type",
                "Relevant_Documents",
                "Retrieved_Documents",
                "Number_of_Results",
                "Expected_Document_Retrieved",
                "Relevant_Document_Ranks"
            ]
        )

        writer.writeheader()

        for row in csv_rows:

            csv_row = row.copy()

            csv_row["Relevant_Document_Ranks"] = json.dumps(
                csv_row["Relevant_Document_Ranks"]
            )

            writer.writerow(csv_row)

    # Save JSON
    with open(
        JSON_OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "experiment": "Baseline Lexical Retrieval",
                "total_queries": 5,
                "total_documents": 20,
                "queries": json_results
            },
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("BASELINE RETRIEVAL COMPLETED")
    print("=" * 60)

    print(f"CSV saved: {CSV_OUTPUT}")
    print(f"JSON saved: {JSON_OUTPUT}")


if __name__ == "__main__":
    run_baseline()

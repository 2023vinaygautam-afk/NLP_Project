
import json
import csv
import re
from pathlib import Path

from collections import Counter


# ==================================================
# 1. PATH CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results" / "cleaned_corpus.json"

OUTPUT_JSON = ROOT / "results" / "ngram_analysis.json"

OUTPUT_CSV = ROOT / "results" / "ngram_frequency.csv"

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. TOKENIZATION
# ==================================================

def tokenize_text(text):

    pattern = (
        r"\b[A-Za-z]+(?:[-'][A-Za-z0-9]+)*\b"
        r"|\d+(?:\.\d+)?%?"
    )

    return re.findall(pattern, text.lower())


# ==================================================
# 3. GENERATE N-GRAMS
# ==================================================

def generate_ngrams(tokens, n):

    ngrams = []

    for i in range(len(tokens) - n + 1):

        gram = tuple(tokens[i:i + n])

        ngrams.append(gram)

    return ngrams


# ==================================================
# 4. ANALYZE ONE DOCUMENT
# ==================================================

def analyze_document(text):

    tokens = tokenize_text(text)

    document_result = {}

    for n in range(1, 6):

        ngrams = generate_ngrams(tokens, n)

        counter = Counter(ngrams)

        document_result[f"{n}-gram"] = {

            "total_occurrences": len(ngrams),

            "unique_ngrams": len(counter),

            "top_10": [

                {
                    "ngram": " ".join(gram),

                    "frequency": frequency

                }

                for gram, frequency in counter.most_common(10)

            ]

        }

    return document_result


# ==================================================
# 5. NORMALIZE CORPUS
# ==================================================

def normalize_corpus(corpus):

    if isinstance(corpus, dict):

        corpus = corpus.get("documents", [])

    if not isinstance(corpus, list):

        raise TypeError(
            "Expected a list of documents."
        )

    normalized = []

    for index, document in enumerate(corpus, start=1):

        if not isinstance(document, dict):

            raise TypeError(
                f"Document {index} is not a dictionary."
            )

        doc_id = (
            document.get("doc_id")
            or document.get("id")
            or document.get("document_id")
            or f"D{index:02d}"
        )

        name = (
            document.get("document_name")
            or document.get("name")
            or document.get("filename")
            or document.get("document")
            or doc_id
        )

        text = (
            document.get("cleaned_text")
            or document.get("cleaned")
            or document.get("text")
            or document.get("content")
            or document.get("original_text")
            or ""
        )

        if not isinstance(text, str) or not text.strip():

            raise ValueError(
                f"Missing text for {doc_id}. "
                f"Available fields: {list(document.keys())}"
            )

        normalized.append({

            "doc_id": str(doc_id),

            "document_name": str(name),

            "text": text

        })

    return normalized


# ==================================================
# 6. MAIN EXECUTION
# ==================================================

def main():

    print("\nN-GRAM ANALYSIS")
    print("==============")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        raw_corpus = json.load(file)

    corpus = normalize_corpus(raw_corpus)

    # Corpus-level frequency counters
    corpus_counters = {
        n: Counter()
        for n in range(1, 6)
    }

    document_results = []

    csv_rows = []

    # ==================================================
    # 7. DOCUMENT-LEVEL ANALYSIS
    # ==================================================

    for document in corpus:

        doc_id = document["doc_id"]

        document_name = document["document_name"]

        text = document["text"]

        tokens = tokenize_text(text)

        document_analysis = {}

        for n in range(1, 6):

            ngrams = generate_ngrams(tokens, n)

            counter = Counter(ngrams)

            corpus_counters[n].update(counter)

            document_analysis[f"{n}-gram"] = {

                "total_occurrences": len(ngrams),

                "unique_ngrams": len(counter),

                "top_10": [

                    {
                        "ngram": " ".join(gram),

                        "frequency": frequency

                    }

                    for gram, frequency in counter.most_common(10)

                ]

            }

        document_results.append({

            "doc_id": doc_id,

            "document_name": document_name,

            "token_count": len(tokens),

            "analysis": document_analysis

        })

        print(
            f"Processed: {doc_id} - {document_name}"
        )

    # ==================================================
    # 8. CORPUS-LEVEL ANALYSIS
    # ==================================================

    corpus_analysis = {}

    for n in range(1, 6):

        counter = corpus_counters[n]

        corpus_analysis[f"{n}-gram"] = {

            "total_occurrences": sum(counter.values()),

            "unique_ngrams": len(counter),

            "top_10": [

                {
                    "ngram": " ".join(gram),

                    "frequency": frequency

                }

                for gram, frequency in counter.most_common(10)

            ]

        }

        for gram, frequency in counter.most_common():

            csv_rows.append({

                "n": n,

                "ngram": " ".join(gram),

                "frequency": frequency

            })

    # ==================================================
    # 9. FINAL SUMMARY
    # ==================================================

    summary = {

        "total_documents": len(corpus),

        "ngram_range": "1-gram to 5-gram",

        "corpus_analysis": corpus_analysis,

        "document_analysis": document_results

    }

    # ==================================================
    # 10. SAVE JSON
    # ==================================================

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    # ==================================================
    # 11. SAVE CSV
    # ==================================================

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "n",
                "ngram",
                "frequency"
            ]
        )

        writer.writeheader()

        writer.writerows(csv_rows)

    # ==================================================
    # 12. PRINT SUMMARY
    # ==================================================

    print("\nN-GRAM ANALYSIS COMPLETED")
    print("=========================")

    print("Total documents:", len(corpus))

    for n in range(1, 6):

        key = f"{n}-gram"

        data = corpus_analysis[key]

        print(f"\n{key.upper()}")

        print(
            "Total occurrences:",
            data["total_occurrences"]
        )

        print(
            "Unique n-grams:",
            data["unique_ngrams"]
        )

        print("Top 5:")

        for item in data["top_10"][:5]:

            print(
                f"  {item['ngram']}: "
                f"{item['frequency']}"
            )

    print("\nGenerated files:")

    print(OUTPUT_JSON)

    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()

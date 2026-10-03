
from pathlib import Path
import json
import csv
import re
import nltk

from nltk.corpus import stopwords


# ==================================================
# 1. PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = PROJECT_ROOT / "results"

INPUT_FILE = RESULT_DIR / "cleaned_corpus.json"

OUTPUT_JSON = RESULT_DIR / "stopwords_analysis.json"
OUTPUT_CSV = RESULT_DIR / "stopwords_comparison.csv"


# ==================================================
# 2. STOP WORDS
# ==================================================

def load_stopwords():

    try:
        stop_words = set(stopwords.words("english"))

    except LookupError:

        nltk.download("stopwords", quiet=True)

        stop_words = set(stopwords.words("english"))

    # Preserve important healthcare terms
    protected_terms = {
        "no",
        "not",
        "nor",
        "against",
        "before",
        "after",
        "during",
        "between",
        "under",
        "over",
        "only",
        "without"
    }

    stop_words = stop_words - protected_terms

    return stop_words


# ==================================================
# 3. TOKENIZATION
# ==================================================

def tokenize(text):

    return re.findall(
        r"\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b",
        text.lower()
    )


# ==================================================
# 4. REMOVE STOP WORDS
# ==================================================

def remove_stopwords(tokens, stop_words):

    return [
        token
        for token in tokens
        if token.lower() not in stop_words
    ]


# ==================================================
# 5. STATISTICS
# ==================================================

def calculate_statistics(tokens):

    return {
        "token_count": len(tokens),
        "vocabulary_size": len(set(tokens))
    }


# ==================================================
# 6. PROCESS DOCUMENTS
# ==================================================

def process_documents(documents, stop_words):

    processed_documents = []
    comparison_results = []

    all_original_tokens = []
    all_filtered_tokens = []

    for document in documents:

        document_id = document["document_id"]

        text = document["cleaned_text"]

        original_tokens = tokenize(text)

        filtered_tokens = remove_stopwords(
            original_tokens,
            stop_words
        )

        all_original_tokens.extend(original_tokens)
        all_filtered_tokens.extend(filtered_tokens)

        original_stats = calculate_statistics(
            original_tokens
        )

        filtered_stats = calculate_statistics(
            filtered_tokens
        )

        removed_count = (
            len(original_tokens) - len(filtered_tokens)
        )

        removal_percentage = (
            removed_count / len(original_tokens) * 100
            if original_tokens else 0
        )

        processed_documents.append({
            "document_id": document_id,
            "file_name": document["file_name"],
            "tokens_before_removal": original_tokens,
            "tokens_after_removal": filtered_tokens,
            "token_count_before": original_stats["token_count"],
            "token_count_after": filtered_stats["token_count"],
            "vocabulary_before": original_stats["vocabulary_size"],
            "vocabulary_after": filtered_stats["vocabulary_size"]
        })

        comparison_results.append({
            "document_id": document_id,
            "token_count_before": original_stats["token_count"],
            "token_count_after": filtered_stats["token_count"],
            "removed_tokens": removed_count,
            "removal_percentage": round(
                removal_percentage, 2
            ),
            "vocabulary_before": original_stats["vocabulary_size"],
            "vocabulary_after": filtered_stats["vocabulary_size"]
        })

    corpus_summary = {
        "total_tokens_before": len(all_original_tokens),
        "total_tokens_after": len(all_filtered_tokens),
        "total_removed_tokens": (
            len(all_original_tokens) - len(all_filtered_tokens)
        ),
        "vocabulary_before": len(set(all_original_tokens)),
        "vocabulary_after": len(set(all_filtered_tokens))
    }

    corpus_summary["removal_percentage"] = round(
        corpus_summary["total_removed_tokens"]
        / corpus_summary["total_tokens_before"] * 100,
        2
    ) if corpus_summary["total_tokens_before"] else 0

    return (
        processed_documents,
        comparison_results,
        corpus_summary
    )


# ==================================================
# 7. SAVE RESULTS
# ==================================================

def save_results(
    processed_documents,
    comparison_results,
    corpus_summary
):

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "corpus_summary": corpus_summary,
                "documents": processed_documents
            },
            file,
            indent=4,
            ensure_ascii=False
        )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(comparison_results[0].keys())
        )

        writer.writeheader()
        writer.writerows(comparison_results)


# ==================================================
# 8. MAIN
# ==================================================

def main():

    print("=" * 65)
    print("MODULE 4: STOP-WORD ANALYSIS")
    print("=" * 65)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "cleaned_corpus.json not found. "
            "Run previous modules first."
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        documents = json.load(file)

    stop_words = load_stopwords()

    (
        processed_documents,
        comparison_results,
        corpus_summary
    ) = process_documents(documents, stop_words)

    save_results(
        processed_documents,
        comparison_results,
        corpus_summary
    )

    print("\nSTOP-WORD CORPUS COMPARISON")
    print("-" * 65)

    for key, value in corpus_summary.items():

        print(f"{key.replace('_', ' ').title()}: {value}")

    print("\nSAMPLE COMPARISON")
    print("-" * 65)

    sample = processed_documents[0]

    print("\nBefore removal:")
    print(sample["tokens_before_removal"][:30])

    print("\nAfter removal:")
    print(sample["tokens_after_removal"][:30])

    print("\nOUTPUT FILES")
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)

    print("\nMODULE 4 COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()

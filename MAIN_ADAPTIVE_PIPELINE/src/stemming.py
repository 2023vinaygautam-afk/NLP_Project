
from pathlib import Path
import json
import csv
import re
import nltk

from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer


# ==================================================
# 1. PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = PROJECT_ROOT / "results"

INPUT_FILE = RESULT_DIR / "cleaned_corpus.json"

OUTPUT_JSON = RESULT_DIR / "stemming_comparison.json"
OUTPUT_CSV = RESULT_DIR / "stemming_comparison.csv"


# ==================================================
# 2. INITIALIZE STEMMERS
# ==================================================

stemmer_methods = {
    "Porter": PorterStemmer(),
    "Lancaster": LancasterStemmer(),
    "Snowball": SnowballStemmer("english")
}


# ==================================================
# 3. TOKENIZATION
# ==================================================

def tokenize(text):

    return re.findall(
        r"\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b",
        text.lower()
    )


# ==================================================
# 4. APPLY STEMMING
# ==================================================

def apply_stemming(tokens, stemmer):

    return [
        stemmer.stem(token)
        for token in tokens
    ]


# ==================================================
# 5. PROCESS CORPUS
# ==================================================

def process_corpus(documents):

    summary = {}
    document_results = []
    word_examples = {}

    for method_name, stemmer in stemmer_methods.items():

        all_stems = []
        total_tokens = 0

        for document in documents:

            tokens = tokenize(document["cleaned_text"])

            stems = apply_stemming(tokens, stemmer)

            total_tokens += len(stems)
            all_stems.extend(stems)

            document_results.append({
                "document_id": document["document_id"],
                "method": method_name,
                "token_count": len(stems),
                "vocabulary_size": len(set(stems)),
                "stemmed_tokens": stems
            })

            for original, stemmed in zip(tokens, stems):

                if original != stemmed:

                    word_examples.setdefault(
                        original, {}
                    )

                    word_examples[original][method_name] = stemmed

        summary[method_name] = {
            "total_tokens": total_tokens,
            "vocabulary_size": len(set(all_stems)),
            "average_tokens_per_document": round(
                total_tokens / len(documents), 2
            ) if documents else 0
        }

    return summary, document_results, word_examples


# ==================================================
# 6. SAVE RESULTS
# ==================================================

def save_results(summary, document_results, word_examples):

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "summary": summary,
                "word_examples": word_examples,
                "document_results": document_results
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
            fieldnames=[
                "document_id",
                "method",
                "token_count",
                "vocabulary_size",
                "stemmed_tokens"
            ]
        )

        writer.writeheader()

        for row in document_results:

            writer.writerow({
                **row,
                "stemmed_tokens": " ".join(
                    row["stemmed_tokens"]
                )
            })


# ==================================================
# 7. MAIN
# ==================================================

def main():

    print("=" * 65)
    print("MODULE 5: STEMMING COMPARISON")
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

    summary, document_results, word_examples = process_corpus(
        documents
    )

    save_results(
        summary,
        document_results,
        word_examples
    )

    print("\nSTEMMING COMPARISON")
    print("-" * 65)

    for method, stats in summary.items():

        print(f"\nMethod: {method}")
        print(f"Total Tokens: {stats['total_tokens']}")
        print(f"Vocabulary Size: {stats['vocabulary_size']}")
        print(
            "Average Tokens per Document:",
            stats["average_tokens_per_document"]
        )

    print("\nWORD STEMMING EXAMPLES")
    print("-" * 65)

    for word, methods in list(word_examples.items())[:20]:

        print(f"\nOriginal: {word}")

        for method, stem in methods.items():

            print(f"{method}: {stem}")

    print("\nOUTPUT FILES")
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)

    print("\nMODULE 5 COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()

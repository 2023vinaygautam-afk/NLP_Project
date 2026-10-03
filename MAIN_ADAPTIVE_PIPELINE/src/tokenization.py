
from pathlib import Path
import json
import re
import csv

from nltk.tokenize import TreebankWordTokenizer
import spacy


# ==================================================
# 1. PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = PROJECT_ROOT / "results"

INPUT_FILE = RESULT_DIR / "cleaned_corpus.json"

OUTPUT_JSON = RESULT_DIR / "tokenization_comparison.json"
OUTPUT_CSV = RESULT_DIR / "tokenization_comparison.csv"

nlp = spacy.load("en_core_web_sm")

nltk_tokenizer = TreebankWordTokenizer()


# ==================================================
# 2. CUSTOM HEALTHCARE TOKENIZER
# ==================================================

def custom_tokenize(text):

    pattern = (
        r"COVID-19"
        r"|SARS-CoV-2"
        r"|MERS-CoV"
        r"|H1N1"
        r"|PCR"
        r"|RT-PCR"
        r"|[a-zA-Z]+(?:-[a-zA-Z0-9]+)*"
        r"|\d+(?:\.\d+)?%?"
    )

    return re.findall(pattern, text, flags=re.IGNORECASE)


# ==================================================
# 3. TOKENIZATION METHODS
# ==================================================

def nltk_tokenize(text):

    return nltk_tokenizer.tokenize(text)


def spacy_tokenize(text):

    doc = nlp(text)

    return [
        token.text
        for token in doc
        if not token.is_space
    ]


def hybrid_tokenize(text):

    # Use custom rules for healthcare terms,
    # followed by spaCy tokenization for ordinary text.

    domain_terms = [
        "COVID-19",
        "SARS-CoV-2",
        "MERS-CoV",
        "H1N1",
        "RT-PCR"
    ]

    placeholders = {}

    protected_text = text

    for index, term in enumerate(domain_terms):

        placeholder = f"DOMAINTERM{index}TOKEN"

        pattern = re.compile(
            re.escape(term),
            flags=re.IGNORECASE
        )

        if pattern.search(protected_text):

            protected_text = pattern.sub(
                placeholder,
                protected_text
            )

            placeholders[placeholder.lower()] = term

    tokens = spacy_tokenize(protected_text)

    final_tokens = []

    for token in tokens:

        key = token.lower()

        if key in placeholders:
            final_tokens.append(placeholders[key])
        else:
            final_tokens.append(token)

    return final_tokens


# ==================================================
# 4. VOCABULARY STATISTICS
# ==================================================

def calculate_statistics(token_lists):

    all_tokens = [
        token.lower()
        for tokens in token_lists
        for token in tokens
    ]

    total_tokens = len(all_tokens)

    vocabulary = set(all_tokens)

    document_count = len(token_lists)

    average_tokens = (
        total_tokens / document_count
        if document_count else 0
    )

    return {
        "total_tokens": total_tokens,
        "vocabulary_size": len(vocabulary),
        "average_tokens_per_document": round(
            average_tokens, 2
        )
    }


# ==================================================
# 5. PROCESS CORPUS
# ==================================================

def run_comparison(documents):

    methods = {
        "NLTK": nltk_tokenize,
        "spaCy": spacy_tokenize,
        "Custom": custom_tokenize,
        "Hybrid": hybrid_tokenize
    }

    results = {}
    document_results = []

    for method_name, tokenizer in methods.items():

        token_lists = []

        for document in documents:

            text = document["cleaned_text"]

            tokens = tokenizer(text)

            token_lists.append(tokens)

            document_results.append({
                "document_id": document["document_id"],
                "method": method_name,
                "token_count": len(tokens),
                "tokens": tokens
            })

        results[method_name] = calculate_statistics(
            token_lists
        )

    return results, document_results


# ==================================================
# 6. SAVE RESULTS
# ==================================================

def save_results(summary, document_results):

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "summary": summary,
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
                "tokens"
            ]
        )

        writer.writeheader()

        for row in document_results:

            writer.writerow({
                **row,
                "tokens": " ".join(row["tokens"])
            })


# ==================================================
# 7. MAIN
# ==================================================

def main():

    print("=" * 65)
    print("MODULE 3: TOKENIZATION COMPARISON")
    print("=" * 65)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "cleaned_corpus.json not found. "
            "Run document_loader.py and "
            "text_preprocessing.py first."
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        documents = json.load(file)

    summary, document_results = run_comparison(documents)

    save_results(summary, document_results)

    print("\nTOKENIZATION RESULTS")
    print("-" * 65)

    for method, stats in summary.items():

        print(f"\nMethod: {method}")
        print(f"Total Tokens: {stats['total_tokens']}")
        print(f"Vocabulary Size: {stats['vocabulary_size']}")
        print(
            "Average Tokens per Document:",
            stats["average_tokens_per_document"]
        )

    print("\nSAMPLE TOKEN COMPARISON")
    print("-" * 65)

    first_document = documents[0]["document_id"]

    for row in document_results:

        if row["document_id"] == first_document:

            print(f"\n{row['method']}:")
            print(row["tokens"][:30])

    print("\nOUTPUT FILES")
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)

    print("\nTOKENIZATION COMPARISON COMPLETED.")


if __name__ == "__main__":
    main()

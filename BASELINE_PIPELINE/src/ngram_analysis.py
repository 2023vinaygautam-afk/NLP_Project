
from pathlib import Path
from collections import Counter
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "documents"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def tokenize(text):
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b[a-zA-Z]+(?:[-'][a-zA-Z]+)*\b", text.lower())


def generate_ngrams(tokens, n):
    """Generate n-grams from a token list."""
    return [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]


def main():
    print("\nBASELINE PIPELINE")
    print("MODULE 6: N-GRAM ANALYSIS")

    all_tokens = []
    document_data = []

    for file_path in sorted(DOCS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        tokens = tokenize(text)

        all_tokens.extend(tokens)

        document_data.append({
            "Document_ID": file_path.stem,
            "Token_Count": len(tokens)
        })

    if not all_tokens:
        print("No tokens found. Check the documents folder.")
        return

    summary = []
    frequency_rows = []

    for n in range(1, 6):
        ngrams = generate_ngrams(all_tokens, n)
        counts = Counter(ngrams)

        total_count = len(ngrams)
        unique_count = len(counts)

        summary.append({
            "N": n,
            "Ngram_Type": {
                1: "Unigram",
                2: "Bigram",
                3: "Trigram",
                4: "Four-Gram",
                5: "Five-Gram"
            }[n],
            "Total_Ngrams": total_count,
            "Unique_Ngrams": unique_count
        })

        for phrase, frequency in counts.most_common(100):
            frequency_rows.append({
                "N": n,
                "Ngram": " ".join(phrase),
                "Frequency": frequency
            })

        print(
            f"{n}-gram | Total: {total_count} | "
            f"Unique: {unique_count}"
        )

    summary_df = pd.DataFrame(summary)
    frequency_df = pd.DataFrame(frequency_rows)
    document_df = pd.DataFrame(document_data)

    summary_df.to_csv(
        RESULTS_DIR / "ngram_summary.csv",
        index=False
    )

    frequency_df.to_csv(
        RESULTS_DIR / "ngram_frequencies.csv",
        index=False
    )

    document_df.to_csv(
        RESULTS_DIR / "ngram_document_statistics.csv",
        index=False
    )

    print("\nTOP 10 BIGRAMS")

    bigrams = Counter(generate_ngrams(all_tokens, 2))

    for phrase, frequency in bigrams.most_common(10):
        print(f"{' '.join(phrase)}: {frequency}")

    print("\nN-Gram analysis completed successfully.")
    print("Results saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()

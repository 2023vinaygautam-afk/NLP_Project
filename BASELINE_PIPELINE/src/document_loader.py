
from pathlib import Path
import re
import pandas as pd


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# DOCUMENT LOADING
# --------------------------------------------------

def load_documents():

    documents = {}

    for file_path in sorted(DOCUMENTS_DIR.glob("*.txt")):

        doc_id = file_path.stem

        text = file_path.read_text(encoding="utf-8")

        documents[doc_id] = text

    if not documents:
        raise ValueError("No documents found in the dataset folder.")

    return documents


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):

    text = text.replace("\ufeff", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# --------------------------------------------------
# BASIC TOKENIZATION
# --------------------------------------------------

def tokenize_text(text):

    pattern = r"[A-Za-z]+(?:[-'][A-Za-z0-9]+)*|\d+(?:\.\d+)?"

    return re.findall(pattern, text.lower())


# --------------------------------------------------
# SENTENCE COUNT
# --------------------------------------------------

def count_sentences(text):

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    return len([s for s in sentences if s])


# --------------------------------------------------
# CORPUS STATISTICS
# --------------------------------------------------

def generate_corpus_statistics(documents):

    rows = []
    all_tokens = []

    for doc_id, original_text in documents.items():

        cleaned_text = clean_text(original_text)

        tokens = tokenize_text(cleaned_text)

        all_tokens.extend(tokens)

        rows.append({
            "Document_ID": doc_id,
            "Characters": len(cleaned_text),
            "Sentences": count_sentences(cleaned_text),
            "Tokens": len(tokens),
            "Vocabulary_Size": len(set(tokens))
        })

    statistics_df = pd.DataFrame(rows)

    total_documents = len(documents)
    total_characters = int(statistics_df["Characters"].sum())
    total_sentences = int(statistics_df["Sentences"].sum())
    total_tokens = int(statistics_df["Tokens"].sum())

    vocabulary_size = len(set(all_tokens))

    average_document_length = (
        total_tokens / total_documents
    )

    summary = {
        "Total_Documents": total_documents,
        "Total_Characters": total_characters,
        "Total_Sentences": total_sentences,
        "Total_Tokens": total_tokens,
        "Vocabulary_Size": vocabulary_size,
        "Average_Document_Length": round(
            average_document_length, 2
        )
    }

    return statistics_df, summary


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    print("=" * 60)
    print("BASELINE NLP PIPELINE")
    print("MODULE 1: DOCUMENT LOADING AND STATISTICS")
    print("=" * 60)

    documents = load_documents()

    statistics_df, summary = generate_corpus_statistics(documents)

    print("\nCORPUS STATISTICS")
    print("-" * 40)

    for metric, value in summary.items():
        print(f"{metric}: {value}")

    print("\nDOCUMENT STATISTICS")
    print(statistics_df.to_string(index=False))

    statistics_df.to_csv(
        RESULTS_DIR / "document_statistics.csv",
        index=False
    )

    pd.DataFrame(
        list(summary.items()),
        columns=["Metric", "Value"]
    ).to_csv(
        RESULTS_DIR / "corpus_summary.csv",
        index=False
    )

    print("\nResults saved successfully!")
    print(f"Output directory: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

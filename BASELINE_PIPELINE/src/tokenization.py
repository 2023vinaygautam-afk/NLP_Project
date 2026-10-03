
from pathlib import Path
import re
import pandas as pd
import nltk
import spacy

from nltk.tokenize import word_tokenize, sent_tokenize


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# INITIALIZE NLP TOOLS
# --------------------------------------------------

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy English model missing. Run: "
        "python -m spacy download en_core_web_sm"
    )


# --------------------------------------------------
# CUSTOM HEALTHCARE TOKENIZER
# --------------------------------------------------

CUSTOM_PATTERN = (
    r"COVID-19|SARS-CoV-2|"
    r"\d+(?:\.\d+)?%?|"
    r"[A-Za-z]+(?:[-'][A-Za-z0-9]+)*|"
    r"[^\w\s]"
)


def custom_tokenize(text):

    return re.findall(CUSTOM_PATTERN, text, flags=re.IGNORECASE)


# --------------------------------------------------
# TOKENIZATION METHODS
# --------------------------------------------------

def nltk_tokenize(text):

    return word_tokenize(text)


def spacy_tokenize(text):

    doc = nlp(text)

    return [token.text for token in doc]


def hybrid_tokenize(text):

    """
    Use spaCy for sentence-aware processing and
    custom rules to preserve healthcare terminology.
    """

    doc = nlp(text)

    tokens = []

    for sentence in doc.sents:

        sentence_text = sentence.text

        tokens.extend(custom_tokenize(sentence_text))

    return tokens


# --------------------------------------------------
# PROCESS DOCUMENTS
# --------------------------------------------------

def load_documents():

    documents = {}

    for file_path in sorted(DOCUMENTS_DIR.glob("*.txt")):

        documents[file_path.stem] = file_path.read_text(
            encoding="utf-8"
        )

    if not documents:
        raise ValueError("No healthcare documents found.")

    return documents


# --------------------------------------------------
# TOKENIZATION COMPARISON
# --------------------------------------------------

def compare_tokenizers(documents):

    results = []

    for doc_id, text in documents.items():

        methods = {
            "NLTK": nltk_tokenize(text),
            "spaCy": spacy_tokenize(text),
            "Custom": custom_tokenize(text),
            "Hybrid": hybrid_tokenize(text)
        }

        for method, tokens in methods.items():

            results.append({
                "Document_ID": doc_id,
                "Method": method,
                "Token_Count": len(tokens),
                "Vocabulary_Size": len(set(
                    token.lower() for token in tokens
                )),
                "Sample_Tokens": " | ".join(tokens[:20])
            })

    return pd.DataFrame(results)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 60)
    print("BASELINE PIPELINE")
    print("MODULE 2: TOKENIZATION COMPARISON")
    print("=" * 60)

    documents = load_documents()

    results_df = compare_tokenizers(documents)

    summary = (
        results_df.groupby("Method")
        .agg(
            Total_Tokens=("Token_Count", "sum"),
            Average_Tokens=("Token_Count", "mean"),
            Total_Document_Vocabulary=(
                "Vocabulary_Size", "sum"
            )
        )
        .reset_index()
    )

    print("\nTOKENIZATION COMPARISON")
    print(summary.to_string(index=False))

    results_df.to_csv(
        RESULTS_DIR / "tokenization_comparison.csv",
        index=False
    )

    summary.to_csv(
        RESULTS_DIR / "tokenization_summary.csv",
        index=False
    )

    print("\nResults saved successfully!")


if __name__ == "__main__":
    main()
